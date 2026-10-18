# BigGAN-deep generator.
# Adapted from huggingface/pytorch-pretrained-BigGAN via lucidrains/big-sleep (MIT).
#
# Optimizations vs. the original (all numerically equivalent to eval-mode output):
#   * spectral norm is baked into plain weights once at load time, instead of
#     re-normalising every conv/linear weight on every forward pass
#   * the generator is frozen (requires_grad=False), so backward only computes
#     gradients w.r.t. the latents, never the ~50M generator weights
#   * conv_to_rgb computed 128 channels at full resolution and kept 3 - the weight
#     is now sliced to those 3 output channels
#   * conditional batch-norm is folded into a single scale/shift (one fused kernel)
#   * self-attention uses fused scaled_dot_product_attention
#   * latents are batched (B, L, dim) so several images optimise in one pass
import json
import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint


class BigGANConfig:
    def __init__(self, **kw):
        self.output_dim = 128
        self.z_dim = 128
        self.class_embed_dim = 128
        self.channel_width = 128
        self.num_classes = 1000
        self.layers = []
        self.attention_layer_position = 8
        self.eps = 1e-4
        self.n_stats = 51
        self.__dict__.update(kw)

    @classmethod
    def from_json_file(cls, path):
        with open(path, "r", encoding="utf-8") as f:
            return cls(**json.load(f))

    @property
    def num_latents(self):
        return len(self.layers) + 1


def _sn(module, eps):
    return nn.utils.spectral_norm(module, eps=eps)


def snconv2d(eps=1e-12, **kw):
    return _sn(nn.Conv2d(**kw), eps)


def snlinear(eps=1e-12, **kw):
    return _sn(nn.Linear(**kw), eps)


class SelfAttn(nn.Module):
    def __init__(self, in_channels, eps=1e-12):
        super().__init__()
        self.in_channels = in_channels
        self.snconv1x1_theta = snconv2d(in_channels=in_channels, out_channels=in_channels // 8, kernel_size=1, bias=False, eps=eps)
        self.snconv1x1_phi = snconv2d(in_channels=in_channels, out_channels=in_channels // 8, kernel_size=1, bias=False, eps=eps)
        self.snconv1x1_g = snconv2d(in_channels=in_channels, out_channels=in_channels // 2, kernel_size=1, bias=False, eps=eps)
        self.snconv1x1_o_conv = snconv2d(in_channels=in_channels // 2, out_channels=in_channels, kernel_size=1, bias=False, eps=eps)
        self.gamma = nn.Parameter(torch.zeros(1))
        self.upcast = False  # set for fp16 autocast: unscaled logits can overflow half

    def forward(self, x):
        b, ch, h, w = x.shape
        q = self.snconv1x1_theta(x).flatten(2).transpose(1, 2)                      # b, hw, ch/8
        k = F.max_pool2d(self.snconv1x1_phi(x), 2).flatten(2).transpose(1, 2)        # b, hw/4, ch/8
        v = F.max_pool2d(self.snconv1x1_g(x), 2).flatten(2).transpose(1, 2)          # b, hw/4, ch/2
        if self.upcast:
            q, k, v = q.float(), k.float(), v.float()
        # original: softmax(theta^T phi) with no 1/sqrt(d) scaling -> scale=1.0
        o = F.scaled_dot_product_attention(q.unsqueeze(1), k.unsqueeze(1), v.unsqueeze(1), scale=1.0).squeeze(1)
        o = o.to(x.dtype).transpose(1, 2).reshape(b, ch // 2, h, w)
        return x + self.gamma * self.snconv1x1_o_conv(o)


class BigGANBatchNorm(nn.Module):
    """Conditional batch norm with fixed (truncation=1) statistics, folded into scale/shift."""

    def __init__(self, num_features, condition_vector_dim=None, n_stats=51, eps=1e-4, conditional=True):
        super().__init__()
        self.num_features = num_features
        self.eps = eps
        self.conditional = conditional
        self.register_buffer("running_means", torch.zeros(n_stats, num_features))
        self.register_buffer("running_vars", torch.ones(n_stats, num_features))
        self.step_size = 1.0 / (n_stats - 1)
        if conditional:
            self.scale = snlinear(in_features=condition_vector_dim, out_features=num_features, bias=False, eps=eps)
            self.offset = snlinear(in_features=condition_vector_dim, out_features=num_features, bias=False, eps=eps)
        else:
            self.weight = nn.Parameter(torch.ones(num_features))
            self.bias = nn.Parameter(torch.zeros(num_features))
        self.register_buffer("_mean", torch.zeros(num_features), persistent=False)
        self.register_buffer("_inv_std", torch.ones(num_features), persistent=False)

    @torch.no_grad()
    def prepare(self, truncation=1.0):
        coef, start = math.modf(truncation / self.step_size)
        start = int(start)
        if coef != 0.0:
            mean = self.running_means[start] * coef + self.running_means[start + 1] * (1 - coef)
            var = self.running_vars[start] * coef + self.running_vars[start + 1] * (1 - coef)
        else:
            mean, var = self.running_means[start], self.running_vars[start]
        self._mean.copy_(mean)
        self._inv_std.copy_(torch.rsqrt(var + self.eps))

    def forward(self, x, condition_vector=None):
        if self.conditional:
            a = (1 + self.scale(condition_vector)) * self._inv_std          # (B, C)
            shift = self.offset(condition_vector) - self._mean * a
        else:
            a = self.weight * self._inv_std
            shift = (self.bias - self._mean * a).unsqueeze(0)
            a = a.unsqueeze(0)
        a = a.to(x.dtype)[:, :, None, None]
        shift = shift.to(x.dtype)[:, :, None, None]
        return torch.addcmul(shift, x, a)


class GenBlock(nn.Module):
    def __init__(self, in_size, out_size, condition_vector_dim, reduction_factor=4, up_sample=False, n_stats=51, eps=1e-12):
        super().__init__()
        self.up_sample = up_sample
        self.drop_channels = in_size != out_size
        mid = in_size // reduction_factor
        self.bn_0 = BigGANBatchNorm(in_size, condition_vector_dim, n_stats=n_stats, eps=eps)
        self.conv_0 = snconv2d(in_channels=in_size, out_channels=mid, kernel_size=1, eps=eps)
        self.bn_1 = BigGANBatchNorm(mid, condition_vector_dim, n_stats=n_stats, eps=eps)
        self.conv_1 = snconv2d(in_channels=mid, out_channels=mid, kernel_size=3, padding=1, eps=eps)
        self.bn_2 = BigGANBatchNorm(mid, condition_vector_dim, n_stats=n_stats, eps=eps)
        self.conv_2 = snconv2d(in_channels=mid, out_channels=mid, kernel_size=3, padding=1, eps=eps)
        self.bn_3 = BigGANBatchNorm(mid, condition_vector_dim, n_stats=n_stats, eps=eps)
        self.conv_3 = snconv2d(in_channels=mid, out_channels=out_size, kernel_size=1, eps=eps)

    def forward(self, x, cond):
        x0 = x
        x = self.conv_0(F.relu(self.bn_0(x, cond)))
        x = F.relu(self.bn_1(x, cond))
        if self.up_sample:
            x = F.interpolate(x, scale_factor=2, mode="nearest")
        x = self.conv_1(x)
        x = self.conv_2(F.relu(self.bn_2(x, cond)))
        x = self.conv_3(F.relu(self.bn_3(x, cond)))
        if self.drop_channels:
            x0 = x0[:, : x0.shape[1] // 2]
        if self.up_sample:
            x0 = F.interpolate(x0, scale_factor=2, mode="nearest")
        return x + x0


class Generator(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        ch = config.channel_width
        cond_dim = config.z_dim * 2
        self.gen_z = snlinear(in_features=cond_dim, out_features=4 * 4 * 16 * ch, eps=config.eps)
        layers = []
        for i, layer in enumerate(config.layers):
            if i == config.attention_layer_position:
                layers.append(SelfAttn(ch * layer[1], eps=config.eps))
            layers.append(GenBlock(ch * layer[1], ch * layer[2], cond_dim, up_sample=layer[0], n_stats=config.n_stats, eps=config.eps))
        self.layers = nn.ModuleList(layers)
        self.bn = BigGANBatchNorm(ch, n_stats=config.n_stats, eps=config.eps, conditional=False)
        self.conv_to_rgb = snconv2d(in_channels=ch, out_channels=ch, kernel_size=3, padding=1, eps=config.eps)
        self.checkpoint = False  # recompute block activations in backward to save VRAM

    def forward(self, cond):  # cond: (B, L, 2*z_dim)
        ch = self.config.channel_width
        z = self.gen_z(cond[:, 0])
        # TF weight layout: (B, 4, 4, C) -> NCHW view, which is already channels_last in memory
        z = z.view(-1, 4, 4, 16 * ch).permute(0, 3, 1, 2)
        ckpt = self.checkpoint and torch.is_grad_enabled()
        idx = 1
        for layer in self.layers:
            args = (z, cond[:, idx]) if isinstance(layer, GenBlock) else (z,)
            if isinstance(layer, GenBlock):
                idx += 1
            z = checkpoint(layer, *args, use_reentrant=False) if ckpt else layer(*args)
        z = F.relu(self.bn(z))
        z = self.conv_to_rgb(z)
        return torch.tanh(z[:, :3])


class BigGAN(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.embeddings = nn.Linear(config.num_classes, config.z_dim, bias=False)
        self.generator = Generator(config)

    def forward(self, z, classes):
        """z: (B, L, z_dim), classes: (B, L, num_classes) -> image (B, 3, S, S) in [-1, 1]"""
        cond = torch.cat((z, self.embeddings(classes)), dim=-1)
        return self.generator(cond)

    @classmethod
    def load(cls, config_path, weights_path, log=print):
        config = BigGANConfig.from_json_file(config_path)
        model = cls(config)
        state = torch.load(weights_path, map_location="cpu", weights_only=True)
        missing, unexpected = model.load_state_dict(state, strict=False)
        missing = [k for k in missing if not k.endswith(("_mean", "_inv_std"))]
        if missing:
            raise RuntimeError(f"BigGAN checkpoint is missing {len(missing)} tensors, e.g. {missing[:3]}")
        if unexpected:
            log(f"BigGAN: ignoring {len(unexpected)} unexpected checkpoint tensors")
        model.bake()
        return model

    @torch.no_grad()
    def bake(self):
        """Freeze into a fast inference graph. Equivalent to the original's eval-mode output."""
        self.eval()
        for m in list(self.modules()):
            if hasattr(m, "weight_orig"):
                nn.utils.remove_spectral_norm(m)  # uses stored u/v, no power iteration (== eval mode)
        for m in self.modules():
            if isinstance(m, BigGANBatchNorm):
                m.prepare(1.0)
        rgb = self.generator.conv_to_rgb
        slim = nn.Conv2d(rgb.in_channels, 3, 3, padding=1, bias=rgb.bias is not None)
        slim.weight.copy_(rgb.weight[:3])
        if rgb.bias is not None:
            slim.bias.copy_(rgb.bias[:3])
        self.generator.conv_to_rgb = slim
        # running stats are no longer needed once folded
        for m in self.modules():
            if isinstance(m, BigGANBatchNorm):
                del m.running_means, m.running_vars
        self.requires_grad_(False)
        return self

    def set_fp16_attention_guard(self, enabled):
        for m in self.modules():
            if isinstance(m, SelfAttn):
                m.upcast = enabled
