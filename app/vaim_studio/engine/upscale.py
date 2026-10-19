"""Real-ESRGAN x2 (RRDBNet) upscaler - the same model family people used on BigGAN/VQGAN
outputs in 2021-22. Architecture from xinntao/Real-ESRGAN & BasicSR (BSD-3-Clause)."""
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image


class ResidualDenseBlock(nn.Module):
    def __init__(self, num_feat=64, num_grow_ch=32):
        super().__init__()
        g = num_grow_ch
        self.conv1 = nn.Conv2d(num_feat, g, 3, 1, 1)
        self.conv2 = nn.Conv2d(num_feat + g, g, 3, 1, 1)
        self.conv3 = nn.Conv2d(num_feat + 2 * g, g, 3, 1, 1)
        self.conv4 = nn.Conv2d(num_feat + 3 * g, g, 3, 1, 1)
        self.conv5 = nn.Conv2d(num_feat + 4 * g, num_feat, 3, 1, 1)
        self.lrelu = nn.LeakyReLU(0.2, inplace=True)

    def forward(self, x):
        x1 = self.lrelu(self.conv1(x))
        x2 = self.lrelu(self.conv2(torch.cat((x, x1), 1)))
        x3 = self.lrelu(self.conv3(torch.cat((x, x1, x2), 1)))
        x4 = self.lrelu(self.conv4(torch.cat((x, x1, x2, x3), 1)))
        x5 = self.conv5(torch.cat((x, x1, x2, x3, x4), 1))
        return x5 * 0.2 + x


class RRDB(nn.Module):
    def __init__(self, num_feat, num_grow_ch=32):
        super().__init__()
        self.rdb1 = ResidualDenseBlock(num_feat, num_grow_ch)
        self.rdb2 = ResidualDenseBlock(num_feat, num_grow_ch)
        self.rdb3 = ResidualDenseBlock(num_feat, num_grow_ch)

    def forward(self, x):
        return self.rdb3(self.rdb2(self.rdb1(x))) * 0.2 + x


class RRDBNet(nn.Module):
    def __init__(self, num_in_ch=3, num_out_ch=3, scale=2, num_feat=64, num_block=23, num_grow_ch=32):
        super().__init__()
        self.scale = scale
        if scale == 2:
            num_in_ch *= 4
        elif scale == 1:
            num_in_ch *= 16
        self.conv_first = nn.Conv2d(num_in_ch, num_feat, 3, 1, 1)
        self.body = nn.Sequential(*[RRDB(num_feat, num_grow_ch) for _ in range(num_block)])
        self.conv_body = nn.Conv2d(num_feat, num_feat, 3, 1, 1)
        self.conv_up1 = nn.Conv2d(num_feat, num_feat, 3, 1, 1)
        self.conv_up2 = nn.Conv2d(num_feat, num_feat, 3, 1, 1)
        self.conv_hr = nn.Conv2d(num_feat, num_feat, 3, 1, 1)
        self.conv_last = nn.Conv2d(num_feat, num_out_ch, 3, 1, 1)
        self.lrelu = nn.LeakyReLU(0.2, inplace=True)

    def forward(self, x):
        if self.scale == 2:
            feat = F.pixel_unshuffle(x, 2)
        elif self.scale == 1:
            feat = F.pixel_unshuffle(x, 4)
        else:
            feat = x
        feat = self.conv_first(feat)
        feat = feat + self.conv_body(self.body(feat))
        feat = self.lrelu(self.conv_up1(F.interpolate(feat, scale_factor=2, mode="nearest")))
        feat = self.lrelu(self.conv_up2(F.interpolate(feat, scale_factor=2, mode="nearest")))
        return self.conv_last(self.lrelu(self.conv_hr(feat)))


class Upscaler:
    def __init__(self, weights_path, device, half=True):
        sd = torch.load(str(weights_path), map_location="cpu", weights_only=True)
        sd = sd.get("params_ema", sd.get("params", sd))
        self.net = RRDBNet(scale=2)
        self.net.load_state_dict(sd, strict=True)
        self.device = device
        self.half = half and device.type == "cuda"
        self.net = self.net.eval().requires_grad_(False).to(device)
        if self.half:
            self.net.half()
        self.tile = 0  # 0 = whole image; lowered automatically on out-of-memory

    @torch.no_grad()
    def _forward(self, x, tile, on_tile=None):
        if not tile or (x.shape[-1] <= tile and x.shape[-2] <= tile):
            return self.net(x)
        # overlapped tiles, each with a margin that is cropped away (no seams)
        t, pad = tile, 32
        _, c, h, w = x.shape
        out = x.new_zeros(1, 3, h * 2, w * 2)
        n = -(-h // t) * -(-w // t)
        i = 0
        for y0 in range(0, h, t):
            for x0 in range(0, w, t):
                if on_tile and i:
                    if x.is_cuda:
                        torch.cuda.synchronize(x.device)  # report finished tiles, not queued ones
                    on_tile(i, n)
                i += 1
                y1, x1 = min(y0 + t, h), min(x0 + t, w)
                py0, px0 = max(y0 - pad, 0), max(x0 - pad, 0)
                py1, px1 = min(y1 + pad, h), min(x1 + pad, w)
                o = self.net(x[:, :, py0:py1, px0:px1])
                oy, ox = (y0 - py0) * 2, (x0 - px0) * 2
                out[:, :, y0 * 2:y1 * 2, x0 * 2:x1 * 2] = o[:, :, oy:oy + (y1 - y0) * 2, ox:ox + (x1 - x0) * 2]
        return out

    @torch.no_grad()
    def run(self, im: Image.Image, on_tile=None) -> Image.Image:
        """x2 upscale. on_tile(done, total) reports progress while tiles are processed."""
        x = torch.from_numpy(np.array(im.convert("RGB"))).permute(2, 0, 1)[None].float().div(255).to(self.device)
        tile = self.tile or (0 if max(im.size) <= 640 else 512)  # big inputs go straight to tiles
        while True:
            try:
                y = self._forward(x.half() if self.half else x, tile, on_tile).float()
                if self.half and not torch.isfinite(y).all():  # GTX 16xx fp16 trouble -> fp32
                    self.half = False
                    self.net.float()
                    continue
                break
            except torch.cuda.OutOfMemoryError:
                pass
            torch.cuda.empty_cache()
            if tile == 64:
                raise MemoryError("Not enough GPU memory to upscale.")
            tile = 512 if not tile else max(64, tile // 2)
            self.tile = tile  # remember for the next image
        a = (y[0].clamp(0, 1) * 255 + 0.5).byte().permute(1, 2, 0).cpu().numpy()
        if on_tile:
            on_tile(1, 1)
        return Image.fromarray(a)
