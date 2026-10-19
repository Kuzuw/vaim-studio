"""VAIM Studio engine: the Big Sleep optimisation loop (BigGAN latents steered by CLIP), rebuilt for speed.

Faithful to lucidrains/big-sleep's objective (same latent/class/similarity losses,
same EMA, same random-crop scheme) but restructured:

  * BigGAN and CLIP are frozen -> backward only flows to the latents
  * all cutouts are produced by ONE grid_sample call instead of a 128-iteration
    Python loop of crop + interpolate + cat
  * mixed precision for BigGAN (bf16 on RTX 30xx+, fp16 + loss scaling otherwise),
    CLIP in fp16 like the original; previews/finals are rendered in fp32
  * loss terms are vectorised (no per-latent Python loop, no topk(999) per step)
  * previews render the generator once with EMA latents (the original ran BigGAN +
    CLIP twice per save just to pick the "best" of a batch of one)
  * no host<->device syncs inside the step loop except at preview time
  * several images ("variations") optimise in a single batched pass
  * optional torch.compile
  * per-GPU autotuner (memory savers vs. speed), fp16 sanity probe for GTX 16xx,
    VRAM cap so Windows never silently spills into system RAM, NaN recovery
  * optional 1080px (or any size) output via Real-ESRGAN x2 + Lanczos
"""
from __future__ import annotations

import gc
import json
import math
import time
from dataclasses import asdict, dataclass, field
from typing import Callable, Optional

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

from . import weights
from .biggan import BigGAN
from .clip_model import build_model as build_clip, tokenize

CLIP_MEAN = (0.48145466, 0.4578275, 0.40821073)
CLIP_STD = (0.26862954, 0.26130258, 0.27577711)


@dataclass
class DreamSettings:
    prompt: str = "a lonely house in the woods"
    negative: str = ""                 # "|"-separated phrases to steer away from
    image_prompt: Optional[str] = None  # path to an image to blend into the prompt
    image_size: int = 512               # 128 / 256 / 512
    steps: int = 500                    # original default was 20 epochs x 1050 = 21000
    lr: float = 0.07
    cutouts: int = 128
    max_classes: Optional[int] = None
    class_temperature: float = 2.0
    ema_decay: float = 0.99
    seed: Optional[int] = None
    batch: int = 1
    preview_every: int = 10
    interp: str = "nearest"            # nearest = classic crunchy look, bilinear = smoother
    center_bias: bool = False
    clip_model: str = "ViT-B/32"
    precision: str = "auto"            # auto / bf16 / fp16 / fp32
    compile: bool = False
    checkpointing: str = "auto"        # auto / on / off  (BigGAN gradient checkpointing, saves VRAM)
    cutout_chunk: int = 0              # crops per CLIP pass; 0 = autotune
    output_size: int = 0               # final image size; 0 = native BigGAN size (e.g. 1080 = upscale)
    upscale_strength: float = 1.0      # 1 = full Real-ESRGAN detail, 0 = plain Lanczos
    auto_stop: bool = True             # finish early once the image has stopped changing
    auto_stop_min: int = 150           # never before this many steps
    auto_stop_threshold: float = 0.0025  # mean pixel change per 10 steps (0-1 scale) counted as "settled"
    init_latents: Optional[list] = field(default=None, repr=False)  # list of dicts from a previous run

    def to_meta(self):
        d = asdict(self)
        d.pop("init_latents", None)
        return d


@dataclass
class Preview:
    step: int
    total: int
    images: list           # PIL images, one per batch item (upscaled on the final preview)
    loss: float
    its: float             # optimisation steps per second
    final: bool = False
    latents: Optional[list] = None  # on final: [{"normu","cls","z","classes"} numpy] per image
    native: Optional[list] = None   # on final: the un-upscaled BigGAN images
    vram: float = 0.0               # GB reserved by PyTorch
    change: float = -1.0            # how much the image moved per 10 steps (0-1), -1 = unknown
    note: str = ""                  # e.g. "settled at step 210"


# ----------------------------------------------------------------- latent state

def differentiable_topk(x, k, temperature=1.0):
    n, dim = x.shape
    out = []
    for i in range(k):
        values, indices = (x / temperature).softmax(dim=-1).topk(1, dim=-1)
        out.append(torch.zeros_like(x).scatter_(-1, indices, values))
        if i != k - 1:
            x = x.scatter(-1, indices, float("-inf"))
    return torch.stack(out, 1).sum(1)


class Latents:
    """Per-layer noise + class logits for B images, with the original's bias-corrected EMA."""

    def __init__(self, b, num_latents, z_dim, num_classes, device, seed, decay, max_classes, temperature, init=None):
        g = torch.Generator(device="cpu")
        if seed is not None:
            g.manual_seed(seed)
        else:
            g.seed()
        if init:
            normu = torch.stack([torch.as_tensor(d["normu"]) for d in init[:b]]).float()
            cls = torch.stack([torch.as_tensor(d["cls"]) for d in init[:b]]).float()
            if normu.shape[0] < b:  # fill extra variations with fresh noise
                extra = b - normu.shape[0]
                normu = torch.cat([normu, torch.randn(extra, num_latents, z_dim, generator=g)])
                cls = torch.cat([cls, torch.randn(extra, num_latents, num_classes, generator=g) * 0.3 - 3.9])
        else:
            normu = torch.randn(b, num_latents, z_dim, generator=g)
            cls = torch.randn(b, num_latents, num_classes, generator=g) * 0.3 - 3.9
        self.normu = normu.to(device).requires_grad_()
        self.cls = cls.to(device).requires_grad_()
        self.params = [self.normu, self.cls]
        self.decay = decay
        self.max_classes = max_classes
        self.temperature = temperature
        self.accum = 1.0
        self.biased = [torch.zeros_like(p) for p in self.params]
        self.avg = [torch.zeros_like(p) for p in self.params]
        self.update()

    def activate(self, normu, cls):
        if self.max_classes:
            b, l, n = cls.shape
            classes = differentiable_topk(cls.reshape(b * l, n), self.max_classes, self.temperature).view(b, l, n)
        else:
            classes = torch.sigmoid(cls)
        return normu, classes

    def current(self):
        return self.activate(*self.params)

    def averaged(self):
        return self.activate(*self.avg)

    @torch.no_grad()
    def update(self):
        self.accum *= self.decay
        for p, b, a in zip(self.params, self.biased, self.avg):
            b.mul_(self.decay).add_(p.detach(), alpha=1 - self.decay)
            torch.div(b, 1 - self.accum, out=a)

    @torch.no_grad()
    def reset_avg_to_current(self):
        for p, b, a in zip(self.params, self.biased, self.avg):
            a.copy_(p)
            b.copy_(p * (1 - self.accum))

    @torch.no_grad()
    def export(self):
        z, classes = self.averaged()
        out = []
        for i in range(z.shape[0]):
            out.append({
                "normu": self.avg[0][i].float().cpu().numpy(),
                "cls": self.avg[1][i].float().cpu().numpy(),
                "z": z[i].float().cpu().numpy(),
                "classes": classes[i].float().cpu().numpy(),
            })
        return out


# ----------------------------------------------------------------------- losses

def latent_loss(z):
    """Vectorised version of the original per-latent loop. z: (B, L, D) -> (B,)"""
    L = z.shape[1]
    mean = z.mean(2)
    loss = (1 - z.std(2)).abs().mean(1) + mean.abs().mean(1) + 4 * z.square().mean((1, 2)).clamp(min=1.0)
    diffs = z - mean.unsqueeze(2)
    std = diffs.square().mean(2).sqrt().unsqueeze(2)
    zs = diffs / std
    skew = zs.pow(3).mean(2)
    kurt = zs.pow(4).mean(2) - 3.0
    return loss + (kurt.abs() + skew.abs()).sum(1) / L


def class_loss(classes):
    """== ((50 * topk(classes, 999, smallest)) ** 2).mean() without the sort. classes >= 0."""
    b, l, n = classes.shape
    sq = (classes * 50).square()
    return (sq.sum(2) - sq.amax(2)).sum(1) / (l * (n - 1))


def make_cutouts(img, n, size=224, mode="nearest", center_bias=False, _crop=None, _off=None):
    """All random square crops (50-95% of the image, resized to `size`) in one grid_sample."""
    b, c, s, _ = img.shape
    dev = img.device
    crop = _crop if _crop is not None else (s * (torch.randn(b, n, device=dev) * 0.3 + 0.8).clamp(0.5, 0.95)).floor()
    max_off = s - crop
    if _off is not None:
        off = _off
    elif center_bias:
        center = max_off / 2
        off = torch.randn(2, b, n, device=dev) * (center / 2) + center
        off = torch.round(off)
        uniform = torch.floor(torch.rand(2, b, n, device=dev) * (max_off + 1))
        off = torch.where((off < 0) | (off > max_off), uniform, off)
    else:
        off = torch.floor(torch.rand(2, b, n, device=dev) * (max_off + 1))
    scale = crop / s
    cy = (off[0] + crop / 2) / s * 2 - 1
    cx = (off[1] + crop / 2) / s * 2 - 1
    lin = (torch.arange(size, device=dev, dtype=torch.float32) + 0.5) / size * 2 - 1
    gx = cx[..., None] + scale[..., None] * lin              # (b, n, size)
    gy = cy[..., None] + scale[..., None] * lin
    grid = torch.stack(torch.broadcast_tensors(gx[:, :, None, :], gy[:, :, :, None]), dim=-1)  # (b,n,size,size,2)
    grid = grid.view(b, n * size, size, 2).to(img.dtype)
    out = F.grid_sample(img, grid, mode=mode, padding_mode="border", align_corners=False)  # (b,c,n*size,size)
    return out.view(b, c, n, size, size).transpose(1, 2).reshape(b * n, c, size, size)


def to_pil(t):
    """(3,H,W) float in [0,1] -> PIL"""
    a = (t.clamp(0, 1) * 255 + 0.5).to(torch.uint8).permute(1, 2, 0).cpu().numpy()
    return Image.fromarray(a)


# ----------------------------------------------------------------------- engine


class _Trainer:
    """One optimisation configuration: precision, checkpointing and CLIP chunk size."""

    def __init__(self, d: "Dreamer", s: DreamSettings, lat: Latents, pos, neg, amp, ckpt, chunk):
        self.d, self.s, self.lat, self.pos, self.neg = d, s, lat, pos, neg
        self.ckpt, self.chunk = ckpt, chunk
        self.profile = None  # set to [] to collect per-stage CUDA timings
        self.set_amp(amp)

    def set_amp(self, amp):
        dev = self.d.device
        self.amp = amp
        self.d.biggan.set_fp16_attention_guard(amp == torch.float16)
        fused = {"fused": True} if dev.type == "cuda" else {}
        self.opt = torch.optim.Adam(self.lat.params, lr=self.s.lr, **fused)
        self.scaler = torch.amp.GradScaler(dev.type, enabled=amp == torch.float16)

    def _mark(self):
        if self.profile is not None:
            e = torch.cuda.Event(enable_timing=True)
            e.record()
            self.profile.append(e)

    def timings(self):
        """ms per stage, averaged over profiled steps: generator fwd, CLIP fwd+bwd, generator bwd, optimiser"""
        if not self.profile:
            return None
        torch.cuda.synchronize(self.d.device)
        ev = self.profile
        n = len(ev) // 5
        tot = [0.0] * 4
        for k in range(n):
            g = ev[k * 5:(k + 1) * 5]
            for i in range(4):
                tot[i] += g[i].elapsed_time(g[i + 1])
        return [t / n for t in tot]

    def step(self):
        d, s, lat, dev = self.d, self.s, self.lat, self.d.device
        self._mark()
        gen, enc = d._maybe_compile(s)
        cdtype = d.clip_dtype
        mean, std = d._norm(cdtype)
        n, res, coef = s.cutouts, d.clip_res, 100.0  # loss_coef from the original
        d.biggan.generator.checkpoint = self.ckpt
        z, classes = lat.current()
        with torch.autocast(dev.type, dtype=self.amp or torch.float32, enabled=self.amp is not None):
            out = gen(z, classes)
        img = ((out + 1) / 2).to(cdtype)
        self._mark()
        # CLIP runs on a detached copy of the image, in chunks, accumulating d(loss)/d(image).
        # Same gradient as one big batch, but CLIP memory is bounded by the chunk size.
        leaf = img.detach().requires_grad_()
        sim_total = torch.zeros(s.batch, device=dev)
        done = 0
        while done < n:
            m = min(self.chunk, n - done)
            crops = make_cutouts(leaf, m, res, s.interp, s.center_bias)
            emb = F.normalize(enc((crops - mean) / std).float(), dim=-1).view(s.batch, m, -1)
            sim = -coef * (emb @ self.pos.T).sum(1).sum(1) / n        # (B,), mean over all n crops
            if self.neg is not None:
                sim = sim + coef * (emb @ self.neg.T).sum(1).sum(1) / n
            self.scaler.scale(sim.sum()).backward()
            sim_total += sim.detach()
            done += m
            del crops, emb, sim
        reg = latent_loss(z) + class_loss(classes)
        self._mark()
        torch.autograd.backward([self.scaler.scale(reg.sum()), img], [None, leaf.grad])
        self._mark()
        self.scaler.step(self.opt)
        self.scaler.update()
        self.opt.zero_grad(set_to_none=True)
        lat.update()
        self._mark()
        return (reg.detach() + sim_total).sum()


class Dreamer:
    def __init__(self, device=None, log=print):
        self.log = log
        # stage(key, fraction, detail): what the engine is doing, for the progress display.
        # key is "prepare" | "paint" | "upscale"; fraction is 0..1 of that stage, or -1 if unknown.
        self.stage = lambda key, frac=-1.0, detail="": None
        self._tune_n = 0
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        if self.device.type == "cuda" and self.device.index is None:
            self.device = torch.device("cuda", torch.cuda.current_device())  # some CUDA APIs need the index
        self.biggan: Optional[BigGAN] = None
        self.biggan_size = None
        self.clip = None
        self.clip_name = None
        self.upscaler = None
        self._compiled = {}
        self._norms = {}
        self.gen_fp16_ok = {}    # biggan size -> bool (probed)
        self.clip_fp16_ok = {}   # clip name -> bool (probed)
        self._tune_path = weights.app_home() / "autotune.json"
        if self.device.type == "cuda":
            torch.backends.cudnn.benchmark = True
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True
            self._cap_vram()

    # -- hardware ---------------------------------------------------------------
    def _cap_vram(self):
        """Cap PyTorch below the VRAM that is actually free.

        On Windows the NVIDIA driver (536+) by default *spills into system RAM* when VRAM runs
        out instead of failing - which makes every step crawl with no error. Capping the allocator
        makes PyTorch raise a normal out-of-memory error instead, which the engine recovers from by
        switching on memory savers.
        """
        try:
            idx = self.device.index
            free, total = torch.cuda.mem_get_info(idx)
            frac = float(min(max((free - 192 * 2**20) / total, 0.3), 0.97))
            torch.cuda.set_per_process_memory_fraction(frac, idx)
            self.vram_cap_gb = frac * total / 2**30
            self.log(f"VRAM: {free / 2**30:.2f} GB free of {total / 2**30:.2f} GB - PyTorch capped at {self.vram_cap_gb:.2f} GB")
        except Exception as e:
            self.vram_cap_gb = None
            self.log(f"(could not cap VRAM: {e})")

    def gpu_name(self):
        if self.device.type == "cuda":
            p = torch.cuda.get_device_properties(self.device)
            return f"{p.name} ({p.total_memory / 2**30:.0f} GB)"
        return "CPU (no CUDA GPU found - this will be very slow)"

    def vram_gb(self):
        if self.device.type != "cuda":
            return 0.0
        return torch.cuda.get_device_properties(self.device).total_memory / 2**30

    def has_tensor_cores(self):
        cc = self._cc()
        name = torch.cuda.get_device_name(self.device).upper() if self.device.type == "cuda" else ""
        return cc >= (8, 0) or (cc >= (7, 0) and "GTX" not in name)

    def _set_clip_half(self, half):
        want = torch.float16 if half else torch.float32
        if self.clip_dtype != want:
            self.clip.visual.to(want)
            self._compiled.clear()

    def _cc(self):
        return torch.cuda.get_device_capability(self.device) if self.device.type == "cuda" else (0, 0)

    def _free(self):
        gc.collect()
        if self.device.type == "cuda":
            torch.cuda.empty_cache()

    # -- models ---------------------------------------------------------------
    def ensure_models(self, image_size, clip_name, progress=None, cancelled=None):
        if self.biggan_size != image_size:
            key = f"biggan-{image_size}"
            weights.ensure(key, progress, cancelled, self.log)
            self.biggan = None
            self._compiled.clear()
            self._free()
            self.log(f"Loading BigGAN-deep {image_size} ...")
            self.stage("prepare", -1, f"Loading the BigGAN {image_size} painter onto your GPU")
            m = BigGAN.load(weights.file_path(key, "config.json"), weights.file_path(key, "pytorch_model.bin"), self.log)
            if self.device.type == "cuda":
                # channels_last feeds tensor cores (RTX); on GTX cards plain NCHW kernels are faster
                fmt = torch.channels_last if self.has_tensor_cores() else torch.contiguous_format
                m = m.to(self.device, memory_format=fmt)
            self.biggan, self.biggan_size = m, image_size
        if clip_name and self.clip_name != clip_name:
            key = f"clip-{clip_name}"
            weights.ensure(key, progress, cancelled, self.log)
            self.clip = None
            self._compiled.clear()
            self._free()
            self.log(f"Loading CLIP {clip_name} ...")
            self.stage("prepare", -1, "Loading CLIP, the model that reads your prompt")
            path = next(iter(weights.MODELS[key]["files"]))
            self.clip, self.clip_name = self._load_clip(weights.file_path(key, path)), clip_name

    def ensure_upscaler(self, progress=None, cancelled=None):
        if self.upscaler is None:
            from .upscale import Upscaler
            weights.ensure("esrgan-x2", progress, cancelled, self.log)
            self.log("Loading Real-ESRGAN x2 ...")
            self.stage("prepare", -1, "Loading the Real-ESRGAN upscaler")
            self.upscaler = Upscaler(weights.file_path("esrgan-x2", "RealESRGAN_x2plus.pth"), self.device,
                                     half=self.has_tensor_cores())

    def _load_clip(self, path):
        try:
            sd = torch.jit.load(str(path), map_location="cpu").state_dict()
        except Exception:
            sd = torch.load(str(path), map_location="cpu", weights_only=True)
            if "state_dict" in sd:
                sd = sd["state_dict"]
        sd = {k.removeprefix("module."): v for k, v in sd.items()}
        model = build_clip(sd).float().eval().requires_grad_(False)
        # Only the image tower lives on the GPU. The text tower runs once per prompt, so it stays
        # on the CPU in fp32 and costs no VRAM.
        if self.device.type == "cuda":
            if self.has_tensor_cores():
                model.visual.half()  # like the original; fast on RTX tensor cores
            # GTX cards (10xx/16xx): measured ~3.4x slower in fp16 on a GTX 1650, so fp32 there
            # (the autotuner still double-checks on each card)
            model.visual.to(self.device)
        return model

    @property
    def clip_dtype(self):
        return self.clip.visual.conv1.weight.dtype

    @property
    def clip_res(self):
        return self.clip.visual.input_resolution

    def _norm(self, dtype):
        if dtype not in self._norms:
            self._norms[dtype] = (torch.tensor(CLIP_MEAN, device=self.device, dtype=dtype)[:, None, None],
                                  torch.tensor(CLIP_STD, device=self.device, dtype=dtype)[:, None, None])
        return self._norms[dtype]

    # -- fp16 sanity probe -------------------------------------------------------------
    @torch.no_grad()
    def _probe_fp16(self):
        """GTX 16xx cards have a history of producing NaN/garbage in half precision with some
        driver/cuDNN combinations. Test both models once against fp32 and fall back if needed."""
        if self.device.type != "cuda" or self._cc() < (7, 0):
            return
        dev = self.device
        g = torch.Generator().manual_seed(0)
        L = self.biggan.config.num_latents
        z = torch.randn(1, L, self.biggan.config.z_dim, generator=g).to(dev)
        c = torch.sigmoid(torch.randn(1, L, self.biggan.config.num_classes, generator=g) * 0.3 - 3.9).to(dev)
        ref = self.biggan(z, c).float()
        if self.biggan_size not in self.gen_fp16_ok:
            self.biggan.set_fp16_attention_guard(True)
            with torch.autocast("cuda", dtype=torch.float16):
                out = self.biggan(z, c).float()
            err = (out - ref).abs().mean().item()
            ok = bool(torch.isfinite(out).all()) and err < 0.03
            self.gen_fp16_ok[self.biggan_size] = ok
            if not ok:
                self.log(f"BigGAN half precision is unreliable on this GPU (err {err:.3g}) - using fp32")
        if self.clip_name not in self.clip_fp16_ok and self.clip_dtype == torch.float16:
            x = F.interpolate((ref + 1) / 2, size=self.clip_res, mode="bilinear", align_corners=False)
            mean, std = self._norm(torch.float32)
            x = (x - mean) / std
            e16 = self.clip.visual(x.half()).float()
            self.clip.visual.float()
            e32 = self.clip.visual(x).float()
            cos = F.cosine_similarity(e16, e32).item() if torch.isfinite(e16).all() else float("nan")
            ok = cos > 0.99
            if ok:
                self.clip.visual.half()
            else:
                self.log(f"CLIP half precision is unreliable on this GPU (cos {cos:.3f}) - using fp32")
            self._compiled.clear()
            self.clip_fp16_ok[self.clip_name] = ok

    # -- precision ---------------------------------------------------------------
    def _amp_dtype(self, s):
        if self.device.type != "cuda" or s.precision == "fp32":
            return None
        if s.precision == "bf16":
            return torch.bfloat16
        if s.precision == "fp16":
            return torch.float16
        cc = self._cc()
        if cc >= (8, 0):
            return torch.bfloat16   # Ampere+: bf16 has fp32 range, no loss scaling needed
        if cc >= (7, 0) and self.gen_fp16_ok.get(self.biggan_size, True):
            return torch.float16    # Volta/Turing incl. GTX 16xx: fast fp16
        return None                 # Pascal & older (slow fp16), or fp16 failed the probe

    # -- memory / speed planning ---------------------------------------------------------
    def _tune_key(self, s):
        return "|".join(map(str, ["v3", self.gpu_name(), torch.__version__, s.image_size, self.clip_name,
                                  s.precision, s.cutouts, s.batch, self.gen_fp16_ok.get(self.biggan_size),
                                  self.clip_fp16_ok.get(self.clip_name)]))

    def _load_tune(self):
        try:
            return json.loads(self._tune_path.read_text())
        except Exception:
            return {}

    def _save_tune(self, key, val):
        data = self._load_tune()
        data[key] = val
        try:
            self._tune_path.parent.mkdir(parents=True, exist_ok=True)
            self._tune_path.write_text(json.dumps(data, indent=1))
        except Exception:
            pass

    def _time_config(self, s, pos, neg, amp, ckpt, chunk):
        """it/s over 3 steps after a warm-up step, or None if it does not fit in VRAM."""
        cfg = self.biggan.config
        lat = Latents(s.batch, cfg.num_latents, cfg.z_dim, cfg.num_classes, self.device, 0, s.ema_decay,
                      s.max_classes, s.class_temperature)
        t = _Trainer(self, s, lat, pos, neg, amp, ckpt, chunk)
        its = None
        self._tune_n += 1
        self.stage("prepare", -1, f"Finding the fastest settings for your GPU (one time only) · test {self._tune_n}")
        torch.cuda.reset_peak_memory_stats(self.device)
        try:
            t.step()                      # warm-up (cuDNN picks algorithms here)
            torch.cuda.synchronize(self.device)
            t0 = time.perf_counter()
            for _ in range(3):
                t.step()
            torch.cuda.synchronize(self.device)
            its = 3 / (time.perf_counter() - t0)
        except torch.cuda.OutOfMemoryError:
            pass
        peak = torch.cuda.max_memory_reserved(self.device) / 2**30
        t = lat = None
        self._free()
        clip = "fp16" if self.clip_dtype == torch.float16 else "fp32"
        desc = (f"BigGAN {str(amp).replace('torch.', '') if amp else 'fp32'}, CLIP {clip}, "
                f"{'checkpointing, ' if ckpt else ''}{chunk} crops/pass")
        self.log(f"  {desc}: " + (f"{its:.2f} it/s, {peak:.2f} GB" if its else "does not fit"))
        return its

    def _autotune(self, s, pos, neg, amp, should_stop):
        """Find the fastest configuration that fits: precision first, then memory savers.
        Each candidate runs 1 warm-up + 3 timed steps. Returns (amp, ckpt, chunk, clip_half)."""
        key = self._tune_key(s)
        clip_half0 = self.clip_dtype == torch.float16
        cached = self._load_tune().get(key)
        if cached:
            amp_c = {"fp16": torch.float16, "bf16": torch.bfloat16}.get(cached.get("amp"))
            return amp_c, cached["ckpt"], cached["chunk"], cached.get("clip_half", clip_half0)
        self.log("Tuning for your GPU (one time for these settings, ~20-40 s) ...")
        self._tune_n = 0
        n = s.cutouts
        stop = lambda: bool(should_stop and should_stop())  # noqa: E731
        ckpts = [c for c in (False, True) if not (s.checkpointing == "on" and not c or s.checkpointing == "off" and c)]
        results = []  # (its, amp, ckpt, chunk, clip_half)

        def try_fit(a, chunk):
            """time at the first checkpointing mode that fits"""
            for ck in ckpts:
                if stop():
                    return None
                its = self._time_config(s, pos, neg, a, ck, chunk)
                if its:
                    results.append((its, a, ck, chunk, self.clip_dtype == torch.float16))
                    return its, ck
            return None

        # 1) precision. On RTX the defaults (bf16/fp16 + fp16 CLIP) are right; GTX cards without
        #    tensor cores are often faster in fp32, and some (GTX 16xx) break in fp16.
        probe_chunk = min(n, 16)
        amp_opts = [amp]
        clip_opts = [clip_half0]
        if s.precision == "auto" and not self.has_tensor_cores():
            if amp is None and self._cc() >= (7, 0) and self.gen_fp16_ok.get(self.biggan_size, True):
                amp_opts.append(torch.float16)
            elif amp is not None:
                amp_opts.append(None)
            if self.clip_fp16_ok.get(self.clip_name, True) and self._cc() >= (7, 0):
                clip_opts = [clip_half0, not clip_half0]
        if len(amp_opts) * len(clip_opts) > 1:
            for ch in clip_opts:
                self._set_clip_half(ch)
                for a in amp_opts:
                    try_fit(a, probe_chunk)
            if results:
                best = max(results, key=lambda r: r[0])
                amp, clip_half0 = best[1], best[4]
            self._set_clip_half(clip_half0)

        # 2) memory: largest CLIP chunk that fits (fewer passes = faster), stop once it gets slower
        chunks = sorted({n, max(1, math.ceil(n / 2)), max(1, math.ceil(n / 4)), min(n, 8)}, reverse=True)
        prev = None
        for chunk in chunks:
            if stop():
                break
            done = next((r for r in results if r[1] == amp and r[3] == chunk and r[4] == clip_half0), None)
            r = (done[0], done[2]) if done else try_fit(amp, chunk)
            if r is None:
                continue
            if prev is not None and r[0] < prev * 0.97:
                break
            prev = r[0] if prev is None else max(prev, r[0])

        mine = [r for r in results if r[4] == clip_half0 and r[1] == amp]
        if not mine:
            return amp, True, 1, clip_half0  # let the runtime OOM handler report the problem
        its, amp, ckpt, chunk, clip_half = max(mine, key=lambda r: r[0])
        amp_name = {torch.float16: "fp16", torch.bfloat16: "bf16"}.get(amp)
        if not stop():
            self._save_tune(key, {"amp": amp_name, "ckpt": ckpt, "chunk": chunk, "clip_half": clip_half,
                                  "its": round(its, 3)})
        return amp, ckpt, chunk, clip_half

    def _memory_plan(self, s, pos, neg, amp, should_stop):
        """-> (amp, ckpt, chunk, clip_half)"""
        clip_half = self.clip_dtype == torch.float16
        if s.cutout_chunk and s.checkpointing != "auto":
            return amp, s.checkpointing == "on", max(1, min(s.cutouts, s.cutout_chunk)), clip_half
        if s.cutout_chunk:
            return amp, False, max(1, min(s.cutouts, s.cutout_chunk)), clip_half
        if self.device.type != "cuda":
            return amp, False, s.cutouts, clip_half
        return self._autotune(s, pos, neg, amp, should_stop)

    def _maybe_compile(self, s):
        if not s.compile:
            return self.biggan, self.clip.encode_image
        if "gen" not in self._compiled:
            try:
                import importlib.util
                if importlib.util.find_spec("triton") is None:  # torch.compile needs it (Windows: triton-windows)
                    raise ImportError("triton is not installed")
                self.log("Compiling models with torch.compile (first run takes a minute or two) ...")
                self._compiled["gen"] = torch.compile(self.biggan, dynamic=False)
                self._compiled["clip"] = torch.compile(self.clip.encode_image, dynamic=False)
            except Exception as e:
                self.log(f"torch.compile unavailable ({type(e).__name__}); using eager mode")
                self._compiled["gen"], self._compiled["clip"] = self.biggan, self.clip.encode_image
        return self._compiled["gen"], self._compiled["clip"]

    # -- encoding ---------------------------------------------------------------
    @torch.no_grad()
    def _encode_text(self, text):
        m = self.clip
        tok = tokenize(text)
        x = m.token_embedding(tok) + m.positional_embedding
        x = m.transformer(x.permute(1, 0, 2)).permute(1, 0, 2)
        x = m.ln_final(x)
        e = x[torch.arange(x.shape[0]), tok.argmax(dim=-1)] @ m.text_projection
        return F.normalize(e.float(), dim=-1).to(self.device)

    @torch.no_grad()
    def _encode_image(self, path):
        img = Image.open(path).convert("RGB")
        r = self.clip_res
        w, h = img.size
        sc = r / min(w, h)
        img = img.resize((max(r, round(w * sc)), max(r, round(h * sc))), Image.BICUBIC)
        w, h = img.size
        img = img.crop(((w - r) // 2, (h - r) // 2, (w - r) // 2 + r, (h - r) // 2 + r))
        t = torch.from_numpy(np.asarray(img)).permute(2, 0, 1).float().div(255).unsqueeze(0).to(self.device)
        mean, std = self._norm(torch.float32)
        t = (t - mean) / std
        return F.normalize(self.clip.encode_image(t.to(self.clip_dtype)).float(), dim=-1)

    def _targets(self, s: DreamSettings):
        img_e = self._encode_image(s.image_prompt) if s.image_prompt else None

        def phrases(text):
            ps = [p.strip() for p in (text or "").split("|") if p.strip()]
            out = []
            for p in ps:
                e = self._encode_text(p)
                if img_e is not None:
                    e = F.normalize((e + img_e) / 2, dim=-1)  # original: average of text + image encodings
                out.append(e)
            return out

        pos = phrases(s.prompt)
        if not pos and img_e is not None:
            pos = [img_e]
        if not pos:
            raise ValueError("Enter a prompt (or pick an image prompt).")
        neg = phrases(s.negative)
        return torch.cat(pos), (torch.cat(neg) if neg else None)

    # -- rendering ---------------------------------------------------------------
    @torch.no_grad()
    def render(self, z, classes, fast=False):
        """fp32 render for finals (no banding); fast=True uses the training precision for previews."""
        z = torch.as_tensor(z, device=self.device).float()
        classes = torch.as_tensor(classes, device=self.device).float()
        if z.dim() == 2:
            z, classes = z[None], classes[None]
        amp = self._amp_dtype(DreamSettings()) if fast else None
        with torch.autocast(self.device.type, dtype=amp or torch.float32, enabled=amp is not None):
            img = self.biggan(z, classes)
        return [to_pil(im) for im in (img.float() + 1) / 2]

    def finish_size(self, images, s: DreamSettings, progress=None, cancelled=None):
        """Upscale native BigGAN images to s.output_size (Real-ESRGAN x2 blended with Lanczos)."""
        target = s.output_size
        if not target or target <= images[0].width:
            return images
        if s.upscale_strength <= 0:
            return [im.resize((target, target), Image.LANCZOS) for im in images]
        self.ensure_upscaler(progress, cancelled)
        return [self._upscale_to(im, target, target, s.upscale_strength, cancelled, k, len(images))
                for k, im in enumerate(images)]

    def _upscale_to(self, im, tw, th, strength, cancelled=None, k=0, n=1, hard_cancel=False):
        """Real-ESRGAN x2 passes until within 1.5x of the target (512 -> 1080: 1, -> 2K: 2, -> 4K: 3),
        Lanczos to the exact size, blended with a plain Lanczos enlargement by `strength`.
        hard_cancel: stopping raises Cancelled mid-pass instead of finishing with fewer passes."""
        target = max(tw, th)
        passes, w = 0, max(im.size)
        while True:
            passes += 1
            w *= 2
            if w * 1.5 >= target:
                break
        per_img = sum(4 ** q for q in range(passes))  # each pass has 4x the pixels of the previous one
        up = im
        for p in range(passes):
            if cancelled and cancelled():
                if hard_cancel:
                    raise weights.Cancelled()
                break
            t0 = time.perf_counter()
            self.log(f"Upscaling image {k + 1}/{n}: pass {p + 1}/{passes} ({up.width} -> {up.width * 2}px) ...")
            done = k * per_img + sum(4 ** q for q in range(p))
            what = (f"image {k + 1} of {n} · " if n > 1 else "") + \
                f"pass {p + 1} of {passes} · {up.width} → {up.width * 2} px"

            def on_tile(i, m, done=done, p=p, what=what):
                if hard_cancel and cancelled and cancelled():
                    raise weights.Cancelled()
                self.stage("upscale", (done + 4 ** p * i / m) / (per_img * n), what)
            on_tile(0, 1)
            up = self.upscaler.run(up, on_tile)
            self.log(f"  pass {p + 1} took {time.perf_counter() - t0:.1f} s")
        if up.size != (tw, th):
            up = up.resize((tw, th), Image.LANCZOS)
        if strength < 1:
            up = Image.blend(im.resize((tw, th), Image.LANCZOS), up, strength)
        return up

    def upscale_image(self, image, target, strength=0.75, latents=None, image_size=None,
                      progress=None, cancelled=None):
        """Upscale a finished image so its long side is `target` px. With the image's saved latents,
        BigGAN first re-renders the original native canvas: a sharper source than a file that was
        already upscaled once."""
        w, h = image.size
        sc = target / max(w, h)
        tw, th = max(1, round(w * sc)), max(1, round(h * sc))
        src = image
        if latents is not None and image_size and w == h:
            self.ensure_models(int(image_size), None, progress, cancelled)
            self.stage("prepare", -1, "Re-rendering the original canvas from its saved latents (sharpest source)")
            src = self.render(latents["z"], latents["classes"])[0]
        if strength <= 0:
            return src.resize((tw, th), Image.LANCZOS)
        self.ensure_upscaler(progress, cancelled)
        self.stage("upscale", 0.0, "starting")
        return self._upscale_to(src, tw, th, strength, cancelled, hard_cancel=True)

    # -- main loop -------------------------------------------------------------
    def run(self, s: DreamSettings, on_preview: Callable[[Preview], None] = None,
            should_stop: Callable[[], bool] = None, progress=None):
        dev = self.device
        self.stage("prepare", -1, "Checking what your GPU handles best")
        self._probe_fp16()
        if s.seed is not None:
            torch.manual_seed(s.seed)
        self.stage("prepare", -1, "Reading your prompt")
        pos, neg = self._targets(s)
        amp = self._amp_dtype(s)
        amp, ckpt, chunk, clip_half = self._memory_plan(s, pos, neg, amp, should_stop)
        self._set_clip_half(clip_half)
        if s.seed is not None:
            torch.manual_seed(s.seed)  # autotune consumed random numbers
        cfg = self.biggan.config
        lat = Latents(s.batch, cfg.num_latents, cfg.z_dim, cfg.num_classes, dev, s.seed, s.ema_decay,
                      s.max_classes, s.class_temperature, s.init_latents)
        tr = _Trainer(self, s, lat, pos, neg, amp, ckpt, chunk)
        self.log(f"{'Checkpointing on, ' if ckpt else ''}{chunk}/{s.cutouts} crops per CLIP pass, "
                 f"BigGAN {str(amp).replace('torch.', '') if amp else 'fp32'}, CLIP {str(self.clip_dtype).replace('torch.', '')}")

        def snapshot():
            return [t.detach().clone() for t in lat.params + lat.avg + lat.biased] + [lat.accum]

        def restore(snap):
            with torch.no_grad():
                for t, v in zip(lat.params + lat.avg + lat.biased, snap[:-1]):
                    t.copy_(v)
            lat.accum = snap[-1]

        good, good_step = snapshot(), 0

        def safe_step():
            while True:
                try:
                    return tr.step()
                except torch.cuda.OutOfMemoryError:
                    pass  # handled below, once the traceback (and its tensors) is released
                tr.opt.zero_grad(set_to_none=True)
                self._free()
                if not tr.ckpt and dev.type == "cuda":
                    tr.ckpt = True
                    self.log("Out of VRAM - turning on gradient checkpointing and retrying")
                elif tr.chunk > 1:
                    tr.chunk = max(1, tr.chunk // 2)
                    self.log(f"Out of VRAM - lowering to {tr.chunk} crops per CLIP pass and retrying")
                else:
                    raise MemoryError(
                        "Not enough GPU memory even with every memory saver on. "
                        "Try a smaller image size, fewer variations, or the ViT-B/32 CLIP model.")

        t0 = time.perf_counter()
        last_t, last_step = t0, 0
        loss_val = float("nan")
        step = 0
        note = ""
        prev_small, prev_small_step, calm = None, 0, 0
        patience = 3
        early = {1, 5} if s.preview_every > 5 else set()  # show something right away
        self.stage("paint", 0.0, f"step 0 of {s.steps} · the first steps warm up the GPU")
        while step < s.steps:
            if should_stop and should_stop():
                break
            step += 1
            if step == 4 and dev.type == "cuda":
                tr.profile = []           # time steps 4-8 per stage (after cuDNN warm-up)
            loss_t = safe_step()
            if step == 8 and tr.profile is not None:
                t = tr.timings()
                tr.profile = None
                if t:
                    self.log("Step breakdown: BigGAN forward {:.0f} ms, CLIP fwd+bwd {:.0f} ms, "
                             "BigGAN backward {:.0f} ms, optimiser {:.0f} ms".format(*t))
            self.stage("paint", step / s.steps, f"step {step} of {s.steps}")
            if step % s.preview_every == 0 or step == s.steps or step in early:
                loss_val = loss_t.item() / s.batch  # the only per-preview sync
                if not math.isfinite(loss_val):
                    # fall back one precision level and resume from the last good state
                    if tr.amp is not None:
                        self.log("Numerical trouble in half precision - switching BigGAN to fp32")
                        self.gen_fp16_ok[self.biggan_size] = False
                        tr.set_amp(None)
                    elif self.clip_dtype == torch.float16:
                        self.log("Numerical trouble in CLIP half precision - switching CLIP to fp32")
                        self.clip.visual.float()
                        self.clip_fp16_ok[self.clip_name] = False
                        self._compiled.clear()
                        tr.set_amp(None)
                    else:
                        raise RuntimeError("The optimisation diverged (NaN). Try a lower learning rate.")
                    restore(good)
                    step = good_step
                    last_t, last_step = time.perf_counter(), step
                    continue
                good, good_step = snapshot(), step
                now = time.perf_counter()
                its = (step - last_step) / max(now - last_t, 1e-6)
                last_t, last_step = now, step
                z, classes = lat.averaged()
                imgs = self.render(z, classes, fast=True)
                # convergence: how much did the (EMA) image change since the last preview?
                small = np.stack([np.asarray(im.resize((64, 64), Image.BILINEAR), np.float32) / 255 for im in imgs])
                change = -1.0
                if prev_small is not None and step > prev_small_step:
                    change = float(np.abs(small - prev_small).mean()) * 10 / (step - prev_small_step)
                prev_small, prev_small_step = small, step
                if step % 50 == 0 and change >= 0:
                    self.log(f"step {step}: image change {change:.4f} per 10 steps, loss {loss_val:.2f}")
                if s.auto_stop and step >= s.auto_stop_min and 0 <= change < s.auto_stop_threshold:
                    calm += 1
                else:
                    calm = 0
                if on_preview:
                    vram = torch.cuda.memory_reserved(dev) / 2**30 if dev.type == "cuda" else 0.0
                    on_preview(Preview(step, s.steps, imgs, loss_val, its, vram=vram, change=change))
                if calm >= patience and step < s.steps:
                    note = f"settled at step {step} of {s.steps}"
                    self.log(f"Image has stopped changing - {note} (auto-stop)")
                    break
        elapsed = time.perf_counter() - t0
        self.biggan.generator.checkpoint = False
        z, classes = lat.averaged()
        native = self.render(z, classes)
        latents = lat.export()
        tr = lat = None  # release optimiser state before upscaling
        self._free()
        if s.output_size and s.output_size > native[0].width:
            self.log(f"Upscaling to {s.output_size}x{s.output_size} ...")
            self.stage("upscale", 0.0, "starting")
        images = self.finish_size(native, s, progress, should_stop)
        final = Preview(step, s.steps, images, loss_val, step / max(elapsed, 1e-6), final=True,
                        latents=latents, native=native, note=note)
        if on_preview:
            on_preview(final)
        return final
