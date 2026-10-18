"""VAIM Studio self-test and benchmark.

    python tools/selftest.py            correctness checks (no downloads, CPU is fine)
    python tools/selftest.py --bench    + time the ORIGINAL big-sleep step vs the new one on your GPU
                                        (downloads the models on first use)
Options: --size 256 --cutouts 64 --steps 20
"""
import argparse
import copy
import os
import sys
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "tools"))

import torch
import torch.nn.functional as F

import legacy_biggan as legacy
from vaim_studio.engine.biggan import BigGAN, BigGANConfig
from vaim_studio.engine.dreamer import class_loss, differentiable_topk, latent_loss, make_cutouts

OK, BAD = "\033[92mPASS\033[0m", "\033[91mFAIL\033[0m"
if os.name == "nt":
    os.system("")  # enable ANSI colours in the Windows console
failures = 0


def check(name, err, tol):
    global failures
    good = err <= tol
    failures += not good
    print(f"  [{OK if good else BAD}] {name:<52} max err {err:.2e} (tol {tol:.0e})")


CFG = dict(output_dim=128, z_dim=128, class_embed_dim=128, channel_width=32, num_classes=1000,
           layers=[[False, 16, 16], [True, 16, 16], [False, 16, 16], [True, 16, 8], [False, 8, 8],
                   [True, 8, 4], [False, 4, 4], [True, 4, 2], [False, 2, 2], [True, 2, 1]],
           attention_layer_position=8, eps=1e-4, n_stats=51)


def random_legacy(cfg):
    m = legacy.BigGAN(legacy.BigGANConfig.from_dict(cfg))
    with torch.no_grad():
        for n, p in m.named_parameters():
            p.copy_(torch.full_like(p, 0.3) if n.endswith("gamma") else torch.randn_like(p) * 0.05)
        for n, b in m.named_buffers():
            if n.endswith("running_means"):
                b.normal_(0, 0.1)
            elif n.endswith("running_vars"):
                b.uniform_(0.5, 1.5)
        # Spectral norm divides each weight by u^T W v. With random u/v that can be ~0 and blow the
        # network up to NaN, so use the true top singular vectors (what the trained model has).
        for mod in m.modules():
            if hasattr(mod, "weight_orig"):
                W = mod.weight_orig.reshape(mod.weight_orig.shape[0], -1)
                U, S, Vh = torch.linalg.svd(W, full_matrices=False)
                mod.weight_u.copy_(U[:, 0])
                mod.weight_v.copy_(Vh[0])
    return m.eval()


def correctness():
    print("\nCorrectness (new code vs. the original big-sleep code)")
    torch.manual_seed(0)
    leg = random_legacy(CFG)
    new = BigGAN(BigGANConfig(**CFG))
    missing, _ = new.load_state_dict(leg.state_dict(), strict=False)
    new.bake()
    L = len(CFG["layers"]) + 1
    z = torch.randn(3, L, 128)
    cls = torch.sigmoid(torch.randn(3, L, 1000) - 2)
    with torch.no_grad():
        ref = torch.cat([leg(z[i], cls[i], 1.0) for i in range(3)])
        out = new(z, cls)
    check("BigGAN baked/batched output == original (batch of 3)", (ref - out).abs().max().item(), 1e-4)

    # gradient w.r.t. latents
    zr, cr = z[:1].clone().requires_grad_(), cls[:1].clone().requires_grad_()
    (leg(zr[0], cr[0], 1.0) ** 2).mean().backward()
    zn, cn = z[:1].clone().requires_grad_(), cls[:1].clone().requires_grad_()
    (new(zn, cn) ** 2).mean().backward()
    check("BigGAN latent gradients == original", max((zr.grad - zn.grad).abs().max().item(), (cr.grad - cn.grad).abs().max().item()), 1e-5)

    new.generator.checkpoint = True
    zc = z[:1].clone().requires_grad_()
    (new(zc, cls[:1]) ** 2).mean().backward()
    new.generator.checkpoint = False
    check("gradient checkpointing gives identical gradients", (zc.grad - zn.grad).abs().max().item(), 1e-6)

    # losses
    lat = torch.randn(2, L, 128) * 1.3 + 0.1
    ref = []
    for b in range(2):
        latents = lat[b]
        l = torch.abs(1 - torch.std(latents, dim=1)).mean() + torch.abs(torch.mean(latents, dim=1)).mean() + \
            4 * torch.max(torch.square(latents).mean(), torch.tensor(1))
        for array in latents:
            mean = torch.mean(array); diffs = array - mean
            std = torch.pow(torch.mean(torch.pow(diffs, 2.0)), 0.5); zs = diffs / std
            l = l + torch.abs(torch.mean(torch.pow(zs, 4.0)) - 3.0) / L + torch.abs(torch.mean(torch.pow(zs, 3.0))) / L
        ref.append(l)
    check("latent loss (vectorised) == original loop", (torch.stack(ref) - latent_loss(lat)).abs().max().item(), 1e-4)

    for name, c in [("sigmoid", torch.sigmoid(torch.randn(2, L, 1000) * 0.3 - 3.9)),
                    ("top-15", differentiable_topk(torch.randn(2 * L, 1000), 15, 2.0).view(2, L, 1000))]:
        ref = torch.stack([((50 * torch.topk(c[b], largest=False, dim=1, k=999)[0]) ** 2).mean() for b in range(2)])
        check(f"class loss without topk == original ({name})", ((ref - class_loss(c)).abs() / ref.abs()).max().item(), 1e-5)

    # cutouts: one grid_sample == crop + interpolate per cutout
    img = torch.rand(2, 3, 128, 128)
    crop = torch.tensor([[64., 100.], [121., 77.]])
    off = torch.stack([torch.tensor([[3., 20.], [0., 51.]]), torch.tensor([[60., 0.], [7., 30.]])])
    got = make_cutouts(img, 2, 48, "bilinear", _crop=crop, _off=off).view(2, 2, 3, 48, 48)
    err = 0
    for b in range(2):
        for i in range(2):
            s, oy, ox = int(crop[b, i]), int(off[0, b, i]), int(off[1, b, i])
            r = F.interpolate(img[b:b + 1, :, oy:oy + s, ox:ox + s], (48, 48), mode="bilinear", align_corners=False)
            err = max(err, (r[0] - got[b, i]).abs().max().item())
    check("batched cutouts == per-crop crop+interpolate (bilinear)", err, 1e-4)

    # chunked image-gradient trick == direct backward
    zz = z[:2].clone().requires_grad_()
    out = new(zz, cls[:2]); w = torch.randn_like(out)
    (out * w).sum().backward(); direct = zz.grad.clone(); zz.grad = None
    out = new(zz, cls[:2]); leaf = out.detach().requires_grad_()
    (leaf[:, :, :64] * w[:, :, :64]).sum().backward(); (leaf[:, :, 64:] * w[:, :, 64:]).sum().backward()
    torch.autograd.backward([out], [leaf.grad])
    check("chunked CLIP backward == single backward", (zz.grad - direct).abs().max().item(), 1e-6)


def smoke():
    """Run the whole engine (probe, planning, chunked CLIP, EMA, previews, upscaling) on tiny
    random models, so the pipeline is exercised without downloading anything."""
    print("\nPipeline smoke test (tiny random models, CPU)")
    from vaim_studio.engine import Dreamer, DreamSettings
    from vaim_studio.engine.clip_model import CLIP
    from vaim_studio.engine.upscale import RRDBNet, Upscaler
    from PIL import Image
    torch.manual_seed(0)
    d = Dreamer(device="cpu", log=lambda m: None)
    leg = random_legacy(CFG)
    g = BigGAN(BigGANConfig(**CFG))
    g.load_state_dict(leg.state_dict(), strict=False)
    d.biggan, d.biggan_size = g.bake(), 128
    clip = CLIP(32, 32, 2, 64, 16, 77, 49408, 64, 1, 1).float().eval().requires_grad_(False)
    d.clip, d.clip_name = clip, "tiny"
    seen = []
    s = DreamSettings(prompt="a lantern | fog", negative="text", image_size=128, steps=6, cutouts=6,
                      cutout_chunk=4, preview_every=3, seed=3, batch=2)
    out = d.run(s, on_preview=seen.append)
    ok = (len(seen) == 3 and seen[-1].final and len(out.images) == 2 and out.images[0].size == (128, 128)
          and all(abs(p.loss) < 1e9 for p in seen) and out.latents[0]["z"].shape == (11, 128))
    check("dream run: 2 variations, chunked CLIP, previews, latents", 0.0 if ok else 1.0, 0.5)
    s2 = DreamSettings(prompt="more fog", image_size=128, steps=3, cutouts=4, preview_every=10, seed=3,
                       init_latents=out.latents[:1])
    out2 = d.run(s2)
    check("continue from saved latents", 0.0 if out2.images[0].size == (128, 128) else 1.0, 0.5)
    s4 = DreamSettings(prompt="fog", image_size=128, steps=60, cutouts=4, preview_every=3, seed=3,
                       auto_stop=True, auto_stop_min=3, auto_stop_threshold=10.0)
    out4 = d.run(s4)
    check("auto-stop ends a settled run early", 0.0 if out4.step == 12 and "settled" in out4.note else 1.0, 0.5)
    # upscaler: tiny RRDBNet, whole-image and tiled paths
    up = Upscaler.__new__(Upscaler)
    up.net, up.device, up.half, up.tile = RRDBNet(scale=2, num_block=1).eval(), torch.device("cpu"), False, 0
    a = up.run(Image.new("RGB", (96, 96), (90, 40, 140)))
    up.tile = 32
    b = up.run(Image.new("RGB", (96, 96), (90, 40, 140)))
    check("Real-ESRGAN x2 whole + tiled output size", 0.0 if a.size == b.size == (192, 192) else 1.0, 0.5)
    s3 = DreamSettings(output_size=300, upscale_strength=0.0)
    r = d.finish_size([Image.new("RGB", (128, 128))], s3)
    check("Lanczos-only upscale to custom size", 0.0 if r[0].size == (300, 300) else 1.0, 0.5)


def ui_test():
    """Build the real window off-screen with a fake engine, drive the main flows, save a screenshot."""
    print("\nUser interface test (off-screen)")
    import tempfile
    import numpy as np
    from PIL import Image
    try:
        from PySide6.QtCore import QSettings, Qt
        from PySide6.QtWidgets import QApplication
    except Exception as e:
        check(f"PySide6 import ({e})", 1.0, 0.5)
        return
    from vaim_studio.engine import Preview
    from vaim_studio.engine.weights import app_home
    from vaim_studio.ui import theme
    from vaim_studio.ui.main_window import MainWindow
    from vaim_studio import storage

    app = QApplication.instance() or QApplication(sys.argv)
    app.setStyle("Fusion")
    app.setStyleSheet(theme.QSS)
    tmp = tempfile.mkdtemp(prefix="vaim-ui-")
    qs = QSettings(os.path.join(tmp, "settings.ini"), QSettings.IniFormat)
    qs.setValue("out_dir", os.path.join(tmp, "renders"))
    errors = []
    try:
        w = MainWindow(start_engine=False, settings=qs)
        w.setAttribute(Qt.WA_DontShowOnScreen)
        w.resize(1440, 900)
        w.show()
        w.on_ready("NVIDIA GeForce GTX 1650 (4 GB)", 4.0)
        w.prompt.setPlainText("a lantern dangling from a tree in a foggy graveyard")
        sent = []
        w.start_dream.connect(sent.append)
        from vaim_studio.ui.main_window import SIZES
        for idx in range(len(SIZES)):  # every size option builds valid settings
            w.size_seg.set_index(idx)
            w.update_hints()
            s = w.build_settings()
            assert s.output_size in (0, 1080, 2048, 4096, w.custom_size.value()), s.output_size
        w.apply_preset("Portrait")
        assert w.center_bias.isChecked() and w.max_classes.value() == 15 and w.active_preset == "Portrait"
        w.steps_spin.setValue(123)        # a manual change turns the style into "custom"
        assert w.active_preset is None
        w.apply_preset("Sharpest")
        assert w.cutouts.value() == 128 and w.output_size() == 2048 and w.active_preset == "Sharpest"
        w.use_example("a cathedral made of glass")
        shot0 = app_home() / "ui_preview_welcome.png"
        shot0.parent.mkdir(parents=True, exist_ok=True)
        app.processEvents()
        w.grab().save(str(shot0))
        w.size_seg.set_index(0)
        w.add_to_queue()           # idle -> starts immediately
        w.add_to_queue()           # running -> waits in the queue
        assert len(sent) == 1 and len(w.queue) == 1, (len(sent), len(w.queue))
        job = sent[0]
        yy, xx = np.mgrid[0:512, 0:512] / 511.0   # a soft test image (stand-in for a render)
        r = np.hypot(xx - 0.5, yy - 0.45)
        im = Image.fromarray((np.stack([0.75 - 0.6 * r + 0.2 * xx, 0.45 - 0.5 * r + 0.25 * yy,
                                        0.7 - 0.4 * r], -1).clip(0, 1) * 255).astype(np.uint8))
        w.on_preview(Preview(10, 300, [im], 12.3, 1.49, vram=2.3))
        lat = [{"normu": np.zeros((15, 128)), "cls": np.zeros((15, 1000)), "z": np.zeros((15, 128)),
                "classes": np.zeros((15, 1000))}]
        big = im.resize((1080, 1080))
        w.on_finished(Preview(300, 300, [big], 10.0, 1.5, final=True, latents=lat, native=[im]), job)
        saved = storage.list_images(w.out_dir)
        assert len(saved) == 1 and Image.open(saved[0]).size == (1080, 1080)
        assert w.gallery.count() == 1
        w.use_as_start(saved[0])   # re-dream path
        assert w.init_from is not None and w.build_settings().init_latents
        w.clear_init()
        app.processEvents()
        shot = app_home() / "ui_preview.png"
        shot.parent.mkdir(parents=True, exist_ok=True)
        w.grab().save(str(shot))
        print(f"    screenshot: {shot}")
        w.running = False
        w.close()
    except Exception as e:
        import traceback
        traceback.print_exc()
        errors.append(e)
    check("main window builds, queues, previews, saves, re-dreams", 1.0 if errors else 0.0, 0.5)


def bench(size, cutouts, steps):
    from vaim_studio.engine import Dreamer, DreamSettings
    from vaim_studio.engine import weights
    if not torch.cuda.is_available():
        print("\nNo CUDA GPU visible to PyTorch - skipping benchmark.")
        return
    dev = torch.device("cuda")
    d = Dreamer(log=lambda m: print("   ", m))
    print(f"\nBenchmark on {d.gpu_name()}  ({size}px, {cutouts} cutouts, {steps} timed steps)")
    d.ensure_models(size, "ViT-B/32", progress=lambda f, a, t: print(f"\r    downloading {f} {a / 2**20:.0f}/{t / 2**20:.0f} MB", end=""))

    # ---- new engine
    s = DreamSettings(prompt="a pyramid made of ice", image_size=size, cutouts=cutouts, steps=steps + 3, preview_every=10**9, seed=1)

    # tune once up front so only optimisation steps are timed below
    d._probe_fp16()
    pos, neg = d._targets(s)
    amp, ckpt, chunk, clip_half = d._autotune(s, pos, neg, d._amp_dtype(s), None)
    s.checkpointing, s.cutout_chunk = ("on" if ckpt else "off"), chunk
    s.precision = {torch.float16: "fp16", torch.bfloat16: "bf16"}.get(amp, "fp32")
    d._set_clip_half(clip_half)
    torch.cuda.reset_peak_memory_stats()

    class T:
        n = 0
        t0 = None

    def stop():
        T.n += 1
        if T.n == 4:
            torch.cuda.synchronize(); T.t0 = time.perf_counter()
        return False
    d.run(s, should_stop=stop)
    torch.cuda.synchronize()
    new_its = steps / (time.perf_counter() - T.t0)
    new_mem = torch.cuda.max_memory_allocated() / 2**30
    print(f"  new engine : {new_its:6.2f} it/s   peak VRAM {new_mem:.2f} GB")

    # ---- the original algorithm (same weights), reproduced step for step
    leg_cfg = legacy.BigGANConfig.from_json_file(str(weights.file_path(f"biggan-{size}", "config.json")))
    leg = legacy.BigGAN(leg_cfg)
    leg.load_state_dict(torch.load(weights.file_path(f"biggan-{size}", "pytorch_model.bin"), map_location="cpu", weights_only=True), strict=False)
    leg = leg.to(dev).eval()
    clip_full = copy.deepcopy(d.clip).half().to(dev).requires_grad_(True)   # original: CLIP fp16, not frozen
    d.biggan = d.biggan.cpu(); torch.cuda.empty_cache()
    nl = leg_cfg.layers.__len__() + 1
    normu = torch.randn(nl, 128, device=dev, requires_grad=True)
    clsp = (torch.randn(nl, 1000, device=dev) * 0.3 - 3.9).requires_grad_()
    opt = torch.optim.Adam([normu, clsp], 0.07)
    text = F.normalize(torch.randn(1, 512, device=dev), dim=-1).half()
    mean = torch.tensor([0.48145466, 0.4578275, 0.40821073], device=dev)[:, None, None]
    std = torch.tensor([0.26862954, 0.26130258, 0.27577711], device=dev)[:, None, None]
    import random as pyrandom

    def legacy_step():
        out = (leg(normu, torch.sigmoid(clsp), 1) + 1) / 2
        pieces = []
        for _ in range(cutouts):
            sz = int(size * torch.zeros(1,).normal_(mean=.8, std=.3).clip(.5, .95))
            ox, oy = pyrandom.randint(0, size - sz), pyrandom.randint(0, size - sz)
            pieces.append(F.interpolate(out[:, :, ox:ox + sz, oy:oy + sz], (224, 224), mode="nearest"))
        emb = clip_full.encode_image((torch.cat(pieces) - mean) / std)
        cls = torch.sigmoid(clsp)
        lat_loss = torch.abs(1 - torch.std(normu, dim=1)).mean() + torch.abs(torch.mean(normu, dim=1)).mean() + \
            4 * torch.max(torch.square(normu).mean(), torch.tensor(1., device=dev))
        for array in normu:
            m = torch.mean(array); df = array - m; sd = torch.pow(torch.mean(torch.pow(df, 2.0)), 0.5); zs = df / sd
            lat_loss = lat_loss + torch.abs(torch.mean(torch.pow(zs, 4.0)) - 3.0) / nl + torch.abs(torch.mean(torch.pow(zs, 3.0))) / nl
        cls_loss = ((50 * torch.topk(cls, largest=False, dim=1, k=999)[0]) ** 2).mean()
        sim = -100 * torch.cosine_similarity(text, emb, dim=-1).mean()
        (lat_loss + cls_loss + sim).backward()
        opt.step(); opt.zero_grad()

    torch.cuda.reset_peak_memory_stats()
    try:
        for _ in range(3):
            legacy_step()
        torch.cuda.synchronize(); t0 = time.perf_counter()
        for _ in range(steps):
            legacy_step()
        torch.cuda.synchronize()
        old_its = steps / (time.perf_counter() - t0)
        old_mem = torch.cuda.max_memory_allocated() / 2**30
        print(f"  original   : {old_its:6.2f} it/s   peak VRAM {old_mem:.2f} GB")
        print(f"\n  => {new_its / old_its:.1f}x faster, {old_mem / max(new_mem, 1e-6):.1f}x less VRAM")
    except torch.cuda.OutOfMemoryError:
        print("  original   : OUT OF MEMORY at these settings (the new engine fits)")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--bench", action="store_true")
    ap.add_argument("--size", type=int, default=256)
    ap.add_argument("--cutouts", type=int, default=64)
    ap.add_argument("--steps", type=int, default=20)
    ap.add_argument("--no-ui", action="store_true")
    a = ap.parse_args()
    for name, fn in [("correctness", correctness), ("smoke", smoke), ("ui", ui_test)]:
        if name == "ui" and a.no_ui:
            continue
        try:
            fn()
        except Exception:  # one broken section must not hide the others
            import traceback
            traceback.print_exc()
            check(f"{name} section ran to completion", 1.0, 0.5)
    if a.bench:
        bench(a.size, a.cutouts, a.steps)
    print("\nAll checks passed." if not failures else f"\n{failures} check(s) FAILED.")
    sys.exit(1 if failures else 0)
