# VAIM Studio: notes for Claude Code

Windows desktop app that makes images with BigGAN-deep steered by CLIP ("Big Sleep",
lucidrains/big-sleep), rebuilt for speed, with Real-ESRGAN upscaling to 4K. Owner: Kuzuw (Turkish
speaker; answer in the language they write in). Built in a claude.ai cloud session on 2026-10-18/19.

## Layout
- `app/vaim_studio/engine/`: `dreamer.py` (optimisation loop, autotuner, fp16 probe, VRAM cap, NaN
  recovery, auto-stop, upscaling), `biggan.py` (baked spectral norm, batched latents, checkpointing),
  `clip_model.py`, `upscale.py` (RRDBNet x2, tiled), `weights.py` (downloads, mirrors, app_home).
- `app/vaim_studio/ui/`: `main_window.py` (PySide6 UI: style cards, size/quality pills, Fine-tune
  drawer, queue, gallery, welcome page), `theme.py` (QSS), `worker.py` (QThread render worker).
- `app/vaim_studio/storage.py`: saving PNG/JPEG/WebP, sidecar `.vaim/` (json meta, npz latents, thumbs).
- `app/tools/selftest.py`: correctness vs the unmodified original (`legacy_biggan.py`), a CPU pipeline
  smoke test, an off-screen UI test that saves screenshots, and `--bench`.
- `installer/`: Go. `setup/` = VAIM-Studio-Setup.exe (embeds payload.zip, uv + CUDA PyTorch 2.7.1,
  MSVC runtime check, shortcuts, uninstall, HKCU uninstall key); `launcher/` = VAIM Studio.exe.
- `scripts/build.py`: one-command cross-compiled installer -> `dist/`.
- CI (`.github/workflows/ci.yml`): pyflakes + self-test (CPU, offscreen Qt), screenshots pushed to the
  `screenshots` branch, installer build, release `v<app version>` published from main.

## Release flow
Bump `app/vaim_studio/__init__.py` `__version__`, update CHANGELOG.md, push to main: CI builds and
publishes release `v<version>` with `VAIM-Studio-Setup-<version>.exe`.

## Things learned the hard way
- Import torch BEFORE PySide6 (app.py). Otherwise PySide6's bundled msvcp140.dll gets used by torch
  and Windows crashes with heap corruption 0xc0000374 on the first model load.
- GTX 16xx: BigGAN fp16 produces NaN (the probe catches it); CLIP fp16 is ~3.4x SLOWER than fp32. The
  defaults use fp32 on cards without tensor cores, and the tuner re-checks.
- `torch.cuda.set_per_process_memory_fraction` needs a device index (`cuda:0`), not `cuda`.
- The VRAM cap stops Windows' sysmem fallback, which otherwise silently makes everything crawl.
- Owner's GTX 1650 4 GB measurements: CLIP ~18 ms per cutout (fp32) is ~95% of a step; BigGAN 512
  ~0.1 s/step. Draft(32) 1.49 it/s, Best(128) 0.41 it/s. So base 128/256 is barely faster than 512 and
  much blurrier: 512 is the recommended base. Upscale passes 256->512 1.4 s, 512->1024 3.1 s.
- Auto-stop threshold 0.0025 mean pixel change per 10 steps; real runs settled around steps 300-400.
- Qt button text: escape `&` as `&&`. Sidebar widgets use QSizePolicy.Ignored so they never overflow.
- Settings keys are versioned (`v11/`, `v13/`, `v14/`) for one-time migrations.

## Owner's machine
Windows, GTX 1650 4 GB (likely laptop), 16 GB RAM, app installed at `D:\vaimstudio`. Data and logs at
`%LOCALAPPDATA%\VAIM Studio` (`log.txt`, `log-previous.txt`, `autotune.json`, `ui_preview*.png`).

## Open ideas
- Real Windows screenshot for the README (the current one comes from CI on Linux).
- Optional torch.compile via triton-windows, gated by the tuner.
- Non-square output, for example a 16:9 crop or outpainting.
