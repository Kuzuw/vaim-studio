# Changelog

## 1.0.0 — first release

**App**
- Clean, calm interface: prompt, style, size, quality and save location up front, with everything
  technical in a *Fine-tune* drawer. Welcome screen with example prompts.
- Style presets: **Sharpest**, Balanced, Portrait, Landscape, Classic 2021, Fast draft.
- Output at native 512 px, 1080, 2K, 4K or a custom size via Real-ESRGAN ×2 passes, with an adjustable
  sharpness blend.
- Render queue, render-to-folder picker, PNG / JPEG / WebP, re-dream from a finished image, batched
  variations, image prompts and avoid words.
- Time estimate learned from your own renders, auto-finish when the image stops changing, sleep
  prevention during renders, taskbar alert when done.

**Engine**
- Big Sleep (BigGAN-deep + CLIP) rebuilt: frozen models, spectral norm baked once, batched cutouts in
  one `grid_sample`, chunked CLIP backward with bounded memory, vectorised losses.
- Per-GPU autotuner for precision and memory, fp16 safety probe and NaN recovery, VRAM cap so Windows
  never silently spills into system RAM, out-of-memory recovery.
- Self-test that checks the rewrite against the unmodified original code.

**Installer**
- `VAIM-Studio-Setup.exe`: no admin rights, GPU detection, private Python + CUDA PyTorch,
  Microsoft C++ runtime check, shortcuts, uninstaller and Apps & features entry.
