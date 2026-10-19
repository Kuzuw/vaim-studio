# Changelog

## 1.1.0 — you can always see what it's doing

- New progress panel under the image: **Prepare › Paint › Upscale › Save**, with the current stage
  highlighted, a bar that keeps moving (it never looks frozen), what is happening right now
  ("Loading CLIP…", "Finding the fastest settings for your GPU · test 3", "step 210 of 350 · 0.41 steps/s",
  "pass 2 of 2 · 1024 → 2048 px"), time left and time elapsed.
- Before the first preview the canvas shows a spinner with the current step instead of the old welcome page.
- The first preview now appears after 1 step instead of 10.
- Upscaling reports real progress tile by tile.
- Progress percentage in the window title and on the Windows taskbar button, so you can follow a render
  from another app.
- Clear end states: *Done* (with where it was saved and how long it took), *Stopped*, or *Something went wrong*.

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
