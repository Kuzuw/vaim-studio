# Third-party notices

VAIM Studio is MIT-licensed (see [LICENSE](LICENSE)). It builds on the following work.

## Source code adapted into this repository

| Component | Where | License |
|---|---|---|
| [big-sleep](https://github.com/lucidrains/big-sleep) by Ryan Murdock & Phil Wang | optimisation objective in `app/vaim_studio/engine/dreamer.py`; `app/tools/legacy_biggan.py` (unmodified reference copy) | MIT, Copyright (c) 2021 Ryan Murdock, Phil Wang ([full text](docs/LICENSE-big-sleep.txt)) |
| [pytorch-pretrained-BigGAN](https://github.com/huggingface/pytorch-pretrained-BigGAN) by Hugging Face | `app/vaim_studio/engine/biggan.py` (restructured) | MIT |
| [CLIP](https://github.com/openai/CLIP) by OpenAI | `app/vaim_studio/engine/clip_model.py`, BPE vocabulary in `engine/data/` | MIT, Copyright (c) 2021 OpenAI |
| [Real-ESRGAN](https://github.com/xinntao/Real-ESRGAN) / [BasicSR](https://github.com/XPixelGroup/BasicSR) by Xintao Wang et al. | RRDBNet architecture in `app/vaim_studio/engine/upscale.py` | BSD-3-Clause |

## Model weights (downloaded on first use, not distributed here)

| Model | Source | License |
|---|---|---|
| BigGAN-deep 128 / 256 / 512 | DeepMind, converted by Hugging Face | Apache-2.0 |
| CLIP ViT-B/32 (and optional B/16, L/14) | OpenAI | MIT |
| Real-ESRGAN x2plus | Xintao Wang et al. | BSD-3-Clause |

## Installed at setup time (not bundled)

PyTorch (BSD-3-Clause), PySide6 / Qt for Python (LGPL-3.0), NumPy (BSD-3-Clause), Pillow (MIT-CMU),
ftfy (Apache-2.0), regex (Apache-2.0), and the [uv](https://github.com/astral-sh/uv) package manager
(MIT / Apache-2.0). These are downloaded from their official sources by the installer and keep their
own licenses.
