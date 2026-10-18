"""Model weight locations, downloads (with mirrors + resume) and caching."""
import hashlib
import os
import shutil
import sys
import urllib.request
from pathlib import Path


def app_home() -> Path:
    if os.environ.get("VAIM_HOME"):
        return Path(os.environ["VAIM_HOME"])
    if sys.platform == "win32":
        return Path(os.environ.get("LOCALAPPDATA", Path.home() / "AppData" / "Local")) / "VAIM Studio"
    return Path.home() / ".vaim_studio"


def models_dir() -> Path:
    d = app_home() / "models"
    d.mkdir(parents=True, exist_ok=True)
    return d


_S3 = "https://s3.amazonaws.com/models.huggingface.co/biggan"
_HF = "https://huggingface.co"
_OPENAI = "https://openaipublic.azureedge.net/clip/models"
_OPENAI_BLOB = "https://openaipublic.blob.core.windows.net/clip/models"


def _biggan(size):
    # The original 2019 S3 bucket still serves these; 128px also has a Hub mirror.
    hub = [f"{_HF}/osanseviero/BigGAN-deep-128/resolve/main"] if size == 128 else []
    return {
        "title": f"BigGAN-deep {size}px",
        "files": {
            "config.json": {"urls": [f"{_S3}/biggan-deep-{size}-config.json"] + [f"{h}/config.json" for h in hub]},
            "pytorch_model.bin": {"urls": [f"{_S3}/biggan-deep-{size}-pytorch_model.bin"] + [f"{h}/pytorch_model.bin" for h in hub]},
        },
    }


def _clip(fname, sha, timm_repo):
    return {
        "title": f"CLIP {fname[:-3]}",
        "files": {
            fname: {
                # OpenAI original (sha256-verified), then the open_clip re-upload of the same weights
                "urls": [f"{_OPENAI}/{sha}/{fname}", f"{_OPENAI_BLOB}/{sha}/{fname}",
                         f"{_HF}/timm/{timm_repo}/resolve/main/open_clip_pytorch_model.bin"],
                "sha256": {0: sha, 1: sha},
                "legacy_cache": Path.home() / ".cache" / "clip" / fname,
            }
        },
    }


MODELS = {
    "biggan-128": _biggan(128),
    "biggan-256": _biggan(256),
    "biggan-512": _biggan(512),
    "clip-ViT-B/32": _clip("ViT-B-32.pt", "40d365715913c9da98579312b702a82c18be219cc2a73407c4526f58eba950af", "vit_base_patch32_clip_224.openai"),
    "clip-ViT-B/16": _clip("ViT-B-16.pt", "5806e77cd80f8b59890b7e101eabd078d9fb84e6937f9e85e4ecb61988df416f", "vit_base_patch16_clip_224.openai"),
    "esrgan-x2": {
        "title": "Real-ESRGAN x2",
        "files": {"RealESRGAN_x2plus.pth": {"urls": [
            "https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.1/RealESRGAN_x2plus.pth"]}},
    },
    "clip-ViT-L/14": _clip("ViT-L-14.pt", "b8cca3fd41ae0c99ba7e8951adf17d267cdb84cd88be6f7c2e0eca1737a03836", "vit_large_patch14_clip_224.openai"),
}


class Cancelled(Exception):
    pass


def model_dir(key) -> Path:
    d = models_dir() / key.replace("/", "-")
    d.mkdir(parents=True, exist_ok=True)
    return d


def is_installed(key) -> bool:
    d = model_dir(key)
    return all((d / f).is_file() and (d / f).stat().st_size > 0 for f in MODELS[key]["files"])


def file_path(key, fname) -> Path:
    return model_dir(key) / fname


def _sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _fetch(url, dest: Path, progress, cancelled):
    part = dest.with_suffix(dest.suffix + ".part")
    have = part.stat().st_size if part.exists() else 0
    req = urllib.request.Request(url, headers={"User-Agent": "VAIMStudio/1.0"})
    if have:
        req.add_header("Range", f"bytes={have}-")
    with urllib.request.urlopen(req, timeout=30) as r:
        if have and r.status != 206:  # server ignored the range, start over
            have = 0
        total = int(r.headers.get("Content-Length") or 0) + have
        with open(part, "ab" if have else "wb") as out:
            done = have
            while True:
                if cancelled and cancelled():
                    raise Cancelled()
                buf = r.read(1 << 20)
                if not buf:
                    break
                out.write(buf)
                done += len(buf)
                if progress:
                    progress(dest.name, done, total)
    part.replace(dest)


def ensure(key, progress=None, cancelled=None, log=print):
    """Download any missing files for a model. Tries each mirror in turn."""
    spec = MODELS[key]
    for fname, f in spec["files"].items():
        dest = file_path(key, fname)
        if dest.is_file() and dest.stat().st_size > 0:
            continue
        legacy = f.get("legacy_cache")
        if legacy and Path(legacy).is_file():
            log(f"Reusing existing {legacy}")
            shutil.copyfile(legacy, dest)
            continue
        errors = []
        for i, url in enumerate(f["urls"]):
            try:
                log(f"Downloading {fname} from {url.split('/')[2]} ...")
                _fetch(url, dest, progress, cancelled)
                want = f.get("sha256", {}).get(i)
                if want and _sha256(dest) != want:
                    dest.unlink()
                    raise RuntimeError("checksum mismatch")
                break
            except Cancelled:
                raise
            except Exception as e:  # try next mirror
                errors.append(f"{url}: {e}")
                stale = dest.with_suffix(dest.suffix + ".part")
                if i + 1 < len(f["urls"]) and stale.exists():
                    stale.unlink()
        else:
            raise RuntimeError(
                f"Could not download {fname} for {spec['title']}.\n" + "\n".join(errors) +
                f"\n\nYou can place the file manually in:\n{model_dir(key)}"
            )
