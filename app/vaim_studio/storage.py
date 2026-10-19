"""Where images go, how their settings/latents are stored, thumbnails."""
import json
import os
import re
import sys
import time
from pathlib import Path

import numpy as np
from PIL import Image, PngImagePlugin

META_KEY = "vaim"
SIDE = ".vaim"                       # hidden folder next to the images
FORMATS = {                          # label -> (extension, PIL save kwargs)
    "PNG": (".png", {"compress_level": 4}),
    "JPEG": (".jpg", {"quality": 95, "subsampling": 0, "optimize": True}),
    "WebP": (".webp", {"quality": 95, "method": 4}),
}
IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".webp"}
THUMB = 160


def default_output_dir() -> Path:
    pics = Path.home() / "Pictures"
    if sys.platform == "win32":
        try:  # honour a relocated Pictures folder
            import ctypes
            from ctypes import wintypes
            buf = ctypes.create_unicode_buffer(wintypes.MAX_PATH)
            ctypes.windll.shell32.SHGetFolderPathW(None, 0x27, None, 0, buf)  # CSIDL_MYPICTURES
            if buf.value:
                pics = Path(buf.value)
        except Exception:
            pass
    return pics / "VAIM Studio"


def slug(text, n=60):
    s = re.sub(r"[^\w\s-]", "", text or "dream").strip().lower()
    s = re.sub(r"[\s_-]+", "_", s)
    return (s[:n].rstrip("_") or "dream")


def _side(png: Path) -> Path:
    return png.parent / SIDE


def _hide(folder: Path):
    if sys.platform == "win32":
        try:
            import ctypes
            ctypes.windll.kernel32.SetFileAttributesW(str(folder), 0x02)
        except Exception:
            pass


def latents_path(img: Path) -> Path:
    new = _side(img) / (img.name + ".npz")
    old = img.parent / ".latents" / (img.stem + ".npz")  # v1.0 location
    return old if (not new.exists() and old.exists()) else new


def meta_path(img: Path) -> Path:
    return _side(img) / (img.name + ".json")


def thumb_path(img: Path) -> Path:
    return _side(img) / "thumbs" / (img.name + ".jpg")


def save_dream(out_dir: Path, image: Image.Image, meta: dict, latents: dict | None, index=0,
               fmt="PNG") -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    ext, kw = FORMATS.get(fmt, FORMATS["PNG"])
    stamp = time.strftime("%Y%m%d-%H%M%S")
    name = f"{stamp}_{slug(meta.get('prompt'))}_{meta.get('seed', 0)}"
    if meta.get("batch", 1) > 1:
        name += f"_{index + 1}"
    path = _free_name(out_dir, name, ext)
    _write(path, image, meta, latents, kw)
    return path


def _free_name(folder: Path, name: str, ext: str) -> Path:
    path = folder / f"{name}{ext}"
    k = 2
    while path.exists() or meta_path(path).exists():
        path = folder / f"{name}-{k}{ext}"
        k += 1
    return path


def save_upscaled(src: Path, image: Image.Image, meta: dict, latents: dict | None) -> Path:
    """Save an upscaled copy next to `src` (same format), e.g. name_4K.png, keeping its latents
    so it can still be re-dreamed."""
    ext = src.suffix.lower()
    kw = next((k for e, k in FORMATS.values() if e == {".jpeg": ".jpg"}.get(ext, ext)), None)
    if kw is None:
        ext, kw = FORMATS["PNG"]
    n = max(image.size)
    tag = {1080: "1080", 2048: "2K", 4096: "4K"}.get(n, f"{n}px")
    path = _free_name(src.parent, f"{src.stem}_{tag}", ext)
    _write(path, image, meta, latents, kw)
    return path


def _write(path: Path, image: Image.Image, meta: dict, latents: dict | None, kw: dict):
    ext = path.suffix.lower()
    meta_json = json.dumps(meta)
    if ext == ".png":
        info = PngImagePlugin.PngInfo()
        info.add_text(META_KEY, meta_json)
        info.add_text("Software", "VAIM Studio (BigGAN + CLIP)")
        image.save(path, pnginfo=info, **kw)
    else:
        exif = Image.Exif()
        exif[0x010E] = meta_json[:60000]              # ImageDescription
        exif[0x0131] = "VAIM Studio (BigGAN + CLIP)"   # Software
        image.convert("RGB").save(path, exif=exif, **kw)

    side = _side(path)
    (side / "thumbs").mkdir(parents=True, exist_ok=True)
    _hide(side)
    meta_path(path).write_text(meta_json, encoding="utf-8")
    t = image.copy()
    t.thumbnail((THUMB, THUMB), Image.LANCZOS)
    t.convert("RGB").save(thumb_path(path), quality=88)
    if latents is not None:
        np.savez_compressed(_side(path) / (path.name + ".npz"), **latents)


def read_meta(img: Path) -> dict | None:
    try:
        mp = meta_path(img)
        if mp.is_file():
            return json.loads(mp.read_text(encoding="utf-8"))
        with Image.open(img) as im:
            raw = im.text.get(META_KEY) if hasattr(im, "text") else None
            if not raw:
                raw = im.getexif().get(0x010E)
        return json.loads(raw) if raw else None
    except Exception:
        return None


def read_latents(img: Path) -> dict | None:
    p = latents_path(img)
    if not p.is_file():
        return None
    with np.load(p) as d:
        return {k: d[k] for k in d.files}


def thumbnail(img: Path) -> Path | None:
    """Cached small JPEG for the gallery; created on demand for older/foreign images."""
    tp = thumb_path(img)
    try:
        if tp.is_file() and tp.stat().st_mtime >= img.stat().st_mtime - 1:
            return tp
        tp.parent.mkdir(parents=True, exist_ok=True)
        _hide(tp.parent.parent)
        with Image.open(img) as im:
            im.draft("RGB", (THUMB * 2, THUMB * 2))  # fast JPEG downscale on decode
            im = im.convert("RGB")
            im.thumbnail((THUMB, THUMB), Image.LANCZOS)
            im.save(tp, quality=88)
        return tp
    except Exception:
        return None


def delete_dream(img: Path):
    for f in (img, latents_path(img), meta_path(img), thumb_path(img)):
        try:
            f.unlink()
        except (FileNotFoundError, OSError):
            pass


def list_images(out_dir: Path):
    if not out_dir.is_dir():
        return []
    files = [p for p in out_dir.iterdir() if p.suffix.lower() in IMAGE_EXTS and p.is_file()]
    return sorted(files, key=lambda p: p.stat().st_mtime, reverse=True)


def open_file(path: Path):
    if sys.platform == "win32":
        os.startfile(str(path))  # noqa
    else:
        import subprocess
        subprocess.Popen(["xdg-open", str(path)])


def open_in_explorer(path: Path):
    path = Path(path)
    import subprocess
    if sys.platform == "win32":
        if path.is_file():
            subprocess.Popen(["explorer", "/select,", str(path)])
        else:
            path.mkdir(parents=True, exist_ok=True)
            os.startfile(str(path))  # noqa
    else:
        subprocess.Popen(["xdg-open", str(path.parent if path.is_file() else path)])
