#!/usr/bin/env python3
"""Build the Windows installer from any OS (Go cross-compiles; no Windows needed).

    python scripts/build.py            -> dist/VAIM-Studio-Setup-<version>.exe

Steps: syntax-check the app, build the launcher (VAIM Studio.exe), zip the app + launcher into
the installer payload, then build the setup exe with the version baked in. If llvm-windres is on
PATH the icon / version resources are regenerated, otherwise the committed .syso files are used.
"""
import os
import re
import shutil
import subprocess
import sys
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
APP = ROOT / "app"
INST = ROOT / "installer"
DIST = ROOT / "dist"


def version():
    text = (APP / "vaim_studio" / "__init__.py").read_text()
    return re.search(r'__version__\s*=\s*"([^"]+)"', text).group(1)


def run(cmd, cwd, env=None):
    print("  $", " ".join(cmd))
    subprocess.run(cmd, cwd=cwd, check=True, env={**os.environ, **(env or {})})


def windres():
    for name in ("llvm-windres", "llvm-windres-18", "llvm-windres-17", "x86_64-w64-mingw32-windres"):
        path = shutil.which(name)
        if path:
            return path
    return None


def resources(ver, tool):
    v = (ver.split("-")[0].split(".") + ["0", "0", "0"])[:3]
    template = (INST / "res.rc.in").read_text()
    for target, desc, fname in [("setup", "VAIM Studio Setup", "VAIM-Studio-Setup.exe"),
                                ("launcher", "VAIM Studio", "VAIM Studio.exe")]:
        rc = (template.replace("@V1@", v[0]).replace("@V2@", v[1]).replace("@V3@", v[2])
              .replace("@VERSION@", ver).replace("@DESC@", desc).replace("@NAME@", target)
              .replace("@FILE@", fname))
        (INST / target / "res.rc").write_text(rc)
        args = [tool, "-i", "res.rc", "-o", "rsrc_windows_amd64.syso", "-O", "coff"]
        if "llvm" in Path(tool).name:
            args.insert(1, "--target=pe-x86-64")
        run(args, INST / target)
        (INST / target / "res.rc").unlink()


def main():
    ver = version()
    print(f"VAIM Studio {ver}")

    print("[1/4] checking Python sources")
    for p in [*APP.rglob("*.py"), APP / "launch.pyw"]:
        compile(p.read_text(encoding="utf-8"), str(p), "exec")  # syntax check, writes nothing

    tool = windres()
    if tool:
        print("[2/4] icon + version resources via", Path(tool).name)
        resources(ver, tool)
    else:
        print("[2/4] llvm-windres not found - using committed resources")

    go_env = {"GOOS": "windows", "GOARCH": "amd64", "CGO_ENABLED": "0"}
    build = INST / "build"
    build.mkdir(exist_ok=True)
    print("[3/4] launcher + payload")
    run(["go", "build", "-trimpath", "-ldflags", "-s -w -H windowsgui", "-o", str(build / "VAIM Studio.exe"),
         "./launcher"], INST, go_env)
    payload = INST / "setup" / "payload.zip"
    with zipfile.ZipFile(payload, "w", zipfile.ZIP_DEFLATED, compresslevel=9) as z:
        for p in sorted(APP.rglob("*")):
            if p.is_file() and "__pycache__" not in p.parts:
                z.write(p, p.relative_to(APP).as_posix())
        for extra in ("README.md", "LICENSE", "THIRD_PARTY_NOTICES.md", "CHANGELOG.md"):
            if (ROOT / extra).is_file():
                z.write(ROOT / extra, extra)
        z.write(build / "VAIM Studio.exe", "VAIM Studio.exe")

    print("[4/4] installer")
    DIST.mkdir(exist_ok=True)
    out = DIST / f"VAIM-Studio-Setup-{ver}.exe"
    run(["go", "build", "-trimpath", "-ldflags", f"-s -w -X main.appVersion={ver}", "-o", str(out), "./setup"],
        INST, go_env)
    payload.unlink()
    print(f"\nBuilt {out.relative_to(ROOT)}  ({out.stat().st_size / 2**20:.1f} MB)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
