"""Download the licensed source clip and HDRI listed in shots/<name>.json.

Footage is never committed. Each computer downloads it again from the original
distributor and checks the SHA-256 recorded when the license was reviewed.
"""
from __future__ import annotations

from pathlib import Path
import shutil
import urllib.request

from .common import StepError, checkpoint, save_json, sha256, shot_dir

AGENT = "Mozilla/5.0 (compatible; colmap-opengl-composite classroom fetch)"


def download(url, target, expected=None):
    target = Path(target)
    if target.is_file():
        if expected and sha256(target) != expected:
            raise StepError(f"{target} exists but its SHA-256 differs. Keep it and inspect it before replacing.")
        print("exists", target, flush=True)
        return target
    target.parent.mkdir(parents=True, exist_ok=True)
    part = target.with_name(target.name+".part")
    request = urllib.request.Request(url, headers={"User-Agent": AGENT})
    print("download", url, flush=True)
    with urllib.request.urlopen(request, timeout=60) as response, part.open("wb") as out:
        total = int(response.headers.get("Content-Length") or 0)
        done = 0
        while chunk := response.read(1 << 20):
            out.write(chunk)
            done += len(chunk)
            if total and done % (16 << 20) < (1 << 20):
                print(f"  {done/total:5.1%}", flush=True)
    digest = sha256(part)
    if expected and digest != expected:
        part.unlink()
        raise StepError(f"SHA-256 mismatch for {url}. The distributor changed the file; review the license and pick it again.")
    part.replace(target)
    return target


def lighting_path(cfg, work):
    return Path(work)/"lighting"/cfg["hdri"]["file"]


def main(cfg, work, args):
    d = shot_dir(cfg, work)
    src = cfg["source"]
    clip = download(src["download"], d/"source/original.mp4", src.get("sha256"))
    hdri = download(cfg["hdri"]["download"], lighting_path(cfg, work), cfg["hdri"].get("sha256"))
    save_json(d/"source/license.json", dict(source=src, hdri=cfg["hdri"], downloaded_sha256=sha256(clip),
              note="Keep this record with every render. Pexels License is not CC0; follow its restrictions."))
    if shutil.which("ffprobe") is None:
        print("ffprobe not on PATH; prepare will use tools.local.json", flush=True)
    checkpoint(d, "fetched", clip=str(clip), hdri=str(hdri))
    print(f"source ready: {clip}\nHDRI ready: {hdri}")
