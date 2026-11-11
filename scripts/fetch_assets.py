"""Download the inputs that are not kept in git: the free Pexels clips the examples use and the MediaPipe pose model.

Standard library only.

    python scripts/fetch_assets.py list                  what exists, what is missing, where each file comes from
    python scripts/fetch_assets.py footage 7525471       one clip by its Pexels number (several numbers allowed)
    python scripts/fetch_assets.py footage all
    python scripts/fetch_assets.py pose-model            MediaPipe Pose Landmarker (heavy), about 30 MB

The clips are under the Pexels License (free to use and modify, https://www.pexels.com/license/). They are downloaded
from the Pexels file server to your computer and are never committed (*.mp4 is ignored by git).
If a download is refused, open the page shown, press the site's download button yourself and save the file under the
name printed here. A file whose SHA-256 differs from the verified copy is kept, with a warning: Pexels may re-encode,
and then frame counts in the READMEs may not match."""
from __future__ import annotations

import hashlib
import sys
import urllib.error
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CDN = "https://videos.pexels.com/video-files/"
CLIPS = {
    "7525471": ("examples/fruit_physics/source/Pexels_7525471_720x1280.mp4", "7525471/7525471-hd_1080_1920_25fps.mp4",
                "0b51020f49a03bcf109b7705076dfd823e0fc73ea3fa04071cf1495114275a04", "woman-dancing-in-the-kitchen-7525471", "fruit_physics"),
    "10708812": ("examples/shader_live/footage/Pexels_10708812.mp4", "10708812/10708812-hd_1920_1080_30fps.mp4",
                 "7ea653251995bf6e1ef89351f6034a4cf2dd1feec1b10a6ef89587312affee19", "top-view-of-an-empty-basketball-court-10708812", "shader_live 06_Marbled_Court"),
    "34696869": ("examples/shader_live/footage/Pexels_34696869.mp4", "34696869/14706282_1920_1080_30fps.mp4",
                 "cd9e45fea707586d2b90e650f8b3638135f2f7515afd62ff58f30ec47a0b9a28", "aerial-view-of-spacious-empty-parking-lot-34696869", "shader_live 08_Lava_Fissures"),
    "10872797": ("examples/shader_live/footage/Pexels_10872797.mp4", "10872797/10872797-hd_1920_1080_30fps.mp4",
                 "ff6fbba79f2478643b689d1c106c85caa5dc3dbc07ec5f492b628dc3c33c6aad", "road-through-sandy-desert-at-sunrise-10872797", "shader_live 01_Desert_Causeway_Sea"),
}
POSE = ("examples/fruit_physics/models/pose_landmarker_heavy.task",
        "https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_heavy/float16/latest/pose_landmarker_heavy.task")


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def download(url: str, dst: Path) -> bool:
    dst.parent.mkdir(parents=True, exist_ok=True); tmp = dst.with_suffix(dst.suffix + ".part")
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0 (fetch_assets.py; classroom example)"})
    try:
        with urllib.request.urlopen(req, timeout=120) as r, open(tmp, "wb") as f:
            while True:
                chunk = r.read(1 << 20)
                if not chunk:
                    break
                f.write(chunk)
    except (urllib.error.URLError, OSError) as exc:
        tmp.unlink(missing_ok=True); print(f"  download failed: {exc}"); return False
    tmp.replace(dst); return True


def clip(vid: str) -> bool:
    rel, part, want, slug, used = CLIPS[vid]; dst = ROOT / rel
    if dst.exists():
        print(f"[have] {rel}"); return True
    print(f"[get ] Pexels {vid} -> {rel}")
    if not download(CDN + part, dst):
        print(f"  Open https://www.pexels.com/video/{slug}/ , download the 1080-wide file yourself and save it as\n  {dst}")
        return False
    got = sha256(dst)
    print(f"  saved {dst.stat().st_size:,} bytes" + ("" if got == want else "\n  WARNING: this file differs from the verified copy (Pexels may have re-encoded it); frame counts may differ"))
    return True


def pose_model() -> bool:
    rel, url = POSE; dst = ROOT / rel
    if dst.exists() and dst.stat().st_size > 20_000_000:
        print(f"[have] {rel}"); return True
    print(f"[get ] MediaPipe pose model -> {rel}")
    if not download(url, dst) or dst.stat().st_size < 20_000_000:
        print(f"  Download {url}\n  and save it as {dst}"); return False
    print(f"  saved {dst.stat().st_size:,} bytes"); return True


def main():
    a = sys.argv[1:]
    if not a or a[0] == "list":
        for vid, (rel, part, _, slug, used) in CLIPS.items():
            print(f"{'[have]' if (ROOT / rel).exists() else '[none]'} Pexels {vid:9s} {rel}\n         used by {used} | https://www.pexels.com/video/{slug}/")
        print(f"{'[have]' if (ROOT / POSE[0]).exists() else '[none]'} pose model       {POSE[0]}\n         used by fruit_physics | {POSE[1]}")
        return
    if a[0] == "pose-model":
        sys.exit(0 if pose_model() else 2)
    if a[0] == "footage":
        ids = list(CLIPS) if a[1:] == ["all"] else a[1:]
        bad = [i for i in ids if i not in CLIPS]
        if bad or not ids:
            sys.exit("give Pexels numbers from: " + ", ".join(CLIPS) + " (or 'all')")
        sys.exit(0 if all([clip(i) for i in ids]) else 2)
    sys.exit(__doc__)


if __name__ == "__main__":
    main()
