"""Cut the working plate: explicit trim, constant frame rate, fixed size, PNG frames.

Every later step counts frames from 0 in `frames/frame_000000.png`. Keyframes
(every Nth frame plus the last) are copied, not linked, so Windows works too.
"""
from __future__ import annotations

import shutil

from .common import StepError, checkpoint, probe, run, save_json, shot_dir, tool


def main(cfg, work, args):
    d = shot_dir(cfg, work)
    src = d/"source/original.mp4"
    if not src.is_file():
        raise StepError("No source clip. Run: python -m pipeline fetch "+cfg["name"])
    plate = cfg["plate"]
    w, h = plate["size"]
    fps = plate["fps"]
    frames = d/"frames"
    expected = round(plate["duration"]*fps)
    if frames.exists() and any(frames.iterdir()):
        if len(list(frames.glob("frame_*.png"))) == expected and (d/"keyframes").is_dir():
            print("already prepared:", frames)
            return
        raise StepError(f"{frames} is incomplete. Keep it for inspection and use a new --work folder.")
    save_json(d/"source/probe.json", probe(src))
    run([tool("ffmpeg"), "-v", "error", "-n", "-ss", plate.get("start", 0), "-i", src, "-t", plate["duration"],
         "-vf", f"fps={fps},scale={w}:{h}:flags=lanczos", "-an", "-c:v", "libx264", "-crf", "14",
         "-pix_fmt", "yuv420p", "-threads", "4", d/"plate.mp4"])
    frames.mkdir(parents=True, exist_ok=True)
    run([tool("ffmpeg"), "-v", "error", "-i", d/"plate.mp4", "-start_number", "0", "-threads", "4",
         frames/"frame_%06d.png"])
    files = sorted(frames.glob("frame_*.png"))
    if len(files) != expected:
        raise StepError(f"Expected {expected} frames, got {len(files)}. Check the clip length and trim.")
    stride = plate.get("keyframe_stride", 3)
    keys = d/"keyframes"
    keys.mkdir(exist_ok=True)
    chosen = sorted(set(range(0, len(files), stride)) | {len(files)-1})
    for i in chosen:
        shutil.copy2(files[i], keys/files[i].name)
    checkpoint(d, "prepared", frames=len(files), keyframes=len(chosen), size=[w, h], fps=fps)
    print(f"{len(files)} frames, {len(chosen)} keyframes in {d}")
