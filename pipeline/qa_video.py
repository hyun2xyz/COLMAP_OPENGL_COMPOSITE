"""Decode every frame of Composite.mp4 + Alpha.mov and check the numbers.

Checks: frame count, no duplicate frames, CG present in every frame, the plate
untouched outside the alpha (mean abs diff < 4/255, decoded by ffmpeg with BT.709).
Writes review sheets (24 thumbnails each). A numerical pass is NOT a visual pass:
look at the sheets and play the movie before calling the shot done.
"""
from __future__ import annotations

import hashlib
from pathlib import Path
import subprocess

import cv2
import numpy as np

from .common import StepError, checkpoint, decode_rgb, frame_path, load_json, probe, save_json, shot_dir, tool


def check(d, out):
    s = load_json(d/"Camera_Three_ground.json")
    W, H, N = s["width"], s["height"], s["count"]
    streams = {n: probe(out/n)["streams"][0] for n in ("Composite.mp4", "Alpha.mov")}
    alpha = subprocess.Popen([tool("ffmpeg"), "-v", "error", "-i", str(out/"Alpha.mov"), "-vf", "alphaextract",
                              "-f", "rawvideo", "-pix_fmt", "gray", "-"], stdout=subprocess.PIPE)
    records, thumbs, hashes = [], [], []
    for i, frame in enumerate(decode_rgb(out/"Composite.mp4", W, H)):
        raw = alpha.stdout.read(W*H)
        if len(raw) != W*H:
            raise StepError("Alpha ended before the composite")
        al = np.frombuffer(raw, np.uint8).reshape(H, W)
        plate = cv2.imread(str(frame_path(d, i)))[:, :, ::-1]
        mask = al < 1
        bg = float(np.abs(frame.astype(float)-plate)[mask].mean()) if mask.any() else None
        records.append(dict(frame=i, alpha_pixels=int((al > 8).sum()), background_mae=bg))
        hashes.append(hashlib.md5(frame.tobytes()).hexdigest())
        thumb = cv2.resize(frame[:, :, ::-1], (320, 180))
        cv2.rectangle(thumb, (0, 0), (55, 22), (0, 0, 0), -1)
        cv2.putText(thumb, f"{i:03}", (6, 17), 0, .5, (255, 255, 255), 1)
        thumbs.append(thumb)
    if alpha.stdout.read() or alpha.wait():
        raise StepError("Alpha frame count differs from the composite")
    for start in range(0, len(thumbs), 24):
        tiles = thumbs[start:start+24]
        tiles += [np.zeros((180, 320, 3), np.uint8)]*(24-len(tiles))
        sheet = np.vstack([np.hstack(tiles[k:k+4]) for k in range(0, 24, 4)])
        cv2.imwrite(str(out/f"review_{start:03}_{min(start+23, N-1):03}.jpg"), sheet)
    passed = bool(len(records) == N and len(set(hashes)) == N and min(r["alpha_pixels"] for r in records) > 50
                  and max(r["background_mae"] for r in records) < 4
                  and all(int(v.get("nb_frames", -1)) == N for v in streams.values())
                  and "a" in streams["Alpha.mov"]["pix_fmt"])
    report = dict(numerical_pass=passed, visual_review="pending: open the review sheets and play Composite.mp4",
                  frame_count=len(records), expected=N, unique_frames=len(set(hashes)), size=[W, H], fps=s["fps"],
                  alpha_pixel_format=streams["Alpha.mov"]["pix_fmt"],
                  maximum_background_mae=max(r["background_mae"] for r in records),
                  minimum_alpha_pixels=min(r["alpha_pixels"] for r in records), records=records)
    save_json(out/"video-qa.json", report)
    return report


def main(cfg, work, args):
    d = shot_dir(cfg, work)
    out = Path(args.render) if args.render else d/"render/final"
    report = check(d, out)
    print({k: v for k, v in report.items() if k != "records"})
    if not report["numerical_pass"]:
        raise StepError("Video QA failed; see video-qa.json")
    checkpoint(d, "video_numbers_ok", output=str(out), next="Look at review_*.jpg and play Composite.mp4 before accepting")
