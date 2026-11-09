"""Show what is finished for a shot and the next command. Reads only small JSON files."""
from __future__ import annotations

from .common import load_json, shot_dir

FILES = {
    "source/original.mp4": "fetch", "frames": "prepare", "Camera_Three.json": "solve",
    "qa/acceptance-camera.json": "qa-camera", "layout.json": "ground", "sim/simulation-qa.json": "simulate",
    "render/final/Composite.mp4": "render", "render/final/video-qa.json": "qa-video",
    "light.json": "light", "relight/final/Composite.mp4": "relight",
}


def main(cfg, work, args):
    d = shot_dir(cfg, work)
    print(f"shot {cfg['name']} ({cfg.get('title', '')}) in {d}")
    for rel, step in FILES.items():
        print(f"  {'x' if (d/rel).exists() else ' '} {step:10} {rel}")
    camera = d/"qa/acceptance-camera.json"
    if camera.is_file():
        r = load_json(camera)
        print(f"  camera QA pass={r['pass']} p95={r['p95_px']:.3f}px max={r['max_px']:.3f}px tracks={r['validation_tracks']}")
    point = d/"checkpoint.json"
    if point.is_file():
        c = load_json(point)
        print(f"  last stage: {c['stage']}" + (f"\n  next: {c['next']}" if c.get("next") else ""))
