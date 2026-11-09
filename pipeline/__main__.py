"""python -m pipeline <step> <shot> [options]

Steps, in order (each one reads the previous step's files in work/<shot>/):
  fetch       download the licensed clip + HDRI, verify SHA-256
  prepare     trim / resample / scale -> frames/, keyframes/
  solve       COLMAP keyframes (fixed focal) -> held-out refit -> per-frame PnP -> Camera_Three.json
  qa-camera   grade the camera with held-out landmarks (gate)
  ground      fit the floor, Y up, origin, scale -> Camera_Three_ground.json, layout.json
  simulate    water SPH / rigid spheres / cloth (CPU)
  render      native OpenGL composite -> Composite.mp4 + Alpha.mov
  qa-video    decode everything, numerical checks, review sheets
  light       measure where the light comes from in the plate; turn and scale the HDRI to match
  relight     light integration: cloth (estimated sun) or spheres (measured light, HDRI reflections,
              soft + contact shadows, texture, grain)
  status      what is done for a shot and the next command
  composite   fetch -> prepare -> solve -> qa-camera -> ground -> simulate -> render -> qa-video
"""
from __future__ import annotations

import argparse
import importlib
import sys
import time

from .common import StepError, shot_config, work_root

STEPS = {
    "fetch": "fetch", "prepare": "prepare", "solve": "solve", "qa-camera": "qa_camera", "ground": "ground",
    "simulate": "simulate", "render": "render", "qa-video": "qa_video", "light": "light", "relight": "relight",
    "status": "status",
}
COMPOSITE = ["fetch", "prepare", "solve", "qa-camera", "ground", "simulate", "render", "qa-video"]


def parser():
    p = argparse.ArgumentParser(prog="python -m pipeline", description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("step", choices=[*STEPS, "composite"])
    p.add_argument("shot", help="shots/<name>.json name, e.g. hall_water, courtyard_spheres, quarry_cloth")
    p.add_argument("--work", help="work folder (default: VFX_WORK or ./work). Use an external SSD for big runs.")
    p.add_argument("--output", help="new output folder for render/relight")
    p.add_argument("--render", help="qa-video: render folder to check (default work/<shot>/render/final)")
    p.add_argument("--frames", type=int, nargs="*", help="render/relight: only these frames as PNG stills")
    p.add_argument("--shadow", choices=["analytic", "ring", "none"], help="render: override the shot setting")
    p.add_argument("--ssaa", type=float, default=1.5, help="render: supersampling factor")
    return p


def main(argv=None):
    args = parser().parse_args(argv)
    try:
        cfg = shot_config(args.shot)
        work = work_root(args.work)
        for step in (COMPOSITE if args.step == "composite" else [args.step]):
            started = time.time()
            print(f"== {step} {cfg['name']}", flush=True)
            importlib.import_module(f".{STEPS[step]}", __package__).main(cfg, work, args)
            print(f"== {step} done in {time.time()-started:.0f}s", flush=True)
    except StepError as exc:
        print(f"STOP: {exc}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
