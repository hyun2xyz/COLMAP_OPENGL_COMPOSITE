"""Shared helpers for every pipeline step.

Paths, shot settings, external tools (ffmpeg / ffprobe / node), atomic JSON,
video encoding and the camera math that all renderers share.
"""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SHOTS = ROOT/"shots"
TOOLS = ("ffmpeg", "ffprobe", "node", "npm")


class StepError(RuntimeError):
    """A failed gate or missing input. The message tells the student what to do."""


# ---------------------------------------------------------------- files

def load_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def save_json(path, value):
    """Write JSON atomically so an interrupted run never leaves half a file."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name+".tmp")
    tmp.write_text(json.dumps(value, ensure_ascii=False, indent=2)+"\n", encoding="utf-8")
    os.replace(tmp, path)


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def new_output(path):
    """Outputs are never overwritten. Use a new folder for a re-render."""
    path = Path(path)
    if path.exists() and any(path.iterdir()):
        raise StepError(f"{path} already has files. Keep it and choose a new --output folder.")
    path.mkdir(parents=True, exist_ok=True)
    return path


# ---------------------------------------------------------------- tools

def tool(name):
    """Resolve an executable: NAME_BINARY env -> tools.local.json -> PATH."""
    configured = {}
    local = ROOT/"tools.local.json"
    if local.is_file():
        configured = load_json(local)
    value = os.environ.get(name.upper()+"_BINARY") or configured.get(name) or shutil.which(name)
    if not value or not Path(value).exists():
        raise StepError(f"{name} not found. Run: python -B scripts/setup_environment.py --check")
    return str(value)


def run(command, **kwargs):
    command = [str(v) for v in command]
    print("RUN", " ".join(command), flush=True)
    return subprocess.run(command, check=True, **kwargs)


def probe(path):
    out = subprocess.check_output([tool("ffprobe"), "-v", "error", "-show_streams", "-show_format",
                                   "-of", "json", str(path)])
    return json.loads(out)


# ---------------------------------------------------------------- shots

def shot_config(name):
    """Load shots/<name>.json (or an explicit .json path) and fill defaults."""
    path = Path(name)
    if path.suffix != ".json":
        path = SHOTS/f"{name}.json"
    if not path.is_file():
        options = ", ".join(sorted(p.stem for p in SHOTS.glob("*.json")))
        raise StepError(f"Unknown shot {name}. Available: {options}")
    cfg = load_json(path)
    cfg.setdefault("name", path.stem)
    cfg["config_path"] = str(path)
    return cfg


def work_root(value=None):
    """--work, then VFX_WORK, then ./work inside the repository (ignored by git)."""
    chosen = value or os.environ.get("VFX_WORK") or ROOT/"work"
    path = Path(chosen).expanduser().resolve()
    path.mkdir(parents=True, exist_ok=True)
    return path


def shot_dir(cfg, work):
    d = Path(work)/cfg["name"]
    d.mkdir(parents=True, exist_ok=True)
    return d


def checkpoint(d, stage, **info):
    """Record the last finished stage; `python -m pipeline status` reads this."""
    path = Path(d)/"checkpoint.json"
    history = load_json(path).get("history", []) if path.is_file() else []
    history.append(dict(stage=stage, time=time.strftime("%Y-%m-%dT%H:%M:%S%z")))
    save_json(path, dict(stage=stage, **info, history=history[-30:]))


def frame_path(d, i):
    return Path(d)/"frames"/f"frame_{i:06}.png"


# ---------------------------------------------------------------- cameras

def load_cameras(path):
    """Return (camera json, Nx4x4 camera-to-world matrices in Three/OpenGL axes)."""
    data = load_json(path)
    worlds = np.array([np.array(f["matrixWorld"], float).reshape(4, 4).T for f in data["frames"]])
    return data, worlds


def projection(cam, near=.02, far=50.):
    """OpenGL projection built directly from the pinhole K (fx, fy, cx, cy)."""
    w, h = cam["width"], cam["height"]
    return np.array([
        [2*cam["fx"]/w, 0, 1-2*cam["cx"]/w, 0],
        [0, 2*cam["fy"]/h, 2*cam["cy"]/h-1, 0],
        [0, 0, -(far+near)/(far-near), -2*far*near/(far-near)],
        [0, 0, -1, 0],
    ])


def gl_matrix(program, name, matrix):
    if name in program:
        program[name].write(np.ascontiguousarray(np.asarray(matrix).T, dtype="f4").tobytes())


def lookat(eye, target):
    z = eye-target
    z = z/np.linalg.norm(z)
    x = np.cross([0, 1, 0], z)
    x = x/np.linalg.norm(x)
    y = np.cross(z, x)
    c = np.eye(4)
    c[:3, :3] = np.stack([x, y, z], axis=1)
    c[:3, 3] = eye
    return np.linalg.inv(c)


def gl_context():
    """Native OpenGL 3.3+ context. Linux servers without a display need EGL."""
    import moderngl
    backend = os.environ.get("VFX_GL_BACKEND")
    try:
        return moderngl.create_standalone_context(require=330, **({"backend": backend} if backend else {}))
    except Exception as exc:  # moderngl raises plain Exception subclasses per platform
        raise StepError(f"OpenGL 3.3 context failed ({exc}). On Linux without a display set VFX_GL_BACKEND=egl.") from exc


def gl_info(ctx):
    return {k: ctx.info[k] for k in ("GL_VERSION", "GL_RENDERER", "GL_VENDOR")}


# ---------------------------------------------------------------- video

COLOR_TAGS = ["-color_primaries", "bt709", "-color_trc", "bt709", "-colorspace", "bt709", "-color_range", "tv"]


class Encoder:
    """Pipe raw RGB/RGBA frames into ffmpeg. H.264 for previews, ProRes 4444 for straight alpha."""

    def __init__(self, path, width, height, fps, alpha=False, crf=16):
        path = Path(path)
        if path.exists():
            raise StepError(f"{path} exists; outputs are never overwritten")
        codec = (["-c:v", "prores_ks", "-profile:v", "4", "-pix_fmt", "yuva444p10le"] if alpha else
                 ["-c:v", "libx264", "-crf", str(crf), "-preset", "medium", "-pix_fmt", "yuv420p"])
        self.path = path
        self.size = width*height*(4 if alpha else 3)
        self.process = subprocess.Popen(
            [tool("ffmpeg"), "-v", "error", "-n", "-f", "rawvideo", "-pix_fmt", "rgba" if alpha else "rgb24",
             "-s", f"{width}x{height}", "-r", str(fps), "-i", "-", "-an",
             "-vf", "scale=in_range=pc:out_range=tv:out_color_matrix=bt709", *codec, "-threads", "4",
             *COLOR_TAGS, "-movflags", "+faststart", str(path)], stdin=subprocess.PIPE)

    def write(self, frame):
        data = np.ascontiguousarray(frame, dtype=np.uint8).tobytes()
        if len(data) != self.size:
            raise StepError("Frame size does not match the encoder")
        self.process.stdin.write(data)

    def finish(self):
        self.process.stdin.close()
        if self.process.wait():
            raise StepError(f"ffmpeg failed while writing {self.path}")


def decode_rgb(path, width, height):
    """Yield RGB frames decoded by ffmpeg (respects BT.709; OpenCV's decoder may not)."""
    process = subprocess.Popen([tool("ffmpeg"), "-v", "error", "-i", str(path), "-f", "rawvideo",
                                "-pix_fmt", "rgb24", "-"], stdout=subprocess.PIPE)
    size = width*height*3
    while True:
        raw = process.stdout.read(size)
        if not raw:
            break
        if len(raw) != size:
            raise StepError(f"Partial frame in {path}")
        yield np.frombuffer(raw, np.uint8).reshape(height, width, 3)
    if process.wait():
        raise StepError(f"Decoding failed: {path}")


def python():
    return sys.executable
