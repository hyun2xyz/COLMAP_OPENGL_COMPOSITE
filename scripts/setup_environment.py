"""Project-only setup helper. Runs with Python's standard library before any install.

--check      read/probe only: Python, .venv packages, ffmpeg/ffprobe/node/npm, Source/node_modules, OpenGL 3.3
--configure  write the verified absolute tool paths to tools.local.json (no PATH change)
--install    create/reuse this folder's .venv, pip install requirements.txt, npm ci --prefix Source
             add "--with body" for the optional body-tracking packages (examples/fruit_physics)
The check also lists, under "examples", what each example outside `python -m pipeline` still needs.
It never installs OS applications, edits system PATH or creates schedulers.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import platform
import shutil
import subprocess
import sys
import tempfile
import venv

ROOT = Path(__file__).resolve().parents[1]
TOOLS = ("ffmpeg", "ffprobe", "node", "npm")
SUPPORTED_PYTHON = {(3, 12), (3, 13)}
PACKAGES = ("numpy", "opencv-python-headless", "scipy", "pycolmap", "moderngl", "pillow")
# Optional package groups: --install --with <group>
EXTRAS = {"body": ("requirements-body.txt", ("mediapipe", "mujoco"))}
# Examples outside the pipeline. Their inputs are not in git; scripts/fetch_assets.py downloads them.
FRUIT_CLIP = "examples/fruit_physics/source/Pexels_7525471_720x1280.mp4"
EXAMPLES = {
    "backboard_water": {},
    "shader_live": {"files": ("examples/shader_live/footage/Pexels_10708812.mp4",)},
    "fruit_physics": {"extras": ("body",), "files": (FRUIT_CLIP,"examples/fruit_physics/models/pose_landmarker_heavy.task")},
}


def settings():
    path = ROOT/"tools.local.json"
    if not path.exists():
        return {}
    data = json.loads(path.read_text(encoding="utf-8-sig"))
    if not isinstance(data, dict) or set(data)-set(TOOLS):
        raise ValueError("tools.local.json may only contain ffmpeg/ffprobe/node/npm paths")
    return data


def common_candidates(name):
    system = platform.system()
    if system == "Darwin":
        return [Path(p)/"bin"/name for p in ("/opt/homebrew", "/usr/local")]
    if system != "Windows":
        return [Path("/usr/bin")/name, Path("/usr/local/bin")/name]
    exe = name+(".cmd" if name == "npm" else ".exe")
    bases = []
    if name in ("node", "npm"):
        bases += [Path(os.environ.get("ProgramFiles", "C:/Program Files"))/"nodejs"]
    else:
        local = os.environ.get("LOCALAPPDATA")
        bases += [Path(local)/"Microsoft/WinGet/Links"] if local else []
        bases += [Path("C:/Tools/FFmpeg/bin"), Path("C:/ffmpeg/bin")]
    return [b/exe for b in bases]


def probe_tool(name, path):
    try:
        result = subprocess.run([str(path), "--version" if name in ("node", "npm") else "-version"],
                                capture_output=True, encoding="utf-8", errors="replace", timeout=20,
                                shell=(os.name == "nt" and name == "npm"))
    except (OSError, subprocess.SubprocessError) as exc:
        return {"error": str(exc)}
    if result.returncode:
        return {"error": f"exited {result.returncode}", "tail": (result.stdout+result.stderr)[-500:]}
    return {"version": (result.stdout+result.stderr).strip().splitlines()[0][:120]}


def find_tools(explicit=None):
    explicit, configured, found = explicit or {}, settings(), {}
    for name in TOOLS:
        options = [(explicit.get(name), "argument"), (os.environ.get(name.upper()+"_BINARY"), "environment"),
                   (configured.get(name), "tools.local.json"), (shutil.which(name), "PATH")]
        options += [(str(p), "common location") for p in common_candidates(name)]
        chosen = next(((v, s) for v, s in options if v and Path(v).exists()), None)
        if not chosen:
            found[name] = {"path": None, "error": "Not found; follow docs/setup-macos.md or docs/setup-windows.md"}
            continue
        path = Path(os.path.abspath(chosen[0]))
        found[name] = {"path": str(path), "source": chosen[1], **probe_tool(name, path)}
    return found


def venv_python():
    return ROOT/".venv"/("Scripts/python.exe" if os.name == "nt" else "bin/python")


def package_versions(python, names):
    """Python version and the installed version (or None) of each distribution in that interpreter."""
    code = ("import importlib.metadata as m,json,sys\np={}\nfor n in %r:\n try:p[n]=m.version(n)\n"
            " except m.PackageNotFoundError:p[n]=None\nprint(json.dumps(dict(version=list(sys.version_info[:3]),packages=p)))") % (tuple(names),)
    return json.loads(subprocess.run([str(python), "-I", "-B", "-c", code], check=True, capture_output=True,
                                     encoding="utf-8", timeout=30).stdout)


def venv_state():
    python = venv_python()
    state = {"python": str(python), "ready": False}
    if not python.is_file():
        return {**state, "error": "Project .venv missing; run --install"}
    try:
        data = package_versions(python, PACKAGES)
    except (OSError, ValueError, subprocess.SubprocessError) as exc:
        return {**state, "error": str(exc)}
    missing = [n for n, v in data["packages"].items() if not v]
    ok = tuple(data["version"][:2]) in SUPPORTED_PYTHON and not missing
    return {**state, **data, "missing": missing, "ready": ok}


def opengl_state():
    python = venv_python()
    if not python.is_file():
        return {"ready": False, "error": "needs .venv"}
    code = ("import moderngl,json,os\nb=os.environ.get('VFX_GL_BACKEND')\n"
            "c=moderngl.create_standalone_context(require=330,**({'backend':b} if b else {}))\n"
            "print(json.dumps({k:c.info[k] for k in ('GL_VERSION','GL_RENDERER','GL_VENDOR')}))")
    try:
        result = subprocess.run([str(python), "-I", "-B", "-c", code], capture_output=True, encoding="utf-8", timeout=60)
    except (OSError, subprocess.SubprocessError) as exc:
        return {"ready": False, "error": str(exc)}
    if result.returncode:
        return {"ready": False, "error": result.stderr.strip()[-400:],
                "hint": "Update the GPU driver. On Linux without a display set VFX_GL_BACKEND=egl."}
    return {"ready": True, **json.loads(result.stdout)}


def examples_state(core_ready):
    """What each example outside the pipeline still needs. Does not change the top-level "ready"."""
    python, extras = venv_python(), {}
    for name, (_, names) in EXTRAS.items():
        try:
            versions = package_versions(python, names)["packages"] if python.is_file() else dict.fromkeys(names)
        except (OSError, ValueError, subprocess.SubprocessError):
            versions = dict.fromkeys(names)
        extras[name] = {"ready": all(versions.values()), "packages": versions}
    examples = {}
    for name, need in EXAMPLES.items():
        missing = [] if core_ready else ["core setup: --install, then --check must say ready"]
        missing += [f"packages: --install --with {e}" for e in need.get("extras", ()) if not extras[e]["ready"]]
        missing += [f"input: {f} (python scripts/fetch_assets.py list)" for f in need.get("files", ()) if not (ROOT/f).exists()]
        missing += [f"tool: {t} on PATH" for t in need.get("tools", ()) if not shutil.which(t)]
        examples[name] = {"ready": not missing, "missing": missing}
    return extras, examples


def check(explicit=None):
    tools, env = find_tools(explicit), venv_state()
    node_modules = (ROOT/"Source/node_modules/three").is_dir()
    gl = opengl_state() if env["ready"] else {"ready": False, "error": "needs a ready .venv"}
    ready = env["ready"] and node_modules and gl["ready"] and all("error" not in t for t in tools.values())
    extras, examples = examples_state(ready)
    return {"schema": "colmap-opengl-composite.setup.v1", "root": str(ROOT), "os": platform.platform(),
            "python": {"path": sys.executable, "version": platform.python_version(),
                       "supported": sys.version_info[:2] in SUPPORTED_PYTHON},
            "venv": env, "tools": tools, "three_js": node_modules, "opengl": gl,
            "extras": extras, "examples": examples, "ready": ready}


def configure(explicit=None):
    tools = find_tools(explicit)
    missing = [n for n, t in tools.items() if "error" in t]
    if missing:
        return {"configured": False, "tools": tools, "error": "No file written; missing: "+", ".join(missing)}
    with tempfile.NamedTemporaryFile("w", encoding="utf-8", dir=ROOT, suffix=".json", delete=False) as stream:
        json.dump({n: tools[n]["path"] for n in TOOLS}, stream, indent=2)
        temporary = stream.name
    os.replace(temporary, ROOT/"tools.local.json")
    return {"configured": True, "config": str(ROOT/"tools.local.json"), "tools": tools, "system_path_changed": False}


def install(extras=()):
    if not (ROOT/".venv").exists():
        if sys.version_info[:2] not in SUPPORTED_PYTHON:
            raise ValueError("Create the project environment with Python 3.13 (or 3.12)")
        venv.EnvBuilder(with_pip=True).create(ROOT/".venv")
    env = {k: v for k, v in os.environ.items() if not k.startswith("PIP_")}
    env["PIP_CONFIG_FILE"] = os.devnull
    subprocess.run([str(venv_python()), "-I", "-m", "pip", "install", "--require-virtualenv", "-r",
                    str(ROOT/"requirements.txt")], check=True, env=env)
    for name in extras:
        subprocess.run([str(venv_python()), "-I", "-m", "pip", "install", "--require-virtualenv", "-r",
                        str(ROOT/EXTRAS[name][0])], check=True, env=env)
    npm = find_tools().get("npm", {})
    if not npm.get("path"):
        raise ValueError("npm not found; install Node.js LTS first (docs/setup-*.md)")
    subprocess.run([npm["path"], "ci", "--prefix", str(ROOT/"Source"), "--no-audit", "--no-fund"], check=True,
                   shell=(os.name == "nt"))
    return {"installed": True, "venv_python": str(venv_python()), "node_modules": str(ROOT/"Source/node_modules"),
            "extras": list(extras)}


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    action = p.add_mutually_exclusive_group(required=True)
    action.add_argument("--check", action="store_true")
    action.add_argument("--configure", action="store_true")
    action.add_argument("--install", action="store_true")
    p.add_argument("--with", dest="extras", nargs="+", choices=sorted(EXTRAS), default=[],
                   help="with --install: also install an optional package group")
    for name in TOOLS:
        p.add_argument("--"+name, help="actual executable path")
    args = p.parse_args(argv)
    explicit = {n: getattr(args, n) for n in TOOLS if getattr(args, n)}
    try:
        result = install(args.extras) if args.install else configure(explicit) if args.configure else check(explicit)
    except (ValueError, OSError, subprocess.SubprocessError) as exc:
        print(json.dumps({"error": str(exc)}, ensure_ascii=False), file=sys.stderr)
        return 1
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0 if result.get("ready") or result.get("installed") or result.get("configured") else 1


if __name__ == "__main__":
    raise SystemExit(main())
