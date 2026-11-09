"""Check Water/water-particles.bin after `node Water/bake-water.mjs` (used by CI).

The bake is deterministic on one CPU architecture, but Apple Silicon (arm64) and
x64 PCs round some floating-point steps differently and the SPH solver amplifies
that. So the file is checked for its layout on every machine, and its SHA-256 is
compared only with the hash recorded for the same architecture.
"""
from __future__ import annotations

import hashlib
from pathlib import Path
import platform
import struct
import sys

ROOT = Path(__file__).resolve().parents[1]
KNOWN = {
    # recorded 2026-11-09: macOS arm64 (M2 Max, GitHub macos-latest) and x64 (GitHub windows-latest + ubuntu-latest)
    "arm64": "5048a92137328b22df71bc9b2b8918512cdcab0082a4f5ecffe5cc22b98102b4",
    "x86_64": "3a5e2101b302807d0bd0d41c95a9bd316e301ead539573ce08a8291c1e124f84",
}
ALIASES = {"aarch64": "arm64", "amd64": "x86_64", "x64": "x86_64"}


def main(path=ROOT/"Water/water-particles.bin"):
    data = Path(path).read_bytes()
    magic, frames, count, num, den = data[:4], *struct.unpack_from("<4I", data, 4)
    problems = []
    if (magic, frames, count, num, den) != (b"WTR2", 192, 370, 24000, 1001):
        problems.append(f"header {magic!r} frames={frames} particles={count} fps={num}/{den}")
    if len(data) != 20+frames*count*6:
        problems.append(f"size {len(data)} != {20+frames*count*6}")
    values = struct.unpack_from(f"<{frames*count*3}h", data, 20) if not problems else ()
    if values and max(abs(v) for v in values) > round(.79*32767):
        problems.append("a particle left the reconstruction volume")
    digest = hashlib.sha256(data).hexdigest()
    machine = platform.machine().lower()
    arch = ALIASES.get(machine, machine)
    expected = KNOWN.get(arch)
    print(f"sha256 {digest} on {arch}; recorded for {arch}: {expected or 'none'}")
    if expected and digest != expected:
        problems.append(f"hash differs from the {arch} record")
    for problem in problems:
        print("FAIL:", problem)
    return 1 if problems else 0


if __name__ == "__main__":
    raise SystemExit(main(*sys.argv[1:]))
