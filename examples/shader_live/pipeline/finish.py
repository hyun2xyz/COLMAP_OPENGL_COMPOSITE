"""Full render, tracking pass and breakdown for one piece, run with the same Python that runs this file.

    python finish.py <piece_dir>        log: <piece_dir>/work/finish.log, last line FINISHED_OK on success"""
import subprocess
import sys
from pathlib import Path

here = Path(__file__).resolve().parent
piece = Path(sys.argv[1]).resolve()
log = piece / "work" / "finish.log"
steps = [
    [sys.executable, str(here / "render.py"), str(piece)],
    [sys.executable, str(here / "slv.py"), "track", str(piece)],
    [sys.executable, str(here / "breakdown.py"), str(piece)],
]
with open(log, "w", encoding="utf-8") as f:
    for cmd in steps:
        print("$", *cmd, file=f, flush=True)
        rc = subprocess.run(cmd, stdout=f, stderr=subprocess.STDOUT, cwd=str(here)).returncode
        if rc != 0:
            print(f"FAILED rc={rc}", file=f)
            print("failed, see", log)
            sys.exit(rc)
    print("FINISHED_OK", file=f)
print("FINISHED_OK", piece.name)
