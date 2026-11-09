# COLMAP_OPENGL_COMPOSITE — classroom package

Read `.agents/skills/opengl-composite/SKILL.md` for this repository's workflow.
For a computer without prerequisites start with `START-HERE.md` and only the
current OS setup document. Do not run Python commands before locating Python.
This repository is the source of its project skill; do not install or change
global agent rules or skills.

- Teach one step at a time. Inspect the student's real files, tools and footage
  before choosing commands. Preserve originals, earlier outputs and student edits.
- Run steps through `python -m pipeline <step> <shot>` with the project `.venv`.
  Never overwrite an output folder; a re-render goes to a new `--output`.
- Gates are real: if `qa-camera` fails, do not ground/simulate/render on that
  camera. Change `camera.focal_px`, the trim or the clip, and say why.
- Name the render path exactly: ModernGL native OpenGL (3.3+, macOS 4.1 Metal).
  Three.js runs in Node only for water surface meshes and the Drift props. Do not
  call it browser WebGL.
- Focal length, zero distortion, scale (camera height 1.6), sun direction and HDRI
  lighting are assumptions. Report them as assumptions, never as measurements.
- Numerical QA is not visual QA. Before calling a shot done, look at the review
  sheets/stills and play the whole movie, or say it was not played.
- New footage and HDRI: find them yourself under `docs/materials.md` (licence
  checked on the distributor's page, parallax, no cuts). Never use footage whose
  licence you could not confirm, and never commit it.
- Long steps (solve, render on slow machines) run through `scripts/jobs.py`;
  read `docs/background.md`. No repeated polling loops.
- Footage, frames, renders, logos and `tools.local.json` stay in ignored `work/`
  (or the student's `VFX_WORK`). Before a commit check `git status` and stop if
  media or private paths would be included.
- The student works in their own fork: `origin` is the fork, `upstream` is
  `hyun2xyz/COLMAP_OPENGL_COMPOSITE`. Follow `docs/fork-and-update.md`; never
  reset, force-push or discard the student's commits.
- Agents ask for approval before network commands (git, gh, pip, npm, brew,
  winget, downloads) and writes outside this folder. Explain why; never change
  sandbox or security settings to avoid it.
- Read small JSON reports (`checkpoint.json`, `qa/*.json`, `render-report.json`)
  instead of long logs or full arrays.
