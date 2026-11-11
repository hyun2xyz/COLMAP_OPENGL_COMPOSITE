---
name: opengl-composite
description: Set up a macOS/Windows/Linux computer and guide a student, one step at a time, from finding a licensed live-action clip to a COLMAP/PnP camera, a CPU simulation (SPH water, rigid spheres, PBD cloth) rendered with native OpenGL, and numerical + visual QA. Also covers the examples outside the pipeline: world-locked GLSL effects (Shader Live), MediaPipe body tracking with MuJoCo physics, and backboard water. Use inside this repository only.
---

# OpenGL composite

Repository-bound teaching skill. Find the repository root that contains
`pipeline/__main__.py` and run everything from there with the project `.venv`
(`.venv/bin/python` or `.venv\Scripts\python.exe`). Copying this file alone does
not install anything.

## First interaction

1. Inspect the actual OS/shell and the repository root. If Python or tools are
   missing, follow [START-HERE](../../../START-HERE.md) and only the matching
   setup: [macOS](../../../docs/setup-macos.md) or [Windows](../../../docs/setup-windows.md).
   Then run `python -B scripts/setup_environment.py --check` and require
   `"ready": true` (it really opens an OpenGL 3.3 context).
2. Choose the work folder. Default `work/` in the repo; for big runs use an
   external drive via `--work` or `VFX_WORK`. Never write footage into git.
3. Explain the next step in two or three sentences, then run it. Do not ask the
   student to read every document.

## Lesson path (one example shot first: `hall_water`)

Commands and what to check after each are in [workflow](../../../docs/workflow.md).

- `fetch` downloads from the distributor recorded in `shots/<shot>.json` and
  checks SHA-256. A mismatch means the file changed: stop, re-check the license.
- `prepare` cuts the plate (trim, CFR, size) and copies keyframes.
- `solve` (background job): pycolmap SIFT on CPU, sequential matching with NO
  focal prior, then fixed-focal incremental mapping, 20% landmark hold-out, BA,
  per-frame LK+PnP. All keyframes must register; never interpolate missing poses.
- `qa-camera` is a gate: p95 < 2 px, max < 8 px, movement p95 < 1 px, >= 40
  tracks. On failure do not continue; try another `camera.focal_px`, a shorter
  trim or another clip, one evidence-based change at a time.
- `ground`: open `qa/ground-fit.jpg` and confirm the green dots lie on the real
  floor and the red cross is where the object should stand. Adjust
  `ground.roi` / `ground.anchor` / `ground.width_fraction` otherwise.
- `simulate`, `render`, `qa-video`: then open the `review_*.jpg` sheets and play
  `Composite.mp4` start to end. Check drift, sliding, jitter, wrong occlusion,
  black or missing frames. Only then call the shot done.
- Light integration (`relight`, then `qa-video --render <shot>/relight/final`):
  cloth uses an estimated sun (record it as estimated). Sphere shots first run
  `light`: open `qa/light-estimate.jpg` and confirm the arrows point at the real
  lamps/windows before relighting. Say which values are measured (directions,
  colours, floor, grain) and which are assumed (`clipped_gain`, light radius).
  A top-down plate cannot measure light; keep its `render` result.

The lesson ends at a reviewed composite.

## Examples outside the pipeline

`python -m pipeline` covers the shots in `shots/`. Three more examples live in `examples/`, each with its own README
and commands; read only the one in use. `scripts/setup_environment.py --check` lists what each still needs under
`"examples"`. Inputs: `python scripts/fetch_assets.py list | footage <number> | pose-model`. These examples call
`ffmpeg`/`ffprobe` from PATH and keep their outputs inside the example folder (ignored by git), not in `work/`.

| Example | What it teaches | Check that it worked |
| --- | --- | --- |
| `backboard_water` | four-corner PnP, GLSL drops | `--demo --frames 12` writes `preview_006.png` with drops on a backboard |
| `shader_live` | camera and plane solve (COLMAP or plane homography), one world-locked GLSL fragment shader | `work/check_plane.jpg` grid stays on the surface in all three frames, then stills, then `finish.py` ends with `FINISHED_OK` |
| `fruit_physics` | MediaPipe pose to invisible rigid bodies, MuJoCo zero gravity, OpenGL fruit (needs `--install --with body`) | `Fruit_ZeroG_Physics.mp4`; fruit bounce off the dancer and pass behind her |

Rules that came from real failures:

- A solve that registers every frame with a small error can still be wrong. Open `work/check_plane.jpg` before
  rendering. If the grid slides or tilts, change the solver, not the shader: straight-down or flat shots use
  `"solver": "homography"`; clouds, sun flare and water are kept out with `mask_roi`.
- In GLSL edited through string replacement, a `//` comment in the middle of a line removes the rest of that line.
  Use `/* */` unless the comment ends the line.
- A passed still does not prove the full render. Check `work/finish.log` for `FINISHED_OK`, then read frames back
  from the encoded mp4 (`verify_sheet.py`).
- The Straight-alpha layers of `shader_live` are exact only over their own plate.

Body tracking: MediaPipe first, SAM 3D Body only after approval. `fruit_physics` works with MediaPipe, which downloads
freely. SAM 3D Body gives a full body mesh, but its weights are gated: the student must request access on Hugging
Face and wait, and it needs PyTorch and a capable GPU. Give the student [docs/sam-3d-body.md](../../../docs/sam-3d-body.md),
let them submit the form themselves, and continue with MediaPipe meanwhile. Never look for the weights elsewhere.

## Honesty rules

- Render path: ModernGL native OpenGL, not browser WebGL. Three.js only makes
  surface meshes in Node.
- Focal length, zero distortion, scale (camera height 1.6), lighting, sun
  direction and simulation constants are assumptions; say so.
- Reflections come from a separate CC0 HDRI, turned and scaled to the plate's
  light; it is not captured on set. `light` measures only directions the camera
  saw, and clipped lamps make their strength an assumption. Never claim a full
  measured light probe.
- Automatic checks, still review and full playback are three different claims.

## Long calculations

Read [background](../../../docs/background.md). Start with
`python scripts/jobs.py start --job <absolute new folder> --estimate-minutes N --timeout-minutes M -- <python> -m pipeline <step> <shot>` from the repository root, tell the
student the job and a check time based on a previous similar run (or "unknown"),
create one one-shot wakeup with the agent's real scheduling tool if it exists,
record it with `jobs.py wakeup`, and end the turn. On wake, remove the
reservation first, read `jobs.py status` once, and continue or report. Never
restart a failed job on your own and never claim a schedule you did not create.

## New shots and materials

When the student wants their own shot, find the footage and HDRI yourself by
following [materials](../../../docs/materials.md); do not ask the student to
search first. In short:

- Search with `python -B scripts/find_materials.py footage "<keywords>"` (Pexels
  API when `PEXELS_API_KEY` is set, otherwise it prints search pages to open) and
  `python -B scripts/find_materials.py hdri "<keywords>"` (Poly Haven, CC0).
- Accept a clip only if the camera translates (parallax), there is no cut,
  zoom is fixed, static texture fills most of the frame and 4-5 s are usable.
  Look at frames before choosing; say what you looked at.
- Verify the license on the distributor's own page (Pexels License, Pixabay
  Content License, CC0). "Free" is not "no copyright". Never use footage from
  social media, film/TV, YouTube downloads, or a page without a license.
- Scaffold with `python -B scripts/find_materials.py new-shot <name> --like <shot>
  --page <url> --download <url> --creator <name> --license <id>`, then run
  `fetch` and copy the printed SHA-256 into the shot file.
- Show the student the candidate (page, creator, license, why it fits) before a
  long solve. If `qa-camera` fails, try another clip rather than forcing it.

Save a short `work/<shot>/next.md` after each lesson: files, what was verified,
what failed, one next step.
