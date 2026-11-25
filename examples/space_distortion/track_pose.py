"""MediaPipe Pose Landmarker (heavy) over the piece's frames (slv.py prep makes them).

    python track_pose.py [piece_dir]        default: piece/

out: work/pose.json         per frame: 33 image landmarks [x_px, y_px, z, visibility] and 33 metric world landmarks (hip-centred)
     work/mask/%04d.png     soft person mask (8 bit), used by render.py to keep the dancer readable
     work/solve_mask/%04d.png  the mask dilated by ~4% of the frame height and its shadow side: white = keep out of the COLMAP solve
Model: examples/fruit_physics/models/pose_landmarker_heavy.task (python scripts/fetch_assets.py pose-model)."""
import json, sys
from pathlib import Path
import cv2, numpy as np
import mediapipe as mp
from mediapipe.tasks.python import vision
from mediapipe.tasks.python.core import base_options

here = Path(__file__).resolve().parent
piece = Path(sys.argv[1]).resolve() if len(sys.argv) > 1 else here / "piece"
model = here.parent / "fruit_physics" / "models" / "pose_landmarker_heavy.task"
if not model.exists(): sys.exit(f"missing {model}: run  python scripts/fetch_assets.py pose-model")
meta = json.loads((piece / "work" / "meta.json").read_text()); fps = meta["fps"]
frames = sorted((piece / "work" / "frames").glob("*.jpg"))
for d in ("mask", "solve_mask"): (piece / "work" / d).mkdir(parents=True, exist_ok=True)
opts = vision.PoseLandmarkerOptions(
    base_options=base_options.BaseOptions(model_asset_path=str(model)),
    running_mode=vision.RunningMode.VIDEO, num_poses=1, output_segmentation_masks=True,
    min_pose_detection_confidence=0.5, min_tracking_confidence=0.5)
out = []
with vision.PoseLandmarker.create_from_options(opts) as lm:
    for i, f in enumerate(frames):
        bgr = cv2.imread(str(f)); H, W = bgr.shape[:2]; rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        res = lm.detect_for_video(mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb), int(round(i * 1000 / fps)))
        rec = {"frame": i, "ok": bool(res.pose_landmarks)}
        m8 = np.zeros((H, W), np.uint8)
        if res.pose_landmarks:
            rec["img"] = [[p.x * W, p.y * H, p.z, p.visibility or 0.0] for p in res.pose_landmarks[0]]
            rec["world"] = [[p.x, p.y, p.z] for p in res.pose_world_landmarks[0]]
            m8 = np.clip(res.segmentation_masks[0].numpy_view().squeeze() * 255, 0, 255).astype(np.uint8)
        cv2.imwrite(str(piece / "work" / "mask" / f"{i:04d}.png"), m8)
        r = max(3, int(0.04 * H)); k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * r + 1, 2 * r + 1))
        sm = cv2.dilate((m8 > 60).astype(np.uint8) * 255, k)
        if rec["ok"]:                                         # the cast shadow on the floor moves with the dancer: keep a box around the feet out too
            ys = [rec["img"][j][1] for j in (27, 28, 29, 30, 31, 32)]; xs = [rec["img"][j][0] for j in (27, 28, 29, 30, 31, 32)]
            cv2.rectangle(sm, (int(min(xs) - 0.06 * W), int(min(ys) - 0.05 * H)), (int(max(xs) + 0.30 * W), H), 255, -1)   # shadow falls to the right
        cv2.imwrite(str(piece / "work" / "solve_mask" / f"{i:04d}.png"), sm)
        out.append(rec)
(piece / "work" / "pose.json").write_text(json.dumps(out))
print("frames", len(out), "detected", sum(r["ok"] for r in out))
