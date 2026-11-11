"""MediaPipe Pose Landmarker (heavy) over every frame: 33 image landmarks, 33 metric world landmarks
(hip-centred, metres) and the person segmentation mask.
out: work/pose.json, work/mask/%04d.png (8-bit soft mask)"""
import json, sys
from pathlib import Path
import cv2, numpy as np
import mediapipe as mp
from mediapipe.tasks.python import vision
from mediapipe.tasks.python.core import base_options

root = Path(__file__).parent
frames = sorted((root / "work" / "frames").glob("*.png"))
(root / "work" / "mask").mkdir(parents=True, exist_ok=True)
opts = vision.PoseLandmarkerOptions(
    base_options=base_options.BaseOptions(model_asset_path=str(root / "models" / "pose_landmarker_heavy.task")),
    running_mode=vision.RunningMode.VIDEO, num_poses=1, output_segmentation_masks=True,
    min_pose_detection_confidence=0.5, min_tracking_confidence=0.5)
out = []
with vision.PoseLandmarker.create_from_options(opts) as lm:
    for i, f in enumerate(frames):
        rgb = cv2.cvtColor(cv2.imread(str(f)), cv2.COLOR_BGR2RGB)
        res = lm.detect_for_video(mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb), int(i * 40))
        rec = {"frame": i, "ok": bool(res.pose_landmarks)}
        if res.pose_landmarks:
            rec["img"] = [[p.x, p.y, p.z, p.visibility or 0.0, p.presence or 0.0] for p in res.pose_landmarks[0]]
            rec["world"] = [[p.x, p.y, p.z] for p in res.pose_world_landmarks[0]]
            m = res.segmentation_masks[0].numpy_view()
            cv2.imwrite(str(root / "work" / "mask" / f"{i:04d}.png"), np.clip(m.squeeze() * 255, 0, 255).astype(np.uint8))
        else:
            cv2.imwrite(str(root / "work" / "mask" / f"{i:04d}.png"), np.zeros(rgb.shape[:2], np.uint8))
        out.append(rec)
(root / "work" / "pose.json").write_text(json.dumps(out))
print("frames", len(out), "detected", sum(r["ok"] for r in out))
