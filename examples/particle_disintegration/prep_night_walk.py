"""Preprocess Pexels 11839713 (Foggy Snowy Night Walk — 100% Real Clean Plate + Hybrid Matte + 3D Road Depth).

Extracts 200 frames (src f=260..459, 8.00s @ 25fps), builds a 58-frame real temporal-median clean plate from the
empty end of the locked-tripod shot (src f=610..667), tracks the walking man with MediaPipe Pose + Clean-Plate
Difference Fusion for a razor-crisp silhouette matte, and fits true 3D perspective depth along the snowy road.
"""
import json, math
from pathlib import Path
import numpy as np, cv2
import mediapipe as mp

here = Path(__file__).resolve().parent
VID  = here / "footage" / "Pexels_11839713.mp4"
WORK = here / "piece_night" / "work"


def find_pose_model():
    root = here.parents[1]
    for p in root.rglob("pose_landmarker*.task"):
        return str(p)
    return None


def main():
    WORK.mkdir(parents=True, exist_ok=True)
    frames_dir = WORK / "frames"; frames_dir.mkdir(exist_ok=True)
    mask_dir   = WORK / "mask";   mask_dir.mkdir(exist_ok=True)

    cap = cv2.VideoCapture(str(VID))
    n_tot = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    W = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    H = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0

    # 1. Build 100% REAL zero-AI temporal-median clean plate from empty-street tail (src f=610..667)!
    tail_frames = []
    cap.set(cv2.CAP_PROP_POS_FRAMES, 610)
    for fi in range(610, min(668, n_tot)):
        ok, im = cap.read()
        if not ok:
            break
        if fi % 2 == 0:
            tail_frames.append(im)
    clean = np.median(np.stack(tail_frames, axis=0), axis=0).astype(np.uint8)
    cv2.imwrite(str(WORK / "clean_plate.png"), clean)
    clean_f = clean.astype(np.float32)
    clean_g = cv2.cvtColor(clean, cv2.COLOR_BGR2GRAY).astype(np.float32)

    # 2. Extract 200 hero frames (src f=260..459 -> 8.00s cinematic walk from mid-shot to dramatic close-up!)
    src_start, N = 260, 200
    cap.set(cv2.CAP_PROP_POS_FRAMES, src_start)
    imgs = []
    for i in range(N):
        ok, im = cap.read()
        if not ok:
            break
        imgs.append(im)
        cv2.imwrite(str(frames_dir / f"{i:04d}.jpg"), im, [cv2.IMWRITE_JPEG_QUALITY, 95])
    cap.release()
    N = len(imgs)

    # 3. Hybrid MediaPipe Pose Landmarker + Clean-Plate Difference Matte (handles dark night winter coat cleanly!)
    model_path = find_pose_model()
    mp_masks = [None] * N
    if model_path:
        print("Using MediaPipe PoseLandmarker:", model_path)
        BaseOptions = mp.tasks.BaseOptions
        PoseLandmarker = mp.tasks.vision.PoseLandmarker
        PoseLandmarkerOptions = mp.tasks.vision.PoseLandmarkerOptions
        RunningMode = mp.tasks.vision.RunningMode
        opts = PoseLandmarkerOptions(
            base_options=BaseOptions(model_asset_path=model_path),
            running_mode=RunningMode.VIDEO,
            output_segmentation_masks=True,
            num_poses=1,
            min_pose_detection_confidence=0.25,
            min_tracking_confidence=0.25,
        )
        with PoseLandmarker.create_from_options(opts) as landmarker:
            for i, im in enumerate(imgs):
                # Boost exposure slightly ONLY for MediaPipe vision detector so it locks solidly in dark fog!
                vis_in = np.clip(im.astype(np.float32) * 2.2 + 12.0, 0, 255).astype(np.uint8)
                mp_img = mp.Image(image_format=mp.ImageFormat.SRGB, data=cv2.cvtColor(vis_in, cv2.COLOR_BGR2RGB))
                res = landmarker.detect_for_video(mp_img, int(round(i * 1000.0 / fps)))
                if res.segmentation_masks:
                    sm = res.segmentation_masks[0].numpy_view()
                    if sm.ndim == 3:
                        sm = sm[:, :, 0]
                    mp_masks[i] = cv2.resize(sm.astype(np.float32), (W, H), interpolation=cv2.INTER_LINEAR)

    # 4. Fuse MediaPipe mask + Exact Locked-Tripod Difference Matte -> razor-sharp silhouette + clean 3D road depth!
    bboxes = np.zeros((N, 4), dtype=np.float64)  # x0, y0, x1, y1
    k_small = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    k_med   = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (11, 11))
    k_close = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (17, 17))

    # Track main connected component smoothly across frames (starts around center road x~1100..1250, y~450..850)
    prev_cx, prev_cy = 1190.0, 620.0
    qa_panels = []
    for i, im in enumerate(imgs):
        # Exact color + luminance difference against real empty-street clean plate
        diff_rgb = np.linalg.norm(im.astype(np.float32) - clean_f, axis=2)
        diff_g   = np.abs(cv2.cvtColor(im, cv2.COLOR_BGR2GRAY).astype(np.float32) - clean_g)
        diff = 0.65 * diff_rgb + 0.35 * diff_g
        diff = cv2.GaussianBlur(diff, (0, 0), 1.4)

        # Restrict search corridor to road zone (ignore tiny foliage shimmer on extreme right/top)
        roi = np.zeros((H, W), dtype=np.uint8)
        roi[300:H, 680:1620] = 1

        # Strong diff core (dark coat against misty streetlamp backdrop!)
        diff_bin = ((diff > 14.5) & (roi > 0)).astype(np.uint8)
        diff_bin = cv2.morphologyEx(diff_bin, cv2.MORPH_OPEN, k_small)
        diff_bin = cv2.morphologyEx(diff_bin, cv2.MORPH_CLOSE, k_close)

        # Combine with MediaPipe segmentation when available
        if mp_masks[i] is not None:
            mp_bin = ((mp_masks[i] > 0.35) & (roi > 0)).astype(np.uint8)
            comb = np.maximum(diff_bin, mp_bin)
        else:
            comb = diff_bin

        # Keep largest connected component near previous person centroid (eliminates any stray snow/fog noise!)
        num_l, labels, stats, cents = cv2.connectedComponentsWithStats(comb, connectivity=8)
        best_l = -1
        best_score = -1e9
        for l in range(1, num_l):
            area = stats[l, cv2.CC_STAT_AREA]
            if area < 4000:
                continue
            cx, cy = cents[l]
            dist = math.hypot(cx - prev_cx, cy - prev_cy)
            score = area - dist * 120.0
            if score > best_score:
                best_score = score
                best_l = l

        if best_l > 0:
            body = (labels == best_l).astype(np.uint8)
            prev_cx, prev_cy = float(cents[best_l, 0]), float(cents[best_l, 1])
        else:
            body = comb

        # Fill interior holes cleanly (e.g. if coat collar briefly matches mid-grey fog luminance)
        body = cv2.morphologyEx(body, cv2.MORPH_CLOSE, k_close)
        cnts, _ = cv2.findContours(body, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        solid = np.zeros((H, W), dtype=np.uint8)
        if cnts:
            big = max(cnts, key=cv2.contourArea)
            cv2.drawContours(solid, [big], -1, 255, thickness=cv2.FILLED)

        # Refine silhouette alpha right along the edge using soft diff + MediaPipe probability
        dil = cv2.dilate((solid > 0).astype(np.uint8), k_med)
        erode = cv2.erode((solid > 0).astype(np.uint8), k_small)
        soft_diff = np.clip((diff - 8.5) / 14.0, 0.0, 1.0)
        if mp_masks[i] is not None:
            edge_prob = np.maximum(soft_diff, mp_masks[i])
        else:
            edge_prob = soft_diff
        matte = np.where(erode > 0, 1.0, np.where(dil > 0, edge_prob, 0.0)).astype(np.float32)
        matte = cv2.GaussianBlur(matte, (0, 0), 1.0)
        matte_u8 = np.clip(matte * 255.0 + 0.5, 0, 255).astype(np.uint8)
        cv2.imwrite(str(mask_dir / f"{i:04d}.png"), matte_u8)

        ys, xs = np.where(matte_u8 > 100)
        if len(ys) > 0:
            bboxes[i] = [xs.min(), ys.min(), xs.max(), ys.max()]
        elif i > 0:
            bboxes[i] = bboxes[i - 1]

        if i in (0, 40, 80, 120, 160, 195):
            vis = im.copy()
            # Tint masked silhouette with warm amber border overlay for quick QA
            edge = cv2.Canny(matte_u8, 60, 160)
            edge = cv2.dilate(edge, np.ones((3, 3), np.uint8))
            vis[edge > 0] = (40, 180, 255)
            cv2.putText(vis, f"f={i} h={int(bboxes[i,3]-bboxes[i,1])}px", (25, 70),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.7, (80, 220, 255), 3)
            qa_panels.append(cv2.resize(vis, (640, 360)))

    # 5. Fit smooth 3D metric camera & walking depth along flat snowy road!
    #    Locked tripod at eye/chest height z_cam = 1.45m, focal f_px = 1680px (~32mm cinematic lens)
    #    Person height H_real = 1.78m -> raw depth = f_px * H_real / h_px -> smooth cubic fit!
    f_px = 1680.0
    h_px = np.maximum(bboxes[:, 3] - bboxes[:, 1], 120.0)
    raw_depth = f_px * 1.78 / h_px
    ts = np.arange(N, dtype=np.float64)
    poly = np.poly1d(np.polyfit(ts, raw_depth, deg=2))
    depths = np.clip(poly(ts), 3.0, 18.0)

    # Fit smooth camera pitch from horizon (vanishing point of streetlamps sits around y_hor ~ 435px!)
    y_hor = 435.0
    pitch = math.atan2(y_hor - 0.5 * H, f_px)  # slight downward/upward tilt so z=0 road matches foot y!
    cp, sp = math.cos(pitch), math.sin(pitch)
    # OpenCV cam->world: cam +X = world +X (right), cam +Y = down (-Z tilted), cam +Z = forward (+Y tilted)
    R0 = np.array([
        [1.0,  0.0,  0.0],
        [0.0, -sp,   cp ],
        [0.0, -cp,  -sp ],
    ], dtype=np.float64)
    # Calibrate camera height z_cam so feet at frame 0 project cleanly to z = 0.0m!
    foot_y0 = float(bboxes[0, 3])
    ray_foot0 = R0 @ np.array([0.0, (foot_y0 - 0.5 * H) / f_px, 1.0])
    z_cam = float(-ray_foot0[2] * depths[0])
    ro0 = np.array([0.0, 0.0, z_cam], dtype=np.float64)

    # Save scene.json, meta.json, track.npz
    (WORK / "meta.json").write_text(json.dumps({
        "W": W, "H": H, "N": N, "fps": float(fps), "fps_num": 25, "fps_den": 1, "src_start": src_start
    }, indent=2))
    (WORK / "scene.json").write_text(json.dumps({
        "f": f_px, "cam": [ro0.tolist()] * N, "rot": [R0.tolist()] * N
    }))
    np.savez(WORK / "track.npz", bboxes=bboxes, depth=depths)

    if len(qa_panels) == 6:
        sheet = np.vstack([np.hstack(qa_panels[:3]), np.hstack(qa_panels[3:])])
        cv2.imwrite(str(WORK / "check_matte_night.jpg"), sheet, [cv2.IMWRITE_JPEG_QUALITY, 92])
    print(f"Prepared piece_night: {N}f @ {fps}fps | depth {depths[0]:.2f}m -> {depths[-1]:.2f}m | z_cam={z_cam:.2f}m | h_px {h_px[0]:.0f}->{h_px[-1]:.0f}")


if __name__ == "__main__":
    main()
