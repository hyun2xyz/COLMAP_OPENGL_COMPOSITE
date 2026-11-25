"""Preprocess Pexels 31808102 (Misty Forest Walk — SIFT+RANSAC Homography Real Clean Plate + Hybrid Matte + 3D Depth).

Extracts 240 hero frames (src f=45..284, 8.01s @ 29.97fps), aligns 40 frames across the full 22s clip via
static-background SIFT+RANSAC homographies to build a 100% REAL zero-AI empty misty-forest clean plate,
computes per-frame homographies H[i] for the 240 hero frames, fuses MediaPipe Pose + aligned difference matte,
and fits smooth 3D walking depth receding into the misty forest (~4.6m -> 9.8m).
"""
import json, math
from pathlib import Path
import numpy as np, cv2
import mediapipe as mp

here = Path(__file__).resolve().parent
VID  = here / "footage" / "Pexels_31808102.mp4"
WORK = here / "piece_forest" / "work"


def find_pose_model():
    root = here.parents[1]
    for p in root.rglob("pose_landmarker*.task"):
        return str(p)
    return None


def compute_homography(src_gray, ref_gray, sift, kp_ref, des_ref):
    """Estimate static-background homography mapping ref -> src (or src -> ref via invert)."""
    kp_s, des_s = sift.detectAndCompute(src_gray, None)
    if des_s is None or len(kp_s) < 20:
        return np.eye(3, dtype=np.float64), 99.0
    bf = cv2.BFMatcher(cv2.NORM_L2)
    matches = bf.knnMatch(des_ref, des_s, k=2)
    good = [m for m, n in matches if m.distance < 0.72 * n.distance]
    if len(good) < 16:
        return np.eye(3, dtype=np.float64), 99.0
    pts_ref = np.float32([kp_ref[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
    pts_src = np.float32([kp_s[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)
    H_r2s, mask = cv2.findHomography(pts_ref, pts_src, cv2.RANSAC, 2.5)
    if H_r2s is None or mask is None or mask.sum() < 12:
        return np.eye(3, dtype=np.float64), 99.0
    inl = mask.ravel() > 0
    proj = cv2.perspectiveTransform(pts_ref[inl], H_r2s)
    err = float(np.median(np.linalg.norm(proj - pts_src[inl], axis=2)))
    return H_r2s, err


def main():
    WORK.mkdir(parents=True, exist_ok=True)
    frames_dir = WORK / "frames"; frames_dir.mkdir(exist_ok=True)
    mask_dir   = WORK / "mask";   mask_dir.mkdir(exist_ok=True)

    cap = cv2.VideoCapture(str(VID))
    n_tot = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    W = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    H = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS) or 29.97003

    src_start, N = 45, 240
    if (WORK / "clean_plate_ref.png").exists() and (WORK / "clean_homographies.npz").exists() and len(list(frames_dir.glob("*.jpg"))) >= N:
        clean_ref = cv2.imread(str(WORK / "clean_plate_ref.png"))
        Hs = np.load(WORK / "clean_homographies.npz")["H"]
        imgs = [cv2.imread(str(frames_dir / f"{i:04d}.jpg")) for i in range(N)]
        cap.release()
        print("Reusing existing aligned forest clean plate, homographies, and 240 frames!")
    else:
        # 1. Pick reference frame (src f=160, middle of hero window) & extract SIFT features on static forest perimeter
        ref_idx = 160
        cap.set(cv2.CAP_PROP_POS_FRAMES, ref_idx)
        ok, ref_bgr = cap.read()
        assert ok
        static_mask = np.full((H, W), 255, dtype=np.uint8)
        static_mask[240:980, 180:1150] = 0
        sift = cv2.SIFT_create(nfeatures=2400)
        ref_g = cv2.cvtColor(ref_bgr, cv2.COLOR_BGR2GRAY)
        kp_ref, des_ref = sift.detectAndCompute(ref_g, static_mask)

        samps_warped = []
        errs_bg = []
        for fi in np.linspace(0, n_tot - 1, 36, dtype=int):
            cap.set(cv2.CAP_PROP_POS_FRAMES, int(fi))
            ok, im = cap.read()
            if not ok:
                continue
            im_g = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
            H_r2s, err = compute_homography(im_g, ref_g, sift, kp_ref, des_ref)
            errs_bg.append(err)
            H_s2r = np.linalg.inv(H_r2s)
            warped = cv2.warpPerspective(im, H_s2r, (W, H), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
            samps_warped.append(warped)
        stack = np.stack(samps_warped, axis=0)
        clean_ref = np.percentile(stack, 78, axis=0).astype(np.uint8)
        perim = static_mask > 0
        for ch in range(3):
            delta = float(ref_bgr[:, :, ch][perim].mean()) - float(clean_ref[:, :, ch][perim].mean())
            clean_ref[:, :, ch] = np.clip(clean_ref[:, :, ch].astype(np.float32) + delta, 0, 255).astype(np.uint8)
        cv2.imwrite(str(WORK / "clean_plate_ref.png"), clean_ref)
        cv2.imwrite(str(WORK / "check_clean_forest.jpg"), cv2.resize(clean_ref, (960, 540)), [cv2.IMWRITE_JPEG_QUALITY, 92])
        print(f"Real forest clean plate built from {len(samps_warped)} aligned frames | median H err={np.median(errs_bg):.2f}px")

        cap.set(cv2.CAP_PROP_POS_FRAMES, src_start)
        imgs = []
        Hs = np.zeros((N, 3, 3), dtype=np.float64)
        h_errs = []
        for i in range(N):
            ok, im = cap.read()
            if not ok:
                break
            imgs.append(im)
            cv2.imwrite(str(frames_dir / f"{i:04d}.jpg"), im, [cv2.IMWRITE_JPEG_QUALITY, 95])
            im_g = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
            H_r2s, err = compute_homography(im_g, ref_g, sift, kp_ref, des_ref)
            Hs[i] = H_r2s
            h_errs.append(err)
        cap.release()
        N = len(imgs)
        Hs = Hs[:N]
        np.savez(WORK / "clean_homographies.npz", H=Hs)
        print(f"Hero 240f homographies computed | median err={np.median(h_errs):.2f}px, p90={np.percentile(h_errs, 90):.2f}px")

    # 4. Run MediaPipe PoseLandmarker on all 240 frames
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
                mp_img = mp.Image(image_format=mp.ImageFormat.SRGB, data=cv2.cvtColor(im, cv2.COLOR_BGR2RGB))
                res = landmarker.detect_for_video(mp_img, int(round(i * 1000.0 / fps)))
                if res.segmentation_masks:
                    sm = res.segmentation_masks[0].numpy_view()
                    if sm.ndim == 3:
                        sm = sm[:, :, 0]
                    mp_masks[i] = cv2.resize(sm.astype(np.float32), (W, H), interpolation=cv2.INTER_LINEAR)

    # 5. Fuse MediaPipe Segmentation + Aligned Clean-Plate Difference Matte -> razor-sharp silhouette!
    bboxes = np.zeros((N, 4), dtype=np.float64)
    k_small = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    k_med   = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (11, 11))
    k_close = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (17, 17))
    prev_cx, prev_cy = 390.0, 560.0
    qa_panels = []

    for i, im in enumerate(imgs):
        clean_i = cv2.warpPerspective(clean_ref, Hs[i], (W, H), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
        # Difference against aligned clean plate (person is a crisp dark coat silhouette against bright mist!)
        diff_dark = np.maximum(cv2.cvtColor(clean_i, cv2.COLOR_BGR2GRAY).astype(np.float32) -
                               cv2.cvtColor(im, cv2.COLOR_BGR2GRAY).astype(np.float32), 0.0)
        diff_rgb = np.linalg.norm(im.astype(np.float32) - clean_i.astype(np.float32), axis=2)
        diff = cv2.GaussianBlur(0.65 * diff_dark + 0.35 * diff_rgb, (0, 0), 1.4)

        # Constrain ROI cleanly around moving person corridor (excludes tiny branch wind flutter on far right!)
        roi = np.zeros((H, W), dtype=np.uint8)
        rx0 = max(0, int(prev_cx - 260)); rx1 = min(W, int(prev_cx + 260))
        ry0 = max(120, int(prev_cy - 360)); ry1 = min(H, int(prev_cy + 360))
        roi[ry0:ry1, rx0:rx1] = 1

        diff_bin = ((diff > 24.0) & (roi > 0)).astype(np.uint8)
        diff_bin = cv2.morphologyEx(diff_bin, cv2.MORPH_OPEN, k_small)
        diff_bin = cv2.morphologyEx(diff_bin, cv2.MORPH_CLOSE, k_close)

        if mp_masks[i] is not None:
            mp_bin = ((mp_masks[i] > 0.35) & (roi > 0)).astype(np.uint8)
            # Gate diff_bin to a tight halo around MediaPipe human body -> razor-sharp silhouette with ZERO grass shadow!
            mp_gate = cv2.dilate(mp_bin, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (23, 23)))
            comb = np.maximum(mp_bin, diff_bin & mp_gate)
        else:
            comb = diff_bin

        num_l, labels, stats, cents = cv2.connectedComponentsWithStats(comb, connectivity=8)
        best_l, best_score = -1, -1e9
        for l in range(1, num_l):
            area = stats[l, cv2.CC_STAT_AREA]
            if area < 3500:
                continue
            cx, cy = cents[l]
            score = area - math.hypot(cx - prev_cx, cy - prev_cy) * 140.0
            if score > best_score:
                best_score = score; best_l = l

        if best_l > 0:
            body = (labels == best_l).astype(np.uint8)
            prev_cx, prev_cy = float(cents[best_l, 0]), float(cents[best_l, 1])
        else:
            body = comb

        body = cv2.morphologyEx(body, cv2.MORPH_CLOSE, k_close)
        cnts, _ = cv2.findContours(body, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        solid = np.zeros((H, W), dtype=np.uint8)
        if cnts:
            cv2.drawContours(solid, [max(cnts, key=cv2.contourArea)], -1, 255, thickness=cv2.FILLED)

        dil = cv2.dilate((solid > 0).astype(np.uint8), k_med)
        erode = cv2.erode((solid > 0).astype(np.uint8), k_small)
        soft_diff = np.clip((diff - 14.0) / 20.0, 0.0, 1.0)
        edge_prob = np.maximum(soft_diff, mp_masks[i]) if mp_masks[i] is not None else soft_diff
        matte = np.where(erode > 0, 1.0, np.where(dil > 0, edge_prob, 0.0)).astype(np.float32)
        matte = cv2.GaussianBlur(matte, (0, 0), 1.0)
        matte_u8 = np.clip(matte * 255.0 + 0.5, 0, 255).astype(np.uint8)
        cv2.imwrite(str(mask_dir / f"{i:04d}.png"), matte_u8)

        ys, xs = np.where(matte_u8 > 100)
        if len(ys) > 0:
            bboxes[i] = [xs.min(), ys.min(), xs.max(), ys.max()]
        elif i > 0:
            bboxes[i] = bboxes[i - 1]

        if i in (0, 48, 96, 144, 192, 235):
            vis = im.copy()
            edge = cv2.dilate(cv2.Canny(matte_u8, 60, 160), np.ones((3, 3), np.uint8))
            vis[edge > 0] = (40, 220, 255)
            cv2.putText(vis, f"f={i} h={int(bboxes[i,3]-bboxes[i,1])}px", (25, 70),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.7, (80, 220, 255), 3)
            qa_panels.append(cv2.resize(vis, (640, 360)))

    # 6. Fit smooth 3D metric camera + receding forest trail depth (~4.6m -> 9.8m!)
    f_px = 1650.0
    h_px = np.maximum(bboxes[:, 3] - bboxes[:, 1], 120.0)
    raw_depth = f_px * 1.78 / h_px
    ts = np.arange(N, dtype=np.float64)
    poly = np.poly1d(np.polyfit(ts, raw_depth, deg=2))
    depths = np.clip(poly(ts), 2.8, 18.0)

    # Horizon of the misty woodland clearing sits right around y_hor ~ 615px (gentle eye-level perspective!)
    y_hor = 615.0
    pitch = math.atan2(y_hor - 0.5 * H, f_px)
    cp, sp = math.cos(pitch), math.sin(pitch)
    R0 = np.array([
        [1.0,  0.0,  0.0],
        [0.0, -sp,   cp ],
        [0.0, -cp,  -sp ],
    ], dtype=np.float64)
    foot_y0 = float(bboxes[0, 3])
    ray_foot0 = R0 @ np.array([0.0, (foot_y0 - 0.5 * H) / f_px, 1.0])
    z_cam = float(-ray_foot0[2] * depths[0])
    ro0 = np.array([0.0, 0.0, max(z_cam, 0.85)], dtype=np.float64)

    (WORK / "meta.json").write_text(json.dumps({
        "W": W, "H": H, "N": N, "fps": float(fps), "fps_num": 30000, "fps_den": 1001, "src_start": src_start
    }, indent=2))
    (WORK / "scene.json").write_text(json.dumps({
        "f": f_px, "cam": [ro0.tolist()] * N, "rot": [R0.tolist()] * N
    }))
    np.savez(WORK / "track.npz", bboxes=bboxes, depth=depths)

    if len(qa_panels) == 6:
        sheet = np.vstack([np.hstack(qa_panels[:3]), np.hstack(qa_panels[3:])])
        cv2.imwrite(str(WORK / "check_matte_forest.jpg"), sheet, [cv2.IMWRITE_JPEG_QUALITY, 92])
    print(f"Prepared piece_forest: {N}f @ {fps:.2f}fps | depth {depths[0]:.2f}m -> {depths[-1]:.2f}m | z_cam={z_cam:.2f}m | h_px {h_px[0]:.0f}->{h_px[-1]:.0f}")


if __name__ == "__main__":
    main()
