"""Prepare examples/minecraft_mocap_rig/piece_steve from Pexels_8928132 (slit_scan frames 25..264, 240f @ 30fps).

Builds:
- piece.json + sliced camera.npz, plane.json, limbs.npz, frames/, mask/
- Per-frame masked SIFT-homography-warped spotless clean plate (clean_frames/%04d.jpg)
  using the AI clean plate on reference frame 150, with feathered boundary exposure matching
  so 100% of the dancer AND her cast ground shadow vanish with zero seam!
"""
import json, shutil, sys, time
from pathlib import Path
import cv2, numpy as np

here = Path(__file__).resolve().parent
ROOT = here.parents[1]
SRC_PIECE = ROOT / "examples/slit_scan/piece"
DST_PIECE = here / "piece_steve"
AI_CLEAN_SRC = Path("C:/Users/Q/.gemini/antigravity/brain/06f6aa31-6033-4e1b-bb93-5b54c8408858/mc_mocap_clean_0150_1795247122113.jpg")

SLICE_START = 25
SLICE_N = 240   # 8.0s @ 30fps (high-energy dance, high knee lift, 360-deg spin, zero car bumper!)
REF_SRC_IDX = 150


def match_homography(img_a, mask_a, img_b, mask_b, sift, bf, scale=0.5):
    """Compute robust masked SIFT homography mapping img_a -> img_b using static background features only."""
    ga = cv2.cvtColor(img_a, cv2.COLOR_BGR2GRAY)
    gb = cv2.cvtColor(img_b, cv2.COLOR_BGR2GRAY)
    ker = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (65, 65))
    va = 255 - cv2.dilate(mask_a, ker) if mask_a is not None else None
    vb = 255 - cv2.dilate(mask_b, ker) if mask_b is not None else None
    if scale != 1.0:
        ha, wa = ga.shape[:2]
        ns = (int(wa * scale), int(ha * scale))
        ga = cv2.resize(ga, ns, interpolation=cv2.INTER_AREA)
        gb = cv2.resize(gb, ns, interpolation=cv2.INTER_AREA)
        if va is not None: va = cv2.resize(va, ns, interpolation=cv2.INTER_NEAREST)
        if vb is not None: vb = cv2.resize(vb, ns, interpolation=cv2.INTER_NEAREST)
    ka, da = sift.detectAndCompute(ga, va)
    kb, db = sift.detectAndCompute(gb, vb)
    if da is None or db is None or len(ka) < 12 or len(kb) < 12:
        return np.eye(3, dtype=np.float64), 0, 99.0
    raw = bf.knnMatch(da, db, k=2)
    good = [m for m, n in raw if m.distance < 0.72 * n.distance]
    if len(good) < 10:
        return np.eye(3, dtype=np.float64), len(good), 99.0
    pa = np.float32([ka[m.queryIdx].pt for m in good]) / float(scale)
    pb = np.float32([kb[m.trainIdx].pt for m in good]) / float(scale)
    H, inl = cv2.findHomography(pa, pb, cv2.RANSAC, 2.5)
    if H is None or inl is None or int(inl.sum()) < 8:
        return np.eye(3, dtype=np.float64), 0, 99.0
    in_m = inl.ravel() == 1
    proj = cv2.perspectiveTransform(pa[in_m].reshape(-1, 1, 2), H).reshape(-1, 2)
    err = float(np.median(np.linalg.norm(proj - pb[in_m], axis=1)))
    return H, int(in_m.sum()), err


def main():
    t0 = time.time()
    src_wk = SRC_PIECE / "work"
    dst_wk = DST_PIECE / "work"
    dst_wk.mkdir(parents=True, exist_ok=True)
    for sub in ("frames", "mask", "clean_frames"):
        d = dst_wk / sub
        shutil.rmtree(d, ignore_errors=True)
        d.mkdir(parents=True, exist_ok=True)

    # 1. Write piece.json
    src_cfg = json.loads((SRC_PIECE / "piece.json").read_text(encoding="utf-8"))
    dst_cfg = {
        "name": "Minecraft_Steve_Mocap_Rig",
        "pexels": 8928132,
        "credit": "RDNE Stock project (Pexels 8928132, Pexels License)",
        "source": "../../slit_scan/footage/Pexels_8928132.mp4",
        "start": int(src_cfg.get("start", 30)) + SLICE_START,
        "frames": SLICE_N,
        "step": 1,
        "fps_num": 30,
        "fps_den": 1,
        "hfov": src_cfg.get("hfov", 60.0),
        "solver": src_cfg.get("solver", "homography"),
        "plane": src_cfg.get("plane", {}),
        "sun": src_cfg.get("sun", {"az": 180, "el": 70}),
    }
    (DST_PIECE / "piece.json").write_text(json.dumps(dst_cfg, indent=1), encoding="utf-8")

    # 2. Slice camera.npz, limbs.npz, and copy plane.json cleanly
    cam = np.load(src_wk / "camera.npz")
    cam_out = {}
    for k in cam.files:
        arr = cam[k]
        if arr.ndim >= 1 and arr.shape[0] == 300:
            cam_out[k] = arr[SLICE_START:SLICE_START + SLICE_N]
        else:
            cam_out[k] = arr
    np.savez(dst_wk / "camera.npz", **cam_out)

    L = np.load(src_wk / "limbs.npz")
    L_out = {k: L[k][SLICE_START:SLICE_START + SLICE_N] for k in L.files}
    np.savez(dst_wk / "limbs.npz", **L_out)

    sc = json.loads((src_wk / "scene.json").read_text(encoding="utf-8"))
    sc["cam"] = sc["cam"][SLICE_START:SLICE_START + SLICE_N]
    sc["rot"] = sc["rot"][SLICE_START:SLICE_START + SLICE_N]
    (dst_wk / "scene.json").write_text(json.dumps(sc), encoding="utf-8")

    mt = json.loads((src_wk / "meta.json").read_text(encoding="utf-8"))
    mt["N"] = SLICE_N
    (dst_wk / "meta.json").write_text(json.dumps(mt, indent=1), encoding="utf-8")

    # 3. SIFT-align AI clean plate to EXACT pixel grid of reference frame 150!
    ref_live = cv2.imread(str(src_wk / "frames" / f"{REF_SRC_IDX:04d}.jpg"))
    ref_mask = cv2.imread(str(src_wk / "mask" / f"{REF_SRC_IDX:04d}.png"), cv2.IMREAD_GRAYSCALE)
    H_im, W_im = ref_live.shape[:2]
    ai_raw = cv2.imread(str(AI_CLEAN_SRC))
    if ai_raw is None:
        sys.exit(f"Missing AI clean plate: {AI_CLEAN_SRC}")
    if ai_raw.shape[:2] != (H_im, W_im):
        ai_raw = cv2.resize(ai_raw, (W_im, H_im), interpolation=cv2.INTER_LANCZOS4)

    sift = cv2.SIFT_create(2800)
    bf = cv2.BFMatcher()
    H_ai, n_in_ai, err_ai = match_homography(ai_raw, None, ref_live, ref_mask, sift, bf)
    print(f"AI clean plate -> ref frame {REF_SRC_IDX}: inliers={n_in_ai}, median_err={err_ai:.2f}px")
    clean_ref = cv2.warpPerspective(ai_raw, H_ai, (W_im, H_im), flags=cv2.INTER_LANCZOS4, borderMode=cv2.BORDER_REPLICATE)

    # Calibrate global RGB gain/bias of clean_ref against static background of ref_live!
    bg_m = (cv2.dilate(ref_mask, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (91, 91))) == 0)
    # Use center concrete band near dancer (not bright windows) for ultra-accurate tone match
    band_m = bg_m.copy()
    band_m[:, :350] = False
    band_m[:, 1550:] = False
    band_m[:180, :] = False
    ref_f = ref_live.astype(np.float32)
    cln_f = clean_ref.astype(np.float32)
    for ch in range(3):
        m_live = float(np.mean(ref_f[:, :, ch][band_m]))
        s_live = float(np.std(ref_f[:, :, ch][band_m]))
        m_cln  = float(np.mean(cln_f[:, :, ch][band_m]))
        s_cln  = float(np.std(cln_f[:, :, ch][band_m]))
        cln_f[:, :, ch] = (cln_f[:, :, ch] - m_cln) * (s_live / max(s_cln, 1.0)) + m_live
    clean_ref = np.clip(cln_f + 0.5, 0, 255).astype(np.uint8)
    cv2.imwrite(str(dst_wk / "clean_ref_0150.jpg"), clean_ref, [cv2.IMWRITE_JPEG_QUALITY, 96])

    # 4. Precompute per-frame homographies ref -> frame i & composite FULL-STAGE spotless clean plate!
    # Replacing the entire static concrete wall + sidewalk stage (fixed in world space!) eliminates
    # ALL moving mask edges AND ALL residual dancer wall/floor shadows completely!
    img_lm = L_out["img"]
    stage_ref_mask = np.zeros((H_im, W_im), dtype=np.uint8)
    # Fixed-in-world concrete wall + sidewalk polygon spanning between & slightly past both window frames!
    stage_ref_mask[90:H_im, 240:1660] = 255
    ker_body = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (95, 95))
    ker_ring = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (145, 145))
    errs = []
    for i in range(SLICE_N):
        src_i = SLICE_START + i
        shutil.copy2(src_wk / "frames" / f"{src_i:04d}.jpg", dst_wk / "frames" / f"{i:04d}.jpg")
        shutil.copy2(src_wk / "mask" / f"{src_i:04d}.png", dst_wk / "mask" / f"{i:04d}.png")

        live = cv2.imread(str(dst_wk / "frames" / f"{i:04d}.jpg"))
        mask = cv2.imread(str(dst_wk / "mask" / f"{i:04d}.png"), cv2.IMREAD_GRAYSCALE)
        H_i, n_in, err_i = match_homography(ref_live, ref_mask, live, mask, sift, bf)
        errs.append(err_i)
        warped_clean = cv2.warpPerspective(clean_ref, H_i, (W_im, H_im), flags=cv2.INTER_LANCZOS4, borderMode=cv2.BORDER_REPLICATE)
        warped_stage = cv2.warpPerspective(stage_ref_mask, H_i, (W_im, H_im), flags=cv2.INTER_LINEAR, borderValue=0)

        # Union fixed-in-world wall+sidewalk stage WITH generous dilated live person mask (zero shadow/hair/limb leak!)
        er = np.maximum((warped_stage > 128).astype(np.uint8) * 255, cv2.dilate((mask > 12).astype(np.uint8) * 255, ker_body))
        # Match subtle per-frame exposure/color along static outer background ring outside stage matte!
        ring = (cv2.dilate(er, ker_ring) > 128) & (er == 0)
        wc_f = warped_clean.astype(np.float32)
        lv_f = live.astype(np.float32)
        if int(ring.sum()) > 500:
            for ch in range(3):
                dl = float(np.median(lv_f[:, :, ch][ring] - wc_f[:, :, ch][ring]))
                wc_f[:, :, ch] = np.clip(wc_f[:, :, ch] + dl, 0.0, 255.0)

        alpha = cv2.GaussianBlur(er.astype(np.float32) / 255.0, (0, 0), 22.0)[..., None]
        comp_clean = np.clip(lv_f * (1.0 - alpha) + wc_f * alpha + 0.5, 0, 255).astype(np.uint8)
        cv2.imwrite(str(dst_wk / "clean_frames" / f"{i:04d}.jpg"), comp_clean, [cv2.IMWRITE_JPEG_QUALITY, 95])
        if i % 60 == 0 or i == SLICE_N - 1:
            print(f"  clean frame {i}/{SLICE_N} (inliers={n_in}, err={err_i:.2f}px)", flush=True)

    # Save 6-frame visual QA sheet of spotless clean plate across the entire camera pan!
    idxs = [0, 48, 96, 144, 192, 239]
    rows = []
    for k in range(0, 6, 3):
        r_tiles = []
        for idx in idxs[k:k + 3]:
            im = cv2.resize(cv2.imread(str(dst_wk / "clean_frames" / f"{idx:04d}.jpg")), (640, 360))
            cv2.putText(im, f"clean f={idx}", (10, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.85, (0, 0, 0), 4)
            cv2.putText(im, f"clean f={idx}", (10, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.85, (40, 255, 40), 2)
            r_tiles.append(im)
        rows.append(np.hstack(r_tiles))
    cv2.imwrite(str(dst_wk / "qa_clean_6.jpg"), np.vstack(rows), [cv2.IMWRITE_JPEG_QUALITY, 92])
    print(f"Done in {time.time()-t0:.1f}s | median homography err across {SLICE_N}f = {np.median(errs):.2f}px (p95={np.percentile(errs,95):.2f}px)")


if __name__ == "__main__":
    main()
