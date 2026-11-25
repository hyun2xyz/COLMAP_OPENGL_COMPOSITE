import json
import math
import subprocess
from pathlib import Path
import numpy as np
import cv2
from scipy.signal import savgol_filter

ROOT = Path(r"D:\VFX_Work_Claude\_repo\COLMAP_OPENGL_COMPOSITE")
PROJ = ROOT / "examples" / "foreground_cafe_peek"
WORK = PROJ / "work"
FRAMES_DIR = WORK / "frames"

tools = json.loads((ROOT / "tools.local.json").read_text(encoding="utf-8"))
FFMPEG = tools["ffmpeg"]
RAW_VIDEO = PROJ / "footage" / "Pexels_7263095.mp4"
PLATE_MP4 = WORK / "plate.mp4"

# Best 6.00s acting beat: man lifts coffee cup, takes a calm sip while reading newspaper, sets cup down
START_SEC = 7.00
NUM_FRAMES = 150
FPS = 25.0

# Assumed 35mm-equiv lens focal length for the café shot (~35mm medium-wide interior; foreground CG
# uses wide-angle 22mm perspective framing OR matched 24mm with close-up 0.24..0.52m foreground placement!
# Wait: to lock 100% pixel-for-pixel with camera rotation (dx/dy px -> d_yaw/d_pitch rad), we use exact focal_px!)
FOCAL_MM_35 = 24.0
FOCAL_PX = 1920.0 * FOCAL_MM_35 / 36.0  # 1280.0 px (dramatic wide-angle close-up distortion at 0.22..0.50m!)

def srgb_to_linear(img):
    return np.where(img <= 0.04045, img / 12.92, ((img + 0.055) / 1.055) ** 2.4)

def linear_to_srgb(img):
    c = np.clip(img, 0.0, 1.0)
    return np.where(c <= 0.0031308, c * 12.92, 1.055 * (c ** (1.0 / 2.4)) - 0.055)

def extract_plate():
    FRAMES_DIR.mkdir(parents=True, exist_ok=True)
    if not PLATE_MP4.exists():
        print("=== 1. Trimming clean 6.00s café plate (t=7.0..13.0s, 150f @ 25fps) ===", flush=True)
        subprocess.run([
            FFMPEG, "-y", "-v", "error",
            "-ss", f"{START_SEC:.2f}", "-i", str(RAW_VIDEO),
            "-frames:v", str(NUM_FRAMES),
            "-c:v", "libx264", "-crf", "14", "-preset", "medium", "-pix_fmt", "yuv420p",
            "-an", str(PLATE_MP4),
        ], check=True)
    existing = sorted(FRAMES_DIR.glob("frame_*.png"))
    if len(existing) != NUM_FRAMES:
        print("=== Extracting 150 full-res 1920x1080 PNG frames ===", flush=True)
        for p in existing: p.unlink()
        subprocess.run([
            FFMPEG, "-y", "-v", "error", "-i", str(PLATE_MP4),
            "-start_number", "0", str(FRAMES_DIR / "frame_%04d.png"),
        ], check=True)
    existing = sorted(FRAMES_DIR.glob("frame_*.png"))
    print(f"Frames ready: {len(existing)}", flush=True)
    return existing

def track_subtle_camera(frames):
    """
    High-precision sub-pixel camera tracker tailored for subtle micro-handheld shots:
    Tracks static interior architectural features ONLY (brick wall mortar, black steel window frames,
    hanging lamp cord brackets, radiator — explicitly masking out the moving person & outdoor street!).
    Recovers exact frame-to-frame similarity/rotation (tx, ty, roll, scale) via RANSAC + cumulative
    drift anchoring to frame 0 for zero drift accumulation, then converts to Blender camera pose!
    """
    print("=== 2. Tracking exact subtle handheld camera motion (static architecture RANSAC) ===", flush=True)
    g0 = cv2.cvtColor(cv2.imread(str(frames[0])), cv2.COLOR_BGR2GRAY)
    H, W = g0.shape

    # Static interior mask: brick wall upper/left, black steel window mullions, radiator left, ceiling edge
    static_mask = np.zeros((H, W), dtype=np.uint8)
    static_mask[20:420, 40:540] = 255      # upper-left brick & wall
    static_mask[420:850, 20:320] = 255     # left radiator & wall
    static_mask[20:260, 540:1080] = 255    # brick wall above man & lamp brackets
    static_mask[20:900, 1060:1220] = 255   # vertical black steel window frame column (super high contrast!)
    static_mask[20:900, 1340:1460] = 255   # second vertical steel window mullion
    # Explicitly exclude moving person + his table region!
    static_mask[240:980, 520:1060] = 0

    # Detect strong sub-pixel corners on frame 0 and track BOTH directly against frame 0 AND sequentially
    # (since total motion is <20px across 6s, direct tracking from frame 0 has ZERO cumulative drift!!)
    pts0 = cv2.goodFeaturesToTrack(g0, maxCorners=450, qualityLevel=0.015, minDistance=14, mask=static_mask)
    term = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 0.005)
    pts0 = cv2.cornerSubPix(g0, pts0, (7, 7), (-1, -1), term)
    print(f"Detected {len(pts0)} static architectural anchor features on frame 0", flush=True)

    lk_params = dict(winSize=(25, 25), maxLevel=4, criteria=term)

    tx_arr = np.zeros(NUM_FRAMES, dtype=np.float64)
    ty_arr = np.zeros(NUM_FRAMES, dtype=np.float64)
    rot_arr = np.zeros(NUM_FRAMES, dtype=np.float64)
    scl_arr = np.ones(NUM_FRAMES, dtype=np.float64)
    inlier_counts = []

    prev_g = g0
    pts_chain = pts0.copy()
    for i in range(1, NUM_FRAMES):
        gi = cv2.cvtColor(cv2.imread(str(frames[i])), cv2.COLOR_BGR2GRAY)
        # Track from previous frame to get warm-start initial guess, then refine directly against frame 0!
        p_step, st1, _ = cv2.calcOpticalFlowPyrLK(prev_g, gi, pts_chain, None, **lk_params)
        p_direct, st2, _ = cv2.calcOpticalFlowPyrLK(
            g0, gi, pts0, p_step.copy(),
            flags=cv2.OPTFLOW_USE_INITIAL_FLOW, **lk_params
        )
        # Backward check gi -> g0 for ultra-clean outlier rejection (< 0.4px round-trip error!)
        p_back, st3, _ = cv2.calcOpticalFlowPyrLK(gi, g0, p_direct, pts0.copy(), flags=cv2.OPTFLOW_USE_INITIAL_FLOW, **lk_params)
        fb_err = np.linalg.norm((pts0 - p_back).reshape(-1, 2), axis=1)
        ok = (st1.ravel() == 1) & (st2.ravel() == 1) & (st3.ravel() == 1) & (fb_err < 0.45)

        src = pts0[ok].reshape(-1, 2)
        dst = p_direct[ok].reshape(-1, 2)
        M, inl = cv2.estimateAffinePartial2D(src, dst, method=cv2.RANSAC, ransacReprojThreshold=0.6)
        if M is not None and inl is not None and inl.sum() >= 15:
            # Image motion of background point (x, y) around principal point (cx, cy)
            cx, cy = W * 0.5, H * 0.5
            a, b, tx = M[0]
            c, d, ty = M[1]
            scl = math.hypot(a, c)
            ang = math.atan2(c, a)
            # Displacement of image center (cx, cy):
            dcx = (a * cx + b * cy + tx) - cx
            dcy = (c * cx + d * cy + ty) - cy
            tx_arr[i] = dcx
            ty_arr[i] = dcy
            rot_arr[i] = ang
            scl_arr[i] = scl
            inlier_counts.append(int(inl.sum()))
        else:
            tx_arr[i], ty_arr[i], rot_arr[i], scl_arr[i] = tx_arr[i-1], ty_arr[i-1], rot_arr[i-1], scl_arr[i-1]
            inlier_counts.append(0)

        pts_chain = p_direct
        prev_g = gi

    print(f"Static RANSAC tracking complete: min_inliers={min(inlier_counts)}, median_inliers={int(np.median(inlier_counts))}", flush=True)
    print(f"Raw background motion range (1920x1080): dx={tx_arr.max()-tx_arr.min():.2f}px, dy={ty_arr.max()-ty_arr.min():.2f}px, roll={math.degrees(rot_arr.max()-rot_arr.min()):.3f}deg", flush=True)

    # Very light Savitzky-Golay smoothing (window=5, poly=2) to eliminate <0.1px quantization noise while preserving 100% real breathing!
    tx_s = savgol_filter(tx_arr, 5, 2)
    ty_s = savgol_filter(ty_arr, 5, 2)
    rot_s = savgol_filter(rot_arr, 5, 2)
    scl_s = savgol_filter(scl_arr, 5, 2)

    # Convert exact pixel image shift of background (dcx, dcy) into physical camera motion in Blender space (+X right, +Y forward, +Z up):
    # When camera yaws RIGHT (+d_yaw around -Z -> negative Z rot in Blender), background shifts LEFT (-dcx):
    #   d_yaw_bl = +dcx / FOCAL_PX  (rotation around Blender +Z!)
    # When camera pitches UP (+d_pitch around +X in Blender), background shifts DOWN (+dcy):
    #   d_pitch_bl = +dcy / FOCAL_PX (rotation around Blender +X!)
    # When camera rolls CCW (+d_roll around +Y in Blender), background rotates CW (-ang in image Y-down -> +ang around +Z_cv -> +ang around +Y_bl):
    #   Wait: in OpenCV image (X right, Y down), CW image rotation has M = [[cos, -sin],[sin, cos]] with ang > 0 (since y'=x*sin + y*cos moves (1,0)->(1,+sin) DOWN = CW).
    #   Camera rolling CCW around forward (+Y_bl) makes background rotate CW (ang > 0) -> d_roll_bl = -ang (Wait: right-hand rule around +Y: thumb forward, fingers curl from +Z toward +X = CW! So camera rolling CW makes background move CCW (ang < 0), camera rolling CCW (-roll_y) makes background move CW (ang > 0) -> rot_y_bl = -ang!).
    #
    # PLUS subtle real handheld translational breathing (80% rotational pan/tilt matching background pixel-for-pixel,
    # 20% gentle physical handheld translation ~1.2..2.0 cm giving silky subtle real depth parallax between close foreground objects & cafe background!)
    frames_out = []
    for i in range(NUM_FRAMES):
        # Exact background-locked angular drift (radians)
        # Split background shift: 82% camera rotation + 18% subtle translational sway at background depth D_bg=3.5m
        # Wait: for a background point at D_bg = 3.5m, a camera translation dx_cam shifts image by -FOCAL_PX * (dx_cam / D_bg),
        # while yaw rotation rot_z shifts image by +FOCAL_PX * rot_z!
        # So total background image shift = FOCAL_PX * (rot_z - dx_cam / D_bg) == tx_s[i] EXACTLY!
        # That guarantees ZERO slipping against the background while giving physically authentic close-up parallax!
        D_bg = 3.5
        dx_cam = -0.22 * (tx_s[i] / FOCAL_PX) * D_bg  # ~0.01m subtle lateral breathing!
        dz_cam = +0.22 * (ty_s[i] / FOCAL_PX) * D_bg  # ~0.01m subtle vertical breathing!
        dy_cam = (scl_s[i] - 1.0) * D_bg * 0.6        # ~0.005m subtle forward breathing!

        # Exact compensating rotation so background features stay 100.0% pixel-locked!
        rot_z_bl = (tx_s[i] / FOCAL_PX) + (dx_cam / D_bg)
        rot_x_bl = (ty_s[i] / FOCAL_PX) - (dz_cam / D_bg)
        rot_y_bl = -rot_s[i]

        frames_out.append({
            "frame": i,
            "tx_px": float(tx_s[i]),
            "ty_px": float(ty_s[i]),
            "roll_rad": float(rot_s[i]),
            "pos_bl_m": [float(dx_cam), float(dy_cam), float(dz_cam)],
            "rot_xyz_bl": [float(rot_x_bl), float(rot_y_bl), float(rot_z_bl)],
        })

    track = {
        "width": W,
        "height": H,
        "fps": FPS,
        "num_frames": NUM_FRAMES,
        "focal_mm_35": FOCAL_MM_35,
        "focal_px": FOCAL_PX,
        "max_bg_shift_px": [float(tx_s.max() - tx_s.min()), float(ty_s.max() - ty_s.min())],
        "max_cam_pos_cm": [
            float((max(f["pos_bl_m"][0] for f in frames_out) - min(f["pos_bl_m"][0] for f in frames_out)) * 100.0),
            float((max(f["pos_bl_m"][1] for f in frames_out) - min(f["pos_bl_m"][1] for f in frames_out)) * 100.0),
            float((max(f["pos_bl_m"][2] for f in frames_out) - min(f["pos_bl_m"][2] for f in frames_out)) * 100.0),
        ],
        "frames": frames_out,
    }
    (WORK / "camera_track.json").write_text(json.dumps(track, indent=2), encoding="utf-8")
    print(f"Saved camera_track.json: subtle camera sway = {np.round(track['max_cam_pos_cm'], 2)} cm (100% locked to plate!)", flush=True)
    return track

def build_cafe_hdri(frames, track):
    """
    Builds a 2048x1024 linear float32 Radiance 360° HDRI physically matched to Pexels 7263095:
    - Exact forward (+Y) café projection with inverse-tonemapped HDR lobes:
        1) Huge cool overcast street windows on the RIGHT (+X / +Y, ~6200K bright daylight!)
        2) Two warm glowing Edison filament pendant bulbs in upper center-left (~2600K warm amber!)
        3) Warm textured brick wall on center-left + dark green ZZ plant & radiator on far left
    - Extended 360° room shell:
        - Huge bright cool-daylight window continuing around the right side (+X)!
        - Dark cozy shadowed brick/wood café interior behind camera (-Y) and far left (-X)
        - Warm wood tabletop bounce directly below camera (-Z)
    """
    print("=== 3. Building scene-matched 360° float32 Café HDRI ===", flush=True)
    sample_idxs = np.linspace(0, len(frames) - 1, 15, dtype=int)
    stack = []
    for idx in sample_idxs:
        im = cv2.imread(str(frames[idx]))[:, :, ::-1].astype(np.float32) / 255.0
        stack.append(srgb_to_linear(im))
    med_lin = np.median(np.stack(stack, axis=0), axis=0).astype(np.float32)

    # Measure real café linear colors from plate regions!
    brick_col = np.median(med_lin[120:420, 320:540], axis=(0, 1))
    window_col = np.median(med_lin[100:500, 1450:1880], axis=(0, 1))
    left_shadow = np.median(med_lin[500:850, 20:220], axis=(0, 1))
    table_col = np.median(med_lin[980:1070, 500:1150], axis=(0, 1))
    print(f"Measured linear café palette: brick={np.round(brick_col,3)} window={np.round(window_col,3)} L_shadow={np.round(left_shadow,3)} table={np.round(table_col,3)}", flush=True)

    # Inverse-tonemap the real highlights in the plate:
    lum = 0.2126 * med_lin[..., 0] + 0.7152 * med_lin[..., 1] + 0.0722 * med_lin[..., 2]
    y_norm = np.linspace(0, 1, 1080, dtype=np.float32)[:, None]
    x_norm = np.linspace(0, 1, 1920, dtype=np.float32)[None, :]

    # 1) Warm Edison bulbs hanging at x=0.33..0.44, y=0.00..0.30
    is_edison = (x_norm > 0.30) & (x_norm < 0.45) & (y_norm < 0.32)
    edison_excess = np.clip((lum - 0.35) / 0.50, 0.0, 1.0) * is_edison.astype(np.float32)
    edison_hdr = (edison_excess ** 1.8)[..., None] * np.array([28.0, 14.5, 4.2], dtype=np.float32)

    # 2) Cool bright overcast street windows on right (x > 0.56)
    is_win = (x_norm > 0.55)
    win_excess = np.clip((lum - 0.28) / 0.55, 0.0, 1.0) * is_win.astype(np.float32)
    win_hdr = (win_excess ** 1.6)[..., None] * np.array([9.5, 10.5, 11.8], dtype=np.float32)

    plate_hdr = med_lin * 1.15 + edison_hdr + win_hdr

    # Build 2048x1024 equirectangular map in Blender World convention (+Y forward at u=0.25!)
    W, H = 2048, 1024
    jj, ii = np.mgrid[0:H, 0:W]
    u = (ii + 0.5) / W
    v = (jj + 0.5) / H
    lon = (0.5 - u) * 2.0 * np.pi
    lat = (0.5 - v) * np.pi
    dx = np.cos(lat) * np.cos(lon)
    dy = np.cos(lat) * np.sin(lon)
    dz = np.sin(lat)

    # Base 360° café enclosure:
    # Right side (+X) is the continuation of the huge cool-daylight café window wall!
    # Left & Behind (-X, -Y) is cozy shadowed brick café interior with subtle warm overhead pendant bounce!
    # Bottom (-Z) is warm oak tabletop & dark café floor!
    right_win_mask = np.clip((dx - 0.15) / 0.65, 0.0, 1.0) * np.clip(1.0 - np.abs(dz - 0.12) * 1.4, 0.0, 1.0)
    right_win_mask = right_win_mask * right_win_mask * (3.0 - 2.0 * right_win_mask)

    up_w = np.clip(dz * 1.3, 0.0, 1.0)
    dn_w = np.clip(-dz * 1.3, 0.0, 1.0)
    mid_w = np.clip(1.0 - up_w - dn_w, 0.0, 1.0)

    # Cozy dark-warm café interior base (rear & left shadows)
    cafe_dark = (
        up_w[..., None] * np.array([0.075, 0.062, 0.048], dtype=np.float32)
        + mid_w[..., None] * (brick_col * 0.45 + np.array([0.025, 0.018, 0.014], dtype=np.float32))
        + dn_w[..., None] * (table_col * 0.55)
    )
    # Keep behind-camera (-Y) darker so camera-facing foreground sides have rich sculptural contrast
    rear_dim = 0.35 + 0.65 * np.clip((dy + 0.45) / 1.45, 0.0, 1.0)
    hdri = cafe_dark * rear_dim[..., None]

    # Add the large continuous cool-daylight right café window wall (dx > 0.25, wrapping right beside the foreground table!)
    side_window_col = np.array([3.6, 4.1, 4.7], dtype=np.float32)
    hdri += right_win_mask[..., None] * side_window_col

    # Project actual HDR-expanded café plate onto forward +Y cone!
    fx = float(track["focal_px"])
    cx, cy = 960.0, 540.0
    in_front = dy > 0.05
    px = cx + fx * (dx / np.maximum(dy, 1e-4))
    py = cy - fx * (dz / np.maximum(dy, 1e-4))
    sampled = cv2.remap(
        plate_hdr, px.astype(np.float32), py.astype(np.float32),
        cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE
    )
    margin_x = np.minimum(px / 130.0, (1919.0 - px) / 130.0)
    margin_y = np.minimum(py / 95.0, (1079.0 - py) / 95.0)
    blend = np.clip(np.minimum(margin_x, margin_y), 0.0, 1.0) * in_front.astype(np.float32)
    blend = blend * blend * (3.0 - 2.0 * blend)
    hdri = hdri * (1.0 - blend[..., None]) + sampled * blend[..., None]

    # Add a soft warm Edison pendant overhead above the foreground table (matches the café's Edison fixtures!)
    fg_edison = np.exp(-((dx - 0.10) ** 2 / 0.08 + (dy - 0.05) ** 2 / 0.10 + (dz - 0.88) ** 2 / 0.04))
    hdri += fg_edison[..., None] * np.array([7.5, 4.2, 1.4], dtype=np.float32)

    # Save float32 Radiance .hdr (OpenCV expects BGR!)
    hdr_path = WORK / "scene_matched_cafe_2k.hdr"
    ok = cv2.imwrite(str(hdr_path), hdri[:, :, ::-1].astype(np.float32))
    assert ok, "Failed to save cafe HDRI"

    tm = hdri / (1.0 + hdri)
    prev = (linear_to_srgb(tm) * 255.0).clip(0, 255).astype(np.uint8)
    cv2.imwrite(str(WORK / "scene_hdri_preview.jpg"), prev[:, :, ::-1], [int(cv2.IMWRITE_JPEG_QUALITY), 92])
    print(f"Saved café HDRI: {hdr_path} (max radiance={hdri.max():.2f}, mean={hdri.mean():.3f})", flush=True)

if __name__ == "__main__":
    frames = extract_plate()
    track = track_subtle_camera(frames)
    build_cafe_hdri(frames, track)
    print("CAFE TRACK & HDRI DONE!", flush=True)
