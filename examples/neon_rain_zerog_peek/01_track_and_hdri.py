import json
import math
import subprocess
from pathlib import Path
import numpy as np
import cv2
from scipy.signal import savgol_filter

ROOT = Path(r"D:\VFX_Work_Claude\_repo\COLMAP_OPENGL_COMPOSITE")
PROJ = ROOT / "examples" / "neon_rain_zerog_peek"
WORK = PROJ / "work"
FRAMES_DIR = WORK / "frames"

tools = json.loads((ROOT / "tools.local.json").read_text(encoding="utf-8"))
FFMPEG = tools["ffmpeg"]
RAW_VIDEO = PROJ / "footage" / "Pexels_7062407.mp4"
PLATE_MP4 = WORK / "plate.mp4"

# Pick frames 60..179 (120 frames @ 24000/1001 = 5.005s):
# Silky slow cinema drift under crimson Chinatown neon + dramatic blue-white headlight bloom at the end!
START_FRAME = 60
NUM_FRAMES = 120
FPS = 24000.0 / 1001.0

# Wide-normal cinema prime (24mm eq on 36mm sensor -> dramatic close-up wide-angle foreground distortion at 0.22..0.55m!)
FOCAL_MM_35 = 24.0
FOCAL_PX = 1920.0 * FOCAL_MM_35 / 36.0  # 1280.0 px

def srgb_to_linear(img):
    return np.where(img <= 0.04045, img / 12.92, ((img + 0.055) / 1.055) ** 2.4)

def linear_to_srgb(img):
    c = np.clip(img, 0.0, 1.0)
    return np.where(c <= 0.0031308, c * 12.92, 1.055 * (c ** (1.0 / 2.4)) - 0.055)

def extract_plate():
    FRAMES_DIR.mkdir(parents=True, exist_ok=True)
    if not PLATE_MP4.exists():
        print(f"=== 1. Trimming 5.00s neon rain cinema plate (frames {START_FRAME}..{START_FRAME+NUM_FRAMES-1}) ===", flush=True)
        subprocess.run([
            FFMPEG, "-y", "-v", "error",
            "-i", str(RAW_VIDEO),
            "-vf", f"select='between(n\\,{START_FRAME}\\,{START_FRAME+NUM_FRAMES-1})',setpts=PTS-STARTPTS",
            "-vsync", "0",
            "-c:v", "libx264", "-crf", "14", "-preset", "medium", "-pix_fmt", "yuv420p",
            "-an", str(PLATE_MP4),
        ], check=True)
    existing = sorted(FRAMES_DIR.glob("frame_*.png"))
    if len(existing) != NUM_FRAMES:
        print(f"=== Extracting {NUM_FRAMES} full-res 1920x1080 PNG frames ===", flush=True)
        for p in existing: p.unlink()
        subprocess.run([
            FFMPEG, "-y", "-v", "error", "-i", str(PLATE_MP4),
            "-start_number", "0", str(FRAMES_DIR / "frame_%04d.png"),
        ], check=True)
    existing = sorted(FRAMES_DIR.glob("frame_*.png"))
    print(f"Frames ready: {len(existing)}", flush=True)
    return existing

def track_cinema_camera_and_lighting(frames):
    """
    Tracks the slow silky cinema drift of Pexels 7062407 using sequential RANSAC affine optical flow
    on foreground-to-midground rigid features (the wet cobalt metallic lamppost + neon sign frame + protagonist silhouette anchor),
    producing ultra-smooth subtle camera rotation & gentle parallax (~2.5cm) locked 100% to the cinema drift!
    Also measures per-frame dynamic lighting intensities:
      - left crimson Chinatown neon brightness
      - right blue-white car headlight bloom (flares dramatically in last ~35 frames!)
    """
    print("=== 2. Tracking slow cinema arc & measuring dynamic neon/headlight curves ===", flush=True)
    H, W = 1080, 1920
    term = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 0.005)
    lk_params = dict(winSize=(27, 27), maxLevel=4, criteria=term)

    # Track sequential rigid motion on lamppost + neon sign + street fixtures
    step_dx = np.zeros(NUM_FRAMES, dtype=np.float64)
    step_dy = np.zeros(NUM_FRAMES, dtype=np.float64)
    step_rot = np.zeros(NUM_FRAMES, dtype=np.float64)

    # Dynamic lighting meters per frame
    neon_red_curve = np.zeros(NUM_FRAMES, dtype=np.float64)
    right_flare_curve = np.zeros(NUM_FRAMES, dtype=np.float64)

    prev_bgr = cv2.imread(str(frames[0]))
    prev_g = cv2.cvtColor(prev_bgr, cv2.COLOR_BGR2GRAY)

    def measure_lights(bgr, i):
        lin = srgb_to_linear(bgr[:, :, ::-1].astype(np.float32) / 255.0)
        # Left neon region (x=0..45%, y=0..80%)
        neon_patch = lin[0:860, 0:860]
        neon_red_curve[i] = float(np.percentile(neon_patch[:, :, 0], 96))
        # Right car headlight flare region (x=72..100%, y=25..95%)
        flare_patch = lin[260:1020, 1380:1920]
        right_flare_curve[i] = float(np.mean(np.clip(flare_patch - 0.15, 0.0, 1.0)))

    measure_lights(prev_bgr, 0)

    # Mask focused on rigid structures (neon sign + metallic lamppost right), excluding top blank black sky
    track_mask = np.zeros((H, W), dtype=np.uint8)
    track_mask[60:920, 140:780] = 255      # Chinatown neon sign & red structure
    track_mask[80:980, 1320:1760] = 255    # Cobalt wet metallic lamppost

    for i in range(1, NUM_FRAMES):
        cur_bgr = cv2.imread(str(frames[i]))
        cur_g = cv2.cvtColor(cur_bgr, cv2.COLOR_BGR2GRAY)
        measure_lights(cur_bgr, i)

        pts = cv2.goodFeaturesToTrack(prev_g, maxCorners=300, qualityLevel=0.015, minDistance=14, mask=track_mask)
        if pts is not None and len(pts) >= 20:
            p1, st1, _ = cv2.calcOpticalFlowPyrLK(prev_g, cur_g, pts, None, **lk_params)
            p0_b, st2, _ = cv2.calcOpticalFlowPyrLK(cur_g, prev_g, p1, pts.copy(), flags=cv2.OPTFLOW_USE_INITIAL_FLOW, **lk_params)
            fb = np.linalg.norm((pts - p0_b).reshape(-1, 2), axis=1)
            ok = (st1.ravel() == 1) & (st2.ravel() == 1) & (fb < 0.6)
            if ok.sum() >= 12:
                M, inl = cv2.estimateAffinePartial2D(pts[ok].reshape(-1, 2), p1[ok].reshape(-1, 2), method=cv2.RANSAC, ransacReprojThreshold=0.8)
                if M is not None and inl is not None and inl.sum() >= 10:
                    cx, cy = W * 0.5, H * 0.5
                    a, b, tx = M[0]
                    c, d, ty = M[1]
                    step_dx[i] = (a * cx + b * cy + tx) - cx
                    step_dy[i] = (c * cx + d * cy + ty) - cy
                    step_rot[i] = math.atan2(c, a)
        prev_g = cur_g

    cum_dx = np.cumsum(step_dx)
    cum_dy = np.cumsum(step_dy)
    cum_rot = np.cumsum(step_rot)

    # Smooth gently to preserve silky cinema dolly motion with zero rain-noise jitter!
    win = 11
    dx_s = savgol_filter(cum_dx, win, 2)
    dy_s = savgol_filter(cum_dy, win, 2)
    rot_s = savgol_filter(cum_rot, win, 2)

    # Normalize right headlight flare curve (0.0 baseline -> 1.0 peak flare!)
    rf_min = float(np.percentile(right_flare_curve[:50], 50))
    rf_max = max(float(right_flare_curve.max()), rf_min + 0.01)
    rf_norm = np.clip((right_flare_curve - rf_min) / (rf_max - rf_min), 0.0, 1.0)
    rf_norm = savgol_filter(rf_norm, 9, 2).clip(0.0, 1.2)

    print(f"Tracked silky cinema drift (1920w): dx={dx_s.max()-dx_s.min():.1f}px, dy={dy_s.max()-dy_s.min():.1f}px, roll={math.degrees(rot_s.max()-rot_s.min()):.2f}deg", flush=True)
    print(f"Dynamic right headlight flare curve: f0={rf_norm[0]:.2f}, f60={rf_norm[60]:.2f}, f95={rf_norm[95]:.2f}, f119={rf_norm[-1]:.2f}", flush=True)

    # Convert to subtle, physically believable camera motion in Blender (+X right, +Y forward, +Z up):
    # Keep foreground motion VERY subtle & cinematic (never sliding awkwardly!):
    # Subtle ~2.2cm gentle lateral arc + exact proportional slow yaw/pitch matching the Steadicam arc!
    frames_out = []
    # Clamp gentle subtle translational sway to max ~2.5cm so extreme-foreground objects stay cleanly framing him!
    max_dx_abs = max(float(np.abs(dx_s).max()), 1.0)
    for i in range(NUM_FRAMES):
        # Gentle subtle Steadicam parallax (only ~2.0cm lateral, ~0.8cm vertical over 5 seconds — silky & grounded!)
        u_t = i / float(NUM_FRAMES - 1)
        dx_cam = -0.022 * (dx_s[i] / max_dx_abs)
        dy_cam = 0.014 * u_t  # whisper-subtle 1.4cm slow cinematic push-in!
        dz_cam = 0.006 * (dy_s[i] / max(float(np.abs(dy_s).max()), 1.0))

        # Subtle matching angular drift (damped gently for ultra-close objects so framing stays dead-on!)
        rot_z_bl = (dx_s[i] / FOCAL_PX) * 0.25
        rot_x_bl = (dy_s[i] / FOCAL_PX) * 0.25
        rot_y_bl = -rot_s[i] * 0.65

        frames_out.append({
            "frame": i,
            "pos_bl_m": [float(dx_cam), float(dy_cam), float(dz_cam)],
            "rot_xyz_bl": [float(rot_x_bl), float(rot_y_bl), float(rot_z_bl)],
            "right_flare": float(rf_norm[i]),
            "neon_red": float(neon_red_curve[i]),
        })

    track = {
        "width": W,
        "height": H,
        "fps": FPS,
        "num_frames": NUM_FRAMES,
        "focal_mm_35": FOCAL_MM_35,
        "focal_px": FOCAL_PX,
        "frames": frames_out,
    }
    (WORK / "camera_track.json").write_text(json.dumps(track, indent=2), encoding="utf-8")
    print("Saved camera_track.json!", flush=True)
    return track

def build_neon_rain_hdri(frames, track):
    """
    Builds a breathtaking 2048x1024 linear float32 360° Neon Rain Night HDRI matched directly to Pexels 7062407:
    - Blazing crimson-red & hot-white Chinatown neon sign HDR lobe on left-forward (-X, +Y)
    - Warm amber/gold streetlamp bokeh pinpoints in background night
    - Deep cobalt-blue night sky & wet blue metallic streetlamp reflections on right (+X, +Y)
    - Rain-slicked dark asphalt reflections below (-Z) catching red & blue neon pools
    - Moody near-black rainy alley behind camera (-Y) for ultra-high cinema contrast!
    """
    print("=== 3. Building scene-matched 360° float32 Neon Rain Night HDRI ===", flush=True)
    sample_idxs = np.linspace(0, len(frames) - 1, 12, dtype=int)
    stack = [srgb_to_linear(cv2.imread(str(frames[k]))[:, :, ::-1].astype(np.float32) / 255.0) for k in sample_idxs]
    med_lin = np.median(np.stack(stack, axis=0), axis=0).astype(np.float32)

    # Expand real neon highlights physically into HDR radiance!
    r, g, b = med_lin[..., 0], med_lin[..., 1], med_lin[..., 2]
    lum = 0.2126 * r + 0.7152 * g + 0.0722 * b
    # Hot white/pink neon tube core (>0.55 linear) -> huge crimson-white HDR emission!
    neon_core = np.clip((lum - 0.45) / 0.55, 0.0, 1.0) ** 1.7
    red_dom = np.clip((r - b * 0.85) / 0.45, 0.0, 1.0)
    # Warm golden streetlamp bokeh highlights vs crimson neon
    x_norm = np.linspace(0, 1, 1920, dtype=np.float32)[None, :]
    w_left = np.clip((0.52 - x_norm) / 0.16, 0.0, 1.0)
    w_left = w_left * w_left * (3.0 - 2.0 * w_left)
    left_boost = neon_core[..., None] * np.array([22.0, 5.2, 4.5], dtype=np.float32) + (red_dom * 3.8)[..., None] * np.array([1.0, 0.06, 0.09], dtype=np.float32)
    right_boost = neon_core[..., None] * np.array([10.5, 10.5, 13.5], dtype=np.float32)
    hdr_boost = left_boost * w_left[..., None] + right_boost * (1.0 - w_left[..., None])
    plate_hdr = med_lin * 1.10 + hdr_boost

    # 2048x1024 equirectangular map in Blender World convention (+Y forward at u=0.25!)
    W, H = 2048, 1024
    jj, ii = np.mgrid[0:H, 0:W]
    u = (ii + 0.5) / W
    v = (jj + 0.5) / H
    lon = (0.5 - u) * 2.0 * np.pi
    lat = (0.5 - v) * np.pi
    dx = np.cos(lat) * np.cos(lon)
    dy = np.cos(lat) * np.sin(lon)
    dz = np.sin(lat)

    # Dark moody rainy night sky (deep midnight blue-indigo above, rain-slicked dark reflections below)
    up_w = np.clip(dz * 1.2, 0.0, 1.0)
    dn_w = np.clip(-dz * 1.3, 0.0, 1.0)
    night_base = (
        up_w[..., None] * np.array([0.012, 0.018, 0.042], dtype=np.float32)
        + (1.0 - up_w - dn_w).clip(0, 1)[..., None] * np.array([0.018, 0.014, 0.022], dtype=np.float32)
        + dn_w[..., None] * np.array([0.024, 0.012, 0.018], dtype=np.float32)
    )
    hdri = night_base.copy()

    # Large atmospheric crimson-red Chinatown neon glow extending along left street (-X)
    left_neon_glow = np.exp(-((dx + 0.72) ** 2 / 0.22 + (dy - 0.35) ** 2 / 0.35 + (dz - 0.12) ** 2 / 0.25))
    hdri += left_neon_glow[..., None] * np.array([5.8, 0.45, 0.55], dtype=np.float32)

    # Cool cobalt-blue streetlamp & wet night atmosphere on right (+X)
    right_blue_glow = np.exp(-((dx - 0.75) ** 2 / 0.24 + (dy - 0.30) ** 2 / 0.35 + (dz - 0.18) ** 2 / 0.28))
    hdri += right_blue_glow[..., None] * np.array([0.55, 1.25, 3.8], dtype=np.float32)

    # Distant warm amber streetlamp bokeh point behind-left & behind-right so glossy rings & water catch warm rear glints!
    rear_gold = np.exp(-((dx - 0.28) ** 2 / 0.04 + (dy + 0.85) ** 2 / 0.10 + (dz - 0.22) ** 2 / 0.04))
    hdri += rear_gold[..., None] * np.array([6.0, 3.4, 0.9], dtype=np.float32)
    rear_red = np.exp(-((dx + 0.45) ** 2 / 0.06 + (dy + 0.78) ** 2 / 0.12 + (dz - 0.05) ** 2 / 0.06))
    hdri += rear_red[..., None] * np.array([3.2, 0.25, 0.35], dtype=np.float32)

    # Project actual HDR-expanded Chinatown neon rain plate onto forward +Y cone!
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

    hdr_path = WORK / "scene_matched_neon_rain_2k.hdr"
    ok = cv2.imwrite(str(hdr_path), hdri[:, :, ::-1].astype(np.float32))
    assert ok, "Failed to write neon rain HDRI"

    tm = hdri / (1.0 + hdri)
    prev = (linear_to_srgb(tm) * 255.0).clip(0, 255).astype(np.uint8)
    cv2.imwrite(str(WORK / "scene_hdri_preview.jpg"), prev[:, :, ::-1], [int(cv2.IMWRITE_JPEG_QUALITY), 92])
    print(f"Saved Neon Rain HDRI: {hdr_path} (max radiance={hdri.max():.2f}, mean={hdri.mean():.3f})", flush=True)

if __name__ == "__main__":
    frames = extract_plate()
    track = track_cinema_camera_and_lighting(frames)
    build_neon_rain_hdri(frames, track)
    print("NEON RAIN TRACK & HDRI DONE!", flush=True)
