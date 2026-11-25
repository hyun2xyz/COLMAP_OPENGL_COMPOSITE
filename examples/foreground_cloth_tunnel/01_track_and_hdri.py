import os, sys, json, math, subprocess, shutil
from pathlib import Path
import numpy as np
import cv2
import pycolmap
from scipy.signal import savgol_filter

ROOT = Path(r"D:\VFX_Work_Claude\_repo\COLMAP_OPENGL_COMPOSITE")
PROJ = ROOT / "examples" / "foreground_cloth_tunnel"
WORK = PROJ / "work"
FRAMES_DIR = WORK / "frames"
KF_DIR = WORK / "colmap_kf"
COLMAP_DIR = WORK / "colmap"

tools = json.loads((ROOT / "tools.local.json").read_text(encoding="utf-8"))
FFMPEG = tools["ffmpeg"]
FFPROBE = tools["ffprobe"]

RAW_VIDEO = PROJ / "footage" / "Pexels_3769965.mp4"
PLATE_MP4 = WORK / "plate.mp4"

# Use clean 5.00s interior push-in segment: t = 2.96s .. 7.96s (125 frames @ 25fps, past left entry door)
START_SEC = 2.96
NUM_FRAMES = 125
FPS = 25.0

def srgb_to_linear(img):
    return np.where(img <= 0.04045, img / 12.92, ((img + 0.055) / 1.055) ** 2.4)

def linear_to_srgb(img):
    c = np.clip(img, 0.0, 1.0)
    return np.where(c <= 0.0031308, c * 12.92, 1.055 * (c ** (1.0 / 2.4)) - 0.055)

def extract_plate():
    FRAMES_DIR.mkdir(parents=True, exist_ok=True)
    if not PLATE_MP4.exists():
        print("=== 1. Trimming clean 5.00s plate (125f @ 25fps) ===", flush=True)
        subprocess.run([
            FFMPEG, "-y", "-v", "error",
            "-ss", f"{START_SEC:.2f}", "-i", str(RAW_VIDEO),
            "-frames:v", str(NUM_FRAMES),
            "-c:v", "libx264", "-crf", "14", "-preset", "medium", "-pix_fmt", "yuv420p",
            "-an", str(PLATE_MP4)
        ], check=True)
    existing = sorted(FRAMES_DIR.glob("frame_*.png"))
    if len(existing) != NUM_FRAMES:
        print("=== Extracting 125 full-res PNG frames ===", flush=True)
        for p in existing: p.unlink()
        subprocess.run([
            FFMPEG, "-y", "-v", "error", "-i", str(PLATE_MP4),
            "-start_number", "0", str(FRAMES_DIR / "frame_%04d.png")
        ], check=True)
    existing = sorted(FRAMES_DIR.glob("frame_*.png"))
    print(f"Frames ready: {len(existing)}", flush=True)
    return existing

def run_colmap(frames):
    print("=== 2. Running pycolmap SfM camera tracking ===", flush=True)
    if KF_DIR.exists(): shutil.rmtree(KF_DIR)
    if COLMAP_DIR.exists(): shutil.rmtree(COLMAP_DIR)
    KF_DIR.mkdir(parents=True, exist_ok=True)
    COLMAP_DIR.mkdir(parents=True, exist_ok=True)

    # Downscale to 960x540 for fast, accurate SIFT sequential tracking across all 125 frames
    scale = 0.5
    for idx, fp in enumerate(frames):
        im = cv2.imread(str(fp))
        im_s = cv2.resize(im, (960, 540), interpolation=cv2.INTER_AREA)
        cv2.imwrite(str(KF_DIR / f"f_{idx:04d}.jpg"), im_s, [int(cv2.IMWRITE_JPEG_QUALITY), 94])

    db_path = COLMAP_DIR / "database.db"
    reader_opts = pycolmap.ImageReaderOptions(camera_model="SIMPLE_RADIAL")
    pycolmap.extract_features(
        database_path=str(db_path),
        image_path=str(KF_DIR),
        camera_mode=pycolmap.CameraMode.SINGLE,
        reader_options=reader_opts,
    )
    pair_opts = pycolmap.SequentialPairingOptions()
    pair_opts.overlap = 12
    pair_opts.quadratic_overlap = True
    pycolmap.match_sequential(
        database_path=str(db_path),
        pairing_options=pair_opts,
    )
    sparse_dir = COLMAP_DIR / "sparse"
    sparse_dir.mkdir(exist_ok=True)
    recons = pycolmap.incremental_mapping(
        database_path=str(db_path),
        image_path=str(KF_DIR),
        output_path=str(sparse_dir),
    )
    if not recons:
        raise RuntimeError("pycolmap produced no reconstruction!")
    best_id = max(recons.keys(), key=lambda k: len(recons[k].images))
    rec = recons[best_id]
    print(f"COLMAP solved {len(rec.images)}/{len(frames)} frames, {len(rec.points3D)} 3D points", flush=True)

    cam = next(iter(rec.cameras.values()))
    f_px_s = float(cam.focal_length)
    f_px_full = f_px_s / scale
    # Full frame 1920x1080 -> 36mm equivalent focal length
    focal_mm_35 = f_px_full * 36.0 / 1920.0
    print(f"Solved camera: model={cam.model}, f_full={f_px_full:.1f}px ({focal_mm_35:.2f}mm eq)", flush=True)

    # Extract per-frame camera-to-world pose in OpenCV convention (X right, Y down, Z forward)
    poses = {}
    for img_id, img in rec.images.items():
        idx = int(img.name.split("_")[1].split(".")[0])
        c_from_w = img.cam_from_world() if callable(img.cam_from_world) else img.cam_from_world
        R_cw = np.array(c_from_w.rotation.matrix(), dtype=np.float64)
        t_cw = np.array(c_from_w.translation, dtype=np.float64)
        R_wc = R_cw.T
        pos_w = -R_wc @ t_cw
        poses[idx] = (R_wc, pos_w)

    # Interpolate any missing frames smoothly in Euler + position
    solved_idxs = sorted(poses.keys())
    print(f"Solved frame span: {solved_idxs[0]}..{solved_idxs[-1]} ({len(solved_idxs)} frames)", flush=True)

    # Align world frame to frame 0 (frame 0 sits at origin, looking along +Z in OpenCV -> +Y in Blender)
    R0, p0 = poses[solved_idxs[0]]
    rel_eulers = []
    rel_pos = []
    for idx in solved_idxs:
        R, p = poses[idx]
        R_rel = R0.T @ R
        p_rel = R0.T @ (p - p0)
        # Small-angle / Rodrigues vector
        rvec, _ = cv2.Rodrigues(R_rel)
        rel_eulers.append(rvec.ravel())
        rel_pos.append(p_rel)

    solved_arr = np.array(solved_idxs, dtype=np.float64)
    rel_eulers = np.array(rel_eulers)
    rel_pos = np.array(rel_pos)

    all_t = np.arange(NUM_FRAMES, dtype=np.float64)
    rvecs_all = np.column_stack([np.interp(all_t, solved_arr, rel_eulers[:, i]) for i in range(3)])
    pos_all = np.column_stack([np.interp(all_t, solved_arr, rel_pos[:, i]) for i in range(3)])

    # Gentle smoothing to remove single-frame SfM jitter while preserving real handheld sway
    win = 7
    rvecs_s = np.column_stack([savgol_filter(rvecs_all[:, i], win, 2) for i in range(3)])
    pos_s = np.column_stack([savgol_filter(pos_all[:, i], win, 2) for i in range(3)])

    # Normalize forward travel scale so total forward camera path length = 2.4 meters (realistic walking speed ~0.48 m/s)
    total_disp = float(np.linalg.norm(pos_s[-1] - pos_s[0]))
    meter_scale = 2.4 / max(total_disp, 1e-5)
    pos_m = pos_s * meter_scale

    # Export sparse 3D points (aligned to frame 0 & scaled) for optional 3D tracking visualization
    pts = []
    for pid, p3 in rec.points3D.items():
        if p3.error < 2.0:
            xyz = (R0.T @ (np.array(p3.xyz) - p0)) * meter_scale
            if 0.3 < xyz[2] < 12.0:
                pts.append([float(xyz[0]), float(xyz[1]), float(xyz[2]),
                            int(p3.color[0]), int(p3.color[1]), int(p3.color[2])])

    # Convert OpenCV camera-rel (X right, Y down, Z forward) -> Blender camera-rel (X right, Y forward, Z up)
    # In Blender: pos_bl = (x, z, -y)
    # Rotation matrix in Blender world for camera originally looking +Y with +Z up:
    # A Blender camera at identity looks along -Z with +Y up. Looking along +Y with +Z up is Rx(+90 deg).
    track_frames = []
    for i in range(NUM_FRAMES):
        R_cv, _ = cv2.Rodrigues(rvecs_s[i])
        # Convert R_cv to Blender world basis: M_b_cv maps (x, y, z)_cv -> (x, z, -y)_bl
        M = np.array([[1, 0, 0], [0, 0, 1], [0, -1, 0]], dtype=np.float64)
        R_bl_delta = M @ R_cv @ M.T
        p_bl = M @ pos_m[i]
        track_frames.append({
            "frame": i,
            "pos_cv_m": [float(v) for v in pos_m[i]],
            "rvec_cv": [float(v) for v in rvecs_s[i]],
            "pos_bl_m": [float(v) for v in p_bl],
            "R_bl_delta": R_bl_delta.tolist(),
        })

    # Compute camera velocity & acceleration in Blender space to drive cloth/hanger physics
    pos_bl_arr = np.array([f["pos_bl_m"] for f in track_frames])
    vel = np.gradient(pos_bl_arr, 1.0 / FPS, axis=0)
    acc = np.gradient(vel, 1.0 / FPS, axis=0)
    for i in range(NUM_FRAMES):
        track_frames[i]["vel_bl"] = [float(v) for v in vel[i]]
        track_frames[i]["acc_bl"] = [float(v) for v in acc[i]]

    out_data = {
        "width": 1920,
        "height": 1080,
        "fps": FPS,
        "num_frames": NUM_FRAMES,
        "solved_count": len(solved_idxs),
        "focal_px": f_px_full,
        "focal_mm_35": focal_mm_35,
        "meter_scale": meter_scale,
        "total_forward_m": float(pos_bl_arr[-1, 1] - pos_bl_arr[0, 1]),
        "frames": track_frames,
    }
    (WORK / "camera_track.json").write_text(json.dumps(out_data, indent=2), encoding="utf-8")
    (WORK / "colmap_points.json").write_text(json.dumps({"points": pts[:4000]}), encoding="utf-8")
    print(f"Saved camera_track.json (forward travel={out_data['total_forward_m']:.2f}m, {len(pts)} pts)", flush=True)
    return out_data

def build_scene_hdri(frames, track):
    print("=== 3. Building scene-matched 360° float32 HDRI from plate ===", flush=True)
    # Sample 15 frames across the shot and take temporal median to remove moving person & get pure room illumination
    sample_idxs = np.linspace(0, len(frames) - 1, 15, dtype=int)
    stack = []
    for idx in sample_idxs:
        im = cv2.imread(str(frames[idx]))[:, :, ::-1].astype(np.float32) / 255.0  # RGB sRGB
        stack.append(srgb_to_linear(im))
    med_lin = np.median(np.stack(stack, axis=0), axis=0).astype(np.float32)  # (1080, 1920, 3) linear RGB

    # Measure room surface colors from actual plate regions!
    left_wall = np.median(med_lin[200:550, 40:260], axis=(0, 1))
    right_wall = np.median(med_lin[200:550, 1660:1880], axis=(0, 1))
    ceiling_col = np.median(med_lin[20:140, 600:1300], axis=(0, 1))
    floor_col = np.median(med_lin[880:1060, 400:1400], axis=(0, 1))
    print(f"Measured linear room palette: L_wall={np.round(left_wall,3)} R_wall={np.round(right_wall,3)} ceil={np.round(ceiling_col,3)} floor={np.round(floor_col,3)}", flush=True)

    # Inverse-tonemap the plate's bright highlights into physical HDR radiance:
    # 1) Blazing sunlit back doorway (high luminance pixels -> strong warm-daylight HDR key)
    lum = 0.2126 * med_lin[..., 0] + 0.7152 * med_lin[..., 1] + 0.0722 * med_lin[..., 2]
    # Strong HDR expansion for bright doorway/window (>0.72 linear)
    excess = np.clip((lum - 0.68) / 0.32, 0.0, 1.0)
    door_boost = (excess ** 2.0) * 20.0
    # Warm daylight tint for sunlit doorway, warmer tungsten tint for upper string lights/chandelier
    y_norm = np.linspace(0, 1, 1080, dtype=np.float32)[:, None]
    x_norm = np.linspace(0, 1, 1920, dtype=np.float32)[None, :]
    is_upper_lights = (y_norm < 0.42) & (x_norm < 0.55)
    hdr_tint = np.where(
        is_upper_lights[..., None],
        np.array([1.00, 0.76, 0.44], dtype=np.float32),
        np.array([1.00, 0.95, 0.88], dtype=np.float32),
    )
    plate_hdr = med_lin * 1.15 + hdr_tint * door_boost[..., None]

    # Build 2048x1024 equirectangular float32 HDRI in Blender's exact World convention!
    # In Blender World (Z-up):
    #   u = 0.5 - atan2(dy, dx) / (2*pi)   => dx = cos(lon), dy = sin(lon) where lon = (0.5 - u)*2*pi
    #   v = 0.5 - asin(dz) / pi            => lat = (0.5 - v)*pi, dz = sin(lat)
    # Wait: let's verify standard equirectangular:
    #   When camera looks along +Y (dx=0, dy=+1, dz=0): atan2(1, 0) = +pi/2 => u = 0.25!
    #   And we can place forward (+Y) directly at direction (0, 1, 0)!
    W, H = 2048, 1024
    jj, ii = np.mgrid[0:H, 0:W]
    u = (ii + 0.5) / W
    v = (jj + 0.5) / H
    # Blender cycles equirectangular mapping:
    #   phi = (0.5 - u) * 2*pi, theta = v * pi -> dir = (cos(phi)*sin(theta), sin(phi)*sin(theta), cos(theta))
    # Wait: Blender's actual formula in intern/cycles/kernel/svm/tex_coord.h is:
    #   u = -atan2(d.y, d.x) / (2*pi) + 0.5, v = atan2(d.z, hypot(d.x, d.y)) / pi + 0.5 (with v=0 bottom, v=1 top in OpenGL/Blender image)
    #   In image row jj (top=0 -> v_img=1): lat = (0.5 - (jj+0.5)/H) * pi, lon = (0.5 - (ii+0.5)/W) * 2*pi
    lon = (0.5 - u) * 2.0 * np.pi
    lat = (0.5 - v) * np.pi
    dx = np.cos(lat) * np.cos(lon)
    dy = np.cos(lat) * np.sin(lon)
    dz = np.sin(lat)

    # Base room shell illumination (warm interior walls ahead, dark shadowed closet behind camera!)
    # Forward factor: dy = +1 is straight ahead into sunlit room, dy = -1 is straight behind inside dark closet!
    fwd = np.clip((dy + 0.15) / 1.15, 0.0, 1.0)
    fwd_smooth = fwd * fwd * (3.0 - 2.0 * fwd)

    # Vertical gradient: ceiling above (dz>0), warm wall middle, dark walnut floor below (dz<0)
    up_w = np.clip(dz * 1.4, 0.0, 1.0)
    dn_w = np.clip(-dz * 1.4, 0.0, 1.0)
    side_w = np.clip(1.0 - up_w - dn_w, 0.0, 1.0)
    wall_col = np.where((dx < 0)[..., None], left_wall, right_wall)

    room_base = (
        up_w[..., None] * ceiling_col * 0.85
        + side_w[..., None] * wall_col * 0.72
        + dn_w[..., None] * floor_col * 0.55
    )

    # Dark moody closet/wardrobe behind the camera (dy < 0)
    closet_shadow = np.array([0.022, 0.016, 0.011], dtype=np.float32) * (0.7 + 0.5 * np.clip(dz + 0.5, 0.0, 1.0))[..., None]
    # Add a very soft dim warm bounce on the camera-facing side so dark cloth folds remain richly legible (never crushed black)
    closet_bounce = np.array([0.085, 0.064, 0.045], dtype=np.float32) * np.exp(-((dx + 0.15) ** 2 + (dz - 0.10) ** 2) / 0.85)[..., None]
    rear_base = closet_shadow + closet_bounce

    hdri = rear_base * (1.0 - fwd_smooth[..., None]) + room_base * fwd_smooth[..., None]

    # Project actual HDR-expanded living room plate onto the +Y forward camera cone!
    # Camera basis: right=+X, up=+Z, forward=+Y
    fx = float(track["focal_px"])
    fy = fx
    cx, cy = 960.0, 540.0
    in_front = dy > 0.05
    px = cx + fx * (dx / np.maximum(dy, 1e-4))
    py = cy - fy * (dz / np.maximum(dy, 1e-4))

    sampled = cv2.remap(
        plate_hdr, px.astype(np.float32), py.astype(np.float32),
        cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE
    )
    # Smooth feathered blend into the equirectangular shell
    margin_x = np.minimum(px / 140.0, (1919.0 - px) / 140.0)
    margin_y = np.minimum(py / 100.0, (1079.0 - py) / 100.0)
    blend = np.clip(np.minimum(margin_x, margin_y), 0.0, 1.0) * in_front.astype(np.float32)
    blend = blend * blend * (3.0 - 2.0 * blend)

    hdri = hdri * (1.0 - blend[..., None]) + sampled * blend[..., None]

    # Add subtle overhead warm chandelier glow slightly above the forward FOV (real fixture right above doorway)
    ch_glow = np.exp(-((dx + 0.05) ** 2 / 0.10 + (dy - 0.55) ** 2 / 0.18 + (dz - 0.82) ** 2 / 0.06))
    hdri += ch_glow[..., None] * np.array([4.5, 3.3, 1.9], dtype=np.float32)

    # Add a soft right-side warm doorway bounce (from the open white right doorway visible in frame!)
    right_door = np.exp(-((dx - 0.78) ** 2 / 0.14 + (dy - 0.52) ** 2 / 0.18 + (dz - 0.05) ** 2 / 0.25))
    hdri += right_door[..., None] * np.array([2.2, 1.95, 1.65], dtype=np.float32)

    # Save float32 Radiance .hdr (OpenCV expects BGR!)
    hdr_path = WORK / "scene_matched_livingroom_2k.hdr"
    ok = cv2.imwrite(str(hdr_path), hdri[:, :, ::-1].astype(np.float32))
    assert ok, "Failed to write HDRI .hdr"

    # Save tone-mapped sRGB preview for visual QA (Reinhard tone map)
    tm = hdri / (1.0 + hdri)
    prev = (linear_to_srgb(tm) * 255.0).clip(0, 255).astype(np.uint8)
    cv2.imwrite(str(WORK / "scene_hdri_preview.jpg"), prev[:, :, ::-1], [int(cv2.IMWRITE_JPEG_QUALITY), 92])
    print(f"Saved scene HDRI: {hdr_path} (max radiance={hdri.max():.2f}, mean={hdri.mean():.3f})", flush=True)

if __name__ == "__main__":
    frames = extract_plate()
    track = run_colmap(frames)
    build_scene_hdri(frames, track)
    print("ALL DONE!", flush=True)
