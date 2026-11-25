import json
import math
import subprocess
from pathlib import Path
import numpy as np
import cv2
from PIL import Image, ImageDraw, ImageFont

ROOT = Path(r"D:\VFX_Work_Claude\_repo\COLMAP_OPENGL_COMPOSITE")
PROJ = ROOT / "examples" / "foreground_cloth_tunnel"
WORK = PROJ / "work"
OUT = PROJ / "out"
OUT.mkdir(parents=True, exist_ok=True)

PLATE_DIR = WORK / "frames"
BEAUTY_DIR = WORK / "render_beauty"
WIRE_DIR = WORK / "render_wire"
COMP_FRAMES = WORK / "comp_frames"
BREAKDOWN_FRAMES = WORK / "breakdown_frames"
COMPARE_FRAMES = WORK / "compare_frames"

tools = json.loads((ROOT / "tools.local.json").read_text(encoding="utf-8"))
FFMPEG = tools["ffmpeg"]
FFPROBE = tools["ffprobe"]

NUM_FRAMES = 125
FPS = 25

def srgb_to_linear(img):
    return np.where(img <= 0.04045, img / 12.92, ((img + 0.055) / 1.055) ** 2.4)

def linear_to_srgb(img):
    c = np.clip(img, 0.0, 1.0)
    return np.where(c <= 0.0031308, c * 12.92, 1.055 * (c ** (1.0 / 2.4)) - 0.055)

def load_font(size, italic=False, bold=False):
    candidates = []
    if italic and bold:
        candidates += [r"C:\Windows\Fonts\segoeuiz.ttf", r"C:\Windows\Fonts\arialbi.ttf"]
    elif italic:
        candidates += [r"C:\Windows\Fonts\segoeuii.ttf", r"C:\Windows\Fonts\ariali.ttf"]
    elif bold:
        candidates += [r"C:\Windows\Fonts\segoeuib.ttf", r"C:\Windows\Fonts\arialbd.ttf"]
    candidates += [r"C:\Windows\Fonts\segoeui.ttf", r"C:\Windows\Fonts\arial.ttf"]
    for p in candidates:
        if Path(p).exists():
            return ImageFont.truetype(p, size)
    return ImageFont.load_default()

def apply_subtle_radial_ca(img_bgr, strength=1.25):
    """Whisper-subtle wide-angle lateral chromatic aberration on outer corners of CG layer."""
    h, w = img_bgr.shape[:2]
    cx, cy = w * 0.5, h * 0.5
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    dx = (xx - cx) / cx
    dy = (yy - cy) / cy
    r2 = dx * dx + dy * dy
    shift = strength * r2
    map_rx = (xx + dx * shift).astype(np.float32)
    map_ry = (yy + dy * shift).astype(np.float32)
    map_bx = (xx - dx * shift * 0.85).astype(np.float32)
    map_by = (yy - dy * shift * 0.85).astype(np.float32)

    out = img_bgr.copy()
    out[:, :, 2] = cv2.remap(img_bgr[:, :, 2], map_rx, map_ry, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT101)
    out[:, :, 0] = cv2.remap(img_bgr[:, :, 0], map_bx, map_by, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT101)
    return out

def composite_frame(idx, rng):
    plate_bgr = cv2.imread(str(PLATE_DIR / f"frame_{idx:04d}.png"))
    cg_raw = cv2.imread(str(BEAUTY_DIR / f"frame_{idx:04d}.png"), cv2.IMREAD_UNCHANGED)
    wire_raw = cv2.imread(str(WIRE_DIR / f"frame_{idx:04d}.png"), cv2.IMREAD_UNCHANGED)

    plate_lin = srgb_to_linear(plate_bgr.astype(np.float32) / 255.0)
    if cg_raw.dtype == np.uint16:
        cg_srgb = cg_raw[:, :, :3].astype(np.float32) / 65535.0
        alpha = (cg_raw[:, :, 3].astype(np.float32) / 65535.0)
    else:
        cg_srgb = cg_raw[:, :, :3].astype(np.float32) / 255.0
        alpha = (cg_raw[:, :, 3].astype(np.float32) / 255.0)

    # 1) Subtle wide-angle optical CA on CG RGB (matching 16.5mm lens corner optics)
    cg_srgb_ca = apply_subtle_radial_ca(cg_srgb, strength=1.15)
    cg_lin = srgb_to_linear(np.clip(cg_srgb_ca, 0.0, 1.0))

    # 2) Match live-action shadow black-level toe (warm lifted interior shadow floor in BGR)
    toe_bgr = np.array([0.0068, 0.0074, 0.0086], dtype=np.float32)
    cg_lin = cg_lin * 0.985 + toe_bgr * np.clip(1.0 - cg_lin * 2.5, 0.0, 1.0)

    # 3) Two-scale physical doorway light-wrap (warm bright background halation bleeding softly onto cloth inner edges!)
    lum_plate = 0.0722 * plate_lin[:, :, 0] + 0.7152 * plate_lin[:, :, 1] + 0.2126 * plate_lin[:, :, 2]
    bright_plate = plate_lin * np.clip((lum_plate - 0.18) / 0.60, 0.0, 1.5)[..., None]
    wrap_tight = cv2.GaussianBlur(bright_plate, (0, 0), 9.0)
    wrap_wide = cv2.GaussianBlur(bright_plate, (0, 0), 28.0)

    a_blur_tight = cv2.GaussianBlur(alpha, (0, 0), 8.0)
    a_blur_wide = cv2.GaussianBlur(alpha, (0, 0), 26.0)
    edge_tight = np.clip(alpha * (1.0 - a_blur_tight) * 1.9, 0.0, 1.0)[..., None]
    edge_wide = np.clip(alpha * (1.0 - a_blur_wide) * 1.4, 0.0, 1.0)[..., None]

    light_wrap = wrap_tight * edge_tight * 0.16 + wrap_wide * edge_wide * 0.11

    # 4) Linear Over compositing
    a3 = alpha[..., None]
    comp_lin = (cg_lin + light_wrap) * a3 + plate_lin * (1.0 - a3)
    comp_srgb = linear_to_srgb(comp_lin)

    # 5) Fine matched camera sensor grain over CG alpha region
    noise = rng.normal(0.0, 0.0042, size=(plate_bgr.shape[0], plate_bgr.shape[1], 1)).astype(np.float32)
    comp_srgb = np.clip(comp_srgb + noise * a3, 0.0, 1.0)
    comp_u8 = (comp_srgb * 255.0 + 0.5).clip(0, 255).astype(np.uint8)

    # Clean RENDER ONLY on matte black (exact straight->premul in linear space -> sRGB!)
    ro_lin = cg_lin * a3
    ro_u8 = (linear_to_srgb(ro_lin) * 255.0 + 0.5).clip(0, 255).astype(np.uint8)

    # Clean 3D VIEW on matte black
    w_rgb = wire_raw[:, :, :3].astype(np.float32) / (65535.0 if wire_raw.dtype == np.uint16 else 255.0)
    w_a = (wire_raw[:, :, 3].astype(np.float32) / (65535.0 if wire_raw.dtype == np.uint16 else 255.0))[..., None]
    wire_u8 = (w_rgb * w_a * 255.0 + 0.5).clip(0, 255).astype(np.uint8)

    return plate_bgr, comp_u8, ro_u8, wire_u8

def draw_labels_pil(img_bgr, labels):
    pil = Image.fromarray(img_bgr[:, :, ::-1])
    draw = ImageDraw.Draw(pil)
    for text, xy, font, fill in labels:
        x, y = xy
        # Soft drop shadow for legibility on any bright/dark plate corner
        draw.text((x + 2, y + 2), text, font=font, fill=(0, 0, 0))
        draw.text((x, y), text, font=font, fill=fill)
    return np.array(pil)[:, :, ::-1]

def encode_h264(frame_pattern, out_mp4, fps=FPS):
    cmd = [
        FFMPEG, "-y", "-v", "error",
        "-framerate", str(fps),
        "-i", str(frame_pattern),
        "-c:v", "libx264", "-profile:v", "high", "-crf", "16", "-preset", "slow",
        "-pix_fmt", "yuv420p",
        "-color_primaries", "bt709", "-color_trc", "bt709", "-colorspace", "bt709",
        "-movflags", "+faststart",
        str(out_mp4),
    ]
    subprocess.run(cmd, check=True)
    print(f"Encoded: {out_mp4} ({out_mp4.stat().st_size / 1e6:.2f} MB)", flush=True)

def main():
    COMP_FRAMES.mkdir(parents=True, exist_ok=True)
    BREAKDOWN_FRAMES.mkdir(parents=True, exist_ok=True)
    COMPARE_FRAMES.mkdir(parents=True, exist_ok=True)

    f_label = load_font(26, italic=True, bold=True)
    f_footer = load_font(27, italic=False, bold=True)
    f_cmp = load_font(28, italic=False, bold=True)

    rng = np.random.default_rng(20261123)

    # Pre-cache all 125 frames (1920x1080 comp + 1080x608 resized panels for ultra-clean 2x loop breakdown)
    W_V, H_V = 1080, 1350
    P_W, P_H = 1080, 608
    TOP_Y = 24
    BOT_Y = TOP_Y + P_H + 14  # 646..1254

    yy, xx = np.mgrid[0:P_H, 0:P_W].astype(np.float32)

    cache_comp = []
    cache_ro = []
    cache_wire = []

    print("=== 1. Compositing 125 full-HD frames (1920x1080) ===", flush=True)
    for i in range(NUM_FRAMES):
        plate_u8, comp_u8, ro_u8, wire_u8 = composite_frame(i, rng)
        cv2.imwrite(str(COMP_FRAMES / f"frame_{i:04d}.png"), comp_u8)

        # Downscale cleanly with INTER_AREA to 1080x608 for 4:5 vertical showcase
        comp_s = cv2.resize(comp_u8, (P_W, P_H), interpolation=cv2.INTER_AREA)
        plate_s = cv2.resize(plate_u8, (P_W, P_H), interpolation=cv2.INTER_AREA)
        ro_s = cv2.resize(ro_u8, (P_W, P_H), interpolation=cv2.INTER_AREA)
        wire_s = cv2.resize(wire_u8, (P_W, P_H), interpolation=cv2.INTER_AREA)

        cache_comp.append(comp_s)
        cache_ro.append(ro_s)
        cache_wire.append(wire_s)

        # Build RESULT / ORIGINAL comparison frame (1080x1350)
        cmp_canvas = np.zeros((H_V, W_V, 3), dtype=np.uint8)
        cmp_canvas[TOP_Y:TOP_Y + P_H, :] = comp_s
        cmp_canvas[BOT_Y:BOT_Y + P_H, :] = plate_s
        cmp_canvas = draw_labels_pil(cmp_canvas, [
            ("RESULT", (36, TOP_Y + 20), f_cmp, (245, 245, 245)),
            ("ORIGINAL", (36, BOT_Y + 20), f_cmp, (245, 245, 245)),
            ("Foreground Cloth FX · Scene-Matched HDRI · Comp", (36, BOT_Y + P_H + 28), f_footer, (220, 220, 220)),
        ])
        cv2.imwrite(str(COMPARE_FRAMES / f"frame_{i:04d}.png"), cmp_canvas)

        if (i + 1) % 25 == 0 or i == 0:
            print(f"  Composited {i+1:03d}/{NUM_FRAMES}", flush=True)

    print("=== 2. Assembling 2x-Loop @_.meekat-style Breakdown (1080x1350, 250f / 10.0s) ===", flush=True)
    # Loop 1 (0..124): Iconic center diagonal split (x_mid gently breathing around 0.50, tilt 15 deg!)
    # Loop 2 (125..249): Smooth sweeping diagonal wipe revealing full left 3D wireframe & full right render beauty!
    for f_out in range(NUM_FRAMES * 2):
        idx = f_out % NUM_FRAMES
        t_loop = idx / float(NUM_FRAMES - 1)

        if f_out < NUM_FRAMES:
            # First pass: steady editorial diagonal split with subtle organic drift (0.48..0.55)
            x_center = 0.50 + 0.045 * math.sin(t_loop * math.pi * 2.0)
        else:
            # Second pass: smooth cinema wipe left -> right -> back to center
            if t_loop < 0.30:
                u = t_loop / 0.30
                u = u * u * (3 - 2 * u)
                x_center = 0.50 - 0.36 * u  # 0.50 -> 0.14 (reveals left hangers/pearls/eyelet wireframe!)
            elif t_loop < 0.72:
                u = (t_loop - 0.30) / 0.42
                u = u * u * (3 - 2 * u)
                x_center = 0.14 + 0.74 * u  # 0.14 -> 0.88 (reveals right cream blouse beauty!)
            else:
                u = (t_loop - 0.72) / 0.28
                u = u * u * (3 - 2 * u)
                x_center = 0.88 - 0.38 * u  # 0.88 -> 0.50

        # Diagonal split line: x_split(y) = x_center * P_W + slope * (P_H*0.5 - y)
        slope = 0.27  # ~15 deg stylish editorial slant matching @_.meekat!
        x_line = x_center * P_W + slope * (P_H * 0.5 - yy)
        dist = xx - x_line  # <0 left (RENDER ONLY), >0 right (3D VIEW)
        mask_wire = np.clip((dist + 1.0) * 0.5, 0.0, 1.0)[..., None]

        bot_split = (
            cache_ro[idx].astype(np.float32) * (1.0 - mask_wire)
            + cache_wire[idx].astype(np.float32) * mask_wire
        )
        # Subtle razor-thin bright silver-white diagonal divider line (1.5px anti-aliased)
        line_alpha = np.clip(1.0 - np.abs(dist) / 1.35, 0.0, 1.0)[..., None] * 0.65
        bot_split = bot_split * (1.0 - line_alpha) + 235.0 * line_alpha
        bot_u8 = bot_split.clip(0, 255).astype(np.uint8)

        canvas = np.zeros((H_V, W_V, 3), dtype=np.uint8)
        canvas[TOP_Y:TOP_Y + P_H, :] = cache_comp[idx]
        canvas[BOT_Y:BOT_Y + P_H, :] = bot_u8

        labels = [
            ("FINAL", (46, TOP_Y + 20), f_label, (245, 245, 245)),
            ("Cloth FX · Scene HDRI · Comp", (46, BOT_Y + P_H + 30), f_footer, (238, 238, 238)),
        ]
        # Keep RENDER ONLY / 3D VIEW labels cleanly visible whenever their side has room
        if x_center > 0.24:
            labels.append(("RENDER ONLY", (46, BOT_Y + 20), f_label, (240, 240, 240)))
        if x_center < 0.76:
            labels.append(("3D VIEW", (W_V - 165, BOT_Y + 20), f_label, (240, 240, 240)))

        canvas = draw_labels_pil(canvas, labels)
        cv2.imwrite(str(BREAKDOWN_FRAMES / f"frame_{f_out:04d}.png"), canvas)

    print("=== 3. Encoding final H.264 BT.709 deliverables ===", flush=True)
    encode_h264(COMP_FRAMES / "frame_%04d.png", OUT / "Foreground_Cloth_Tunnel_Composite.mp4", fps=FPS)
    encode_h264(BREAKDOWN_FRAMES / "frame_%04d.png", OUT / "Foreground_Cloth_Tunnel_Breakdown_1080x1350.mp4", fps=FPS)
    encode_h264(COMPARE_FRAMES / "frame_%04d.png", OUT / "Foreground_Cloth_Tunnel_Compare_1080x1350.mp4", fps=FPS)

    # Save visual QA contact sheets for both Composite and Breakdown!
    qa_idxs = [0, 30, 62, 95, 120]
    c_row = [cv2.resize(cv2.imread(str(COMP_FRAMES / f"frame_{k:04d}.png")), (640, 360)) for k in qa_idxs]
    cv2.imwrite(str(WORK / "qa_composite_strip.jpg"), np.hstack(c_row), [int(cv2.IMWRITE_JPEG_QUALITY), 92])

    b_idxs = [10, 62, 155, 200, 240]
    b_row = [cv2.resize(cv2.imread(str(BREAKDOWN_FRAMES / f"frame_{k:04d}.png")), (432, 540)) for k in b_idxs]
    cv2.imwrite(str(WORK / "qa_breakdown_strip.jpg"), np.hstack(b_row), [int(cv2.IMWRITE_JPEG_QUALITY), 92])

    print("ALL COMPOSITE & BREAKDOWN VIDEOS READY!", flush=True)

if __name__ == "__main__":
    main()
