import json
import subprocess
from pathlib import Path
import numpy as np
import cv2

ROOT = Path(r"D:\VFX_Work_Claude\_repo\COLMAP_OPENGL_COMPOSITE")
PROJ = ROOT / "examples" / "foreground_cafe_peek"
WORK = PROJ / "work"
OUT = PROJ / "out"
OUT.mkdir(parents=True, exist_ok=True)

PLATE_DIR = WORK / "frames"
BEAUTY_DIR = WORK / "render_beauty"
COMP_FRAMES = WORK / "comp_frames"

tools = json.loads((ROOT / "tools.local.json").read_text(encoding="utf-8"))
FFMPEG = tools["ffmpeg"]
NUM_FRAMES = 150
FPS = 25

def srgb_to_linear(img):
    return np.where(img <= 0.04045, img / 12.92, ((img + 0.055) / 1.055) ** 2.4)

def linear_to_srgb(img):
    c = np.clip(img, 0.0, 1.0)
    return np.where(c <= 0.0031308, c * 12.92, 1.055 * (c ** (1.0 / 2.4)) - 0.055)

def apply_subtle_radial_ca(img_bgr, strength=1.10):
    """Whisper-subtle outer-corner lateral chromatic aberration matching real cinema glass optics."""
    h, w = img_bgr.shape[:2]
    cx, cy = w * 0.5, h * 0.5
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    dx = (xx - cx) / cx
    dy = (yy - cy) / cy
    r2 = dx * dx + dy * dy
    shift = strength * r2
    out = img_bgr.copy()
    out[:, :, 2] = cv2.remap(img_bgr[:, :, 2], (xx + dx * shift).astype(np.float32), (yy + dy * shift).astype(np.float32), cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT101)
    out[:, :, 0] = cv2.remap(img_bgr[:, :, 0], (xx - dx * shift * 0.85).astype(np.float32), (yy - dy * shift * 0.85).astype(np.float32), cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT101)
    return out

def main():
    COMP_FRAMES.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(20261124)

    # Measure exact dark shadow floor (black-level toe) from the real café plate!
    p0 = cv2.imread(str(PLATE_DIR / "frame_0000.png")).astype(np.float32) / 255.0
    p0_lin = srgb_to_linear(p0)
    # 1st percentile in dark shadows (matches real café black-level toe in BGR!)
    toe_bgr = np.percentile(p0_lin.reshape(-1, 3), 1.5, axis=0).astype(np.float32)
    print(f"Measured real café linear black-level toe (BGR): {np.round(toe_bgr, 4)}", flush=True)

    print(f"=== Compositing {NUM_FRAMES} Full-HD frames (1920x1080 linear filmic comp) ===", flush=True)
    for i in range(NUM_FRAMES):
        plate_bgr = cv2.imread(str(PLATE_DIR / f"frame_{i:04d}.png"))
        cg_raw = cv2.imread(str(BEAUTY_DIR / f"frame_{i:04d}.png"), cv2.IMREAD_UNCHANGED)

        plate_lin = srgb_to_linear(plate_bgr.astype(np.float32) / 255.0)
        scale = 65535.0 if cg_raw.dtype == np.uint16 else 255.0
        cg_srgb = cg_raw[:, :, :3].astype(np.float32) / scale
        alpha = cg_raw[:, :, 3].astype(np.float32) / scale

        # 1) Subtle wide-angle outer-corner lateral CA on CG layer
        cg_srgb_ca = apply_subtle_radial_ca(cg_srgb, strength=1.05)
        cg_lin = srgb_to_linear(np.clip(cg_srgb_ca, 0.0, 1.0))

        # 2) Exact black-level toe match to Pexels 7263095 dark shadows (never crushed below real plate black!)
        cg_lin = cg_lin * 0.985 + toe_bgr * np.clip(1.0 - cg_lin * 3.0, 0.0, 1.0)

        # 3) Two-scale physical window light-wrap (bright overcast window softly bleeding around kettle, glass & leaves!)
        lum_p = 0.0722 * plate_lin[:, :, 0] + 0.7152 * plate_lin[:, :, 1] + 0.2126 * plate_lin[:, :, 2]
        bright_p = plate_lin * np.clip((lum_p - 0.16) / 0.55, 0.0, 1.4)[..., None]
        wrap_tight = cv2.GaussianBlur(bright_p, (0, 0), 8.0)
        wrap_wide = cv2.GaussianBlur(bright_p, (0, 0), 26.0)

        a_blur_tight = cv2.GaussianBlur(alpha, (0, 0), 7.5)
        a_blur_wide = cv2.GaussianBlur(alpha, (0, 0), 24.0)
        edge_tight = np.clip(alpha * (1.0 - a_blur_tight) * 1.85, 0.0, 1.0)[..., None]
        edge_wide = np.clip(alpha * (1.0 - a_blur_wide) * 1.35, 0.0, 1.0)[..., None]
        light_wrap = wrap_tight * edge_tight * 0.15 + wrap_wide * edge_wide * 0.10

        # 4) Subtle warm optical bloom ONLY around the foreground glowing Edison filament bulb (upper-right!)
        cg_lum = 0.0722 * cg_lin[:, :, 0] + 0.7152 * cg_lin[:, :, 1] + 0.2126 * cg_lin[:, :, 2]
        hot = cg_lin * np.clip((cg_lum - 0.80) / 0.25, 0.0, 1.0)[..., None] * alpha[..., None]
        bloom = cv2.GaussianBlur(hot, (0, 0), 22.0) * 0.22

        # 5) Linear Over + soft additive Edison optical halation
        a3 = alpha[..., None]
        comp_lin = (cg_lin + light_wrap) * a3 + plate_lin * (1.0 - a3) + bloom
        comp_srgb = linear_to_srgb(comp_lin)

        # 6) Subtle fine camera sensor grain matching real video texture over CG alpha
        grain = rng.normal(0.0, 0.0040, size=(plate_bgr.shape[0], plate_bgr.shape[1], 1)).astype(np.float32)
        comp_srgb = np.clip(comp_srgb + grain * a3, 0.0, 1.0)
        comp_u8 = (comp_srgb * 255.0 + 0.5).clip(0, 255).astype(np.uint8)
        cv2.imwrite(str(COMP_FRAMES / f"frame_{i:04d}.png"), comp_u8)

        if (i + 1) % 30 == 0 or i == 0:
            print(f"  Composited {i+1:03d}/{NUM_FRAMES}", flush=True)

    # Encode clean final 1920x1080 H.264 High CRF 15 BT.709 deliverable (NO breakdown per user request!)
    out_mp4 = OUT / "Foreground_Cafe_Peek_Composite.mp4"
    cmd = [
        FFMPEG, "-y", "-v", "error",
        "-framerate", str(FPS),
        "-i", str(COMP_FRAMES / "frame_%04d.png"),
        "-c:v", "libx264", "-profile:v", "high", "-crf", "15", "-preset", "slow",
        "-pix_fmt", "yuv420p",
        "-color_primaries", "bt709", "-color_trc", "bt709", "-colorspace", "bt709",
        "-movflags", "+faststart",
        str(out_mp4),
    ]
    subprocess.run(cmd, check=True)
    print(f"Encoded final composite: {out_mp4} ({out_mp4.stat().st_size / 1e6:.2f} MB)", flush=True)

    # Save 5-frame QA strip
    qa_idxs = [0, 35, 75, 112, 145]
    strip = np.hstack([cv2.resize(cv2.imread(str(COMP_FRAMES / f"frame_{k:04d}.png")), (640, 360)) for k in qa_idxs])
    cv2.imwrite(str(WORK / "qa_composite_strip.jpg"), strip, [int(cv2.IMWRITE_JPEG_QUALITY), 92])
    print("FINAL CAFE PEEK COMPOSITE READY!", flush=True)

if __name__ == "__main__":
    main()
