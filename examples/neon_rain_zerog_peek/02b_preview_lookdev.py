from pathlib import Path
import numpy as np
import cv2

WORK = Path(r"D:\VFX_Work_Claude\_repo\COLMAP_OPENGL_COMPOSITE\examples\neon_rain_zerog_peek\work")

def srgb_to_linear(img):
    return np.where(img <= 0.04045, img / 12.92, ((img + 0.055) / 1.055) ** 2.4)

def linear_to_srgb(img):
    c = np.clip(img, 0.0, 1.0)
    return np.where(c <= 0.0031308, c * 12.92, 1.055 * (c ** (1.0 / 2.4)) - 0.055)

def comp_one(fnum):
    plate = cv2.imread(str(WORK / "frames" / f"frame_{fnum-1:04d}.png"))
    cg = cv2.imread(str(WORK / f"lookdev_f{fnum:03d}.png"), cv2.IMREAD_UNCHANGED)
    h, w = cg.shape[:2]
    plate = cv2.resize(plate, (w, h), interpolation=cv2.INTER_AREA)
    plate_lin = srgb_to_linear(plate.astype(np.float32) / 255.0)
    scale = 65535.0 if cg.dtype == np.uint16 else 255.0
    cg_lin = srgb_to_linear(cg[:, :, :3].astype(np.float32) / scale)
    alpha = (cg[:, :, 3].astype(np.float32) / scale)[..., None]

    # Neon & headlight light-wrap + cinema specular halation preview
    blur_p = cv2.GaussianBlur(plate_lin, (0, 0), 12.0)
    blur_a = cv2.GaussianBlur(alpha[:, :, 0], (0, 0), 10.0)[..., None]
    edge = np.clip(alpha * (1.0 - blur_a) * 1.6, 0.0, 1.0)

    cg_lum = 0.0722 * cg_lin[:, :, 0] + 0.7152 * cg_lin[:, :, 1] + 0.2126 * cg_lin[:, :, 2]
    hot = cg_lin * np.clip((cg_lum - 0.70) / 0.35, 0.0, 1.0)[..., None] * alpha
    bloom = cv2.GaussianBlur(hot, (0, 0), 14.0) * 0.24 + cv2.GaussianBlur(hot, (0, 0), sigmaX=34.0, sigmaY=2.5) * 0.14

    comp_lin = (cg_lin + blur_p * edge * 0.18) * alpha + plate_lin * (1.0 - alpha) + bloom
    comp = (linear_to_srgb(comp_lin) * 255.0).clip(0, 255).astype(np.uint8)
    cv2.putText(comp, f"NEON ZERO-G LOOKDEV f{fnum:03d}", (24, 44), cv2.FONT_HERSHEY_SIMPLEX, 0.95, (255, 255, 255), 2, cv2.LINE_AA)
    return comp

panels = [comp_one(10), comp_one(60), comp_one(112)]
probe = cv2.imread(str(WORK / "lookdev_hdri_probe.png"))
probe = cv2.resize(probe, (panels[0].shape[1], panels[0].shape[0]))
cv2.putText(probe, "NEON RAIN HDRI PROBE", (24, 44), cv2.FONT_HERSHEY_SIMPLEX, 0.95, (255, 255, 255), 2, cv2.LINE_AA)

sheet = np.vstack([np.hstack([panels[0], panels[1]]), np.hstack([panels[2], probe])])
out = WORK / "lookdev_comp_sheet.jpg"
cv2.imwrite(str(out), sheet, [int(cv2.IMWRITE_JPEG_QUALITY), 92])
print(f"Saved {out}", flush=True)
