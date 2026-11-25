from pathlib import Path
import numpy as np
import cv2

WORK = Path(r"D:\VFX_Work_Claude\_repo\COLMAP_OPENGL_COMPOSITE\examples\foreground_cafe_peek\work")

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

    # Subtle window light-wrap preview
    blur_p = cv2.GaussianBlur(plate_lin, (0, 0), 12.0)
    blur_a = cv2.GaussianBlur(alpha[:, :, 0], (0, 0), 10.0)[..., None]
    edge = np.clip(alpha * (1.0 - blur_a) * 1.5, 0.0, 1.0)
    comp_lin = (cg_lin + blur_p * edge * 0.14) * alpha + plate_lin * (1.0 - alpha)
    comp = (linear_to_srgb(comp_lin) * 255.0).clip(0, 255).astype(np.uint8)
    cv2.putText(comp, f"CAFE LOOKDEV f{fnum:03d}", (24, 44), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2, cv2.LINE_AA)
    return comp

panels = [comp_one(1), comp_one(75), comp_one(145)]
probe = cv2.imread(str(WORK / "lookdev_hdri_probe.png"))
probe = cv2.resize(probe, (panels[0].shape[1], panels[0].shape[0]))
cv2.putText(probe, "CAFE HDRI ORIENTATION PROBE", (24, 44), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2, cv2.LINE_AA)

sheet = np.vstack([np.hstack([panels[0], panels[1]]), np.hstack([panels[2], probe])])
out = WORK / "lookdev_comp_sheet.jpg"
cv2.imwrite(str(out), sheet, [int(cv2.IMWRITE_JPEG_QUALITY), 92])
print(f"Saved {out}", flush=True)
