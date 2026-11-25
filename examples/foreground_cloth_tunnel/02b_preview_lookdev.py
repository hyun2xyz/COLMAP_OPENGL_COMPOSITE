from pathlib import Path
import numpy as np
import cv2

WORK = Path(r"D:\VFX_Work_Claude\_repo\COLMAP_OPENGL_COMPOSITE\examples\foreground_cloth_tunnel\work")

def srgb_to_linear(img):
    return np.where(img <= 0.04045, img / 12.92, ((img + 0.055) / 1.055) ** 2.4)

def linear_to_srgb(img):
    c = np.clip(img, 0.0, 1.0)
    return np.where(c <= 0.0031308, c * 12.92, 1.055 * (c ** (1.0 / 2.4)) - 0.055)

def comp_one(fnum):
    plate_path = WORK / "frames" / f"frame_{fnum-1:04d}.png"
    cg_path = WORK / f"lookdev_f{fnum:03d}.png"
    plate = cv2.imread(str(plate_path))
    cg = cv2.imread(str(cg_path), cv2.IMREAD_UNCHANGED)
    h, w = cg.shape[:2]
    plate = cv2.resize(plate, (w, h), interpolation=cv2.INTER_AREA)

    plate_f = srgb_to_linear(plate.astype(np.float32) / 255.0)
    if cg.dtype == np.uint16:
        cg_rgb = srgb_to_linear(cg[:, :, :3].astype(np.float32) / 65535.0)
        alpha = (cg[:, :, 3].astype(np.float32) / 65535.0)[..., None]
    else:
        cg_rgb = srgb_to_linear(cg[:, :, :3].astype(np.float32) / 255.0)
        alpha = (cg[:, :, 3].astype(np.float32) / 255.0)[..., None]

    # Subtle doorway light wrap preview
    blur_plate = cv2.GaussianBlur(plate_f, (0, 0), 14.0)
    blur_a = cv2.GaussianBlur(alpha[:, :, 0], (0, 0), 12.0)[..., None]
    edge_wrap = np.clip(alpha * (1.0 - blur_a) * 1.4, 0.0, 1.0)
    wrap_add = blur_plate * edge_wrap * 0.16

    comp_lin = (cg_rgb + wrap_add) * alpha + plate_f * (1.0 - alpha)
    comp = (linear_to_srgb(comp_lin) * 255.0).clip(0, 255).astype(np.uint8)
    cv2.putText(comp, f"LOOKDEV COMP f{fnum:03d}", (24, 44), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2, cv2.LINE_AA)
    return comp

panels = [comp_one(1), comp_one(62), comp_one(120)]
wire_path = WORK / "lookdev_wire_f062.png"
if wire_path.exists():
    w_cg = cv2.imread(str(wire_path), cv2.IMREAD_UNCHANGED)
    if w_cg.dtype == np.uint16:
        w_rgb = w_cg[:, :, :3].astype(np.float32) / 65535.0
        w_a = (w_cg[:, :, 3].astype(np.float32) / 65535.0)[..., None]
    else:
        w_rgb = w_cg[:, :, :3].astype(np.float32) / 255.0
        w_a = (w_cg[:, :, 3].astype(np.float32) / 255.0)[..., None]
    probe = (w_rgb * w_a * 255.0).clip(0, 255).astype(np.uint8)
    probe = cv2.resize(probe, (panels[0].shape[1], panels[0].shape[0]))
    cv2.putText(probe, "3D VIEW WIREFRAME f062", (24, 44), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2, cv2.LINE_AA)
else:
    probe = cv2.imread(str(WORK / "lookdev_hdri_probe.png"))
    probe = cv2.resize(probe, (panels[0].shape[1], panels[0].shape[0]))
    cv2.putText(probe, "HDRI ORIENTATION PROBE", (24, 44), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2, cv2.LINE_AA)

top = np.hstack([panels[0], panels[1]])
bot = np.hstack([panels[2], probe])
sheet = np.vstack([top, bot])
out = WORK / "lookdev_comp_sheet.jpg"
cv2.imwrite(str(out), sheet, [int(cv2.IMWRITE_JPEG_QUALITY), 92])
print(f"Saved {out}", flush=True)
