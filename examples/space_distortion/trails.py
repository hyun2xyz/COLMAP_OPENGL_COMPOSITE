"""Lift the MediaPipe limbs into the solved scene so the distortion trails stay where the arm passed, in the air.

    python trails.py [piece_dir]        needs work/pose.json (track_pose.py) and work/scene.json (slv.py solve + plane)
out: work/limbs.npz   P[N,33,3] landmark positions in PLANE coordinates (metres, z up from the fitted floor), vis[N,33]
     work/check_limbs.jpg   first / middle / last frame: projected skeleton, floor grid and the lifted foot point

Depth: the feet stand on the floor, so the ray through the lowest foot pixel hits the fitted plane at the dancer's
distance. Every joint keeps its own 2D pixel (no slide against the picture) and gets that distance plus MediaPipe's
relative world depth. Height above the floor therefore follows the assumed scale (plane.cam_h); the script prints the
head height so the scale can be checked against a ~1.7 m dancer."""
import json, sys
from pathlib import Path
import numpy as np, cv2
from scipy.signal import savgol_filter

here = Path(__file__).resolve().parent
sys.path.insert(0, str(here.parent / "shader_live" / "pipeline"))
from slv import load, work, frame, scene, overlay   # noqa: E402

c = load(sys.argv[1] if len(sys.argv) > 1 else here / "piece"); S = scene(c); W, H, N = c["W"], c["H"], c["N"]; f = S["f"]
pose = json.loads(work(c, "pose.json").read_text())
img = np.zeros((N, 33, 2)); vis = np.zeros((N, 33)); wz = np.zeros((N, 33)); ok = np.zeros(N, bool)
for r in pose:
    if r["ok"]:
        i = r["frame"]; a = np.array(r["img"]); img[i] = a[:, :2]; vis[i] = a[:, 3]; wz[i] = np.array(r["world"])[:, 2]; ok[i] = True
idx = np.flatnonzero(ok)
for arr in (img, wz, vis):                                                      # fill frames with no detection from the neighbours
    flat = arr.reshape(N, -1)
    for k in range(flat.shape[1]): flat[:, k] = np.interp(np.arange(N), idx, flat[idx, k])
img = savgol_filter(img, 7, 2, axis=0); wz = savgol_filter(wz, 9, 2, axis=0)   # pose jitter would shake the air

def ray(i, px):
    d = S["rot"][i] @ np.array([(px[0] - W / 2) / f, (px[1] - H / 2) / f, 1.0]); return d
depth = np.zeros(N); foot = np.zeros((N, 3))
for i in range(N):
    j = max((29, 30, 31, 32), key=lambda q: img[i, q, 1])                       # lowest heel / toe = standing contact
    ro = S["cam"][i]; rd = ray(i, img[i, j]); t = -ro[2] / rd[2] if rd[2] < 0 else np.nan
    foot[i] = ro + t * rd; depth[i] = (S["rot"][i].T @ (foot[i] - ro))[2]
bad = ~np.isfinite(depth) | (depth <= 0)
if bad.all(): sys.exit("TRAILS FAILED: no foot ray hits the floor plane; check work/check_plane.jpg")
depth[bad] = np.interp(np.flatnonzero(bad), np.flatnonzero(~bad), depth[~bad])
depth = savgol_filter(depth, 15, 2)
P = np.zeros((N, 33, 3))
for i in range(N):
    z = depth[i] + (wz[i] - (wz[i, 31] + wz[i, 32]) / 2)                       # relative to the feet; MediaPipe z: metres, + away from the camera
    q = np.stack([(img[i, :, 0] - W / 2) / f, (img[i, :, 1] - H / 2) / f, np.ones(33)], 1) * z[:, None]
    P[i] = S["cam"][i] + q @ S["rot"][i].T
head = P[:, 0, 2]; print("dancer distance %.2f-%.2f m, nose height median %.2f m (expect ~1.6 for a 1.7 m dancer), camera height %.2f m"
                         % (depth.min(), depth.max(), np.median(head), np.median(S["cam"][:, 2])))
np.savez_compressed(work(c, "limbs.npz"), P=P.astype(np.float32), vis=vis.astype(np.float32), img=img.astype(np.float32), depth=depth.astype(np.float32))

BONES = [(11, 13), (13, 15), (15, 19), (12, 14), (14, 16), (16, 20), (11, 12), (11, 23), (12, 24), (23, 24), (23, 25), (25, 27), (24, 26), (26, 28)]
rows = []
for i in (0, N // 2, N - 1):
    im = overlay(c, i, None, pts=False)
    def pr(X):
        q = S["rot"][i].T @ (X - S["cam"][i]); return int(f * q[0] / q[2] + W / 2), int(f * q[1] / q[2] + H / 2)
    for a, b in BONES: cv2.line(im, pr(P[i, a]), pr(P[i, b]), (255, 0, 255), 3, cv2.LINE_AA)
    for k in range(1, 15):                                                     # where the wrists were in the last 14 frames, seen from THIS camera
        for j in (15, 16):
            if i - k >= 0: cv2.circle(im, pr(P[i - k, j]), 5, (0, 255, 255), -1, cv2.LINE_AA)
    cv2.drawMarker(im, pr(foot[i]), (0, 0, 255), cv2.MARKER_CROSS, 40, 3)
    rows.append(cv2.resize(im, (960, 540)))
cv2.imwrite(str(work(c, "check_limbs.jpg")), np.vstack(rows), [cv2.IMWRITE_JPEG_QUALITY, 88]); print("->", work(c, "check_limbs.jpg"))
