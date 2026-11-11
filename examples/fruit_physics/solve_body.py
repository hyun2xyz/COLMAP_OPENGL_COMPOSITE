"""Camera-space 3D body from MediaPipe: per frame solvePnP(world landmarks [m] -> image landmarks [px]).
Camera = OpenCV convention (x right, y down, z forward), principal point at the image centre.
The focal length is unknown, so a sweep reports the reprojection error per f; usage:
  python solve_body.py sweep
  python solve_body.py <f_px>     -> work/body_cam.json (33 joints per frame in metres, smoothed)"""
import json, sys
from pathlib import Path
import numpy as np, cv2
from scipy.signal import savgol_filter

root = Path(__file__).parent
P = json.loads((root / "work" / "pose.json").read_text())
W, H = 720, 1280
CORE = [0, 2, 5, 7, 8, 11, 12, 13, 14, 15, 16, 23, 24]   # nose, eyes, ears, shoulders, elbows, wrists, hips

def solve(f):
    K = np.array([[f, 0, W / 2], [0, f, H / 2], [0, 0, 1]], float)
    errs, Rs, ts = [], [], []
    prev = None
    for r in P:
        img = np.array(r["img"]); wl = np.array(r["world"], float)
        idx = [k for k in CORE if img[k, 3] > 0.5]
        obj = wl[idx]; uv = img[idx, :2] * [W, H]
        if prev is None:
            ok, rv, tv = cv2.solvePnP(obj, uv, K, None, flags=cv2.SOLVEPNP_SQPNP)
        else:
            ok, rv, tv = cv2.solvePnP(obj, uv, K, None, prev[0].copy(), prev[1].copy(), True, cv2.SOLVEPNP_ITERATIVE)
        prev = (rv, tv)
        pr, _ = cv2.projectPoints(obj, rv, tv, K, None)
        errs.append(np.linalg.norm(pr.reshape(-1, 2) - uv, axis=1).mean())
        Rs.append(cv2.Rodrigues(rv)[0]); ts.append(tv.ravel())
    return np.array(errs), np.array(Rs), np.array(ts)

if sys.argv[1] == "sweep":
    for f in [550, 650, 750, 850, 950, 1100, 1300, 1600]:
        e, R, t = solve(f)
        print(f"f={f:5d}  hfov={np.degrees(2*np.arctan(W/2/f)):5.1f}  reproj mean {e.mean():6.2f}px  median depth {np.median(t[:,2]):5.2f} m  depth std {t[:,2].std():.3f}")
    sys.exit()

f = float(sys.argv[1])
e, R, t = solve(f)
# camera-space joints, then temporal smoothing (pose jitter would kick the fruits)
J = np.array([(R[i] @ np.array(P[i]["world"]).T).T + t[i] for i in range(len(P))])   # (N,33,3)
# keep the depth from the metric 3D pose but put every joint exactly on the ray of its 2D landmark,
# so the invisible body lines up with the picture (PnP alone is off by ~18 px)
UV = np.array([[p[:2] for p in r["img"]] for r in P]) * [W, H]
J[:, :, 0] = (UV[:, :, 0] - W / 2) / f * J[:, :, 2]
J[:, :, 1] = (UV[:, :, 1] - H / 2) / f * J[:, :, 2]
Js = savgol_filter(J, 9, 2, axis=0)
vis = np.array([[p[3] for p in r["img"]] for r in P])
(root / "work" / "body_cam.json").write_text(json.dumps({
    "f": f, "W": W, "H": H, "reproj_mean_px": float(e.mean()),
    "joints": Js.round(5).tolist(), "visibility": vis.round(3).tolist()}))
print("saved; reproj", e.mean().round(2), "px; hip-centre depth range", J[:, [23, 24], 2].mean(1).min().round(2), J[:, [23, 24], 2].mean(1).max().round(2))
