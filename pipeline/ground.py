"""World align: fit the visible floor plane, put Y up, choose an origin and a scale.

The plane comes from reconstructed landmarks inside a floor region of the middle
frame (RANSAC, then SVD). The origin is where a chosen anchor pixel's ray hits
that plane. Scale is normalized so the middle camera is 1.6 units above the
floor. That is a convention, not a measured altitude: write it down as an assumption.
"""
from __future__ import annotations

import cv2
import numpy as np

from .common import StepError, checkpoint, frame_path, load_json, save_json, shot_dir
from .fetch import lighting_path


def main(cfg, work, args):
    import pycolmap
    d = shot_dir(cfg, work)
    g = cfg["ground"]
    s = load_json(d/"Camera_Three.json")
    W, H = s["width"], s["height"]
    r = pycolmap.Reconstruction(str(d/"colmap/fit"))
    by_index = {int(im.name[6:12]): im for im in r.images.values()}
    idx = min(by_index, key=lambda k: abs(k-len(s["frames"])//2))
    im = by_index[idx]
    c = np.array(s["frames"][idx]["matrixWorld"]).reshape(4, 4).T
    cloud, pixels = [], []
    for pt in im.points2D:
        x, y = pt.xy
        if g.get("roi"):
            x0, y0, x1, y1 = g["roi"]
            inside = x0*W < x < x1*W and y0*H < y < y1*H
        elif g.get("topdown"):
            inside = .3*H < y < .7*H and .25*W < x < .75*W
        else:
            inside = .63*H < y < .96*H and .18*W < x < .82*W
        if not pt.has_point3D() or not inside:
            continue
        v = r.points3D[pt.point3D_id]
        if v.error > 1.2 or v.track.length() < 8:
            continue
        cloud.append(v.xyz)
        pixels.append(pt.xy)
    cloud, pixels = np.array(cloud), np.array(pixels)
    if len(cloud) < 30:
        raise StepError(f"Only {len(cloud)} floor landmarks in the region. Adjust ground.roi in the shot settings.")
    depth = np.median(np.linalg.norm(cloud-c[:3, 3], axis=1))
    tol = depth*.008
    rng = np.random.default_rng(926)
    best = np.zeros(len(cloud), bool)
    for _ in range(800):
        u, v, w = cloud[rng.choice(len(cloud), 3, replace=False)]
        n = np.cross(v-u, w-u)
        if np.linalg.norm(n) < 1e-7:
            continue
        n /= np.linalg.norm(n)
        # A floor normal must roughly face the camera's up (or view axis, looking straight down).
        if abs(np.dot(n, c[:3, 2] if g.get("topdown") else c[:3, 1])) < (.9 if g.get("topdown") else .5):
            continue
        good = abs((cloud-u)@n) < tol
        if good.sum() > best.sum():
            best = good
    if best.sum() < 30 or best.mean() < .45:
        raise StepError("No reliable floor plane in that region. Pick a flatter, textured floor area (ground.roi).")
    center = cloud[best].mean(0)
    n = np.linalg.svd(cloud[best]-center)[2][-1]
    if np.dot(c[:3, 3]-center, n) < 0:
        n = -n
    u, v = np.array(g["anchor"])*[W, H]
    ray = c[:3, :3]@np.array([(u-s["cx"])/s["fx"], -(v-s["cy"])/s["fy"], -1.])
    t = np.dot(center-c[:3, 3], n)/np.dot(ray, n)
    if t <= 0:
        raise StepError("The anchor pixel's ray does not hit the floor in front of the camera. Move ground.anchor.")
    origin = c[:3, 3]+t*ray
    x = c[:3, 0]-n*np.dot(c[:3, 0], n)
    x /= np.linalg.norm(x)
    z = np.cross(x, n)
    R = np.stack([x, n, z])
    scale = 1.6/np.dot(c[:3, 3]-origin, n)
    out = dict(s, world="Floor aligned, Y up. Middle camera height normalized to 1.6; absolute scale not measured.")
    out["frames"] = []
    for row in s["frames"]:
        old = np.array(row["matrixWorld"]).reshape(4, 4).T
        new = np.eye(4)
        new[:3, :3] = R@old[:3, :3]
        new[:3, 3] = scale*(R@(old[:3, 3]-origin))
        out["frames"].append(dict(row, matrixWorld=new.T.ravel().tolist()))
    newc = np.array(out["frames"][idx]["matrixWorld"]).reshape(4, 4).T
    depth_anchor = -(np.linalg.inv(newc)@np.array([0, .65, 0, 1]))[2]
    size = g["width_fraction"]*W*depth_anchor/s["fx"]/1.12
    kind = cfg["kind"]
    layout = dict(kind=kind, water_scale=float(size), center=[0, float(size*.72) if kind == "water" else 0., 0],
                  base_radius=float(size*.29), anchor_pixel=[float(u), float(v)], reference_frame=idx,
                  plane_inliers=int(best.sum()), plane_candidates=len(cloud), plane_inlier_ratio=float(best.mean()),
                  plane_rms_world=float(np.sqrt(np.mean(((cloud[best]-center)@n)**2))*scale),
                  assumed_camera_height=1.6, metric_scale_measured=False,
                  transform=dict(rotation=R.tolist(), origin=origin.tolist(), scale=float(scale)),
                  hdri=str(lighting_path(cfg, work)), occlusion=g.get("occlusion", ""), shadow=g.get("shadow", ""))
    save_json(d/"Camera_Three_ground.json", out)
    save_json(d/"layout.json", layout)
    vis = cv2.imread(str(frame_path(d, idx)))
    for px in pixels[best]:
        cv2.circle(vis, tuple(np.round(px).astype(int)), 3, (30, 240, 20), -1)
    cv2.drawMarker(vis, (int(u), int(v)), (0, 0, 255), cv2.MARKER_CROSS, 30, 2)
    (d/"qa").mkdir(exist_ok=True)
    cv2.imwrite(str(d/"qa/ground-fit.jpg"), vis)
    checkpoint(d, "grounded", next="python -m pipeline simulate "+cfg["name"])
    print({k: layout[k] for k in ("kind", "water_scale", "plane_inliers", "plane_inlier_ratio", "plane_rms_world")})
    print("Look at qa/ground-fit.jpg: green dots must sit on the floor, the red cross where the object stands.")
