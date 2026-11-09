"""Grade the camera with landmarks it never saw.

Each held-out landmark is triangulated from its first and last SIFT observation,
then reprojected into every frame in between. Short gaps are also checked with
bidirectional LK samples. Nothing is rejected for having a large error.
Gate: >=40 tracks, every interior frame >=15 observations, p95 < 2 px,
max < 8 px, frame-to-frame movement p95 < 1 px (at the working resolution).
"""
from __future__ import annotations

import sys

import cv2
import numpy as np

from .common import ROOT, StepError, checkpoint, frame_path, load_json, save_json, shot_dir

sys.path.insert(0, str(ROOT/"tracking"))


def main(cfg, work, args):
    from measure import flow
    d = shot_dir(cfg, work)
    s = load_json(d/"Camera_Three.json")
    K = np.array([[s["fx"], 0, s["cx"]-.5], [0, s["fy"], s["cy"]-.5], [0, 0, 1.]])
    cams = np.array([np.array(f["matrixWorld"]).reshape(4, 4).T for f in s["frames"]])
    views = np.array([np.diag([1, -1, -1, 1])@np.linalg.inv(c) for c in cams])
    Ps = K@views[:, :3]
    ledger = load_json(d/"qa/withheld-observations.json")
    N = len(Ps)
    imgs = [cv2.imread(str(frame_path(d, i)), 0) for i in range(N)]
    records, key_errors, xyzs, by_pair, rejected = [], [], {}, {}, 0
    for tr in ledger:
        rows = np.array(tr["observations"])
        i, j = int(rows[0, 0]), int(rows[-1, 0])
        h = cv2.triangulatePoints(Ps[i], Ps[j], rows[0, 1:].reshape(2, 1), rows[-1, 1:].reshape(2, 1))
        X = (h[:3]/h[3]).ravel()
        if not np.isfinite(X).all() or any((views[k, :3, :3]@X+views[k, :3, 3])[2] <= 0 for k in [i, j]):
            rejected += 1
            continue
        xyzs[tr["id"]] = X
        for k, u, v in rows[1:-1]:
            k = int(k)
            q = Ps[k]@np.r_[X, 1]
            pred = q[:2]/q[2]
            err = float(np.linalg.norm(pred-[u, v]))
            key_errors.append(err)
            records.append([k, tr["id"], u, v, *pred, err, 0])
        for left, right in zip(rows[:-1], rows[1:]):
            l, r = int(left[0]), int(right[0])
            if 2 <= r-l <= 6:
                by_pair.setdefault((l, r), []).append((tr["id"], left[1:], right[1:]))
    for (l, r), points in by_pair.items():
        ids = np.array([x[0] for x in points])
        uv0 = np.array([x[1] for x in points], dtype="f")
        uv1 = np.array([x[2] for x in points], dtype="f")
        for n in range(l+1, r):
            x, ok = flow(imgs[l], imgs[n], uv0, win=23, iterations=30, epsilon=.005, tolerance=.5, border=8)
            y, good = flow(imgs[r], imgs[n], uv1, win=23, iterations=30, epsilon=.005, tolerance=.5, border=8)
            keep = ok & good & (np.linalg.norm(x-y, axis=1) < .7)
            for k in np.where(keep)[0]:
                obs = (x[k]*(r-n)+y[k]*(n-l))/(r-l)
                q = Ps[n]@np.r_[xyzs[int(ids[k])], 1]
                pred = q[:2]/q[2]
                records.append([n, int(ids[k]), *obs, *pred, float(np.linalg.norm(pred-obs)), 1])
    if not records:
        raise StepError("No held-out observations; the solve is too sparse to grade.")
    arr = np.array(records)
    errs = arr[:, 6]
    dense = arr[arr[:, 7] == 1, 6]
    movement = []
    for pid in np.unique(arr[:, 1]):
        rows = arr[arr[:, 1] == pid]
        rows = rows[np.argsort(rows[:, 0])]
        step = np.diff(rows[:, 0])
        err = np.linalg.norm(np.diff(rows[:, 4:6], axis=0)-np.diff(rows[:, 2:4], axis=0), axis=1)
        movement.extend(err[step == 1])
    coverage = {str(i): int(np.sum(arr[:, 0] == i)) for i in range(1, N-1)}
    report = dict(
        method="20% of landmarks removed before the final bundle adjustment and PnP; endpoint triangulation, "
               "interior SIFT and short-span bidirectional LK observations; no outlier rejection.",
        validation_tracks=len(xyzs), invalid_depth_tracks=rejected, observations=len(arr),
        key_observations=len(key_errors), dense_observations=len(dense),
        median_px=float(np.median(errs)), p95_px=float(np.percentile(errs, 95)), max_px=float(errs.max()),
        rms_px=float(np.sqrt(np.mean(errs**2))), key_p95_px=float(np.percentile(key_errors, 95)) if key_errors else None,
        dense_p95_px=float(np.percentile(dense, 95)) if len(dense) else None,
        movement_p95_px=float(np.percentile(movement, 95)) if movement else None, coverage=coverage,
        thresholds=dict(min_tracks=40, min_interior_frame_observations=15, p95_px=2., max_px=8., movement_p95_px=1.),
        resolution=[s["width"], s["height"]],
        limitations="Pinhole lens and absolute scale are assumptions; this is relative geometry, not lens calibration.")
    report["pass"] = bool(len(xyzs) >= 40 and min(coverage.values()) >= 15 and report["p95_px"] < 2
                          and report["max_px"] < 8 and (report["movement_p95_px"] or 9) < 1)
    save_json(d/"qa/acceptance-camera.json", report)
    np.save(d/"qa/withheld-validation.npy", arr)
    print({k: v for k, v in report.items() if k not in ("coverage", "method", "limitations")})
    if not report["pass"]:
        checkpoint(d, "camera_failed", next="Inspect qa/acceptance-camera.json; change focal_px or the clip.")
        raise StepError("Camera QA failed. Do not composite on this camera.")
    checkpoint(d, "camera_accepted", next="python -m pipeline ground "+cfg["name"])
