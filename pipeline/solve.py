"""Camera solve: COLMAP keyframes -> held-out refit -> per-frame PnP -> Three.js cameras.

1. pycolmap SIFT (CPU) + sequential matching on the keyframes.
2. Incremental mapping with a FIXED pinhole focal length from the shot settings.
   Free self-calibration diverged on rotation-heavy drone shots; the focal is an
   assumption and is recorded as one.
3. Remove every 5th landmark (id % 5 == 0) BEFORE the final bundle adjustment.
   Those landmarks never touch the camera and are used only to grade it (qa_camera).
4. tracking/measure.py: every in-between frame gets its own pose from
   bidirectional LK + RANSAC PnP + LM. No pose interpolation.
5. tracking/export_colmap.py: COLMAP optical axes -> Three/OpenGL matrixWorld.
"""
from __future__ import annotations

import re
import shutil
import sys

import numpy as np

from .common import ROOT, StepError, checkpoint, load_json, python, run, save_json, shot_dir

sys.path.insert(0, str(ROOT/"tracking"))


def frame_index(name):
    return int(re.search(r"(\d+)\.png$", name)[1])


def colmap_steps(cfg, d):
    import pycolmap
    w, h = cfg["plate"]["size"]
    focal = float(cfg["camera"]["focal_px"])
    threads = int(cfg["camera"].get("threads", 6))
    colmap = d/"colmap"
    db = colmap/"database.db"
    sparse = colmap/"sparse"
    if not (sparse/"done.json").is_file():
        if colmap.exists():
            raise StepError(f"{colmap} is incomplete from an earlier run. Keep it for inspection and use a new --work folder.")
        colmap.mkdir(parents=True)
        # Match WITHOUT a focal prior (uncalibrated F/H verification), exactly like the
        # validated runs. A wrong calibrated prior here classified quarry pairs badly
        # (held-out p95 3.5 px instead of 1.3 px). The fixed focal is set only for mapping.
        reader = pycolmap.ImageReaderOptions(camera_model="SIMPLE_RADIAL")
        extract = pycolmap.FeatureExtractionOptions()
        extract.use_gpu = False
        extract.num_threads = threads
        extract.max_image_size = max(w, h)
        extract.sift.max_num_features = 2500
        print("SIFT feature extraction (CPU)", flush=True)
        pycolmap.extract_features(db, d/"keyframes", camera_mode=pycolmap.CameraMode.SINGLE,
                                  reader_options=reader, extraction_options=extract, device=pycolmap.Device.cpu)
        match = pycolmap.FeatureMatchingOptions()
        match.use_gpu = False
        match.num_threads = threads
        pairing = pycolmap.SequentialPairingOptions()
        pairing.overlap = 8
        pairing.quadratic_overlap = True
        print("sequential matching", flush=True)
        pycolmap.match_sequential(db, matching_options=match, pairing_options=pairing, device=pycolmap.Device.cpu)
        with pycolmap.Database.open(db) as database:
            cameras = database.read_all_cameras()
            if len(cameras) != 1:
                raise StepError("Expected one shared camera for all keyframes")
            camera = cameras[0]
            camera.model = pycolmap.CameraModelId.SIMPLE_PINHOLE
            camera.params = [focal, w/2, h/2]
            database.update_camera(camera)
        o = pycolmap.IncrementalPipelineOptions()
        o.num_threads = threads
        o.random_seed = 42
        o.ba_refine_focal_length = False
        o.ba_refine_principal_point = False
        o.ba_refine_extra_params = False
        o.ba_global_max_num_iterations = 30
        o.ba_global_max_refinements = 2
        o.ba_global_frames_ratio = 1.3
        o.ba_global_points_ratio = 1.5
        o.mapper.init_min_tri_angle = 2
        o.mapper.init_max_forward_motion = .9999
        o.mapper.abs_pose_refine_focal_length = False
        o.mapper.abs_pose_refine_extra_params = False
        o.mapper.filter_max_reproj_error = 3
        o.constant_cameras = {1}
        o.max_runtime_seconds = 900
        sparse.mkdir()
        print("incremental mapping, fixed focal", focal, flush=True)
        models = pycolmap.incremental_mapping(db, d/"keyframes", sparse, options=o)
        save_json(sparse/"done.json", {str(k): r.summary() for k, r in models.items()})
    models = [(pycolmap.Reconstruction(f), f) for f in sparse.iterdir() if f.is_dir()]
    if not models:
        raise StepError("COLMAP found no model. Try a clip with more parallax/texture or another focal_px.")
    model, path = max(models, key=lambda m: m[0].num_reg_images())
    expected = len(list((d/"keyframes").glob("*.png")))
    if model.num_reg_images() != expected:
        raise StepError(f"Only {model.num_reg_images()}/{expected} keyframes registered. Do not export a partial camera; "
                        "shorten the clip, change focal_px or pick another shot.")
    return model, path


def withhold(model, d):
    """Delete 20% of landmarks, keep their observations as a grading ledger, refit."""
    import pycolmap
    ledger, removed = [], []
    for pid, point in list(model.points3D.items()):
        if pid % 5:
            continue
        removed.append(pid)
        track = []
        for e in point.track.elements:
            image = model.images[e.image_id]
            u, v = image.points2D[e.point2D_idx].xy
            track.append([frame_index(image.name), float(u-.5), float(v-.5)])
        if len(track) >= 8:
            ledger.append(dict(id=pid, observations=sorted(track)))
    for pid in removed:
        model.delete_point3D(pid)
    o = pycolmap.BundleAdjustmentOptions()
    o.refine_focal_length = False
    o.refine_extra_params = False
    o.refine_principal_point = False
    o.ceres.solver_options.max_num_iterations = 100
    o.ceres.solver_options.function_tolerance = 1e-7
    o.ceres.solver_options.num_threads = 6
    pycolmap.bundle_adjustment(model, options=o)
    fit = d/"colmap/fit"
    fit.mkdir(parents=True, exist_ok=True)
    model.write(fit)
    save_json(d/"qa/withheld-observations.json", ledger)
    return model, fit, dict(removed_points=len(removed), validation_tracks=len(ledger),
                            fit_points=model.num_points3D(), fit_error=model.compute_mean_reprojection_error())


def export_cameras(cfg, d, model, fit):
    from fractions import Fraction
    from scipy.spatial.transform import Rotation
    import export_colmap
    native = d/"native.npz"
    if not native.is_file():
        run([python(), ROOT/"tracking/measure.py", "native", fit, d/"frames", native])
    poses = np.load(native)["c2w"]
    dense = d/"colmap/dense-txt"
    dense.mkdir(parents=True, exist_ok=True)
    model.write_text(str(dense))
    camera_id = next(iter(model.cameras))
    lines = []
    for i, c2w in enumerate(poses):
        view = np.linalg.inv(c2w)
        qx, qy, qz, qw = Rotation.from_matrix(view[:3, :3]).as_quat()
        lines += [" ".join(map(str, [i+1, qw, qx, qy, qz, *view[:3, 3], camera_id]))+f" frame_{i:06}.png", ""]
    (dense/"images.txt").write_text("\n".join(lines)+"\n")
    (dense/"points3D.txt").unlink(missing_ok=True)
    data = export_colmap.export(dense, Fraction(cfg["plate"]["fps"]))
    data["intrinsics_note"] = "Assumed fixed pinhole focal length and zero distortion; not a lens calibration."
    save_json(d/"Camera_Three.json", data)
    return len(poses)


def main(cfg, work, args):
    d = shot_dir(cfg, work)
    if not (d/"keyframes").is_dir():
        raise StepError("Run prepare first.")
    if (d/"Camera_Three.json").is_file():
        print("already solved:", d/"Camera_Three.json", "(use a new --work folder to solve again)")
        return
    model, path = colmap_steps(cfg, d)
    camera = next(iter(model.cameras.values()))
    summary = dict(registered_keys=model.num_reg_images(), points=model.num_points3D(),
                   mean_reprojection_error=model.compute_mean_reprojection_error(),
                   camera_model=camera.model_name, camera_params=camera.params.tolist(),
                   intrinsics="Assumed fixed pinhole focal length; not physically calibrated", model=str(path))
    model, fit, refit = withhold(model, d)
    summary["heldout_refit"] = refit
    summary["dense_frames"] = export_cameras(cfg, d, model, fit)
    summary["dense_method"] = "tracking/measure.py bidirectional LK + RANSAC PnP + LM on every frame; no interpolation"
    save_json(d/"solve-summary.json", summary)
    checkpoint(d, "solved", next="python -m pipeline qa-camera "+cfg["name"])
    print({k: v for k, v in summary.items() if k != "heldout_refit"})
