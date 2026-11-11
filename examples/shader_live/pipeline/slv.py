"""Shader Live Series: tracked live-action plate + world-locked GLSL effect.

    python slv.py prep  <piece_dir>     frames from the source clip            -> work/frames, work/meta.json
    python slv.py solve <piece_dir>     COLMAP camera (pycolmap, half res)      -> work/camera.npz
    python slv.py plane <piece_dir>     plane fit + plane frame + check sheet   -> work/scene.json, work/check_plane.jpg
    python slv.py track <piece_dir>     tracking pass for the breakdown         -> work/track/%04d.jpg

piece.json (all pixel values are in the full-res frame, origin top-left):
  source, start, frames, step     source video (relative to the piece folder), first source frame, frame count, take every step-th frame
  hfov, refine_focal              assumed horizontal field of view in degrees (stock clips carry no lens data); let COLMAP refine it
  plane.frame, plane.roi          the plane is fitted to the SfM points that project into roi [x0,y0,x1,y1] in that frame
  plane.anchor, plane.xdir        pixel that becomes the plane origin, pixel that gives the +x direction on the plane
  plane.cam_h                     height of that frame's camera above the plane in metres (sets the scale: COLMAP has none)
  plane.offset                    optional: shift the plane along its normal by this many metres after the fit
  sun                             {"az": deg, "el": deg} direction TO the sun in plane coordinates (az from +x towards +y), or {"px": [x, y], "frame": k}

World = the reference camera (x right, y down, z forward). Plane coordinates: origin at the anchor, x along xdir,
z = plane normal towards the camera, y = z cross x; metres. Everything the shader gets is in plane coordinates."""
import json, math, shutil, subprocess, sys, tempfile, time
from pathlib import Path
import numpy as np, cv2

def load(d):
    d = Path(d).resolve(); c = json.loads((d / "piece.json").read_text(encoding="utf-8")); c["dir"] = d
    m = d / "work" / "meta.json"
    if m.exists(): c.update(json.loads(m.read_text()))
    return c
def work(c, *p): return c["dir"].joinpath("work", *p)
def frame(c, i): return cv2.imread(str(work(c, "frames", f"{i:04d}.jpg")))
def focal0(c): return c["W"] / 2 / math.tan(math.radians(c.get("hfov", 73.7)) / 2)

def prep(c):
    src = (c["dir"] / c["source"]).resolve(); out = work(c, "frames"); shutil.rmtree(out, ignore_errors=True); out.mkdir(parents=True)
    pr = subprocess.run(["ffprobe", "-v", "error", "-select_streams", "v:0", "-show_entries", "stream=width,height,r_frame_rate", "-of", "csv=p=0", str(src)],
                        capture_output=True, text=True).stdout.strip().split(",")
    num, den = pr[2].split("/"); fps = float(num) / float(den); S, st, N = c.get("start", 0), c.get("step", 1), c["frames"]
    vf = f"select='gte(n\\,{S})*not(mod(n-{S}\\,{st}))',setpts=N/FRAME_RATE/TB,scale=1920:1080:flags=lanczos"
    subprocess.run(["ffmpeg", "-v", "error", "-y", "-i", str(src), "-vf", vf, "-frames:v", str(N), "-q:v", "1", "-qmin", "1", str(out / "%04d.jpg")], check=True)
    for k, p in enumerate(sorted(out.glob("*.jpg"))): p.rename(out / f"t{k:04d}.jpg")                 # ffmpeg numbers from 1
    for p in sorted(out.glob("t*.jpg")): p.rename(out / p.name[1:])
    n = len(list(out.glob("*.jpg"))); im = cv2.imread(str(out / "0000.jpg"))
    meta = {"W": im.shape[1], "H": im.shape[0], "N": n, "fps": fps / st, "fps_num": int(num), "fps_den": int(den) * st, "src_fps": fps}
    work(c, "meta.json").write_text(json.dumps(meta)); print(c["dir"].name, "prep", meta)

def solve(c):
    import pycolmap
    from scipy.signal import savgol_filter
    from scipy.spatial.transform import Rotation as Rot, Slerp
    W, H, N = c["W"], c["H"], c["N"]; s = 0.5; w2, h2 = int(W * s), int(H * s); f = focal0(c)
    tmp = Path(tempfile.gettempdir()) / "slv" / c["dir"].name; shutil.rmtree(tmp, ignore_errors=True)   # short ASCII path: COLMAP dislikes long ones
    (tmp / "img").mkdir(parents=True); (tmp / "sfm").mkdir(); (tmp / "mask").mkdir()
    mr = c.get("mask_roi")                         # features only inside [x0,y0,x1,y1] (keeps sky, sun flare, water out of the solve)
    for i in range(N):
        if mr:
            m = np.zeros((h2, w2), np.uint8); m[int(mr[1] * s):int(mr[3] * s), int(mr[0] * s):int(mr[2] * s)] = 255; cv2.imwrite(str(tmp / "mask" / f"{i:04d}.jpg.png"), m)
        cv2.imwrite(str(tmp / "img" / f"{i:04d}.jpg"), cv2.resize(frame(c, i), (w2, h2), interpolation=cv2.INTER_AREA), [cv2.IMWRITE_JPEG_QUALITY, 96])
    db = tmp / "sfm" / "database.db"; t0 = time.time()
    reader = pycolmap.ImageReaderOptions(); reader.camera_model = "SIMPLE_PINHOLE"; reader.camera_params = f"{f * s},{w2 / 2},{h2 / 2}"
    if mr: reader.mask_path = str(tmp / "mask")
    ext = pycolmap.FeatureExtractionOptions(); ext.sift.max_num_features = int(c.get("sift", 5000))
    pycolmap.extract_features(db, tmp / "img", camera_mode=pycolmap.CameraMode.SINGLE, reader_options=reader, extraction_options=ext)
    pair = pycolmap.SequentialPairingOptions(); pair.overlap = int(c.get("overlap", 10)); pair.loop_detection = False
    pycolmap.match_sequential(db, pairing_options=pair)
    o = pycolmap.IncrementalPipelineOptions(); o.ba_refine_focal_length = bool(c.get("refine_focal", False))
    o.ba_refine_principal_point = False; o.ba_refine_extra_params = False
    rec = None
    for tri in ([float(c["init_tri"])] if "init_tri" in c else [16.0, 6.0, 2.0]):       # slow drone moves: little parallax between neighbours
        o.mapper.init_min_tri_angle = tri
        if tri < 16: o.mapper.init_max_forward_motion = 1.0; o.mapper.init_min_num_inliers = 60
        for d in (tmp / "sfm").glob("[0-9]*"): shutil.rmtree(d, ignore_errors=True)
        recs = pycolmap.incremental_mapping(db, tmp / "img", tmp / "sfm", options=o)
        if recs:
            rec = max(recs.values(), key=lambda r: r.num_reg_images())
            if rec.num_reg_images() >= 0.6 * N: break
        print(c["dir"].name, "init_min_tri_angle", tri, "->", rec.num_reg_images() if rec else 0, "registered; retrying", flush=True)
    if rec is None or rec.num_reg_images() < 0.3 * N: print(c["dir"].name, "SOLVE FAILED: no usable reconstruction"); sys.exit(1)
    cam = list(rec.cameras.values())[0]; f_half = float(cam.params[0])
    Rs, Ts, ok = np.zeros((N, 3, 3)), np.zeros((N, 3)), np.zeros(N, bool)
    for im in rec.images.values():
        k = int(Path(im.name).stem); cfw = im.cam_from_world() if callable(im.cam_from_world) else im.cam_from_world
        Rs[k] = cfw.rotation.matrix(); Ts[k] = np.asarray(cfw.translation); ok[k] = True
    idx = np.flatnonzero(ok); ref = idx[0]; R0, t0_ = Rs[ref], Ts[ref]
    Rw = np.array([Rs[k] @ R0.T if ok[k] else np.eye(3) for k in range(N)])                       # world (= ref camera) -> camera k
    Cw = np.array([-(Rw[k].T @ (Ts[k] - Rw[k] @ t0_)) if ok[k] else np.zeros(3) for k in range(N)])
    ti = np.clip(np.arange(N), idx[0], idx[-1]); Rw = Slerp(idx, Rot.from_matrix(Rw[idx]))(ti).as_matrix()
    Cw = np.stack([np.interp(ti, idx, Cw[idx, a]) for a in range(3)], 1)
    raw_R, raw_C = Rw.copy(), Cw.copy(); win = int(c.get("smooth", 9))
    if N > win and win >= 5:
        rv = Rot.from_matrix(Rw).as_rotvec(); rv = savgol_filter(rv, win, 2, axis=0); Rw = Rot.from_rotvec(rv).as_matrix(); Cw = savgol_filter(Cw, win, 2, axis=0)
    P = [(R0 @ p.xyz + t0_, p.color, p.error, p.track.length()) for p in rec.points3D.values()]
    P = [q for q in P if q[3] >= 3 and q[2] < 2.0]
    np.savez_compressed(work(c, "camera.npz"), R=Rw, C=Cw, R_raw=raw_R, C_raw=raw_C, f=f_half / s, ok=ok,
                        pts=np.array([q[0] for q in P], np.float32), rgb=np.array([q[1] for q in P], np.uint8))
    info = {"registered": int(ok.sum()), "N": N, "points": len(P), "reproj_px_half_res": float(rec.compute_mean_reprojection_error()),
            "f_px": f_half / s, "f_init_px": f, "refine_focal": bool(c.get("refine_focal", False)), "seconds": round(time.time() - t0)}
    work(c, "solve.json").write_text(json.dumps(info)); print(c["dir"].name, "solve", info)

def solve_h(c):
    """Planar scenes (top-down drone shots): COLMAP cannot initialise on a plane, so the camera comes from homographies.
    Frame 0 is ASSUMED to look at the plane with a known tilt from straight down (piece.json "tilt", degrees, default 0)
    from 1 unit above it; each later frame's pose is the one that reproduces its plane homography to frame 0.
    Exact for everything ON the plane; things above it inherit the error of the assumed tilt and focal length."""
    from scipy.signal import savgol_filter
    from scipy.spatial.transform import Rotation as Rot
    W, H, N = c["W"], c["H"], c["N"]; f = focal0(c); K = np.array([[f, 0, W / 2], [0, f, H / 2], [0, 0, 1.0]]); Ki = np.linalg.inv(K)
    th = math.radians(c.get("tilt", 0.0)); n = np.array([0.0, -math.sin(th), -math.cos(th)]); d = -1.0          # plane n.X = d in camera-0 coords
    u1 = np.array([1.0, 0, 0]); u2 = np.cross(n, u1)
    sift = cv2.SIFT_create(nfeatures=int(c.get("sift", 6000))); bf = cv2.BFMatcher(cv2.NORM_L2)
    mr = c.get("mask_roi"); fm = None
    if mr: fm = np.zeros((H, W), np.uint8); fm[int(mr[1]):int(mr[3]), int(mr[0]):int(mr[2])] = 255
    def feat(i):
        g = cv2.cvtColor(frame(c, i), cv2.COLOR_BGR2GRAY); return sift.detectAndCompute(g, fm)
    def homog(a, b):
        m = [x for x, y in bf.knnMatch(a[1], b[1], k=2) if x.distance < 0.75 * y.distance]
        if len(m) < 30: return None, 0
        pa = np.float32([a[0][x.queryIdx].pt for x in m]); pb = np.float32([b[0][x.trainIdx].pt for x in m])
        Hm, inl = cv2.findHomography(pa, pb, cv2.USAC_MAGSAC, 2.0); return Hm, int(inl.sum()) if inl is not None else 0
    key = feat(0); Gk = np.eye(3); G = [np.eye(3)]; nin = [0]; f0 = key; prev = key; Gprev = np.eye(3)
    for i in range(1, N):
        cur = feat(i); Hm, k = homog(key, cur)
        if Hm is None or k < 0.35 * max(nin[1:] or [k]) or k < 150:                      # the key frame is drifting out of view: re-key on the previous frame
            key, Gk = prev, Gprev; Hm, k = homog(key, cur)
            if Hm is None: print("SOLVE_H FAILED at frame", i); sys.exit(1)
        Gi = Hm @ Gk; G.append(Gi / Gi[2, 2]); nin.append(k); prev, Gprev = cur, Gi
    Rs, Cs = [], []
    for Gi in G:
        A = Ki @ Gi @ K; A /= np.linalg.svd(A, compute_uv=False)[1]
        if (A @ np.array([0, 0, 1.0]))[2] < 0: A = -A
        r1, r2 = A @ u1, A @ u2; Rm = np.stack([r1, r2, np.cross(r1, r2)], 1) @ np.stack([u1, u2, n], 0)
        U_, _, Vt = np.linalg.svd(Rm); Rm = U_ @ np.diag([1, 1, np.linalg.det(U_ @ Vt)]) @ Vt
        t = d * (A @ n - Rm @ n); Rs.append(Rm); Cs.append(-Rm.T @ t)
    Rw, Cw = np.array(Rs), np.array(Cs); raw_R, raw_C = Rw.copy(), Cw.copy(); win = int(c.get("smooth", 9))
    if N > win and win >= 5:
        rv = savgol_filter(Rot.from_matrix(Rw).as_rotvec(), win, 2, axis=0); Rw = Rot.from_rotvec(rv).as_matrix(); Cw = savgol_filter(Cw, win, 2, axis=0)
    im0 = frame(c, 0); P, col = [], []
    for kp in f0[0][::2]:
        r = Ki @ np.array([kp.pt[0], kp.pt[1], 1.0]); P.append(r * (d / (n @ r))); col.append(im0[int(kp.pt[1]), int(kp.pt[0]), ::-1])
    np.savez_compressed(work(c, "camera.npz"), R=Rw, C=Cw, R_raw=raw_R, C_raw=raw_C, f=f, ok=np.ones(N, bool), pts=np.array(P, np.float32), rgb=np.array(col, np.uint8))
    info = {"solver": "homography", "tilt_deg_assumed": c.get("tilt", 0.0), "N": N, "min_inliers": int(min(nin[1:])), "median_inliers": int(np.median(nin[1:])), "f_px": f,
            "camera_travel_in_heights": float(np.linalg.norm(Cw[-1] - Cw[0]))}
    work(c, "solve.json").write_text(json.dumps(info)); print(c["dir"].name, "solve_h", info)

def project(P, R, C, f, W, H):
    X = (P - C) @ R.T; z = X[:, 2]; return np.stack([f * X[:, 0] / z + W / 2, f * X[:, 1] / z + H / 2], 1), z
def ray(px, R, f, W, H):
    d = R.T @ np.array([(px[0] - W / 2) / f, (px[1] - H / 2) / f, 1.0]); return d / np.linalg.norm(d)

def scene(c):
    s = json.loads(work(c, "scene.json").read_text()); return {k: (np.array(v) if isinstance(v, list) else v) for k, v in s.items()}

def plane(c):
    cam = np.load(work(c, "camera.npz")); R, C, f, P = cam["R"], cam["C"], float(cam["f"]), cam["pts"].astype(np.float64)
    W, H, N = c["W"], c["H"], c["N"]; pl = c["plane"]; k = pl.get("frame", 0)
    uv, z = project(P, R[k], C[k], f, W, H); x0, y0, x1, y1 = pl.get("roi", [0, 0, W, H])
    m = (z > 0) & (uv[:, 0] > x0) & (uv[:, 0] < x1) & (uv[:, 1] > y0) & (uv[:, 1] < y1)
    if "pixels" in pl:                                   # only the SfM points next to these pixels (a shoreline, the foot of a wall, ...)
        m = np.zeros(len(P), bool); r = pl.get("radius", 30)
        for px in pl["pixels"]: m |= (z > 0) & (np.hypot(uv[:, 0] - px[0], uv[:, 1] - px[1]) < r)
    Q = P[m]
    if len(Q) < (8 if "pixels" in pl else 30): print("PLANE FAILED: only", len(Q), "points selected"); sys.exit(1)
    thr = pl.get("thresh", 0.004) * float(np.median(z[m])); rng = np.random.default_rng(0); best = (0, None)
    if pl.get("level_travel"):
        # open water has no features. Assume level flight and a gimbal with no roll: 'up' is perpendicular to the camera's travel,
        # inside the reference camera's vertical plane. The surface height is a low percentile of ALL point heights (nothing is below the water).
        v = R[k] @ (C[-1] - C[0]); pp = math.atan2(-v[1], v[2]) if abs(pl.get("pitch", 999)) > 90 else math.radians(pl["pitch"])
        nc = np.array([0.0, -math.cos(pp), math.sin(pp)]) if False else np.array([0.0, -math.cos(pp), -math.sin(pp)])
        Pc = (P[z > 0] - C[k]) @ R[k].T; h = Pc @ nc; d0 = float(np.percentile(h, pl.get("pct", 2.0)))
        n = R[k].T @ nc; d = float(d0 + n @ C[k]); inl = np.abs(h - d0) < thr * 3; best = (int(inl.sum()), inl); Q = P[z > 0]; Qi = Q[inl]; cen = Qi.mean(0) if len(Qi) else Q.mean(0)
        print("level plane from travel: pitch %.1f deg below the flight direction, travel in camera coords %s" % (math.degrees(pp), np.round(v, 2)))
    elif pl.get("level"):
        # a level surface seen by a gimbal camera with no roll: the normal is (0, -cos p, -sin p) in the reference camera, p = pitch
        # below the horizon. Two unknowns (p and the offset) instead of three, so a thin band of points is enough.
        Qc = (Q - C[k]) @ R[k].T
        for pdeg in np.arange(0.0, 90.01, 0.1):
            pp = math.radians(pdeg); nc = np.array([0.0, -math.cos(pp), -math.sin(pp)]); h = Qc @ nc
            for d0 in np.percentile(h, [10, 25, 40, 50, 60, 75]):
                inl = np.abs(h - d0) < thr
                if inl.sum() > best[0]: best = (int(inl.sum()), inl, nc, float(np.median(h[inl])))
        n = R[k].T @ best[2]; d = float(best[3] + n @ C[k]); Qi = Q[best[1]]; cen = Qi.mean(0)
        print("level plane: pitch %.1f deg below the horizon" % math.degrees(math.atan2(-best[2][2], -best[2][1])))
    else:
        for _ in range(3000):
            q = Q[rng.choice(len(Q), 3, replace=False)]; n = np.cross(q[1] - q[0], q[2] - q[0]); nn = np.linalg.norm(n)
            if nn < 1e-12: continue
            n /= nn; inl = np.abs((Q - q[0]) @ n) < thr
            if inl.sum() > best[0]: best = (int(inl.sum()), inl)
        Qi = Q[best[1]]; cen = Qi.mean(0); n = np.linalg.svd(Qi - cen, full_matrices=False)[2][2]
        n = n if n @ (C[k] - cen) > 0 else -n; d = float(n @ cen)
    def hit(px):
        r = ray(px, R[k], f, W, H); t = (d - n @ C[k]) / (n @ r); return C[k] + t * r
    scale = pl.get("cam_h", 1.6) / float(n @ C[k] - d)
    d += pl.get("offset", 0.0) / scale                                    # shift along the normal, metres
    O = hit(pl.get("anchor", [W / 2, H * 0.7])); X = hit(pl.get("xdir", [W * 0.75, H * 0.7])) - O; X -= (X @ n) * n; X /= np.linalg.norm(X)
    Y = np.cross(n, X); M = np.stack([X, Y, n])
    sun = c.get("sun", {"az": 45, "el": 45})
    if "px" in sun:
        sd = M @ ray(sun["px"], R[sun.get("frame", 0)], f, W, H)
    else:
        a, e = math.radians(sun["az"]), math.radians(sun["el"]); sd = np.array([math.cos(e) * math.cos(a), math.cos(e) * math.sin(a), math.sin(e)])
    Cp = scale * (C - O) @ M.T; Rcp = np.einsum("ij,nkj->nik", M, R)       # camera -> plane rotation per frame (M @ R^T)
    out = {"O": O.tolist(), "M": M.tolist(), "scale": scale, "f": f, "sun": (sd / np.linalg.norm(sd)).tolist(), "inliers": best[0], "roi_points": int(len(Q)),
           "cam": Cp.tolist(), "rot": Rcp.tolist(), "cam_h_first_last": [float(Cp[0, 2]), float(Cp[-1, 2])]}
    work(c, "scene.json").write_text(json.dumps(out))
    print(c["dir"].name, "plane: %d/%d inliers, scale %.4f, camera height %.2f -> %.2f m, travel %.2f m, sun %s" %
          (best[0], len(Q), scale, Cp[0, 2], Cp[-1, 2], np.linalg.norm(Cp[-1] - Cp[0]), np.round(out["sun"], 2)))
    sheet = [cv2.resize(overlay(c, i, P), (960, 540)) for i in (0, N // 2, N - 1)]
    cv2.imwrite(str(work(c, "check_plane.jpg")), np.vstack(sheet), [cv2.IMWRITE_JPEG_QUALITY, 88])

_SC = {}
def overlay(c, i, P=None, cell=None, ext=None, pts=True, bg=None):
    """plate (or bg) + SfM points + the plane grid, axes (x red, y green, z blue) and the sun's shadow direction (yellow)"""
    key = str(c["dir"]); S = _SC.get(key) or _SC.setdefault(key, scene(c)); im = frame(c, i) if bg is None else bg; W, H = c["W"], c["H"]; f = S["f"]; Cp, Rcp = S["cam"][i], S["rot"][i]
    pl = c["plane"]; cell = cell or pl.get("grid", max(0.5, round(abs(Cp[2]) / 8, 1))); ext = ext or pl.get("grid_n", 14)
    def pr(Xp):                                                            # plane coords -> pixels (None behind the camera)
        q = (np.asarray(Xp, float) - Cp) @ Rcp; return None if q[2] < 1e-3 * abs(Cp[2]) else (f * q[0] / q[2] + W / 2, f * q[1] / q[2] + H / 2)
    def seg(a, b, col, th):
        n = 24; prev = None
        for t in np.linspace(0, 1, n + 1):
            p = pr(np.asarray(a) * (1 - t) + np.asarray(b) * t)
            if p is not None and prev is not None and max(abs(p[0]), abs(p[1]), abs(prev[0]), abs(prev[1])) < 20000:
                cv2.line(im, (int(prev[0]), int(prev[1])), (int(p[0]), int(p[1])), col, th, cv2.LINE_AA)
            prev = p
    if pts and P is not None:
        cam = _SC.get(key + "cam") or _SC.setdefault(key + "cam", dict(np.load(work(c, "camera.npz")))); uv, z = project(P, cam["R"][i], cam["C"][i], f, W, H)
        hgt = S["scale"] * ((P - S["O"]) @ S["M"][2])
        for (u, v), zz, hh in zip(uv, z, hgt):
            if zz > 0 and 0 <= u < W and 0 <= v < H: cv2.circle(im, (int(u), int(v)), 2, (60, 255, 60) if abs(hh) < 0.02 * abs(Cp[2]) else (60, 160, 255), -1, cv2.LINE_AA)
    L = cell * ext
    for g in range(-ext, ext + 1):
        seg((g * cell, -L, 0), (g * cell, L, 0), (255, 255, 255), 1); seg((-L, g * cell, 0), (L, g * cell, 0), (255, 255, 255), 1)
    seg((0, 0, 0), (2 * cell, 0, 0), (40, 40, 255), 3); seg((0, 0, 0), (0, 2 * cell, 0), (40, 220, 40), 3); seg((0, 0, 0), (0, 0, 2 * cell), (255, 120, 40), 3)
    sd = S["sun"]; sh = -np.array([sd[0], sd[1], 0.0]) / max(sd[2], 0.05) * cell; seg((0, 0, cell), (sh[0], sh[1], 0), (0, 230, 255), 2); seg((0, 0, 0), (sh[0], sh[1], 0), (0, 230, 255), 2)
    return im

def pick(c):
    """python slv.py pick <piece> frame x y [x y ...]: plane coordinates (metres) under those pixels; with 'h=1.5' the point on that height"""
    S = scene(c); a = sys.argv[3:]; i = int(a[0]); h = 0.0; xy = []
    for t in a[1:]:
        if t.startswith("h="): h = float(t[2:])
        else: xy.append(float(t))
    ro, R = S["cam"][i], S["rot"][i]
    for x, y in zip(xy[0::2], xy[1::2]):
        rd = R @ np.array([(x - c["W"] / 2) / S["f"], (y - c["H"] / 2) / S["f"], 1.0]); t = (h - ro[2]) / rd[2]
        print("px (%g, %g) frame %d -> plane (%.3f, %.3f, %.3f)%s" % (x, y, i, *(ro + t * rd), "" if t > 0 else "  [behind the camera / above the horizon]"))

def pick3d(c):
    """python slv.py pick3d <piece> frameA xA yA frameB xB yB: the 3D point seen at those two pixels (closest point of the two rays), plane coords"""
    S = scene(c); a = sys.argv[3:]; ia, xa, ya, ib, xb, yb = int(a[0]), float(a[1]), float(a[2]), int(a[3]), float(a[4]), float(a[5])
    def r(i, x, y): d = S["rot"][i] @ np.array([(x - c["W"] / 2) / S["f"], (y - c["H"] / 2) / S["f"], 1.0]); return S["cam"][i], d / np.linalg.norm(d)
    (o1, d1), (o2, d2) = r(ia, xa, ya), r(ib, xb, yb); w = o1 - o2; b_ = d1 @ d2; den = 1 - b_ * b_
    t1 = (b_ * (d2 @ w) - d1 @ w) / den; t2 = ((d2 @ w) - b_ * (d1 @ w)) / den; p1, p2 = o1 + t1 * d1, o2 + t2 * d2
    print("3D point (%.3f, %.3f, %.3f)  ray gap %.3f m, baseline %.2f m, depth %.1f m" % (*((p1 + p2) / 2), np.linalg.norm(p1 - p2), np.linalg.norm(o1 - o2), t1))

def track(c):
    cam = np.load(work(c, "camera.npz")); P = cam["pts"].astype(np.float64); out = work(c, "track"); shutil.rmtree(out, ignore_errors=True); out.mkdir()
    for i in range(c["N"]): cv2.imwrite(str(out / f"{i:04d}.jpg"), overlay(c, i, P), [cv2.IMWRITE_JPEG_QUALITY, 92])
    print(c["dir"].name, "track pass", c["N"])

if __name__ == "__main__":
    step, c = sys.argv[1], load(sys.argv[2]); {"prep": prep, "solve": (solve_h if c.get("solver") == "homography" else solve), "plane": plane, "track": track, "pick": pick, "pick3d": pick3d}[step](c)
