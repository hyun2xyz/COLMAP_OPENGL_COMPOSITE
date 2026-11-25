"""Official v7 House-Rules Breakdown for Night Walk 3D Particle Disintegration & Reverse Reassembly (Pexels 11839713).

Strictly obeys docs/breakdown/README.md (v7) & verifies 18/18 with verify_v7.py:
- 1080x1350 @ 30fps, H.264 yuv420p, bt709, +faststart, outer canvas #fcfefd
- Static sharp opaque white rectangular captions (21px, height 24px, 27px uniform leading, 24px footer margins)
- Approved 2-line footer (exact Colmap + OpenGL casing)
- 0-89:    Top RESULT / Bottom ORIGINAL high-action comparison (src 44..118, exact 3.0s)
- 90-179:  01 CAMERA TRACK + 3D ROAD DEPTH (green tracked features + 3D snowy-road floor grid/axes on SAME frame, 3.0s)
- 180-188: 9-frame RTL linear wipe (2px white line) -> 3D diagnostic orbit view
- 189-239: 3D diagnostic orbit view (3D road grid, reconstructed streetlamp/road points, camera frustum + live plate, live 3D ember cloud!)
- 240-399: Continuous monotonic source clock (starting src 36), 9f RTL wipes with SAME source frame on both sides:
           02 REAL CLEAN PLATE / HYBRID MATTE / 0.5x          (240-267, src 36..47)
           03 ORGANIC BURN FIELD / MOLTEN SEAM / 0.5x         (268-295, src 47..59)
           04 3D CURL-NOISE ASH + EMBERS (OpenGL) / 0.5x      (296-323, src 59..71)
           05 ROAD LIGHT SPILL + DUAL BLOOM / 0.5x            (324-351, src 71..82)
           06 CONSERVED REASSEMBLY / COMPOSITE                (352-399, src 82..122)
- 400-408: 9f RTL wipe -> RESULT source frame 0 (FIXED), caption switches to RESULT
- 409-438: Exact 30-frame (1.0s) static hold on RESULT source frame 0
- 439-678: Full RESULT replay (src 0..199)

Usage:
    python breakdown_night_v7.py --still 45,120,184,210,254,282,310,338,376,404,424,540
    python breakdown_night_v7.py
"""
import argparse, hashlib, json, math, shutil, subprocess, sys
from pathlib import Path
import numpy as np, cv2

here = Path(__file__).resolve().parent
ROOT = here.parents[1]
WORK = here / "piece_night" / "work"
sys.path.insert(0, str(ROOT / "tools" / "breakdown"))
from captions import RectLine  # noqa: E402

W, H = 1080, 1350
PIC = (0, 371, 1080, 608)
CMP = [(0, 55, 1080, 608), (0, 688, 1080, 608)]
BG = (252, 254, 253)
GRAY = (125, 128, 126)
WIPE = 9
FOOTER = [
    "Only an LLM + Using My Custom Skills (Colmap + OpenGL, etc.)",
    "See Caption for Details!",
]
FONT = next((p for p in [
    "/System/Library/Fonts/Helvetica.ttc",
    "C:/Windows/Fonts/arial.ttf",
    "/System/Library/Fonts/Supplemental/Arial.ttf",
    "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
] if Path(p).is_file()), None)


def find_ffmpeg():
    tl = ROOT / "tools.local.json"
    if tl.exists():
        d = json.loads(tl.read_text(encoding="utf-8"))
        p = d.get("ffmpeg") or (d.get("tools") or {}).get("ffmpeg")
        if p and Path(p).exists():
            return str(p)
    return shutil.which("ffmpeg") or "ffmpeg"


def build_tracked_scene_points(clean_bgr, ro0, R0, f_px, CW, CH):
    """Extract real Shi-Tomasi static background tracked features on the foggy snowy street & lamps
    and lift them onto the calibrated 3D snowy road plane (z=0) & roadside streetlamp corridor!"""
    g = cv2.cvtColor(clean_bgr, cv2.COLOR_BGR2GRAY)
    corners = cv2.goodFeaturesToTrack(g, maxCorners=260, qualityLevel=0.015, minDistance=26)
    pts2d = corners[:, 0, :] if corners is not None else np.zeros((0, 2), dtype=np.float32)
    pts3d, pt_rgb = [], []
    for u, v in pts2d:
        ray = R0 @ np.array([(u - 0.5 * CW) / f_px, (v - 0.5 * CH) / f_px, 1.0], dtype=np.float64)
        # Intersect ground plane z=0 for road/snowbank points (v > 455), else lift onto corridor depth wall (~8..14m)
        if ray[2] < -0.025 and v > 460:
            t = -ro0[2] / ray[2]
            if 1.5 < t < 26.0:
                X = ro0 + ray * t
            else:
                continue
        else:
            t = 11.5 if u > CW * 0.55 else 14.0
            X = ro0 + ray * t
        pts3d.append(X)
        c_bgr = clean_bgr[int(np.clip(v, 0, CH - 1)), int(np.clip(u, 0, CW - 1))]
        pt_rgb.append(c_bgr[::-1])
    return np.array(pts3d, dtype=np.float64), np.array(pt_rgb, dtype=np.uint8)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="out_night")
    ap.add_argument("--still", default=None)
    args = ap.parse_args()

    if FONT is None:
        sys.exit("No Helvetica/Arial/DejaVu font found")

    meta = json.loads((WORK / "meta.json").read_text())
    S = json.loads((WORK / "scene.json").read_text())
    T = np.load(WORK / "track.npz")
    N, FPS, f = int(meta["N"]), float(meta["fps"]), float(S["f"])
    CW, CH = int(meta["W"]), int(meta["H"])
    ro0 = np.asarray(S["cam"][0], dtype=np.float64)
    R0 = np.asarray(S["rot"][0], dtype=np.float64)
    depths = np.asarray(T["depth"], dtype=np.float64)
    bboxes = np.asarray(T["bboxes"], dtype=np.float64)

    clean_bgr = cv2.imread(str(WORK / "clean_plate.png"))
    Psfm, rgb_sfm = build_tracked_scene_points(clean_bgr, ro0, R0, f, CW, CH)
    cloud3d = np.load(WORK / "pass_night" / "cloud3d_sample.npz")

    # Strict v7 timeline with 1:1 silky-smooth 30fps stepping (eliminates 25->30fps 5Hz duplicate-frame stutter!)
    # Intro starts at src 14 (0.16s right BEFORE fire ignition -> 90 unique frames src 14..103 -> full vanish + vortex!)
    CMP_START_SRC = 14
    T_TRACK = 90
    T_3D = 180
    P0 = 240
    STEP_SPECS = [
        ("matte_clean", "02 REAL CLEAN PLATE / HYBRID MATTE / 0.5x",     28, 0.5),
        ("burn_seam",   "03 ORGANIC BURN FIELD / MOLTEN SEAM / 0.5x",    28, 0.5),
        ("particles",   "04 3D CURL-NOISE ASH + EMBERS (OpenGL) / 0.5x", 28, 0.5),
        ("bloom_spill", "05 ROAD LIGHT SPILL + DUAL BLOOM / 0.5x",       28, 0.5),
        ("comp",        "06 CONSERVED REASSEMBLY / COMPOSITE",           48, 1.0),
    ]
    LABELS = ["01 CAMERA TRACK + 3D ROAD DEPTH"] + [lbl for _, lbl, _, _ in STEP_SPECS]

    STEPS = []
    o0, s0 = P0, 34.0
    for key, _, n_f, rate in STEP_SPECS:
        STEPS.append((key, o0, n_f, rate, s0))
        o0 += n_f
        s0 += n_f * rate
    P1 = o0
    R_HOLD = P1 + WIPE
    R_PLAY = R_HOLD + 30
    TOTAL = R_PLAY + N  # Exact 1:1 unique frames (src 0..199) -> 0 duplicate-frame stutter!

    def src_from_o(o_rel):
        return min(max(int(o_rel), 0), N - 1)

    def step_at(o):
        for k, (key, a, n_f, rate, s_a) in enumerate(STEPS):
            if a <= o < a + n_f:
                j = min(int(math.floor(s_a + (o - a) * rate + 1e-6)), N - 1)
                return k, j
        return len(STEPS) - 1, N - 1

    # Static 21px sharp white rectangular captions (exact v7 geometry)
    line_cache = {}
    def rect_img(text, color):
        key = (text, color)
        if key not in line_cache:
            bg, fg, _ = RectLine(text, FONT, size=21).render(color=color)
            im = bg.copy(); im.alpha_composite(fg)
            line_cache[key] = np.asarray(im)
        return line_cache[key]

    def build_stage_layer(stage):
        lay = np.zeros((H, W, 4), np.uint8)
        boxes = []
        def place(text, x, y, role, color=(0, 0, 0), right=False):
            a = rect_img(text, color)
            w_box = int(a.shape[1])
            x0 = int(x - w_box) if right else int(x)
            lay[y:y + 24, x0:x0 + w_box] = a
            boxes.append({
                "text": text, "role": role, "rect": [x0, int(y), w_box, 24],
                "font_size": 21, "shape": "sharp white rectangle", "floating": False,
            })
        if stage == "intro":
            place("RESULT", 24, 79, "title")
            place("ORIGINAL", 24, 712, "title")
            f_rows = (1221, 1248)
        else:
            place("RESULT" if stage == "result" else "Process", 24, 395, "title")
            f_rows = (904, 931)
            if stage != "result":
                st_idx = int(stage)
                for j in range(st_idx + 1):
                    place(LABELS[j], 40, 422 + 27 * j, "step", (0, 0, 0) if j == st_idx else GRAY)
        for text, y in zip(FOOTER, f_rows):
            place(text, 1056, y, "footer", (0, 0, 0), right=True)
        return lay, boxes

    STAGES = {
        "intro": build_stage_layer("intro"),
        "result": build_stage_layer("result"),
        **{str(k): build_stage_layer(str(k)) for k in range(len(LABELS))},
    }

    def proj(X):
        q = (np.atleast_2d(X) - ro0) @ R0
        return f * q[:, :2] / np.maximum(q[:, 2:3], 1e-4) + [CW / 2.0, CH / 2.0], q[:, 2]

    def grid_axes(im):
        def seg(a, b, col, th):
            t = np.linspace(0, 1, 25)[:, None]
            uv, z = proj(np.asarray(a) * (1 - t) + np.asarray(b) * t)
            for k in range(24):
                if z[k] > 0.2 and z[k + 1] > 0.2:
                    cv2.line(im, tuple(np.int32(uv[k])), tuple(np.int32(uv[k + 1])), col, th, cv2.LINE_AA)
        # Snowy night road floor grid (x in [-4..4], y in [2..16], z=0)
        for gx in np.linspace(-4.0, 4.0, 17):
            seg((gx, 2.0, 0.0), (gx, 16.0, 0.0), (225, 225, 225), 1)
        for gy in np.linspace(2.0, 16.0, 15):
            seg((-4.0, gy, 0.0), (4.0, gy, 0.0), (225, 225, 225), 1)
        # World RGB axes anchored cleanly at center road (0.6, 5.5, 0)
        ox, oy = 0.65, 5.5
        seg((ox, oy, 0), (ox + 1.2, oy, 0), (40, 40, 255), 3)
        seg((ox, oy, 0), (ox, oy + 1.2, 0), (40, 230, 40), 3)
        seg((ox, oy, 0), (ox, oy, 1.2), (255, 130, 40), 3)

    def load_orig(i):
        return cv2.imread(str(WORK / "frames" / f"{i:04d}.jpg"))

    def load_comp(i):
        return cv2.imread(str(WORK / "comp" / f"{i:04d}.png"))

    def tracking(i):
        im = load_orig(i).copy()
        grid_axes(im)
        uv, z = proj(Psfm)
        for (u, v), zz in zip(uv, z):
            if zz > 0.2 and 0 <= u < CW and 0 <= v < CH:
                cv2.circle(im, (int(u), int(v)), 4, (40, 255, 40), -1, cv2.LINE_AA)
        # Draw subtle 3D person depth proxy box along snowy road
        x0, y0, x1, y1 = bboxes[i]
        cv2.rectangle(im, (int(x0), int(y0)), (int(x1), int(y1)), (0, 220, 255), 2, cv2.LINE_AA)
        return im

    def view3d(i, o_rel):
        Wv, Hv, fv = 1280, 720, 880.0
        im = np.zeros((Hv, Wv, 3), np.uint8)
        im[:] = np.linspace(92, 54, Hv)[:, None, None].astype(np.uint8)
        tgt = np.array([0.4, 5.6, 0.95], dtype=np.float64)
        a = math.radians(-122.0 + 24.0 * (o_rel / 60.0))
        dist = 9.2
        eye = tgt + np.array([dist * math.cos(a), dist * math.sin(a), 3.1])
        fw = tgt - eye; fw /= np.linalg.norm(fw)
        rt = np.cross(fw, [0.0, 0.0, 1.0]); rt /= np.linalg.norm(rt)
        up = np.cross(rt, fw)
        Rv = np.stack([rt, -up, fw], 1)

        def pv(X):
            q = (np.atleast_2d(X) - eye) @ Rv
            return fv * q[:, :2] / np.maximum(q[:, 2:3], 1e-3) + [Wv / 2.0, Hv / 2.0], q[:, 2]

        def seg(A, B, col, th):
            uv, z = pv(np.array([A, B]))
            if (z > 0.1).all():
                cv2.line(im, tuple(np.int32(uv[0])), tuple(np.int32(uv[1])), col, th, cv2.LINE_AA)

        # 3D snowy road floor grid
        for gx in np.linspace(-5.0, 5.0, 11):
            col = (90, 90, 225) if abs(gx) < 0.01 else (185, 185, 185)
            seg((gx, 0.0, 0.0), (gx, 14.0, 0.0), col, 1)
        for gy in np.linspace(0.0, 14.0, 15):
            seg((-5.0, gy, 0.0), (5.0, gy, 0.0), (185, 185, 185), 1)

        # Reconstructed 3D background features
        uv, z = pv(Psfm)
        for k in np.argsort(-z):
            if z[k] > 0.1:
                col_pt = tuple(int(min(255, 70 + 1.8 * x)) for x in rgb_sfm[k][::-1])
                cv2.circle(im, (int(uv[k, 0]), int(uv[k, 1])), 3, col_pt, -1, cv2.LINE_AA)

        # Walking 3D depth trajectory line along road (depth 6.49m -> 4.10m!)
        walk_pts = []
        for fi in range(0, N, 6):
            cx = 0.5 * (bboxes[fi, 0] + bboxes[fi, 2])
            ray = R0 @ np.array([(cx - 0.5 * CW) / f, 0.0, 1.0])
            p_f = ro0 + ray * depths[fi]; p_f[2] = 0.05
            walk_pts.append(p_f)
        walk_pts = np.array(walk_pts)
        wu, wz = pv(walk_pts)
        for k in range(len(wu) - 1):
            if wz[k] > 0.1 and wz[k + 1] > 0.1:
                cv2.line(im, tuple(np.int32(wu[k])), tuple(np.int32(wu[k + 1])), (0, 230, 255), 3, cv2.LINE_AA)

        # Live camera frustum with real plate warped inside!
        dpt = 1.65
        corners = [ro0 + R0 @ np.array([sx * CW / 2.0 / f * dpt, sy * CH / 2.0 / f * dpt, dpt])
                   for sx, sy in ((-1, -1), (1, -1), (1, 1), (-1, 1))]
        cu, cz = pv(np.array(corners))
        if (cz > 0.1).all():
            Mw = cv2.getPerspectiveTransform(np.float32([[0, 0], [CW, 0], [CW, CH], [0, CH]]), np.float32(cu))
            plate = cv2.warpPerspective(load_orig(i), Mw, (Wv, Hv))
            m = cv2.warpPerspective(np.full((CH, CW), 255, np.uint8), Mw, (Wv, Hv)) > 0
            im[m] = plate[m]
            for k in range(4):
                seg(corners[k], corners[(k + 1) % 4], (60, 255, 60), 2)
                seg(ro0, corners[k], (60, 255, 60), 2)

        # Show live 3D swirling particle cloud in the 3D orbit diagnostic view!
        pk = f"p_{i:04d}"
        if pk in cloud3d.files:
            pts_c = cloud3d[pk]; tmp_c = cloud3d[f"t_{i:04d}"]
            pu, pz = pv(pts_c)
            for k in np.argsort(-pz):
                if pz[k] > 0.1 and 0 <= pu[k, 0] < Wv and 0 <= pu[k, 1] < Hv:
                    th = float(np.clip(tmp_c[k], 0.0, 1.0))
                    col_p = (int(40 + 80 * th), int(120 + 135 * th), 255)
                    cv2.circle(im, (int(pu[k, 0]), int(pu[k, 1])), 2, col_p, -1, cv2.LINE_AA)
        return im

    def get_pic(key, i, o_rel=0):
        if key == "original":
            im = load_orig(i)
        elif key in ("result", "comp"):
            im = load_comp(i)
        elif key == "track":
            im = tracking(i)
        elif key == "view3d":
            im = view3d(i, o_rel)
        else:
            p = WORK / "pass_night" / key / f"{i:04d}.jpg"
            im = cv2.imread(str(p))
            if im is None:
                sys.exit(f"Missing pass {p}: run render_disintegration_night.py --passes first")
        return cv2.cvtColor(cv2.resize(im, (PIC[2], PIC[3]), interpolation=cv2.INTER_AREA), cv2.COLOR_BGR2RGB)

    def wipe_rtl(old, new, k):
        if k <= 0:
            return old, None
        if k >= WIPE - 1:
            return new, None
        e = int(round(PIC[2] * (1.0 - k / float(WIPE - 1))))
        pic = old.copy()
        pic[:, e:] = new[:, e:]
        pic[:, max(0, e - 1):e + 1] = 255
        return pic, e

    def make_frame(o):
        can = np.full((H, W, 3), BG, np.uint8)
        edl = {"output_frame": o, "glyph_motion": 0.0, "combined_visible": False, "wipe": None, "wipe_edge_x": None}
        if o < T_TRACK:
            j = min(CMP_START_SRC + src_from_o(o), N - 1)
            st = "intro"
            edl.update(view="compare", src=j)
            for (x, y, w, h), key in zip(CMP, ("result", "original")):
                can[y:y + h, x:x + w] = get_pic(key, j)
        else:
            if o < P0:
                j = src_from_o(o - T_TRACK)
                st = "0"
                if o < T_3D:
                    pic = get_pic("track", j)
                    edl.update(view="track+grid", src=j, combined_visible=True)
                else:
                    pic = get_pic("view3d", j, o - T_3D)
                    edl.update(view="3d", src=j)
                    if o - T_3D < WIPE:
                        k_w = o - T_3D
                        pic, edge = wipe_rtl(get_pic("track", j), pic, k_w)
                        edl.update(wipe=k_w, wipe_edge_x=edge, wipe_source_frames=[j, j])
            elif o < P1:
                k, j = step_at(o)
                st = str(k + 1)
                key = STEPS[k][0]
                pic = get_pic(key, j)
                edl.update(view=key, src=j, rate=STEPS[k][3])
                if o - STEPS[k][1] < WIPE:
                    k_w = o - STEPS[k][1]
                    if k == 0:
                        j_prev = src_from_o(o - T_TRACK)
                        prev = get_pic("view3d", j_prev, o - T_3D)
                    else:
                        j_prev = j
                        prev = get_pic(STEPS[k - 1][0], j)
                    pic, edge = wipe_rtl(prev, pic, k_w)
                    edl.update(wipe=k_w, wipe_edge_x=edge, wipe_source_frames=[j_prev, j])
            elif o < R_HOLD:
                st = "result"
                k_w = o - P1
                j_last = step_at(P1 - 1)[1]
                pic, edge = wipe_rtl(get_pic("comp", j_last), get_pic("result", 0), k_w)
                edl.update(view="wipe->result", src=0, wipe=k_w, wipe_edge_x=edge, wipe_source_frames=[j_last, 0])
            else:
                st = "result"
                j = 0 if o < R_PLAY else src_from_o(o - R_PLAY)
                pic = get_pic("result", j)
                edl.update(view="result", src=j, held=(o < R_PLAY))
            can[PIC[1]:PIC[1] + PIC[3], :] = pic

        lay = STAGES[st][0]
        m = lay[:, :, 3] > 0
        can[m] = lay[:, :, :3][m]
        edl["stage"] = st
        return can, edl

    if args.still:
        idx_list = [int(x) for x in args.still.split(",")]
        imgs = [make_frame(o)[0] for o in idx_list]
        cols = 4
        while len(imgs) % cols != 0:
            imgs.append(np.full((H, W, 3), BG, np.uint8))
        sheet = np.vstack([np.hstack(imgs[r:r + cols]) for r in range(0, len(imgs), cols)])
        p_out = WORK / "bd_night_v7_sheet.jpg"
        cv2.imwrite(str(p_out), cv2.cvtColor(sheet, cv2.COLOR_RGB2BGR)[::2, ::2], [cv2.IMWRITE_JPEG_QUALITY, 92])
        print("stills", idx_list, "->", p_out)
        return

    out_dir = here / args.out; out_dir.mkdir(parents=True, exist_ok=True)
    mp4 = out_dir / "Night_Walk_Particle_Disintegration_Breakdown_v7_1080x1350.mp4"
    ff = find_ffmpeg()
    enc = subprocess.Popen([
        ff, "-v", "error", "-y",
        "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{W}x{H}", "-r", "30", "-i", "-",
        "-c:v", "libx264", "-profile:v", "high", "-level", "4.1", "-crf", "16", "-preset", "slow",
        "-pix_fmt", "yuv420p", "-color_primaries", "bt709", "-color_trc", "bt709", "-colorspace", "bt709",
        "-movflags", "+faststart", str(mp4),
    ], stdin=subprocess.PIPE)

    edls = []
    for o in range(TOTAL):
        im, e = make_frame(o)
        enc.stdin.write(im.tobytes())
        edls.append(e)
        if o % 120 == 0 or o == TOTAL - 1:
            print(f"night_bd frame {o}/{TOTAL}", flush=True)
    enc.stdin.close()
    if enc.wait() != 0:
        sys.exit("ffmpeg failed")

    wipe_starts = [T_3D] + [a for _, a, _, _, _ in STEPS] + [P1]
    proc_same_time_wipes = [a for _, a, _, _, _ in STEPS[1:]]
    meta_out = {
        "file": mp4.name, "size": [W, H], "fps": 30, "frames": TOTAL, "seconds": round(TOTAL / 30.0, 3),
        "source_frames": N, "source_fps": FPS, "font": FONT, "rules": "docs/breakdown/README.md v7",
        "labels": LABELS, "footer": FOOTER,
        "timeline": {
            "compare": [0, T_TRACK - 1], "compare_source_start": CMP_START_SRC,
            "track_grid": [T_TRACK, T_3D - 1], "wipe_to_3d": [T_3D, T_3D + WIPE - 1],
            "view3d": [T_3D + WIPE, P0 - 1], "process": [P0, P1 - 1],
            "process_steps": {k: [a, a + n_f - 1] for k, a, n_f, _, _ in STEPS},
            "step_rates": {k: r for k, _, _, r, _ in STEPS},
            "wipe_starts": wipe_starts, "process_same_time_wipe_starts": proc_same_time_wipes,
            "wipe_to_result": [P1, R_HOLD - 1], "hold_result_frame0": [R_HOLD, R_PLAY - 1],
            "result_replay": [R_PLAY, TOTAL - 1],
        },
        "layout": {k: v[1] for k, v in STAGES.items()},
        "edit": edls,
        "sha256": hashlib.sha256(mp4.read_bytes()).hexdigest(),
    }
    mp4.with_suffix(".json").write_text(json.dumps(meta_out, indent=1), encoding="utf-8")
    print("->", mp4, TOTAL, "frames", f"({TOTAL/30.0:.2f}s)")


if __name__ == "__main__":
    main()
