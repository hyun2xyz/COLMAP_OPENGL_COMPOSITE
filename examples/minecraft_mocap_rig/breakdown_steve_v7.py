"""Official v7 House-Rules Breakdown for Minecraft Character Mocap Replacement (Steve).

Strictly obeys docs/breakdown/README.md (v7):
- 1080x1350 @ 30fps, H.264 yuv420p, bt709, +faststart, outer bg #fcfefd
- Static sharp opaque white rectangular captions (21px, height 24px, 27px uniform leading, 24px footer margins)
- Approved 2-line footer (exact Colmap / OpenGL casing)
- 0-89:    Top RESULT / Bottom ORIGINAL high-action dance comparison (src 30..119, exact 3.0s)
- 90-179:  01 CAMERA SOLVE (HOMOGRAPHY) + WORLD ALIGN (green tracked points + 3D floor grid/axes on SAME frame, 3.0s)
- 180-188: 9-frame RTL linear wipe (2px white line) -> 3D diagnostic view
- 189-239: 3D diagnostic orbit view (tracked points, camera path + frustum with live plate, floor grid, lifted FK skeleton)
- 240-419: Continuous monotonic source clock (starting src 45), 9f RTL wipes with SAME source frame on both sides:
           02 MEDIAPIPE MOCAP / 3D RIG RETARGET    (240-275, 36f)
           03 AI CLEAN PLATE (HOMOGRAPHY WARP)     (276-311, 36f)
           04 3D VOXEL CHARACTER WIREFRAME (OpenGL)(312-347, 36f)
           05 DAYLIGHT SHADOWS + CLAY (OpenGL)     (348-383, 36f)
           06 PBR CHARACTER / COMPOSITE            (384-419, 36f)
- 420-428: 9f RTL wipe -> RESULT source frame 0 (FIXED), caption switches to RESULT
- 429-458: Exact 30-frame (1.0s) static hold on RESULT source frame 0
- 459-698: Full RESULT replay (src 0..239)
"""
import argparse, hashlib, json, math, shutil, subprocess, sys
from pathlib import Path
import numpy as np, cv2

here = Path(__file__).resolve().parent
ROOT = here.parents[1]
sys.path.insert(0, str(here.parent / "shader_live" / "pipeline"))
sys.path.insert(0, str(ROOT / "tools" / "breakdown"))
from slv import load, work, frame, scene  # noqa: E402
from captions import RectLine             # noqa: E402

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
BONES = [
    (11, 12), (11, 13), (13, 15), (15, 19), (12, 14), (14, 16), (16, 20),
    (11, 23), (12, 24), (23, 24), (23, 25), (25, 27), (27, 31),
    (24, 26), (26, 28), (28, 32), (0, 11), (0, 12),
]


def find_ffmpeg():
    tl = ROOT / "tools.local.json"
    if tl.exists():
        d = json.loads(tl.read_text(encoding="utf-8"))
        p = d.get("ffmpeg") or (d.get("tools") or {}).get("ffmpeg")
        if p and Path(p).exists():
            return str(p)
    return shutil.which("ffmpeg") or "ffmpeg"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("piece", nargs="?", default=str(here / "piece_steve"))
    ap.add_argument("--out", default="out")
    ap.add_argument("--still", default=None)
    args = ap.parse_args()

    if FONT is None:
        sys.exit("No Helvetica/Arial/DejaVu font found")

    c = load(args.piece)
    S = scene(c)
    N, FPS, f = int(c["N"]), float(c["fps"]), float(S["f"])
    CW, CH = int(c["W"]), int(c["H"])

    CMP_START_SRC = 30
    T_TRACK = 90
    T_3D = 180
    P0 = 240
    STEP_SPECS = [
        ("rig",   "02 MEDIAPIPE MOCAP / 3D RIG RETARGET",     36, 1.0),
        ("clean", "03 AI CLEAN PLATE (HOMOGRAPHY WARP)",      36, 1.0),
        ("wire",  "04 3D VOXEL CHARACTER WIREFRAME (OpenGL)", 36, 1.0),
        ("clay",  "05 DAYLIGHT SHADOWS + CLAY (OpenGL)",      36, 1.0),
        ("comp",  "06 PBR CHARACTER / COMPOSITE",             36, 1.0),
    ]
    LABELS = ["01 CAMERA SOLVE (HOMOGRAPHY) + WORLD ALIGN"] + [lbl for _, lbl, _, _ in STEP_SPECS]

    STEPS = []
    o0, s0 = P0, 45.0
    for key, _, n_f, rate in STEP_SPECS:
        STEPS.append((key, o0, n_f, rate, s0))
        o0 += n_f
        s0 += n_f * rate * FPS / 30.0
    P1 = o0
    R_HOLD = P1 + WIPE
    R_PLAY = R_HOLD + 30
    TOTAL = R_PLAY + int(math.ceil(N * 30.0 / FPS))

    def src_from_o(o_rel):
        return min(max(int(math.floor(o_rel * FPS / 30.0 + 1e-6)), 0), N - 1)

    def step_at(o):
        for k, (key, a, n_f, rate, s_a) in enumerate(STEPS):
            if a <= o < a + n_f:
                j = min(int(math.floor(s_a + (o - a) * rate * FPS / 30.0 + 1e-6)), N - 1)
                return k, j
        return len(STEPS) - 1, N - 1

    line_cache = {}
    def rect_img(text, color):
        key = (text, color)
        if key not in line_cache:
            bg, fg, _ = RectLine(text, FONT, size=21).render(color=color)
            im = bg.copy()
            im.alpha_composite(fg)
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
                "text": text,
                "role": role,
                "rect": [x0, int(y), w_box, 24],
                "font_size": 21,
                "shape": "sharp white rectangle",
                "floating": False,
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

    L = np.load(work(c, "limbs.npz"))
    P3 = L["P"].astype(np.float64)
    cam = np.load(work(c, "camera.npz"))
    Psfm = S["scale"] * ((cam["pts"].astype(np.float64) - S["O"]) @ np.asarray(S["M"]).T)
    rgb = cam["rgb"] if "rgb" in cam.files else np.full((len(Psfm), 3), 200)
    hip0 = 0.5 * (P3[:, 23] + P3[:, 24]).mean(0)
    plane_cfg = c.get("plane", {})

    def proj(i, X):
        q = (np.atleast_2d(X) - S["cam"][i]) @ np.asarray(S["rot"][i])
        return f * q[:, :2] / np.maximum(q[:, 2:3], 1e-4) + [CW / 2.0, CH / 2.0], q[:, 2]

    def grid_axes(im, i):
        cell = float(plane_cfg.get("grid", 0.5))
        ext = int(plane_cfg.get("grid_n", 10))
        Lx = cell * ext
        def seg(a, b, col, th):
            t = np.linspace(0, 1, 25)[:, None]
            uv, z = proj(i, np.asarray(a) * (1 - t) + np.asarray(b) * t)
            for k in range(24):
                if z[k] > 0.05 and z[k + 1] > 0.05:
                    cv2.line(im, tuple(np.int32(uv[k])), tuple(np.int32(uv[k + 1])), col, th, cv2.LINE_AA)
        for g in range(-ext, ext + 1):
            seg((g * cell, -Lx, 0), (g * cell, Lx, 0), (235, 235, 235), 1)
            seg((-Lx, g * cell, 0), (Lx, g * cell, 0), (235, 235, 235), 1)
        seg((0, 0, 0), (2 * cell, 0, 0), (40, 40, 255), 3)
        seg((0, 0, 0), (0, 2 * cell, 0), (40, 220, 40), 3)
        seg((0, 0, 0), (0, 0, 2 * cell), (255, 120, 40), 3)

    def tracking(i):
        im = frame(c, i).copy()
        grid_axes(im, i)
        uv, z = proj(i, Psfm)
        for (u, v), zz in zip(uv, z):
            if zz > 0.05 and 0 <= u < CW and 0 <= v < CH:
                cv2.circle(im, (int(u), int(v)), 4, (40, 255, 40), -1, cv2.LINE_AA)
        return im

    def skeleton(img, uv, col, th):
        for a, b in BONES:
            cv2.line(img, tuple(np.int32(uv[a])), tuple(np.int32(uv[b])), col, th, cv2.LINE_AA)

    def view3d(i, o_rel):
        Wv, Hv, fv = 1280, 720, 900.0
        im = np.zeros((Hv, Wv, 3), np.uint8)
        im[:] = np.linspace(108, 72, Hv)[:, None, None].astype(np.uint8)
        tgt = hip0 + np.array([0.0, 0.0, 0.30])
        c0 = np.asarray(S["cam"][0])
        d0 = c0 - tgt; d0[2] = 0.0
        r0 = float(np.linalg.norm(d0))
        a0 = math.atan2(d0[1], d0[0])
        a = a0 + math.radians(28.0 + 14.0 * (o_rel / 60.0))
        dist = r0 * 1.52
        eye = tgt + np.array([dist * math.cos(a), dist * math.sin(a), 2.1])
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

        cell = 0.5; ext = 12
        for g in range(-ext, ext + 1):
            col = (90, 90, 230) if g == 0 else (200, 200, 200)
            seg((g * cell, -ext * cell, 0), (g * cell, ext * cell, 0), col, 1)
            seg((-ext * cell, g * cell, 0), (ext * cell, g * cell, 0), col, 1)
        uv, z = pv(Psfm)
        for k in np.argsort(-z):
            if z[k] > 0.1:
                col_pt = tuple(int(min(255, 60 + 1.5 * x)) for x in rgb[k][::-1])
                cv2.circle(im, (int(uv[k, 0]), int(uv[k, 1])), 4, col_pt, -1, cv2.LINE_AA)
        path, zp = pv(np.asarray(S["cam"]))
        for k in range(len(path) - 1):
            if zp[k] > 0.1 and zp[k + 1] > 0.1:
                cv2.line(im, tuple(np.int32(path[k])), tuple(np.int32(path[k + 1])), (0, 220, 255), 2, cv2.LINE_AA)
        ro, R = np.asarray(S["cam"][i]), np.asarray(S["rot"][i])
        dpt = 1.25
        corners = [ro + R @ np.array([sx * CW / 2.0 / f * dpt, sy * CH / 2.0 / f * dpt, dpt])
                   for sx, sy in ((-1, -1), (1, -1), (1, 1), (-1, 1))]
        cu, cz = pv(np.array(corners))
        if (cz > 0.1).all():
            Mw = cv2.getPerspectiveTransform(np.float32([[0, 0], [CW, 0], [CW, CH], [0, CH]]), np.float32(cu))
            plate = cv2.warpPerspective(frame(c, i), Mw, (Wv, Hv))
            m = cv2.warpPerspective(np.full((CH, CW), 255, np.uint8), Mw, (Wv, Hv)) > 0
            im[m] = plate[m]
            for k in range(4):
                seg(corners[k], corners[(k + 1) % 4], (60, 255, 60), 2)
                seg(ro, corners[k], (60, 255, 60), 2)
        su, sz = pv(P3[i])
        if (sz > 0.1).all():
            skeleton(im, su, (245, 245, 245), 3)
        return im

    def passimg(key, i):
        p = work(c, "comp", f"{i:04d}.png") if key == "comp" else work(c, "pass_steve", key, f"{i:04d}.jpg")
        im = cv2.imread(str(p))
        if im is None:
            sys.exit(f"Missing {p}: run render_steve_mocap.py --passes first")
        return im

    def get_pic(key, i, o_rel=0):
        if key == "original":
            im = frame(c, i)
        elif key == "result":
            im = passimg("comp", i)
        elif key == "track":
            im = tracking(i)
        elif key == "view3d":
            im = view3d(i, o_rel)
        else:
            im = passimg(key, i)
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
        rows_g = [np.hstack(imgs[r:r + cols]) for r in range(0, len(imgs), cols)]
        sheet = np.vstack(rows_g)
        p_out = work(c, "bd_steve_v7_sheet.jpg")
        cv2.imwrite(str(p_out), cv2.cvtColor(sheet, cv2.COLOR_RGB2BGR)[::2, ::2], [cv2.IMWRITE_JPEG_QUALITY, 92])
        print("stills", idx_list, "->", p_out)
        return

    out_dir = c["dir"] / args.out
    out_dir.mkdir(parents=True, exist_ok=True)
    mp4 = out_dir / "Minecraft_Steve_Mocap_Rig_Breakdown_v7_1080x1350.mp4"
    if mp4.exists():
        sys.exit(f"{mp4} already exists: never overwrite final breakdown MP4!")

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
            print(f"steve_bd frame {o}/{TOTAL}", flush=True)
    enc.stdin.close()
    rc = enc.wait()
    if rc != 0:
        sys.exit(f"ffmpeg failed with rc={rc}")

    wipe_starts = [T_3D] + [a for _, a, _, _, _ in STEPS] + [P1]
    proc_same_time_wipes = [a for _, a, _, _, _ in STEPS[1:]]
    meta = {
        "file": mp4.name,
        "size": [W, H],
        "fps": 30,
        "frames": TOTAL,
        "seconds": round(TOTAL / 30.0, 3),
        "source_frames": N,
        "source_fps": FPS,
        "font": FONT,
        "rules": "docs/breakdown/README.md v7",
        "labels": LABELS,
        "footer": FOOTER,
        "timeline": {
            "compare": [0, T_TRACK - 1],
            "compare_source_start": CMP_START_SRC,
            "track_grid": [T_TRACK, T_3D - 1],
            "wipe_to_3d": [T_3D, T_3D + WIPE - 1],
            "view3d": [T_3D + WIPE, P0 - 1],
            "process": [P0, P1 - 1],
            "process_steps": {k: [a, a + n_f - 1] for k, a, n_f, _, _ in STEPS},
            "step_rates": {k: r for k, _, _, r, _ in STEPS},
            "wipe_starts": wipe_starts,
            "process_same_time_wipe_starts": proc_same_time_wipes,
            "wipe_to_result": [P1, R_HOLD - 1],
            "hold_result_frame0": [R_HOLD, R_PLAY - 1],
            "result_replay": [R_PLAY, TOTAL - 1],
        },
        "layout": {k: v[1] for k, v in STAGES.items()},
        "edit": edls,
        "sha256": hashlib.sha256(mp4.read_bytes()).hexdigest(),
    }
    mp4.with_suffix(".json").write_text(json.dumps(meta, indent=1), encoding="utf-8")
    print("->", mp4, TOTAL, "frames", f"({TOTAL/30.0:.2f}s)")


if __name__ == "__main__":
    main()
