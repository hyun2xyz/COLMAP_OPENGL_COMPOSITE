"""3D Liquid Glass breakdown following docs/breakdown/README.md (v7) + spec.json + optional LMS logo end card.

    python breakdown_glass.py [piece_dir] [--still 30,150,230,310,410] [--out out_v2] [--lms]

Strict v7 rules:
- 1080x1350 @ 30fps, #fcfefd page, RectLine sharp white 21px boxes, 27px leading, zero box radius/shadow/motion.
- Fixed footer right-aligned at x=1056:
    "Only an LLM + Using My Custom Skills (Colmap + OpenGL, etc.)"
    "See Caption for Details!"
- Right-to-left linear 9-frame wipes with a 2px white vertical edge inside the picture rectangle.
- Continuous single source clock across all OpenGL process passes (02 -> 03 -> 04 -> 05), then wipe to RESULT frame 0,
  1.0s hold, and full RESULT replay from frame 0 to N-1.
"""
import hashlib, json, math, shutil, subprocess, sys
from pathlib import Path
import numpy as np, cv2

here = Path(__file__).resolve().parent
ROOT = here.parents[1]
sys.path.insert(0, str(here.parent / "shader_live" / "pipeline"))
sys.path.insert(0, str(ROOT / "tools" / "breakdown"))
from slv import load, work, frame, scene   # noqa: E402
from captions import RectLine              # noqa: E402

W, H = 1080, 1350
PIC = (0, 371, 1080, 608)
CMP = [(0, 55, 1080, 608), (0, 688, 1080, 608)]
BG = (252, 254, 253)
GRAY = (125, 128, 126)
WIPE = 9
STEP = 36
FOOTER = ["Only an LLM + Using My Custom Skills (Colmap + OpenGL, etc.)", "See Caption for Details!"]
FONT = next((p for p in [
    "/System/Library/Fonts/Helvetica.ttc",
    "C:/Windows/Fonts/arial.ttf",
    "/System/Library/Fonts/Supplemental/Arial.ttf",
    "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
] if Path(p).is_file()), None)
LMS_MOV = Path(r"C:\Users\Q\Documents\카카오톡 받은 파일\hyun2LMS_logo_motion_1920x1080_30fps_ProRes4444.mov")


def find_ffmpeg():
    tl = ROOT / "tools.local.json"
    if tl.exists():
        d = json.loads(tl.read_text(encoding="utf-8"))
        p = d.get("ffmpeg") or (d.get("tools") or {}).get("ffmpeg")
        if p and Path(p).exists():
            return str(p)
    return shutil.which("ffmpeg") or "ffmpeg"


def main():
    if FONT is None:
        sys.exit("no Helvetica/Arial/DejaVu font found")
    c = load(sys.argv[1] if len(sys.argv) > 1 and not sys.argv[1].startswith("--") else here / "piece_glass")
    S = scene(c)
    stills = [int(x) for x in sys.argv[sys.argv.index("--still") + 1].split(",")] if "--still" in sys.argv else None
    out_dir_name = sys.argv[sys.argv.index("--out") + 1] if "--out" in sys.argv else "out_v2"
    make_lms = "--lms" in sys.argv

    N, FPS, f = c["N"], float(c["fps"]), float(S["f"])
    CW, CH = c["W"], c["H"]
    g = c.get("glass", {})
    BD = c.get("breakdown", {})

    PROCESS = ["normal", "refract", "caustic", "comp"]
    LABELS = [
        "01 COLMAP CAMERA TRACK + WORLD ALIGN",
        "02 3D SDF GEOMETRY / NORMALS (OpenGL)",
        "03 OPTICAL WARP / REFRACTION (OpenGL)",
        "04 FLOOR CAUSTICS / SHADOW (OpenGL)",
        "05 LIQUID GLASS / COMPOSITE",
    ]

    src = lambda o: min(max(int(math.floor(o * FPS / 30 + 1e-6)), 0), N - 1)
    CMP_SRC = BD.get("compare_src", [0, 120])
    T_TRACK = int(math.ceil((CMP_SRC[1] - CMP_SRC[0]) * 30 / FPS)) if CMP_SRC[1] is not None else 90
    T_3D = T_TRACK + 90
    P0 = T_3D + 60

    STEPS = []
    o0, s0 = P0, float(BD.get("process_start_src", 60))
    for key in PROCESS:
        rate, n = BD.get("slow", {}).get(key, [1.0, STEP])
        STEPS.append((key, o0, int(n), float(rate), s0))
        o0 += int(n)
        s0 += int(n) * rate * FPS / 30
    P1 = o0
    R_HOLD = P1 + WIPE
    R_PLAY = R_HOLD + 30
    TOTAL = R_PLAY + int(math.ceil(N * 30 / FPS))

    SLOW = {k: v[0] for k, v in BD.get("slow", {}).items() if v[0] != 1.0}
    LABELS = [l + (f" / {SLOW[PROCESS[i - 1]]:g}x" if i >= 1 and PROCESS[i - 1] in SLOW else "") for i, l in enumerate(LABELS)]

    def step_at(o):
        for k, (key, a, n, rate, s_a) in enumerate(STEPS):
            if a <= o < a + n:
                return k, min(int(math.floor(s_a + (o - a) * rate * FPS / 30 + 1e-6)), N - 1)
        return len(STEPS) - 1, N - 1

    # ---- captions (RectLine, exact v7 coordinates)
    lines = {}
    def rect(text, color):
        key = (text, color)
        if key not in lines:
            bg, fg, _ = RectLine(text, FONT).render(color=color)
            im = bg.copy()
            im.alpha_composite(fg)
            lines[key] = np.asarray(im)
        return lines[key]

    def layer(stage):
        lay = np.zeros((H, W, 4), np.uint8)
        boxes = []
        def place(t, x, y, color=(0, 0, 0), right=False):
            a = rect(t, color)
            x = x - a.shape[1] if right else x
            lay[y:y + 24, x:x + a.shape[1]] = a
            boxes.append([t, x, y, a.shape[1], 24])
        if stage == "intro":
            place("RESULT", 24, 79)
            place("ORIGINAL", 24, 712)
            rows = (1221, 1248)
        else:
            place("RESULT" if stage == "result" else "Process", 24, 395)
            rows = (904, 931)
            if stage != "result":
                for j in range(int(stage) + 1):
                    place(LABELS[j], 40, 422 + 27 * j, (0, 0, 0) if j == int(stage) else GRAY)
        for t, y in zip(FOOTER, rows):
            place(t, 1056, y, right=True)
        return lay, boxes

    STAGE = {"intro": layer("intro"), "result": layer("result"), **{str(k): layer(str(k)) for k in range(len(LABELS))}}

    # ---- 3D tracking & diagnostic view
    cam = np.load(work(c, "camera.npz"))
    Psfm = S["scale"] * ((cam["pts"].astype(np.float64) - S["O"]) @ np.asarray(S["M"]).T)
    rgb = cam["rgb"] if "rgb" in cam.files else np.full((len(Psfm), 3), 200)
    cam_pts = np.asarray(S["cam"], np.float64)
    c1 = np.array(g.get("cluster1", [1.12, -7.6, 0.94, 0.86]))
    c2 = np.array(g.get("cluster2", [0.05, 3.65, 1.55, 1.28]))
    fl_pos = np.array(g.get("flute_pos", [-0.88, -2.2, 1.22]))
    fl_sz = np.array(g.get("flute_size", [0.42, 0.048, 0.88]))
    fl_yaw = math.radians(float(g.get("flute_yaw_deg", 18.0)))

    def proj(i, X):
        q = (np.atleast_2d(X) - S["cam"][i]) @ np.asarray(S["rot"][i])
        return f * q[:, :2] / q[:, 2:3] + [CW / 2, CH / 2], q[:, 2]

    def grid_axes(im, i):
        cell = c["plane"].get("grid", 0.5)
        ext = c["plane"].get("grid_n", 18)
        Lx = cell * ext
        def seg(a, b, col, th):
            t = np.linspace(0, 1, 25)[:, None]
            uv, z = proj(i, np.asarray(a) * (1 - t) + np.asarray(b) * t)
            for k in range(24):
                if z[k] > 0.05 and z[k + 1] > 0.05:
                    p0, p1 = uv[k], uv[k + 1]
                    if max(abs(p0[0]), abs(p0[1]), abs(p1[0]), abs(p1[1])) < 15000:
                        cv2.line(im, tuple(np.int32(p0)), tuple(np.int32(p1)), col, th, cv2.LINE_AA)
        for gi in range(-ext, ext + 1):
            seg((gi * cell, -Lx, 0), (gi * cell, Lx, 0), (235, 235, 235), 1)
            seg((-Lx, gi * cell, 0), (Lx, gi * cell, 0), (235, 235, 235), 1)
        seg((0, 0, 0), (2 * cell, 0, 0), (40, 40, 255), 3)
        seg((0, 0, 0), (0, 2 * cell, 0), (40, 220, 40), 3)
        seg((0, 0, 0), (0, 0, 2 * cell), (255, 120, 40), 3)

    def tracking(i):
        """01: green reprojected COLMAP SfM points + white fitted floor grid & axes simultaneously on the plate"""
        im = frame(c, i).copy()
        grid_axes(im, i)
        uv, z = proj(i, Psfm)
        for (u, v), zz in zip(uv, z):
            if zz > 0.05 and 0 <= u < CW and 0 <= v < CH:
                cv2.circle(im, (int(u), int(v)), 3, (40, 255, 40), -1, cv2.LINE_AA)
        return im

    def view3d(i, o):
        """Diagnostic 3D orbit view: COLMAP pointcloud, camera path + frustum (with live plate), floor grid, and 3D glass stations"""
        Wv, Hv, fv = 1280, 720, 820.0
        im = np.zeros((Hv, Wv, 3), np.uint8)
        im[:] = np.linspace(104, 68, Hv)[:, None, None].astype(np.uint8)
        tgt = np.array([0.0, -3.5, 1.1])
        a = math.radians(-115.0 + 26.0 * (o / 60.0))
        dist = 13.5
        eye = tgt + np.array([dist * math.cos(a), dist * math.sin(a), 5.6])
        fw = tgt - eye
        fw /= np.linalg.norm(fw)
        rt = np.cross(fw, [0, 0, 1.0])
        rt /= np.linalg.norm(rt)
        up = np.cross(rt, fw)
        Rv = np.stack([rt, -up, fw], 1)

        def pv(X):
            q = (np.atleast_2d(X) - eye) @ Rv
            return fv * q[:, :2] / np.maximum(q[:, 2:3], 1e-3) + [Wv / 2, Hv / 2], q[:, 2]

        def seg(A, B, col, th):
            uv, z = pv(np.array([A, B]))
            if (z > 0.1).all():
                cv2.line(im, tuple(np.int32(uv[0])), tuple(np.int32(uv[1])), col, th, cv2.LINE_AA)

        cell = 1.0
        for g_x in range(-8, 9):
            seg((g_x * cell, -14.0, 0), (g_x * cell, 8.0, 0), (120, 120, 220) if g_x == 0 else (180, 180, 180), 1)
        for g_y in range(-14, 9):
            seg((-8.0, g_y * cell, 0), (8.0, g_y * cell, 0), (120, 220, 120) if g_y == 0 else (180, 180, 180), 1)

        # SfM points
        uv, z = pv(Psfm)
        order = np.argsort(-z)
        for k in order[::2]:
            if z[k] > 0.2 and 0 <= uv[k, 0] < Wv and 0 <= uv[k, 1] < Hv:
                cv2.circle(im, (int(uv[k, 0]), int(uv[k, 1])), 2, tuple(int(min(255, 55 + 1.35 * x)) for x in rgb[k][::-1]), -1, cv2.LINE_AA)

        # Draw 3D wireframes of the 3 liquid-glass sculptures in world space!
        for cl in (c1, c2):
            cen, rad = cl[:3], cl[3] * 0.55
            for plane_idx in range(3):
                pts = []
                for ang in np.linspace(0, 2 * math.pi, 32):
                    v = np.zeros(3)
                    v[plane_idx] = rad * math.cos(ang)
                    v[(plane_idx + 1) % 3] = rad * math.sin(ang)
                    pts.append(cen + v)
                u_c, z_c = pv(np.array(pts))
                if (z_c > 0.2).all():
                    cv2.polylines(im, [np.int32(u_c)], True, (255, 225, 110), 2, cv2.LINE_AA)

        # Fluted slab 3D wireframe box
        ca, sa = math.cos(fl_yaw), math.sin(fl_yaw)
        Rfl = np.array([[ca, -sa, 0], [sa, ca, 0], [0, 0, 1]])
        bc = [fl_pos + Rfl @ np.array([sx * fl_sz[0], sy * fl_sz[1], sz * fl_sz[2]])
              for sz in (-1, 1) for sy in (-1, 1) for sx in (-1, 1)]
        for a_, b_ in ((0,1),(2,3),(0,2),(1,3),(4,5),(6,7),(4,6),(5,7),(0,4),(1,5),(2,6),(3,7)):
            seg(bc[a_], bc[b_], (255, 225, 110), 2)

        # Camera path + live frustum with warped plate
        path, zp = pv(cam_pts)
        for k in range(len(path) - 1):
            if zp[k] > 0.1 and zp[k + 1] > 0.1:
                cv2.line(im, tuple(np.int32(path[k])), tuple(np.int32(path[k + 1])), (0, 225, 255), 3, cv2.LINE_AA)
        ro, R = np.asarray(S["cam"][i]), np.asarray(S["rot"][i])
        dpt = 1.6
        corners = [ro + R @ np.array([sx * CW / 2 / f * dpt, sy * CH / 2 / f * dpt, dpt]) for sx, sy in ((-1, -1), (1, -1), (1, 1), (-1, 1))]
        cu, cz = pv(np.array(corners))
        if (cz > 0.1).all():
            Mw = cv2.getPerspectiveTransform(np.float32([[0, 0], [CW, 0], [CW, CH], [0, CH]]), np.float32(cu))
            plate = cv2.warpPerspective(frame(c, i), Mw, (Wv, Hv))
            m = cv2.warpPerspective(np.full((CH, CW), 255, np.uint8), Mw, (Wv, Hv)) > 0
            im[m] = plate[m]
            for k in range(4):
                seg(corners[k], corners[(k + 1) % 4], (60, 255, 60), 2)
                seg(ro, corners[k], (60, 255, 60), 2)
        return im

    def passimg(key, i):
        p = work(c, "comp", f"{i:04d}.png") if key == "comp" else work(c, "pass", key, f"{i:04d}.jpg")
        im = cv2.imread(str(p))
        if im is None:
            sys.exit(f"missing {p}: run render_glass.py --passes first")
        return im

    def get(key, i, o=0):
        im = {
            "original": lambda: frame(c, i),
            "result": lambda: passimg("comp", i),
            "track": lambda: tracking(i),
            "view3d": lambda: view3d(i, o),
        }.get(key, lambda: passimg(key, i))()
        return cv2.cvtColor(cv2.resize(im, (PIC[2], PIC[3]), interpolation=cv2.INTER_AREA), cv2.COLOR_BGR2RGB)

    def wipe(old, new, k):
        if k <= 0: return old
        if k >= WIPE - 1: return new
        e = round(PIC[2] * (1 - k / (WIPE - 1)))
        pic = old.copy()
        pic[:, e:] = new[:, e:]
        pic[:, max(0, e - 1):e + 1] = 255
        return pic

    def make(o):
        can = np.full((H, W, 3), BG, np.uint8)
        edl = {}
        if o < T_TRACK:
            j = min(CMP_SRC[0] + src(o), N - 1)
            st = "intro"
            edl = dict(view="compare", src=j)
            for (x, y, w, h), key in zip(CMP, ("result", "original")):
                can[y:y + h, x:x + w] = get(key, j)
        else:
            if o < P0:
                j = src(o - T_TRACK)
                st = "0"
                if o < T_3D:
                    pic = get("track", j)
                    edl = dict(view="track+grid", src=j)
                else:
                    pic = get("view3d", j, o - T_3D)
                    edl = dict(view="3d", src=j)
                    if o - T_3D < WIPE:
                        pic = wipe(get("track", j), pic, o - T_3D)
                        edl["wipe"] = o - T_3D
            elif o < P1:
                k, j = step_at(o)
                st = str(k + 1)
                pic = get(PROCESS[k], j)
                edl = dict(view=PROCESS[k], src=j, rate=STEPS[k][3])
                if o - STEPS[k][1] < WIPE:
                    prev = get("view3d", src(o - T_TRACK), o - T_3D) if k == 0 else get(PROCESS[k - 1], j)
                    pic = wipe(prev, pic, o - STEPS[k][1])
                    edl["wipe"] = o - STEPS[k][1]
            elif o < R_HOLD:
                st = "result"
                pic = wipe(get("comp", step_at(P1 - 1)[1]), get("result", 0), o - P1)
                edl = dict(view="wipe->result", src=0, wipe=o - P1)
            else:
                j = 0 if o < R_PLAY else src(o - R_PLAY)
                st = "result"
                pic = get("result", j)
                edl = dict(view="result", src=j)
            can[PIC[1]:PIC[1] + PIC[3], :] = pic
        lay = STAGE[st][0]
        m = lay[:, :, 3] > 0
        can[m] = lay[:, :, :3][m]
        edl["stage"] = st
        return can, edl

    if stills:
        sheet = np.hstack([make(o)[0] for o in stills])
        p = work(c, "bd_glass_sheet.jpg")
        cv2.imwrite(str(p), cv2.cvtColor(sheet, cv2.COLOR_RGB2BGR)[::2, ::2], [cv2.IMWRITE_JPEG_QUALITY, 90])
        print("stills", stills, "->", p)
        return

    out = c["dir"] / out_dir_name
    out.mkdir(exist_ok=True)
    mp4 = out / f"{c['name']}_v2_Breakdown_Final_1080x1350.mp4"
    if mp4.exists():
        sys.exit(f"{mp4} exists: choose a fresh --out folder (outputs are never overwritten)")
    ff = find_ffmpeg()
    enc = subprocess.Popen([
        ff, "-v", "error", "-y", "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{W}x{H}", "-r", "30", "-i", "-",
        "-c:v", "libx264", "-profile:v", "high", "-level", "4.1", "-crf", "14", "-preset", "slow",
        "-pix_fmt", "yuv420p", "-color_primaries", "bt709", "-color_trc", "bt709", "-colorspace", "bt709",
        "-movflags", "+faststart", str(mp4),
    ], stdin=subprocess.PIPE)
    edls = []
    for o in range(TOTAL):
        im, e = make(o)
        enc.stdin.write(im.tobytes())
        edls.append(e)
        if o % 120 == 0 or o == TOTAL - 1:
            print(f"bd frame {o}/{TOTAL}", flush=True)
    enc.stdin.close()
    enc.wait()
    if enc.returncode != 0:
        sys.exit(f"ffmpeg failed rc={enc.returncode}")

    meta = dict(
        file=mp4.name, size=[W, H], fps=30, frames=TOTAL, seconds=TOTAL / 30, font=FONT,
        rules="docs/breakdown/README.md v7", labels=LABELS, footer=FOOTER,
        timeline=dict(
            compare=[0, T_TRACK - 1], track_grid=[T_TRACK, T_3D - 1], wipe_to_3d=[T_3D, T_3D + WIPE - 1],
            view3d=[T_3D + WIPE, P0 - 1], process=[P0, P1 - 1],
            process_steps={k: [a, a + n - 1] for k, a, n, r, s_ in STEPS},
            step_rates={k: r for k, a, n, r, s_ in STEPS},
            compare_source=[CMP_SRC[0], CMP_SRC[1]],
            wipe_to_result=[P1, R_HOLD - 1], hold_result_frame0=[R_HOLD, R_PLAY - 1], result_replay=[R_PLAY, TOTAL - 1],
        ),
        layout={k: v[1] for k, v in STAGE.items()}, edit=edls,
        sha256=hashlib.sha256(mp4.read_bytes()).hexdigest(),
    )
    mp4.with_suffix(".json").write_text(json.dumps(meta, indent=1), encoding="utf-8")
    print("->", mp4, TOTAL, "frames")

    if make_lms and LMS_MOV.exists():
        # Match Breakdown_x2_LMS end-card spec:
        # 1) 1.0s dissolve where LMS logo holds frame 0, then plays full typing motion, then holds final lms.hyun2.cloud 2.5s
        # 2) Instagram-safe H.264 High 4.1 + silent AAC 48kHz + faststart
        lms_out = out / f"{c['name']}_v2_Breakdown_LMS_1080x1350.mp4"
        bd_dur = TOTAL / 30.0
        off = bd_dur - 1.0
        # Pad start of logo by 1.0s (clone frame 0 so full typing starts after the 1s dissolve!) and end by 2.5s hold
        fc = (
            "[1:v]scale=1080:-2:flags=lanczos,format=rgba,tpad=start_duration=1.0:start_mode=clone:stop_duration=2.5:stop_mode=clone[lg];"
            "color=c=#fcfefd:s=1080x1350:r=30:d=11.17[bg];"
            "[bg][lg]overlay=(W-w)/2:(H-h)/2:shortest=1,format=yuv420p,fps=30[endcard];"
            "[0:v]format=yuv420p,fps=30[v0];"
            f"[v0][endcard]xfade=transition=fade:duration=1.0:offset={off:.3f}[vout]"
        )
        subprocess.run([
            ff, "-v", "error", "-y",
            "-i", str(mp4), "-i", str(LMS_MOV),
            "-f", "lavfi", "-i", "anullsrc=r=48000:cl=stereo",
            "-filter_complex", fc,
            "-map", "[vout]", "-map", "2:a", "-shortest",
            "-c:v", "libx264", "-profile:v", "high", "-level", "4.1", "-crf", "12", "-maxrate", "9M", "-bufsize", "18M", "-g", "60",
            "-pix_fmt", "yuv420p", "-color_primaries", "bt709", "-color_trc", "bt709", "-colorspace", "bt709",
            "-c:a", "aac", "-b:a", "128k", "-movflags", "+faststart",
            str(lms_out),
        ], check=True)
        print("-> LMS:", lms_out, f"({lms_out.stat().st_size/1e6:.2f} MB)")


if __name__ == "__main__":
    main()
