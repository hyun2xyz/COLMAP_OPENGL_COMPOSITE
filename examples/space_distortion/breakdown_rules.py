"""Space-distortion breakdown by the house rules: docs/breakdown/README.md (v7) and spec.json, as the later reels use them
(TOS UE5, LTX hair alpha, street boxing): white #fcfefd page, 1080x1350 @30fps, sharp white 21px caption boxes, 27px leading,
the two fixed footer lines, right-to-left linear 9-frame wipes with a 2px white edge.

    python breakdown_rules.py [piece_dir] [--still 30,120,250]

Default timeline below; piece.json "breakdown" can set the comparison's source range, the process start frame and slower,
longer steps (e.g. the warp grid at 0.5x for 4 s), which shift the later frames.
Timeline (output frames at 30 fps; src(o) = source frame at o/30 s, clamped):
  0-89      RESULT (top) / ORIGINAL (bottom), same source frames
  90-179    01 camera: green tracked points + floor grid/axes on the plate, in the SAME frames (3 s)
  180-188   wipe to the 3D diagnostic view: SfM points, camera path + frustum (with the plate in it), floor grid, lifted skeleton
  189-239   3D view (its own orbit camera; this is a view change, not a process step)
  240-419   one continuous process clock src(o - 240), a step every 30 frames, each entered by a wipe:
            02 pose -> 3D limbs, 03 distortion field (thermal), 04 warp grid, 05 glass normals, 06 spectral prism, 07 composite
  420-428   wipe to RESULT source frame 0 (label RESULT), 429-458 hold that frame 1 s, then the whole result from frame 0
Needs render.py to have run (work/comp, work/pass/{heat,grid,normal,prism}) and slv.py solve/plane, trails.py.
out: out/<name>_Breakdown_Final_1080x1350.mp4 (never overwritten), plus <same>.json (edit decisions, layout, font, checks)."""
import json, math, subprocess, sys, hashlib
from pathlib import Path
import numpy as np, cv2
from PIL import Image

here = Path(__file__).resolve().parent; ROOT = here.parents[1]
sys.path.insert(0, str(here.parent / "shader_live" / "pipeline")); sys.path.insert(0, str(ROOT / "tools" / "breakdown"))
from slv import load, work, frame, scene   # noqa: E402
from captions import RectLine              # noqa: E402

W, H = 1080, 1350; PIC = (0, 371, 1080, 608); CMP = [(0, 55, 1080, 608), (0, 688, 1080, 608)]
BG = (252, 254, 253); GRAY = (125, 128, 126); WIPE = 9; STEP = 30
FOOTER = ["Only an LLM + Using My Custom Skills (Colmap + OpenGL, etc.)", "See Caption for Details!"]
FONT = next((p for p in ["/System/Library/Fonts/Helvetica.ttc", "C:/Windows/Fonts/arial.ttf", "/System/Library/Fonts/Supplemental/Arial.ttf",
                         "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"] if Path(p).is_file()), None)
BONES = [(11, 12), (11, 13), (13, 15), (15, 19), (12, 14), (14, 16), (16, 20), (11, 23), (12, 24), (23, 24), (23, 25), (25, 27), (27, 31), (24, 26), (26, 28), (28, 32), (0, 11), (0, 12)]
PROCESS = ["pose", "heat", "grid", "normal", "prism", "comp"]

c = load(sys.argv[1] if len(sys.argv) > 1 and not sys.argv[1].startswith("--") else here / "piece"); S = scene(c)
stills = [int(x) for x in sys.argv[sys.argv.index("--still") + 1].split(",")] if "--still" in sys.argv else None
N, FPS, f = c["N"], float(c["fps"]), float(S["f"]); CW, CH = c["W"], c["H"]
homog = c.get("solver") == "homography"
LABELS = ["01 CAMERA SOLVE (HOMOGRAPHY) + WORLD ALIGN" if homog else "01 COLMAP CAMERA TRACK + WORLD ALIGN",
          "02 MEDIAPIPE POSE / 3D LIMBS", "03 AIR TRAILS + SOUND WAVES (OpenGL)", "04 WARP GRID / REFRACTION",
          "05 GLASS NORMALS (OpenGL)", "06 SPECTRAL PRISM (CAUCHY + CIE)", "07 DISTORTION / COMPOSITE"]
if c.get("breakdown", {}).get("steps"):                                          # other effects: their own passes (work/pass/<key>) and captions
    PROCESS = [k for k, _ in c["breakdown"]["steps"]]; LABELS = LABELS[:1] + [l for _, l in c["breakdown"]["steps"]]
src = lambda o: min(max(int(math.floor(o * FPS / 30 + 1e-6)), 0), N - 1)
# per-piece edit (piece.json "breakdown"): which source frames the opening comparison plays, where the process clock starts,
# and steps shown slower and longer, e.g. {"compare_src": [40, 136], "process_start_src": 31, "slow": {"grid": [0.5, 120]}}
BD = c.get("breakdown", {})
CMP_SRC = BD.get("compare_src", [0, None])
T_TRACK = int(math.ceil((CMP_SRC[1] - CMP_SRC[0]) * 30 / FPS)) if CMP_SRC[1] is not None else 90
T_3D = T_TRACK + 90; P0 = T_3D + 60
STEPS = []                                                                       # (key, first output frame, frames, rate, source frame at start)
o0, s0 = P0, float(BD.get("process_start_src", 0))
for key in PROCESS:
    rate, n = BD.get("slow", {}).get(key, [1.0, STEP]); STEPS.append((key, o0, int(n), float(rate), s0)); o0 += int(n); s0 += int(n) * rate * FPS / 30
P1 = o0; R_HOLD = P1 + WIPE; R_PLAY = R_HOLD + 30
TOTAL = R_PLAY + int(math.ceil(N * 30 / FPS))
def step_at(o):
    for k, (key, a, n, rate, s_a) in enumerate(STEPS):
        if a <= o < a + n: return k, min(int(math.floor(s_a + (o - a) * rate * FPS / 30 + 1e-6)), N - 1)
    return len(STEPS) - 1, N - 1
SLOW = {k: v[0] for k, v in BD.get("slow", {}).items() if v[0] != 1.0}
LABELS = [l + (" / %gx" % SLOW[PROCESS[i - 1]] if i >= 1 and PROCESS[i - 1] in SLOW else "") for i, l in enumerate(LABELS)]

# ---------------------------------------------------------------- captions (tools/breakdown/captions.py, fixed rectangles)
lines = {}
def rect(text, color):
    key = (text, color)
    if key not in lines:
        bg, fg, _ = RectLine(text, FONT).render(color=color); im = bg.copy(); im.alpha_composite(fg); lines[key] = np.asarray(im)
    return lines[key]
def layer(stage):
    lay = np.zeros((H, W, 4), np.uint8); boxes = []
    def place(t, x, y, color=(0, 0, 0), right=False):
        a = rect(t, color); x = x - a.shape[1] if right else x; lay[y:y + 24, x:x + a.shape[1]] = a; boxes.append([t, x, y, a.shape[1], 24])
    if stage == "intro":
        place("RESULT", 24, 79); place("ORIGINAL", 24, 712); rows = (1221, 1248)
    else:
        place("RESULT" if stage == "result" else "Process", 24, 395); rows = (904, 931)
        if stage != "result":
            for j in range(int(stage) + 1): place(LABELS[j], 40, 422 + 27 * j, (0, 0, 0) if j == int(stage) else GRAY)
    for t, y in zip(FOOTER, rows): place(t, 1056, y, right=True)
    return lay, boxes

# ---------------------------------------------------------------- passes
L = np.load(work(c, "limbs.npz")); P3 = L["P"].astype(np.float64)
cam = np.load(work(c, "camera.npz")); Psfm = S["scale"] * ((cam["pts"].astype(np.float64) - S["O"]) @ np.asarray(S["M"]).T)
rgb = cam["rgb"] if "rgb" in cam.files else np.full((len(Psfm), 3), 200)
def proj(i, X):                                                                   # plane coords -> plate pixels, depth
    q = (np.atleast_2d(X) - S["cam"][i]) @ np.asarray(S["rot"][i]); return f * q[:, :2] / q[:, 2:3] + [CW / 2, CH / 2], q[:, 2]
def grid_axes(im, i):
    cell = c["plane"].get("grid", 0.5); ext = c["plane"].get("grid_n", 10); Lx = cell * ext
    def seg(a, b, col, th):
        t = np.linspace(0, 1, 25)[:, None]; uv, z = proj(i, np.asarray(a) * (1 - t) + np.asarray(b) * t)
        for k in range(24):
            if z[k] > 0.05 and z[k + 1] > 0.05: cv2.line(im, tuple(np.int32(uv[k])), tuple(np.int32(uv[k + 1])), col, th, cv2.LINE_AA)
    for g in range(-ext, ext + 1):
        seg((g * cell, -Lx, 0), (g * cell, Lx, 0), (235, 235, 235), 1); seg((-Lx, g * cell, 0), (Lx, g * cell, 0), (235, 235, 235), 1)
    seg((0, 0, 0), (2 * cell, 0, 0), (40, 40, 255), 3); seg((0, 0, 0), (0, 2 * cell, 0), (40, 220, 40), 3); seg((0, 0, 0), (0, 0, 2 * cell), (255, 120, 40), 3)
def tracking(i):
    """01: green = the solved 3D points re-projected by this frame's camera (the track itself), white grid = the fitted floor"""
    im = frame(c, i).copy(); grid_axes(im, i); uv, z = proj(i, Psfm)
    for (u, v), zz in zip(uv, z):
        if zz > 0.05 and 0 <= u < CW and 0 <= v < CH: cv2.circle(im, (int(u), int(v)), 4, (40, 255, 40), -1, cv2.LINE_AA)
    return im
def skeleton(img, uv, col, th):
    for a, b in BONES: cv2.line(img, tuple(np.int32(uv[a])), tuple(np.int32(uv[b])), col, th, cv2.LINE_AA)
def pose(i):
    """02: the MediaPipe joints lifted into plane coordinates (trails.py) and projected back by the solved camera"""
    im = frame(c, i).copy(); uv, z = proj(i, P3[i]); skeleton(im, uv, (40, 255, 40), 4)
    for j in (13, 14, 15, 16, 19, 20, 25, 26): cv2.circle(im, tuple(np.int32(uv[j])), 8, (255, 255, 255), -1, cv2.LINE_AA)
    return im
hip0 = 0.5 * (P3[:, 23] + P3[:, 24]).mean(0)
def view3d(i, o):
    """diagnostic 3D view: orbiting look-at camera; SfM points in their colours, floor grid, camera path, frustum with the plate"""
    Wv, Hv, fv = 1280, 720, 900.0; im = np.zeros((Hv, Wv, 3), np.uint8)
    im[:] = np.linspace(108, 72, Hv)[:, None, None].astype(np.uint8)
    tgt = hip0 + [0, 0, 0.4]; c0 = np.asarray(S["cam"][0]); d0 = c0 - tgt; d0[2] = 0; r0 = np.linalg.norm(d0); a0 = math.atan2(d0[1], d0[0])
    a = a0 + math.radians(34 + 12 * (o / 60.0)); dist = r0 * 1.25
    eye = tgt + [dist * math.cos(a), dist * math.sin(a), 1.9]; fw = tgt - eye; fw /= np.linalg.norm(fw)
    rt = np.cross(fw, [0, 0, 1.0]); rt /= np.linalg.norm(rt); up = np.cross(rt, fw); Rv = np.stack([rt, -up, fw], 1)
    def pv(X):
        q = (np.atleast_2d(X) - eye) @ Rv; return fv * q[:, :2] / np.maximum(q[:, 2:3], 1e-3) + [Wv / 2, Hv / 2], q[:, 2]
    def seg(A, B, col, th):
        uv, z = pv(np.array([A, B]))
        if (z > 0.1).all(): cv2.line(im, tuple(np.int32(uv[0])), tuple(np.int32(uv[1])), col, th, cv2.LINE_AA)
    cell = 0.5; ext = 12
    for g in range(-ext, ext + 1):
        col = (90, 90, 230) if g == 0 else (200, 200, 200)
        seg((g * cell, -ext * cell, 0), (g * cell, ext * cell, 0), col, 1); seg((-ext * cell, g * cell, 0), (ext * cell, g * cell, 0), col, 1)
    uv, z = pv(Psfm); order = np.argsort(-z)
    for k in order:
        if z[k] > 0.1: cv2.circle(im, tuple(np.int32(uv[k])), 4, tuple(int(min(255, 60 + 1.5 * x)) for x in rgb[k][::-1]), -1, cv2.LINE_AA)
    path, zp = pv(np.asarray(S["cam"]))
    for k in range(len(path) - 1):
        if zp[k] > 0.1 and zp[k + 1] > 0.1: cv2.line(im, tuple(np.int32(path[k])), tuple(np.int32(path[k + 1])), (0, 220, 255), 2, cv2.LINE_AA)
    ro, R = np.asarray(S["cam"][i]), np.asarray(S["rot"][i]); dpt = 1.3
    corners = [ro + R @ np.array([sx * CW / 2 / f * dpt, sy * CH / 2 / f * dpt, dpt]) for sx, sy in ((-1, -1), (1, -1), (1, 1), (-1, 1))]
    cu, cz = pv(np.array(corners))
    if (cz > 0.1).all():
        Mw = cv2.getPerspectiveTransform(np.float32([[0, 0], [CW, 0], [CW, CH], [0, CH]]), np.float32(cu))
        plate = cv2.warpPerspective(frame(c, i), Mw, (Wv, Hv)); m = cv2.warpPerspective(np.full((CH, CW), 255, np.uint8), Mw, (Wv, Hv)) > 0; im[m] = plate[m]
        for k in range(4): seg(corners[k], corners[(k + 1) % 4], (60, 255, 60), 2); seg(ro, corners[k], (60, 255, 60), 2)
    su, sz = pv(P3[i])
    if (sz > 0.1).all(): skeleton(im, su, (245, 245, 245), 3)
    return im
def passimg(key, i):
    p = work(c, "comp", f"{i:04d}.png") if key == "comp" else work(c, "pass", key, f"{i:04d}.jpg"); im = cv2.imread(str(p))
    if im is None: sys.exit(f"missing {p}: run render.py first")
    return im
def get(key, i, o=0):
    im = {"original": lambda: frame(c, i), "result": lambda: passimg("comp", i), "track": lambda: tracking(i), "view3d": lambda: view3d(i, o),
          "pose": lambda: pose(i)}.get(key, lambda: passimg(key, i))()
    return cv2.cvtColor(cv2.resize(im, (PIC[2], PIC[3]), interpolation=cv2.INTER_AREA), cv2.COLOR_BGR2RGB)
def wipe(old, new, k):
    """right to left, linear: k = 0..8, edge x = (1 - k/8) * 1080; 2px white edge inside the picture"""
    if k <= 0: return old
    if k >= WIPE - 1: return new
    e = round(PIC[2] * (1 - k / (WIPE - 1))); pic = old.copy(); pic[:, e:] = new[:, e:]; pic[:, max(0, e - 1):e + 1] = 255; return pic

# ---------------------------------------------------------------- edit
STAGE = {"intro": layer("intro"), "result": layer("result"), **{str(k): layer(str(k)) for k in range(len(LABELS))}}
def make(o):
    can = np.full((H, W, 3), BG, np.uint8); edl = {}
    if o < T_TRACK:
        j = min(CMP_SRC[0] + src(o), N - 1); st = "intro"; edl = dict(view="compare", src=j)
        for (x, y, w, h), key in zip(CMP, ("result", "original")): can[y:y + h, x:x + w] = get(key, j)
    else:
        if o < P0:                                                                # 01: tracking (3 s) then the 3D view
            j = src(o - T_TRACK); st = "0"
            if o < T_3D: pic = get("track", j); edl = dict(view="track+grid", src=j)
            else:
                pic = get("view3d", j, o - T_3D); edl = dict(view="3d", src=j)
                if o - T_3D < WIPE: pic = wipe(get("track", j), pic, o - T_3D); edl["wipe"] = o - T_3D
        elif o < P1:
            k, j = step_at(o); st = str(k + 1); pic = get(PROCESS[k], j); edl = dict(view=PROCESS[k], src=j, rate=STEPS[k][3])
            if o - STEPS[k][1] < WIPE:                                            # the same source frame on both sides of the edge
                prev = get("view3d", src(o - T_TRACK), o - T_3D) if k == 0 else get(PROCESS[k - 1], j)
                pic = wipe(prev, pic, o - STEPS[k][1]); edl["wipe"] = o - STEPS[k][1]
        elif o < R_HOLD:
            st = "result"; pic = wipe(get("comp", step_at(P1 - 1)[1]), get("result", 0), o - P1); edl = dict(view="wipe->result", src=0, wipe=o - P1)
        else:
            j = 0 if o < R_PLAY else src(o - R_PLAY); st = "result"; pic = get("result", j); edl = dict(view="result", src=j)
        can[PIC[1]:PIC[1] + PIC[3], :] = pic
    lay = STAGE[st][0]; m = lay[:, :, 3] > 0; can[m] = lay[:, :, :3][m]; edl["stage"] = st
    return can, edl

if __name__ == "__main__":
    if FONT is None: sys.exit("no Helvetica/Arial/DejaVu font found")
    if stills:
        sheet = np.hstack([make(o)[0] for o in stills]); p = work(c, "bd_rules_sheet.jpg")
        cv2.imwrite(str(p), cv2.cvtColor(sheet, cv2.COLOR_RGB2BGR)[::2, ::2], [cv2.IMWRITE_JPEG_QUALITY, 90]); print("stills", stills, "->", p); sys.exit()
    out = c["dir"] / "out"; out.mkdir(exist_ok=True); mp4 = out / f"{c['name']}_Breakdown_Final_1080x1350.mp4"
    if mp4.exists(): sys.exit(f"{mp4} exists: move it away first (outputs are never overwritten)")
    enc = subprocess.Popen(["ffmpeg", "-v", "error", "-y", "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{W}x{H}", "-r", "30", "-i", "-", "-c:v", "libx264", "-crf", "16",
                            "-preset", "slow", "-pix_fmt", "yuv420p", "-color_primaries", "bt709", "-color_trc", "bt709", "-colorspace", "bt709", "-movflags", "+faststart", str(mp4)],
                           stdin=subprocess.PIPE)
    edls = []
    for o in range(TOTAL):
        im, e = make(o); enc.stdin.write(im.tobytes()); edls.append(e)
        if o % 120 == 0: print("frame", o, "/", TOTAL, flush=True)
    enc.stdin.close(); enc.wait()
    meta = dict(file=mp4.name, size=[W, H], fps=30, frames=TOTAL, seconds=TOTAL / 30, font=FONT, rules="docs/breakdown/README.md v7", labels=LABELS, footer=FOOTER,
                timeline=dict(compare=[0, T_TRACK - 1], track_grid=[T_TRACK, T_3D - 1], wipe_to_3d=[T_3D, T_3D + WIPE - 1], view3d=[T_3D + WIPE, P0 - 1],
                              process=[P0, P1 - 1], process_steps={k: [a, a + n - 1] for k, a, n, r, s_ in STEPS}, step_rates={k: r for k, a, n, r, s_ in STEPS},
                              compare_source=[CMP_SRC[0], CMP_SRC[1]],
                              wipe_to_result=[P1, R_HOLD - 1], hold_result_frame0=[R_HOLD, R_PLAY - 1], result_replay=[R_PLAY, TOTAL - 1]),
                layout={k: v[1] for k, v in STAGE.items()}, edit=edls, sha256=hashlib.sha256(mp4.read_bytes()).hexdigest())
    mp4.with_suffix(".json").write_text(json.dumps(meta, indent=1)); print("->", mp4, TOTAL, "frames", "rc", enc.returncode)
