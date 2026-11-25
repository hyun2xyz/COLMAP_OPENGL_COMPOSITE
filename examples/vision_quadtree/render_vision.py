"""3D Surveillance Vision · Adaptive Quadtree & Metric Blob-Tracking HUD (ModernGL, OpenGL 3.3).

    python render_vision.py [piece_dir] --still 35,125,215    quick look -> work/still_sheet.jpg
    python render_vision.py [piece_dir]                        full render -> out/*_Composite.mp4, *_Compare_1080x1350.mp4

Combines a real-time GPU Adaptive Quadtree + Glyph/Loupe cell shader (GLSL 3.30, exact 120->60->30->15 px hierarchy
dividing 1920x1080 cleanly) driven by body-gated dense optical flow + silhouette edges + 3D joint velocities, layered
with a camera-calibrated 3D perspective bounding cage, 3D floor reticle & foot trajectory, multi-node blob connection
web, and a smooth turret-tracked monochrome Picture-in-Picture inset framed in the open architectural bay.
"""
import json, math, shutil, subprocess, sys, time
from pathlib import Path
import numpy as np, cv2, moderngl
from PIL import Image, ImageDraw, ImageFont

here = Path(__file__).resolve().parent
root = here.parents[1]
sys.path.insert(0, str(here.parent / "shader_live" / "pipeline"))
from slv import load, work, frame, scene  # noqa: E402


def find_ffmpeg():
    tl = root / "tools.local.json"
    if tl.exists():
        d = json.loads(tl.read_text(encoding="utf-8"))
        p = d.get("ffmpeg") or (d.get("tools") or {}).get("ffmpeg")
        if p and Path(p).exists():
            return str(p)
    return shutil.which("ffmpeg") or "ffmpeg"


def load_mono_font(size, bold=False):
    cands = (
        (r"C:\Windows\Fonts\consolab.ttf", r"C:\Windows\Fonts\consola.ttf", r"C:\Windows\Fonts\cascadiamono.ttf")
        if bold else
        (r"C:\Windows\Fonts\consola.ttf", r"C:\Windows\Fonts\cascadiamono.ttf", r"C:\Windows\Fonts\lucon.ttf")
    )
    for p in cands:
        if Path(p).exists():
            return ImageFont.truetype(p, size)
    return ImageFont.load_default()


VS = "#version 330\nin vec2 p; void main(){ gl_Position = vec4(p, 0.0, 1.0); }"

FS = """#version 330
uniform sampler2D uPlate;
uniform sampler2D uEnergy;    /* R: motion+edge energy [0..1], G: body mask [0..1], B: peak speed [0..1] */
uniform vec2 uRes;
uniform vec2 uFacePx;         /* face center in top-down pixels */
uniform float uFaceRad;       /* face protection radius in pixels */
uniform float iTime;
uniform float uBaseCell;      /* 120.0 -> divides 1920x1080 into exact 16x9 */
uniform float uSplitThresh;
uniform vec3 uAccent;
uniform int uMode;            /* 0 = composite, 1 = quadtree energy diagnostic */

layout(location=0) out vec4 oOut;

float hash21(vec2 p) {
  p = fract(p * vec2(123.34, 456.21));
  p += dot(p, p + 45.32);
  return fract(p.x * p.y);
}

/* crisp 5x5 procedural bitmap glyph for Code Brutalism glyph-dither cells */
float glyphBit(int idx, ivec2 g) {
  if (idx <= 0) return 0.0;
  if (idx == 1) return (g.x == 2 && g.y == 2) ? 1.0 : 0.0;
  if (idx == 2) return (g.x == 2 && (g.y == 1 || g.y == 3)) ? 1.0 : 0.0;
  if (idx == 3) return ((g.x == 2 && g.y >= 1 && g.y <= 3) || (g.y == 2 && g.x >= 1 && g.x <= 3)) ? 1.0 : 0.0;
  if (idx == 4) return (abs(g.x - 2) == abs(g.y - 2) && abs(g.x - 2) <= 1) ? 1.0 : 0.0;
  bool vline = (g.x == 1 || g.x == 3);
  bool hline = (g.y == 1 || g.y == 3);
  return (vline || hline) ? 1.0 : 0.0;
}

vec3 sampleCell(vec2 cMin, float sz) {
  vec2 c0 = (cMin + vec2(0.50, 0.50) * sz) / uRes;
  vec2 c1 = (cMin + vec2(0.20, 0.20) * sz) / uRes;
  vec2 c2 = (cMin + vec2(0.80, 0.20) * sz) / uRes;
  vec2 c3 = (cMin + vec2(0.20, 0.80) * sz) / uRes;
  vec2 c4 = (cMin + vec2(0.80, 0.80) * sz) / uRes;
  vec3 e0 = texture(uEnergy, clamp(c0, 0.001, 0.999)).rgb;
  vec3 e1 = texture(uEnergy, clamp(c1, 0.001, 0.999)).rgb;
  vec3 e2 = texture(uEnergy, clamp(c2, 0.001, 0.999)).rgb;
  vec3 e3 = texture(uEnergy, clamp(c3, 0.001, 0.999)).rgb;
  vec3 e4 = texture(uEnergy, clamp(c4, 0.001, 0.999)).rgb;
  vec3 mx = max(e0, max(max(e1, e2), max(e3, e4)));
  vec3 av = (e0 + e1 + e2 + e3 + e4) * 0.2;
  return mix(av, mx, 0.70);
}

void main() {
  vec2 px = vec2(gl_FragCoord.x, uRes.y - gl_FragCoord.y);
  vec2 uv = px / uRes;
  vec3 orig = texture(uPlate, uv).rgb;
  float origLuma = dot(orig, vec3(0.299, 0.587, 0.114));

  /* sleek cool editorial base grade: crisp concrete, natural dancer tone */
  float bodyMask = texture(uEnergy, uv).g;
  vec3 bgGrade = mix(vec3(origLuma), orig, 0.74) * vec3(0.95, 0.97, 0.99);
  bgGrade = (bgGrade - 0.5) * 1.07 + 0.49;
  vec3 fgGrade = mix(vec3(origLuma), orig, 0.94);
  vec3 baseCol = clamp(mix(bgGrade, fgGrade, smoothstep(0.15, 0.6, bodyMask)), 0.0, 1.0);

  /* recursive 4-level adaptive quadtree: 120 -> 60 -> 30 -> 15 px */
  float sz = uBaseCell;
  vec2 cMin = floor(px / sz) * sz;
  int lvl = 0;
  vec3 cellInfo = sampleCell(cMin, sz);

  float thr[3] = float[3](uSplitThresh, uSplitThresh * 1.75, uSplitThresh * 2.85);
  for (int k = 0; k < 3; k++) {
    float score = cellInfo.r + cellInfo.b * 0.55;
    if (score > thr[k]) {
      sz *= 0.5;
      vec2 sub = step(cMin + sz, px);
      cMin += sub * sz;
      lvl = k + 1;
      cellInfo = sampleCell(cMin, sz);
    } else {
      break;
    }
  }

  vec2 localPx = px - cMin;
  vec2 localUV = localPx / sz;
  vec2 cellCenter = cMin + 0.5 * sz;
  float cellRnd = hash21(floor(cMin / max(sz, 1.0)) + floor(iTime * 5.0) * 0.07);
  float cellStaticRnd = hash21(floor(cMin / max(sz, 1.0)));

  float faceDist = length(cellCenter - uFacePx);
  float faceSafe = smoothstep(uFaceRad * 0.85, uFaceRad * 1.45, faceDist);
  float peakVel = cellInfo.b;

  if (uMode == 1) {
    float border = (localPx.x < 1.5 || localPx.y < 1.5) ? 1.0 : 0.0;
    vec3 heat = mix(vec3(0.05, 0.07, 0.10), vec3(0.22, 0.82, 0.94), clamp(cellInfo.r * 1.5, 0.0, 1.0));
    heat = mix(heat, uAccent, clamp(peakVel * 1.35, 0.0, 1.0));
    vec3 dbg = mix(heat, vec3(1.0), border * (0.28 + 0.22 * float(lvl)));
    oOut = vec4(dbg, 1.0);
    return;
  }

  vec3 col = baseCol;
  float outsideBody = (1.0 - smoothstep(0.15, 0.55, bodyMask)) * faceSafe;

  /* --- Cell-level Code Brutalism treatments (deliberate, high-contrast, clean) --- */
  if (lvl == 1) {
    /* 60px wake halo: rare editorial glyph-dither or crisp monochrome exposure box */
    if (outsideBody > 0.45 && cellStaticRnd < 0.22) {
      vec3 cellPlate = texture(uPlate, cellCenter / uRes).rgb;
      float cl = dot(cellPlate, vec3(0.299, 0.587, 0.114));
      /* 2x2 sub-tiles of 5x5 glyphs inside 60px cell */
      ivec2 g = ivec2(clamp(floor(fract(localUV * 2.0) * 5.0), 0.0, 4.0));
      int gIdx = int(clamp(floor((1.0 - cl) * 6.0), 0.0, 5.0));
      float bit = glyphBit(gIdx, g);
      vec3 glyphCol = mix(vec3(0.08, 0.09, 0.10), vec3(0.92, 0.94, 0.96), bit);
      col = mix(col, glyphCol, 0.62 * outsideBody);
    } else if (outsideBody > 0.45 && cellStaticRnd > 0.84) {
      col = mix(col, vec3(clamp((origLuma - 0.5) * 1.35 + 0.54, 0.0, 1.0)), 0.45 * outsideBody);
    }
  } else if (lvl == 2) {
    /* 30px active wake: crisp Glyph Dither OR 1.45x Optical Loupe magnification */
    if (outsideBody > 0.35 && cellStaticRnd < 0.34) {
      vec3 cellPlate = texture(uPlate, cellCenter / uRes).rgb;
      float cl = dot(cellPlate, vec3(0.299, 0.587, 0.114));
      ivec2 g = ivec2(clamp(floor(localUV * 5.0), 0.0, 4.0));
      int gIdx = int(clamp(floor((1.0 - cl) * 6.0), 0.0, 5.0));
      float bit = glyphBit(gIdx, g);
      vec3 glyphCol = mix(vec3(0.07, 0.08, 0.10), vec3(0.94, 0.96, 0.98), bit);
      col = mix(col, glyphCol, 0.78 * outsideBody);
    } else if (outsideBody > 0.35 && cellStaticRnd > 0.66) {
      vec2 zoomUV = (cellCenter + (localPx - 0.5 * sz) / 1.45) / uRes;
      vec3 zc = texture(uPlate, clamp(zoomUV, 0.001, 0.999)).rgb;
      float zl = dot(zc, vec3(0.299, 0.587, 0.114));
      col = mix(col, vec3(clamp((zl - 0.5) * 1.42 + 0.52, 0.0, 1.0)), 0.72 * outsideBody);
    }
  } else if (lvl == 3) {
    /* 15px peak-velocity limb wake: razor-sharp micro-inversion or brief acid-yellow velocity flash */
    float edgeW = faceSafe * (1.0 - bodyMask * 0.70);
    if (peakVel > 0.48 && cellRnd < 0.28 && edgeW > 0.25) {
      col = mix(col, uAccent, 0.68 * edgeW);
    } else if (cellStaticRnd < 0.32 && outsideBody > 0.35) {
      col = mix(col, vec3(0.08, 0.09, 0.10), 0.62 * outsideBody);
    }
  }

  /* --- Crisp Quadtree grid lines (subtle dark outline + razor bright hairline) --- */
  float lw = (lvl >= 2) ? 1.35 : 1.0;
  bool isEdge = (localPx.x < lw || localPx.y < lw);
  bool isDark = (!isEdge && (localPx.x < lw + 1.0 || localPx.y < lw + 1.0) && lvl >= 2);
  if (isDark && faceSafe > 0.4) {
    col = mix(col, vec3(0.04, 0.05, 0.06), 0.28);
  }
  if (isEdge) {
    float lineAlpha = (lvl == 0) ? 0.13 : (lvl == 1 ? 0.30 : (lvl == 2 ? 0.52 : 0.78));
    lineAlpha *= mix(0.22, 1.0, faceSafe);
    vec3 lineCol = (lvl == 3 && peakVel > 0.55 && cellRnd < 0.40) ? uAccent : vec3(0.96, 0.98, 1.0);
    col = mix(col, lineCol, lineAlpha);
  }

  /* architectural '+' ticks at 120px macro grid intersections */
  vec2 modM = mod(px + 5.0, uBaseCell) - 5.0;
  bool tick = ((abs(modM.x) < 0.9 && abs(modM.y) < 5.0) || (abs(modM.y) < 0.9 && abs(modM.x) < 5.0));
  if (tick && bodyMask < 0.20) {
    col = mix(col, vec3(0.96, 0.98, 1.0), 0.42);
  }

  /* subtle outer surveillance scale ruler ticks along top/left edges */
  if ((px.y < 14.0 && mod(px.x, 30.0) < 1.2) || (px.x < 14.0 && mod(px.y, 30.0) < 1.2)) {
    col = mix(col, vec3(0.94, 0.96, 0.98), 0.32);
  }

  oOut = vec4(clamp(col, 0.0, 1.0), 1.0);
}
"""


def label(im, s):
    im = im.copy()
    cv2.putText(im, s, (12, im.shape[0] - 14), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 4, cv2.LINE_AA)
    cv2.putText(im, s, (12, im.shape[0] - 14), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 1, cv2.LINE_AA)
    return im


def encode(c, title_label="3D Vision Quadtree HUD"):
    out = c["dir"] / "out"
    out.mkdir(exist_ok=True)
    nm = c["name"]
    r = f"{c['fps_num']}/{c['fps_den']}"
    ff = find_ffmpeg()
    tags = ["-pix_fmt", "yuv420p", "-color_primaries", "bt709", "-color_trc", "bt709", "-colorspace", "bt709", "-movflags", "+faststart"]
    x264 = ["-c:v", "libx264", "-profile:v", "high", "-level", "4.1", "-crf", "14", "-preset", "slow"]
    run = lambda a: subprocess.run([ff, "-v", "error", "-y"] + a, check=True)
    run(["-framerate", r, "-i", str(work(c, "comp", "%04d.png"))] + x264 + tags + [str(out / f"{nm}_Composite.mp4")])
    lab = lambda t: f"drawtext=fontfile='C\\:/Windows/Fonts/arialbd.ttf':text='{t}':x=w-tw-28:y=h-th-24:fontsize=44:fontcolor=white:shadowcolor=black@0.6:shadowx=2:shadowy=2"
    run([
        "-framerate", r, "-i", str(work(c, "comp", "%04d.png")),
        "-framerate", r, "-i", str(work(c, "frames", "%04d.jpg")),
        "-filter_complex",
        f"[0]scale=1080:608,{lab(title_label)}[a];[1]scale=1080:608,{lab('Before')}[b];[a][b]vstack,pad=1080:1350:0:67:color=black",
    ] + x264 + tags + [str(out / f"{nm}_Compare_1080x1350.mp4")])
    print("encoded ->", out)


def proj3d(X, R, C, f, W, H):
    q = (np.asarray(X, float) - C) @ R
    if q[2] <= 0.05:
        return None
    return (float(f * q[0] / q[2] + W * 0.5), float(f * q[1] / q[2] + H * 0.5))


def draw_dashed_line(img, p0, p1, color, thickness=1, dash=9, gap=7):
    p0 = np.asarray(p0, float)
    p1 = np.asarray(p1, float)
    dist = float(np.linalg.norm(p1 - p0))
    if dist < 2.0:
        return
    u = (p1 - p0) / dist
    d = 0.0
    while d < dist:
        a = p0 + u * d
        b = p0 + u * min(d + dash, dist)
        cv2.line(img, (int(round(a[0])), int(round(a[1]))), (int(round(b[0])), int(round(b[1]))), color, thickness, cv2.LINE_AA)
        d += dash + gap


def draw_3d_hud(base_bgr, pl_bgr, i, N, fps, S, P, V, img_j, foot_hist, vel_hist, pip_state, cfg, fonts):
    """Draw razor-crisp 3D perspective floor reticle, 3D metric bounding cage, multi-node blob graph & turret PIP."""
    H, W = base_bgr.shape[:2]
    R, C, f = np.asarray(S["rot"][i], float), np.asarray(S["cam"][i], float), float(S["f"])
    acc_rgb = cfg.get("accent_rgb", [0.92, 1.0, 0.28])
    ACC_BGR = (int(acc_rgb[2] * 255), int(acc_rgb[1] * 255), int(acc_rgb[0] * 255))
    ACC_RGBA = (int(acc_rgb[0] * 255), int(acc_rgb[1] * 255), int(acc_rgb[2] * 255), 250)
    vel_trig = float(cfg.get("vel_trigger", 2.4))

    over = base_bgr.copy()

    # 1. 3D Perspective Floor Reticle & Fading 3D Foot Trajectory (z = 0 in plane coords)
    j_low = min((27, 28, 29, 30, 31, 32), key=lambda q: P[i, q, 2])
    fx, fy = float(P[i, j_low, 0]), float(P[i, j_low, 1])

    for rad, col, th in ((0.48, (248, 252, 255), 2), (0.92, (220, 228, 232), 1)):
        pts = []
        for ang in np.linspace(0, 2 * math.pi, 57):
            p2 = proj3d((fx + rad * math.cos(ang), fy + rad * math.sin(ang), 0.0), R, C, f, W, H)
            if p2 is not None:
                pts.append([int(round(p2[0])), int(round(p2[1]))])
        if len(pts) > 4:
            arr_p = np.array(pts, np.int32)
            cv2.polylines(over, [arr_p], False, (18, 22, 26), th + 2, cv2.LINE_AA)
            cv2.polylines(over, [arr_p], False, col, th, cv2.LINE_AA)

    for a3, b3 in (
        ((fx - 1.2, fy, 0.0), (fx + 1.2, fy, 0.0)),
        ((fx, fy - 1.2, 0.0), (fx, fy + 1.2, 0.0)),
    ):
        pa, pb = proj3d(a3, R, C, f, W, H), proj3d(b3, R, C, f, W, H)
        if pa and pb:
            draw_dashed_line(over, pa, pb, (20, 24, 28), 3, dash=9, gap=8)
            draw_dashed_line(over, pa, pb, (238, 244, 248), 1, dash=9, gap=8)

    # Fading 3D perspective foot trajectory on the asphalt/concrete floor
    if len(foot_hist) >= 2:
        pts_tr = [proj3d((x, y, 0.0), R, C, f, W, H) for (x, y) in foot_hist]
        for k in range(1, len(pts_tr)):
            if pts_tr[k - 1] and pts_tr[k]:
                al = k / len(pts_tr)
                p_a = (int(pts_tr[k - 1][0]), int(pts_tr[k - 1][1]))
                p_b = (int(pts_tr[k][0]), int(pts_tr[k][1]))
                if al > 0.35:
                    cv2.line(over, p_a, p_b, (18, 22, 24), 4 if al > 0.7 else 2, cv2.LINE_AA)
                c_tr = ACC_BGR if al > 0.78 else (int(235 * al + 20), int(245 * al + 10), int(248 * al + 5))
                cv2.line(over, p_a, p_b, c_tr, 2 if al > 0.65 else 1, cv2.LINE_AA)

    # 2. True 3D Metric Bounding Cage around Dancer (AABB in plane metres)
    body_pts = P[i, [0, 11, 12, 13, 14, 15, 16, 23, 24, 25, 26, 27, 28, 31, 32]]
    pad_xy = 0.24
    xmin, xmax = float(body_pts[:, 0].min()) - pad_xy, float(body_pts[:, 0].max()) + pad_xy
    ymin, ymax = float(body_pts[:, 1].min()) - pad_xy, float(body_pts[:, 1].max()) + pad_xy
    zmin, zmax = 0.0, float(body_pts[:, 2].max()) + 0.24

    corners3 = [
        (xmin, ymin, zmin), (xmax, ymin, zmin), (xmax, ymax, zmin), (xmin, ymax, zmin),
        (xmin, ymin, zmax), (xmax, ymin, zmax), (xmax, ymax, zmax), (xmin, ymax, zmax),
    ]
    c2 = [proj3d(p, R, C, f, W, H) for p in corners3]
    edges = [(0,1),(1,2),(2,3),(3,0), (4,5),(5,6),(6,7),(7,4), (0,4),(1,5),(2,6),(3,7)]
    for a, b in edges:
        if c2[a] and c2[b]:
            draw_dashed_line(over, c2[a], c2[b], (20, 24, 28), 2, dash=5, gap=9)
            draw_dashed_line(over, c2[a], c2[b], (228, 234, 240), 1, dash=5, gap=9)
            pa, pb = np.array(c2[a]), np.array(c2[b])
            v_ab = pb - pa
            for s0, s1 in ((0.0, 0.16), (0.84, 1.0)):
                q0, q1 = pa + v_ab * s0, pa + v_ab * s1
                cv2.line(over, (int(q0[0]), int(q0[1])), (int(q1[0]), int(q1[1])), (18, 20, 24), 4, cv2.LINE_AA)
                cv2.line(over, (int(q0[0]), int(q0[1])), (int(q1[0]), int(q1[1])), (250, 252, 255), 2, cv2.LINE_AA)

    out = cv2.addWeighted(over, 0.88, base_bgr, 0.12, 0.0)

    # 3. Multi-Node Kinetic Blob Tracking Graph
    NODES = [
        (0,  "HEAD"),
        (15, "L.WRST"),
        (16, "R.WRST"),
        (13, "L.ELBW"),
        (14, "R.ELBW"),
        (27, "L.ANKL"),
        (28, "R.ANKL"),
        (23, "PELVIS"),
    ]
    node_pos2, node_pos3, node_vel, node_names = [], [], [], []
    for idx, nm in NODES:
        p2 = proj3d(P[i, idx], R, C, f, W, H) or (float(img_j[i, idx, 0]), float(img_j[i, idx, 1]))
        node_pos2.append(p2)
        node_pos3.append(P[i, idx])
        node_vel.append(float(V[i, idx]))
        node_names.append(nm)

    # Dynamic k-NN connection web + collect crisp midpoint distance tags
    web = out.copy()
    mid_tags = []
    for a in range(len(NODES)):
        dists = sorted((float(np.linalg.norm(node_pos3[a] - node_pos3[b])), b) for b in range(len(NODES)) if a != b)
        for d3, b in dists[:2]:
            if a < b and d3 < 1.35:
                pa, pb = node_pos2[a], node_pos2[b]
                fast = max(node_vel[a], node_vel[b]) > vel_trig
                col = ACC_BGR if fast else (244, 248, 252)
                cv2.line(web, (int(pa[0]), int(pa[1])), (int(pb[0]), int(pb[1])), (18, 20, 24), 3, cv2.LINE_AA)
                cv2.line(web, (int(pa[0]), int(pa[1])), (int(pb[0]), int(pb[1])), col, 1, cv2.LINE_AA)
                if fast and len(mid_tags) < 2:
                    mid_tags.append(((pa[0] + pb[0]) * 0.5, (pa[1] + pb[1]) * 0.5, f"{d3:.2f}m"))
    out = cv2.addWeighted(web, 0.80, out, 0.20, 0.0)

    # Active velocity-scaled target brackets on each tracked node
    peak_idx = int(np.argmax(node_vel[1:7])) + 1
    for k, (p2, p3, vel, nm) in enumerate(zip(node_pos2, node_pos3, node_vel, node_names)):
        x, y = int(round(p2[0])), int(round(p2[1]))
        fast = vel > vel_trig
        rad = 26 if k == 0 else int(np.clip(12 + vel * 4.5, 12, 30))
        col = ACC_BGR if (fast or (k == peak_idx and vel > 1.85)) else (248, 250, 252)
        bl = max(6, rad // 2)
        for sx in (-1, 1):
            for sy in (-1, 1):
                cx, cy = x + sx * rad, y + sy * rad
                cv2.line(out, (cx, cy), (cx - sx * bl, cy), (14, 16, 18), 3, cv2.LINE_AA)
                cv2.line(out, (cx, cy), (cx, cy - sy * bl), (14, 16, 18), 3, cv2.LINE_AA)
                cv2.line(out, (cx, cy), (cx - sx * bl, cy), col, 2, cv2.LINE_AA)
                cv2.line(out, (cx, cy), (cx, cy - sy * bl), col, 2, cv2.LINE_AA)
        cv2.circle(out, (x, y), 3, col, -1, cv2.LINE_AA)

    # 4. Smooth Turret-Tracked Monochrome PIP Inset framed cleanly in the right sage shutter bay!
    # Hysteresis target lock: stay locked >= 24 frames unless another node spikes dramatically
    want_k = peak_idx if node_vel[peak_idx] > 2.2 else 0
    if pip_state["hold"] <= 0 or (node_vel[want_k] > node_vel[pip_state["k"]] + 1.4):
        pip_state["k"] = want_k
        pip_state["hold"] = 28
    else:
        pip_state["hold"] -= 1
    tgt_k = pip_state["k"]
    raw_tx, raw_ty = node_pos2[tgt_k]
    if pip_state["xy"] is None:
        pip_state["xy"] = [raw_tx, raw_ty]
    else:
        pip_state["xy"][0] = pip_state["xy"][0] * 0.78 + raw_tx * 0.22
        pip_state["xy"][1] = pip_state["xy"][1] * 0.78 + raw_ty * 0.22
    tx, ty = int(round(pip_state["xy"][0])), int(round(pip_state["xy"][1]))

    # Centered inside the clean right shutter bay (x ~ 1085..1545)
    pip_w, pip_h = 370, 232
    pip_x, pip_y = 1130, 148
    elbow_x = pip_x - 44
    cv2.line(out, (tx, ty), (elbow_x, pip_y + 32), (16, 18, 20), 3, cv2.LINE_AA)
    cv2.line(out, (elbow_x, pip_y + 32), (pip_x - 12, pip_y + 32), (16, 18, 20), 3, cv2.LINE_AA)
    cv2.line(out, (tx, ty), (elbow_x, pip_y + 32), (242, 246, 250), 1, cv2.LINE_AA)
    cv2.line(out, (elbow_x, pip_y + 32), (pip_x - 12, pip_y + 32), (242, 246, 250), 1, cv2.LINE_AA)
    cv2.circle(out, (tx, ty), 6, ACC_BGR if node_vel[tgt_k] > vel_trig else (255, 255, 255), 2, cv2.LINE_AA)

    crop_hw, crop_hh = 110, 69
    cx0 = int(np.clip(tx, crop_hw, W - crop_hw))
    cy0 = int(np.clip(ty, crop_hh, H - crop_hh))
    crop = pl_bgr[cy0 - crop_hh:cy0 + crop_hh, cx0 - crop_hw:cx0 + crop_hw]
    crop_g = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    crop_g = np.clip((crop_g.astype(np.float32) - 128.0) * 1.32 + 130.0, 0, 255).astype(np.uint8)
    crop_res = cv2.resize(cv2.cvtColor(crop_g, cv2.COLOR_GRAY2BGR), (pip_w, pip_h), interpolation=cv2.INTER_CUBIC)
    cv2.line(crop_res, (pip_w // 2, 0), (pip_w // 2, pip_h), (195, 202, 208), 1)
    cv2.line(crop_res, (0, pip_h // 2), (pip_w, pip_h // 2), (195, 202, 208), 1)
    cv2.rectangle(crop_res, (pip_w // 2 - 28, pip_h // 2 - 28), (pip_w // 2 + 28, pip_h // 2 + 28), ACC_BGR, 1)

    card_x0, card_y0 = pip_x - 14, pip_y - 36
    card_w, card_h = pip_w + 28, pip_h + 132
    roi = out[card_y0:card_y0 + card_h, card_x0:card_x0 + card_w]
    dark = np.full_like(roi, (14, 16, 18))
    out[card_y0:card_y0 + card_h, card_x0:card_x0 + card_w] = cv2.addWeighted(roi, 0.22, dark, 0.78, 0.0)
    out[pip_y:pip_y + pip_h, pip_x:pip_x + pip_w] = crop_res
    cv2.rectangle(out, (card_x0, card_y0), (card_x0 + card_w, card_y0 + card_h), (235, 240, 244), 1)
    cv2.rectangle(out, (pip_x, pip_y), (pip_x + pip_w, pip_y + pip_h), (165, 172, 178), 1)

    # Real-time 3D peak limb velocity sparkline inside bottom of PIP card
    sp_x0, sp_y0, sp_w, sp_h = pip_x + 2, pip_y + pip_h + 36, pip_w - 4, 34
    cv2.line(out, (sp_x0, sp_y0 + sp_h), (sp_x0 + sp_w, sp_y0 + sp_h), (90, 98, 104), 1)
    if len(vel_hist) >= 2:
        pts_sp = []
        for k, vv in enumerate(vel_hist):
            sx = int(sp_x0 + (k / max(len(vel_hist) - 1, 1)) * sp_w)
            sy = int(sp_y0 + sp_h - np.clip(vv / 5.5, 0.0, 1.0) * sp_h)
            pts_sp.append([sx, sy])
        cv2.polylines(out, [np.array(pts_sp, np.int32)], False, ACC_BGR, 2, cv2.LINE_AA)

    # 5. High-DPI Legible Monospace Telemetry Typography via PIL (crisp even on mobile Reels!)
    pil_im = Image.fromarray(cv2.cvtColor(out, cv2.COLOR_BGR2RGB)).convert("RGBA")
    draw = ImageDraw.Draw(pil_im)
    f_xs, f_sm, f_md = fonts

    # Clean node telemetry tags with 2D anti-overlap placement (Head + fastest limb + right wrist)
    show_nodes = [0, peak_idx]
    if 2 not in show_nodes:
        show_nodes.append(2)
    placed = []
    for k in show_nodes:
        p2, p3, vel, nm = node_pos2[k], node_pos3[k], node_vel[k], node_names[k]
        x, y = int(round(p2[0])), int(round(p2[1]))
        fast = vel > vel_trig
        txt = f"ID:01  Z:{p3[2]:.2f}m" if k == 0 else f"{nm} {vel:.1f}m/s"
        tw = draw.textlength(txt, font=f_xs)
        cand_offsets = [(32, -18), (-tw - 32, -18), (32, 16), (-tw - 32, 16), (32, -46)]
        if k == 1:
            cand_offsets = [(-tw - 32, -18), (32, -18), (-tw - 32, 16), (32, 16)]
        chosen = None
        for ox, oy in cand_offsets:
            bx0, by0, bx1, by1 = x + ox - 6, y + oy - 3, x + ox + tw + 6, y + oy + 19
            if bx0 < 20 or bx1 > W - 20 or by0 < 70 or by1 > H - 70:
                continue
            if not any(not (bx1 + 6 < rx0 or bx0 - 6 > rx1 or by1 + 4 < ry0 or by0 - 4 > ry1) for (rx0, ry0, rx1, ry1) in placed):
                chosen = (x + ox, y + oy, bx0, by0, bx1, by1)
                break
        if chosen is not None:
            tx_p, ty_p, bx0, by0, bx1, by1 = chosen
            placed.append((bx0, by0, bx1, by1))
            draw.rectangle([bx0, by0, bx1, by1], fill=(12, 14, 16, 195))
            draw.text((tx_p, ty_p), txt, font=f_xs, fill=ACC_RGBA if fast else (246, 249, 252, 245))

    # 3D Cage metric dimension tag
    top_c = c2[5] or c2[6]
    if top_c:
        cam_dist = float(np.linalg.norm(P[i, 0] - C))
        dim_txt = f"3D.BOX  H:{zmax:.2f}m  D:{cam_dist:.2f}m"
        bx, by = int(top_c[0]) + 10, int(top_c[1]) - 26
        tw = draw.textlength(dim_txt, font=f_xs)
        draw.rectangle([bx - 6, by - 3, bx + tw + 6, by + 19], fill=(12, 14, 16, 200))
        draw.text((bx, by), dim_txt, font=f_xs, fill=(242, 247, 250, 245))

    # Midpoint 3D distance tags on peak-velocity connections
    for mx, my, mtxt in mid_tags:
        tw = draw.textlength(mtxt, font=f_xs)
        draw.rectangle([mx - tw / 2 - 4, my - 10, mx + tw / 2 + 4, my + 10], fill=(12, 14, 16, 185))
        draw.text((mx - tw / 2, my - 8), mtxt, font=f_xs, fill=ACC_RGBA)

    # PIP card header & footer telemetry
    hdr = f"LOCK // {node_names[tgt_k]}   V:{node_vel[tgt_k]:.2f} m/s"
    draw.text((pip_x, pip_y - 27), hdr, font=f_sm, fill=ACC_RGBA if node_vel[tgt_k] > vel_trig else (245, 248, 252, 248))
    tp3 = node_pos3[tgt_k]
    draw.text((pip_x, pip_y + pip_h + 10), f"3D.POS  ({tp3[0]:+.2f}, {tp3[1]:+.2f}, {tp3[2]:.2f}) m", font=f_xs, fill=(224, 230, 236, 240))
    draw.text((pip_x, pip_y + pip_h + 75), f"PEAK {max(node_vel):.2f} m/s   //   FRM {i:03d}/{N:03d}", font=f_xs, fill=(195, 202, 208, 230))

    # Minimal top-left & bottom-left editorial system bars
    top_txt = f"VISION.SYS // ADAPTIVE.QUADTREE + 3D.POSE   [{i/fps:05.2f}s]"
    tw_top = draw.textlength(top_txt, font=f_sm)
    draw.rectangle([42, 34, 42 + tw_top + 20, 66], fill=(12, 14, 16, 195))
    draw.text((52, 41), top_txt, font=f_sm, fill=(246, 249, 252, 245))

    bot_txt = f"CAM.H {C[2]:.2f}m  |  FLOOR.Z 0.00m  |  30FPS"
    tw_bot = draw.textlength(bot_txt, font=f_xs)
    draw.rectangle([42, H - 64, 42 + tw_bot + 18, H - 36], fill=(12, 14, 16, 185))
    draw.text((51, H - 58), bot_txt, font=f_xs, fill=(220, 226, 232, 235))

    return cv2.cvtColor(np.array(pil_im.convert("RGB")), cv2.COLOR_RGB2BGR)


def main():
    c = load(sys.argv[1] if len(sys.argv) > 1 and not sys.argv[1].startswith("--") else here / "piece_vision")
    S = scene(c)
    W, H, N, fps = c["W"], c["H"], c["N"], c["fps"]
    vc = c.get("vision", {})
    stills = [int(x) for x in sys.argv[sys.argv.index("--still") + 1].split(",")] if "--still" in sys.argv else None

    L = np.load(work(c, "limbs.npz"))
    P = L["P"].astype(np.float64)
    img_j = L["img"].astype(np.float64)

    from scipy.signal import savgol_filter
    dP = np.zeros_like(P)
    dP[1:] = (P[1:] - P[:-1]) * fps
    dP[0] = dP[1]
    V = np.linalg.norm(dP, axis=2)
    if N >= 9:
        V = np.clip(savgol_filter(V, 7, 2, axis=0), 0.0, 20.0)

    ctx = moderngl.create_standalone_context(require=330)
    prog = ctx.program(vertex_shader=VS, fragment_shader=FS)
    quad = ctx.buffer(np.array([-1, -1, 3, -1, -1, 3], "f4").tobytes())
    va = ctx.vertex_array(prog, [(quad, "2f", "p")])
    plate_t = ctx.texture((W, H), 3)
    plate_t.filter = (moderngl.LINEAR, moderngl.LINEAR)
    plate_t.repeat_x = plate_t.repeat_y = False
    energy_t = ctx.texture((W, H), 3, dtype="f4")
    energy_t.filter = (moderngl.LINEAR, moderngl.LINEAR)
    energy_t.repeat_x = energy_t.repeat_y = False
    out_t = ctx.texture((W, H), 4, dtype="f4")
    fbo = ctx.framebuffer([out_t])

    prog["uRes"].value = (float(W), float(H))
    prog["uBaseCell"].value = float(vc.get("base_cell", 120.0))
    prog["uSplitThresh"].value = float(vc.get("split_thresh", 0.13))
    prog["uAccent"].value = tuple(map(float, vc.get("accent_rgb", [0.92, 1.0, 0.28])))

    fonts = (load_mono_font(16, bold=True), load_mono_font(18, bold=True), load_mono_font(22, bold=True))
    decay = float(vc.get("motion_decay", 0.80))
    vel_trig = float(vc.get("vel_trigger", 2.4))

    prev_gray_s = None
    energy_acc = np.zeros((H // 2, W // 2), np.float32)
    peak_acc = np.zeros((H // 2, W // 2), np.float32)
    foot_hist = []
    vel_hist = []
    pip_state = {"k": 0, "hold": 0, "xy": None}
    dil_k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (95, 95))

    def step_frame(i, render_diag=False):
        nonlocal prev_gray_s, energy_acc, peak_acc
        pl = frame(c, i)
        m8 = cv2.imread(str(work(c, "mask", f"{i:04d}.png")), cv2.IMREAD_GRAYSCALE)
        mf = m8.astype(np.float32) / 255.0

        # Half-res dense optical flow gated cleanly by real intensity diff + broad body-wake ROI
        # (eliminates false-positive flow chatter on repeating static metal shutter slats!)
        gray_s = cv2.resize(cv2.cvtColor(pl, cv2.COLOR_BGR2GRAY), (W // 2, H // 2), interpolation=cv2.INTER_AREA)
        ms = cv2.resize(mf, (W // 2, H // 2), interpolation=cv2.INTER_AREA)
        wake_roi = cv2.GaussianBlur(cv2.dilate((ms > 0.15).astype(np.float32), dil_k), (0, 0), 28.0)
        if prev_gray_s is None:
            flow_mag = np.zeros_like(energy_acc)
        else:
            diff_s = np.abs(gray_s.astype(np.float32) - prev_gray_s.astype(np.float32)) / 255.0
            diff_gate = np.clip((cv2.GaussianBlur(diff_s, (0, 0), 3.0) - 0.018) / 0.06, 0.0, 1.0)
            flow = cv2.calcOpticalFlowFarneback(prev_gray_s, gray_s, None, 0.5, 3, 15, 3, 5, 1.1, 0)
            flow_mag = np.hypot(flow[..., 0], flow[..., 1])
            flow_mag = np.clip((flow_mag - 0.75) / 7.2, 0.0, 1.0) * wake_roi * np.clip(diff_gate + ms, 0.0, 1.0)
        prev_gray_s = gray_s

        # Silhouette edge energy from MediaPipe mask
        gx = cv2.Sobel(ms, cv2.CV_32F, 1, 0, ksize=3)
        gy = cv2.Sobel(ms, cv2.CV_32F, 0, 1, ksize=3)
        edge_e = np.clip(np.hypot(gx, gy) * 1.45, 0.0, 1.0)

        # High-velocity 3D joint hotspot blobs
        joint_e = np.zeros_like(energy_acc)
        joint_pk = np.zeros_like(peak_acc)
        for j in (0, 13, 14, 15, 16, 25, 26, 27, 28):
            jx, jy = int(img_j[i, j, 0] * 0.5), int(img_j[i, j, 1] * 0.5)
            if 0 <= jx < W // 2 and 0 <= jy < H // 2:
                v_n = float(np.clip(V[i, j] / 3.4, 0.18, 1.0))
                cv2.circle(joint_e, (jx, jy), 24, v_n, -1)
                if V[i, j] > vel_trig * 0.75:
                    cv2.circle(joint_pk, (jx, jy), 20, float(np.clip(V[i, j] / 4.0, 0.0, 1.0)), -1)
        joint_e = cv2.GaussianBlur(joint_e, (0, 0), 11.0)
        joint_pk = cv2.GaussianBlur(joint_pk, (0, 0), 8.0)

        inst_e = np.clip(flow_mag * 1.25 + edge_e * 0.62 + joint_e * 0.56, 0.0, 1.0)
        energy_acc = np.maximum(inst_e, energy_acc * decay)
        peak_acc = np.maximum(np.maximum(joint_pk, np.clip((flow_mag - 0.42) * 1.9, 0.0, 1.0)), peak_acc * 0.76)

        en_full = cv2.resize(energy_acc, (W, H), interpolation=cv2.INTER_LINEAR)
        pk_full = cv2.resize(peak_acc, (W, H), interpolation=cv2.INTER_LINEAR)
        pack = np.dstack([en_full, mf, pk_full]).astype(np.float32)

        j_low = min((27, 28, 29, 30, 31, 32), key=lambda q: P[i, q, 2])
        foot_hist.append((float(P[i, j_low, 0]), float(P[i, j_low, 1])))
        if len(foot_hist) > 54:
            foot_hist.pop(0)
        vel_hist.append(float(V[i, [15, 16, 27, 28]].max()))
        if len(vel_hist) > 60:
            vel_hist.pop(0)

        plate_t.write(pl[:, :, ::-1].tobytes())
        energy_t.write(pack.tobytes())
        plate_t.use(0)
        energy_t.use(1)
        prog["uPlate"].value = 0
        prog["uEnergy"].value = 1
        prog["iTime"].value = i / fps
        fx_px, fy_px = float(img_j[i, 0, 0]), float(img_j[i, 0, 1])
        sh_w = float(np.hypot(img_j[i, 11, 0] - img_j[i, 12, 0], img_j[i, 11, 1] - img_j[i, 12, 1]))
        prog["uFacePx"].value = (fx_px, fy_px)
        prog["uFaceRad"].value = max(62.0, sh_w * 0.68)

        prog["uMode"].value = 0
        fbo.use()
        va.render(moderngl.TRIANGLES)
        ctx.finish()
        arr = np.frombuffer(out_t.read(), "f4").reshape(H, W, 4)[::-1]
        qt_bgr = (np.clip(arr[..., :3], 0, 1) * 255 + 0.5).astype(np.uint8)[:, :, ::-1]
        comp = draw_3d_hud(qt_bgr, pl, i, N, fps, S, P, V, img_j, foot_hist, vel_hist, pip_state, vc, fonts)

        diag = None
        if render_diag:
            prog["uMode"].value = 1
            fbo.use()
            va.render(moderngl.TRIANGLES)
            ctx.finish()
            darr = np.frombuffer(out_t.read(), "f4").reshape(H, W, 4)[::-1]
            diag = (np.clip(darr[..., :3], 0, 1) * 255 + 0.5).astype(np.uint8)[:, :, ::-1]
        return pl, comp, diag

    if stills:
        rows = []
        todo = set(stills)
        warmup = max(0, min(stills) - 24)
        hero = None
        for i in range(warmup, max(stills) + 1):
            pl, comp, diag = step_frame(i, render_diag=(i in todo))
            if i in todo:
                if i == stills[len(stills) // 2]:
                    hero = comp
                t = lambda im, s: label(cv2.resize(im, (640, 360)), s)
                rows.append(np.hstack([t(pl, f"plate {i}"), t(diag, "quadtree energy field"), t(comp, "3D vision HUD")]))
        out_p = work(c, "still_sheet.jpg")
        cv2.imwrite(str(out_p), np.vstack(rows), [cv2.IMWRITE_JPEG_QUALITY, 92])
        if hero is not None:
            cv2.imwrite(str(work(c, "still_hero.jpg")), hero, [cv2.IMWRITE_JPEG_QUALITY, 94])
        print("stills ->", out_p)
        return

    comp_dir = work(c, "comp")
    shutil.rmtree(comp_dir, ignore_errors=True)
    comp_dir.mkdir(parents=True)
    ts = time.time()
    for i in range(N):
        _, comp, _ = step_frame(i, render_diag=False)
        cv2.imwrite(str(comp_dir / f"{i:04d}.png"), comp, [cv2.IMWRITE_PNG_COMPRESSION, 1])
        if i % 45 == 0 or i == N - 1:
            print(f"frame {i}/{N} ({time.time() - ts:.1f}s)", flush=True)
    encode(c)


if __name__ == "__main__":
    main()
