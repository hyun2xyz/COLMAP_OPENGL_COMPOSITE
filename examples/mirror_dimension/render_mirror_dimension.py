"""Effect #4: Doctor Strange Mirror Dimension Monumental 3D Architecture Folding (Native OpenGL 3.3).

    python render_mirror_dimension.py --still 15,65,125,185,225   quick QA -> work/still_mirror_sheet.jpg
    python render_mirror_dimension.py --render                    full render -> out/Mirror_Dimension_Folding_*.mp4

Real 6-DOF COLMAP modern architectural hall walkthrough (Pexels 7578547, 240f @ 30fps = 8.00s):
- Monumental synchronized Doctor Strange / Inception origami folding along an OPEN central hallway:
  * Real Left Wall, Right Wall, Ceiling, and Side Oak Floor bays fracture sequentially along a sweeping
    crystal wave into ~72 LARGE beveled mirror-glass architectural slabs (razor-thin 4.5cm glass/wall profile).
  * Every folding slab projects its REST 3D world position back through the live COLMAP camera (C[i], R[i], f)
    so the ACTUAL sunlit oak floor, white walls, artwork, recessed ceiling lights, framed mirror, and marble
    island physically hinge 25-52 deg overhead like a folding Inception cathedral — ZERO muddy dark debris!
  * 24 sleek translucent beveled glass-mirror clockwork fins arranged in 3 concentric octagonal Sanctum rings
    frame the open central walkway with crisp real-room mirror reflections & authentic white-silver Cauchy
    chromatic glass bevel glints (ZERO cartoon neon stripes!).
  * Synchronized Escher kaleidoscope mirror backdrop fills deep wall seams, then all panels snap smoothly
    back along C2 origami hinges into 100% untouched reality by f=218.
- ZERO text/HUD overlays.
"""
import argparse, json, math, shutil, subprocess, sys, time
from pathlib import Path
import numpy as np, cv2, moderngl

here = Path(__file__).resolve().parent
ROOT = here.parents[1]
sys.path.insert(0, str(ROOT / "examples/shader_live/pipeline"))
from slv import load, work, frame, scene  # noqa: E402

SRC_PIECE = ROOT / "examples/lidar_scan/piece_lidar"


def find_ffmpeg():
    tl = ROOT / "tools.local.json"
    if tl.exists():
        d = json.loads(tl.read_text(encoding="utf-8"))
        p = d.get("ffmpeg") or (d.get("tools") or {}).get("ffmpeg")
        if p and Path(p).exists():
            return str(p)
    return shutil.which("ffmpeg") or "ffmpeg"


def smootherstep(e0, e1, x):
    t = max(0.0, min(1.0, (x - e0) / max(e1 - e0, 1e-6)))
    return t * t * t * (t * (t * 6.0 - 15.0) + 10.0)


def axis_angle_mat3(axis, ang):
    ax = np.asarray(axis, dtype=np.float64)
    ax /= np.linalg.norm(ax) + 1e-9
    c, s = math.cos(ang), math.sin(ang)
    x, y, z = ax
    K = np.array([[0, -z, y], [z, 0, -x], [-y, x, 0]], dtype=np.float64)
    return c * np.eye(3) + s * K + (1.0 - c) * np.outer(ax, ax)


def build_beveled_slab_mesh(pts2d, thickness=0.048, bevel=0.015):
    """Build a sleek, razor-crisp architectural glass-mirror panel (pos3, nrm3, rest_local3, edge_flag1)."""
    pts = np.asarray(pts2d, dtype=np.float64)
    n = len(pts)
    ctr = pts.mean(axis=0)
    inner = np.array([ctr + (p - ctr) * max(0.82, 1.0 - bevel / max(np.linalg.norm(p - ctr), 0.08)) for p in pts])

    verts, idxs = [], []

    def add_v(p, nrm, rest_p, ef):
        nn = np.asarray(nrm, dtype=np.float64)
        nn /= np.linalg.norm(nn) + 1e-9
        verts.append([p[0], p[1], p[2], nn[0], nn[1], nn[2], rest_p[0], rest_p[1], rest_p[2], ef])
        return len(verts) - 1

    def add_tri(a, b, c, nrm, ef):
        idxs.extend([add_v(a[0], nrm, a[1], ef), add_v(b[0], nrm, b[1], ef), add_v(c[0], nrm, c[1], ef)])

    def add_quad(a, b, c, d, nrm, ef):
        add_tri(a, b, c, nrm, ef); add_tri(a, c, d, nrm, ef)

    # 1. Front architectural mirror face (edge_flag = 0.0)
    c_pt = (np.array([ctr[0], ctr[1], 0.0]), np.array([ctr[0], ctr[1], 0.0]))
    for k in range(n):
        k2 = (k + 1) % n
        add_tri(c_pt,
                (np.array([inner[k, 0], inner[k, 1], 0.0]), np.array([inner[k, 0], inner[k, 1], 0.0])),
                (np.array([inner[k2, 0], inner[k2, 1], 0.0]), np.array([inner[k2, 0], inner[k2, 1], 0.0])),
                (0.0, 0.0, 1.0), 0.0)

    # 2. Razor-thin 45-deg crystal glass bevel (edge_flag = 0.55) + sleek mirror edge wall (edge_flag = 1.0)
    for k in range(n):
        k2 = (k + 1) % n
        ev = pts[k2] - pts[k]
        out_n = np.array([ev[1], -ev[0]], dtype=np.float64)
        if np.dot(out_n, 0.5 * (pts[k] + pts[k2]) - ctr) < 0:
            out_n = -out_n
        out_n /= np.linalg.norm(out_n) + 1e-9
        nb = (out_n[0] * 0.70, out_n[1] * 0.70, 0.71)
        add_quad(
            (np.array([inner[k, 0], inner[k, 1], 0.0]), np.array([inner[k, 0], inner[k, 1], 0.0])),
            (np.array([pts[k, 0], pts[k, 1], -bevel]), np.array([pts[k, 0], pts[k, 1], 0.0])),
            (np.array([pts[k2, 0], pts[k2, 1], -bevel]), np.array([pts[k2, 0], pts[k2, 1], 0.0])),
            (np.array([inner[k2, 0], inner[k2, 1], 0.0]), np.array([inner[k2, 0], inner[k2, 1], 0.0])),
            nb, 0.55,
        )
        add_quad(
            (np.array([pts[k, 0], pts[k, 1], -bevel]), np.array([pts[k, 0], pts[k, 1], 0.0])),
            (np.array([pts[k, 0], pts[k, 1], -thickness]), np.array([pts[k, 0], pts[k, 1], 0.0])),
            (np.array([pts[k2, 0], pts[k2, 1], -thickness]), np.array([pts[k2, 0], pts[k2, 1], 0.0])),
            (np.array([pts[k2, 0], pts[k2, 1], -bevel]), np.array([pts[k2, 0], pts[k2, 1], 0.0])),
            (out_n[0], out_n[1], 0.0), 1.0,
        )

    # 3. Back mirror face (edge_flag = 0.15)
    cb_pt = (np.array([ctr[0], ctr[1], -thickness]), np.array([ctr[0], ctr[1], 0.0]))
    for k in range(n):
        k2 = (k + 1) % n
        add_tri(cb_pt,
                (np.array([pts[k2, 0], pts[k2, 1], -thickness]), np.array([pts[k2, 0], pts[k2, 1], 0.0])),
                (np.array([pts[k, 0], pts[k, 1], -thickness]), np.array([pts[k, 0], pts[k, 1], 0.0])),
                (0.0, 0.0, -1.0), 0.15)

    return np.array(verts, dtype=np.float32), np.array(idxs, dtype=np.uint32)


def generate_monumental_mirror_architecture():
    """Build a sleek, monumental, SYMMETRICAL Doctor Strange / Inception origami corridor:
    - Open central walkway (r > 1.5m!) so the real bright architectural hall ALWAYS commands center frame!
    - Large clean synchronized architectural slabs along Left Wall, Right Wall, Ceiling, and Side Floor Strips
      hinging gracefully inward & upward along clean architectural bay axes (like Inception folding city!).
    - 24 large crystal-mirror clockwork fins in 3 concentric octagonal Sanctum rings framing the deep hall!
    """
    slabs = []

    def add_arch_bay(name, origin, u_ax, v_ax, n_ax, u_range, v_range, nu, nv, hinge_world, fold_deg, slide_vec, diag_split=False):
        us = np.linspace(u_range[0], u_range[1], nu + 1)
        vs = np.linspace(v_range[0], v_range[1], nv + 1)
        R_rest = np.column_stack([u_ax, v_ax, n_ax]).astype(np.float64)
        for i in range(nu):
            for j in range(nv):
                q00 = np.array([us[i], vs[j]])
                q10 = np.array([us[i + 1], vs[j]])
                q11 = np.array([us[i + 1], vs[j + 1]])
                q01 = np.array([us[i], vs[j + 1]])
                polys = [[q00, q10, q11], [q00, q11, q01]] if diag_split else [[q00, q10, q11, q01]]
                for pk, poly in enumerate(polys):
                    arr = np.array(poly, dtype=np.float64)
                    # Clean architectural hinge along the base/outer edge (so wall panels fold smoothly like giant doors!)
                    piv_uv = 0.5 * (arr[0] + arr[1])
                    local_pts = arr - piv_uv[None, :]
                    P_rest = origin + u_ax * piv_uv[0] + v_ax * piv_uv[1]
                    vdata, idata = build_beveled_slab_mesh(local_pts * 0.988, thickness=0.045, bevel=0.014)
                    # Alternate slight architectural rhythm between bays for Escher staircase depth!
                    alt = 1.0 if ((i + j + pk) % 2 == 0) else 0.72
                    slabs.append({
                        "kind": name, "vdata": vdata, "idata": idata,
                        "P_rest": P_rest, "R_rest": R_rest,
                        "hinge": hinge_world / (np.linalg.norm(hinge_world) + 1e-9),
                        "fold_ang": math.radians(fold_deg * alt),
                        "slide_vec": slide_vec * alt,
                        "y_coord": float(P_rest[1]),
                        "phase": 0.45 * i + 0.9 * j + 0.5 * pk,
                        "is_ring": False,
                    })

    # 1. Monumental Left Wall (x = -2.65m): 5 large clean bays hinging INWARD & UPWARD (+34 deg Inception fold!)
    add_arch_bay("left", np.array([-2.65, 0.0, 0.0]), np.array([0., 1., 0.]), np.array([0., 0., 1.]),
                 np.array([1., 0., 0.]), (-4.2, 5.2), (0.0, 3.35), 5, 2,
                 np.array([0., 1., 0.]), -34.0, np.array([0.42, 0.0, 0.14]), diag_split=False)

    # 2. Monumental Right Wall (x = +2.65m): 5 large clean bays hinging SYMMETRICALLY INWARD & UPWARD!
    add_arch_bay("right", np.array([2.65, 0.0, 0.0]), np.array([0., -1., 0.]), np.array([0., 0., 1.]),
                 np.array([-1., 0., 0.]), (-5.2, 4.2), (0.0, 3.35), 5, 2,
                 np.array([0., 1., 0.]), +34.0, np.array([-0.42, 0.0, 0.14]), diag_split=False)

    # 3. Cathedral Ceiling Vault (z = 3.38m): left & right geometric chevron ceiling panels folding down into diamond arches!
    add_arch_bay("ceilL", np.array([0.0, 0.0, 3.38]), np.array([1., 0., 0.]), np.array([0., 1., 0.]),
                 np.array([0., 0., -1.]), (-2.65, -0.05), (-4.2, 5.2), 1, 5,
                 np.array([0., 1., 0.]), +30.0, np.array([0.16, 0.0, -0.26]), diag_split=True)
    add_arch_bay("ceilR", np.array([0.0, 0.0, 3.38]), np.array([1., 0., 0.]), np.array([0., 1., 0.]),
                 np.array([0., 0., -1.]), (0.05, 2.65), (-4.2, 5.2), 1, 5,
                 np.array([0., 1., 0.]), -30.0, np.array([-0.16, 0.0, -0.26]), diag_split=True)

    # 4. Sleek Stepped Side Oak Floor Terraces (leaving center 1.7m oak hall gleaming & open!)
    add_arch_bay("floorL", np.array([0.0, 0.0, 0.0]), np.array([1., 0., 0.]), np.array([0., 1., 0.]),
                 np.array([0., 0., 1.]), (-2.65, -0.88), (-4.0, 4.8), 1, 5,
                 np.array([0., 1., 0.]), -28.0, np.array([0.12, 0.0, 0.22]), diag_split=False)
    add_arch_bay("floorR", np.array([0.0, 0.0, 0.0]), np.array([1., 0., 0.]), np.array([0., 1., 0.]),
                 np.array([0., 0., 1.]), (0.88, 2.65), (-4.0, 4.8), 1, 5,
                 np.array([0., 1., 0.]), +28.0, np.array([-0.12, 0.0, 0.22]), diag_split=False)

    # 5. 3 Concentric Octagonal Doctor Strange Clockwork Glass-Mirror Rings framing the open archway!
    #    (Guaranteed OPEN inner aperture r >= 1.48m -> frames the hallway like an iconic Sanctum kaleidoscopic portal!)
    ring_specs = [
        (-1.2, 1.88, +1.0, 0.42),
        ( 1.8, 1.72, -1.0, 0.46),
        ( 4.6, 1.62, +1.0, 0.50),
    ]
    for ri, (ry, r_ring, spin_dir, w_fin) in enumerate(ring_specs):
        for k in range(8):
            ang0 = k * (math.pi / 4.0) + ri * (math.pi / 8.0)
            cx = r_ring * math.cos(ang0)
            cz = 1.68 + r_ring * 0.72 * math.sin(ang0)
            P_rest = np.array([cx, ry, cz], dtype=np.float64)
            # Sleek architectural glass chevron blade
            pts = np.array([
                [-w_fin * 1.15, -w_fin * 0.48],
                [ w_fin * 1.15, -w_fin * 0.34],
                [ w_fin * 0.90,  w_fin * 0.48],
                [-w_fin * 0.90,  w_fin * 0.34],
            ], dtype=np.float64)
            vdata, idata = build_beveled_slab_mesh(pts, thickness=0.032, bevel=0.012)
            # Orient Sanctum crystal mirror panes 45 deg toward oncoming camera (-Y) so they gleam bright & refractive (never dark edge-on!)
            inward = np.array([-math.cos(ang0) * 0.65, -0.68, -math.sin(ang0) * 0.34], dtype=np.float64)
            inward /= np.linalg.norm(inward) + 1e-9
            u_a = np.array([-math.sin(ang0), 0.0, math.cos(ang0)], dtype=np.float64)
            v_a = np.cross(inward, u_a); v_a /= np.linalg.norm(v_a) + 1e-9
            R_rest = np.column_stack([u_a, v_a, inward])
            slabs.append({
                "kind": "ring", "vdata": vdata, "idata": idata,
                "P_rest": P_rest, "R_rest": R_rest,
                "hinge": np.array([0.0, 1.0, 0.0]),
                "fold_ang": math.radians(38.0 * spin_dir),
                "slide_vec": np.zeros(3),
                "y_coord": float(ry),
                "ring_r": r_ring, "ring_a0": ang0, "spin_dir": spin_dir,
                "phase": ri * 1.1 + k * 0.4,
                "is_ring": True,
            })

    return slabs


VS_FSQ = "#version 330\nin vec2 p; void main(){ gl_Position = vec4(p, 0.0, 1.0); }"

# Background Pass: Live Hallway Plate + Synchronized Subtle Glass-Kaleidoscope Depth Fold!
BG_FS = """#version 330
uniform sampler2D uPlate;
uniform vec2 uRes;
uniform float uKaleid, uTime;
out vec4 oCol;

void main() {
  vec2 px = vec2(gl_FragCoord.x, uRes.y - gl_FragCoord.y);
  vec2 uv = clamp(px / uRes, vec2(0.002), vec2(0.998));
  vec3 raw = texture(uPlate, uv).rgb;
  if (uKaleid < 0.005) { oCol = vec4(raw, 1.0); return; }

  /* Subtle Escher mirror-plane multiplication around outer frame & deep horizon (preserves bright natural light!) */
  vec2 ctr = vec2(0.50, 0.48);
  vec2 d = (uv - ctr) * vec2(uRes.x / uRes.y, 1.0);
  float r = length(d);
  float ang = atan(d.y, d.x);

  float seg = 3.14159265 / 4.0;
  float rot = 0.16 * sin(uTime * 0.65) * uKaleid;
  float aFold = abs(mod(ang + rot, seg * 2.0) - seg);
  vec2 uvFold = ctr + vec2(cos(aFold), sin(aFold)) * r * vec2(uRes.y / uRes.x, 1.0);
  uvFold = clamp(uvFold, vec2(0.004), vec2(0.996));

  float seam = exp(-pow(min(aFold, seg - aFold) / 0.016, 2.0));
  vec2 ca = normalize(d + vec2(1e-4)) * (0.0028 * uKaleid * (0.5 + seam));
  vec3 kCol = vec3(
    texture(uPlate, clamp(uvFold + ca, vec2(0.004), vec2(0.996))).r,
    texture(uPlate, uvFold).g,
    texture(uPlate, clamp(uvFold - ca, vec2(0.004), vec2(0.996))).b
  );

  float mask = smoothstep(0.18, 0.62, r) * uKaleid * 0.45;
  vec3 col = mix(raw, kCol, mask);
  col += vec3(0.95, 0.97, 1.00) * seam * mask * 0.18;
  oCol = vec4(clamp(col, 0.0, 1.0), 1.0);
}
"""

SLAB_VS = """#version 330
in vec3 inPos;
in vec3 inNrm;
in vec3 inRestWorld;
in float inEdgeFlag;
in float inFoldAmt;

uniform mat4 uProj, uView;

out vec3 vWorldPos;
out vec3 vRestWorld;
out vec3 vNrm;
out float vEdge;
out float vFold;
out float vDepth;

void main() {
  vWorldPos  = inPos;
  vRestWorld = inRestWorld;
  vNrm       = normalize(inNrm);
  vEdge      = inEdgeFlag;
  vFold      = inFoldAmt;
  vec4 vp = uView * vec4(inPos, 1.0);
  vDepth = -vp.z;
  gl_Position = uProj * vp;
}
"""

SLAB_FS = """#version 330
in vec3 vWorldPos;
in vec3 vRestWorld;
in vec3 vNrm;
in float vEdge;
in float vFold;
in float vDepth;

uniform sampler2D uPlate;
uniform vec2 uRes;
uniform float uF, uTime;
uniform vec3 uRo;
uniform mat3 uCam;

layout(location=0) out vec4 oCol;
layout(location=1) out vec4 oMeta;

vec3 sampleWorldProj(vec3 wp, vec2 pxOff) {
  vec3 q = transpose(uCam) * (wp - uRo);
  vec2 px = uF * q.xy / max(q.z, 0.25) + 0.5 * uRes + pxOff;
  vec2 uv = clamp(px / uRes, vec2(0.003), vec2(0.997));
  return texture(uPlate, uv).rgb;
}

vec3 sampleEnvRay(vec3 dirW, vec2 pxOff) {
  vec3 q = transpose(uCam) * dirW;
  if (q.z < 0.18) { q.z = abs(q.z) + 0.42; q.x = -q.x; }
  vec2 px = uF * q.xy / max(q.z, 0.22) + 0.5 * uRes + pxOff;
  vec2 uv = clamp(px / uRes, vec2(0.003), vec2(0.997));
  return texture(uPlate, uv).rgb;
}

void main() {
  if (vFold < 0.004 || vDepth < 0.35) discard;

  vec3 N = normalize(vNrm);
  vec3 V = normalize(uRo - vWorldPos);
  if (dot(N, V) < 0.0) N = -N;
  float NdotV = max(dot(N, V), 0.0);
  vec3 R = reflect(-V, N);

  /* 1. Bright, true-to-plate live camera-projected architecture from slab's REST world pose! */
  vec3 archSrgb = sampleWorldProj(vRestWorld, vec2(0.0));

  /* 2. Authentic optical crystal-glass mirror reflection + delicate natural Cauchy R/G/B dispersion! */
  float caPx = (vEdge > 0.35 && vEdge < 0.75) ? 7.5 : (2.2 * vFold);
  vec3 reflSrgb = vec3(
    sampleEnvRay(R, vec2( caPx, -caPx * 0.3)).r,
    sampleEnvRay(R, vec2( 0.0,   0.0)).g,
    sampleEnvRay(R, vec2(-caPx,  caPx * 0.3)).b
  );

  /* Bright warm sunlit interior architectural shading (keeps white walls & oak bright, NEVER dark muddy grey!) */
  vec3 L1 = normalize(vec3(-0.30, -0.40, 0.86));
  vec3 L2 = normalize(vec3( 0.45,  0.35, 0.82));
  float diff = 0.84 + 0.18 * max(dot(N, L1), 0.0) + 0.10 * max(dot(N, L2), 0.0);
  vec3 H1 = normalize(L1 + V);
  vec3 H2 = normalize(L2 + V);
  float spec = pow(max(dot(N, H1), 0.0), 110.0) * 1.15 + pow(max(dot(N, H2), 0.0), 75.0) * 0.65;

  float fresnel = mix(0.08, 0.78, pow(1.0 - NdotV, 3.0));
  float glow = 0.0;
  vec3 col;

  if (vEdge < 0.30) {
    /* Front architectural mirror surface: real camera-projected room + crisp crystal-mirror Fresnel sheen! */
    float mirrorMix = (vEdge > 0.10) ? 0.72 : (fresnel * 0.56 * vFold);
    col = mix(archSrgb * mix(1.0, diff, vFold * 0.60), reflSrgb * 1.08, mirrorMix);
    col += vec3(1.00, 0.99, 0.97) * spec * (0.25 + 0.75 * vFold);
  } else if (vEdge < 0.75) {
    /* Delicate cut-crystal glass bevel gleam (realistic glass refraction edge, not heavy white wireframe!) */
    col = mix(reflSrgb * 1.18, vec3(0.95, 0.98, 1.00), 0.28) + vec3(1.00, 0.99, 0.96) * (spec * 1.15 + 0.08);
    glow = 0.36 * vFold;
  } else {
    /* Bright architectural glass-mirror slab thickness edge (matches sunlit room, zero dark grey bars!) */
    col = mix(archSrgb * 0.85, reflSrgb * 0.95, 0.45) * diff;
  }

  float alpha = smoothstep(0.004, 0.10, vFold) * ((vEdge < 0.30) ? 0.92 : 0.96);
  oCol = vec4(clamp(col, 0.0, 1.0), alpha);
  oMeta = vec4(alpha, glow * alpha, 0.0, 1.0);
}
"""

POST_FS = """#version 330
uniform sampler2D uScene, uMeta;
uniform vec2 uRes;
uniform float uBlackFloor, uGrainAmt, uFrame;
out vec4 oCol;
float hash21(vec2 p){ vec3 q=fract(vec3(p.xyx)*0.1031); q+=dot(q,q.yzx+33.33); return fract((q.x+q.y)*q.z); }

void main() {
  vec2 uv = gl_FragCoord.xy / uRes;
  vec2 tex = 1.0 / uRes;
  vec3 c0 = texture(uScene, uv).rgb;
  vec2 m0 = texture(uMeta, uv).rg;
  float cgA = clamp(m0.r, 0.0, 1.0);

  vec3 bevelGlow = vec3(0.0);
  for (int k = -2; k <= 2; k++) {
    vec2 o1 = vec2(float(k) * 2.0 * tex.x, 0.0);
    vec2 o2 = vec2(0.0, float(k) * 2.0 * tex.y);
    bevelGlow += texture(uScene, uv + o1).rgb * texture(uMeta, uv + o1).g * 0.055;
    bevelGlow += texture(uScene, uv + o2).rgb * texture(uMeta, uv + o2).g * 0.055;
  }

  vec3 res = c0 + bevelGlow * vec3(0.96, 0.98, 1.00) * 0.45;
  if (cgA > 0.01) {
    vec3 graded = max(res, vec3(uBlackFloor));
    vec2 p1 = gl_FragCoord.xy + vec2(uFrame * 17.3, uFrame * 31.7);
    vec2 p2 = floor(gl_FragCoord.xy * 0.5) + vec2(uFrame * 11.1, uFrame * 23.9);
    float g = (hash21(p1)+hash21(p1+7.13)-1.0)*0.62 + (hash21(p2)+hash21(p2+5.71)-1.0)*0.38;
    float lum = dot(graded, vec3(0.299, 0.587, 0.114));
    graded += vec3(g) * (uGrainAmt * (1.10 - 0.55 * smoothstep(0.55, 0.98, lum)));
    res = mix(res, graded, cgA);
  }
  oCol = vec4(clamp(res, 0.0, 1.0), 1.0);
}
"""


def mat4_perspective_from_opencv(f_px, W, H, near=0.1, far=60.0):
    P = np.zeros((4, 4), dtype=np.float32)
    P[0, 0] = 2.0 * f_px / W; P[1, 1] = 2.0 * f_px / H
    P[2, 2] = -(far + near) / (far - near); P[2, 3] = -2.0 * far * near / (far - near); P[3, 2] = -1.0
    return P


def mat4_view_from_colmap(ro, R_c2w):
    flip = np.diag([1.0, -1.0, -1.0]); R_w2gl = flip @ R_c2w.T
    V = np.eye(4, dtype=np.float32); V[:3, :3] = R_w2gl; V[:3, 3] = -R_w2gl @ ro
    return V


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--still", default=None)
    ap.add_argument("--render", action="store_true")
    args = ap.parse_args()

    c = load(SRC_PIECE)
    S = scene(c)
    C = np.asarray(S["cam"], dtype=np.float64)
    R_all = np.asarray(S["rot"], dtype=np.float64)
    f_px = float(S["f"])
    N = len(C)
    fps = float(c["fps"])
    pl0 = frame(c, 0)
    H, W = pl0.shape[:2]

    slabs = generate_monumental_mirror_architecture()
    all_local_pos, all_local_nrm, all_rest_w, all_eflag, all_idx = [], [], [], [], []
    v_off = 0
    for sl in slabs:
        vd = sl["vdata"]
        lp = vd[:, 0:3].astype(np.float64)
        ln = vd[:, 3:6].astype(np.float64)
        rl = vd[:, 6:9].astype(np.float64)
        rw = sl["P_rest"][None, :] + rl @ sl["R_rest"].T
        all_local_pos.append(lp); all_local_nrm.append(ln)
        all_rest_w.append(rw.astype(np.float32)); all_eflag.append(vd[:, 9].astype(np.float32))
        all_idx.append(sl["idata"] + v_off)
        v_off += len(vd)

    idx_arr = np.concatenate(all_idx).astype(np.uint32)
    n_verts = v_off
    print(f"Monumental Mirror Dimension: {len(slabs)} large architectural slabs & Sanctum rings | {len(idx_arr)//3} tris")

    gray0 = cv2.cvtColor(pl0, cv2.COLOR_BGR2GRAY).astype(np.float32) / 255.0
    black_floor = float(np.clip(np.percentile(gray0, 1.0) * 0.85, 0.008, 0.050))
    hp0 = gray0 - cv2.GaussianBlur(gray0, (0, 0), 1.2)
    grain_amt = float(np.clip(float(np.median(np.abs(hp0 - np.median(hp0)))) * 1.4826 * 1.10, 0.008, 0.022))

    ctx = moderngl.create_standalone_context(require=330)
    bg_prog   = ctx.program(vertex_shader=VS_FSQ,  fragment_shader=BG_FS)
    slab_prog = ctx.program(vertex_shader=SLAB_VS, fragment_shader=SLAB_FS)
    post_prog = ctx.program(vertex_shader=VS_FSQ,  fragment_shader=POST_FS)

    fsq = ctx.buffer(np.array([-1, -1, 1, -1, -1, 1, -1, 1, 1, -1, 1, 1], dtype=np.float32).tobytes())
    va_bg   = ctx.vertex_array(bg_prog,   [(fsq, "2f", "p")])
    va_post = ctx.vertex_array(post_prog, [(fsq, "2f", "p")])

    vbo = ctx.buffer(reserve=n_verts * 11 * 4)
    ibo = ctx.buffer(idx_arr.tobytes())
    va_slab = ctx.vertex_array(slab_prog, [(vbo, "3f 3f 3f 1f 1f", "inPos", "inNrm", "inRestWorld", "inEdgeFlag", "inFoldAmt")], index_buffer=ibo)

    t_scene = ctx.texture((W, H), 4, dtype="f4")
    t_meta  = ctx.texture((W, H), 4, dtype="f4")
    rbo_dep = ctx.depth_renderbuffer((W, H))
    fbo_scene = ctx.framebuffer([t_scene, t_meta], depth_attachment=rbo_dep)
    t_final = ctx.texture((W, H), 4, dtype="f4")
    fbo_final = ctx.framebuffer([t_final])
    plate_t = ctx.texture((W, H), 3); plate_t.filter = (moderngl.LINEAR, moderngl.LINEAR); plate_t.repeat_x = plate_t.repeat_y = False
    for t in (t_scene, t_meta):
        t.filter = (moderngl.LINEAR, moderngl.LINEAR); t.repeat_x = t.repeat_y = False

    proj_m = mat4_perspective_from_opencv(f_px, W, H)
    bg_prog["uRes"].value = (float(W), float(H))
    slab_prog["uProj"].write(proj_m.T.tobytes())
    slab_prog["uRes"].value = (float(W), float(H)); slab_prog["uF"].value = f_px
    post_prog["uRes"].value = (float(W), float(H))
    post_prog["uBlackFloor"].value = black_floor; post_prog["uGrainAmt"].value = grain_amt

    v_stage = np.zeros((n_verts, 11), dtype=np.float32)
    v_stage[:, 6:9] = np.concatenate(all_rest_w, axis=0)
    v_stage[:, 9]   = np.concatenate(all_eflag, axis=0)

    def update_folded_geometry(i):
        t = i / fps
        env_in  = smootherstep(22.0, 68.0, float(i))
        env_out = 1.0 - smootherstep(172.0, 218.0, float(i))
        master = env_in * env_out

        wave_y   = -5.8 + max(0.0, (i - 20.0) / 48.0) * 14.0
        unfold_y = -5.8 + max(0.0, (i - 168.0) / 46.0) * 14.0

        off = 0
        for si, sl in enumerate(slabs):
            lp = all_local_pos[si]; ln = all_local_nrm[si]
            nv = len(lp)
            y_c = sl["y_coord"]
            act_in  = smootherstep(y_c - 1.6, y_c + 1.6, wave_y)
            act_out = 1.0 - smootherstep(y_c - 1.6, y_c + 1.6, unfold_y)
            f_amt = act_in * act_out * master

            if f_amt < 0.002:
                P_w = sl["P_rest"][None, :] + lp @ sl["R_rest"].T
                N_w = ln @ sl["R_rest"].T
            else:
                breath = 0.88 + 0.14 * math.sin(t * 1.15 + sl["phase"])
                if sl["is_ring"]:
                    # Concentric octagonal Sanctum mirror blades rotate cleanly around the OPEN central corridor!
                    ang_now = sl["ring_a0"] + 0.42 * sl["spin_dir"] * t * f_amt
                    r_now = sl["ring_r"] * (1.08 - 0.12 * f_amt)
                    P_cur = np.array([r_now * math.cos(ang_now), y_c, 1.68 + r_now * 0.72 * math.sin(ang_now)])
                    R_rot = axis_angle_mat3(np.array([0.0, 1.0, 0.0]), ang_now - sl["ring_a0"])
                    R_tilt = axis_angle_mat3(np.array([math.cos(ang_now), 0.0, math.sin(ang_now)]), sl["fold_ang"] * f_amt * breath)
                    R_cur = R_tilt @ R_rot @ sl["R_rest"]
                else:
                    # Monumental synchronized architectural bay hinging!
                    ang = sl["fold_ang"] * f_amt * breath
                    R_hinge = axis_angle_mat3(sl["hinge"], ang)
                    R_cur = R_hinge @ sl["R_rest"]
                    P_cur = sl["P_rest"] + sl["slide_vec"] * (f_amt * breath)

                P_w = P_cur[None, :] + lp @ R_cur.T
                N_w = ln @ R_cur.T

            v_stage[off:off + nv, 0:3] = P_w.astype(np.float32)
            v_stage[off:off + nv, 3:6] = N_w.astype(np.float32)
            v_stage[off:off + nv, 10]  = float(f_amt)
            off += nv

        vbo.write(v_stage.tobytes())
        return master

    def render_one(i):
        pl = frame(c, i)
        plate_t.write(pl[:, :, ::-1].tobytes())
        master = update_folded_geometry(i)

        ro = C[i]; R = R_all[i]
        Rt = tuple(R.T.flatten().astype(np.float32))
        view_m = mat4_view_from_colmap(ro, R)

        fbo_scene.use(); ctx.viewport = (0, 0, W, H); ctx.disable(moderngl.DEPTH_TEST); ctx.disable(moderngl.BLEND)
        fbo_scene.clear(0.0, 0.0, 0.0, 0.0, depth=1.0)
        plate_t.use(0)
        bg_prog["uPlate"].value = 0; bg_prog["uKaleid"].value = float(master); bg_prog["uTime"].value = float(i / fps)
        va_bg.render(moderngl.TRIANGLES)

        if master > 0.002:
            ctx.enable(moderngl.DEPTH_TEST); ctx.depth_mask = True
            ctx.enable(moderngl.BLEND); ctx.blend_func = (moderngl.SRC_ALPHA, moderngl.ONE_MINUS_SRC_ALPHA)
            plate_t.use(0)
            slab_prog["uPlate"].value = 0; slab_prog["uView"].write(view_m.T.tobytes())
            slab_prog["uRo"].value = tuple(map(float, ro)); slab_prog["uCam"].value = Rt
            va_slab.render(moderngl.TRIANGLES)

        fbo_final.use(); ctx.viewport = (0, 0, W, H); ctx.disable(moderngl.DEPTH_TEST); ctx.disable(moderngl.BLEND)
        t_scene.use(0); t_meta.use(1)
        post_prog["uScene"].value = 0; post_prog["uMeta"].value = 1; post_prog["uFrame"].value = float(i)
        va_post.render(moderngl.TRIANGLES)
        ctx.finish()

        arr = np.frombuffer(t_final.read(), "f4").reshape(H, W, 4)[::-1]
        img = (np.clip(arr[..., :3], 0, 1) * 255 + 0.5).astype(np.uint8)[:, :, ::-1]
        return pl, img

    wk = here / "work"; wk.mkdir(parents=True, exist_ok=True)
    if args.still:
        idxs = [int(x) for x in args.still.split(",")]
        rows = []
        for idx in idxs:
            pl, comp = render_one(idx)
            a, b = cv2.resize(pl, (640, 360)), cv2.resize(comp, (640, 360))
            cv2.putText(b, f"mirror f={idx}", (10, 344), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 0, 0), 3)
            cv2.putText(b, f"mirror f={idx}", (10, 344), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (140, 235, 255), 1)
            rows.append(np.hstack([a, b]))
        out_p = wk / "still_mirror_sheet.jpg"
        cv2.imwrite(str(out_p), np.vstack(rows), [cv2.IMWRITE_JPEG_QUALITY, 92])
        print("stills ->", out_p)
        return

    if args.render:
        comp_dir = wk / "comp"
        shutil.rmtree(comp_dir, ignore_errors=True); comp_dir.mkdir(parents=True)
        ts = time.time()
        for i in range(N):
            _, comp = render_one(i)
            cv2.imwrite(str(comp_dir / f"{i:04d}.png"), comp, [cv2.IMWRITE_PNG_COMPRESSION, 1])
            if i % 40 == 0 or i == N - 1:
                print(f"mirror_dim frame {i}/{N} ({time.time() - ts:.1f}s)", flush=True)
        out_dir = here / "out"; out_dir.mkdir(parents=True, exist_ok=True)
        r = f"{c['fps_num']}/{c['fps_den']}"; ff = find_ffmpeg()
        tags = ["-pix_fmt", "yuv420p", "-color_primaries", "bt709", "-color_trc", "bt709", "-colorspace", "bt709", "-movflags", "+faststart"]
        x264 = ["-c:v", "libx264", "-profile:v", "high", "-level", "4.1", "-crf", "14", "-preset", "slow"]
        run = lambda a: subprocess.run([ff, "-v", "error", "-y"] + a, check=True)
        run(["-framerate", r, "-i", str(comp_dir / "%04d.png")] + x264 + tags + [str(out_dir / "Mirror_Dimension_Folding_Composite.mp4")])
        run([
            "-framerate", r, "-i", str(comp_dir / "%04d.png"),
            "-framerate", r, "-i", str(work(c, "frames", "%04d.jpg")),
            "-filter_complex", "[0]scale=1080:608[a];[1]scale=1080:608[b];[a][b]vstack,pad=1080:1350:0:67:color=black",
        ] + x264 + tags + [str(out_dir / "Mirror_Dimension_Folding_Compare_1080x1350.mp4")])
        print("encoded ->", out_dir)


if __name__ == "__main__":
    main()
