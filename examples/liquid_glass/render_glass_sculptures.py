"""Diverse Architectural & Organic 3D Liquid Glass Sculpture Gallery (ModernGL, OpenGL 3.3).

    python render_glass_sculptures.py [piece_dir] --still 0,25,85,155,195,245 --tag sculptures_v2
    python render_glass_sculptures.py [piece_dir] --tag sculptures_v2 --out out_sculptures_v2 --suffix _Sculptures_v2 --passes

Zero-Duplicate 11-Sculpture Edition (supports multi-piece config via piece.json):
- 11 completely distinct museum-grade sculptural families (ZERO repeated ring/twist silhouettes):
  0) Fluted Concave Hourglass / Optical Amphora Vessel (turned concave vase with spiral flutes)
  1) Intersecting Architectural Cross-Blade Glass Prism (+ blades with dual circular viewing apertures)
  2) Arched Hollow Fluted Glass Portal (Roman arch monolith with open doorway & vertical lens ribs)
  3) Helically Twisted Triangular Optical Prism Column (vertical crystal obelisk)
  4) Organic Rippling Liquid-Glass Metaball Wave (morphing fluid mass)
  5) Sinuous S-Curved Wave Fluted Glass Screen (curved reed-glass partition)
  6) Cut-Crystal Faceted Octahedral Gem Polyhedron (museum cut diamond)
  7) Sculptural 3D Murano Trefoil Glass Knot (the ONLY tubular loop/knot in the scene!)
  8) Stacked 4-Tier Biconvex Optical Lens Totem (breathing horizontal Fresnel magnifying discs)
  9) Carved Hollow Rippled Glass Shell Vessel (diagonally sliced translucent orb + floating inner pearl)
  10) Grand Climax: Open Beveled Crystal Hypercube / Tesseract Frame enclosing a Morphing Liquid-Glass Heart
- Restrained accent micro-droplets (~25 subtle dew beads tucked near organic liquid forms)
- Up to 12 3D COLMAP architectural holdout boxes (walls + structural columns)
- Single-pass MRT diagnostic export (--passes -> normal, refract, caustic) respecting real 3D pillar occlusion!
"""
import json, math, shutil, subprocess, sys, time
from pathlib import Path
import numpy as np, cv2, moderngl

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


DEFAULT_OCC_BOXES_HALL = [
    (-4.00, -0.71, -13.50, -8.56, 0.0, 5.0),  # Left entry hallway solid wall
    ( 2.28,  4.50, -13.50, -8.85, 0.0, 5.0),  # Right entry hallway solid wall
    (-1.50, -0.90,  -1.12, -0.57, 0.0, 5.0),  # Left white structural pillar #1 (foreground/mid)
    ( 2.24,  2.82,  -1.00, -0.42, 0.0, 5.0),  # Right white structural pillar #1 (beside stairs)
    (-1.56, -1.07,   7.10,  7.65, 0.0, 5.0),  # Left white structural pillar #2 (far hall)
    ( 1.54,  2.05,   7.10,  7.65, 0.0, 5.0),  # Right white structural pillar #2 (far hall)
]

# Default 11 ZERO-DUPLICATE sculptures for piece_glass (Sunlit Architectural Hall)
DEFAULT_SCULPT_DEFS_HALL = [
    # 0: Fluted Concave Hourglass / Optical Amphora Vessel — front-upper-left entryway (f0..55)
    ((-0.14, -9.12, 1.44, 0.54), (0.0, math.radians(15.0), 0.4, 0.62)),
    # 1: Intersecting Architectural Cross-Blade Prism w/ Apertures — front-right past entry (f0..95)
    (( 1.14, -8.00, 1.06, 0.84), (1.0, math.radians(-20.0), 1.3, 0.86)),
    # 2: Arched Hollow Fluted Glass Portal — front-left lounge peeking cleanly past LeftEntryWall (f20..125)
    ((-1.10, -5.85, 1.10, 1.00), (2.0, math.radians(18.0), 2.1, 1.14)),
    # 3: Helically Twisted Triangular Optical Prism Column — front-right stairs (f15..125)
    (( 1.36, -6.18, 1.24, 0.72), (3.0, math.radians(-22.0), 0.9, 0.96)),
    # 4: Organic Rippling Liquid-Glass Metaball Wave — mid-right beside staircase (f30..165)
    (( 1.20, -4.05, 1.30, 0.90), (4.0, math.radians(-12.0), 3.1, 1.18)),
    # 5: Sinuous S-Curved Wave Fluted Glass Screen — mid-left hero 1m IN FRONT of White Pillar #1 (f75..195)
    ((-0.90, -2.18, 1.32, 1.00), (5.0, math.radians(20.0), 1.7, 1.32)),
    # 6: Cut-Crystal Faceted Gem Polyhedron — mid-upper floating jewel near ceiling beams
    (( 0.88, -2.80, 2.16, 0.50), (6.0, math.radians(25.0), 4.2, 0.55)),
    # 7: Sculptural 3D Murano Trefoil Glass Knot — ONLY tubular knot in hall! (f95..255)
    (( 1.26,  0.72, 1.60, 0.92), (7.0, math.radians(-24.0), 2.6, 0.92)),
    # 8: Stacked Floating Biconvex Optical Lens Totem — far-left BEHIND White Pillar #1 until f~190!
    ((-1.26,  1.35, 1.54, 0.88), (8.0, math.radians(14.0), 0.7, 1.05)),
    # 9: Carved Hollow Rippled Glass Shell Vessel — far-right near dining chandelier (f140..270)
    (( 1.48,  2.55, 1.44, 0.78), (9.0, math.radians(-30.0), 5.0, 0.76)),
    # 10: Grand Center-Hall Climax — Open Beveled Crystal Hypercube Frame enclosing a Morphing Liquid-Glass Heart!
    (( 0.06,  3.85, 1.62, 1.24), (10.0, math.radians(14.0), 1.1, 1.42)),
]


def get_piece_sculptures_and_occluders(c):
    """Loads per-piece sculptures & occ_boxes from piece.json if provided, else defaults to Sunlit Hall v2."""
    occ_raw = c.get("occ_boxes")
    if occ_raw is not None:
        occ_boxes = [tuple(map(float, b)) for b in occ_raw]
    else:
        occ_boxes = list(DEFAULT_OCC_BOXES_HALL)

    sc_raw = c.get("sculptures")
    if sc_raw is not None:
        sculpt_defs = []
        for s in sc_raw:
            pos = tuple(map(float, s["pos"]))
            cfg = (float(s["type"]), math.radians(float(s.get("yaw_deg", 0.0))),
                   float(s.get("phase", 0.0)), float(s.get("bound", 1.1)))
            sculpt_defs.append((pos, cfg))
    else:
        sculpt_defs = list(DEFAULT_SCULPT_DEFS_HALL)
    return sculpt_defs, occ_boxes


VS = "#version 330\nin vec2 p; void main(){ gl_Position = vec4(p, 0.0, 1.0); }"

# Pass 1: 11 Zero-Duplicate Sculptural Glass Forms + Floor Caustics + 12 3D Architectural Holdout Boxes + Diagnostic Passes
HERO_FS = """#version 330
uniform sampler2D uPlate;
uniform vec2 uRes;
uniform float uF, iTime;
uniform vec3 uRo, uSun;
uniform mat3 uCam;

/* Up to 11 diverse hero sculptures: uSculptPos[i] = (x, y, z, scale), uSculptCfg[i] = (typeId, yawRad, phase, boundRad) */
uniform vec4 uSculptPos[11];
uniform vec4 uSculptCfg[11];
uniform int uNumSculpt;

/* Up to 12 3D architectural holdout boxes (walls + structural columns) */
uniform vec3 uOccMin[12];
uniform vec3 uOccMax[12];
uniform int uNumOcc;

uniform float uIor, uDisp, uCausticGain, uShadowGain, uFluteFreq, uFluteAmp, uMetaK, uProxyDist;
uniform vec3 uTint, uSunlitCol;
uniform vec2 uFloorLuma;
uniform int uExportPasses;

layout(location=0) out vec4 oComp;
layout(location=1) out vec4 oDepth;   /* R: min(tGlass, tOcc), G: tOcc */
layout(location=2) out vec4 oMeta;    /* R: cgAlpha, G: cgInvDepth, B: occInvDepth, A: 1.0 */
layout(location=3) out vec4 oNormal;
layout(location=4) out vec4 oRefract;
layout(location=5) out vec4 oCaustic;

float hash31(vec3 p) {
  p = fract(p * 0.1031);
  p += dot(p, p.yzx + 33.33);
  return fract((p.x + p.y) * p.z);
}

float vnoise(vec3 p) {
  vec3 i = floor(p), f = fract(p);
  f = f * f * (3.0 - 2.0 * f);
  return mix(
    mix(mix(hash31(i), hash31(i + vec3(1,0,0)), f.x),
        mix(hash31(i + vec3(0,1,0)), hash31(i + vec3(1,1,0)), f.x), f.y),
    mix(mix(hash31(i + vec3(0,0,1)), hash31(i + vec3(1,0,1)), f.x),
        mix(hash31(i + vec3(0,1,1)), hash31(i + vec3(1,1,1)), f.x), f.y),
    f.z
  );
}

float smin(float a, float b, float k) {
  float h = max(k - abs(a - b), 0.0) / max(k, 1e-5);
  return min(a, b) - h * h * k * 0.25;
}

float smax(float a, float b, float k) {
  return -smin(-a, -b, k);
}

float sdRoundBox(vec3 p, vec3 b, float r) {
  vec3 q = abs(p) - b + r;
  return length(max(q, 0.0)) + min(max(q.x, max(q.y, q.z)), 0.0) - r;
}

float sdBoxFrame(vec3 p, vec3 b, float e) {
  p = abs(p) - b;
  vec3 q = abs(p + e) - e;
  return min(min(
      length(max(vec3(p.x, q.y, q.z), 0.0)) + min(max(p.x, max(q.y, q.z)), 0.0),
      length(max(vec3(q.x, p.y, q.z), 0.0)) + min(max(q.x, max(p.y, q.z)), 0.0)),
      length(max(vec3(q.x, q.y, p.z), 0.0)) + min(max(q.x, max(q.y, p.z)), 0.0));
}

float sdCylinderY(vec3 p, float r, float hy) {
  vec2 d = abs(vec2(length(p.xz), p.y)) - vec2(r, hy);
  return min(max(d.x, d.y), 0.0) + length(max(d, 0.0));
}

float sdCylinderZ(vec3 p, float r, float hz) {
  vec2 d = abs(vec2(length(p.xy), p.z)) - vec2(r, hz);
  return min(max(d.x, d.y), 0.0) + length(max(d, 0.0));
}

/* Exact beveled octahedron for crystal gem polyhedron */
float sdOctahedron(vec3 p, float s) {
  p = abs(p);
  float m = p.x + p.y + p.z - s;
  vec3 q;
  if (3.0 * p.x < m) q = p.xyz;
  else if (3.0 * p.y < m) q = p.yzx;
  else if (3.0 * p.z < m) q = p.zxy;
  else return m * 0.57735027;
  float k = clamp(0.5 * (q.z - q.y + s), 0.0, s);
  return length(vec3(q.x, q.y - s + k, q.z - k));
}

/* Exact equilateral triangular prism along Z */
float sdTriPrismZ(vec3 p, vec2 h) {
  vec3 q = abs(p);
  return max(q.z - h.y, max(q.x * 0.866025 + p.y * 0.5, -p.y) - h.x * 0.5);
}

mat2 rot2(float a) {
  float c = cos(a), s = sin(a);
  return mat2(c, -s, s, c);
}

/* Exact ray-AABB intersection against real walls & structural columns */
float hitOccluders(vec3 ro, vec3 rd) {
  float tBest = 1e4;
  vec3 invD = 1.0 / mix(rd, vec3(1e-8), lessThan(abs(rd), vec3(1e-8)));
  for (int i = 0; i < 12; i++) {
    if (i >= uNumOcc) break;
    vec3 t0 = (uOccMin[i] - ro) * invD;
    vec3 t1 = (uOccMax[i] - ro) * invD;
    vec3 mn = min(t0, t1);
    vec3 mx = max(t0, t1);
    float tNear = max(max(mn.x, mn.y), mn.z);
    float tFar  = min(min(mx.x, mx.y), mx.z);
    if (tFar > max(tNear, 0.05)) {
      float tHit = (tNear > 0.05) ? tNear : 0.05;
      tBest = min(tBest, tHit);
    }
  }
  return tBest;
}

bool insideOccXY(vec2 xy) {
  for (int i = 0; i < 12; i++) {
    if (i >= uNumOcc) break;
    if (xy.x >= uOccMin[i].x && xy.x <= uOccMax[i].x &&
        xy.y >= uOccMin[i].y && xy.y <= uOccMax[i].y) return true;
  }
  return false;
}

/* ============================================================================
   11 ZERO-DUPLICATE MUSEUM-GRADE SCULPTURAL GLASS SDFs (Lipschitz-safe & crisp!)
   matKind returned in y:
     0 = organic/smooth blown crystal glass
     1 = fluted vertical optical lens ribs (for fluted arch & S-wave screen)
     2 = faceted prismatic architectural crystal (extra crisp dispersion!)
   ============================================================================ */
vec2 evalSculpture(vec3 p, int idx, out vec3 localP) {
  vec4 sp = uSculptPos[idx];
  vec4 cfg = uSculptCfg[idx];
  int stype = int(cfg.x + 0.5);
  float yaw = cfg.y;
  float ph = cfg.z;
  float scl = sp.w;
  float t = iTime * 0.72 + ph;

  vec3 q = p - sp.xyz;
  q.z -= 0.030 * sin(iTime * 0.85 + ph);
  q.xy = rot2(-yaw) * q.xy;
  localP = q;
  vec3 qs = q / scl;

  /* Type 0: FLUTED CONCAVE HOURGLASS / OPTICAL AMPHORA VESSEL (replaces Mobius Ring) */
  if (stype == 0) {
    vec3 qa = qs;
    qa.xy = rot2(t * 0.30) * qa.xy;
    qa.xz = rot2(0.18 * sin(t * 0.50) + 0.14) * qa.xz;
    float zn = clamp(qa.z / 0.68, -1.0, 1.0);
    /* Pinched hourglass waist flaring gracefully to curved chalice rim & foot */
    float rProf = 0.17 + 0.27 * (zn * zn) + 0.035 * zn;
    float rxy = length(qa.xy);
    vec2 dd = vec2(rxy - rProf, abs(qa.z) - 0.66);
    float dBody = length(max(dd, 0.0)) + min(max(dd.x, dd.y), 0.0) - 0.028;
    /* Carve deep concave upper chalice bowl & lower foot recess */
    float cavTop = length(qa - vec3(0.0, 0.0,  0.66)) - 0.34;
    float cavBot = length(qa - vec3(0.0, 0.0, -0.66)) - 0.30;
    float dVase = smax(smax(dBody, -cavTop, 0.045), -cavBot, 0.045);
    /* Crisp diagonal spiral optical flutes along hourglass exterior */
    float ang = atan(qa.y, qa.x);
    float fluteEnv = smoothstep(0.66, 0.44, abs(qa.z));
    dVase += sin(ang * 14.0 + qa.z * 5.2) * 0.011 * fluteEnv;
    return vec2(dVase * 0.62 * scl, 0.0);
  }

  /* Type 1: INTERSECTING ARCHITECTURAL CROSS-BLADE PRISM WITH DUAL APERTURES (replaces Gyro Rings) */
  if (stype == 1) {
    vec3 qb = qs;
    qb.xy = rot2(t * 0.26) * qb.xy;
    qb.yz = rot2(0.18 * sin(t * 0.45) + 0.12) * qb.yz;
    float b1 = sdRoundBox(qb, vec3(0.56, 0.072, 0.62), 0.028);
    float b2 = sdRoundBox(qb, vec3(0.072, 0.56, 0.62), 0.028);
    float crossB = smin(b1, b2, 0.048);
    /* Dual orthogonal circular porthole viewing apertures right through the cross center! */
    float holeY = length(qb.xz) - 0.22;
    float holeX = length(qb.yz) - 0.22;
    float dCross = smax(smax(crossB, -holeY, 0.034), -holeX, 0.034);
    /* Subtle architectural horizontal optical grooves near upper & lower blade tips */
    dCross += sin(qb.z * 28.0) * 0.004 * smoothstep(0.24, 0.45, abs(qb.z));
    return vec2(dCross * 0.78 * scl, 2.0);
  }

  /* Type 2: ARCHED HOLLOW FLUTED GLASS PORTAL (Roman arch monolith with open central arch aperture!) */
  if (stype == 2) {
    vec3 qb = qs;
    qb.z += 0.18;
    float lowerBox = sdRoundBox(qb, vec3(0.38, 0.052, 0.58), 0.028);
    vec3 qa = qs - vec3(0.0, 0.0, 0.34);
    float topArch = sdCylinderY(qa, 0.38, 0.052) - 0.012;
    float outerArch = smin(lowerBox, topArch, 0.025);

    vec3 qh = qs - vec3(0.0, 0.0, -0.02);
    float holeBox = sdRoundBox(qh + vec3(0.0, 0.0, 0.14), vec3(0.145, 0.20, 0.28), 0.015);
    float holeTop = sdCylinderY(qh - vec3(0.0, 0.0, 0.16), 0.145, 0.20);
    float hole = min(holeBox, holeTop);
    float portal = smax(outerArch, -hole, 0.028);

    float faceMask = smoothstep(0.018, 0.065, 0.38 - abs(qs.x)) * smoothstep(0.02, 0.07, 0.055 - abs(qs.y) + 0.04);
    float d = portal + sin(qs.x * uFluteFreq * scl) * (0.0050 / max(scl, 0.4)) * faceMask;
    localP = qs * scl;
    return vec2(d * 0.82 * scl, 1.0);
  }

  /* Type 3: HELICALLY TWISTED TRIANGULAR OPTICAL PRISM COLUMN (vertical crystal obelisk) */
  if (stype == 3) {
    vec3 qt = qs;
    qt.xy = rot2(qt.z * 1.85 + t * 0.40) * qt.xy;
    float prism = sdTriPrismZ(qt, vec2(0.40, 0.86)) - 0.052;
    float a3 = atan(qt.y, qt.x);
    prism += sin(a3 * 6.0) * 0.0035 * smoothstep(0.86, 0.62, abs(qt.z));
    return vec2(prism * 0.46 * scl, 2.0);
  }

  /* Type 4: ORGANIC RIPPLING LIQUID-GLASS METABALL SCULPTURE */
  if (stype == 4) {
    vec3 c0 = vec3(0.06 * sin(t * 0.7), 0.05 * cos(t * 0.6), 0.08 * sin(t * 0.9));
    float d = length(qs - c0) - 0.32;
    vec3 c1 = vec3( 0.44 * cos(t * 0.95),        0.32 * sin(t * 0.80),  0.22 * sin(t * 1.15));
    vec3 c2 = vec3(-0.40 * cos(t * 0.78 + 1.2),  0.36 * sin(t * 0.92), -0.20 * cos(t * 1.05));
    vec3 c3 = vec3( 0.30 * sin(t * 1.22 + 2.4), -0.30 * cos(t * 0.85),  0.34 * cos(t * 0.90));
    vec3 c4 = vec3(-0.32 * sin(t * 0.90 + 4.0), -0.24 * sin(t * 1.10), -0.26 * sin(t * 0.82));
    vec3 c5 = vec3( 0.52 * cos(t * 0.62 + 3.1),  0.16 * cos(t * 1.30),  0.14 * sin(t * 1.40));
    float k = uMetaK;
    d = smin(d, length(qs - c1) - 0.21, k);
    d = smin(d, length(qs - c2) - 0.19, k);
    d = smin(d, length(qs - c3) - 0.18, k);
    d = smin(d, length(qs - c4) - 0.16, k);
    d = smin(d, length(qs - c5) - 0.15, k * 0.85);
    float rip = (vnoise(qs * 3.1 + vec3(0.0, t * 0.40, -t * 0.30)) - 0.5) * 0.018;
    return vec2((d + rip) * scl, 0.0);
  }

  /* Type 5: SINUOUS S-CURVED WAVE FLUTED GLASS SCREEN */
  if (stype == 5) {
    vec3 qw = qs;
    float waveY = 0.145 * sin(qw.x * 4.4 + 0.35 * sin(t * 0.7));
    qw.y -= waveY;
    vec3 sz = vec3(0.48, 0.046, 0.92);
    float slab = sdRoundBox(qw, sz, 0.032);
    float faceMask = smoothstep(0.02, 0.08, sz.x - abs(qw.x)) * smoothstep(0.02, 0.08, sz.z - abs(qw.z));
    float d = slab + sin(qw.x * uFluteFreq * scl) * (0.0052 / max(scl, 0.4)) * faceMask;
    localP = qw * scl;
    return vec2(d * 0.68 * scl, 1.0);
  }

  /* Type 6: CUT-CRYSTAL FACETED GEM POLYHEDRON */
  if (stype == 6) {
    vec3 qg = qs;
    qg.xy = rot2(t * 0.36) * qg.xy;
    qg.xz = rot2(t * 0.24 + 0.5) * qg.xz;
    float oct = sdOctahedron(qg, 0.68);
    vec3 qg2 = qg;
    qg2.xy = rot2(0.7854) * qg2.xy;
    float cub = sdRoundBox(qg2, vec3(0.46, 0.46, 0.54), 0.035);
    float gem = smax(oct, cub, 0.045) - 0.018;
    return vec2(gem * 0.85 * scl, 2.0);
  }

  /* Type 7: SCULPTURAL 3D MURANO TREFOIL GLASS KNOT (THE ONLY tubular knot/loop in the scene!) */
  if (stype == 7) {
    vec3 qk = qs;
    qk.xy = rot2(t * 0.34) * qk.xy;
    qk.yz = rot2(0.38 + 0.18 * sin(t * 0.5)) * qk.yz;
    float a = atan(qk.z, qk.x);
    float r = length(qk.xz) - 0.46;
    vec2 cp = vec2(r, qk.y);
    cp = rot2(a * 1.5) * cp;
    cp.x = abs(cp.x) - 0.145;
    float dTube = length(cp) - 0.095;
    return vec2(dTube * 0.58 * scl, 0.0);
  }

  /* Type 8: STACKED FLOATING BICONVEX OPTICAL LENS TOTEM (4 breathing glass magnifying discs) */
  if (stype == 8) {
    float dBest = 1e4;
    for (int k = 0; k < 4; k++) {
      float fk = float(k);
      float zOff = -0.60 + fk * 0.40 + 0.045 * sin(t * 1.1 + fk * 1.3);
      vec3 ql = qs - vec3(0.06 * sin(fk * 1.8 + t * 0.4), 0.05 * cos(fk * 1.5 + t * 0.4), zOff);
      float tiltA = 0.24 * sin(t * 0.6 + fk * 1.7) + (mod(fk, 2.0) < 0.5 ? 0.16 : -0.16);
      ql.xz = rot2(tiltA) * ql.xz;
      ql.yz = rot2(0.14 * cos(t * 0.5 + fk)) * ql.yz;
      float discR = 0.46 - abs(fk - 1.5) * 0.065;
      float lens = max(length(ql - vec3(0.0, 0.0, -0.62)) - 0.72,
                       length(ql - vec3(0.0, 0.0,  0.62)) - 0.72);
      float cyl = sdCylinderZ(ql, discR, 0.12) - 0.025;
      float dDisc = smax(cyl, lens, 0.032);
      float rr = length(ql.xy);
      dDisc += sin(rr * 38.0) * 0.004 * smoothstep(discR, discR * 0.65, rr);
      dBest = min(dBest, dDisc);
    }
    return vec2(dBest * 0.78 * scl, 2.0);
  }

  /* Type 9: CARVED HOLLOW RIPPLED GLASS SHELL VESSEL */
  if (stype == 9) {
    vec3 qv = qs;
    qv.xy = rot2(t * 0.28) * qv.xy;
    qv.xz = rot2(0.45 + 0.16 * sin(t * 0.6)) * qv.xz;
    float shell = abs(length(qv) - 0.54) - 0.052;
    float cutPlane = dot(qv, normalize(vec3(0.55, -0.68, 0.48))) - 0.10;
    float bowl = smax(shell, cutPlane, 0.035);
    bowl += sin(qv.z * 28.0 + qv.x * 10.0) * 0.007;
    float pearl = length(qv - vec3(-0.08, 0.06, 0.06 * sin(t * 1.2))) - 0.16;
    return vec2(min(bowl * 0.75, pearl) * scl, 0.0);
  }

  /* Type 10: GRAND FINALE — OPEN BEVELED CRYSTAL HYPERCUBE / TESSERACT FRAME + MORPHING LIQUID-GLASS HEART! */
  vec3 qf = qs;
  qf.xy = rot2(t * 0.24) * qf.xy;
  qf.yz = rot2(0.44 + 0.15 * sin(t * 0.42)) * qf.yz;
  qf.xz = rot2(0.32 * cos(t * 0.36) + 0.22) * qf.xz;
  float cubeFrame = sdBoxFrame(qf, vec3(0.54, 0.54, 0.54), 0.066) - 0.016;

  /* Dynamic morphing liquid-glass heart suspended freely inside the open cube frame */
  vec3 c0 = vec3(0.05 * sin(t * 0.8), 0.04 * cos(t * 0.7), 0.06 * sin(t * 1.0));
  float core = length(qs - c0) - 0.25;
  vec3 c1 = vec3( 0.25 * cos(t * 1.0),  0.20 * sin(t * 0.9),  0.17 * sin(t * 1.2));
  vec3 c2 = vec3(-0.23 * sin(t * 0.85), -0.18 * cos(t * 1.1),  0.20 * cos(t * 0.95));
  vec3 c3 = vec3( 0.15 * sin(t * 1.3), -0.22 * sin(t * 0.8), -0.20 * cos(t * 1.05));
  core = smin(core, length(qs - c1) - 0.16, 0.26);
  core = smin(core, length(qs - c2) - 0.15, 0.26);
  core = smin(core, length(qs - c3) - 0.14, 0.24);
  core += (vnoise(qs * 3.4 + vec3(0.0, t * 0.4, -t * 0.3)) - 0.5) * 0.014;

  float dAll = min(cubeFrame, core);
  return vec2(dAll * scl, (dAll == core) ? 0.0 : 2.0);
}

/* Returns vec3(dist, matKind, sculptIndex) */
vec3 mapScene(vec3 p) {
  float dBest = 1e4;
  float mBest = 0.0;
  float iBest = -1.0;
  for (int i = 0; i < 11; i++) {
    if (i >= uNumSculpt) break;
    float bs = length(p - uSculptPos[i].xyz) - uSculptCfg[i].w;
    if (bs < dBest) {
      vec3 lp;
      vec2 res = evalSculpture(p, i, lp);
      if (res.x < dBest) {
        dBest = res.x;
        mBest = res.y;
        iBest = float(i);
      }
    }
  }
  return vec3(dBest, mBest, iBest);
}

float mapDist(vec3 p) { return mapScene(p).x; }

vec3 calcNormal(vec3 p) {
  vec2 e = vec2(0.0026, -0.0026);
  return normalize(
    e.xyy * mapDist(p + e.xyy) +
    e.yyx * mapDist(p + e.yyx) +
    e.yxy * mapDist(p + e.yxy) +
    e.xxx * mapDist(p + e.xxx)
  );
}

vec2 projUV(vec3 X, out float zcam) {
  vec3 q = transpose(uCam) * (X - uRo);
  zcam = q.z;
  vec2 px = uF * q.xy / max(q.z, 0.05) + 0.5 * uRes;
  return px / uRes;
}

vec3 sampleBackdrop(vec3 ro, vec3 rd, float blurPx, out vec2 uvOut) {
  float tf = (rd.z < -0.012) ? (-ro.z / rd.z) : 1e4;
  if (tf < 0.05 || tf > 26.0) tf = 1e4;
  vec3 X = ro + rd * min(tf, uProxyDist);
  float zc;
  vec2 uv = projUV(X, zc);
  if (zc < 0.1) uv = clamp(vec2(1.0 - uv.x, uv.y), 0.01, 0.99);
  uv = abs(fract(uv * 0.5 + 0.5) * 2.0 - 1.0);
  uv = clamp(uv, vec2(0.008), vec2(0.992));
  uvOut = uv;
  if (blurPx <= 0.25) return texture(uPlate, uv).rgb;
  vec2 tx = blurPx / uRes;
  vec3 acc = texture(uPlate, uv).rgb * 0.28;
  acc += texture(uPlate, clamp(uv + vec2( tx.x,  tx.y * 0.5), 0.008, 0.992)).rgb * 0.18;
  acc += texture(uPlate, clamp(uv + vec2(-tx.x,  tx.y * 0.5), 0.008, 0.992)).rgb * 0.18;
  acc += texture(uPlate, clamp(uv + vec2( tx.x, -tx.y * 0.5), 0.008, 0.992)).rgb * 0.18;
  acc += texture(uPlate, clamp(uv + vec2(-tx.x, -tx.y * 0.5), 0.008, 0.992)).rgb * 0.18;
  return acc;
}

bool hitSphere(vec3 ro, vec3 rd, vec3 cen, float rad) {
  vec3 oc = cen - ro;
  float b = dot(oc, rd);
  float c = dot(oc, oc) - rad * rad;
  return (b > -rad) && (b * b - c > 0.0);
}

float warpGrid(vec2 uv) {
  vec2 g = uv * uRes / 48.0;
  vec2 f = abs(fract(g + 0.5) - 0.5) * 48.0;
  return 1.0 - smoothstep(0.8, 2.2, min(f.x, f.y));
}

void floorCaustic(vec3 Xf, out float caustic, out float shadow) {
  caustic = 0.0; shadow = 0.0;
  if (insideOccXY(Xf.xy)) return;
  vec3 sun = normalize(uSun);
  float nearD = 1e4;
  for (int i = 0; i < 11; i++) {
    if (i >= uNumSculpt) break;
    vec3 cpos = uSculptPos[i].xyz;
    float cscl = max(uSculptPos[i].w, 0.55);
    vec3 sh = cpos - sun * (cpos.z / max(sun.z, 0.20));
    nearD = min(nearD, length(Xf.xy - sh.xy));
    float g = exp(-dot(Xf.xy - cpos.xy, Xf.xy - cpos.xy) / (0.62 * cscl * cscl)) * 0.16;
    shadow = max(shadow, g);
  }
  if (nearD > 2.65) return;

  float t = 0.05;
  bool hit = false;
  float minD = 1e4;
  float hitMat = 0.0;
  int hitIdx = 0;
  for (int i = 0; i < 48; i++) {
    vec3 p = Xf + sun * t;
    if (p.z > 3.8) break;
    vec3 h = mapScene(p);
    minD = min(minD, h.x);
    if (h.x < 0.012) { hit = true; hitMat = h.y; hitIdx = int(h.z + 0.5); break; }
    t += max(h.x * 0.85, 0.026);
  }
  shadow = max(shadow, (1.0 - smoothstep(0.0, 0.42, minD)) * 0.34);

  if (hit) {
    for (int k = 0; k < 3; k++) t += mapDist(Xf + sun * t);
    vec3 P = Xf + sun * t;
    vec3 N = calcNormal(P);
    if (abs(hitMat - 1.0) < 0.3) {
      vec3 lp; evalSculpture(P, hitIdx, lp);
      float bands = pow(0.5 + 0.5 * cos(lp.x * uFluteFreq * 2.0), 2.8);
      caustic = (0.24 + 1.35 * bands) * smoothstep(0.0, 0.14, 0.48 - abs(lp.x));
      shadow = mix(0.44, 0.05, clamp(caustic * 0.55, 0.0, 1.0));
    } else {
      float nds = max(abs(dot(N, sun)), 0.0);
      vec3 r0 = refract(-sun, N, 1.0 / uIor);
      vec2 focusPt = P.xy + r0.xy * (-P.z / min(r0.z, -0.15));
      float n1 = vnoise(vec3(focusPt * 2.0, iTime * 0.36));
      float n2 = vnoise(vec3(focusPt * 3.3 + vec2(2.4, -1.3), -iTime * 0.44));
      float fil = pow(clamp(1.0 - abs(n1 - 0.5) * 2.2, 0.0, 1.0), 2.0) * 0.72
                + pow(clamp(1.0 - abs(n2 - 0.5) * 2.4, 0.0, 1.0), 2.2) * 0.42;
      float softEnv = pow(smoothstep(0.12, 0.65, nds), 1.30);
      caustic = (pow(nds, 2.1) * 0.90 + fil * 0.82) * softEnv;
      shadow = mix(0.32, 0.03, clamp(caustic * 0.65, 0.0, 1.0));
    }
  }
}

void traceGlass(vec3 ro, vec3 rd, vec2 uv0,
                out vec3 cComp, out vec3 cNorm, out vec3 cRefr, out vec3 cCaus,
                out float outDist, out float outOcc, out float outCgAlpha) {
  vec3 plateCol = texture(uPlate, uv0).rgb;
  outOcc = hitOccluders(ro, rd);
  outDist = outOcc;
  outCgAlpha = 0.0;

  cNorm = plateCol;
  float gl0 = (uExportPasses != 0) ? warpGrid(uv0) : 0.0;
  cRefr = mix(plateCol, vec3(0.95, 0.98, 1.0), gl0 * 0.28);
  cCaus = plateCol;

  float tFloor = (rd.z < -0.004) ? (-ro.z / rd.z) : 1e4;
  if (tFloor < 0.05 || tFloor > 24.0) tFloor = 1e4;

  bool nearAny = false;
  for (int i = 0; i < 11; i++) {
    if (i >= uNumSculpt) break;
    if (hitSphere(ro, rd, uSculptPos[i].xyz, uSculptCfg[i].w)) { nearAny = true; break; }
  }

  float tHit = 1e4;
  float matKind = -1.0;
  int hitIdx = -1;
  if (nearAny) {
    float t = 0.04;
    float tMax = min(min(tFloor + 0.05, outOcc), 22.0);
    for (int i = 0; i < 104; i++) {
      if (t > tMax) break;
      vec3 p = ro + rd * t;
      vec3 h = mapScene(p);
      if (h.x < 0.0018) { tHit = t; matKind = h.y; hitIdx = int(h.z + 0.5); break; }
      t += max(h.x * 0.88, 0.0038);
    }
  }

  vec3 bgCol = plateCol;
  if (tFloor < min(outOcc, 22.0)) {
    vec3 Xf = ro + rd * tFloor;
    float caust, shad;
    floorCaustic(Xf, caust, shad);
    if (caust > 0.001 || shad > 0.001) {
      float luma = dot(plateCol, vec3(0.299, 0.587, 0.114));
      float floorMask = smoothstep(uFloorLuma.x, uFloorLuma.y, luma) * (1.0 - smoothstep(14.0, 21.0, tFloor));
      bgCol = bgCol * (1.0 - shad * uShadowGain * floorMask);
      bgCol += uSunlitCol * (caust * uCausticGain * 0.30 * floorMask) * (0.42 + 0.58 * bgCol);
      if (uExportPasses != 0) {
        vec3 cHi = plateCol * (1.0 - shad * 0.45 * floorMask) + vec3(1.0, 0.82, 0.32) * (caust * 0.88 * floorMask);
        cCaus = clamp(cHi, 0.0, 1.0);
      }
    }
  }

  /* On caustic diagnostic pass, subtly show cyan architectural pillar holdout where a pillar masks a sculpture behind it */
  if (uExportPasses != 0 && (matKind < -0.5 || tHit >= outOcc) && outOcc < min(tFloor, 18.0) && nearAny) {
    /* Check briefly if a sculpture actually sits behind this occluder pixel */
    float tb = outOcc + 0.08;
    bool behindOcc = false;
    for (int s = 0; s < 40; s++) {
      if (tb > min(tFloor, 18.0)) break;
      float d = mapDist(ro + rd * tb);
      if (d < 0.008) { behindOcc = true; break; }
      tb += max(d * 0.9, 0.025);
    }
    if (behindOcc) {
      cCaus = mix(plateCol, vec3(0.18, 0.92, 0.96), 0.34);
    }
  }

  if (matKind < -0.5 || tHit >= outOcc) {
    cComp = bgCol;
    return;
  }
  outDist = tHit;
  outCgAlpha = 1.0;

  vec3 p1 = ro + rd * tHit;
  vec3 n1 = calcNormal(p1);
  bool isFlute = (abs(matKind - 1.0) < 0.3);
  bool isCrystal = (matKind > 1.5);

  float fluteCrest = 0.0;
  if (isFlute) {
    vec3 lp; evalSculpture(p1, hitIdx, lp);
    float yaw = uSculptCfg[hitIdx].y;
    float slope = cos(lp.x * uFluteFreq) * uFluteAmp;
    fluteCrest = pow(abs(sin(lp.x * uFluteFreq)), 4.0);
    n1 = normalize(n1 + vec3(cos(yaw), sin(yaw), 0.0) * slope * 0.9);
  }

  cNorm = 0.5 + 0.5 * n1;

  float localIor = isCrystal ? (uIor + 0.06) : uIor;
  float localDisp = isCrystal ? (uDisp * 1.45) : uDisp;

  vec3 rdInG = refract(rd, n1, 1.0 / localIor);
  if (length(rdInG) < 0.001) rdInG = reflect(rd, n1);

  float thick = 0.06;
  vec3 p2 = p1 + rdInG * 0.04;
  for (int i = 0; i < 36; i++) {
    float dIn = -mapDist(p2);
    if (dIn < 0.003) break;
    float stepD = max(dIn * 0.85, 0.012);
    p2 += rdInG * stepD;
    thick += stepD;
    if (thick > 2.0) break;
  }
  vec3 n2 = -calcNormal(p2);

  float iorR = localIor - localDisp, iorG = localIor, iorB = localIor + localDisp;
  vec3 rdOutG = refract(rdInG, n2, iorG);
  if (length(rdOutG) < 0.001) rdOutG = normalize(mix(rdInG, reflect(rdInG, n2), 0.35));
  vec3 bend = rdOutG - rd;
  vec3 rdOutR = normalize(rd + bend * (iorR - 1.0) / max(iorG - 1.0, 0.05));
  vec3 rdOutB = normalize(rd + bend * (iorB - 1.0) / max(iorG - 1.0, 0.05));

  float blurPx = isFlute ? (3.2 + 2.2 * fluteCrest) : (isCrystal ? 0.30 : 0.45);
  vec2 uvR, uvG, uvB, uvEnv;
  vec3 refrCol = vec3(
    sampleBackdrop(p2, rdOutR, blurPx, uvR).r,
    sampleBackdrop(p2, rdOutG, blurPx, uvG).g,
    sampleBackdrop(p2, rdOutB, blurPx, uvB).b
  );

  if (uExportPasses != 0) {
    float glR = warpGrid(uvG);
    float bendMag = clamp(length(uvG - uv0) * 9.0, 0.0, 1.0);
    cRefr = mix(refrCol * 0.88, vec3(0.45, 0.92, 1.0), bendMag * 0.25);
    cRefr = mix(cRefr, vec3(1.0, 1.0, 1.0), glR * 0.82);
  }

  refrCol = refrCol * exp(-thick * uTint * (isFlute ? 1.30 : 0.85)) * 1.04;

  float NdotV = max(dot(-rd, n1), 0.0);
  float fresnel = 0.035 + 0.965 * pow(1.0 - NdotV, isCrystal ? 4.2 : 4.6);
  vec3 reflDir = reflect(rd, n1);
  vec3 envRefl = sampleBackdrop(p1 + n1 * 0.01, reflDir, isFlute ? 6.5 : 2.8, uvEnv);
  float skyUp = smoothstep(-0.1, 0.85, reflDir.z);
  envRefl = mix(envRefl * 0.96, envRefl * 1.14 + vec3(0.18, 0.19, 0.21), skyUp * 0.54);

  vec3 sunDir = normalize(uSun);
  vec3 H1 = normalize(sunDir - rd);
  float specSun = pow(max(dot(n1, H1), 0.0), isFlute ? 115.0 : (isCrystal ? 260.0 : 220.0)) * (isFlute ? 0.95 : (isCrystal ? 1.75 : 1.50));
  float specBroad = pow(max(dot(n1, H1), 0.0), 24.0) * 0.15;
  vec3 H2 = normalize(normalize(vec3(0.04, 0.12, 0.98)) - rd);
  float specCeil = pow(max(dot(n1, H2), 0.0), 120.0) * 0.46;
  float rimGlow = pow(1.0 - NdotV, 3.1) * (0.34 + 0.28 * skyUp);

  float reflMix = clamp(fresnel * (isFlute ? 0.44 : (isCrystal ? 0.64 : 0.58)), 0.0, 0.80);
  vec3 col = mix(refrCol, envRefl, reflMix);
  col += vec3(1.0, 0.97, 0.91) * (specSun + specBroad);
  col += vec3(0.95, 0.98, 1.0) * (specCeil + rimGlow);
  if (isFlute) {
    col += vec3(0.96, 0.98, 1.0) * fluteCrest * 0.085 * (0.35 + 0.65 * fresnel);
  }
  cComp = col;
  if (uExportPasses != 0) {
    cCaus = mix(col, vec3(1.0, 0.95, 0.85), rimGlow * 0.32);
  }
}

void main() {
  vec2 px0 = vec2(gl_FragCoord.x, uRes.y - gl_FragCoord.y);
  vec3 acc = vec3(0.0), aNorm = vec3(0.0), aRefr = vec3(0.0), aCaus = vec3(0.0);
  float minD = 1e4;
  float minOcc = 1e4;
  float sumA = 0.0;
  float sumInvD = 0.0;
  vec2 offs[4] = vec2[4](vec2(-0.25, -0.25), vec2(0.25, -0.25), vec2(-0.25, 0.25), vec2(0.25, 0.25));
  for (int k = 0; k < 4; k++) {
    vec2 px = px0 + offs[k];
    vec3 rd = normalize(uCam * vec3((px - 0.5 * uRes) / uF, 1.0));
    vec3 c0, c1, c2, c3;
    float d, occ, cga;
    traceGlass(uRo, rd, px / uRes, c0, c1, c2, c3, d, occ, cga);
    acc += c0;
    if (uExportPasses != 0) {
      aNorm += c1; aRefr += c2; aCaus += c3;
    }
    minD = min(minD, d);
    minOcc = min(minOcc, occ);
    sumA += cga;
    if (cga > 0.5) sumInvD += 1.0 / max(d, 0.20);
  }
  oComp    = vec4(clamp(acc * 0.25, 0.0, 1.0), 1.0);
  oDepth   = vec4(minD, minOcc, 0.0, 1.0);
  oMeta    = vec4(sumA * 0.25, sumInvD * 0.25, 1.0 / max(minOcc, 0.20), 1.0);
  oNormal  = vec4(clamp(aNorm * 0.25, 0.0, 1.0), 1.0);
  oRefract = vec4(clamp(aRefr * 0.25, 0.0, 1.0), 1.0);
  oCaustic = vec4(clamp(aCaus * 0.25, 0.0, 1.0), 1.0);
}
"""

# Pass 2: Restrained Subtle Accent Droplets (~25 delicate micro-beads strictly tucked near organic liquid sculptures!)
BEAD_VS = """#version 330
in vec2 inQuad;
in vec4 inPosRad;
in vec4 inAxisAsp;
in vec4 inSeed;
uniform vec2 uRes;
uniform float uF;
uniform vec3 uRo;
uniform mat3 uCam;

out vec3 vCenter;
out float vRad;
out vec3 vAxis;
out float vAsp;
out vec4 vSeed;

void main() {
  vCenter = inPosRad.xyz;
  vRad = inPosRad.w;
  vAxis = normalize(inAxisAsp.xyz);
  vAsp = inAxisAsp.w;
  vSeed = inSeed;

  vec3 q = transpose(uCam) * (vCenter - uRo);
  float boundR = vRad * max(vAsp, 1.0) * 1.55;
  if (q.z <= 0.08 + boundR * 0.25) {
    gl_Position = vec4(2.0, 2.0, 2.0, 1.0);
    return;
  }
  vec2 ctrPx = uF * q.xy / q.z + 0.5 * uRes;
  float pxR = clamp(uF * boundR / max(q.z - boundR * 0.75, 0.06) + 4.0, 4.0, 900.0);
  vec2 px = ctrPx + inQuad * pxR;
  vec2 ndc = vec2(px.x / uRes.x * 2.0 - 1.0, (1.0 - px.y / uRes.y) * 2.0 - 1.0);
  gl_Position = vec4(ndc, 0.0, 1.0);
}
"""

BEAD_FS = """#version 330
in vec3 vCenter;
in float vRad;
in vec3 vAxis;
in float vAsp;
in vec4 vSeed;

uniform sampler2D uBase;
uniform sampler2D uDepth;
uniform vec2 uRes;
uniform float uF, iTime, uIor, uDisp, uProxyDist;
uniform vec3 uRo, uSun, uTint;
uniform mat3 uCam;

layout(location=0) out vec4 oCol;
layout(location=1) out vec4 oMeta;

float sdBead(vec3 p) {
  vec3 q = p - vCenter;
  float along = dot(q, vAxis);
  vec3 perp = q - vAxis * along;
  float taper = 1.0 + 0.16 * clamp(along / max(vRad * vAsp, 1e-4), -1.0, 1.0);
  vec3 qs = perp / max(taper, 0.65) + vAxis * (along / vAsp);
  float d = (length(qs) - vRad) * min(1.0, vAsp * 0.72);
  float ph = vSeed.x * 19.3;
  float wob = sin(q.x / vRad * 2.5 + ph) * cos(q.y / vRad * 2.3 - ph * 0.7) * sin(q.z / vRad * 2.4 + ph * 1.3);
  return d + wob * vRad * vSeed.y;
}

vec3 beadNormal(vec3 p) {
  float h = max(vRad * 0.015, 0.0004);
  vec2 e = vec2(h, -h);
  return normalize(
    e.xyy * sdBead(p + e.xyy) +
    e.yyx * sdBead(p + e.yyx) +
    e.yxy * sdBead(p + e.yxy) +
    e.xxx * sdBead(p + e.xxx)
  );
}

vec3 sampleEnv(vec3 ro, vec3 rd) {
  float tf = (rd.z < -0.012) ? (-ro.z / rd.z) : 1e4;
  if (tf < 0.05 || tf > 24.0) tf = 1e4;
  vec3 X = ro + rd * min(tf, uProxyDist);
  vec3 q = transpose(uCam) * (X - uRo);
  vec2 uv = (uF * q.xy / max(q.z, 0.08) + 0.5 * uRes) / uRes;
  if (q.z < 0.08) uv = clamp(vec2(1.0 - uv.x, uv.y), 0.01, 0.99);
  uv = abs(fract(uv * 0.5 + 0.5) * 2.0 - 1.0);
  uv = clamp(uv, vec2(0.008), vec2(0.992));
  return texture(uBase, vec2(uv.x, 1.0 - uv.y)).rgb;
}

void main() {
  vec2 px0 = vec2(gl_FragCoord.x, uRes.y - gl_FragCoord.y);
  vec2 dep = texture(uDepth, gl_FragCoord.xy / uRes).rg;
  float heroD = dep.r;
  float occD  = dep.g;

  vec3 acc = vec3(0.0);
  float cov = 0.0;
  float minT = 1e4;
  vec2 offs[4] = vec2[4](vec2(-0.25, -0.25), vec2(0.25, -0.25), vec2(-0.25, 0.25), vec2(0.25, 0.25));
  float boundR = vRad * max(vAsp, 1.0) * 1.60;

  for (int k = 0; k < 4; k++) {
    vec2 px = px0 + offs[k];
    vec3 rd = normalize(uCam * vec3((px - 0.5 * uRes) / uF, 1.0));
    vec3 oc = vCenter - uRo;
    float b = dot(oc, rd);
    float c = dot(oc, oc) - boundR * boundR;
    float disc = b * b - c;
    if (disc < 0.0) continue;
    float sq = sqrt(disc);
    float tStart = max(b - sq, 0.04);
    float tEnd = min(b + sq, heroD);
    if (tStart >= tEnd) continue;

    float t = tStart;
    float tHit = -1.0;
    for (int s = 0; s < 36; s++) {
      float d = sdBead(uRo + rd * t);
      if (d < max(vRad * 0.008, 0.0003)) { tHit = t; break; }
      t += d * 0.90;
      if (t > tEnd) break;
    }
    if (tHit < 0.0) continue;
    minT = min(minT, tHit);

    vec3 p1 = uRo + rd * tHit;
    vec3 n1 = beadNormal(p1);
    vec3 rdIn = refract(rd, n1, 1.0 / uIor);
    if (length(rdIn) < 0.001) rdIn = reflect(rd, n1);

    float thick = vRad * 0.35;
    vec3 p2 = p1 + rdIn * max(vRad * 0.25, 0.002);
    for (int s = 0; s < 16; s++) {
      float dIn = -sdBead(p2);
      if (dIn < max(vRad * 0.015, 0.0004)) break;
      float st = max(dIn * 0.90, vRad * 0.08);
      p2 += rdIn * st;
      thick += st;
      if (thick > vRad * 4.5) break;
    }
    vec3 n2 = -beadNormal(p2);
    vec3 rdOutG = refract(rdIn, n2, uIor);
    if (length(rdOutG) < 0.001) rdOutG = normalize(mix(rdIn, reflect(rdIn, n2), 0.35));

    vec3 bend = rdOutG - rd;
    vec3 rdOutR = normalize(rd + bend * ((uIor - uDisp) - 1.0) / max(uIor - 1.0, 0.05));
    vec3 rdOutB = normalize(rd + bend * ((uIor + uDisp) - 1.0) / max(uIor - 1.0, 0.05));

    vec3 refrCol = vec3(
      sampleEnv(p2, rdOutR).r,
      sampleEnv(p2, rdOutG).g,
      sampleEnv(p2, rdOutB).b
    );
    refrCol *= exp(-thick * uTint * 0.55) * 1.06;

    float ndv = max(dot(-rd, n1), 0.0);
    float fresnel = 0.04 + 0.96 * pow(1.0 - ndv, 4.0);
    vec3 reflDir = reflect(rd, n1);
    vec3 envRefl = sampleEnv(p1 + n1 * 0.005, reflDir);
    float skyUp = smoothstep(-0.15, 0.85, reflDir.z);
    envRefl = mix(envRefl * 1.02 + vec3(0.05), envRefl * 1.18 + vec3(0.24, 0.26, 0.28), skyUp * 0.56);

    vec3 H1 = normalize(normalize(uSun) - rd);
    float spec1 = pow(max(dot(n1, H1), 0.0), 210.0) * 2.05 + pow(max(dot(n1, H1), 0.0), 28.0) * 0.16;
    vec3 H2 = normalize(normalize(vec3(0.05, 0.15, 0.98)) - rd);
    float spec2 = pow(max(dot(n1, H2), 0.0), 130.0) * 0.68;
    float rim = pow(1.0 - ndv, 2.8) * (0.38 + 0.28 * skyUp);

    vec3 col = mix(refrCol, envRefl, clamp(fresnel * 0.58, 0.0, 0.80));
    col += vec3(1.0, 0.98, 0.93) * spec1 + vec3(0.95, 0.98, 1.0) * (spec2 + rim);
    acc += col;
    cov += 1.0;
  }

  if (cov < 0.5) discard;
  float a = cov * 0.25;
  oCol = vec4(clamp(acc / cov, 0.0, 1.0), a);
  oMeta = vec4(1.0, 1.0 / max(minT, 0.20), 1.0 / max(occD, 0.20), a);
}
"""

# Pass 3: Camera Optical Finish (Depth-Aware Bokeh + OLPF + Halation + Edge CA + Sensor Grain)
POST_FS = """#version 330
uniform sampler2D uScene;
uniform sampler2D uMeta;
uniform vec2 uRes;
uniform float uFrame;
uniform float uFocusDist, uNearBlur, uFarBlur, uBaseSoft, uMaxCoc;
uniform float uGrainAmt, uBlackFloor;

out vec4 oCol;

float hash12(vec2 p) {
  vec3 p3 = fract(vec3(p.xyx) * 0.1031);
  p3 += dot(p3, p3.yzx + 33.33);
  return fract((p3.x + p3.y) * p3.z);
}

float calcCoc(float invD) {
  float invF = 1.0 / max(uFocusDist, 0.5);
  float nearC = uNearBlur * max(0.0, invD - invF);
  float farC  = uFarBlur  * max(0.0, invF - invD);
  return clamp(uBaseSoft + nearC + farC, uBaseSoft, uMaxCoc);
}

void main() {
  vec2 uv = gl_FragCoord.xy / uRes;
  vec2 pxSize = 1.0 / uRes;
  vec3 col0 = texture(uScene, uv).rgb;
  vec3 m0 = texture(uMeta, uv).rgb;
  float cgA0 = m0.r;
  float occInv0 = m0.b;
  float dInv0 = (cgA0 > 0.01) ? (m0.g / cgA0) : 0.0;
  float coc0 = (cgA0 > 0.01) ? calcCoc(dInv0) : 0.0;

  const float GA = 2.39996323;
  vec3 cgSum = vec3(0.0);
  float cgW = 0.0;
  vec3 halation = vec3(0.0);

  if (cgA0 > 0.005) {
    float rA = max(coc0, 1.35);
    cgSum += col0 * (0.28 * cgA0);
    cgW += 0.28 * cgA0;
    for (int k = 1; k <= 10; k++) {
      float f = sqrt(float(k) / 10.0);
      float ang = float(k) * GA;
      vec2 off = vec2(cos(ang), sin(ang)) * (f * rA);
      vec2 suv = clamp(uv + off * pxSize, vec2(0.001), vec2(0.999));
      vec3 mj = texture(uMeta, suv).rgb;
      float occInvJ = mj.b;
      if (occInvJ > dInv0 + 0.02) continue;
      float w = smoothstep(coc0 + 0.65, coc0 - 0.65, f * rA);
      if (w <= 0.001) continue;
      vec3 cj = texture(uScene, suv).rgb;
      float bgAllow = smoothstep(1.6, 5.5, coc0) * 0.55;
      float tw = w * mix(bgAllow, 1.0, mj.r);
      cgSum += cj * tw;
      cgW += tw;
    }
  }

  float fgHaloW = 0.0;
  vec3 fgHaloCol = vec3(0.0);
  for (int k = 0; k < 24; k++) {
    float f = sqrt((float(k) + 0.5) / 24.0);
    float ang = float(k) * GA + 0.5;
    float r = mix(1.0, uMaxCoc, f);
    vec2 off = vec2(cos(ang), sin(ang)) * r;
    vec2 suv = clamp(uv + off * pxSize, vec2(0.001), vec2(0.999));
    vec3 mj = texture(uMeta, suv).rgb;
    if (mj.r < 0.01) continue;
    float dInvJ = mj.g / mj.r;
    if (dInvJ <= occInv0 + 0.015) continue;
    float cocJ = calcCoc(dInvJ);
    vec3 cj = texture(uScene, suv).rgb;

    float lum = dot(cj, vec3(0.299, 0.587, 0.114));
    if (lum > 0.78) {
      float hw = exp(-r * r / 28.0) * (lum - 0.78) * mj.r;
      halation += cj * hw;
    }

    if (cocJ > 1.35 && r <= cocJ + 0.85 && (cgA0 < 0.02 || dInvJ > dInv0 + 0.08)) {
      float disk = smoothstep(cocJ + 0.85, cocJ - 0.85, r);
      float normArea = (uMaxCoc * uMaxCoc) / max(cocJ * cocJ * 24.0, 1.0);
      float w = disk * mj.r * min(normArea, 0.35);
      fgHaloCol += cj * w;
      fgHaloW += w;
    }
  }

  vec3 cgCol = (cgW > 0.001) ? (cgSum / cgW) : col0;
  if (cgA0 > 0.05) {
    vec2 radV = uv - 0.5;
    float r2 = dot(radV, radV);
    vec2 caOff = radV * (r2 * 1.85) * pxSize;
    float rS = texture(uScene, clamp(uv + caOff, 0.001, 0.999)).r;
    float bS = texture(uScene, clamp(uv - caOff, 0.001, 0.999)).b;
    float mR = texture(uMeta, clamp(uv + caOff, 0.001, 0.999)).r;
    float mB = texture(uMeta, clamp(uv - caOff, 0.001, 0.999)).r;
    cgCol.r = mix(cgCol.r, rS, 0.45 * mR);
    cgCol.b = mix(cgCol.b, bS, 0.45 * mB);
  }

  vec3 res = mix(col0, cgCol, clamp(cgA0, 0.0, 1.0));
  float fgAlpha = clamp(fgHaloW * 1.15, 0.0, 0.92);
  if (fgHaloW > 0.001) {
    vec3 haloRgb = fgHaloCol / fgHaloW;
    float blendFg = fgAlpha * (1.0 - cgA0 * 0.55);
    res = mix(res, haloRgb, blendFg);
  }

  float cgPresence = clamp(max(cgA0, fgAlpha), 0.0, 1.0);
  res += halation * vec3(1.02, 0.98, 0.93) * 0.055;

  if (cgPresence > 0.005) {
    vec3 graded = res * (1.0 - uBlackFloor) + vec3(uBlackFloor * 0.95, uBlackFloor * 0.98, uBlackFloor * 1.05);
    graded = graded / (1.0 + 0.055 * max(graded - 0.86, vec3(0.0)));

    vec2 p1 = gl_FragCoord.xy + vec2(uFrame * 17.3, uFrame * 31.7);
    vec2 p2 = floor(gl_FragCoord.xy * 0.5) + vec2(uFrame * 11.1, uFrame * 23.9);
    float g1 = (hash12(p1) + hash12(p1 + 7.13) - 1.0);
    float g2 = (hash12(p2) + hash12(p2 + 5.71) - 1.0);
    float gLuma = (g1 * 0.62 + g2 * 0.38);
    vec3 gChroma = vec3(hash12(p2 + 13.7) - 0.5, hash12(p2 + 29.3) - 0.5, hash12(p2 + 47.1) - 0.5) * 0.28;
    float lum = dot(graded, vec3(0.299, 0.587, 0.114));
    float grainEnv = (1.15 - 0.65 * smoothstep(0.55, 0.98, lum));
    graded += (vec3(gLuma) + gChroma) * (uGrainAmt * grainEnv);

    res = mix(res, graded, cgPresence);
  }

  oCol = vec4(clamp(res, 0.0, 1.0), 1.0);
}
"""


def label(im, s):
    im = im.copy()
    cv2.putText(im, s, (12, im.shape[0] - 14), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 4, cv2.LINE_AA)
    cv2.putText(im, s, (12, im.shape[0] - 14), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 1, cv2.LINE_AA)
    return im


def encode_composite(c, tag="sculptures_v2", out_dir="out_sculptures_v2", suffix="_Sculptures_v2"):
    out = c["dir"] / out_dir
    out.mkdir(exist_ok=True)
    nm = c["name"] + suffix
    r = f"{c['fps_num']}/{c['fps_den']}"
    ff = find_ffmpeg()
    tags = ["-pix_fmt", "yuv420p", "-color_primaries", "bt709", "-color_trc", "bt709", "-colorspace", "bt709", "-movflags", "+faststart"]
    x264 = ["-c:v", "libx264", "-profile:v", "high", "-level", "4.1", "-crf", "14", "-preset", "slow"]
    out_mp4 = out / f"{nm}_Composite.mp4"
    subprocess.run([ff, "-v", "error", "-y", "-framerate", r, "-i", str(work(c, f"comp_{tag}", "%04d.png"))] + x264 + tags + [str(out_mp4)], check=True)
    print(f"Encoded final composite -> {out_mp4} ({out_mp4.stat().st_size / 1e6:.2f} MB)", flush=True)
    return out_mp4


def build_restrained_accent_beads(cam_path, sculpt_pos, sculpt_cfg, occ_boxes, post_cfg):
    """Generates ONLY ~24 subtle micro-droplets tightly orbiting organic liquid pieces."""
    rng = np.random.default_rng(int(post_cfg.get("bead_seed", 20261124)))
    beads = []
    x_lo, x_hi = post_cfg.get("bead_x_clamp", [-2.15, 2.15])
    z_lo, z_hi = post_cfg.get("bead_z_clamp", [0.35, 2.55])

    def clears_cam_and_walls(pos, rad):
        d = np.linalg.norm(cam_path - pos[None, :], axis=1).min()
        if d <= (0.45 + rad * 2.0):
            return False
        pad = rad * 1.8 + 0.14
        for (x0, x1, y0, y1, z0, z1) in occ_boxes:
            if (x0 - pad <= pos[0] <= x1 + pad and
                y0 - pad <= pos[1] <= y1 + pad and
                z0 - pad <= pos[2] <= z1 + pad):
                return False
        return True

    organic_idxs = [i for i, cfg in enumerate(sculpt_cfg) if int(cfg[0]) in (0, 4, 7, 9, 10)]
    for idx in organic_idxs:
        anc = np.array(sculpt_pos[idx][:3], dtype=np.float64)
        spread = sculpt_pos[idx][3] * 0.92
        got = 0
        for _ in range(180):
            if got >= 5:
                break
            off = rng.normal(0.0, 1.0, 3)
            off /= max(np.linalg.norm(off), 1e-4)
            dist = rng.uniform(0.55, 1.12) * spread
            pos = anc + off * dist
            pos[2] = float(np.clip(pos[2], z_lo, z_hi))
            pos[0] = float(np.clip(pos[0], x_lo, x_hi))
            rad = float(np.exp(rng.uniform(math.log(0.014), math.log(0.040))))
            if not clears_cam_and_walls(pos, rad):
                continue
            axis = rng.normal(0, 1, 3)
            axis /= max(np.linalg.norm(axis), 1e-4)
            beads.append((pos, rad, axis, float(rng.uniform(1.05, 1.45)), float(rng.uniform(0, 10)),
                          float(rng.uniform(0.02, 0.06)), 0.0, float(rng.uniform(0, 6.28))))
            got += 1

    print(f"Diverse Sculpture Gallery: {len(sculpt_pos)} distinct hero sculptures + {len(beads)} restrained accent droplets + {len(occ_boxes)} 3D architectural occluders")
    return beads


def main():
    c = load(sys.argv[1] if len(sys.argv) > 1 and not sys.argv[1].startswith("--") else here / "piece_glass")
    S = scene(c)
    W, H, N, fps = c["W"], c["H"], c["N"], c["fps"]
    f = float(S["f"])
    g = c.get("glass", {})
    post_cfg = c.get("post", {})

    stills = [int(x) for x in sys.argv[sys.argv.index("--still") + 1].split(",")] if "--still" in sys.argv else None
    tag = sys.argv[sys.argv.index("--tag") + 1] if "--tag" in sys.argv else "sculptures_v2"
    out_dir = sys.argv[sys.argv.index("--out") + 1] if "--out" in sys.argv else f"out_{tag}"
    suffix = sys.argv[sys.argv.index("--suffix") + 1] if "--suffix" in sys.argv else f"_{tag}"
    export_passes = "--passes" in sys.argv

    cam_path = np.asarray(S["cam"], np.float64)
    sculpt_defs, occ_boxes = get_piece_sculptures_and_occluders(c)
    if len(sculpt_defs) > 11:
        sys.exit("max 11 sculptures supported")
    if len(occ_boxes) > 12:
        sys.exit("max 12 occ_boxes supported")

    sculpt_pos = [d[0] for d in sculpt_defs]
    sculpt_cfg = [d[1] for d in sculpt_defs]
    # Pad uniform arrays to fixed GLSL array sizes
    sculpt_pos_pad = sculpt_pos + [(0.0, 0.0, -99.0, 1.0)] * (11 - len(sculpt_pos))
    sculpt_cfg_pad = sculpt_cfg + [(0.0, 0.0, 0.0, 0.1)] * (11 - len(sculpt_cfg))
    occ_min_pad = [(b[0], b[2], b[4]) for b in occ_boxes] + [(0.0, 0.0, -99.0)] * (12 - len(occ_boxes))
    occ_max_pad = [(b[1], b[3], b[5]) for b in occ_boxes] + [(0.0, 0.0, -98.0)] * (12 - len(occ_boxes))

    for idx, (sp, sc) in enumerate(sculpt_defs):
        md = np.linalg.norm(cam_path - np.array(sp[:3]), axis=1).min()
        print(f"  sculpt[{idx:02d}] type={int(sc[0]):2d} pos=({sp[0]:+.2f},{sp[1]:+.2f},{sp[2]:+.2f}) scl={sp[3]:.2f} min_cam_dist={md:.2f}m (boundR={sc[3]:.2f}m)")

    beads = build_restrained_accent_beads(cam_path, sculpt_pos, sculpt_cfg, occ_boxes, post_cfg)

    ctx = moderngl.create_standalone_context(require=330)
    hero_prog = ctx.program(vertex_shader=VS, fragment_shader=HERO_FS)
    bead_prog = ctx.program(vertex_shader=BEAD_VS, fragment_shader=BEAD_FS)
    post_prog = ctx.program(vertex_shader=VS, fragment_shader=POST_FS)
    copy2_prog = ctx.program(
        vertex_shader=VS,
        fragment_shader="""#version 330
        uniform sampler2D uSrc0, uSrc1; uniform vec2 uRes;
        layout(location=0) out vec4 o0; layout(location=1) out vec4 o1;
        void main(){ vec2 uv = gl_FragCoord.xy / uRes; o0 = texture(uSrc0, uv); o1 = texture(uSrc1, uv); }""",
    )

    tri = ctx.buffer(np.array([-1, -1, 3, -1, -1, 3], "f4").tobytes())
    va_hero = ctx.vertex_array(hero_prog, [(tri, "2f", "p")])
    va_copy2 = ctx.vertex_array(copy2_prog, [(tri, "2f", "p")])
    va_post = ctx.vertex_array(post_prog, [(tri, "2f", "p")])

    quad_vbo = ctx.buffer(np.array([-1, -1, 1, -1, 1, 1, -1, -1, 1, 1, -1, 1], "f4").tobytes())
    inst_buf = ctx.buffer(reserve=max(len(beads), 1) * 12 * 4)
    va_bead = ctx.vertex_array(bead_prog, [
        (quad_vbo, "2f", "inQuad"),
        (inst_buf, "4f 4f 4f/i", "inPosRad", "inAxisAsp", "inSeed"),
    ])

    plate_t = ctx.texture((W, H), 3)
    plate_t.filter = (moderngl.LINEAR, moderngl.LINEAR)
    plate_t.repeat_x = plate_t.repeat_y = False

    t_hero = ctx.texture((W, H), 4, dtype="f4")
    t_hero.filter = (moderngl.LINEAR, moderngl.LINEAR)
    t_hero.repeat_x = t_hero.repeat_y = False
    t_depth = ctx.texture((W, H), 4, dtype="f4")
    t_depth.filter = (moderngl.NEAREST, moderngl.NEAREST)
    t_meta0 = ctx.texture((W, H), 4, dtype="f4")
    t_meta0.filter = (moderngl.NEAREST, moderngl.NEAREST)
    t_norm = ctx.texture((W, H), 4, dtype="f4")
    t_refr = ctx.texture((W, H), 4, dtype="f4")
    t_caus = ctx.texture((W, H), 4, dtype="f4")
    fbo_hero = ctx.framebuffer([t_hero, t_depth, t_meta0, t_norm, t_refr, t_caus])

    t_scene = ctx.texture((W, H), 4, dtype="f4")
    t_scene.filter = (moderngl.LINEAR, moderngl.LINEAR)
    t_scene.repeat_x = t_scene.repeat_y = False
    t_meta = ctx.texture((W, H), 4, dtype="f4")
    t_meta.filter = (moderngl.NEAREST, moderngl.NEAREST)
    fbo_scene = ctx.framebuffer([t_scene, t_meta])

    t_final = ctx.texture((W, H), 4, dtype="f4")
    fbo_final = ctx.framebuffer([t_final])

    copy2_prog["uRes"].value = (float(W), float(H))

    hero_prog["uRes"].value = (float(W), float(H))
    hero_prog["uF"].value = f
    hero_prog["uSun"].value = tuple(map(float, S["sun"]))
    hero_prog["uSculptPos"].value = sculpt_pos_pad
    hero_prog["uSculptCfg"].value = sculpt_cfg_pad
    hero_prog["uNumSculpt"].value = len(sculpt_pos)
    hero_prog["uOccMin"].value = occ_min_pad
    hero_prog["uOccMax"].value = occ_max_pad
    hero_prog["uNumOcc"].value = len(occ_boxes)
    hero_prog["uIor"].value = float(g.get("ior", 1.46))
    hero_prog["uDisp"].value = float(g.get("dispersion", 0.019))
    hero_prog["uTint"].value = tuple(map(float, g.get("tint", [0.070, 0.024, 0.018])))
    hero_prog["uSunlitCol"].value = tuple(map(float, g.get("sunlit_col", [1.0, 0.94, 0.82])))
    hero_prog["uFloorLuma"].value = tuple(map(float, g.get("floor_luma", [0.08, 0.24])))
    hero_prog["uCausticGain"].value = float(g.get("caustic_gain", 1.08))
    hero_prog["uShadowGain"].value = float(g.get("shadow_gain", 0.34))
    hero_prog["uFluteFreq"].value = float(g.get("flute_freq", 56.0))
    hero_prog["uFluteAmp"].value = float(g.get("flute_amp", 0.19))
    hero_prog["uMetaK"].value = float(g.get("metaball_k", 0.32))
    hero_prog["uProxyDist"].value = float(g.get("proxy_dist", 7.0))
    hero_prog["uExportPasses"].value = 1 if (export_passes or stills) else 0

    bead_prog["uRes"].value = (float(W), float(H))
    bead_prog["uF"].value = f
    bead_prog["uSun"].value = tuple(map(float, S["sun"]))
    bead_prog["uIor"].value = float(g.get("ior", 1.46))
    bead_prog["uDisp"].value = float(g.get("dispersion", 0.018))
    bead_prog["uTint"].value = tuple(map(float, g.get("tint", [0.070, 0.024, 0.018])))
    bead_prog["uProxyDist"].value = float(g.get("proxy_dist", 7.0))

    post_prog["uRes"].value = (float(W), float(H))
    post_prog["uFocusDist"].value = float(post_cfg.get("focus_dist", 4.2))
    post_prog["uNearBlur"].value = float(post_cfg.get("near_blur", 4.4))
    post_prog["uFarBlur"].value = float(post_cfg.get("far_blur", 5.5))
    post_prog["uBaseSoft"].value = float(post_cfg.get("base_soft", 0.72))
    post_prog["uMaxCoc"].value = float(post_cfg.get("max_coc", 10.5))
    post_prog["uGrainAmt"].value = float(post_cfg.get("grain", 0.012))
    post_prog["uBlackFloor"].value = float(post_cfg.get("black_floor", 0.016))

    def tex_to_bgr(tex):
        arr = np.frombuffer(tex.read(), "f4").reshape(H, W, 4)[::-1]
        return (np.clip(arr[..., :3], 0, 1) * 255 + 0.5).astype(np.uint8)[:, :, ::-1]

    def render_one(i, want_passes=False):
        pl = frame(c, i)
        t_sec = i / fps
        ro = np.asarray(S["cam"][i], np.float64)
        R = np.asarray(S["rot"][i], np.float64)
        Rt = tuple(R.astype(np.float32).T.flatten())

        plate_t.write(pl[:, :, ::-1].tobytes())
        plate_t.use(0)
        hero_prog["uPlate"].value = 0
        hero_prog["uRo"].value = tuple(map(float, ro))
        hero_prog["uCam"].value = Rt
        hero_prog["iTime"].value = t_sec
        hero_prog["uExportPasses"].value = 1 if want_passes else 0

        fbo_hero.use()
        ctx.disable(moderngl.BLEND)
        va_hero.render(moderngl.TRIANGLES)

        fbo_scene.use()
        t_hero.use(0)
        t_meta0.use(1)
        copy2_prog["uSrc0"].value = 0
        copy2_prog["uSrc1"].value = 1
        va_copy2.render(moderngl.TRIANGLES)

        rows_b = []
        for pos, rad, axis, asp, s_w, w_amp, lobe, ph in beads:
            p_now = pos + np.array([0.022 * math.sin(t_sec * 0.65 + ph),
                                    0.022 * math.cos(t_sec * 0.55 + ph * 1.2),
                                    0.032 * math.sin(t_sec * 0.80 + ph * 0.8)])
            z_cam = float(((p_now - ro) @ R)[2])
            if z_cam > 0.08:
                rows_b.append((z_cam, [*p_now, rad, *axis, asp, s_w, w_amp, lobe, ph]))
        rows_b.sort(key=lambda x: -x[0])
        if rows_b:
            data = np.array([r[1] for r in rows_b], dtype=np.float32)
            inst_buf.write(data.tobytes())
            t_hero.use(0)
            t_depth.use(1)
            bead_prog["uBase"].value = 0
            bead_prog["uDepth"].value = 1
            bead_prog["uRo"].value = tuple(map(float, ro))
            bead_prog["uCam"].value = Rt
            if "iTime" in bead_prog:
                bead_prog["iTime"].value = t_sec
            ctx.enable(moderngl.BLEND)
            ctx.blend_func = (moderngl.SRC_ALPHA, moderngl.ONE_MINUS_SRC_ALPHA)
            va_bead.render(moderngl.TRIANGLES, instances=len(rows_b))
            ctx.disable(moderngl.BLEND)

        fbo_final.use()
        t_scene.use(0)
        t_meta.use(1)
        post_prog["uScene"].value = 0
        post_prog["uMeta"].value = 1
        post_prog["uFrame"].value = float(i)
        va_post.render(moderngl.TRIANGLES)

        ctx.finish()
        img = tex_to_bgr(t_final)
        if want_passes:
            return pl, img, tex_to_bgr(t_norm), tex_to_bgr(t_refr), tex_to_bgr(t_caus)
        return pl, img

    if stills:
        rows = []
        hero = None
        hero_end = None
        for i in stills:
            pl, comp, pn, pr, pc = render_one(i, want_passes=True)
            if i == stills[len(stills) // 2]:
                hero = comp
            if i == stills[-1]:
                hero_end = comp
            t_im = lambda im, s: label(cv2.resize(im, (640, 360)), s)
            rows.append(np.hstack([t_im(pl, f"plate {i}"), t_im(comp, f"comp {i}"), t_im(pc, f"caustic+occ {i}")]))
        out_p = work(c, f"still_{tag}_sheet.jpg")
        cv2.imwrite(str(out_p), np.vstack(rows), [int(cv2.IMWRITE_JPEG_QUALITY), 92])
        if hero is not None:
            cv2.imwrite(str(work(c, f"still_{tag}_hero.jpg")), hero, [int(cv2.IMWRITE_JPEG_QUALITY), 94])
        if hero_end is not None:
            cv2.imwrite(str(work(c, f"still_{tag}_end.jpg")), hero_end, [int(cv2.IMWRITE_JPEG_QUALITY), 94])
        print("stills ->", out_p)
        return

    comp_dir = work(c, f"comp_{tag}")
    shutil.rmtree(comp_dir, ignore_errors=True)
    comp_dir.mkdir(parents=True)

    pass_dirs = {}
    if export_passes:
        pbase = work(c, f"pass_{tag}")
        shutil.rmtree(pbase, ignore_errors=True)
        for nm in ("normal", "refract", "caustic"):
            d = pbase / nm
            d.mkdir(parents=True)
            pass_dirs[nm] = d

    ts = time.time()
    for i in range(N):
        if export_passes:
            _, comp, pn, pr, pc = render_one(i, want_passes=True)
            cv2.imwrite(str(pass_dirs["normal"] / f"{i:04d}.jpg"), pn, [int(cv2.IMWRITE_JPEG_QUALITY), 92])
            cv2.imwrite(str(pass_dirs["refract"] / f"{i:04d}.jpg"), pr, [int(cv2.IMWRITE_JPEG_QUALITY), 92])
            cv2.imwrite(str(pass_dirs["caustic"] / f"{i:04d}.jpg"), pc, [int(cv2.IMWRITE_JPEG_QUALITY), 92])
        else:
            _, comp = render_one(i, want_passes=False)
        cv2.imwrite(str(comp_dir / f"{i:04d}.png"), comp, [int(cv2.IMWRITE_PNG_COMPRESSION), 1])
        if i % 45 == 0 or i == N - 1:
            print(f"{tag} frame {i}/{N} ({time.time() - ts:.1f}s)", flush=True)
    encode_composite(c, tag=tag, out_dir=out_dir, suffix=suffix)


if __name__ == "__main__":
    main()
