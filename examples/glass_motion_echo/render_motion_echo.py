"""#3 Frozen Glass Motion Clone — 3D Volumetric Liquid-Glass Human Echo Statues + Shedding Crystal Beads (ModernGL, OpenGL 3.3).

    python render_motion_echo.py [piece_dir] --still 20,58,98,132,158   quick look -> work/still_echo_sheet.jpg
    python render_motion_echo.py [piece_dir] --out out                  full render -> out/*_Composite.mp4, *_Compare_1080x1350.mp4

Zero text/HUD overlays — 100% pure optical 3D sculpture VFX:
- Uses MediaPipe 33-joint 3D lifted skeleton (limbs.npz) + soft person segmentation matte (mask/) + COLMAP 3D camera.
- At 10 rhythmic choreography beats, the dancer's exact 3D pose crystallizes into a volumetric 3D smooth-min (smin)
  liquid-glass human sculpture (15 anatomical 3D capsules + rounded head + flowing liquid-glass surface ripples),
  glides smoothly into an architectural 3D V-chorus flanking & surrounding the dancer, and locks frozen in 3D space.
- Each clone casts warm golden 3D floor caustics onto the stone plaza and sheds ~55 frozen irregular crystal droplets
  along its fingertip/arm trail (~550 sparkling droplets total).
- True 3D depth occlusion against the live moving dancer (MediaPipe matte + depth) + Pass-3 cinema lens bokeh & grain.
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


VS = "#version 330\nin vec2 p; void main(){ gl_Position = vec4(p, 0.0, 1.0); }"

# Pass 1: Floor Caustics & Soft Contact Shadows on z=0 stone plaza beneath all active crystal clones
FLOOR_FS = """#version 330
uniform sampler2D uPlate;
uniform sampler2D uPersonMask;
uniform vec2 uRes;
uniform float uF, uTime;
uniform vec3 uRo, uSun;
uniform mat3 uCam;
uniform vec4 uCloneFeet[12];    /* xyz = clone center xy + height z, w = visibility alpha (0..1) */
uniform int uNumClones;
uniform float uPersonDepth;

layout(location=0) out vec4 oComp;
layout(location=1) out vec4 oDepth;   /* R: 1e4, G: effOcc (person depth inside mask) */
layout(location=2) out vec4 oMeta;    /* R: cgAlpha, G: cgInvDepth, B: occInvDepth, A: 1.0 */

float hash21(vec2 p) {
  p = fract(p * vec2(123.34, 456.21));
  p += dot(p, p + 45.32);
  return fract(p.x * p.y);
}

float vnoise2(vec2 p) {
  vec2 i = floor(p), f = fract(p);
  f = f * f * (3.0 - 2.0 * f);
  return mix(
    mix(hash21(i), hash21(i + vec2(1,0)), f.x),
    mix(hash21(i + vec2(0,1)), hash21(i + vec2(1,1)), f.x),
    f.y
  );
}

void main() {
  vec2 px0 = vec2(gl_FragCoord.x, uRes.y - gl_FragCoord.y);
  vec2 uv0 = px0 / uRes;
  vec3 plateCol = texture(uPlate, clamp(uv0, vec2(0.002), vec2(0.998))).rgb;
  float pMask = texture(uPersonMask, clamp(uv0, vec2(0.002), vec2(0.998))).r;
  vec3 rd = normalize(uCam * vec3((px0 - 0.5 * uRes) / uF, 1.0));

  vec3 col = plateCol;
  float tFloor = (rd.z < -0.004) ? (-uRo.z / rd.z) : 1e4;
  if (tFloor > 0.15 && tFloor < 16.0 && pMask < 0.45) {
    vec3 Xf = uRo + rd * tFloor;
    vec3 sun = normalize(uSun);
    float invSz = 1.0 / max(sun.z, 0.25);
    float caust = 0.0;
    float shad = 0.0;
    for (int i = 0; i < 12; i++) {
      if (i >= uNumClones) break;
      float vis = uCloneFeet[i].w;
      if (vis < 0.01) continue;
      /* Elongated sunlight caustic projection from clone torso onto stone plaza */
      vec2 sp = uCloneFeet[i].xy - sun.xy * (uCloneFeet[i].z * 0.72 * invSz);
      vec2 dVec = Xf.xy - sp;
      float r = length(dVec);
      if (r < 1.35) {
        float n1 = vnoise2(Xf.xy * 3.4 + vec2(float(i) * 1.7, uTime * 0.35));
        float n2 = vnoise2(Xf.xy * 6.2 - vec2(uTime * 0.4, float(i)));
        float web = pow(clamp(1.0 - abs(n1 - 0.5) * 2.3, 0.0, 1.0), 2.0) * 0.75
                  + pow(clamp(1.0 - abs(n2 - 0.5) * 2.5, 0.0, 1.0), 2.2) * 0.45;
        float env = exp(-(r * r) / 0.26);
        caust += (0.55 + 0.95 * web) * env * vis;
        shad += exp(-(r * r) / 0.45) * 0.32 * vis;
      }
      /* Crisp contact AO directly under grounded clone feet */
      float rf = length(Xf.xy - uCloneFeet[i].xy);
      shad += exp(-(rf * rf) / 0.16) * 0.42 * vis;
    }
    float floorMask = (1.0 - smoothstep(0.12, 0.45, pMask)) * (1.0 - smoothstep(9.0, 15.0, tFloor));
    vec3 golden = vec3(1.0, 0.92, 0.74);
    col = col * (1.0 - clamp(shad, 0.0, 0.52) * floorMask);
    col += golden * (clamp(caust, 0.0, 1.4) * 0.38 * floorMask) * (0.45 + 0.55 * col);
  }

  float effOcc = (pMask > 0.48) ? uPersonDepth : 1e4;
  oComp = vec4(clamp(col, 0.0, 1.0), 1.0);
  oDepth = vec4(1e4, effOcc, pMask, 1.0);
  oMeta = vec4(0.0, 0.0, 1.0 / max(effOcc, 0.25), 1.0);
}
"""

# Pass 2A: Individual 3D Volumetric Sculpted Liquid-Glass Human Clone (drawn per clone over its screen bounding box, back-to-front!)
CLONE_VS = """#version 330
in vec2 inQuad;          /* [-1..1] quad */
uniform vec4 uRectNdc;   /* (xmin, ymin, xmax, ymax) in NDC */
void main() {
  vec2 t = inQuad * 0.5 + 0.5;
  gl_Position = vec4(mix(uRectNdc.xy, uRectNdc.zw, t), 0.0, 1.0);
}
"""

CLONE_FS = """#version 330
uniform sampler2D uBase;        /* Pass-1 scene color (bottom-up) */
uniform sampler2D uPersonMask;  /* MediaPipe live person matte (top-down) */
uniform vec2 uRes;
uniform float uF, uTime, uIor, uDisp, uProxyDist, uPersonDepth;
uniform vec3 uRo, uSun, uTint;
uniform mat3 uCam;

/* 17 anatomical 3D capsules forming the exact grounded human pose (incl. feet!) + rounded head sphere */
uniform vec4 uCapA[17];   /* xyz = joint A (m), w = radius A (m) */
uniform vec4 uCapB[17];   /* xyz = joint B (m), w = radius B (m) */
uniform vec4 uHead;       /* xyz = head center, w = radius */
uniform vec4 uBound;      /* xyz = bounding sphere center, w = bounding radius */
uniform float uAlpha;     /* fade-in / visibility alpha */
uniform float uPhase;     /* unique liquid-glass surface ripple phase */

layout(location=0) out vec4 oCol;
layout(location=1) out vec4 oMeta;

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

float sdTaperCapsule(vec3 p, vec3 a, vec3 b, float ra, float rb) {
  vec3 pa = p - a, ba = b - a;
  float h = clamp(dot(pa, ba) / max(dot(ba, ba), 1e-6), 0.0, 1.0);
  return length(pa - ba * h) - mix(ra, rb, h);
}

float sdClone(vec3 p) {
  float d = length(p - uHead.xyz) - uHead.w;
  for (int i = 0; i < 17; i++) {
    float dc = sdTaperCapsule(p, uCapA[i].xyz, uCapB[i].xyz, uCapA[i].w, uCapB[i].w);
    float k = (i < 4) ? 0.085 : 0.048;
    d = smin(d, dc, k);
  }
  /* Subtle sculptural liquid-glass flow folds across torso & limbs (frozen in time with gentle crystalline shimmer!) */
  float rip = (vnoise(p * 7.2 + vec3(uPhase, uPhase * 0.6, uTime * 0.22)) - 0.5) * 0.018;
  float rip2 = sin(p.z * 26.0 + uPhase * 3.0) * cos(p.x * 18.0 - uPhase) * 0.005;
  return d + rip + rip2;
}

vec3 cloneNormal(vec3 p) {
  vec2 e = vec2(0.0028, -0.0028);
  return normalize(
    e.xyy * sdClone(p + e.xyy) +
    e.yyx * sdClone(p + e.yyx) +
    e.yxy * sdClone(p + e.yxy) +
    e.xxx * sdClone(p + e.xxx)
  );
}

vec3 sampleEnv(vec3 ro, vec3 rd) {
  float tf = (rd.z < -0.012) ? (-ro.z / rd.z) : 1e4;
  if (tf < 0.05 || tf > 22.0) tf = 1e4;
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
  vec2 uvTop = px0 / uRes;
  float pMask = texture(uPersonMask, clamp(uvTop, vec2(0.002), vec2(0.998))).r;
  float effOcc = (pMask > 0.48) ? uPersonDepth : 1e4;

  vec3 acc = vec3(0.0);
  float cov = 0.0;
  float minT = 1e4;
  vec2 offs[4] = vec2[4](vec2(-0.25, -0.25), vec2(0.25, -0.25), vec2(-0.25, 0.25), vec2(0.25, 0.25));
  vec3 L_sun = normalize(uSun);
  vec3 L_sky = normalize(vec3(0.35, -0.35, 0.87));

  for (int k = 0; k < 4; k++) {
    vec2 px = px0 + offs[k];
    vec3 rd = normalize(uCam * vec3((px - 0.5 * uRes) / uF, 1.0));
    vec3 oc = uBound.xyz - uRo;
    float b = dot(oc, rd);
    float c = dot(oc, oc) - uBound.w * uBound.w;
    float disc = b * b - c;
    if (disc < 0.0) continue;
    float sq = sqrt(disc);
    float tStart = max(b - sq, 0.05);
    float tEnd = b + sq;
    if (tStart >= tEnd) continue;

    float t = tStart;
    float tHit = -1.0;
    for (int s = 0; s < 58; s++) {
      float d = sdClone(uRo + rd * t);
      if (d < 0.0018) { tHit = t; break; }
      t += max(d * 0.88, 0.003);
      if (t > tEnd) break;
    }
    if (tHit < 0.0) continue;

    /* Exact MediaPipe depth occlusion: parts of clone behind live dancer disappear cleanly behind him! */
    float visP = (tHit > uPersonDepth + 0.04) ? (1.0 - smoothstep(0.22, 0.72, pMask)) : 1.0;
    if (visP < 0.02) continue;
    minT = min(minT, tHit);

    vec3 p1 = uRo + rd * tHit;
    vec3 n1 = cloneNormal(p1);
    vec3 rdIn = refract(rd, n1, 1.0 / uIor);
    if (length(rdIn) < 0.001) rdIn = reflect(rd, n1);

    float thick = 0.04;
    vec3 p2 = p1 + rdIn * 0.025;
    for (int s = 0; s < 20; s++) {
      float dIn = -sdClone(p2);
      if (dIn < 0.0025) break;
      float st = max(dIn * 0.88, 0.010);
      p2 += rdIn * st;
      thick += st;
      if (thick > 0.65) break;
    }
    vec3 n2 = -cloneNormal(p2);
    vec3 rdOutG = refract(rdIn, n2, uIor);
    if (length(rdOutG) < 0.001) rdOutG = normalize(mix(rdIn, reflect(rdIn, n2), 0.36));

    vec3 bend = rdOutG - rd;
    vec3 rdOutR = normalize(rd + bend * ((uIor - uDisp) - 1.0) / max(uIor - 1.0, 0.05));
    vec3 rdOutB = normalize(rd + bend * ((uIor + uDisp) - 1.0) / max(uIor - 1.0, 0.05));

    vec3 refrCol = vec3(sampleEnv(p2, rdOutR).r, sampleEnv(p2, rdOutG).g, sampleEnv(p2, rdOutB).b);
    refrCol = refrCol * exp(-thick * uTint * 0.45) * 1.24 + vec3(0.035, 0.040, 0.045);

    float ndv = max(dot(-rd, n1), 0.0);
    float fresnel = 0.035 + 0.965 * pow(1.0 - ndv, 3.8);
    vec3 reflDir = reflect(rd, n1);
    vec3 envRefl = sampleEnv(p1 + n1 * 0.005, reflDir);
    float skyUp = smoothstep(-0.15, 0.85, reflDir.z);
    envRefl = mix(envRefl * 1.08 + vec3(0.06), envRefl * 1.28 + vec3(0.32, 0.34, 0.37), skyUp * 0.58);

    float spSun = pow(max(dot(n1, normalize(L_sun - rd)), 0.0), 210.0) * 2.55 + pow(max(dot(n1, normalize(L_sun - rd)), 0.0), 24.0) * 0.24;
    float spSky = pow(max(dot(n1, normalize(L_sky - rd)), 0.0), 140.0) * 1.05;
    float backRim = pow(max(dot(-n2, L_sun), 0.0), 4.0) * 0.28;
    float rim = pow(1.0 - ndv, 2.4) * (0.52 + 0.40 * skyUp);

    vec3 col = mix(refrCol, envRefl, clamp(fresnel * 0.48, 0.0, 0.78));
    col += vec3(1.00, 0.95, 0.85) * (spSun + backRim);
    col += vec3(0.92, 0.96, 1.00) * (spSky + rim);
    acc += col * visP;
    cov += visP;
  }

  if (cov < 0.05) discard;
  float a = clamp(cov * 0.25 * uAlpha, 0.0, 1.0);
  oCol = vec4(clamp(acc / cov, 0.0, 1.0), a);
  oMeta = vec4(1.0, 1.0 / max(minT, 0.20), 1.0 / max(effOcc, 0.20), a);
}
"""

# Pass 2B: Instanced 3D Shedding Crystal Echo Droplets trailing from every clone's hands, shoulders & arcs
DROP_VS = """#version 330
in vec2 inQuad;
in vec4 inPosRad;
in vec4 inAxisAsp;
in vec4 inSeed;       /* x = wobSeed, y = wobAmp, z = opacity, w = phase */
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
  if (vRad <= 0.0004 || inSeed.z <= 0.01 || q.z <= 0.08 + boundR * 0.2) {
    gl_Position = vec4(2.0, 2.0, 2.0, 1.0);
    return;
  }
  vec2 ctrPx = uF * q.xy / q.z + 0.5 * uRes;
  float pxR = clamp(uF * boundR / max(q.z - boundR * 0.78, 0.05) + 3.5, 3.5, 860.0);
  vec2 px = ctrPx + inQuad * pxR;
  vec2 ndc = vec2(px.x / uRes.x * 2.0 - 1.0, (1.0 - px.y / uRes.y) * 2.0 - 1.0);
  gl_Position = vec4(ndc, 0.0, 1.0);
}
"""

DROP_FS = """#version 330
in vec3 vCenter;
in float vRad;
in vec3 vAxis;
in float vAsp;
in vec4 vSeed;

uniform sampler2D uBase;
uniform sampler2D uPersonMask;
uniform vec2 uRes;
uniform float uF, uIor, uDisp, uProxyDist, uPersonDepth;
uniform vec3 uRo, uSun;
uniform mat3 uCam;

layout(location=0) out vec4 oCol;
layout(location=1) out vec4 oMeta;

float sdDrop(vec3 p) {
  vec3 q = p - vCenter;
  float along = dot(q, vAxis);
  vec3 perp = q - vAxis * along;
  float taper = 1.0 + 0.22 * clamp(along / max(vRad * vAsp, 1e-4), -1.0, 1.0);
  vec3 qs = perp / max(taper, 0.58) + vAxis * (along / max(vAsp, 1.0));
  float d = (length(qs) - vRad) * min(1.0, 0.72 + 0.28 / max(vAsp, 1.0));
  float ph = vSeed.x * 19.3;
  float wob = sin(q.x / max(vRad, 1e-4) * 2.8 + ph) * cos(q.y / max(vRad, 1e-4) * 2.6 - ph);
  return d + wob * vRad * vSeed.y;
}

vec3 dropN(vec3 p) {
  float h = max(vRad * 0.018, 0.0002);
  vec2 e = vec2(h, -h);
  return normalize(e.xyy*sdDrop(p+e.xyy) + e.yyx*sdDrop(p+e.yyx) + e.yxy*sdDrop(p+e.yxy) + e.xxx*sdDrop(p+e.xxx));
}

vec3 sampleEnv(vec3 ro, vec3 rd) {
  float tf = (rd.z < -0.012) ? (-ro.z / rd.z) : 1e4;
  if (tf < 0.05 || tf > 22.0) tf = 1e4;
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
  float pMask = texture(uPersonMask, clamp(px0 / uRes, vec2(0.002), vec2(0.998))).r;
  float effOcc = (pMask > 0.48) ? uPersonDepth : 1e4;
  float boundR = vRad * max(vAsp, 1.0) * 1.55;

  vec3 acc = vec3(0.0);
  float cov = 0.0;
  float minT = 1e4;
  vec2 offs[4] = vec2[4](vec2(-0.25, -0.25), vec2(0.25, -0.25), vec2(-0.25, 0.25), vec2(0.25, 0.25));
  vec3 L_sun = normalize(uSun);
  vec3 L_sky = normalize(vec3(0.35, -0.35, 0.87));

  for (int k = 0; k < 4; k++) {
    vec2 px = px0 + offs[k];
    vec3 rd = normalize(uCam * vec3((px - 0.5 * uRes) / uF, 1.0));
    vec3 oc = vCenter - uRo;
    float b = dot(oc, rd);
    float disc = b * b - (dot(oc, oc) - boundR * boundR);
    if (disc < 0.0) continue;
    float sq = sqrt(disc);
    float t = max(b - sq, 0.04);
    float tEnd = b + sq;
    if (t >= tEnd) continue;
    float tHit = -1.0;
    for (int s = 0; s < 26; s++) {
      float d = sdDrop(uRo + rd * t);
      if (d < max(vRad * 0.009, 0.00015)) { tHit = t; break; }
      t += d * 0.88;
      if (t > tEnd) break;
    }
    if (tHit < 0.0) continue;
    float visP = (tHit > uPersonDepth + 0.04) ? (1.0 - smoothstep(0.22, 0.72, pMask)) : 1.0;
    if (visP < 0.02) continue;
    minT = min(minT, tHit);

    vec3 p1 = uRo + rd * tHit;
    vec3 n1 = dropN(p1);
    vec3 rdIn = refract(rd, n1, 1.0 / uIor);
    if (length(rdIn) < 0.001) rdIn = reflect(rd, n1);
    vec3 rdOutG = normalize(mix(rd, rdIn, 0.82));
    vec3 bend = rdOutG - rd;
    vec3 refrCol = vec3(
      sampleEnv(p1 + rdIn * vRad, normalize(rd + bend * ((uIor - uDisp) - 1.0) / max(uIor - 1.0, 0.05))).r,
      sampleEnv(p1 + rdIn * vRad, rdOutG).g,
      sampleEnv(p1 + rdIn * vRad, normalize(rd + bend * ((uIor + uDisp) - 1.0) / max(uIor - 1.0, 0.05))).b
    ) * 1.15;
    float ndv = max(dot(-rd, n1), 0.0);
    float fresnel = 0.04 + 0.96 * pow(1.0 - ndv, 3.6);
    vec3 reflDir = reflect(rd, n1);
    vec3 envRefl = sampleEnv(p1 + n1 * 0.004, reflDir) * 1.18 + vec3(0.22) * smoothstep(-0.1, 0.85, reflDir.z);
    float sp1 = pow(max(dot(n1, normalize(L_sun - rd)), 0.0), 190.0) * 2.6;
    float sp2 = pow(max(dot(n1, normalize(L_sky - rd)), 0.0), 130.0) * 1.1;
    float rim = pow(1.0 - ndv, 2.6) * 0.52;
    vec3 col = mix(refrCol, envRefl, clamp(fresnel * 0.54, 0.0, 0.80)) + vec3(1.0, 0.96, 0.88) * sp1 + vec3(0.94, 0.97, 1.0) * (sp2 + rim);
    acc += col * visP;
    cov += visP;
  }
  if (cov < 0.05) discard;
  float a = clamp(cov * 0.25 * vSeed.z, 0.0, 1.0);
  oCol = vec4(clamp(acc / cov, 0.0, 1.0), a);
  oMeta = vec4(1.0, 1.0 / max(minT, 0.20), 1.0 / max(effOcc, 0.20), a);
}
"""

# Pass 3: Camera Optical Finish (Person-Locked Thin-Lens Bokeh + OLPF Softness + Golden Halation + Sensor Grain)
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
    float rA = max(coc0, 1.30);
    cgSum += col0 * (0.28 * cgA0);
    cgW += 0.28 * cgA0;
    for (int k = 1; k <= 10; k++) {
      float f = sqrt(float(k) / 10.0);
      float ang = float(k) * GA;
      vec2 off = vec2(cos(ang), sin(ang)) * (f * rA);
      vec2 suv = clamp(uv + off * pxSize, vec2(0.001), vec2(0.999));
      vec3 mj = texture(uMeta, suv).rgb;
      if (mj.b > dInv0 + 0.02) continue;
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
    if (lum > 0.78) halation += cj * (exp(-r * r / 28.0) * (lum - 0.78) * mj.r);
    if (cocJ > 1.35 && r <= cocJ + 0.85 && (cgA0 < 0.02 || dInvJ > dInv0 + 0.08)) {
      float disk = smoothstep(cocJ + 0.85, cocJ - 0.85, r);
      float normArea = (uMaxCoc * uMaxCoc) / max(cocJ * cocJ * 24.0, 1.0);
      float w = disk * mj.r * min(normArea, 0.35);
      fgHaloCol += cj * w;
      fgHaloW += w;
    }
  }

  vec3 cgCol = (cgW > 0.001) ? (cgSum / cgW) : col0;
  vec3 res = mix(col0, cgCol, clamp(cgA0, 0.0, 1.0));
  float fgAlpha = clamp(fgHaloW * 1.15, 0.0, 0.92);
  if (fgHaloW > 0.001) res = mix(res, fgHaloCol / fgHaloW, fgAlpha * (1.0 - cgA0 * 0.55));
  float cgPresence = clamp(max(cgA0, fgAlpha), 0.0, 1.0);
  res += halation * vec3(1.02, 0.96, 0.88) * 0.060;

  if (cgPresence > 0.005) {
    vec3 graded = res * (1.0 - uBlackFloor) + vec3(uBlackFloor);
    graded = graded / (1.0 + 0.05 * max(graded - 0.86, vec3(0.0)));
    vec2 p1 = gl_FragCoord.xy + vec2(uFrame * 17.3, uFrame * 31.7);
    vec2 p2 = floor(gl_FragCoord.xy * 0.5) + vec2(uFrame * 11.1, uFrame * 23.9);
    float gLuma = (hash12(p1) + hash12(p1 + 7.13) - 1.0) * 0.62 + (hash12(p2) + hash12(p2 + 5.71) - 1.0) * 0.38;
    float lum = dot(graded, vec3(0.299, 0.587, 0.114));
    graded += vec3(gLuma) * (uGrainAmt * (1.15 - 0.65 * smoothstep(0.55, 0.98, lum)));
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


def encode(c, out_dir="out"):
    out = c["dir"] / out_dir
    out.mkdir(exist_ok=True)
    nm = c["name"]
    r = f"{c['fps_num']}/{c['fps_den']}"
    ff = find_ffmpeg()
    tags = ["-pix_fmt", "yuv420p", "-color_primaries", "bt709", "-color_trc", "bt709", "-colorspace", "bt709", "-movflags", "+faststart"]
    x264 = ["-c:v", "libx264", "-profile:v", "high", "-level", "4.1", "-crf", "14", "-preset", "slow"]
    run = lambda a: subprocess.run([ff, "-v", "error", "-y"] + a, check=True)
    run(["-framerate", r, "-i", str(work(c, "comp", "%04d.png"))] + x264 + tags + [str(out / f"{nm}_Composite.mp4")])
    run([
        "-framerate", r, "-i", str(work(c, "comp", "%04d.png")),
        "-framerate", r, "-i", str(work(c, "frames", "%04d.jpg")),
        "-filter_complex",
        "[0]scale=1080:608[a];[1]scale=1080:608[b];[a][b]vstack,pad=1080:1350:0:67:color=black",
    ] + x264 + tags + [str(out / f"{nm}_Compare_1080x1350.mp4")])
    print("encoded ->", out)


def pose_to_capsules(P33, offset):
    """Build 17 grounded anatomical 3D tapered capsules (incl. feet touching z=0!) + rounded head."""
    J = P33 + offset[None, :]
    # Ground the lowest heel/toe firmly to stone plaza z = 0.035m so NO clone ever floats above the floor!
    min_foot_z = float(min(J[29, 2], J[30, 2], J[31, 2], J[32, 2]))
    J[:, 2] -= (min_foot_z - 0.035)
    sh_mid = 0.5 * (J[11] + J[12])
    hip_mid = 0.5 * (J[23] + J[24])
    chest = 0.70 * sh_mid + 0.30 * hip_mid
    neck = 0.80 * sh_mid + 0.20 * J[0]
    head_c = 0.5 * (J[7] + J[8]) + np.array([0.0, 0.0, 0.03])

    caps = [
        (J[11], J[12], 0.088, 0.088),     # 0: shoulders
        (sh_mid, hip_mid, 0.132, 0.122),  # 1: torso core
        (J[23], J[24], 0.105, 0.105),     # 2: pelvis / hips
        (neck, head_c, 0.052, 0.068),     # 3: neck
        (J[11], J[13], 0.056, 0.048),     # 4: L upper arm
        (J[13], J[15], 0.046, 0.036),     # 5: L forearm
        (J[15], J[19], 0.036, 0.032),     # 6: L hand
        (J[12], J[14], 0.056, 0.048),     # 7: R upper arm
        (J[14], J[16], 0.046, 0.036),     # 8: R forearm
        (J[16], J[20], 0.036, 0.032),     # 9: R hand
        (J[23], J[25], 0.095, 0.080),     # 10: L thigh
        (J[25], J[27], 0.078, 0.066),     # 11: L calf
        (J[24], J[26], 0.095, 0.080),     # 12: R thigh
        (J[26], J[28], 0.078, 0.066),     # 13: R calf
        (chest, J[11]*0.5 + J[23]*0.5, 0.105, 0.098),  # 14: ribcage fill
        (J[29], J[31], 0.044, 0.036),     # 15: L foot grounded on plaza
        (J[30], J[32], 0.044, 0.036),     # 16: R foot grounded on plaza
    ]
    pts = np.vstack([c[0] for c in caps] + [c[1] for c in caps] + [head_c])
    b_cen = 0.5 * (pts.min(0) + pts.max(0))
    b_rad = float(np.linalg.norm(pts - b_cen[None, :], axis=1).max()) + 0.18
    return caps, (head_c, 0.105), (b_cen, b_rad), pts


def project_rect_ndc(pts, ro, R, f, W, H, pad_m=0.20):
    """Conservative screen-space NDC rect covering a clone's 3D bounding box."""
    mn = pts.min(0) - pad_m
    mx = pts.max(0) + pad_m
    corners = np.array([[x, y, z] for x in (mn[0], mx[0]) for y in (mn[1], mx[1]) for z in (mn[2], mx[2])])
    q = (corners - ro[None, :]) @ R
    if (q[:, 2] <= 0.10).all():
        return None
    z = np.maximum(q[:, 2], 0.10)
    ux = f * q[:, 0] / z + 0.5 * W
    uy = f * q[:, 1] / z + 0.5 * H
    x0 = max(-1.0, min(1.0, float(ux.min()) / W * 2.0 - 1.0))
    x1 = max(-1.0, min(1.0, float(ux.max()) / W * 2.0 - 1.0))
    y0 = max(-1.0, min(1.0, (1.0 - float(uy.max()) / H) * 2.0 - 1.0))
    y1 = max(-1.0, min(1.0, (1.0 - float(uy.min()) / H) * 2.0 - 1.0))
    if x1 <= x0 or y1 <= y0:
        return None
    return (x0, y0, x1, y1)


def main():
    c = load(sys.argv[1] if len(sys.argv) > 1 and not sys.argv[1].startswith("--") else here / "piece_echo")
    S = scene(c)
    L = np.load(work(c, "limbs.npz"))
    P_limbs = np.asarray(L["P"], np.float64)
    p_depths = np.asarray(L["depth"], np.float64)
    W, H, N, fps = c["W"], c["H"], c["N"], c["fps"]
    f = float(S["f"])
    cfg = c.get("echo", {})
    stills = [int(x) for x in sys.argv[sys.argv.index("--still") + 1].split(",")] if "--still" in sys.argv else None
    out_dir = sys.argv[sys.argv.index("--out") + 1] if "--out" in sys.argv else "out"

    cam_path = np.asarray(S["cam"], np.float64)
    cam_rot = np.asarray(S["rot"], np.float64)
    spawn_frames = [int(x) for x in cfg.get("spawn_frames", [8, 22, 36, 50, 64, 78, 92, 106, 120, 134])]
    glide_dur_s = float(cfg.get("glide_dur_s", 0.65))

    # 10 sculptural 3D V-chorus pedestal offsets flanking & surrounding the live dancer (mix of behind & foreground!)
    # (dx_right, dy_back, dz) in metres relative to live dancer pose at spawn_frame
    chorus_offsets = [
        (-0.58,  0.22, 0.0),
        ( 0.62,  0.26, 0.0),
        (-1.12,  0.54, 0.0),
        ( 1.18,  0.58, 0.0),
        (-1.62,  0.90, 0.0),
        ( 1.66,  0.92, 0.0),
        (-0.86, -0.32, 0.0),  # foreground-left flanking glass clone (closer to camera -> creamy near bokeh!)
        ( 0.90, -0.28, 0.0),  # foreground-right flanking glass clone
        (-1.38,  0.18, 0.0),
        ( 1.42,  0.20, 0.0),
    ]

    # Build clone definitions + 55 frozen shedding crystal droplets per clone (~550 droplets total!)
    rng = np.random.default_rng(int(cfg.get("seed", 2029)))
    clones = []
    echo_drops = []
    for k, sf in enumerate(spawn_frames):
        if sf >= N:
            continue
        off = np.array(chorus_offsets[k % len(chorus_offsets)], np.float64)
        P_pose = P_limbs[sf].copy()
        clones.append({"sf": sf, "pose": P_pose, "off": off, "phase": float(k) * 1.7 + 0.5})
        # Shed 55 irregular crystal pearls along wrists, elbows, shoulders, knees & glide trajectory!
        key_joints = [0, 11, 12, 13, 14, 15, 16, 19, 20, 23, 24, 25, 26]
        for j in range(55):
            jid = key_joints[j % len(key_joints)]
            frac = float(rng.uniform(0.18, 1.12))
            p0 = P_pose[jid]
            p1 = P_pose[jid] + off * frac + rng.normal(0.0, [0.16, 0.18, 0.16])
            p1[2] = float(np.clip(p1[2], 0.12, 2.25))
            ax = off + rng.normal(0.0, 0.45, 3); ax /= max(float(np.linalg.norm(ax)), 1e-4)
            rad = float(np.exp(rng.uniform(math.log(0.006), math.log(0.032))))
            echo_drops.append({
                "sf": sf, "p0": p0, "p1": p1, "rad": rad, "axis": ax,
                "asp": float(rng.uniform(1.12, 1.85)), "seed": float(rng.uniform(0, 10)),
                "wob": float(rng.uniform(0.04, 0.12)), "delay": float(rng.uniform(0.0, 0.22)),
            })
    print(f"Motion Echo: {len(clones)} 3D volumetric glass clones + {len(echo_drops)} crystal echo droplets")

    ctx = moderngl.create_standalone_context(require=330)
    floor_prog = ctx.program(vertex_shader=VS, fragment_shader=FLOOR_FS)
    clone_prog = ctx.program(vertex_shader=CLONE_VS, fragment_shader=CLONE_FS)
    drop_prog = ctx.program(vertex_shader=DROP_VS, fragment_shader=DROP_FS)
    post_prog = ctx.program(vertex_shader=VS, fragment_shader=POST_FS)
    copy2_prog = ctx.program(
        vertex_shader=VS,
        fragment_shader="""#version 330
        uniform sampler2D uSrc0, uSrc1; uniform vec2 uRes;
        layout(location=0) out vec4 o0; layout(location=1) out vec4 o1;
        void main(){ vec2 uv = gl_FragCoord.xy / uRes; o0 = texture(uSrc0, uv); o1 = texture(uSrc1, uv); }""",
    )

    tri = ctx.buffer(np.array([-1, -1, 3, -1, -1, 3], "f4").tobytes())
    va_floor = ctx.vertex_array(floor_prog, [(tri, "2f", "p")])
    va_copy2 = ctx.vertex_array(copy2_prog, [(tri, "2f", "p")])
    va_post = ctx.vertex_array(post_prog, [(tri, "2f", "p")])

    quad_vbo = ctx.buffer(np.array([-1, -1, 1, -1, 1, 1, -1, -1, 1, 1, -1, 1], "f4").tobytes())
    va_clone = ctx.vertex_array(clone_prog, [(quad_vbo, "2f", "inQuad")])
    inst_buf = ctx.buffer(reserve=max(len(echo_drops), 1) * 12 * 4)
    va_drop = ctx.vertex_array(drop_prog, [
        (quad_vbo, "2f", "inQuad"),
        (inst_buf, "4f 4f 4f/i", "inPosRad", "inAxisAsp", "inSeed"),
    ])

    plate_t = ctx.texture((W, H), 3); plate_t.filter = (moderngl.LINEAR, moderngl.LINEAR); plate_t.repeat_x = plate_t.repeat_y = False
    mask_t = ctx.texture((W, H), 1); mask_t.filter = (moderngl.LINEAR, moderngl.LINEAR); mask_t.repeat_x = mask_t.repeat_y = False

    t_hero = ctx.texture((W, H), 4, dtype="f4"); t_hero.filter = (moderngl.LINEAR, moderngl.LINEAR); t_hero.repeat_x = t_hero.repeat_y = False
    t_depth = ctx.texture((W, H), 4, dtype="f4"); t_depth.filter = (moderngl.NEAREST, moderngl.NEAREST)
    t_meta0 = ctx.texture((W, H), 4, dtype="f4"); t_meta0.filter = (moderngl.NEAREST, moderngl.NEAREST)
    fbo_hero = ctx.framebuffer([t_hero, t_depth, t_meta0])

    t_scene = ctx.texture((W, H), 4, dtype="f4"); t_scene.filter = (moderngl.LINEAR, moderngl.LINEAR); t_scene.repeat_x = t_scene.repeat_y = False
    t_meta = ctx.texture((W, H), 4, dtype="f4"); t_meta.filter = (moderngl.NEAREST, moderngl.NEAREST)
    fbo_scene = ctx.framebuffer([t_scene, t_meta])

    t_final = ctx.texture((W, H), 4, dtype="f4")
    fbo_final = ctx.framebuffer([t_final])

    ior = float(cfg.get("ior", 1.46))
    disp = float(cfg.get("dispersion", 0.018))
    tint = (0.042, 0.022, 0.014)

    copy2_prog["uRes"].value = (float(W), float(H))
    floor_prog["uRes"].value = (float(W), float(H)); floor_prog["uF"].value = f; floor_prog["uSun"].value = tuple(map(float, S["sun"]))
    clone_prog["uRes"].value = (float(W), float(H)); clone_prog["uF"].value = f; clone_prog["uSun"].value = tuple(map(float, S["sun"]))
    clone_prog["uIor"].value = ior; clone_prog["uDisp"].value = disp; clone_prog["uTint"].value = tint; clone_prog["uProxyDist"].value = 5.5
    drop_prog["uRes"].value = (float(W), float(H)); drop_prog["uF"].value = f; drop_prog["uSun"].value = tuple(map(float, S["sun"]))
    drop_prog["uIor"].value = ior; drop_prog["uDisp"].value = disp; drop_prog["uProxyDist"].value = 5.5

    post_prog["uRes"].value = (float(W), float(H))
    post_prog["uNearBlur"].value = 4.6
    post_prog["uFarBlur"].value = 4.2
    post_prog["uBaseSoft"].value = 0.68
    post_prog["uMaxCoc"].value = 9.5
    post_prog["uGrainAmt"].value = 0.012
    post_prog["uBlackFloor"].value = 0.018

    def render_one(i):
        pl = frame(c, i)
        m_img = cv2.GaussianBlur(cv2.imread(str(work(c, "mask", f"{i:04d}.png")), cv2.IMREAD_GRAYSCALE), (0, 0), 1.0)
        t_sec = i / fps
        ro = cam_path[i]; R = cam_rot[i]
        Rt = tuple(R.astype(np.float32).T.flatten())
        p_depth = float(p_depths[i])

        # Compute active clones and their current smoothly glided-and-frozen 3D offsets
        active_clones = []
        feet_list = []
        for cl in clones:
            age = (i - cl["sf"]) / fps
            if age <= 0.0:
                continue
            # Smooth C2 glide into 3D chorus position then DEAD-STILL frozen!
            x = min(max(age / glide_dur_s, 0.0), 1.0)
            w = 1.0 - (1.0 - x) ** 3.2
            off_now = cl["off"] * w
            alpha = min(1.0, age / 0.14)
            caps, head, bound, pts = pose_to_capsules(cl["pose"], off_now)
            z_cam = float(((bound[0] - ro) @ R)[2])
            feet_xy = 0.5 * (cl["pose"][27, :2] + cl["pose"][28, :2]) + off_now[:2]
            feet_list.append((float(feet_xy[0]), float(feet_xy[1]), float(bound[0][2]), float(alpha)))
            active_clones.append((z_cam, cl, caps, head, bound, pts, alpha))

        n_feet = len(feet_list)
        while len(feet_list) < 12:
            feet_list.append((0.0, 0.0, 0.0, 0.0))

        plate_t.write(pl[:, :, ::-1].tobytes())
        mask_t.write(m_img.tobytes())

        # Pass 1: Golden 3D floor caustics & contact shadows beneath all crystal clones
        plate_t.use(0); mask_t.use(1)
        floor_prog["uPlate"].value = 0; floor_prog["uPersonMask"].value = 1
        floor_prog["uRo"].value = tuple(map(float, ro)); floor_prog["uCam"].value = Rt
        floor_prog["uTime"].value = float(t_sec)
        floor_prog["uCloneFeet"].value = feet_list[:12]
        floor_prog["uNumClones"].value = n_feet
        floor_prog["uPersonDepth"].value = p_depth
        fbo_hero.use()
        ctx.disable(moderngl.BLEND)
        va_floor.render(moderngl.TRIANGLES)

        fbo_scene.use()
        t_hero.use(0); t_meta0.use(1)
        copy2_prog["uSrc0"].value = 0; copy2_prog["uSrc1"].value = 1
        va_copy2.render(moderngl.TRIANGLES)

        # Pass 2A: Render all active 3D volumetric sculpted liquid-glass clones sorted back-to-front!
        active_clones.sort(key=lambda x: -x[0])
        if active_clones:
            t_hero.use(0); mask_t.use(1)
            clone_prog["uBase"].value = 0; clone_prog["uPersonMask"].value = 1
            clone_prog["uRo"].value = tuple(map(float, ro)); clone_prog["uCam"].value = Rt
            clone_prog["uTime"].value = float(t_sec); clone_prog["uPersonDepth"].value = p_depth
            ctx.enable(moderngl.BLEND)
            ctx.blend_func = (moderngl.SRC_ALPHA, moderngl.ONE_MINUS_SRC_ALPHA)
            for z_cam, cl, caps, (hc, hr), (bc, br), pts, alpha in active_clones:
                rect = project_rect_ndc(pts, ro, R, f, W, H, pad_m=0.22)
                if rect is None:
                    continue
                clone_prog["uRectNdc"].value = rect
                clone_prog["uCapA"].value = [(*a, ra) for a, b, ra, rb in caps]
                clone_prog["uCapB"].value = [(*b, rb) for a, b, ra, rb in caps]
                clone_prog["uHead"].value = (*hc, hr)
                clone_prog["uBound"].value = (*bc, br)
                clone_prog["uAlpha"].value = float(alpha)
                clone_prog["uPhase"].value = float(cl["phase"])
                va_clone.render(moderngl.TRIANGLES)
            ctx.disable(moderngl.BLEND)

        # Pass 2B: Render all shedding frozen crystal echo droplets sorted back-to-front!
        rows_d = []
        for dr in echo_drops:
            age = (i - dr["sf"]) / fps - dr["delay"]
            if age <= 0.0:
                continue
            x = min(max(age / (glide_dur_s * 1.15), 0.0), 1.0)
            w = 1.0 - (1.0 - x) ** 3.4
            p_now = dr["p0"] + (dr["p1"] - dr["p0"]) * w
            op = min(1.0, age / 0.10)
            z_cam = float(((p_now - ro) @ R)[2])
            if z_cam > 0.12:
                rows_d.append((z_cam, [*p_now, dr["rad"] * (0.4 + 0.6 * op), *dr["axis"], dr["asp"], dr["seed"], dr["wob"], op, 0.0]))
        rows_d.sort(key=lambda x: -x[0])
        if rows_d:
            data = np.array([r[1] for r in rows_d], dtype=np.float32)
            inst_buf.write(data.tobytes())
            t_hero.use(0); mask_t.use(1)
            drop_prog["uBase"].value = 0; drop_prog["uPersonMask"].value = 1
            drop_prog["uRo"].value = tuple(map(float, ro)); drop_prog["uCam"].value = Rt
            drop_prog["uPersonDepth"].value = p_depth
            ctx.enable(moderngl.BLEND)
            ctx.blend_func = (moderngl.SRC_ALPHA, moderngl.ONE_MINUS_SRC_ALPHA)
            va_drop.render(moderngl.TRIANGLES, instances=len(rows_d))
            ctx.disable(moderngl.BLEND)

        # Pass 3: Camera optical finish locked onto live dancer depth
        fbo_final.use()
        t_scene.use(0); t_meta.use(1)
        post_prog["uScene"].value = 0; post_prog["uMeta"].value = 1
        post_prog["uFocusDist"].value = p_depth
        post_prog["uFrame"].value = float(i)
        va_post.render(moderngl.TRIANGLES)

        ctx.finish()
        arr = np.frombuffer(t_final.read(), "f4").reshape(H, W, 4)[::-1]
        img = (np.clip(arr[..., :3], 0, 1) * 255 + 0.5).astype(np.uint8)[:, :, ::-1]
        return pl, img

    if stills:
        rows = []
        hero = None
        for i in stills:
            pl, comp = render_one(i)
            if i == stills[len(stills) // 2]:
                hero = comp
            t_im = lambda im, s: label(cv2.resize(im, (960, 540)), s)
            rows.append(np.hstack([t_im(pl, f"plate {i}"), t_im(comp, f"glass echo {i}")]))
        out_p = work(c, "still_echo_sheet.jpg")
        cv2.imwrite(str(out_p), np.vstack(rows), [cv2.IMWRITE_JPEG_QUALITY, 92])
        if hero is not None:
            cv2.imwrite(str(work(c, "still_echo_hero.jpg")), hero, [cv2.IMWRITE_JPEG_QUALITY, 94])
        print("stills ->", out_p)
        return

    comp_dir = work(c, "comp")
    shutil.rmtree(comp_dir, ignore_errors=True)
    comp_dir.mkdir(parents=True)
    ts = time.time()
    for i in range(N):
        _, comp = render_one(i)
        cv2.imwrite(str(comp_dir / f"{i:04d}.png"), comp, [cv2.IMWRITE_PNG_COMPRESSION, 1])
        if i % 35 == 0 or i == N - 1:
            print(f"echo frame {i}/{N} ({time.time() - ts:.1f}s)", flush=True)
    encode(c, out_dir=out_dir)


if __name__ == "__main__":
    main()
