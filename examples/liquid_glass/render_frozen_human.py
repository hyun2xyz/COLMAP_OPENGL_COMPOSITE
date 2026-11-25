"""Human-Centered Frozen-Time Liquid Glass Splash (MediaPipe Pose + Segmentation + COLMAP 3D + ModernGL OpenGL 3.3).

    python render_frozen_human.py [piece_dir] --still 10,28,60,135,210   quick look -> work/still_human_sheet.jpg
    python render_frozen_human.py [piece_dir] --out out                  full render -> out/*_Composite.mp4, *_Compare_1080x1350.mp4

3-Pass MediaPipe + COLMAP Pro Compositing Pipeline:
- Uses MediaPipe Pose 33-joint 3D lifted skeleton (limbs.npz) + soft person segmentation matte (mask/%04d.png)
  + COLMAP 3D camera trajectory (scene.json) + optional 3D cylindrical/box architectural pillar holdout.
- Frames 0..19: Tense liquid-glass fluid ring/aura coalescing around the person's torso (wrapped both behind & in front of body).
- Frame 20 ("팡!"): Explosive 360-degree radial burst around the person into ~1,040 irregular, randomly sized crystal droplets
  (36 curved/pinched hero splash crowns + 184 mid organic drops + 820 fine '송골송골' dewy micro-beads).
- Frames 20..52: Smooth bullet-time brake freezing all 1,040 droplets dead-still in 3D world space (0 drift).
- Frames 52..224: Camera orbits smoothly around the person and frozen cloud in true 3D parallax:
  droplets behind the person are cleanly occluded by the MediaPipe person matte; droplets in front sweep cleanly across
  his coat and past the lens with person-locked thin-lens bokeh, halation, edge CA, and plate-matched sensor grain.
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

# Pass 1: Pre-burst liquid-glass aura ring wrapped around person + 3D floor caustics + 3D cylindrical column holdout
CORE_FS = """#version 330
uniform sampler2D uPlate;
uniform sampler2D uPersonMask;  /* top-down 0..1 soft MediaPipe person segmentation matte */
uniform vec2 uRes;
uniform float uF, uSimT;
uniform vec3 uRo, uSun;
uniform mat3 uCam;
uniform vec4 uRing;             /* xyz = torso center (m), w = ring intensity/scale (0 once shattered) */
uniform float uRingRad;         /* major radius of orbiting pre-burst ring around torso */
uniform float uPersonDepth;     /* exact 3D ray distance from camera to person torso (m) */
uniform vec4 uCaustDrops[28];
uniform int uNumCaust;
uniform vec4 uCylOcc;           /* xy = cylinder center, z = radius (0 = disabled), w = maxZ */
uniform float uIor, uDisp, uCausticGain, uProxyDist;
uniform vec3 uTint;

layout(location=0) out vec4 oComp;
layout(location=1) out vec4 oDepth;   /* R: min(tCore, tCylOcc), G: tCylOcc */
layout(location=2) out vec4 oMeta;    /* R: cgAlpha, G: cgInvDepth, B: occInvDepth, A: 1.0 */

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

/* Exact ray intersection with vertical cylindrical concrete column (z in [0, w]) */
float hitCylOcc(vec3 ro, vec3 rd) {
  if (uCylOcc.z <= 0.01) return 1e4;
  vec2 oc = ro.xy - uCylOcc.xy;
  float a = dot(rd.xy, rd.xy);
  if (a < 1e-8) return 1e4;
  float b = dot(oc, rd.xy);
  float c = dot(oc, oc) - uCylOcc.z * uCylOcc.z;
  float disc = b * b - a * c;
  if (disc < 0.0) return 1e4;
  float sq = sqrt(disc);
  float t1 = (-b - sq) / a;
  float t2 = (-b + sq) / a;
  if (t2 <= 0.05) return 1e4;
  float t = (t1 > 0.05) ? t1 : t2;
  float zh = ro.z + rd.z * t;
  if (zh < -0.1 || zh > uCylOcc.w) return 1e4;
  return t;
}

/* Tense pre-burst undulating ring of molten liquid glass circling the person's torso */
float sdRing(vec3 p) {
  if (uRing.w <= 0.005) return 1e4;
  vec3 q = p - uRing.xyz;
  float t = uSimT * 5.2;
  /* slightly tilted elliptical organic fluid ring around torso */
  float tilt = 0.22;
  vec3 qt = vec3(q.x, q.y * cos(tilt) - q.z * sin(tilt), q.y * sin(tilt) + q.z * cos(tilt));
  float ang = atan(qt.y, qt.x);
  float rWave = uRingRad + 0.045 * sin(ang * 3.0 + t) + 0.025 * cos(ang * 5.0 - t * 1.4);
  float zWave = 0.08 * sin(ang * 2.0 - t * 0.9);
  vec2 tr = vec2(length(qt.xy) - rWave, qt.z - zWave);
  float tubeR = (0.048 + 0.020 * sin(ang * 4.0 + t * 1.6)) * uRing.w;
  float d = length(tr) - tubeR;
  /* 6 coalescing bulbous beads swelling around the ring right before the '팡!' shatter */
  for (int k = 0; k < 6; k++) {
    float ka = float(k) * 1.0472 + t * 0.35;
    vec3 bc = vec3(cos(ka) * rWave, sin(ka) * rWave, 0.09 * sin(ka * 2.0 - t));
    d = smin(d, length(qt - bc) - 0.085 * uRing.w, 0.065 * uRing.w);
  }
  float rip = (vnoise(qt * 9.0 + vec3(0.0, t * 0.8, -t * 0.7)) - 0.5) * 0.015 * uRing.w;
  return d + rip;
}

vec3 ringNormal(vec3 p) {
  vec2 e = vec2(0.0022, -0.0022);
  return normalize(
    e.xyy * sdRing(p + e.xyy) +
    e.yyx * sdRing(p + e.yyx) +
    e.yxy * sdRing(p + e.yxy) +
    e.xxx * sdRing(p + e.xxx)
  );
}

vec3 samplePlateRay(vec3 ro, vec3 rd) {
  float tf = (rd.z < -0.012) ? (-ro.z / rd.z) : 1e4;
  if (tf < 0.05 || tf > 22.0) tf = 1e4;
  vec3 X = ro + rd * min(tf, uProxyDist);
  vec3 q = transpose(uCam) * (X - uRo);
  vec2 uv = (uF * q.xy / max(q.z, 0.08) + 0.5 * uRes) / uRes;
  if (q.z < 0.08) uv = clamp(vec2(1.0 - uv.x, uv.y), 0.01, 0.99);
  uv = abs(fract(uv * 0.5 + 0.5) * 2.0 - 1.0);
  return texture(uPlate, clamp(uv, vec2(0.006), vec2(0.994))).rgb;
}

void evalFloorLighting(vec3 Xf, out float caust, out float shad) {
  caust = 0.0;
  shad = 0.0;
  vec3 sun = normalize(uSun);
  float invSz = 1.0 / max(sun.z, 0.30);

  if (uRing.w > 0.005) {
    vec2 sp = uRing.xy - sun.xy * (uRing.z * invSz);
    float r = length(Xf.xy - sp);
    float ring = exp(-pow((r - uRingRad) / 0.24, 2.0));
    float n = vnoise(vec3(Xf.xy * 3.2, uSimT * 1.4));
    caust += ring * (0.45 + 0.65 * n) * uRing.w;
    shad += ring * 0.14 * uRing.w;
  }

  for (int i = 0; i < 28; i++) {
    if (i >= uNumCaust) break;
    vec4 dr = uCaustDrops[i];
    if (dr.w <= 0.004 || dr.z <= 0.05) continue;
    vec2 sp = dr.xy - sun.xy * (dr.z * invSz);
    float r = length(Xf.xy - sp);
    float R = dr.w * 2.2 + dr.z * 0.05;
    if (r > R * 3.0) continue;
    float e2 = exp(-(r * r) / max(R * R * 0.48, 0.001));
    float wgt = clamp(dr.w / 0.065, 0.15, 1.10) * clamp(1.3 / (0.6 + dr.z), 0.22, 1.0);
    caust += e2 * wgt * 0.52;
    shad += exp(-(r * r) / max(R * R * 1.8, 0.004)) * wgt * 0.09;
  }
  caust = clamp(caust, 0.0, 1.35);
  shad = clamp(shad, 0.0, 0.28);
}

void main() {
  vec2 px0 = vec2(gl_FragCoord.x, uRes.y - gl_FragCoord.y);
  vec2 uv0 = px0 / uRes;
  vec3 plateCol = texture(uPlate, clamp(uv0, vec2(0.002), vec2(0.998))).rgb;
  float pMask = texture(uPersonMask, clamp(uv0, vec2(0.002), vec2(0.998))).r;
  vec3 rd = normalize(uCam * vec3((px0 - 0.5 * uRes) / uF, 1.0));
  float tCyl = hitCylOcc(uRo, rd);

  /* Floor caustics ONLY on visible platform floor (never on top of person or concrete column!) */
  vec3 bgCol = plateCol;
  float tFloor = (rd.z < -0.004) ? (-uRo.z / rd.z) : 1e4;
  if (tFloor > 0.10 && tFloor < min(tCyl, 16.0) && pMask < 0.40) {
    vec3 Xf = uRo + rd * tFloor;
    float caust, shad;
    evalFloorLighting(Xf, caust, shad);
    if (caust > 0.001 || shad > 0.001) {
      float luma = dot(plateCol, vec3(0.299, 0.587, 0.114));
      float floorMask = (1.0 - smoothstep(0.15, 0.45, pMask)) * smoothstep(0.12, 0.30, luma) * (1.0 - smoothstep(8.0, 15.0, tFloor));
      vec3 stationLight = vec3(0.96, 0.98, 1.0);
      bgCol = bgCol * (1.0 - shad * floorMask);
      bgCol += stationLight * (caust * uCausticGain * 0.22 * floorMask) * (0.45 + 0.55 * bgCol);
    }
  }

  float outDist = tCyl;
  float cgAlpha = 0.0;
  float cgInvD = 0.0;
  vec3 finalCol = bgCol;

  /* Effective occluder distance for Pass-3 depth-aware bokeh: closer of concrete column OR person silhouette! */
  float effOcc = tCyl;
  if (pMask > 0.48) effOcc = min(effOcc, uPersonDepth);

  if (uRing.w > 0.005) {
    float boundR = uRingRad + 0.24;
    vec3 oc = uRing.xyz - uRo;
    float b = dot(oc, rd);
    float c = dot(oc, oc) - boundR * boundR;
    float disc = b * b - c;
    if (disc > 0.0 && b > -boundR) {
      vec3 acc = vec3(0.0);
      float cov = 0.0;
      float minD = 1e4;
      vec2 offs[4] = vec2[4](vec2(-0.25, -0.25), vec2(0.25, -0.25), vec2(-0.25, 0.25), vec2(0.25, 0.25));
      for (int k = 0; k < 4; k++) {
        vec2 px = px0 + offs[k];
        vec3 rdk = normalize(uCam * vec3((px - 0.5 * uRes) / uF, 1.0));
        float bk = dot(oc, rdk);
        float dk = bk * bk - c;
        if (dk < 0.0) continue;
        float sq = sqrt(dk);
        float t = max(bk - sq, 0.04);
        float tEnd = min(bk + sq, tCyl);
        if (t >= tEnd) continue;
        float tHit = -1.0;
        for (int s = 0; s < 56; s++) {
          float d = sdRing(uRo + rdk * t);
          if (d < 0.0012) { tHit = t; break; }
          t += max(d * 0.85, 0.002);
          if (t > tEnd) break;
        }
        if (tHit < 0.0) continue;
        /* Person foreground/background occlusion: ring back arc stays cleanly behind person! */
        float visP = (tHit > uPersonDepth + 0.04) ? (1.0 - smoothstep(0.22, 0.72, pMask)) : 1.0;
        if (visP < 0.02) continue;

        minD = min(minD, tHit);
        vec3 p1 = uRo + rdk * tHit;
        vec3 n1 = ringNormal(p1);
        vec3 rdIn = refract(rdk, n1, 1.0 / uIor);
        if (length(rdIn) < 0.001) rdIn = reflect(rdk, n1);
        float thick = 0.04;
        vec3 p2 = p1 + rdIn * 0.02;
        for (int s = 0; s < 16; s++) {
          float dIn = -sdRing(p2);
          if (dIn < 0.0018) break;
          float st = max(dIn * 0.88, 0.006);
          p2 += rdIn * st;
          thick += st;
          if (thick > 0.35) break;
        }
        vec3 n2 = -ringNormal(p2);
        vec3 rdOutG = refract(rdIn, n2, uIor);
        if (length(rdOutG) < 0.001) rdOutG = normalize(mix(rdIn, reflect(rdIn, n2), 0.38));
        vec3 bend = rdOutG - rdk;
        vec3 rdOutR = normalize(rdk + bend * ((uIor - uDisp) - 1.0) / max(uIor - 1.0, 0.05));
        vec3 rdOutB = normalize(rdk + bend * ((uIor + uDisp) - 1.0) / max(uIor - 1.0, 0.05));
        vec3 refrCol = vec3(samplePlateRay(p2, rdOutR).r, samplePlateRay(p2, rdOutG).g, samplePlateRay(p2, rdOutB).b);
        refrCol *= exp(-thick * uTint * 0.50) * 1.08;

        float ndv = max(dot(-rdk, n1), 0.0);
        float fresnel = 0.04 + 0.96 * pow(1.0 - ndv, 3.8);
        vec3 reflDir = reflect(rdk, n1);
        vec3 envRefl = samplePlateRay(p1 + n1 * 0.004, reflDir);
        float upGlow = smoothstep(-0.1, 0.85, reflDir.z);
        envRefl = mix(envRefl * 1.04 + vec3(0.04), envRefl * 1.22 + vec3(0.24, 0.26, 0.30), upGlow * 0.55);

        vec3 L1 = normalize(vec3( 0.08,  0.15, 0.98));
        vec3 L2 = normalize(vec3( 0.72,  0.25, 0.65));
        vec3 L3 = normalize(vec3(-0.62, -0.25, 0.74));
        float sp1 = pow(max(dot(n1, normalize(L1 - rdk)), 0.0), 220.0) * 2.4 + pow(max(dot(n1, normalize(L1 - rdk)), 0.0), 28.0) * 0.16;
        float sp2 = pow(max(dot(n1, normalize(L2 - rdk)), 0.0), 170.0) * 1.25;
        float sp3 = pow(max(dot(n1, normalize(L3 - rdk)), 0.0), 170.0) * 0.85;
        float rim = pow(1.0 - ndv, 2.7) * (0.40 + 0.30 * upGlow);

        vec3 c0 = mix(refrCol, envRefl, clamp(fresnel * 0.56, 0.0, 0.82));
        c0 += vec3(0.95, 0.98, 1.0) * sp1 + vec3(1.0, 0.97, 0.92) * sp2 + vec3(0.92, 0.94, 1.0) * sp3 + vec3(0.96, 0.98, 1.0) * rim;
        acc += c0 * visP;
        cov += visP;
      }
      if (cov > 0.05) {
        float a = clamp(cov * 0.25, 0.0, 1.0);
        finalCol = mix(bgCol, clamp(acc / cov, 0.0, 1.0), a);
        outDist = min(tCyl, minD);
        cgAlpha = a;
        cgInvD = a / max(minD, 0.20);
      }
    }
  }

  oComp = vec4(clamp(finalCol, 0.0, 1.0), 1.0);
  oDepth = vec4(outDist, effOcc, 0.0, 1.0);
  oMeta = vec4(cgAlpha, cgInvD, 1.0 / max(effOcc, 0.20), 1.0);
}
"""

# Pass 2: Instanced 3D Raymarched Irregular Frozen Liquid-Glass Droplets with MediaPipe Person Occlusion!
DROP_VS = """#version 330
in vec2 inQuad;
in vec4 inPosRad;
in vec4 inAxisAsp;
in vec4 inShape;
in vec4 inLobeDir;

uniform vec2 uRes;
uniform float uF;
uniform vec3 uRo;
uniform mat3 uCam;

out vec3 vCenter;
out float vRad;
out vec3 vAxis;
out float vAsp;
out vec4 vShape;
out vec4 vLobeDir;
out float vBoundR;

void main() {
  vCenter = inPosRad.xyz;
  vRad = inPosRad.w;
  vAxis = normalize(inAxisAsp.xyz);
  vAsp = inAxisAsp.w;
  vShape = inShape;
  vLobeDir = inLobeDir;

  float lobeExtra = (inShape.z > 0.05) ? (0.45 + 0.55 * inShape.z) : 0.0;
  vBoundR = vRad * (max(vAsp, 1.0) * 1.38 + lobeExtra + abs(inShape.w) * 0.75 + 0.25);

  vec3 q = transpose(uCam) * (vCenter - uRo);
  if (vRad <= 0.0005 || q.z <= 0.05 + vBoundR * 0.15) {
    gl_Position = vec4(2.0, 2.0, 2.0, 1.0);
    return;
  }
  vec2 ctrPx = uF * q.xy / q.z + 0.5 * uRes;
  float pxR = clamp(uF * vBoundR / max(q.z - vBoundR * 0.80, 0.04) + 3.5, 3.5, 960.0);
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
in vec4 vShape;
in vec4 vLobeDir;
in float vBoundR;

uniform sampler2D uBase;        /* Pass-1 FBO color (bottom-up) */
uniform sampler2D uDepth;       /* Pass-1 FBO depth: R = min(tCore, tCylOcc), G = effOcc (bottom-up) */
uniform sampler2D uPersonMask;  /* MediaPipe soft person matte (top-down!) */
uniform vec2 uRes;
uniform float uF, uIor, uDisp, uProxyDist, uPersonDepth;
uniform vec3 uRo, uSun, uTint;
uniform mat3 uCam;

layout(location=0) out vec4 oCol;
layout(location=1) out vec4 oMeta;

float smin(float a, float b, float k) {
  float h = max(k - abs(a - b), 0.0) / max(k, 1e-5);
  return min(a, b) - h * h * k * 0.25;
}

float sdDrop(vec3 p) {
  vec3 q = p - vCenter;
  vec3 ax = vAxis;
  vec3 side = normalize(vLobeDir.xyz - ax * dot(vLobeDir.xyz, ax) + vec3(1e-4, 2e-4, 3e-4));

  float along = dot(q, ax);
  float uLen = clamp(along / max(vRad * vAsp, 1e-4), -1.4, 1.4);

  vec3 qb = q - side * (vShape.w * vRad * uLen * uLen);
  float bAlong = dot(qb, ax);
  vec3 perp = qb - ax * bAlong;

  float taper = 1.0 + 0.26 * clamp(bAlong / max(vRad * vAsp, 1e-4), -1.0, 1.0);
  vec3 qs = perp / max(taper, 0.52) + ax * (bAlong / max(vAsp, 1.0));
  float d = (length(qs) - vRad) * min(1.0, 0.72 + 0.28 / max(vAsp, 1.0));

  if (vShape.z > 0.05) {
    float lOff = vRad * (vAsp * 0.78 + 0.22);
    vec3 c2 = ax * (-lOff * 0.78) + side * (vRad * 0.42 * vShape.z);
    float r2 = vRad * (0.36 + 0.46 * vShape.z);
    d = smin(d, length(q - c2) - r2, vRad * 0.52);

    if (vLobeDir.w > 1.5) {
      vec3 ortho = normalize(cross(ax, side));
      vec3 c3 = ax * (lOff * 0.56) + ortho * (vRad * 0.48) - side * (vRad * 0.22);
      float r3 = vRad * (0.30 + 0.30 * vShape.z);
      d = smin(d, length(q - c3) - r3, vRad * 0.48);
    }
  }

  float ph = vShape.x * 23.7;
  vec3 uq = q / max(vRad, 1e-4);
  float wob = sin(uq.x * 2.7 + ph) * cos(uq.y * 2.5 - ph * 0.63) * sin(uq.z * 2.6 + ph * 1.19);
  float wob2 = cos(uq.x * 4.3 - ph * 1.4) * sin(uq.y * 4.1 + ph * 0.8);
  return d + (wob * 0.75 + wob2 * 0.25) * vRad * vShape.y;
}

vec3 dropNormal(vec3 p) {
  float h = max(vRad * 0.016, 0.00018);
  vec2 e = vec2(h, -h);
  return normalize(
    e.xyy * sdDrop(p + e.xyy) +
    e.yyx * sdDrop(p + e.yyx) +
    e.yxy * sdDrop(p + e.yxy) +
    e.xxx * sdDrop(p + e.xxx)
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
  vec2 dep = texture(uDepth, gl_FragCoord.xy / uRes).rg;
  float heroD = dep.r;   /* min(tCore, tCylOcc) */
  float effOcc = dep.g;  /* min(tCylOcc, personDepth if inside person mask) */
  float pMask = texture(uPersonMask, clamp(uvTop, vec2(0.002), vec2(0.998))).r;

  vec3 acc = vec3(0.0);
  float cov = 0.0;
  float minT = 1e4;
  vec2 offs[4] = vec2[4](vec2(-0.25, -0.25), vec2(0.25, -0.25), vec2(-0.25, 0.25), vec2(0.25, 0.25));

  /* Subway station architectural lights: overhead bright white tube strips + platform right glow + left ambient fill */
  vec3 L_top   = normalize(vec3( 0.08,  0.15, 0.98));
  vec3 L_right = normalize(vec3( 0.72,  0.25, 0.65));
  vec3 L_left  = normalize(vec3(-0.62, -0.25, 0.74));
  bool isMicro = (vLobeDir.w < 0.5);

  for (int k = 0; k < 4; k++) {
    vec2 px = px0 + offs[k];
    vec3 rd = normalize(uCam * vec3((px - 0.5 * uRes) / uF, 1.0));
    vec3 oc = vCenter - uRo;
    float b = dot(oc, rd);
    float c = dot(oc, oc) - vBoundR * vBoundR;
    float disc = b * b - c;
    if (disc < 0.0) continue;
    float sq = sqrt(disc);
    float tStart = max(b - sq, 0.03);
    float tEnd = min(b + sq, heroD);
    if (tStart >= tEnd) continue;

    float t = tStart;
    float tHit = -1.0;
    float eps = max(vRad * 0.007, 0.00012);
    int maxSteps = isMicro ? 26 : 40;
    for (int s = 0; s < 40; s++) {
      if (s >= maxSteps) break;
      float d = sdDrop(uRo + rd * t);
      if (d < eps) { tHit = t; break; }
      t += d * 0.88;
      if (t > tEnd) break;
    }
    if (tHit < 0.0) continue;

    /* MediaPipe person occlusion: droplets BEHIND person torso depth are cleanly masked out! */
    float visP = (tHit > uPersonDepth + 0.04) ? (1.0 - smoothstep(0.22, 0.74, pMask)) : 1.0;
    if (visP < 0.02) continue;
    minT = min(minT, tHit);

    vec3 p1 = uRo + rd * tHit;
    vec3 n1 = dropNormal(p1);
    vec3 rdIn = refract(rd, n1, 1.0 / uIor);
    if (length(rdIn) < 0.001) rdIn = reflect(rd, n1);

    float thick = vRad * 0.35;
    vec3 p2 = p1 + rdIn * max(vRad * 0.24, 0.001);
    int inSteps = isMicro ? 10 : 18;
    for (int s = 0; s < 18; s++) {
      if (s >= inSteps) break;
      float dIn = -sdDrop(p2);
      if (dIn < eps * 1.8) break;
      float st = max(dIn * 0.88, vRad * 0.08);
      p2 += rdIn * st;
      thick += st;
      if (thick > vRad * 5.2) break;
    }
    vec3 n2 = -dropNormal(p2);
    vec3 rdOutG = refract(rdIn, n2, uIor);
    if (length(rdOutG) < 0.001) rdOutG = normalize(mix(rdIn, reflect(rdIn, n2), 0.38));

    vec3 bend = rdOutG - rd;
    vec3 rdOutR = normalize(rd + bend * ((uIor - uDisp) - 1.0) / max(uIor - 1.0, 0.05));
    vec3 rdOutB = normalize(rd + bend * ((uIor + uDisp) - 1.0) / max(uIor - 1.0, 0.05));

    vec3 refrCol = vec3(
      sampleEnv(p2, rdOutR).r,
      sampleEnv(p2, rdOutG).g,
      sampleEnv(p2, rdOutB).b
    );
    float ndv = max(dot(-rd, n1), 0.0);
    float innerMeniscus = 1.0 - 0.18 * exp(-pow((ndv - 0.36) / 0.16, 2.0));
    refrCol *= exp(-thick * uTint * 0.35) * 1.18 * innerMeniscus;

    float fresnel = 0.035 + 0.965 * pow(1.0 - ndv, 4.0);
    vec3 reflDir = reflect(rd, n1);
    vec3 envRefl = sampleEnv(p1 + n1 * 0.004, reflDir);
    float upGlow = smoothstep(-0.15, 0.85, reflDir.z);
    /* Bright overhead linear station LED strip reflections */
    float tubeRefl = pow(max(dot(reflDir, L_top), 0.0), 7.0);
    envRefl = mix(envRefl * 1.06 + vec3(0.05), envRefl * 1.24 + vec3(0.24, 0.27, 0.32), upGlow * 0.54)
            + vec3(0.38, 0.40, 0.44) * tubeRefl;

    float spTop   = pow(max(dot(n1, normalize(L_top   - rd)), 0.0), isMicro ? 130.0 : 230.0) * (isMicro ? 2.65 : 2.25)
                  + pow(max(dot(n1, normalize(L_top   - rd)), 0.0), 26.0) * 0.14;
    float spRight = pow(max(dot(n1, normalize(L_right - rd)), 0.0), isMicro ? 115.0 : 185.0) * (isMicro ? 1.45 : 1.15);
    float spLeft  = pow(max(dot(n1, normalize(L_left  - rd)), 0.0), isMicro ? 125.0 : 195.0) * 0.88;
    float backGlint = pow(max(dot(-n2, L_top), 0.0), 5.0) * (0.24 + 0.24 * (1.0 - ndv));
    float rim = pow(1.0 - ndv, 2.8) * (0.38 + 0.32 * upGlow);

    vec3 col = mix(refrCol, envRefl, clamp(fresnel * 0.46, 0.0, 0.76));
    col += vec3(0.95, 0.98, 1.00) * (spTop + backGlint);
    col += vec3(1.00, 0.97, 0.90) * spRight;
    col += vec3(0.92, 0.94, 1.00) * spLeft;
    col += vec3(0.96, 0.98, 1.00) * rim;

    acc += col * visP;
    cov += visP;
  }

  if (cov < 0.05) discard;
  float a = clamp(cov * 0.25, 0.0, 1.0);
  oCol = vec4(clamp(acc / cov, 0.0, 1.0), a);
  oMeta = vec4(1.0, 1.0 / max(minT, 0.20), 1.0 / max(effOcc, 0.20), a);
}
"""

# Pass 3: Camera Optical Finish (Person-Locked Thin-Lens Bokeh + OLPF Softness + Halation + Edge CA + Sensor Grain)
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
    if (lum > 0.78) {
      float hw = exp(-r * r / 26.0) * (lum - 0.78) * mj.r;
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
    res = mix(res, haloRgb, fgAlpha * (1.0 - cgA0 * 0.55));
  }

  float cgPresence = clamp(max(cgA0, fgAlpha), 0.0, 1.0);
  res += halation * vec3(0.98, 0.99, 1.02) * 0.052;

  if (cgPresence > 0.005) {
    vec3 graded = res * (1.0 - uBlackFloor) + vec3(uBlackFloor * 0.96, uBlackFloor * 0.97, uBlackFloor * 1.04);
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


def encode(c, out_dir="out", title_label="3D Liquid Glass — Human Frozen Splash"):
    out = c["dir"] / out_dir
    out.mkdir(exist_ok=True)
    nm = c["name"]
    r = f"{c['fps_num']}/{c['fps_den']}"
    ff = find_ffmpeg()
    tags = ["-pix_fmt", "yuv420p", "-color_primaries", "bt709", "-color_trc", "bt709", "-colorspace", "bt709", "-movflags", "+faststart"]
    x264 = ["-c:v", "libx264", "-profile:v", "high", "-level", "4.1", "-crf", "14", "-preset", "slow"]
    run = lambda a: subprocess.run([ff, "-v", "error", "-y"] + a, check=True)
    run(["-framerate", r, "-i", str(work(c, "comp", "%04d.png"))] + x264 + tags + [str(out / f"{nm}_Composite.mp4")])
    lab = lambda t: f"drawtext=fontfile='C\\:/Windows/Fonts/arialbd.ttf':text='{t}':x=w-tw-28:y=h-th-24:fontsize=40:fontcolor=white:shadowcolor=black@0.6:shadowx=2:shadowy=2"
    run([
        "-framerate", r, "-i", str(work(c, "comp", "%04d.png")),
        "-framerate", r, "-i", str(work(c, "frames", "%04d.jpg")),
        "-filter_complex",
        f"[0]scale=1080:608,{lab(title_label)}[a];[1]scale=1080:608,{lab('Before')}[b];[a][b]vstack,pad=1080:1350:0:67:color=black",
    ] + x264 + tags + [str(out / f"{nm}_Compare_1080x1350.mp4")])
    print("encoded ->", out)


def build_human_frozen_cloud(cam_path, cam_rot, torso_origin, head_pos, cyl_occ, n_hero=36, n_mid=184, n_micro=820, seed=2027):
    """Build ~1,040 irregular liquid-glass droplets exploding 360-deg around the person's 3D body & freezing in mid-air!

    Coordinate basis around person:
      right_u: camera-right horizontal axis at mid-frame (spans across the subway platform!)
      fwd_u:   camera-forward horizontal axis (separates front-of-person vs behind-person depth!)
      up_u:    +Z
    """
    rng = np.random.default_rng(seed)
    origin = np.asarray(torso_origin, np.float64)
    head = np.asarray(head_pos, np.float64)

    # Derive natural screen-aligned horizontal basis at mid-frame so droplets span the whole 16:9 frame beautifully!
    mid = len(cam_path) // 2
    fwd = origin - cam_path[mid]
    fwd[2] = 0.0
    fwd /= max(np.linalg.norm(fwd), 1e-4)
    right = np.array([fwd[1], -fwd[0], 0.0], np.float64)
    right /= max(np.linalg.norm(right), 1e-4)
    up = np.array([0.0, 0.0, 1.0], np.float64)

    cyl_xy = np.array([cyl_occ[0], cyl_occ[1]], np.float64) if cyl_occ[2] > 0.01 else None
    cyl_r = float(cyl_occ[2])

    def ensure_clearance(pos, rad, asp, lobe_scale):
        p = np.array(pos, np.float64)
        p[2] = float(np.clip(p[2], 0.16, 2.75))
        bound_r = rad * (max(asp, 1.0) * 1.4 + (0.65 if lobe_scale > 0.05 else 0.0) + 0.35)

        # 1. Keep outside person's body cylinder (r=0.34m around torso XY, z < head.z + 0.15) so droplets wrap AROUND him!
        d_xy = p[:2] - origin[:2]
        r_xy = float(np.linalg.norm(d_xy))
        min_body = 0.36 + bound_r * 1.15
        if p[2] < head[2] + 0.22 and r_xy < min_body:
            u = d_xy / r_xy if r_xy > 1e-4 else np.array([math.cos(rng.uniform(0, 6.28)), math.sin(rng.uniform(0, 6.28))])
            p[:2] = origin[:2] + u * (min_body + 0.04)

        # 2. Keep out of cylindrical concrete pillar volume if active
        if cyl_xy is not None:
            dc = p[:2] - cyl_xy
            rc = float(np.linalg.norm(dc))
            if rc < cyl_r + bound_r + 0.08:
                u = dc / max(rc, 1e-4)
                p[:2] = cyl_xy + u * (cyl_r + bound_r + 0.12)

        # 3. Strict clearance from full camera gimbal arc so close foreground drops get silky bokeh without ever clipping!
        min_cam = 0.22 + bound_r * 1.65
        diffs = p[None, :] - cam_path
        dists = np.linalg.norm(diffs, axis=1)
        idx = int(np.argmin(dists))
        if dists[idx] < min_cam:
            v = p - cam_path[idx]
            nrm = float(np.linalg.norm(v))
            v = v / nrm if nrm > 1e-4 else fwd
            p = cam_path[idx] + v * (min_cam + 0.05)
            p[2] = float(np.clip(p[2], 0.16, 2.75))
        return p

    # 20 asymmetric 3D radial splash tendrils exploding 360-deg outward around the man
    # Biased slightly wider along platform-right (+right) so the full 16:9 subway frame fills dynamically!
    tendrils = []
    for k in range(20):
        u = (k + 0.5) / 20.0
        phi = math.acos(1.0 - 2.0 * u) + rng.normal(0.0, 0.16)
        theta = k * 2.3999632 + rng.normal(0.0, 0.20)
        # Local coordinates (r_right, r_fwd, r_up)
        lr = math.sin(phi) * math.cos(theta) * 1.45 + 0.22   # fill right side of 16:9 frame gracefully
        lf = math.sin(phi) * math.sin(theta) * 1.05
        lu = math.cos(phi) * 0.72 + 0.10
        d = right * lr + fwd * lf + up * lu
        d /= max(np.linalg.norm(d), 1e-4)
        reach = float(rng.uniform(1.15, 2.75))
        tendrils.append((d, reach))

    drops = []

    def add_drop(p1_raw, rad, dir_hint, tier, asp_range, lobe_prob, bend_max):
        lobe_scale = float(rng.uniform(0.34, 0.92)) if rng.random() < lobe_prob else 0.0
        asp = float(rng.uniform(*asp_range))
        p1 = ensure_clearance(p1_raw, rad, asp, lobe_scale)

        # Spawn from a 3D shell tightly wrapping the person's torso/shoulders (r ~ 0.36..0.52m around origin!)
        d_out = p1 - origin
        d_len = max(float(np.linalg.norm(d_out)), 1e-4)
        u_out = d_out / d_len
        p0 = origin + u_out * float(rng.uniform(0.32, 0.50))
        p0[2] = float(np.clip(origin[2] + rng.uniform(-0.35, 0.42), 0.35, head[2] + 0.18))

        ax = dir_hint + rng.normal(0.0, 0.26, 3)
        ax /= max(float(np.linalg.norm(ax)), 1e-4)
        lobe_dir = rng.normal(0.0, 1.0, 3)
        lobe_dir /= max(float(np.linalg.norm(lobe_dir)), 1e-4)

        wob_amp = float(rng.uniform(0.055, 0.145) if tier >= 1 else rng.uniform(0.035, 0.105))
        bend = float(rng.uniform(-bend_max, bend_max))
        droop = float(rng.uniform(0.04, 0.17) * min(d_len / 2.0, 1.2))
        delay = float(rng.uniform(0.0, 0.05))

        drops.append({
            "p0": p0, "p1": p1, "droop": droop, "rad": rad, "axis": ax, "asp": asp,
            "wob_seed": float(rng.uniform(0.0, 12.0)), "wob_amp": wob_amp,
            "lobe_scale": lobe_scale, "bend": bend, "lobe_dir": lobe_dir,
            "tier": float(tier), "delay": delay,
        })

    # 1. HERO IRREGULAR SPLASH CROWNS & STRETCHED FILAMENTS (36 pieces)
    for k in range(n_hero):
        td, reach = tendrils[k % len(tendrils)]
        frac = float(rng.uniform(0.32, 0.96)) if k < 20 else float(rng.uniform(0.22, 0.65))
        p1_raw = origin + td * (reach * frac) + rng.normal(0.0, 0.14, 3)
        rad = float(np.exp(rng.uniform(math.log(0.032), math.log(0.072))))
        add_drop(p1_raw, rad, td, tier=2, asp_range=(1.35, 2.35), lobe_prob=0.82, bend_max=0.38)

    # 2. MID-SCALE IRREGULAR ORGANIC DROPLETS (184 pieces: front, sides, behind person!)
    for k in range(n_mid):
        if k < int(n_mid * 0.65):
            td, reach = tendrils[k % len(tendrils)]
            frac = float(rng.uniform(0.20, 1.06))
            p1_raw = origin + td * (reach * frac) + rng.normal(0.0, 0.08 + 0.22 * frac, 3)
            dir_h = td
        else:
            # 3D cloud wrapping all around him & stretching right across the station platform
            lr = float(rng.uniform(-1.35, 2.65))
            lf = float(rng.uniform(-1.95, 1.65))  # negative = IN FRONT of man toward camera; positive = BEHIND man!
            lu = float(rng.uniform(-0.92, 1.35))
            p1_raw = origin + right * lr + fwd * lf + up * lu
            dir_h = p1_raw - origin
            dir_h /= max(float(np.linalg.norm(dir_h)), 1e-4)
        rad = float(np.exp(rng.uniform(math.log(0.012), math.log(0.032))))
        add_drop(p1_raw, rad, dir_h, tier=1, asp_range=(1.15, 1.95), lobe_prob=0.52, bend_max=0.26)

    # 3. ULTRA-FINE "자잘자잘 송골송골" DEW MICRO-BEADS (820 pieces!)
    for k in range(n_micro):
        mode = rng.random()
        if mode < 0.45:
            td, reach = tendrils[k % len(tendrils)]
            frac = float(rng.uniform(0.14, 1.20))
            p1_raw = origin + td * (reach * frac) + rng.normal(0.0, 0.06 + 0.34 * frac, 3)
            dir_h = td
        elif mode < 0.78:
            # Dense glittering 3D rain-stop halo spanning left-to-right & front-to-back around the man
            lr = float(rng.uniform(-1.55, 2.95))
            lf = float(rng.uniform(-2.15, 1.75))
            lu = float(rng.uniform(-0.98, 1.45))
            p1_raw = origin + right * lr + fwd * lf + up * lu
            dir_h = p1_raw - origin
            dir_h /= max(float(np.linalg.norm(dir_h)), 1e-4)
        else:
            # Close-foreground bokeh droplets hovering 0.7..2.0m in front of the orbiting camera for dramatic 3D parallax!
            cam_mid = cam_path[int(rng.integers(0, len(cam_path)))]
            to_man = origin - cam_mid
            dist_m = float(np.linalg.norm(to_man))
            u_to = to_man / max(dist_m, 1e-4)
            # Bias slightly off-center (left/right/top/bottom) so foreground bokeh frames the man without blocking his face
            lat_r = float(rng.uniform(-0.85, 1.55))
            if abs(lat_r) < 0.28:
                lat_r += 0.42 if rng.random() < 0.65 else -0.38
            p1_raw = cam_mid + u_to * float(rng.uniform(0.68, min(2.05, dist_m * 0.52))) + right * lat_r + up * float(rng.uniform(-0.72, 0.62))
            dir_h = p1_raw - origin
            dir_h /= max(float(np.linalg.norm(dir_h)), 1e-4)
        rad = float(np.exp(rng.uniform(math.log(0.0032), math.log(0.0135))))
        add_drop(p1_raw, rad, dir_h, tier=0, asp_range=(1.06, 1.90), lobe_prob=0.30, bend_max=0.18)

    # 4. 24 dedicated creamy foreground bokeh hero/mid droplets (0.85..1.95m from lens) for strong 3D camera-arc parallax!
    cam_c = cam_path[mid]
    u_to_c = (origin - cam_c)
    u_to_c /= max(float(np.linalg.norm(u_to_c)), 1e-4)
    for k in range(24):
        d_fg = float(rng.uniform(0.85, 1.95))
        lat_r = float(rng.uniform(-0.95, 1.45))
        lat_u = float(rng.uniform(-0.68, 0.55))
        # Keep clear of the man's face sightline
        if -0.25 < lat_r < 0.22 and -0.10 < lat_u < 0.35:
            lat_r = 0.45 if rng.random() < 0.65 else -0.45
        p1_raw = cam_c + u_to_c * d_fg + right * lat_r + up * lat_u
        dir_h = p1_raw - origin
        dir_h /= max(float(np.linalg.norm(dir_h)), 1e-4)
        rad = float(np.exp(rng.uniform(math.log(0.010), math.log(0.028))))
        add_drop(p1_raw, rad, dir_h, tier=1, asp_range=(1.15, 1.85), lobe_prob=0.45, bend_max=0.22)

    min_d = min(float(np.linalg.norm(cam_path - d["p1"][None, :], axis=1).min()) for d in drops)
    print(f"Human frozen splash: {n_hero} hero + {n_mid+24} mid + {n_micro} micro = {len(drops)} droplets (min_cam_dist={min_d:.2f}m)")
    return drops


def main():
    c = load(sys.argv[1] if len(sys.argv) > 1 and not sys.argv[1].startswith("--") else here / "piece_human")
    S = scene(c)
    L = np.load(work(c, "limbs.npz"))
    P_limbs = np.asarray(L["P"], np.float64)        # (N, 33, 3)
    person_depths = np.asarray(L["depth"], np.float64)  # (N,)
    W, H, N, fps = c["W"], c["H"], c["N"], c["fps"]
    f = float(S["f"])
    cfg = c.get("frozen_human", {})
    stills = [int(x) for x in sys.argv[sys.argv.index("--still") + 1].split(",")] if "--still" in sys.argv else None
    out_dir = sys.argv[sys.argv.index("--out") + 1] if "--out" in sys.argv else "out"

    cam_path = np.asarray(S["cam"], np.float64)
    cam_rot = np.asarray(S["rot"], np.float64)
    burst_frame = int(cfg.get("burst_frame", 20))
    freeze_dur_s = float(cfg.get("freeze_dur_s", 1.25))

    # Torso center (average of shoulders 11,12 and hips 23,24) and head nose (0) at burst frame!
    torso_traj = 0.25 * (P_limbs[:, 11] + P_limbs[:, 12] + P_limbs[:, 23] + P_limbs[:, 24])
    origin = torso_traj[burst_frame]
    head_pos = P_limbs[burst_frame, 0]
    print(f"Person torso at burst f{burst_frame}: {np.round(origin, 2)} (head z={head_pos[2]:.2f}m, depth={person_depths[burst_frame]:.2f}m)")

    # Optional cylindrical column holdout [cx, cy, radius, maxZ] from piece.json (0 radius = disabled)
    cyl_occ = tuple(map(float, cfg.get("cyl_occ", [0.0, 0.0, 0.0, 4.5])))

    drops = build_human_frozen_cloud(
        cam_path, cam_rot, origin, head_pos, cyl_occ,
        n_hero=int(cfg.get("n_hero", 36)),
        n_mid=int(cfg.get("n_mid", 184)),
        n_micro=int(cfg.get("n_micro", 820)),
        seed=int(cfg.get("seed", 2027)),
    )
    caust_idx = sorted(range(len(drops)), key=lambda idx: -drops[idx]["rad"])[:28]

    ctx = moderngl.create_standalone_context(require=330)
    core_prog = ctx.program(vertex_shader=VS, fragment_shader=CORE_FS)
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
    va_core = ctx.vertex_array(core_prog, [(tri, "2f", "p")])
    va_copy2 = ctx.vertex_array(copy2_prog, [(tri, "2f", "p")])
    va_post = ctx.vertex_array(post_prog, [(tri, "2f", "p")])

    quad_vbo = ctx.buffer(np.array([-1, -1, 1, -1, 1, 1, -1, -1, 1, 1, -1, 1], "f4").tobytes())
    inst_buf = ctx.buffer(reserve=len(drops) * 16 * 4)
    va_drop = ctx.vertex_array(drop_prog, [
        (quad_vbo, "2f", "inQuad"),
        (inst_buf, "4f 4f 4f 4f/i", "inPosRad", "inAxisAsp", "inShape", "inLobeDir"),
    ])

    plate_t = ctx.texture((W, H), 3)
    plate_t.filter = (moderngl.LINEAR, moderngl.LINEAR)
    plate_t.repeat_x = plate_t.repeat_y = False

    mask_t = ctx.texture((W, H), 1)
    mask_t.filter = (moderngl.LINEAR, moderngl.LINEAR)
    mask_t.repeat_x = mask_t.repeat_y = False

    t_hero = ctx.texture((W, H), 4, dtype="f4")
    t_hero.filter = (moderngl.LINEAR, moderngl.LINEAR)
    t_hero.repeat_x = t_hero.repeat_y = False
    t_depth = ctx.texture((W, H), 4, dtype="f4")
    t_depth.filter = (moderngl.NEAREST, moderngl.NEAREST)
    t_meta0 = ctx.texture((W, H), 4, dtype="f4")
    t_meta0.filter = (moderngl.NEAREST, moderngl.NEAREST)
    fbo_hero = ctx.framebuffer([t_hero, t_depth, t_meta0])

    t_scene = ctx.texture((W, H), 4, dtype="f4")
    t_scene.filter = (moderngl.LINEAR, moderngl.LINEAR)
    t_scene.repeat_x = t_scene.repeat_y = False
    t_meta = ctx.texture((W, H), 4, dtype="f4")
    t_meta.filter = (moderngl.NEAREST, moderngl.NEAREST)
    fbo_scene = ctx.framebuffer([t_scene, t_meta])

    t_final = ctx.texture((W, H), 4, dtype="f4")
    fbo_final = ctx.framebuffer([t_final])

    ior = float(cfg.get("ior", 1.44))
    disp = float(cfg.get("dispersion", 0.016))
    tint = tuple(map(float, cfg.get("tint", [0.040, 0.018, 0.014])))
    proxy_dist = float(cfg.get("proxy_dist", 5.8))

    copy2_prog["uRes"].value = (float(W), float(H))

    core_prog["uRes"].value = (float(W), float(H))
    core_prog["uF"].value = f
    core_prog["uSun"].value = tuple(map(float, S["sun"]))
    core_prog["uCylOcc"].value = cyl_occ
    core_prog["uIor"].value = ior
    core_prog["uDisp"].value = disp
    core_prog["uTint"].value = tint
    core_prog["uCausticGain"].value = float(cfg.get("caustic_gain", 0.85))
    core_prog["uProxyDist"].value = proxy_dist

    drop_prog["uRes"].value = (float(W), float(H))
    drop_prog["uF"].value = f
    if "uSun" in drop_prog:
        drop_prog["uSun"].value = tuple(map(float, S["sun"]))
    drop_prog["uIor"].value = ior
    drop_prog["uDisp"].value = disp
    drop_prog["uTint"].value = tint
    drop_prog["uProxyDist"].value = proxy_dist

    post_prog["uRes"].value = (float(W), float(H))
    post_prog["uNearBlur"].value = 4.6
    post_prog["uFarBlur"].value = 4.8
    post_prog["uBaseSoft"].value = 0.68
    post_prog["uMaxCoc"].value = 11.0
    post_prog["uGrainAmt"].value = 0.013
    post_prog["uBlackFloor"].value = 0.025

    def eval_drop_state(d, age_s):
        a = age_s - d["delay"]
        if a <= 0.0:
            return None
        x = min(max(a / freeze_dur_s, 0.0), 1.0)
        w = 1.0 - (1.0 - x) ** 3.8
        p = d["p0"] + (d["p1"] - d["p0"]) * w
        p = p.copy()
        p[2] += d["droop"] * (w * (1.0 - 0.55 * w))
        grow = min(1.0, 0.22 + 0.78 * min(a / 0.09, 1.0) ** 0.75)
        rad = d["rad"] * grow
        asp = d["asp"] * (1.0 + 0.28 * (1.0 - x) ** 2.4)
        return p, rad, asp

    def render_one(i):
        pl = frame(c, i)
        m_img = cv2.imread(str(work(c, "mask", f"{i:04d}.png")), cv2.IMREAD_GRAYSCALE)
        if m_img is None:
            m_img = np.zeros((H, W), np.uint8)
        else:
            # Tiny 1.5px feather on MediaPipe matte for ultra-clean optical edge integration around coat/hair!
            m_img = cv2.GaussianBlur(m_img, (0, 0), 1.0)

        t_sec = i / fps
        ro = cam_path[i]
        R = cam_rot[i]
        Rt = tuple(R.astype(np.float32).T.flatten())
        age_s = (i - burst_frame) / fps

        # Exact camera-space ray distance to the person's torso at frame i
        t_cur = torso_traj[i]
        p_depth = float(np.linalg.norm(t_cur - ro))

        # Pre-burst shimmering fluid ring coalescing tightly around person's torso (0..19), shattering at f=20..26
        if age_s < 0.0:
            u_pre = i / max(float(burst_frame), 1.0)
            ring_scale = 0.45 + 0.65 * (u_pre ** 1.4)
            ring_rad = 0.54 - 0.12 * (u_pre ** 2.0)  # tightens right before '팡!' burst!
            sim_t = t_sec
        elif age_s < 0.22:
            u_sh = age_s / 0.22
            ring_scale = 1.10 * max(0.0, 1.0 - u_sh ** 1.2)
            ring_rad = 0.42 + 0.45 * u_sh
            sim_t = burst_frame / fps + age_s * 1.4
        else:
            ring_scale = 0.0
            ring_rad = 0.45
            sim_t = burst_frame / fps + 0.30

        caust_list = []
        rows_d = []
        if age_s > 0.0:
            for idx, d in enumerate(drops):
                st = eval_drop_state(d, age_s)
                if st is None:
                    continue
                p_now, rad_now, asp_now = st
                if idx in caust_idx:
                    caust_list.append((float(p_now[0]), float(p_now[1]), float(p_now[2]), float(rad_now)))
                z_cam = float(((p_now - ro) @ R)[2])
                if z_cam > 0.06:
                    rows_d.append((z_cam, [
                        float(p_now[0]), float(p_now[1]), float(p_now[2]), float(rad_now),
                        float(d["axis"][0]), float(d["axis"][1]), float(d["axis"][2]), float(asp_now),
                        d["wob_seed"], d["wob_amp"], d["lobe_scale"], d["bend"],
                        float(d["lobe_dir"][0]), float(d["lobe_dir"][1]), float(d["lobe_dir"][2]), d["tier"],
                    ]))

        n_caust = len(caust_list)
        while len(caust_list) < 28:
            caust_list.append((0.0, 0.0, 0.0, 0.0))

        plate_t.write(pl[:, :, ::-1].tobytes())
        mask_t.write(m_img.tobytes())

        # Pass 1: Pre-burst ring + Floor caustics + Column/Person depth
        plate_t.use(0)
        mask_t.use(1)
        core_prog["uPlate"].value = 0
        core_prog["uPersonMask"].value = 1
        core_prog["uRo"].value = tuple(map(float, ro))
        core_prog["uCam"].value = Rt
        core_prog["uSimT"].value = float(sim_t)
        core_prog["uRing"].value = (float(t_cur[0]), float(t_cur[1]), float(t_cur[2]), float(ring_scale))
        core_prog["uRingRad"].value = float(ring_rad)
        core_prog["uPersonDepth"].value = float(p_depth)
        core_prog["uCaustDrops"].value = caust_list[:28]
        core_prog["uNumCaust"].value = int(n_caust)

        fbo_hero.use()
        ctx.disable(moderngl.BLEND)
        va_core.render(moderngl.TRIANGLES)

        fbo_scene.use()
        t_hero.use(0)
        t_meta0.use(1)
        copy2_prog["uSrc0"].value = 0
        copy2_prog["uSrc1"].value = 1
        va_copy2.render(moderngl.TRIANGLES)

        # Pass 2: Back-to-front depth-sorted 3D droplets occluded cleanly behind person matte & column!
        rows_d.sort(key=lambda x: -x[0])
        if rows_d:
            data = np.array([r[1] for r in rows_d], dtype=np.float32)
            inst_buf.write(data.tobytes())
            t_hero.use(0)
            t_depth.use(1)
            mask_t.use(2)
            drop_prog["uBase"].value = 0
            drop_prog["uDepth"].value = 1
            drop_prog["uPersonMask"].value = 2
            drop_prog["uPersonDepth"].value = float(p_depth)
            drop_prog["uRo"].value = tuple(map(float, ro))
            drop_prog["uCam"].value = Rt
            ctx.enable(moderngl.BLEND)
            ctx.blend_func = (moderngl.SRC_ALPHA, moderngl.ONE_MINUS_SRC_ALPHA)
            va_drop.render(moderngl.TRIANGLES, instances=len(rows_d))
            ctx.disable(moderngl.BLEND)

        # Pass 3: Camera optical finish with focus dynamically locked onto the person!
        fbo_final.use()
        t_scene.use(0)
        t_meta.use(1)
        post_prog["uScene"].value = 0
        post_prog["uMeta"].value = 1
        post_prog["uFocusDist"].value = float(p_depth)
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
            rows.append(np.hstack([t_im(pl, f"plate {i}"), t_im(comp, f"human frozen splash {i}")]))
        out_p = work(c, "still_human_sheet.jpg")
        cv2.imwrite(str(out_p), np.vstack(rows), [cv2.IMWRITE_JPEG_QUALITY, 92])
        if hero is not None:
            cv2.imwrite(str(work(c, "still_human_hero.jpg")), hero, [cv2.IMWRITE_JPEG_QUALITY, 94])
        print("stills ->", out_p)
        return

    comp_dir = work(c, "comp")
    shutil.rmtree(comp_dir, ignore_errors=True)
    comp_dir.mkdir(parents=True)
    ts = time.time()
    for i in range(N):
        _, comp = render_one(i)
        cv2.imwrite(str(comp_dir / f"{i:04d}.png"), comp, [cv2.IMWRITE_PNG_COMPRESSION, 1])
        if i % 40 == 0 or i == N - 1:
            print(f"human frozen frame {i}/{N} ({time.time() - ts:.1f}s)", flush=True)
    encode(c, out_dir=out_dir)


if __name__ == "__main__":
    main()
