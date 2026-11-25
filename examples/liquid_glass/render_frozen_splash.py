"""Frozen-Time Liquid Glass Splash with Real 3D Architectural Occlusion & Camera Optical Finish (ModernGL, OpenGL 3.3).

    python render_frozen_splash.py [piece_dir] --still 8,22,58,145,215   quick look -> work/still_frozen_v2_sheet.jpg
    python render_frozen_splash.py [piece_dir] --out out_v2              full render -> out_v2/*_Composite.mp4, *_Compare_1080x1350.mp4

3-Pass Pro Compositing Pipeline:
- Pass 1 (CORE_FS): Pre-burst fracturing liquid-glass core + 3D floor caustics + 3 COLMAP-verified 3D architectural
  holdout boxes (foreground left doorframe + left/right hallway jamb walls).
- Pass 2 (DROP_VS/DROP_FS): ~980 instanced 3D raymarched irregular droplets (30 hero + 160 mid + 790 fine dewy micro-beads),
  bursting at frame 14 ("팡!") and freezing 100% dead-still in 3D space at frame 52, strictly occluded by real walls/jambs.
- Pass 3 (POST_FS): Physical thin-lens depth-of-field bokeh (creamy near-lens defocus on close droplets + OLPF base softness
  + far softness) respecting wall occlusion depth, subtle specular halation, edge radial CA, filmic black-floor lift,
  and temporally animated plate-matched camera sensor grain (applied exclusively to CG regions; background plate stays untouched).
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

# Pass 1: Pre-burst fracturing core + 3D floor caustics + 3D architectural holdout boxes
CORE_FS = """#version 330
uniform sampler2D uPlate;
uniform vec2 uRes;
uniform float uF, uSimT;
uniform vec3 uRo, uSun;
uniform mat3 uCam;
uniform vec4 uCore;            /* xyz = burst origin, w = core radius scale (0 once shattered) */
uniform vec4 uCaustDrops[28];  /* xyz = world pos of largest droplets, w = radius (m) */
uniform int uNumCaust;
uniform vec3 uOccMin[4];       /* 3D architectural holdout boxes (doorframe + left/right hall walls) */
uniform vec3 uOccMax[4];
uniform int uNumOcc;
uniform float uIor, uDisp, uCausticGain, uProxyDist;
uniform vec3 uTint;

layout(location=0) out vec4 oComp;
layout(location=1) out vec4 oDepth;   /* R: min(tCore, tOcc), G: tOcc */
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

float hitOccluders(vec3 ro, vec3 rd) {
  float tBest = 1e4;
  vec3 invD = 1.0 / mix(rd, vec3(1e-8), lessThan(abs(rd), vec3(1e-8)));
  for (int i = 0; i < 4; i++) {
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

float sdCore(vec3 p) {
  if (uCore.w <= 0.005) return 1e4;
  vec3 q = (p - uCore.xyz) / uCore.w;
  float t = uSimT * 4.5;
  float d = length(q) - 0.62;
  d = smin(d, length(q - vec3( 0.42 * cos(t),       0.32 * sin(t * 1.3),  0.22 * sin(t * 1.7))) - 0.36, 0.36);
  d = smin(d, length(q - vec3(-0.38 * sin(t * 1.1), 0.28 * cos(t * 1.5), -0.30 * cos(t * 1.2))) - 0.34, 0.36);
  d = smin(d, length(q - vec3( 0.20 * sin(t * 1.9),-0.40 * cos(t * 1.4),  0.34 * cos(t * 1.6))) - 0.32, 0.34);
  d = smin(d, length(q - vec3(-0.30 * cos(t * 1.6),-0.26 * sin(t * 1.8), -0.25 * sin(t * 1.1))) - 0.30, 0.34);
  float rip = (vnoise(q * 3.8 + vec3(0.0, t * 0.7, -t * 0.6)) - 0.5) * 0.065;
  return (d + rip) * uCore.w;
}

vec3 coreNormal(vec3 p) {
  float h = max(uCore.w * 0.012, 0.0012);
  vec2 e = vec2(h, -h);
  return normalize(
    e.xyy * sdCore(p + e.xyy) +
    e.yyx * sdCore(p + e.yyx) +
    e.yxy * sdCore(p + e.yxy) +
    e.xxx * sdCore(p + e.xxx)
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
  return texture(uPlate, clamp(uv, vec2(0.008), vec2(0.992))).rgb;
}

void evalFloorLighting(vec3 Xf, out float caust, out float shad) {
  caust = 0.0;
  shad = 0.0;
  vec3 sun = normalize(uSun);
  float invSz = 1.0 / max(sun.z, 0.25);

  if (uCore.w > 0.005) {
    vec2 sp = uCore.xyz.xy - sun.xy * (uCore.z * invSz);
    float r = length(Xf.xy - sp);
    float R = uCore.w * 1.45;
    float env = exp(-(r * r) / max(R * R * 0.85, 0.01));
    float ring = exp(-pow((r - R * 0.45) / max(R * 0.42, 0.05), 2.0));
    float n = vnoise(vec3(Xf.xy * 2.6, uSimT * 0.9));
    caust += (0.65 * env + 0.55 * ring * (0.35 + 0.85 * n));
    shad += exp(-(r * r) / max(R * R * 2.2, 0.04)) * 0.22;
  }

  for (int i = 0; i < 28; i++) {
    if (i >= uNumCaust) break;
    vec4 dr = uCaustDrops[i];
    if (dr.w <= 0.004 || dr.z <= 0.05) continue;
    vec2 sp = dr.xy - sun.xy * (dr.z * invSz);
    float r = length(Xf.xy - sp);
    float R = dr.w * 2.0 + dr.z * 0.045;
    if (r > R * 3.2) continue;
    float e2 = exp(-(r * r) / max(R * R * 0.48, 0.001));
    float wgt = clamp(dr.w / 0.075, 0.15, 1.15) * clamp(1.4 / (0.6 + dr.z), 0.25, 1.0);
    caust += e2 * wgt * 0.55;
    shad += exp(-(r * r) / max(R * R * 1.8, 0.004)) * wgt * 0.095;
  }
  caust = clamp(caust, 0.0, 1.45);
  shad = clamp(shad, 0.0, 0.32);
}

void main() {
  vec2 px0 = vec2(gl_FragCoord.x, uRes.y - gl_FragCoord.y);
  vec2 uv0 = px0 / uRes;
  vec3 plateCol = texture(uPlate, clamp(uv0, vec2(0.003), vec2(0.997))).rgb;
  vec3 rd = normalize(uCam * vec3((px0 - 0.5 * uRes) / uF, 1.0));
  float tOcc = hitOccluders(uRo, rd);

  /* Floor caustics ONLY on visible floor strictly in front of walls/doorframes */
  vec3 bgCol = plateCol;
  float tFloor = (rd.z < -0.004) ? (-uRo.z / rd.z) : 1e4;
  if (tFloor > 0.08 && tFloor < min(tOcc, 18.0)) {
    vec3 Xf = uRo + rd * tFloor;
    float roomMask = smoothstep(3.4, 2.2, abs(Xf.x)) * smoothstep(2.2, 0.6, Xf.y) * smoothstep(-6.8, -5.5, Xf.y);
    if (roomMask > 0.01) {
      float caust, shad;
      evalFloorLighting(Xf, caust, shad);
      float luma = dot(plateCol, vec3(0.299, 0.587, 0.114));
      float floorMask = roomMask * smoothstep(0.08, 0.26, luma) * (1.0 - smoothstep(10.0, 16.5, tFloor));
      vec3 warmLight = vec3(1.0, 0.96, 0.88);
      bgCol = bgCol * (1.0 - shad * floorMask);
      bgCol += warmLight * (caust * uCausticGain * 0.24 * floorMask) * (0.45 + 0.55 * bgCol);
    }
  }

  float outDist = tOcc;
  float cgAlpha = 0.0;
  float cgInvD = 0.0;
  vec3 finalCol = bgCol;

  if (uCore.w > 0.005) {
    float boundR = uCore.w * 1.35;
    vec3 oc = uCore.xyz - uRo;
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
        float tEnd = min(bk + sq, tOcc);
        if (t >= tEnd) continue;
        float tHit = -1.0;
        for (int s = 0; s < 56; s++) {
          float d = sdCore(uRo + rdk * t);
          if (d < 0.0012) { tHit = t; break; }
          t += max(d * 0.88, 0.002);
          if (t > tEnd) break;
        }
        if (tHit < 0.0) continue;
        minD = min(minD, tHit);
        vec3 p1 = uRo + rdk * tHit;
        vec3 n1 = coreNormal(p1);
        vec3 rdIn = refract(rdk, n1, 1.0 / uIor);
        if (length(rdIn) < 0.001) rdIn = reflect(rdk, n1);
        float thick = uCore.w * 0.22;
        vec3 p2 = p1 + rdIn * max(uCore.w * 0.16, 0.004);
        for (int s = 0; s < 22; s++) {
          float dIn = -sdCore(p2);
          if (dIn < 0.0018) break;
          float st = max(dIn * 0.88, 0.006);
          p2 += rdIn * st;
          thick += st;
          if (thick > uCore.w * 3.5) break;
        }
        vec3 n2 = -coreNormal(p2);
        vec3 rdOutG = refract(rdIn, n2, uIor);
        if (length(rdOutG) < 0.001) rdOutG = normalize(mix(rdIn, reflect(rdIn, n2), 0.35));
        vec3 bend = rdOutG - rdk;
        vec3 rdOutR = normalize(rdk + bend * ((uIor - uDisp) - 1.0) / max(uIor - 1.0, 0.05));
        vec3 rdOutB = normalize(rdk + bend * ((uIor + uDisp) - 1.0) / max(uIor - 1.0, 0.05));
        vec3 refrCol = vec3(samplePlateRay(p2, rdOutR).r, samplePlateRay(p2, rdOutG).g, samplePlateRay(p2, rdOutB).b);
        refrCol *= exp(-thick * uTint * 0.60) * 1.06;

        float ndv = max(dot(-rdk, n1), 0.0);
        float fresnel = 0.04 + 0.96 * pow(1.0 - ndv, 4.0);
        vec3 reflDir = reflect(rdk, n1);
        vec3 envRefl = samplePlateRay(p1 + n1 * 0.005, reflDir);
        float skyUp = smoothstep(-0.15, 0.85, reflDir.z);
        envRefl = mix(envRefl * 1.02 + vec3(0.05), envRefl * 1.20 + vec3(0.24, 0.26, 0.29), skyUp * 0.56);

        vec3 L1 = normalize(vec3(-0.78, 0.25, 0.56));
        vec3 L2 = normalize(vec3( 0.05, 0.20, 0.98));
        vec3 L3 = normalize(vec3( 0.68, 0.35, 0.64));
        float sp1 = pow(max(dot(n1, normalize(L1 - rdk)), 0.0), 220.0) * 2.2 + pow(max(dot(n1, normalize(L1 - rdk)), 0.0), 30.0) * 0.16;
        float sp2 = pow(max(dot(n1, normalize(L2 - rdk)), 0.0), 160.0) * 1.1;
        float sp3 = pow(max(dot(n1, normalize(L3 - rdk)), 0.0), 180.0) * 0.85;
        float rim = pow(1.0 - ndv, 2.8) * (0.38 + 0.32 * skyUp);

        vec3 c0 = mix(refrCol, envRefl, clamp(fresnel * 0.58, 0.0, 0.80));
        c0 += vec3(0.95, 0.98, 1.0) * sp1 + vec3(1.0, 0.96, 0.88) * sp2 + vec3(1.0, 0.92, 0.78) * sp3 + vec3(0.96, 0.98, 1.0) * rim;
        acc += c0;
        cov += 1.0;
      }
      if (cov > 0.5) {
        float a = cov * 0.25;
        finalCol = mix(bgCol, clamp(acc / cov, 0.0, 1.0), a);
        outDist = min(tOcc, minD);
        cgAlpha = a;
        cgInvD = a / max(minD, 0.20);
      }
    }
  }

  oComp = vec4(clamp(finalCol, 0.0, 1.0), 1.0);
  oDepth = vec4(outDist, tOcc, 0.0, 1.0);
  oMeta = vec4(cgAlpha, cgInvD, 1.0 / max(tOcc, 0.20), 1.0);
}
"""

# Pass 2: Instanced 3D Raymarched Irregular Frozen Liquid-Glass Droplets (strictly occluded by walls/jambs!)
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

uniform sampler2D uBase;    /* Pass-1 FBO color (bottom-up) */
uniform sampler2D uDepth;   /* Pass-1 FBO depth: R = min(tCore, tOcc), G = tOcc (bottom-up) */
uniform vec2 uRes;
uniform float uF, uIor, uDisp, uProxyDist;
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
  vec2 dep = texture(uDepth, gl_FragCoord.xy / uRes).rg;
  float heroD = dep.r;   /* min(tCore, tOcc) */
  float occD  = dep.g;   /* exact real wall/jamb depth */

  vec3 acc = vec3(0.0);
  float cov = 0.0;
  float minT = 1e4;
  vec2 offs[4] = vec2[4](vec2(-0.25, -0.25), vec2(0.25, -0.25), vec2(-0.25, 0.25), vec2(0.25, 0.25));

  vec3 L_win  = normalize(vec3(-0.78, 0.25, 0.56));
  vec3 L_ceil = normalize(vec3( 0.05, 0.20, 0.98));
  vec3 L_pend = normalize(vec3( 0.68, 0.35, 0.64));
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
    float innerMeniscus = 1.0 - 0.22 * exp(-pow((ndv - 0.36) / 0.16, 2.0));
    refrCol *= exp(-thick * uTint * 0.45) * 1.05 * innerMeniscus;

    float fresnel = 0.04 + 0.96 * pow(1.0 - ndv, 3.8);
    vec3 reflDir = reflect(rd, n1);
    vec3 envRefl = sampleEnv(p1 + n1 * 0.004, reflDir);
    float skyUp = smoothstep(-0.15, 0.85, reflDir.z);
    float winRefl = pow(max(dot(reflDir, L_win), 0.0), 6.0);
    envRefl = mix(envRefl * 1.02 + vec3(0.04), envRefl * 1.18 + vec3(0.24, 0.27, 0.31), skyUp * 0.55)
            + vec3(0.32, 0.36, 0.40) * winRefl;

    float spWin  = pow(max(dot(n1, normalize(L_win  - rd)), 0.0), isMicro ? 140.0 : 240.0) * (isMicro ? 2.50 : 2.10)
                 + pow(max(dot(n1, normalize(L_win  - rd)), 0.0), 26.0) * 0.14;
    float spCeil = pow(max(dot(n1, normalize(L_ceil - rd)), 0.0), isMicro ? 110.0 : 180.0) * (isMicro ? 1.30 : 1.00);
    float spPend = pow(max(dot(n1, normalize(L_pend - rd)), 0.0), isMicro ? 130.0 : 200.0) * 0.82;
    float backGlint = pow(max(dot(-n2, L_win), 0.0), 5.0) * (0.19 + 0.21 * (1.0 - ndv));
    float rim = pow(1.0 - ndv, 2.8) * (0.36 + 0.30 * skyUp);

    vec3 col = mix(refrCol, envRefl, clamp(fresnel * 0.56, 0.0, 0.82));
    col += vec3(0.95, 0.98, 1.00) * (spWin + backGlint);
    col += vec3(1.00, 0.96, 0.88) * spCeil;
    col += vec3(1.00, 0.92, 0.78) * spPend;
    col += vec3(0.96, 0.99, 1.00) * rim;

    acc += col;
    cov += 1.0;
  }

  if (cov < 0.5) discard;
  float a = cov * 0.25;
  oCol = vec4(clamp(acc / cov, 0.0, 1.0), a);
  oMeta = vec4(1.0, 1.0 / max(minT, 0.20), 1.0 / max(occD, 0.20), a);
}
"""

# Pass 3: Camera Optical Finish (Depth-Aware Thin-Lens Bokeh Defocus + OLPF Softness + Halation + Edge CA + Sensor Grain)
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

  /* Ring A: tight 10-tap precision optical kernel around center */
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

  /* Ring B: wide 24-tap foreground near-lens bokeh spread + specular halation */
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
  res += halation * vec3(1.02, 0.98, 0.93) * 0.052;

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


def encode(c, out_dir="out_v2", suffix="_v2", title_label="3D Liquid Glass — Frozen Splash (v2 Optics)"):
    out = c["dir"] / out_dir
    out.mkdir(exist_ok=True)
    nm = c["name"] + suffix
    r = f"{c['fps_num']}/{c['fps_den']}"
    ff = find_ffmpeg()
    tags = ["-pix_fmt", "yuv420p", "-color_primaries", "bt709", "-color_trc", "bt709", "-colorspace", "bt709", "-movflags", "+faststart"]
    x264 = ["-c:v", "libx264", "-profile:v", "high", "-level", "4.1", "-crf", "14", "-preset", "slow"]
    run = lambda a: subprocess.run([ff, "-v", "error", "-y"] + a, check=True)
    run(["-framerate", r, "-i", str(work(c, "comp_v2", "%04d.png"))] + x264 + tags + [str(out / f"{nm}_Composite.mp4")])
    lab = lambda t: f"drawtext=fontfile='C\\:/Windows/Fonts/arialbd.ttf':text='{t}':x=w-tw-28:y=h-th-24:fontsize=40:fontcolor=white:shadowcolor=black@0.6:shadowx=2:shadowy=2"
    run([
        "-framerate", r, "-i", str(work(c, "comp_v2", "%04d.png")),
        "-framerate", r, "-i", str(work(c, "frames", "%04d.jpg")),
        "-filter_complex",
        f"[0]scale=1080:608,{lab(title_label)}[a];[1]scale=1080:608,{lab('Before')}[b];[a][b]vstack,pad=1080:1350:0:67:color=black",
    ] + x264 + tags + [str(out / f"{nm}_Compare_1080x1350.mp4")])
    print("encoded ->", out)


def build_frozen_splash_cloud(cam_path, origin, occ_boxes, n_hero=30, n_mid=160, n_micro=790, seed=2026):
    """Build ~980 irregular liquid-glass droplets clear of walls/jambs & camera pinhole."""
    rng = np.random.default_rng(seed)
    origin = np.asarray(origin, np.float64)

    def ensure_room_and_cam_clearance(pos, rad, asp, lobe_scale):
        y = float(np.clip(pos[1], -5.95, 2.05))
        max_x = 2.15 if y < 0.95 else 0.88
        x = float(np.clip(pos[0], -max_x, max_x))
        z = float(np.clip(pos[2], 0.22, 2.60))
        p = np.array([x, y, z], np.float64)

        # Keep clear of all 3D architectural holdout walls/jambs
        bound_r = rad * (max(asp, 1.0) * 1.4 + (0.65 if lobe_scale > 0.05 else 0.0) + 0.35)
        pad = bound_r + 0.08
        for (x0, x1, y0, y1, z0, z1) in occ_boxes:
            if (x0 - pad <= p[0] <= x1 + pad and y0 - pad <= p[1] <= y1 + pad and z0 <= p[2] <= z1):
                if p[1] >= y0 - pad and abs(p[0]) >= 0.85:
                    p[0] = float(np.sign(p[0]) * min(abs(p[0]), 0.82))
                else:
                    p[1] = min(p[1], y0 - pad - 0.05)

        min_clear = 0.16 + bound_r * 1.55
        diffs = p[None, :] - cam_path
        dists = np.linalg.norm(diffs, axis=1)
        idx = int(np.argmin(dists))
        if dists[idx] < min_clear:
            v = p - cam_path[idx]
            xz = np.array([v[0], v[2]])
            nrm = np.linalg.norm(xz)
            if nrm < 1e-4:
                ang = rng.uniform(0, 2 * math.pi)
                xz = np.array([math.cos(ang), math.sin(ang)])
            else:
                xz /= nrm
            push = (min_clear - dists[idx]) + 0.05
            p[0] = float(np.clip(p[0] + xz[0] * push, -max_x, max_x))
            p[2] = float(np.clip(p[2] + xz[1] * push, 0.22, 2.60))
        return p

    tendrils = []
    for k in range(18):
        u = (k + 0.5) / 18.0
        phi = math.acos(1.0 - 2.0 * u) + rng.normal(0.0, 0.18)
        theta = k * 2.3999632 + rng.normal(0.0, 0.22)
        d = np.array([
            math.sin(phi) * math.cos(theta) * 1.05,
            math.sin(phi) * math.sin(theta) * 1.38,
            math.cos(phi) * 0.72 + 0.12,
        ], np.float64)
        d /= max(np.linalg.norm(d), 1e-4)
        reach = float(rng.uniform(1.65, 3.15))
        tendrils.append((d, reach))

    drops = []

    def add_drop(p1_raw, rad, dir_hint, tier, asp_range, lobe_prob, bend_max):
        lobe_scale = float(rng.uniform(0.32, 0.92)) if rng.random() < lobe_prob else 0.0
        asp = float(rng.uniform(*asp_range))
        p1 = ensure_room_and_cam_clearance(p1_raw, rad, asp, lobe_scale)

        d_out = p1 - origin
        d_len = max(np.linalg.norm(d_out), 1e-4)
        u_out = d_out / d_len
        p0 = origin + u_out * float(rng.uniform(0.03, 0.17))

        ax = dir_hint + rng.normal(0.0, 0.26, 3)
        ax /= max(np.linalg.norm(ax), 1e-4)
        lobe_dir = rng.normal(0.0, 1.0, 3)
        lobe_dir /= max(np.linalg.norm(lobe_dir), 1e-4)

        wob_amp = float(rng.uniform(0.055, 0.145) if tier >= 1 else rng.uniform(0.035, 0.105))
        bend = float(rng.uniform(-bend_max, bend_max))
        droop = float(rng.uniform(0.04, 0.18) * min(d_len / 2.2, 1.2))
        delay = float(rng.uniform(0.0, 0.045))

        drops.append({
            "p0": p0, "p1": p1, "droop": droop, "rad": rad, "axis": ax, "asp": asp,
            "wob_seed": float(rng.uniform(0.0, 12.0)), "wob_amp": wob_amp,
            "lobe_scale": lobe_scale, "bend": bend, "lobe_dir": lobe_dir,
            "tier": float(tier), "delay": delay,
        })

    for k in range(n_hero):
        td, reach = tendrils[k % len(tendrils)]
        frac = float(rng.uniform(0.32, 0.95)) if k < 18 else float(rng.uniform(0.14, 0.48))
        p1_raw = origin + td * (reach * frac) + rng.normal(0.0, 0.16, 3)
        rad = float(np.exp(rng.uniform(math.log(0.046), math.log(0.098))))
        add_drop(p1_raw, rad, td, tier=2, asp_range=(1.35, 2.35), lobe_prob=0.82, bend_max=0.38)

    for k in range(n_mid):
        if k < int(n_mid * 0.72):
            td, reach = tendrils[k % len(tendrils)]
            frac = float(rng.uniform(0.16, 1.05))
            p1_raw = origin + td * (reach * frac) + rng.normal(0.0, 0.08 + 0.24 * frac, 3)
            dir_h = td
        elif k < int(n_mid * 0.88):
            d = rng.normal(0.0, 1.0, 3)
            d[1] *= 1.55; d[2] *= 0.72
            d /= max(np.linalg.norm(d), 1e-4)
            p1_raw = origin + d * float(rng.uniform(0.45, 2.95))
            dir_h = d
        else:
            y = float(rng.uniform(-1.55, 1.85))
            x = float(rng.uniform(-0.82, 0.82))
            z = float(rng.uniform(0.42, 2.38))
            p1_raw = np.array([x, y, z], np.float64)
            dir_h = p1_raw - origin
            dir_h /= max(np.linalg.norm(dir_h), 1e-4)
        rad = float(np.exp(rng.uniform(math.log(0.015), math.log(0.045))))
        add_drop(p1_raw, rad, dir_h, tier=1, asp_range=(1.15, 1.95), lobe_prob=0.52, bend_max=0.26)

    for k in range(n_micro):
        mode = rng.random()
        if mode < 0.44:
            td, reach = tendrils[k % len(tendrils)]
            frac = float(rng.uniform(0.08, 1.24))
            p1_raw = origin + td * (reach * frac) + rng.normal(0.0, 0.06 + 0.40 * frac, 3)
            dir_h = td
        elif mode < 0.72:
            y = float(rng.uniform(-5.85, 1.95))
            max_x = 2.00 if y < 0.90 else 0.84
            x = float(rng.uniform(-max_x, max_x))
            z = float(rng.uniform(0.28, 2.54))
            p1_raw = np.array([x, y, z], np.float64)
            dir_h = p1_raw - origin
            dir_h /= max(np.linalg.norm(dir_h), 1e-4)
        elif mode < 0.88:
            y = float(rng.uniform(-5.30, 0.40))
            ang = float(rng.uniform(0.0, 2.0 * math.pi))
            r_lat = float(rng.uniform(0.22, 0.82))
            p1_raw = np.array([-0.08 + math.cos(ang) * r_lat, y, 1.43 + math.sin(ang) * r_lat * 0.78], np.float64)
            dir_h = p1_raw - origin
            dir_h /= max(np.linalg.norm(dir_h), 1e-4)
        else:
            y = float(rng.uniform(-1.60, 2.00))
            x = float(rng.uniform(-0.84, 0.84))
            z = float(rng.uniform(0.38, 2.42))
            p1_raw = np.array([x, y, z], np.float64)
            dir_h = p1_raw - origin
            dir_h /= max(np.linalg.norm(dir_h), 1e-4)

        rad = float(np.exp(rng.uniform(math.log(0.0036), math.log(0.0155))))
        add_drop(p1_raw, rad, dir_h, tier=0, asp_range=(1.06, 1.90), lobe_prob=0.30, bend_max=0.18)

    min_d = min(float(np.linalg.norm(cam_path - d["p1"][None, :], axis=1).min()) for d in drops)
    print(f"Frozen splash v2: {n_hero} hero + {n_mid} mid + {n_micro} micro = {len(drops)} droplets + {len(occ_boxes)} 3D occluders (min_cam_dist={min_d:.2f}m)")
    return drops


def main():
    c = load(sys.argv[1] if len(sys.argv) > 1 and not sys.argv[1].startswith("--") else here / "piece_frozen")
    S = scene(c)
    W, H, N, fps = c["W"], c["H"], c["N"], c["fps"]
    f = float(S["f"])
    cfg = c.get("frozen", {})
    stills = [int(x) for x in sys.argv[sys.argv.index("--still") + 1].split(",")] if "--still" in sys.argv else None
    out_dir = sys.argv[sys.argv.index("--out") + 1] if "--out" in sys.argv else "out_v2"

    cam_path = np.asarray(S["cam"], np.float64)
    origin = np.array(cfg.get("burst_origin", [-0.08, -3.65, 1.44]), np.float64)
    burst_frame = int(cfg.get("burst_frame", 14))
    freeze_dur_s = float(cfg.get("freeze_dur_s", 1.25))

    # 3 COLMAP-triangulated & visually verified 3D architectural holdout boxes for Pexels 7578547
    occ_boxes = [
        (-3.50, -1.55, -8.00, -4.70, 0.0, 5.0),  # Foreground left doorframe pillar
        (-4.50, -1.08,  1.22,  3.80, 0.0, 5.0),  # Left white hallway wall (with abstract painting)
        ( 0.96,  4.50,  1.14,  3.80, 0.0, 5.0),  # Right white hallway wall (with wooden mirror)
    ]

    drops = build_frozen_splash_cloud(
        cam_path, origin, occ_boxes,
        n_hero=int(cfg.get("n_hero", 30)),
        n_mid=int(cfg.get("n_mid", 160)),
        n_micro=int(cfg.get("n_micro", 790)),
        seed=int(cfg.get("seed", 2026)),
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
    tint = tuple(map(float, cfg.get("tint", [0.045, 0.018, 0.014])))
    proxy_dist = float(cfg.get("proxy_dist", 6.2))

    copy2_prog["uRes"].value = (float(W), float(H))

    occ_mins = [(b[0], b[2], b[4]) for b in occ_boxes] + [(0.0, 0.0, 0.0)] * (4 - len(occ_boxes))
    occ_maxs = [(b[1], b[3], b[5]) for b in occ_boxes] + [(0.0, 0.0, 0.0)] * (4 - len(occ_boxes))
    core_prog["uRes"].value = (float(W), float(H))
    core_prog["uF"].value = f
    core_prog["uSun"].value = tuple(map(float, S["sun"]))
    core_prog["uOccMin"].value = occ_mins[:4]
    core_prog["uOccMax"].value = occ_maxs[:4]
    core_prog["uNumOcc"].value = len(occ_boxes)
    core_prog["uIor"].value = ior
    core_prog["uDisp"].value = disp
    core_prog["uTint"].value = tint
    core_prog["uCausticGain"].value = float(cfg.get("caustic_gain", 1.05))
    core_prog["uProxyDist"].value = proxy_dist

    drop_prog["uRes"].value = (float(W), float(H))
    drop_prog["uF"].value = f
    if "uSun" in drop_prog:
        drop_prog["uSun"].value = tuple(map(float, S["sun"]))
    drop_prog["uIor"].value = ior
    drop_prog["uDisp"].value = disp
    drop_prog["uTint"].value = tint
    drop_prog["uProxyDist"].value = proxy_dist

    # Cinema optical finish tuned to Pexels 7578547 (measured black floor ~0.0167)
    post_prog["uRes"].value = (float(W), float(H))
    post_prog["uFocusDist"].value = 2.9
    post_prog["uNearBlur"].value = 4.5
    post_prog["uFarBlur"].value = 4.8
    post_prog["uBaseSoft"].value = 0.68
    post_prog["uMaxCoc"].value = 11.0
    post_prog["uGrainAmt"].value = 0.012
    post_prog["uBlackFloor"].value = 0.017

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
        t_sec = i / fps
        ro = np.asarray(S["cam"][i], np.float64)
        R = np.asarray(S["rot"][i], np.float64)
        Rt = tuple(R.astype(np.float32).T.flatten())
        age_s = (i - burst_frame) / fps

        if age_s < 0.0:
            u_pre = i / max(float(burst_frame), 1.0)
            core_scale = 0.36 * (1.0 + 0.08 * math.sin(i * 0.80) + 0.22 * (u_pre ** 2.4))
            sim_t = t_sec
        elif age_s < 0.22:
            u_sh = age_s / 0.22
            core_scale = 0.44 * max(0.0, 1.0 - u_sh ** 1.25)
            sim_t = burst_frame / fps + age_s * 1.4
        else:
            core_scale = 0.0
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

        # Pass 1: Core + Floor caustics + 3D architectural holdout boxes
        plate_t.write(pl[:, :, ::-1].tobytes())
        plate_t.use(0)
        core_prog["uPlate"].value = 0
        core_prog["uRo"].value = tuple(map(float, ro))
        core_prog["uCam"].value = Rt
        core_prog["uSimT"].value = float(sim_t)
        core_prog["uCore"].value = (float(origin[0]), float(origin[1]), float(origin[2]), float(core_scale))
        core_prog["uCaustDrops"].value = caust_list[:28]
        core_prog["uNumCaust"].value = int(n_caust)

        fbo_hero.use()
        ctx.disable(moderngl.BLEND)
        va_core.render(moderngl.TRIANGLES)

        # Copy Pass-1 color + meta into Pass-2 FBO
        fbo_scene.use()
        t_hero.use(0)
        t_meta0.use(1)
        copy2_prog["uSrc0"].value = 0
        copy2_prog["uSrc1"].value = 1
        va_copy2.render(moderngl.TRIANGLES)

        # Pass 2: Depth-sorted 3D droplets occluded by walls & doorway jambs
        rows_d.sort(key=lambda x: -x[0])
        if rows_d:
            data = np.array([r[1] for r in rows_d], dtype=np.float32)
            inst_buf.write(data.tobytes())
            t_hero.use(0)
            t_depth.use(1)
            drop_prog["uBase"].value = 0
            drop_prog["uDepth"].value = 1
            drop_prog["uRo"].value = tuple(map(float, ro))
            drop_prog["uCam"].value = Rt
            ctx.enable(moderngl.BLEND)
            ctx.blend_func = (moderngl.SRC_ALPHA, moderngl.ONE_MINUS_SRC_ALPHA)
            va_drop.render(moderngl.TRIANGLES, instances=len(rows_d))
            ctx.disable(moderngl.BLEND)

        # Pass 3: Camera optical finish (depth-aware bokeh + OLPF softness + halation + radial CA + sensor grain)
        fbo_final.use()
        t_scene.use(0)
        t_meta.use(1)
        post_prog["uScene"].value = 0
        post_prog["uMeta"].value = 1
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
            rows.append(np.hstack([t_im(pl, f"plate {i}"), t_im(comp, f"frozen v2 (occ+optics) {i}")]))
        out_p = work(c, "still_frozen_v2_sheet.jpg")
        cv2.imwrite(str(out_p), np.vstack(rows), [cv2.IMWRITE_JPEG_QUALITY, 92])
        if hero is not None:
            cv2.imwrite(str(work(c, "still_frozen_v2_hero.jpg")), hero, [cv2.IMWRITE_JPEG_QUALITY, 94])
        print("stills ->", out_p)
        return

    comp_dir = work(c, "comp_v2")
    shutil.rmtree(comp_dir, ignore_errors=True)
    comp_dir.mkdir(parents=True)
    ts = time.time()
    for i in range(N):
        _, comp = render_one(i)
        cv2.imwrite(str(comp_dir / f"{i:04d}.png"), comp, [cv2.IMWRITE_PNG_COMPRESSION, 1])
        if i % 40 == 0 or i == N - 1:
            print(f"frozen v2 frame {i}/{N} ({time.time() - ts:.1f}s)", flush=True)
    encode(c, out_dir=out_dir)


if __name__ == "__main__":
    main()
