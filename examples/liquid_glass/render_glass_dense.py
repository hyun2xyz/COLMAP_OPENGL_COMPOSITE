"""Dense 3D Liquid Glass & Fluted Optics Gallery with Real 3D Architectural Occlusion & Camera Optical Finish (ModernGL, OpenGL 3.3).

    python render_glass_dense.py [piece_dir] --still 0,25,85,155,195,235   quick look -> work/still_dense_v2_sheet.jpg
    python render_glass_dense.py [piece_dir] --out out_dense_v2            full render -> out_dense_v2/*_Composite.mp4, *_Compare_1080x1350.mp4

3-Pass Pro Compositing Pipeline:
- Pass 1 (HERO_FS): 5 world-locked liquid-glass metaball clusters + 4 angled fluted glass monoliths + 3D floor caustics,
  tested against 6 COLMAP-triangulated 3D invisible architectural holdout boxes (left/right entry walls + 4 structural pillars).
- Pass 2 (BEAD_VS/BEAD_FS): ~185 instanced 3D raymarched irregular crystal satellite beads, depth-sorted back-to-front,
  strictly occluded against BOTH hero glass AND all 6 architectural walls/pillars via t_depth.
- Pass 3 (POST_FS): Physical thin-lens depth-of-field bokeh (near-lens defocus + OLPF base optical softness + subtle far softness)
  that respects wall/pillar occlusion depth, subtle specular halation, edge radial CA, filmic black-floor integration,
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

# Pass 1: Hero Liquid-Glass Clusters + Fluted Monoliths + Floor Caustics + 6 3D Architectural Holdout Boxes
HERO_FS = """#version 330
uniform sampler2D uPlate;
uniform vec2 uRes;
uniform float uF, iTime;
uniform vec3 uRo, uSun;
uniform mat3 uCam;
uniform vec4 uClusters[5];     /* xyz = center (m), w = scale */
uniform vec4 uFlutePos[4];     /* xyz = center (m), w = yaw (rad) */
uniform vec3 uFluteSize[4];    /* half-extents (hx, hy, hz) */
uniform vec3 uOccMin[6];       /* 3D architectural holdout boxes (walls + structural columns) */
uniform vec3 uOccMax[6];
uniform int uNumOcc;
uniform float uIor, uDisp, uCausticGain, uShadowGain, uFluteFreq, uFluteAmp, uMetaK, uProxyDist;
uniform vec3 uTint;

layout(location=0) out vec4 oComp;
layout(location=1) out vec4 oDepth;   /* R: min(tGlass, tOcc) for Pass-2 occlusion, G: tOcc (wall/pillar distance) */
layout(location=2) out vec4 oMeta;    /* R: cgAlpha, G: cgInvDepth (alpha/tHit), B: occInvDepth (1/tOcc), A: 1.0 */

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

float sdRoundBox(vec3 p, vec3 b, float r) {
  vec3 q = abs(p) - b + r;
  return length(max(q, 0.0)) + min(max(q.x, max(q.y, q.z)), 0.0) - r;
}

mat2 rot2(float a) {
  float c = cos(a), s = sin(a);
  return mat2(c, -s, s, c);
}

/* Exact ray-AABB intersection against real walls & structural columns */
float hitOccluders(vec3 ro, vec3 rd) {
  float tBest = 1e4;
  vec3 invD = 1.0 / mix(rd, vec3(1e-8), lessThan(abs(rd), vec3(1e-8)));
  for (int i = 0; i < 6; i++) {
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
  for (int i = 0; i < 6; i++) {
    if (i >= uNumOcc) break;
    if (xy.x >= uOccMin[i].x && xy.x <= uOccMax[i].x &&
        xy.y >= uOccMin[i].y && xy.y <= uOccMax[i].y) return true;
  }
  return false;
}

float sdCluster(vec3 p, vec4 cfg, float phase) {
  vec3 q = (p - cfg.xyz) / cfg.w;
  float t = iTime * 0.85 + phase;
  vec3 c0 = vec3(0.06 * sin(t * 0.7), 0.05 * cos(t * 0.6), 0.08 * sin(t * 0.9));
  float d = length(q - c0) - 0.34;

  vec3 c1 = vec3( 0.45 * cos(t * 0.95),        0.34 * sin(t * 0.80),  0.24 * sin(t * 1.15));
  vec3 c2 = vec3(-0.42 * cos(t * 0.78 + 1.2),  0.38 * sin(t * 0.92), -0.22 * cos(t * 1.05));
  vec3 c3 = vec3( 0.32 * sin(t * 1.22 + 2.4), -0.32 * cos(t * 0.85),  0.36 * cos(t * 0.90));
  vec3 c4 = vec3(-0.34 * sin(t * 0.90 + 4.0), -0.26 * sin(t * 1.10), -0.28 * sin(t * 0.82));
  vec3 c5 = vec3( 0.58 * cos(t * 0.62 + 3.1),  0.18 * cos(t * 1.30),  0.12 * sin(t * 1.40));
  vec3 c6 = vec3(-0.54 * sin(t * 0.68 + 0.7), -0.18 * cos(t * 1.18),  0.30 * cos(t * 1.12));
  vec3 c7 = vec3( 0.14 * cos(t * 1.12 + 5.0),  0.46 * cos(t * 0.75), -0.36 * sin(t * 0.95));

  float k = uMetaK;
  d = smin(d, length(q - c1) - 0.21, k);
  d = smin(d, length(q - c2) - 0.19, k);
  d = smin(d, length(q - c3) - 0.18, k);
  d = smin(d, length(q - c4) - 0.17, k);
  d = smin(d, length(q - c5) - 0.15, k * 0.85);
  d = smin(d, length(q - c6) - 0.14, k * 0.85);
  d = smin(d, length(q - c7) - 0.15, k * 0.85);

  float rip = (vnoise(q * 3.1 + vec3(0.0, t * 0.40, -t * 0.30)) - 0.5) * 0.020;
  return (d + rip) * cfg.w;
}

float sdFluteOne(vec3 p, int idx, out vec3 localP) {
  vec3 pos = uFlutePos[idx].xyz;
  float yaw = uFlutePos[idx].w;
  vec3 sz = uFluteSize[idx];
  vec3 q = p - pos;
  q.z -= 0.035 * sin(iTime * 0.82 + float(idx) * 2.1);
  q.xy = rot2(-yaw) * q.xy;
  localP = q;
  float box = sdRoundBox(q, sz, 0.034);
  float faceMask = smoothstep(0.02, 0.08, sz.x - abs(q.x)) * smoothstep(0.02, 0.08, sz.z - abs(q.z));
  return box + sin(q.x * uFluteFreq) * 0.0048 * faceMask;
}

vec2 mapScene(vec3 p) {
  float dBest = 1e4;
  float mBest = 0.0;
  for (int i = 0; i < 5; i++) {
    float bs = length(p - uClusters[i].xyz) - 1.38 * uClusters[i].w;
    if (bs < dBest) {
      float dc = sdCluster(p, uClusters[i], float(i) * 1.9);
      if (dc < dBest) { dBest = dc; mBest = 0.0; }
    }
  }
  for (int i = 0; i < 4; i++) {
    float rad = length(uFluteSize[i]) + 0.25;
    float bs = length(p - uFlutePos[i].xyz) - rad;
    if (bs < dBest) {
      vec3 lp;
      float df = sdFluteOne(p, i, lp);
      if (df < dBest) { dBest = df; mBest = float(i + 1); }
    }
  }
  return vec2(dBest, mBest);
}

float mapDist(vec3 p) { return mapScene(p).x; }

vec3 calcNormal(vec3 p) {
  vec2 e = vec2(0.0028, -0.0028);
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

vec3 sampleBackdrop(vec3 ro, vec3 rd, float blurPx) {
  float tf = (rd.z < -0.012) ? (-ro.z / rd.z) : 1e4;
  if (tf < 0.05 || tf > 26.0) tf = 1e4;
  vec3 X = ro + rd * min(tf, uProxyDist);
  float zc;
  vec2 uv = projUV(X, zc);
  if (zc < 0.1) uv = clamp(vec2(1.0 - uv.x, uv.y), 0.01, 0.99);
  uv = abs(fract(uv * 0.5 + 0.5) * 2.0 - 1.0);
  uv = clamp(uv, vec2(0.008), vec2(0.992));
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

void floorCaustic(vec3 Xf, out float caustic, out float shadow) {
  caustic = 0.0; shadow = 0.0;
  if (insideOccXY(Xf.xy)) return;
  vec3 sun = normalize(uSun);
  float nearD = 1e4;
  for (int i = 0; i < 5; i++) {
    vec3 sh = uClusters[i].xyz - sun * (uClusters[i].z / max(sun.z, 0.20));
    nearD = min(nearD, length(Xf.xy - sh.xy));
    float g = exp(-dot(Xf.xy - uClusters[i].xy, Xf.xy - uClusters[i].xy) / (0.68 * uClusters[i].w * uClusters[i].w)) * 0.16;
    shadow = max(shadow, g);
  }
  for (int i = 0; i < 4; i++) {
    vec3 sh = uFlutePos[i].xyz - sun * (uFlutePos[i].z / max(sun.z, 0.20));
    nearD = min(nearD, length(Xf.xy - sh.xy));
    float g = exp(-dot(Xf.xy - uFlutePos[i].xy, Xf.xy - uFlutePos[i].xy) / 0.52) * 0.18;
    shadow = max(shadow, g);
  }
  if (nearD > 2.4) return;

  float t = 0.05;
  bool hit = false;
  float minD = 1e4;
  float hitMat = 0.0;
  for (int i = 0; i < 46; i++) {
    vec3 p = Xf + sun * t;
    if (p.z > 3.6) break;
    vec2 h = mapScene(p);
    minD = min(minD, h.x);
    if (h.x < 0.012) { hit = true; hitMat = h.y; break; }
    t += max(h.x * 0.85, 0.028);
  }
  shadow = max(shadow, (1.0 - smoothstep(0.0, 0.42, minD)) * 0.34);

  if (hit) {
    for (int k = 0; k < 3; k++) t += mapDist(Xf + sun * t);
    vec3 P = Xf + sun * t;
    vec3 N = calcNormal(P);
    if (hitMat > 0.5) {
      int fi = int(hitMat + 0.5) - 1;
      vec3 lp; sdFluteOne(P, fi, lp);
      vec3 sz = uFluteSize[fi];
      float edgeFade = smoothstep(0.0, 0.11, sz.x - abs(lp.x)) * smoothstep(0.0, 0.13, sz.z - abs(lp.z));
      float bands = pow(0.5 + 0.5 * cos(lp.x * uFluteFreq * 2.0), 2.8);
      caustic = pow(edgeFade, 1.25) * (0.20 + 1.40 * bands);
      shadow = mix(0.45, 0.05, clamp(caustic * 0.55, 0.0, 1.0));
    } else {
      float nds = max(abs(dot(N, sun)), 0.0);
      vec3 r0 = refract(-sun, N, 1.0 / uIor);
      vec2 focusPt = P.xy + r0.xy * (-P.z / min(r0.z, -0.15));
      float n1 = vnoise(vec3(focusPt * 1.9, iTime * 0.38));
      float n2 = vnoise(vec3(focusPt * 3.1 + vec2(2.4, -1.3), -iTime * 0.46));
      float fil = pow(clamp(1.0 - abs(n1 - 0.5) * 2.2, 0.0, 1.0), 2.0) * 0.72
                + pow(clamp(1.0 - abs(n2 - 0.5) * 2.4, 0.0, 1.0), 2.2) * 0.42;
      float softEnv = pow(smoothstep(0.14, 0.68, nds), 1.35);
      caustic = (pow(nds, 2.2) * 0.85 + fil * 0.78) * softEnv;
      shadow = mix(0.32, 0.03, clamp(caustic * 0.65, 0.0, 1.0));
    }
  }
}

vec3 traceGlass(vec3 ro, vec3 rd, vec2 uv0, out float outDist, out float outOcc, out float outCgAlpha) {
  vec3 plateCol = texture(uPlate, uv0).rgb;
  outOcc = hitOccluders(ro, rd);
  outDist = outOcc;
  outCgAlpha = 0.0;

  float tFloor = (rd.z < -0.004) ? (-ro.z / rd.z) : 1e4;
  if (tFloor < 0.05 || tFloor > 24.0) tFloor = 1e4;

  bool nearAny = false;
  for (int i = 0; i < 5; i++) {
    if (hitSphere(ro, rd, uClusters[i].xyz, 1.35 * uClusters[i].w)) { nearAny = true; break; }
  }
  if (!nearAny) {
    for (int i = 0; i < 4; i++) {
      if (hitSphere(ro, rd, uFlutePos[i].xyz, length(uFluteSize[i]) + 0.25)) { nearAny = true; break; }
    }
  }

  float tHit = 1e4;
  float matId = -1.0;
  if (nearAny) {
    float t = 0.04;
    /* Stop raymarching strictly at real wall/pillar occluder distance outOcc! */
    float tMax = min(min(tFloor + 0.05, outOcc), 22.0);
    for (int i = 0; i < 96; i++) {
      if (t > tMax) break;
      vec3 p = ro + rd * t;
      vec2 h = mapScene(p);
      if (h.x < 0.002) { tHit = t; matId = h.y; break; }
      t += max(h.x * 0.88, 0.004);
    }
  }

  vec3 bgCol = plateCol;
  /* Floor caustics ONLY on visible floor strictly in front of walls & columns! */
  if (tFloor < min(outOcc, 22.0)) {
    vec3 Xf = ro + rd * tFloor;
    float caust, shad;
    floorCaustic(Xf, caust, shad);
    if (caust > 0.001 || shad > 0.001) {
      float luma = dot(plateCol, vec3(0.299, 0.587, 0.114));
      float floorMask = smoothstep(0.08, 0.24, luma) * (1.0 - smoothstep(14.0, 21.0, tFloor));
      vec3 sunlit = vec3(1.0, 0.94, 0.82);
      bgCol = bgCol * (1.0 - shad * uShadowGain * floorMask);
      bgCol += sunlit * (caust * uCausticGain * 0.30 * floorMask) * (0.42 + 0.58 * bgCol);
    }
  }

  if (matId < -0.5 || tHit >= outOcc) return bgCol;
  outDist = tHit;
  outCgAlpha = 1.0;

  vec3 p1 = ro + rd * tHit;
  vec3 n1 = calcNormal(p1);
  bool isFlute = (matId > 0.5);
  int fi = int(matId + 0.5) - 1;

  float fluteCrest = 0.0;
  if (isFlute) {
    vec3 lp; sdFluteOne(p1, fi, lp);
    vec3 sz = uFluteSize[fi];
    float yaw = uFlutePos[fi].w;
    float faceMask = smoothstep(0.018, 0.075, sz.x - abs(lp.x)) * smoothstep(0.018, 0.075, sz.z - abs(lp.z));
    float slope = cos(lp.x * uFluteFreq) * uFluteAmp * faceMask;
    fluteCrest = pow(abs(sin(lp.x * uFluteFreq)), 4.0) * faceMask;
    n1 = normalize(n1 + vec3(cos(yaw), sin(yaw), 0.0) * slope);
  }

  vec3 rdInG = refract(rd, n1, 1.0 / uIor);
  if (length(rdInG) < 0.001) rdInG = reflect(rd, n1);

  float thick = 0.08;
  vec3 p2 = p1 + rdInG * 0.05;
  for (int i = 0; i < 34; i++) {
    float dIn = -mapDist(p2);
    if (dIn < 0.003) break;
    float stepD = max(dIn * 0.85, 0.014);
    p2 += rdInG * stepD;
    thick += stepD;
    if (thick > 2.0) break;
  }
  vec3 n2 = -calcNormal(p2);

  float iorR = uIor - uDisp, iorG = uIor, iorB = uIor + uDisp;
  vec3 rdOutG = refract(rdInG, n2, iorG);
  if (length(rdOutG) < 0.001) rdOutG = normalize(mix(rdInG, reflect(rdInG, n2), 0.35));
  vec3 bend = rdOutG - rd;
  vec3 rdOutR = normalize(rd + bend * (iorR - 1.0) / max(iorG - 1.0, 0.05));
  vec3 rdOutB = normalize(rd + bend * (iorB - 1.0) / max(iorG - 1.0, 0.05));

  float blurPx = isFlute ? (3.5 + 2.2 * fluteCrest) : 0.45;
  vec3 refrCol = vec3(
    sampleBackdrop(p2, rdOutR, blurPx).r,
    sampleBackdrop(p2, rdOutG, blurPx).g,
    sampleBackdrop(p2, rdOutB, blurPx).b
  );
  refrCol = refrCol * exp(-thick * uTint * (isFlute ? 1.35 : 0.90)) * 1.03;

  float NdotV = max(dot(-rd, n1), 0.0);
  float fresnel = 0.035 + 0.965 * pow(1.0 - NdotV, 4.6);
  vec3 reflDir = reflect(rd, n1);
  vec3 envRefl = sampleBackdrop(p1 + n1 * 0.01, reflDir, isFlute ? 7.0 : 3.2);
  float skyUp = smoothstep(-0.1, 0.85, reflDir.z);
  envRefl = mix(envRefl * 0.96, envRefl * 1.12 + vec3(0.18, 0.19, 0.21), skyUp * 0.52);

  vec3 sunDir = normalize(uSun);
  vec3 H1 = normalize(sunDir - rd);
  float specSun = pow(max(dot(n1, H1), 0.0), isFlute ? 110.0 : 220.0) * (isFlute ? 0.95 : 1.50);
  float specBroad = pow(max(dot(n1, H1), 0.0), 24.0) * 0.15;
  vec3 H2 = normalize(normalize(vec3(0.04, 0.12, 0.98)) - rd);
  float specCeil = pow(max(dot(n1, H2), 0.0), 120.0) * 0.44;
  float rimGlow = pow(1.0 - NdotV, 3.2) * (0.32 + 0.26 * skyUp);

  float reflMix = clamp(fresnel * (isFlute ? 0.44 : 0.60), 0.0, 0.78);
  vec3 col = mix(refrCol, envRefl, reflMix);
  col += vec3(1.0, 0.97, 0.91) * (specSun + specBroad);
  col += vec3(0.95, 0.98, 1.0) * (specCeil + rimGlow);
  if (isFlute) {
    col += vec3(0.96, 0.98, 1.0) * fluteCrest * 0.085 * (0.35 + 0.65 * fresnel);
    vec3 lp; sdFluteOne(p1, fi, lp);
    vec3 sz = uFluteSize[fi];
    float edgeBevel = smoothstep(0.05, 0.0, min(sz.x - abs(lp.x), sz.z - abs(lp.z)));
    col += vec3(0.86, 0.95, 0.96) * edgeBevel * 0.15;
  }
  return col;
}

void main() {
  vec2 px0 = vec2(gl_FragCoord.x, uRes.y - gl_FragCoord.y);
  vec3 acc = vec3(0.0);
  float minD = 1e4;
  float minOcc = 1e4;
  float sumA = 0.0;
  float sumInvD = 0.0;
  vec2 offs[4] = vec2[4](vec2(-0.25, -0.25), vec2(0.25, -0.25), vec2(-0.25, 0.25), vec2(0.25, 0.25));
  for (int k = 0; k < 4; k++) {
    vec2 px = px0 + offs[k];
    vec3 rd = normalize(uCam * vec3((px - 0.5 * uRes) / uF, 1.0));
    float d, occ, cga;
    acc += traceGlass(uRo, rd, px / uRes, d, occ, cga);
    minD = min(minD, d);
    minOcc = min(minOcc, occ);
    sumA += cga;
    if (cga > 0.5) sumInvD += 1.0 / max(d, 0.20);
  }
  oComp = vec4(clamp(acc * 0.25, 0.0, 1.0), 1.0);
  oDepth = vec4(minD, minOcc, 0.0, 1.0);
  oMeta = vec4(sumA * 0.25, sumInvD * 0.25, 1.0 / max(minOcc, 0.20), 1.0);
}
"""

# Pass 2: Instanced 3D Raymarched Crystal Satellite Beads (occluded by hero glass AND all 6 architectural walls/pillars!)
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

uniform sampler2D uBase;    /* Pass-1 FBO color (bottom-up) */
uniform sampler2D uDepth;   /* Pass-1 FBO depth: R = min(tGlass, tOcc), G = tOcc (bottom-up) */
uniform vec2 uRes;
uniform float uF, iTime, uIor, uDisp, uProxyDist;
uniform vec3 uRo, uSun, uTint;
uniform mat3 uCam;

layout(location=0) out vec4 oCol;
layout(location=1) out vec4 oMeta;   /* blended R: cgAlpha, G: cgInvDepth, B: occInvDepth */

float smin(float a, float b, float k) {
  float h = max(k - abs(a - b), 0.0) / max(k, 1e-5);
  return min(a, b) - h * h * k * 0.25;
}

float sdBead(vec3 p) {
  vec3 q = p - vCenter;
  float along = dot(q, vAxis);
  vec3 perp = q - vAxis * along;
  float taper = 1.0 + 0.18 * clamp(along / max(vRad * vAsp, 1e-4), -1.0, 1.0);
  vec3 qs = perp / max(taper, 0.65) + vAxis * (along / vAsp);
  float d = (length(qs) - vRad) * min(1.0, vAsp * 0.72);

  if (vSeed.z > 0.05) {
    vec3 off = vAxis * (vRad * vAsp * 0.72) + normalize(cross(vAxis, vec3(0.1, 0.2, 1.0))) * (vRad * 0.28);
    float d2 = length(q - off) - vRad * (0.58 * vSeed.z + 0.25);
    d = smin(d, d2, vRad * 0.55);
  }
  float ph = vSeed.x * 19.3;
  float wob = sin(q.x / vRad * 2.6 + ph) * cos(q.y / vRad * 2.4 - ph * 0.7) * sin(q.z / vRad * 2.5 + ph * 1.3);
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
  float heroD = dep.r;   /* min(tHeroGlass, tOcc) */
  float occD  = dep.g;   /* exact real wall/pillar depth */

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

# Pass 3: Camera Optical Finish (Depth-Aware Thin-Lens Bokeh Defocus + OLPF Softness + Halation + Edge CA + Sensor Grain)
POST_FS = """#version 330
uniform sampler2D uScene;     /* Pass-2 composited scene (bottom-up) */
uniform sampler2D uMeta;      /* R: cgAlpha, G: cgInvDepth, B: occInvDepth (bottom-up) */
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

/* Physical thin-lens Circle of Confusion (px) from inverse depth (1/m) */
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

  /* 1. Depth-aware 2-ring Golden-Angle Vogel spiral bokeh + OLPF optical softness
        - Never blurs untouched background plate pixels
        - Never bleeds background glass over a closer white pillar/wall (dInvJ > occInv0)!
        - Foreground glass close to the lens blooms softly over the sharp background behind it! */
  const float GA = 2.39996323;
  vec3 cgSum = vec3(0.0);
  float cgW = 0.0;
  float haloCov = 0.0;
  vec3 halation = vec3(0.0);

  /* Ring A: tight 10-tap precision optical kernel around center (0..max(coc0, 2.2) px) */
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
      /* Do NOT pull color from a closer solid wall/pillar into background glass! */
      if (occInvJ > dInv0 + 0.02) continue;
      float w = smoothstep(coc0 + 0.65, coc0 - 0.65, f * rA);
      if (w <= 0.001) continue;
      vec3 cj = texture(uScene, suv).rgb;
      /* Weight CG taps strongly, allow slight background wrap when strongly defocused near lens */
      float bgAllow = smoothstep(1.6, 5.5, coc0) * 0.55;
      float tw = w * mix(bgAllow, 1.0, mj.r);
      cgSum += cj * tw;
      cgW += tw;
    }
  }

  /* Ring B: wide 24-tap foreground bokeh scatter-gather + subtle glass specular halation (1.2..uMaxCoc px) */
  float fgHaloW = 0.0;
  vec3 fgHaloCol = vec3(0.0);
  float maxFgCoc = 0.0;
  for (int k = 0; k < 24; k++) {
    float f = sqrt((float(k) + 0.5) / 24.0);
    float ang = float(k) * GA + 0.5;
    float r = mix(1.0, uMaxCoc, f);
    vec2 off = vec2(cos(ang), sin(ang)) * r;
    vec2 suv = clamp(uv + off * pxSize, vec2(0.001), vec2(0.999));
    vec3 mj = texture(uMeta, suv).rgb;
    if (mj.r < 0.01) continue;
    float dInvJ = mj.g / mj.r;
    /* Strict architectural holdout check: glass BEHIND wall/pillar at center pixel CANNOT bleed over pillar! */
    if (dInvJ <= occInv0 + 0.015) continue;
    float cocJ = calcCoc(dInvJ);
    vec3 cj = texture(uScene, suv).rgb;

    /* Specular halation bloom on bright glass highlights (> 0.80 luma) */
    float lum = dot(cj, vec3(0.299, 0.587, 0.114));
    if (lum > 0.78) {
      float hw = exp(-r * r / 28.0) * (lum - 0.78) * mj.r;
      halation += cj * hw;
    }

    /* Foreground bokeh disk spread (only when tap is significantly defocused & in front of center) */
    if (cocJ > 1.35 && r <= cocJ + 0.85 && (cgA0 < 0.02 || dInvJ > dInv0 + 0.08)) {
      float disk = smoothstep(cocJ + 0.85, cocJ - 0.85, r);
      float normArea = (uMaxCoc * uMaxCoc) / max(cocJ * cocJ * 24.0, 1.0);
      float w = disk * mj.r * min(normArea, 0.35);
      fgHaloCol += cj * w;
      fgHaloW += w;
      maxFgCoc = max(maxFgCoc, cocJ);
    }
  }

  /* Subtle lens radial chromatic aberration at frame corners on CG pixels */
  vec3 cgCol = (cgW > 0.001) ? (cgSum / cgW) : col0;
  if (cgA0 > 0.05) {
    vec2 radV = uv - 0.5;
    float r2 = dot(radV, radV);
    vec2 caOff = radV * (r2 * 1.85) * pxSize;
    float rS = texture(uScene, clamp(uv + caOff, 0.001, 0.999)).r;
    float bS = texture(uScene, clamp(uv - caOff, 0.001, 0.999)).b;
    /* Only apply radial CA if offset tap isn't a closer wall/pillar */
    float mR = texture(uMeta, clamp(uv + caOff, 0.001, 0.999)).r;
    float mB = texture(uMeta, clamp(uv - caOff, 0.001, 0.999)).r;
    cgCol.r = mix(cgCol.r, rS, 0.45 * mR);
    cgCol.b = mix(cgCol.b, bS, 0.45 * mB);
  }

  /* Combine base CG + defocused foreground bokeh halo over untouched plate */
  vec3 res = mix(col0, cgCol, clamp(cgA0, 0.0, 1.0));
  float fgAlpha = clamp(fgHaloW * 1.15, 0.0, 0.92);
  if (fgHaloW > 0.001) {
    vec3 haloRgb = fgHaloCol / fgHaloW;
    /* Blend foreground bokeh smoothly: strong on background pixels, gentle on already-covered CG */
    float blendFg = fgAlpha * (1.0 - cgA0 * 0.55);
    res = mix(res, haloRgb, blendFg);
  }

  /* Add subtle warm optical halation around bright speculars */
  float cgPresence = clamp(max(cgA0, fgAlpha), 0.0, 1.0);
  res += halation * vec3(1.02, 0.98, 0.93) * 0.055;

  /* Filmic black-floor lift + highlight roll-off + plate-matched animated organic sensor grain ONLY on CG! */
  if (cgPresence > 0.005) {
    /* Soft black floor matching real Sony/Canon Rec.709 plate (0.017) + gentle highlight knee */
    vec3 graded = res * (1.0 - uBlackFloor) + vec3(uBlackFloor * 0.95, uBlackFloor * 0.98, uBlackFloor * 1.05);
    graded = graded / (1.0 + 0.055 * max(graded - 0.86, vec3(0.0)));

    /* Animated 2-scale organic camera sensor grain (fine 1px + soft 2px H.264-like micro-texture) */
    vec2 p1 = gl_FragCoord.xy + vec2(uFrame * 17.3, uFrame * 31.7);
    vec2 p2 = floor(gl_FragCoord.xy * 0.5) + vec2(uFrame * 11.1, uFrame * 23.9);
    float g1 = (hash12(p1) + hash12(p1 + 7.13) - 1.0);
    float g2 = (hash12(p2) + hash12(p2 + 5.71) - 1.0);
    float gLuma = (g1 * 0.62 + g2 * 0.38);
    vec3 gChroma = vec3(hash12(p2 + 13.7) - 0.5, hash12(p2 + 29.3) - 0.5, hash12(p2 + 47.1) - 0.5) * 0.28;
    float lum = dot(graded, vec3(0.299, 0.587, 0.114));
    /* Real camera sensor grain is strongest in shadows/midtones and tapers smoothly in clipped highlights */
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


def encode(c, out_dir="out_dense_v2", suffix="_Dense_v2", title_label="3D Liquid Glass Gallery (v2 Occluded + Optics)"):
    out = c["dir"] / out_dir
    out.mkdir(exist_ok=True)
    nm = c["name"] + suffix
    r = f"{c['fps_num']}/{c['fps_den']}"
    ff = find_ffmpeg()
    tags = ["-pix_fmt", "yuv420p", "-color_primaries", "bt709", "-color_trc", "bt709", "-colorspace", "bt709", "-movflags", "+faststart"]
    x264 = ["-c:v", "libx264", "-profile:v", "high", "-level", "4.1", "-crf", "14", "-preset", "slow"]
    run = lambda a: subprocess.run([ff, "-v", "error", "-y"] + a, check=True)
    run(["-framerate", r, "-i", str(work(c, "comp_dense_v2", "%04d.png"))] + x264 + tags + [str(out / f"{nm}_Composite.mp4")])
    lab = lambda t: f"drawtext=fontfile='C\\:/Windows/Fonts/arialbd.ttf':text='{t}':x=w-tw-28:y=h-th-24:fontsize=40:fontcolor=white:shadowcolor=black@0.6:shadowx=2:shadowy=2"
    run([
        "-framerate", r, "-i", str(work(c, "comp_dense_v2", "%04d.png")),
        "-framerate", r, "-i", str(work(c, "frames", "%04d.jpg")),
        "-filter_complex",
        f"[0]scale=1080:608,{lab(title_label)}[a];[1]scale=1080:608,{lab('Before')}[b];[a][b]vstack,pad=1080:1350:0:67:color=black",
    ] + x264 + tags + [str(out / f"{nm}_Compare_1080x1350.mp4")])
    print("encoded ->", out)


def build_gallery_beads(cam_path, clusters, flutes, occ_boxes):
    """Generate ~185 world-locked multi-scale crystal beads strictly outside all real walls/pillars & clear of camera pinhole."""
    rng = np.random.default_rng(2026)
    beads = []

    def clears_cam_and_walls(pos, rad):
        # 1. Camera pinhole clearance
        d = np.linalg.norm(cam_path - pos[None, :], axis=1).min()
        if d <= (0.34 + rad * 2.0):
            return False
        # 2. Never embed a bead inside any real wall or structural column!
        pad = rad * 1.65 + 0.12
        for (x0, x1, y0, y1, z0, z1) in occ_boxes:
            if (x0 - pad <= pos[0] <= x1 + pad and
                y0 - pad <= pos[1] <= y1 + pad and
                z0 - pad <= pos[2] <= z1 + pad):
                return False
        return True

    # 1. Orbital halos clustered around every hero sculpture (5 clusters + 4 fluted monoliths)
    anchors = [(np.array(cl[:3]), cl[3] * 1.25) for cl in clusters] + [(np.array(fl[:3]), 1.15) for fl in flutes]
    for anc, spread in anchors:
        count = 13
        got = 0
        for _ in range(240):
            if got >= count:
                break
            off = rng.normal(0.0, 1.0, 3)
            off /= max(np.linalg.norm(off), 1e-4)
            dist = rng.uniform(0.42, 1.48) * spread
            pos = anc + off * dist
            pos[2] = float(np.clip(pos[2], 0.24, 2.85))
            pos[0] = float(np.clip(pos[0], -2.35, 2.35))
            rad = float(np.exp(rng.uniform(math.log(0.013), math.log(0.082))))
            if not clears_cam_and_walls(pos, rad):
                continue
            axis = rng.normal(0, 1, 3)
            axis /= max(np.linalg.norm(axis), 1e-4)
            asp = float(rng.uniform(1.05, 1.70))
            wob = float(rng.uniform(0.03, 0.11))
            lobe = float(rng.uniform(0.0, 0.85)) if rng.random() < 0.45 else 0.0
            beads.append((pos, rad, axis, asp, float(rng.uniform(0, 10)), wob, lobe, float(rng.uniform(0, 6.28))))
            got += 1

    # 2. Flowing architectural ribbon + left/right depth fillers (strictly inside entry hallway for y < -8.5!)
    for _ in range(700):
        if len(beads) >= 188:
            break
        y = float(rng.uniform(-10.8, 5.0))
        u_hall = np.clip((y + 11.0) / 15.8, 0.0, 1.0)
        if y < -8.45:
            # Inside the front entryway corridor: strictly between left entry wall (-0.71) & right wall (+2.28)!
            x = float(rng.uniform(-0.52, 1.75))
            z = float(rng.uniform(0.32, 1.95))
        elif rng.random() < 0.44:
            x = float(1.15 * math.sin(y * 0.65) + rng.normal(0.0, 0.42))
            z = float(0.65 + 1.05 * u_hall + 0.42 * math.cos(y * 0.85) + rng.normal(0.0, 0.28))
        else:
            side = -1.0 if rng.random() < 0.5 else 1.0
            x = float(side * rng.uniform(0.48, 2.15))
            z = float(rng.uniform(0.28 + 0.50 * u_hall, 1.85 + 0.95 * u_hall))
        z = float(np.clip(z, 0.22, 2.95))
        pos = np.array([x, y, z], np.float64)
        rad = float(np.exp(rng.uniform(math.log(0.011), math.log(0.066))))
        if not clears_cam_and_walls(pos, rad):
            continue
        axis = rng.normal(0, 1, 3)
        axis /= max(np.linalg.norm(axis), 1e-4)
        beads.append((pos, rad, axis, float(rng.uniform(1.05, 1.60)), float(rng.uniform(0, 10)),
                      float(rng.uniform(0.03, 0.10)), float(rng.uniform(0.0, 0.72)) if rng.random() < 0.38 else 0.0,
                      float(rng.uniform(0, 6.28))))
    print(f"Dense gallery v2: 5 hero clusters + 4 fluted monoliths + {len(beads)} beads + {len(occ_boxes)} 3D architectural occluders")
    return beads


def main():
    c = load(sys.argv[1] if len(sys.argv) > 1 and not sys.argv[1].startswith("--") else here / "piece_glass")
    S = scene(c)
    W, H, N, fps = c["W"], c["H"], c["N"], c["fps"]
    f = float(S["f"])
    g = c.get("glass", {})
    stills = [int(x) for x in sys.argv[sys.argv.index("--still") + 1].split(",")] if "--still" in sys.argv else None
    out_dir = sys.argv[sys.argv.index("--out") + 1] if "--out" in sys.argv else "out_dense_v2"

    cam_path = np.asarray(S["cam"], np.float64)

    # 6 COLMAP-triangulated & visually verified 3D architectural holdout boxes (walls + structural columns)
    occ_boxes = [
        (-4.00, -0.71, -13.50, -8.56, 0.0, 5.0),  # Left entry hallway solid wall (with mirror & switch)
        ( 2.28,  4.50, -13.50, -8.85, 0.0, 5.0),  # Right entry hallway solid wall (with navy door)
        (-1.50, -0.90,  -1.12, -0.57, 0.0, 5.0),  # Left white structural pillar #1 (foreground/mid)
        ( 2.24,  2.82,  -1.00, -0.42, 0.0, 5.0),  # Right white structural pillar #1 (beside stairs)
        (-1.56, -1.07,   7.10,  7.65, 0.0, 5.0),  # Left white structural pillar #2 (far hall)
        ( 1.54,  2.05,   7.10,  7.65, 0.0, 5.0),  # Right white structural pillar #2 (far hall)
    ]

    # 5 staggered liquid-glass metaball clusters (ALL physically inside open air, NEVER buried inside walls/pillars!)
    # Notice cluster[0] now sits cleanly INSIDE the open front entryway corridor (x=-0.16, y=-9.15, z=1.48)
    # so frame 0..55 has huge authentic 3D hallway parallax & depth!
    # Notice cluster[3] sits genuinely BEHIND L_Pillar1 so it is cleanly occluded by the pillar and reveals as camera passes!
    clusters = [
        (-0.16, -9.15, 1.48, 0.52),  # Front-upper-left INSIDE real entryway hallway!
        ( 1.12, -8.10, 0.96, 0.86),  # Front-right past entryway opening
        ( 1.18, -4.10, 1.32, 0.92),  # Mid-right beside wooden staircase
        (-1.26,  1.35, 1.58, 0.94),  # Far-left lounge (cleanly behind L_Pillar1 until f~190!)
        ( 0.06,  3.85, 1.64, 1.38),  # Grand center-hall climax sculpture
    ]
    flutes_pos = [
        (-1.12, -5.85, 1.06, math.radians(17.0)),  # Front-left lounge (peeks out cleanly past LeftEntryWall corner!)
        ( 1.28, -6.25, 1.18, math.radians(-22.0)), # Front-right tall narrow fluted fin
        (-0.92, -2.18, 1.32, math.radians(20.0)),  # Mid-left hero fluted monolith (1.0m IN FRONT of L_Pillar1!)
        ( 1.24,  0.75, 1.66, math.radians(-19.0)), # Far-right dining fluted slab
    ]
    flutes_size = [
        (0.36, 0.044, 0.78),
        (0.24, 0.040, 0.74),
        (0.44, 0.050, 0.94),
        (0.38, 0.046, 0.82),
    ]
    for idx, cl in enumerate(clusters):
        md = np.linalg.norm(cam_path - np.array(cl[:3]), axis=1).min()
        print(f"  cluster[{idx}] min_cam_dist={md:.2f}m (rad~{cl[3]*0.6:.2f}m)")
    for idx, fl in enumerate(flutes_pos):
        md = np.linalg.norm(cam_path - np.array(fl[:3]), axis=1).min()
        print(f"  flute[{idx}]   min_cam_dist={md:.2f}m")

    beads = build_gallery_beads(cam_path, clusters, flutes_pos, occ_boxes)

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
    inst_buf = ctx.buffer(reserve=len(beads) * 12 * 4)
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
    fbo_hero = ctx.framebuffer([t_hero, t_depth, t_meta0])

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
    hero_prog["uClusters"].value = clusters
    hero_prog["uFlutePos"].value = flutes_pos
    hero_prog["uFluteSize"].value = flutes_size
    hero_prog["uOccMin"].value = [(b[0], b[2], b[4]) for b in occ_boxes]
    hero_prog["uOccMax"].value = [(b[1], b[3], b[5]) for b in occ_boxes]
    hero_prog["uNumOcc"].value = len(occ_boxes)
    hero_prog["uIor"].value = float(g.get("ior", 1.46))
    hero_prog["uDisp"].value = float(g.get("dispersion", 0.018))
    hero_prog["uTint"].value = tuple(map(float, g.get("tint", [0.075, 0.025, 0.018])))
    hero_prog["uCausticGain"].value = float(g.get("caustic_gain", 1.05))
    hero_prog["uShadowGain"].value = float(g.get("shadow_gain", 0.34))
    hero_prog["uFluteFreq"].value = float(g.get("flute_freq", 56.0))
    hero_prog["uFluteAmp"].value = float(g.get("flute_amp", 0.19))
    hero_prog["uMetaK"].value = float(g.get("metaball_k", 0.32))
    hero_prog["uProxyDist"].value = float(g.get("proxy_dist", 7.0))

    bead_prog["uRes"].value = (float(W), float(H))
    bead_prog["uF"].value = f
    bead_prog["uSun"].value = tuple(map(float, S["sun"]))
    bead_prog["uIor"].value = float(g.get("ior", 1.46))
    bead_prog["uDisp"].value = float(g.get("dispersion", 0.018))
    bead_prog["uTint"].value = tuple(map(float, g.get("tint", [0.075, 0.025, 0.018])))
    bead_prog["uProxyDist"].value = float(g.get("proxy_dist", 7.0))

    # Camera optical finish parameters tuned to Pexels 7578546 (wide cinema lens, Rec.709 black floor ~0.016)
    post_prog["uRes"].value = (float(W), float(H))
    post_prog["uFocusDist"].value = 4.2
    post_prog["uNearBlur"].value = 4.4
    post_prog["uFarBlur"].value = 5.5
    post_prog["uBaseSoft"].value = 0.72
    post_prog["uMaxCoc"].value = 10.5
    post_prog["uGrainAmt"].value = 0.012
    post_prog["uBlackFloor"].value = 0.016

    def render_one(i):
        pl = frame(c, i)
        t_sec = i / fps
        ro = np.asarray(S["cam"][i], np.float64)
        R = np.asarray(S["rot"][i], np.float64)
        Rt = tuple(R.astype(np.float32).T.flatten())

        # Pass 1: Hero clusters + fluted slabs + 3D architectural occlusion
        plate_t.write(pl[:, :, ::-1].tobytes())
        plate_t.use(0)
        hero_prog["uPlate"].value = 0
        hero_prog["uRo"].value = tuple(map(float, ro))
        hero_prog["uCam"].value = Rt
        hero_prog["iTime"].value = t_sec

        fbo_hero.use()
        ctx.disable(moderngl.BLEND)
        va_hero.render(moderngl.TRIANGLES)

        # Copy Pass-1 color + meta into Pass-2 FBO
        fbo_scene.use()
        t_hero.use(0)
        t_meta0.use(1)
        copy2_prog["uSrc0"].value = 0
        copy2_prog["uSrc1"].value = 1
        va_copy2.render(moderngl.TRIANGLES)

        # Pass 2: Depth-sorted 3D crystal beads occluded against BOTH hero glass AND all 6 architectural walls/pillars
        rows_b = []
        for pos, rad, axis, asp, s_w, w_amp, lobe, ph in beads:
            p_now = pos + np.array([0.025 * math.sin(t_sec * 0.65 + ph),
                                    0.025 * math.cos(t_sec * 0.55 + ph * 1.2),
                                    0.040 * math.sin(t_sec * 0.80 + ph * 0.8)])
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
            rows.append(np.hstack([t_im(pl, f"plate {i}"), t_im(comp, f"dense v2 (occ+optics) {i}")]))
        out_p = work(c, "still_dense_v2_sheet.jpg")
        cv2.imwrite(str(out_p), np.vstack(rows), [cv2.IMWRITE_JPEG_QUALITY, 92])
        if hero is not None:
            cv2.imwrite(str(work(c, "still_dense_v2_hero.jpg")), hero, [cv2.IMWRITE_JPEG_QUALITY, 94])
        print("stills ->", out_p)
        return

    comp_dir = work(c, "comp_dense_v2")
    shutil.rmtree(comp_dir, ignore_errors=True)
    comp_dir.mkdir(parents=True)
    ts = time.time()
    for i in range(N):
        _, comp = render_one(i)
        cv2.imwrite(str(comp_dir / f"{i:04d}.png"), comp, [cv2.IMWRITE_PNG_COMPRESSION, 1])
        if i % 45 == 0 or i == N - 1:
            print(f"dense v2 frame {i}/{N} ({time.time() - ts:.1f}s)", flush=True)
    encode(c, out_dir=out_dir)


if __name__ == "__main__":
    main()
