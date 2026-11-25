"""3D Liquid Glass & Fluted Optics Gallery in a COLMAP-tracked architectural walkthrough (ModernGL, OpenGL 3.3).

    python render_glass.py [piece_dir] --still 30,85,140,235     quick look -> work/still_sheet.jpg
    python render_glass.py [piece_dir] --out out_v2 --passes     full render + breakdown passes -> out_v2/*.mp4

Raymarches three world-locked 3D optical glass installations directly in the solved COLMAP floor-plane coordinates
(metres, z up from floor). No near-camera fade/dissolve: foreground sculptures stay 100% solid optical glass as they
sweep cleanly past the camera lens.
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

FS = """#version 330
uniform sampler2D uPlate;
uniform vec2 uRes;
uniform float uF, iTime;
uniform vec3 uRo, uSun;
uniform mat3 uCam;
uniform vec4 uCluster1;     /* xyz = center (m), w = scale */
uniform vec4 uCluster2;     /* xyz = center (m), w = scale */
uniform vec3 uFlutePos;     /* fluted glass slab centre (m) */
uniform vec3 uFluteSize;    /* half-extents (hx, hy, hz) */
uniform float uFluteYaw;    /* rotation around Z (rad) */
uniform float uIor, uDisp, uCausticGain, uShadowGain, uFluteFreq, uFluteAmp, uMetaK, uProxyDist;
uniform vec3 uTint;

/* 4 simultaneous MRT outputs so all breakdown passes render in ONE single GPU pass! */
layout(location=0) out vec4 oComp;      /* final 3D liquid glass composite */
layout(location=1) out vec4 oNormal;    /* 3D SDF surface normals over full-brightness plate */
layout(location=2) out vec4 oRefract;   /* optical refraction warp-grid over plate */
layout(location=3) out vec4 oCaustic;   /* 3D floor caustic focusing + shadow pass */

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

float sdMetaballCluster(vec3 p, vec4 cfg, float phase) {
  vec3 q = (p - cfg.xyz) / cfg.w;
  float t = iTime * 0.88 + phase;

  vec3 c0 = vec3(0.06 * sin(t * 0.7), 0.05 * cos(t * 0.6), 0.08 * sin(t * 0.9));
  float d = length(q - c0) - 0.32;

  vec3 c1 = vec3( 0.44 * cos(t * 0.95),        0.34 * sin(t * 0.80),  0.22 * sin(t * 1.15));
  vec3 c2 = vec3(-0.40 * cos(t * 0.78 + 1.2),  0.38 * sin(t * 0.92), -0.20 * cos(t * 1.05));
  vec3 c3 = vec3( 0.30 * sin(t * 1.22 + 2.4), -0.32 * cos(t * 0.85),  0.35 * cos(t * 0.90));
  vec3 c4 = vec3(-0.32 * sin(t * 0.90 + 4.0), -0.26 * sin(t * 1.10), -0.28 * sin(t * 0.82));
  vec3 c5 = vec3( 0.56 * cos(t * 0.62 + 3.1),  0.18 * cos(t * 1.30),  0.10 * sin(t * 1.40));
  vec3 c6 = vec3(-0.52 * sin(t * 0.68 + 0.7), -0.16 * cos(t * 1.18),  0.28 * cos(t * 1.12));

  float k = uMetaK;
  d = smin(d, length(q - c1) - 0.20, k);
  d = smin(d, length(q - c2) - 0.18, k);
  d = smin(d, length(q - c3) - 0.17, k);
  d = smin(d, length(q - c4) - 0.16, k);
  d = smin(d, length(q - c5) - 0.14, k * 0.85);
  d = smin(d, length(q - c6) - 0.13, k * 0.85);

  float rip = (vnoise(q * 3.2 + vec3(0.0, t * 0.42, -t * 0.32)) - 0.5) * 0.020;
  return (d + rip) * cfg.w;
}

float sdFlutedSlab(vec3 p, out vec3 localP) {
  vec3 q = p - uFlutePos;
  q.z -= 0.04 * sin(iTime * 0.85 + 1.1);
  q.xy = rot2(-uFluteYaw) * q.xy;
  localP = q;
  float box = sdRoundBox(q, uFluteSize, 0.036);
  float faceMask = smoothstep(0.02, 0.08, uFluteSize.x - abs(q.x)) * smoothstep(0.02, 0.08, uFluteSize.z - abs(q.z));
  float flute = sin(q.x * uFluteFreq) * 0.005 * faceMask;
  return box + flute;
}

vec2 mapScene(vec3 p) {
  float d1 = sdMetaballCluster(p, uCluster1, 0.0);
  float d2 = sdMetaballCluster(p, uCluster2, 2.7);
  float dMeta = min(d1, d2);
  vec3 lp;
  float dFlute = sdFlutedSlab(p, lp);
  if (dFlute < dMeta) return vec2(dFlute, 1.0);
  return vec2(dMeta, 0.0);
}

float mapDist(vec3 p) {
  return mapScene(p).x;
}

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

vec2 backdropUV(vec3 ro, vec3 rd) {
  float tf = (rd.z < -0.012) ? (-ro.z / rd.z) : 1e4;
  if (tf < 0.05 || tf > 26.0) tf = 1e4;
  float tHit = min(tf, uProxyDist);
  vec3 X = ro + rd * tHit;
  float zc;
  vec2 uv = projUV(X, zc);
  if (zc < 0.1) uv = clamp(vec2(1.0 - uv.x, uv.y), 0.01, 0.99);
  uv = abs(fract(uv * 0.5 + 0.5) * 2.0 - 1.0);
  return clamp(uv, vec2(0.008), vec2(0.992));
}

vec3 sampleBackdrop(vec3 ro, vec3 rd, float blurPx, out vec2 hitUV) {
  vec2 uv = backdropUV(ro, rd);
  hitUV = uv;
  if (blurPx <= 0.25) return texture(uPlate, uv).rgb;
  vec2 texel = blurPx / uRes;
  vec3 acc = texture(uPlate, uv).rgb * 0.28;
  acc += texture(uPlate, clamp(uv + vec2( texel.x,  texel.y * 0.5), 0.008, 0.992)).rgb * 0.18;
  acc += texture(uPlate, clamp(uv + vec2(-texel.x,  texel.y * 0.5), 0.008, 0.992)).rgb * 0.18;
  acc += texture(uPlate, clamp(uv + vec2( texel.x, -texel.y * 0.5), 0.008, 0.992)).rgb * 0.18;
  acc += texture(uPlate, clamp(uv + vec2(-texel.x, -texel.y * 0.5), 0.008, 0.992)).rgb * 0.18;
  return acc;
}

bool hitSphere(vec3 ro, vec3 rd, vec3 cen, float rad) {
  vec3 oc = cen - ro;
  float b = dot(oc, rd);
  float c = dot(oc, oc) - rad * rad;
  return (b > -rad) && (b * b - c > 0.0);
}

void floorCaustic(vec3 Xf, out float caustic, out float shadow) {
  caustic = 0.0;
  shadow = 0.0;
  vec3 sun = normalize(uSun);
  vec3 sh1 = uCluster1.xyz - sun * (uCluster1.z / max(sun.z, 0.20));
  vec3 sh2 = uFlutePos     - sun * (uFlutePos.z / max(sun.z, 0.20));
  vec3 sh3 = uCluster2.xyz - sun * (uCluster2.z / max(sun.z, 0.20));
  float d1 = length(Xf.xy - sh1.xy);
  float d2 = length(Xf.xy - sh2.xy);
  float d3 = length(Xf.xy - sh3.xy);
  float g1 = exp(-dot(Xf.xy - uCluster1.xy, Xf.xy - uCluster1.xy) / (0.65 * uCluster1.w * uCluster1.w)) * 0.22;
  float g2 = exp(-dot(Xf.xy - uFlutePos.xy, Xf.xy - uFlutePos.xy) / 0.55) * 0.24;
  float g3 = exp(-dot(Xf.xy - uCluster2.xy, Xf.xy - uCluster2.xy) / (0.65 * uCluster2.w * uCluster2.w)) * 0.20;
  shadow = max(g1, max(g2, g3));

  if (min(d1, min(d2, d3)) > 2.6) return;

  float t = 0.05;
  bool hit = false;
  float minD = 1e4;
  float hitMat = 0.0;
  for (int i = 0; i < 52; i++) {
    vec3 p = Xf + sun * t;
    if (p.z > 3.6) break;
    vec2 h = mapScene(p);
    minD = min(minD, h.x);
    if (h.x < 0.012) {
      hit = true;
      hitMat = h.y;
      break;
    }
    t += max(h.x * 0.85, 0.025);
  }
  float penumbra = 1.0 - smoothstep(0.0, 0.34, minD);
  shadow = max(shadow, penumbra * 0.58);

  if (hit) {
    for (int k = 0; k < 3; k++) {
      t += mapDist(Xf + sun * t);
    }
    vec3 P = Xf + sun * t;
    vec3 N = calcNormal(P);
    if (hitMat > 0.5) {
      vec3 lp;
      sdFlutedSlab(P, lp);
      float edgeFade = smoothstep(0.0, 0.11, uFluteSize.x - abs(lp.x)) * smoothstep(0.0, 0.13, uFluteSize.z - abs(lp.z));
      float bands = pow(0.5 + 0.5 * cos(lp.x * uFluteFreq * 2.0), 2.8);
      caustic = pow(edgeFade, 1.25) * (0.22 + 1.45 * bands);
    } else {
      float nds = max(abs(dot(N, sun)), 0.0);
      vec3 r0 = refract(-sun, N, 1.0 / uIor);
      vec2 focusPt = P.xy + r0.xy * (-P.z / min(r0.z, -0.15));
      float n1 = vnoise(vec3(focusPt * 3.8, iTime * 0.55));
      float n2 = vnoise(vec3(focusPt * 7.2 + vec2(3.1, -1.7), -iTime * 0.70));
      float filament = pow(clamp(1.0 - abs(n1 - 0.5) * 2.6, 0.0, 1.0), 2.2) * 0.85
                     + pow(clamp(1.0 - abs(n2 - 0.5) * 2.8, 0.0, 1.0), 2.4) * 0.55;
      float coreGlow = pow(nds, 2.0) * 0.72;
      caustic = (coreGlow + filament * (0.45 + 0.65 * nds)) * smoothstep(0.06, 0.36, nds);
    }
    shadow = mix(0.62, 0.06, clamp(caustic * 0.55, 0.0, 1.0));
  }
}

/* optical warp grid helper for breakdown refraction pass */
float warpGrid(vec2 uv) {
  vec2 g = uv * uRes / 48.0;
  vec2 f = abs(fract(g + 0.5) - 0.5) * 48.0;
  float line = 1.0 - smoothstep(0.8, 2.2, min(f.x, f.y));
  return line;
}

void traceGlass(vec3 ro, vec3 rd, vec2 uv0, out vec3 cComp, out vec3 cNorm, out vec3 cRefr, out vec3 cCaus) {
  vec3 plateCol = texture(uPlate, uv0).rgb;
  cNorm = plateCol;
  float gl0 = warpGrid(uv0);
  cRefr = mix(plateCol, vec3(0.95, 0.98, 1.0), gl0 * 0.28);
  cCaus = plateCol;

  float tFloor = (rd.z < -0.004) ? (-ro.z / rd.z) : 1e4;
  if (tFloor < 0.05 || tFloor > 24.0) tFloor = 1e4;

  bool nearAny =
    hitSphere(ro, rd, uCluster1.xyz, 1.32 * uCluster1.w) ||
    hitSphere(ro, rd, uCluster2.xyz, 1.32 * uCluster2.w) ||
    hitSphere(ro, rd, uFlutePos, length(uFluteSize) + 0.24);

  float tHit = 1e4;
  float matId = -1.0;
  if (nearAny) {
    /* start very close (0.04m) so approaching glass stays 100% solid until it exits the frame */
    float t = 0.04;
    float tMax = min(tFloor + 0.05, 22.0);
    for (int i = 0; i < 96; i++) {
      vec3 p = ro + rd * t;
      vec2 h = mapScene(p);
      if (h.x < 0.002) {
        tHit = t;
        matId = h.y;
        break;
      }
      t += max(h.x * 0.88, 0.004);
      if (t > tMax) break;
    }
  }

  vec3 bgCol = plateCol;
  if (tFloor < 22.0) {
    vec3 Xf = ro + rd * tFloor;
    float caust, shad;
    floorCaustic(Xf, caust, shad);
    if (caust > 0.001 || shad > 0.001) {
      float luma = dot(plateCol, vec3(0.299, 0.587, 0.114));
      float floorMask = smoothstep(0.08, 0.24, luma) * (1.0 - smoothstep(14.0, 21.0, tFloor));
      vec3 sunlit = vec3(1.0, 0.94, 0.82);
      bgCol = bgCol * (1.0 - shad * uShadowGain * floorMask);
      bgCol += sunlit * (caust * uCausticGain * 0.30 * floorMask) * (0.42 + 0.58 * bgCol);
      /* caustic breakdown pass: full-gain plate + vivid golden caustic emphasis */
      vec3 cHi = plateCol * (1.0 - shad * 0.45 * floorMask) + vec3(1.0, 0.82, 0.32) * (caust * 0.85 * floorMask);
      cCaus = clamp(cHi, 0.0, 1.0);
    }
  }

  if (matId < -0.5) {
    cComp = bgCol;
    return;
  }

  /* --- HIT 3D GLASS SURFACE (NO near-camera fade/dissolve!) --- */
  vec3 p1 = ro + rd * tHit;
  vec3 n1 = calcNormal(p1);
  bool isFlute = (matId > 0.5);

  float fluteCrest = 0.0;
  if (isFlute) {
    vec3 lp;
    sdFlutedSlab(p1, lp);
    float faceMask = smoothstep(0.018, 0.075, uFluteSize.x - abs(lp.x)) * smoothstep(0.018, 0.075, uFluteSize.z - abs(lp.z));
    float slope = cos(lp.x * uFluteFreq) * uFluteAmp * faceMask;
    fluteCrest = pow(abs(sin(lp.x * uFluteFreq)), 4.0) * faceMask;
    vec3 tangentX = vec3(cos(uFluteYaw), sin(uFluteYaw), 0.0);
    n1 = normalize(n1 + tangentX * slope);
  }

  /* pass 1: crisp world-space SDF normal map directly on the glass silhouette over full-gain plate */
  cNorm = 0.5 + 0.5 * n1;

  vec3 rdInG = refract(rd, n1, 1.0 / uIor);
  if (length(rdInG) < 0.001) rdInG = reflect(rd, n1);

  float thick = 0.08;
  vec3 p2 = p1 + rdInG * 0.05;
  for (int i = 0; i < 36; i++) {
    float dIn = -mapDist(p2);
    if (dIn < 0.003) break;
    float stepD = max(dIn * 0.85, 0.014);
    p2 += rdInG * stepD;
    thick += stepD;
    if (thick > 2.0) break;
  }
  vec3 n2 = -calcNormal(p2);

  float iorR = uIor - uDisp;
  float iorG = uIor;
  float iorB = uIor + uDisp;

  vec3 rdOutG = refract(rdInG, n2, iorG);
  if (length(rdOutG) < 0.001) {
    rdOutG = normalize(mix(rdInG, reflect(rdInG, n2), 0.35));
  }
  vec3 bend = rdOutG - rd;
  vec3 rdOutR = normalize(rd + bend * (iorR - 1.0) / max(iorG - 1.0, 0.05));
  vec3 rdOutB = normalize(rd + bend * (iorB - 1.0) / max(iorG - 1.0, 0.05));

  float blurPx = isFlute ? (3.6 + 2.4 * fluteCrest) : 0.35;
  vec2 uvR, uvG, uvB, uvEnv;
  float rCh = sampleBackdrop(p2, rdOutR, blurPx, uvR).r;
  float gCh = sampleBackdrop(p2, rdOutG, blurPx, uvG).g;
  float bCh = sampleBackdrop(p2, rdOutB, blurPx, uvB).b;
  vec3 refrCol = vec3(rCh, gCh, bCh);

  /* pass 2: optical refraction warp grid (shows exact Snell UV bend inside the glass!) */
  float glR = warpGrid(uvG);
  float bendMag = clamp(length(uvG - uv0) * 9.0, 0.0, 1.0);
  cRefr = mix(refrCol * 0.88, vec3(0.45, 0.92, 1.0), bendMag * 0.25);
  cRefr = mix(cRefr, vec3(1.0, 1.0, 1.0), glR * 0.82);

  vec3 absorb = exp(-thick * uTint * (isFlute ? 1.35 : 0.95));
  refrCol = refrCol * absorb * 1.03;

  float NdotV = max(dot(-rd, n1), 0.0);
  float fresnel = 0.035 + 0.965 * pow(1.0 - NdotV, 4.6);

  vec3 reflDir = reflect(rd, n1);
  vec3 envRefl = sampleBackdrop(p1 + n1 * 0.01, reflDir, isFlute ? 7.0 : 3.0, uvEnv);
  float skyUp = smoothstep(-0.1, 0.85, reflDir.z);
  envRefl = mix(envRefl * 0.95, envRefl * 1.12 + vec3(0.18, 0.19, 0.21), skyUp * 0.52);

  vec3 sunDir = normalize(uSun);
  vec3 H1 = normalize(sunDir - rd);
  float specSun = pow(max(dot(n1, H1), 0.0), isFlute ? 110.0 : 240.0) * (isFlute ? 0.95 : 1.55);
  float specBroad = pow(max(dot(n1, H1), 0.0), 24.0) * 0.16;

  vec3 ceilLight = normalize(vec3(0.04, 0.12, 0.98));
  vec3 H2 = normalize(ceilLight - rd);
  float specCeil = pow(max(dot(n1, H2), 0.0), 120.0) * 0.45;

  float rimGlow = pow(1.0 - NdotV, 3.2) * (0.34 + 0.28 * skyUp);

  float reflMix = clamp(fresnel * (isFlute ? 0.44 : 0.62), 0.0, 0.78);
  vec3 col = mix(refrCol, envRefl, reflMix);
  col += vec3(1.0, 0.97, 0.91) * (specSun + specBroad);
  col += vec3(0.95, 0.98, 1.0) * (specCeil + rimGlow);
  if (isFlute) {
    col += vec3(0.96, 0.98, 1.0) * fluteCrest * 0.085 * (0.35 + 0.65 * fresnel);
    vec3 lp; sdFlutedSlab(p1, lp);
    float edgeBevel = smoothstep(0.05, 0.0, min(uFluteSize.x - abs(lp.x), uFluteSize.z - abs(lp.z)));
    col += vec3(0.86, 0.95, 0.96) * edgeBevel * 0.16;
  }

  /* pass 3: glass silhouette tinted softly on caustic pass so caustics + glass relationship reads clearly */
  cCaus = mix(col, vec3(1.0, 0.95, 0.85), rimGlow * 0.35);
  cComp = col;
}

void main() {
  vec2 px0 = vec2(gl_FragCoord.x, uRes.y - gl_FragCoord.y);
  vec3 aComp = vec3(0.0), aNorm = vec3(0.0), aRefr = vec3(0.0), aCaus = vec3(0.0);
  vec2 offs[4] = vec2[4](vec2(-0.25, -0.25), vec2(0.25, -0.25), vec2(-0.25, 0.25), vec2(0.25, 0.25));
  for (int k = 0; k < 4; k++) {
    vec2 px = px0 + offs[k];
    vec2 uv = px / uRes;
    vec3 rd = normalize(uCam * vec3((px - 0.5 * uRes) / uF, 1.0));
    vec3 c0, c1, c2, c3;
    traceGlass(uRo, rd, uv, c0, c1, c2, c3);
    aComp += c0; aNorm += c1; aRefr += c2; aCaus += c3;
  }
  oComp    = vec4(clamp(aComp * 0.25, 0.0, 1.0), 1.0);
  oNormal  = vec4(clamp(aNorm * 0.25, 0.0, 1.0), 1.0);
  oRefract = vec4(clamp(aRefr * 0.25, 0.0, 1.0), 1.0);
  oCaustic = vec4(clamp(aCaus * 0.25, 0.0, 1.0), 1.0);
}
"""


def label(im, s):
    im = im.copy()
    cv2.putText(im, s, (12, im.shape[0] - 14), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 4, cv2.LINE_AA)
    cv2.putText(im, s, (12, im.shape[0] - 14), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 1, cv2.LINE_AA)
    return im


def encode(c, out_name="out_v2", suffix="_v2", title_label="3D Liquid Glass Optics"):
    out = c["dir"] / out_name
    out.mkdir(exist_ok=True)
    nm = c["name"] + suffix
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


def main():
    c = load(sys.argv[1] if len(sys.argv) > 1 and not sys.argv[1].startswith("--") else here / "piece_glass")
    S = scene(c)
    W, H, N, fps = c["W"], c["H"], c["N"], c["fps"]
    f = float(S["f"])
    g = c.get("glass", {})
    stills = [int(x) for x in sys.argv[sys.argv.index("--still") + 1].split(",")] if "--still" in sys.argv else None
    save_passes = "--passes" in sys.argv
    out_dir = sys.argv[sys.argv.index("--out") + 1] if "--out" in sys.argv else "out_v2"

    ctx = moderngl.create_standalone_context(require=330)
    prog = ctx.program(vertex_shader=VS, fragment_shader=FS)
    quad = ctx.buffer(np.array([-1, -1, 3, -1, -1, 3], "f4").tobytes())
    va = ctx.vertex_array(prog, [(quad, "2f", "p")])
    plate_t = ctx.texture((W, H), 3)
    plate_t.filter = (moderngl.LINEAR, moderngl.LINEAR)
    plate_t.repeat_x = plate_t.repeat_y = False
    t_comp, t_norm, t_refr, t_caus = [ctx.texture((W, H), 4, dtype="f4") for _ in range(4)]
    fbo = ctx.framebuffer([t_comp, t_norm, t_refr, t_caus])

    prog["uRes"].value = (float(W), float(H))
    prog["uF"].value = f
    prog["uSun"].value = tuple(map(float, S["sun"]))
    prog["uCluster1"].value = tuple(map(float, g.get("cluster1", [1.12, -7.6, 0.94, 0.86])))
    prog["uCluster2"].value = tuple(map(float, g.get("cluster2", [0.05, 3.65, 1.55, 1.28])))
    prog["uFlutePos"].value = tuple(map(float, g.get("flute_pos", [-0.88, -2.2, 1.22])))
    prog["uFluteSize"].value = tuple(map(float, g.get("flute_size", [0.42, 0.048, 0.88])))
    prog["uFluteYaw"].value = math.radians(float(g.get("flute_yaw_deg", 18.0)))
    prog["uIor"].value = float(g.get("ior", 1.46))
    prog["uDisp"].value = float(g.get("dispersion", 0.018))
    prog["uTint"].value = tuple(map(float, g.get("tint", [0.075, 0.025, 0.018])))
    prog["uCausticGain"].value = float(g.get("caustic_gain", 1.05))
    prog["uShadowGain"].value = float(g.get("shadow_gain", 0.34))
    prog["uFluteFreq"].value = float(g.get("flute_freq", 56.0))
    prog["uFluteAmp"].value = float(g.get("flute_amp", 0.19))
    prog["uMetaK"].value = float(g.get("metaball_k", 0.32))
    prog["uProxyDist"].value = float(g.get("proxy_dist", 7.0))

    def read_bgr(tex):
        arr = np.frombuffer(tex.read(), "f4").reshape(H, W, 4)[::-1]
        return (np.clip(arr[..., :3], 0, 1) * 255 + 0.5).astype(np.uint8)[:, :, ::-1]

    def render_one(i, read_all=False):
        pl = frame(c, i)
        plate_t.write(pl[:, :, ::-1].tobytes())
        plate_t.use(0)
        prog["uPlate"].value = 0
        prog["uRo"].value = tuple(map(float, S["cam"][i]))
        prog["uCam"].value = tuple(np.asarray(S["rot"][i], np.float32).T.flatten())
        prog["iTime"].value = i / fps
        fbo.use()
        va.render(moderngl.TRIANGLES)
        ctx.finish()
        comp = read_bgr(t_comp)
        if not read_all:
            return pl, comp, None, None, None
        return pl, comp, read_bgr(t_norm), read_bgr(t_refr), read_bgr(t_caus)

    if stills:
        rows = []
        for i in stills:
            pl, comp, nm, rf, cs = render_one(i, read_all=True)
            t = lambda im, s: label(cv2.resize(im, (480, 270)), s)
            rows.append(np.hstack([t(pl, f"plate {i}"), t(nm, "3D SDF normal"), t(rf, "optical warp grid"), t(comp, "liquid glass v2")]))
        out_p = work(c, "still_v2_sheet.jpg")
        cv2.imwrite(str(out_p), np.vstack(rows), [cv2.IMWRITE_JPEG_QUALITY, 92])
        print("stills ->", out_p)
        return

    comp_dir = work(c, "comp")
    shutil.rmtree(comp_dir, ignore_errors=True)
    comp_dir.mkdir(parents=True)
    if save_passes:
        for k in ("normal", "refract", "caustic"):
            d = work(c, "pass", k)
            shutil.rmtree(d, ignore_errors=True)
            d.mkdir(parents=True)
    ts = time.time()
    for i in range(N):
        _, comp, nm, rf, cs = render_one(i, read_all=save_passes)
        cv2.imwrite(str(comp_dir / f"{i:04d}.png"), comp, [cv2.IMWRITE_PNG_COMPRESSION, 1])
        if save_passes:
            cv2.imwrite(str(work(c, "pass", "normal", f"{i:04d}.jpg")), nm, [cv2.IMWRITE_JPEG_QUALITY, 93])
            cv2.imwrite(str(work(c, "pass", "refract", f"{i:04d}.jpg")), rf, [cv2.IMWRITE_JPEG_QUALITY, 93])
            cv2.imwrite(str(work(c, "pass", "caustic", f"{i:04d}.jpg")), cs, [cv2.IMWRITE_JPEG_QUALITY, 93])
        if i % 45 == 0 or i == N - 1:
            print(f"frame {i}/{N} ({time.time() - ts:.1f}s)", flush=True)
    encode(c, out_name=out_dir, suffix="_v2")


if __name__ == "__main__":
    main()
