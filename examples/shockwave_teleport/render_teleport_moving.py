"""Moving-Camera 3D Perspective Sonic-Prism Teleport (COLMAP 3D + Homography AI Clean Plate + space_distortion Prism).

    python render_teleport_moving.py [piece_dir] --still 28,38,52,85,100,115   quick look -> work/still_moving_sheet.jpg
    python render_teleport_moving.py [piece_dir] --out out                     full render -> out/Moving_Cam_Sonic_Prism_Teleport_*.mp4

True Moving-Camera 3D Teleportation (Pexels 32109503, 177.4px camera sweep):
- AI-generated spotless clean plate (dancer + cast floor shadow removed) tracked across all 163 moving frames via
  per-frame static-background SIFT+RANSAC homographies (0.50px mean error) + generous shadow-zone wipe -> ZERO ghost!
- 2 true 3D world-space teleports projected every frame through COLMAP (C[i], R[i], f) with real 3D perspective scaling:
  1) CENTER -> VANISH (f=34) -> RIGHT-DEEP (+1.18m right, +0.42m deeper, ~0.90x scale, f=48)
  2) RIGHT-DEEP -> VANISH (f=96) -> LEFT-FOREGROUND (-1.08m left, -0.22m closer, ~1.06x scale, f=110)
- Exact space_distortion optical air physics locked in 3D world space:
  15-capsule body-conforming 3D acoustic sound waves (3 echoes + pre-snap implosion) on floor+facade 3D proxy,
  10-wavelength CIE 1931 Cauchy spectral rainbow prism dispersion (Abbe=1.9), per-wavelength rainbow wave-crest glints,
  vanished hot-air shimmer vortex, live limb air trails, impact camera micro-shake, matched plate sensor grain.
- ZERO liquid water droplets, ZERO glass shards, ZERO text/HUD overlays.
"""
import json, math, shutil, subprocess, sys, time
from pathlib import Path
import numpy as np, cv2, moderngl

here = Path(__file__).resolve().parent
root = here.parents[1]
sys.path.insert(0, str(here.parent / "shader_live" / "pipeline"))
from slv import load, work, frame, scene  # noqa: E402


SEGS = [
    (11, 13), (13, 15), (15, 19), (12, 14), (14, 16), (16, 20),
    (23, 25), (25, 27), (24, 26), (26, 28),
    (11, 12), (23, 24), (11, 23), (12, 24), (0, 0),
]


def find_ffmpeg():
    tl = root / "tools.local.json"
    if tl.exists():
        d = json.loads(tl.read_text(encoding="utf-8"))
        p = d.get("ffmpeg") or (d.get("tools") or {}).get("ffmpeg")
        if p and Path(p).exists():
            return str(p)
    return shutil.which("ffmpeg") or "ffmpeg"


def spectral(abbe=1.9, ns=10):
    lam = 400.0 + 300.0 * (np.arange(ns) + 0.5) / ns
    g = lambda x, m, s1, s2: np.exp(-0.5 * ((x - m) / np.where(x < m, s1, s2)) ** 2)
    X = 1.056 * g(lam, 599.8, 37.9, 31.0) + 0.362 * g(lam, 442.0, 16.0, 26.7) - 0.065 * g(lam, 501.1, 20.4, 26.2)
    Y = 0.821 * g(lam, 568.8, 46.9, 40.5) + 0.286 * g(lam, 530.9, 16.3, 31.1)
    Z = 1.217 * g(lam, 437.0, 11.8, 36.0) + 0.681 * g(lam, 459.0, 26.0, 13.8)
    M = np.array([[3.2406, -1.5372, -0.4986], [-0.9689, 1.8758, 0.0415], [0.0557, -0.2040, 1.0570]])
    rgb = np.clip(np.stack([X, Y, Z], 1) @ M.T, 0, None)
    rgb /= rgb.sum(0, keepdims=True)
    B = (0.5 / abbe) / (1.0 / 0.4861 ** 2 - 1.0 / 0.6563 ** 2)
    A = 1.5 - B / 0.5876 ** 2
    k = (A + B / (lam / 1000.0) ** 2 - 1.0) / 0.5
    return k.astype(np.float32), rgb.astype(np.float32)


FULL_VS = "#version 330\nin vec2 p; void main(){ gl_Position = vec4(p, 0.0, 1.0); }"

NOISE_GLSL = """
float hash21(vec2 p){
  vec3 q = fract(vec3(p.xyx) * 0.1031);
  q += dot(q, q.yzx + 33.33);
  return fract((q.x + q.y) * q.z);
}
float vnoise(vec3 p){
  vec3 i = floor(p), f = fract(p); f = f * f * (3.0 - 2.0 * f);
  float a = mix(hash21(i.xy + i.z * 17.0), hash21(i.xy + vec2(1,0) + i.z * 17.0), f.x);
  float b = mix(hash21(i.xy + vec2(0,1) + i.z * 17.0), hash21(i.xy + vec2(1,1) + i.z * 17.0), f.x);
  float c = mix(hash21(i.xy + (i.z + 1.0) * 17.0), hash21(i.xy + vec2(1,0) + (i.z + 1.0) * 17.0), f.x);
  float d = mix(hash21(i.xy + vec2(0,1) + (i.z + 1.0) * 17.0), hash21(i.xy + vec2(1,1) + (i.z + 1.0) * 17.0), f.x);
  return mix(mix(a, b, f.y), mix(c, d, f.y), f.z);
}
float fbm(vec3 p){
  float a = 0.5, s = 0.0;
  for (int i = 0; i < 4; i++){ s += a * vnoise(p); p = p * 2.03 + vec3(17.1, 5.3, 9.7); a *= 0.5; }
  return s;
}
"""

# Pass 0: Moving-Camera Clean-Plate + 3D Perspective Teleported Dancer Compositor
WORLD_FS = """#version 330
uniform sampler2D uLive;       /* live moving frame (top-down) */
uniform sampler2D uClean;      /* homography-warped moving clean plate for THIS frame (top-down) */
uniform sampler2D uMask;       /* live MediaPipe person matte (top-down) */
uniform sampler2D uRemoveZone; /* generous blurred body+shadow removal matte for THIS frame (top-down) */
uniform vec2 uRes;
uniform float uTime, uVisOrig, uVisDest;
uniform vec2 uFootOrig;        /* live foot anchor px */
uniform vec2 uFootDest;        /* COLMAP-projected 3D teleported foot anchor px */
uniform float uScaleDest;      /* exact 3D perspective scale = depth_orig / depth_dest */

layout(location=0) out vec4 oWorld;  /* RGB = scene (sRGB), A = active person matte */
""" + NOISE_GLSL + """
void main(){
  vec2 px = vec2(gl_FragCoord.x, uRes.y - gl_FragCoord.y);
  vec2 uv = clamp(px / uRes, vec2(0.001), vec2(0.999));
  vec3 cleanCol = texture(uClean, uv).rgb;
  vec3 liveCol  = texture(uLive, uv).rgb;
  float mOrig   = texture(uMask, uv).r;
  float remZ    = texture(uRemoveZone, uv).r;

  float nTear = fbm(vec3(uv * vec2(22.0, 16.0), uTime * 3.8));
  /* Remove BOTH the dancer silhouette AND his entire sunlit floor shadow whenever he is not at Orig! */
  float wipe = clamp(max(smoothstep(0.01, 0.25, mOrig), remZ) * (1.0 - smoothstep(0.15, 0.94, uVisOrig)), 0.0, 1.0);
  vec3 col = mix(liveCol, cleanCol, wipe);
  float pMask = 0.0;

  /* Realistic sunlit stone-plaza cast shadow under teleported dancer (matches Pexels 32109503 right-stretching sun!) */
  if (uVisDest > 0.01){
    float sc = max(uScaleDest, 0.4);
    /* Crisp dark foot contact shadow */
    vec2 d0 = (px - uFootDest) / (vec2(82.0, 16.0) * sc);
    /* Long afternoon sunlit shadow stretching right-and-slightly-back matching sun=[-0.77, -0.28, 0.57] */
    vec2 d1 = (px - (uFootDest + vec2(135.0, -8.0) * sc)) / (vec2(175.0, 26.0) * sc);
    float floorMask = smoothstep(uFootDest.y - 65.0 * sc, uFootDest.y - 10.0 * sc, px.y);
    float sh = max(exp(-dot(d0, d0) * 1.45) * 0.50, exp(-dot(d1, d1) * 1.25) * 0.30) * uVisDest * floorMask;
    col *= (1.0 - sh);
  }

  /* 1. Original position dancer visibility */
  if (uVisOrig > 0.005 && mOrig > 0.01){
    float tear = smoothstep(1.0 - uVisOrig - 0.16, 1.0 - uVisOrig + 0.12, nTear * 0.60 + 0.40 * mOrig);
    float aO = mOrig * mix(tear, 1.0, smoothstep(0.86, 1.0, uVisOrig)) * smoothstep(0.0, 0.14, uVisOrig);
    col = mix(col, liveCol, aO);
    pMask = max(pMask, aO);
  }

  /* 2. True 3D-Perspective Teleported Dancer (COLMAP projected foot anchor + exact depth perspective scale!) */
  if (uVisDest > 0.005){
    vec2 pxSrc = uFootOrig + (px - uFootDest) / max(uScaleDest, 0.35);
    vec2 uvSrc = pxSrc / uRes;
    if (uvSrc.x > 0.008 && uvSrc.x < 0.992 && uvSrc.y > 0.008 && uvSrc.y < 0.992){
      float mD = texture(uMask, uvSrc).r;
      if (mD > 0.01){
        vec3 personD = texture(uLive, uvSrc).rgb;
        float nTearD = fbm(vec3(uvSrc * vec2(22.0, 16.0), -uTime * 3.8));
        float tearD = smoothstep(1.0 - uVisDest - 0.16, 1.0 - uVisDest + 0.12, nTearD * 0.60 + 0.40 * mD);
        float aD = smoothstep(0.06, 0.72, mD) * mix(tearD, 1.0, smoothstep(0.86, 1.0, uVisDest)) * smoothstep(0.0, 0.14, uVisDest);
        col = mix(col, personD, aD);
        pMask = max(pMask, aD);
      }
    }
  }
  oWorld = vec4(clamp(col, 0.0, 1.0), pMask);
}
"""

TRAIL_VS = """#version 330
uniform vec3 uRo; uniform mat3 uCam; uniform float uF; uniform vec2 uRes;
in vec3 pos; in vec3 prv; in vec3 attr;
out vec2 vMv; out vec3 vA;
vec2 proj(vec3 P){ vec3 q = transpose(uCam) * (P - uRo); return uF * q.xy / max(q.z, 0.2) + 0.5 * uRes; }
void main(){
  vec2 p = proj(pos); vMv = p - proj(prv); vA = attr;
  gl_Position = vec4(p.x / uRes.x * 2.0 - 1.0, 1.0 - p.y / uRes.y * 2.0, 0.0, 1.0);
}
"""
TRAIL_FS = """#version 330
in vec2 vMv; in vec3 vA; out vec4 o;
void main(){
  float age = vA.x, u = vA.y;
  float sp = length(vMv);
  float w = vA.z * smoothstep(2.0, 16.0, sp) * smoothstep(0.0, 0.14, age) * pow(1.0 - age, 1.6) * smoothstep(0.0, 0.25, u) * smoothstep(1.0, 0.78, u);
  vec2 dir = vMv / max(sp, 1e-3) * min(sp, 60.0) / 60.0;
  o = vec4(dir * w, w, w * w);
}
"""
BLUR_FS = """#version 330
uniform sampler2D uSrc; uniform vec2 uDir; out vec4 o;
void main(){
  vec2 r = vec2(textureSize(uSrc, 0)); vec2 uv = gl_FragCoord.xy / r;
  vec4 s = vec4(0.0); float tw = 0.0;
  for (int k = -18; k <= 18; k++){
    float g = exp(-float(k * k) / 140.0);
    s += g * texture(uSrc, uv + uDir * float(k) / r);
    tw += g;
  }
  o = s / tw;
}
"""

# Pass 1B: World-Locked 3D Body-Conforming Sonic Waves on moving camera (exact space_distortion RING_FS!)
RING_FS = """#version 330
uniform vec2 uRes, uScale;
uniform float iTime, uF, uZref, uWallD;
uniform vec3 uRo, uWallN;
uniform mat3 uCam;
uniform int uNR;
uniform vec4 uRing[8];
uniform vec4 uRingP[8];
uniform vec4 uRingMode[8];
uniform vec4 uLobe[16];
uniform vec4 uSegA[120];
uniform vec4 uSegB[120];
out vec4 o;
""" + NOISE_GLSL + """
vec3 proxy(vec2 p){
  vec3 d = uCam * vec3((p - 0.5 * uRes) / uF, 1.0);
  float tf = d.z < -1e-5 ? -uRo.z / d.z : 1e4;
  float dn = dot(d, uWallN);
  float tw = dn > 1e-5 ? (uWallD - dot(uRo, uWallN)) / dn : 1e4;
  float mn = min(tf, tw);
  float t = mn - 0.35 * log(exp(-(tf - mn) / 0.35) + exp(-(tw - mn) / 0.35));
  return uRo + max(t, 0.0) * d;
}
float capsule(vec3 p, vec3 a, vec3 b){
  vec3 pa = p - a, ba = b - a;
  float h = clamp(dot(pa, ba) / max(dot(ba, ba), 1e-8), 0.0, 1.0);
  return length(pa - ba * h);
}
float front(int k, vec3 X, out float wn){
  float dd[15]; float dm = 1e4;
  for (int s = 0; s < 15; s++){
    vec4 A = uSegA[k * 15 + s];
    dd[s] = capsule(X, A.xyz, uSegB[k * 15 + s].xyz) + A.w;
    dm = min(dm, dd[s]);
  }
  float sum = 0.0, ws = 0.0;
  for (int s = 0; s < 15; s++){
    float e = exp(-(dd[s] - dm) / 0.14);
    sum += e; ws += e * uSegB[k * 15 + s].w;
  }
  wn = ws / max(sum, 1e-6);
  float d = dm - 0.14 * log(max(sum, 1e-6));
  vec3 u = normalize(X - uRing[k].xyz + vec3(1e-6));
  float nz = fbm(u * 1.85 + vec3(uRingP[k].z, 0.0, iTime * 0.45)) - 0.47;
  return d * (1.0 + uRingP[k].w * nz * 2.0);
}
void main(){
  vec2 p = gl_FragCoord.xy * uScale;
  vec3 s = vec3(0.0);
  float crest = 0.0;
  if (uNR > 0){
    vec3 X0 = proxy(p), Xx = proxy(p + vec2(3.0, 0.0)), Xy = proxy(p + vec2(0.0, 3.0));
    float zc = (transpose(uCam) * (X0 - uRo)).z;
    float A = clamp(uZref / max(zc, 0.05), 0.45, 3.0);
    for (int k = 0; k < 8; k++){
      if (k >= uNR) break;
      float wn, w1, w2;
      float d0 = front(k, X0, wn), dx = front(k, Xx, w1), dy = front(k, Xy, w2);
      vec2 gd = vec2(dx - d0, dy - d0);
      vec2 dir = gd / max(length(gd), 1e-9);
      float x = (d0 - uRing[k].w) / max(uRingP[k].x, 0.04);
      float env = exp(-0.42 * x * x);
      if (env < 1e-3) continue;
      vec3 u = normalize(X0 - uRing[k].xyz + vec3(1e-6));
      vec4 L1 = uLobe[2 * k], L2 = uLobe[2 * k + 1];
      float lobe = min(0.28 + exp(L1.w * (dot(u, L1.xyz) - 1.0)) + exp(L2.w * (dot(u, L2.xyz) - 1.0)), 1.35);
      float amp = uRingP[k].y * A * lobe * mix(0.55, 1.0, wn);
      float dirSign = uRingMode[k].x;
      float freq = uRingMode[k].y;
      float osc = sin(freq * x) * dirSign;
      s.xy += dir * amp * osc * env;
      if (uRingMode[k].z > 0.01 && d0 < uRing[k].w + 0.55){
        float pull = exp(-(d0 * d0) / max(uRing[k].w * uRing[k].w * 0.7, 0.06)) * uRingMode[k].z;
        vec2 tang = vec2(-dir.y, dir.x);
        s.xy += (-dir * 1.15 + tang * 0.42) * amp * 0.65 * pull;
        s.z += amp * pull / 16.0;
      }
      s.z += amp * env / 16.0;
      crest += abs(cos(freq * x)) * env * amp / 20.0;
    }
  }
  o = vec4(s, crest);
}
"""

# Pass 2: 10-Wavelength CIE 1931 Spectral Rainbow Prism + Tilted-Air Rainbow Glint + Moving Hot-Air Ghost!
COMP_FS = """#version 330
uniform sampler2D uWorld;
uniform sampler2D uField;
uniform sampler2D uRingT;
uniform sampler2D uDist;
uniform sampler2D uGhostMask;  /* already warped to current moving camera frame! */
uniform vec2 uRes;
uniform float iTime, uNear;
uniform float uGhostAlpha, uGhostAge;
uniform vec4 uPulse[6];
uniform vec2 uPulseW[6];
uniform int uNPulse;
uniform int uNS;
uniform float uSpecK[12];
uniform vec3 uSpecW[12];
uniform float uGlint, uTilt, uRefl, uSpec, uLift;
uniform vec2 uShakePx;
uniform float uBlackFloor, uGrainAmt, uFrame;

out vec4 oCol;
""" + NOISE_GLSL + """
vec3 toLin(vec3 c){ return pow(max(c, 0.0), vec3(2.2)); }
vec4 sampleWorldPx(vec2 px){
  vec2 uv = clamp(px, vec2(0.5), uRes - 0.5) / uRes;
  return texture(uWorld, vec2(uv.x, 1.0 - uv.y));
}
vec3 plateLin(vec2 px){ return toLin(sampleWorldPx(px).rgb); }
vec4 fieldPx(vec2 px){
  vec2 uv = clamp(px / uRes, vec2(0.001), vec2(0.999));
  return texture(uField, vec2(uv.x, 1.0 - uv.y));
}
vec4 ringPx(vec2 px){
  vec2 uv = clamp(px / uRes, vec2(0.001), vec2(0.999));
  return texture(uRingT, vec2(uv.x, 1.0 - uv.y));
}
float sdist(vec2 px){
  vec2 uv = clamp(px / uRes, vec2(0.001), vec2(0.999));
  return texture(uDist, uv).r;
}
float pulseField(vec2 p){
  float d = sdist(p);
  float s = 0.0;
  for (int k = 0; k < 6; k++){
    if (k >= uNPulse) break;
    vec4 Q = uPulse[k];
    float x = (d - Q.z) / max(uPulseW[k].x, 4.0);
    vec2 r = p - Q.xy;
    float reach = max(uPulseW[k].y, 60.0);
    s += exp(-x * x) * exp(-dot(r, r) / (2.0 * reach * reach)) * Q.w;
  }
  return s * smoothstep(0.0, 4.0, d);
}

void main(){
  vec2 uvScr = gl_FragCoord.xy / uRes;
  vec2 uvSh = clamp((uvScr - 0.5) / 1.008 + 0.5 + uShakePx / uRes, vec2(0.001), vec2(0.999));
  vec2 px = vec2(uvSh.x * uRes.x, (1.0 - uvSh.y) * uRes.y);

  vec4 w0 = sampleWorldPx(px);
  float activeMask = w0.a;
  float keep = 1.0 - 0.68 * smoothstep(0.35, 0.85, activeMask);

  float e = 3.0;
  vec2 gd = vec2(sdist(px + vec2(e, 0)) - sdist(px - vec2(e, 0)),
                 sdist(px + vec2(0, e)) - sdist(px - vec2(0, e)));
  vec2 nout = gd / max(length(gd), 1e-4);
  float pw = pulseField(px);
  float nr = exp(-pow(sdist(px) / max(uNear, 20.0), 2.0));
  vec4 F = fieldPx(px);
  float wTrail = F.z * nr + pw;
  float covTrail = (1.0 - exp(-2.8 * wTrail)) * keep;
  vec2 mv = (F.xy * nr + nout * pw) / max(wTrail, 1e-3);

  float aL = fieldPx(px - vec2(e,0)).z * exp(-pow(sdist(px - vec2(e,0))/max(uNear,20.0),2.0)) + pulseField(px - vec2(e,0));
  float aR = fieldPx(px + vec2(e,0)).z * exp(-pow(sdist(px + vec2(e,0))/max(uNear,20.0),2.0)) + pulseField(px + vec2(e,0));
  float aU = fieldPx(px - vec2(0,e)).z * exp(-pow(sdist(px - vec2(0,e))/max(uNear,20.0),2.0)) + pulseField(px - vec2(0,e));
  float aD = fieldPx(px + vec2(0,e)).z * exp(-pow(sdist(px + vec2(0,e))/max(uNear,20.0),2.0)) + pulseField(px + vec2(0,e));
  vec2 g = vec2(aR - aL, aD - aU) / (2.0 * e);

  vec2 ghostWarp = vec2(0.0);
  float ghostCov = 0.0;
  if (uGhostAlpha > 0.005){
    vec2 uvG = px / uRes;
    vec2 drift = vec2((fbm(vec3(uvG * 16.0, uGhostAge * 3.5)) - 0.5) * 0.020, -0.014 * uGhostAge) * (0.35 + uGhostAge);
    uvG += drift;
    if (uvG.x > 0.01 && uvG.x < 0.99 && uvG.y > 0.01 && uvG.y < 0.99){
      float gm = texture(uGhostMask, uvG).r;
      if (gm > 0.015){
        vec2 eg = vec2(4.0 / uRes.x, 4.0 / uRes.y);
        float gx = texture(uGhostMask, uvG + vec2(eg.x, 0)).r - texture(uGhostMask, uvG - vec2(eg.x, 0)).r;
        float gy = texture(uGhostMask, uvG + vec2(0, eg.y)).r - texture(uGhostMask, uvG - vec2(0, eg.y)).r;
        vec3 nq = vec3(px * 0.018, iTime * 4.5);
        vec2 shimmer = vec2(fbm(nq) - 0.5, fbm(nq + vec3(7.3, -4.1, 2.9)) - 0.5);
        ghostWarp = (vec2(gx, gy) * 36.0 + shimmer * 40.0) * gm * uGhostAlpha;
        ghostCov = gm * uGhostAlpha;
      }
    }
  }

  vec4 rg = ringPx(px);
  vec2 rw = rg.xy * keep;
  float rcov = (1.0 - exp(-rg.z)) * keep;
  float crest = rg.w * keep;

  vec3 qh = vec3(px * 0.013 - mv * iTime * 0.7, iTime * 2.2);
  vec2 hz = vec2(fbm(qh), fbm(qh + vec3(5.2, 1.3, 2.7))) - 0.5;
  vec2 warp = -mv * 16.0 * covTrail
            - g * 3400.0 * covTrail
            + hz * 14.0 * max(covTrail, rcov * 0.55)
            + ghostWarp;

  vec2 totalWarp = warp + rw;
  float cov = clamp(max(max(covTrail, rcov), ghostCov * 0.85), 0.0, 1.0);

  vec3 col;
  if (length(totalWarp) > 0.25 || cov > 0.01){
    vec3 acc = vec3(0.0);
    float tw = 0.0;
    for (int k = 0; k < 7; k++){
      float t = float(k) / 6.0;
      float wt = 1.0 - 0.55 * t;
      vec2 o = warp * (0.30 + 0.70 * t) + rw * (0.55 + 0.45 * t);
      vec3 s = vec3(0.0);
      for (int j = 0; j < 12; j++){
        if (j >= uNS) break;
        s += uSpecW[j] * plateLin(px + o * uSpecK[j]);
      }
      acc += wt * s;
      tw += wt;
    }
    vec3 bent = acc / max(tw, 1e-4);

    vec2 airGrad = -g * 52.0 * covTrail - (rw + ghostWarp * 0.65) * uTilt;
    vec3 n = normalize(vec3(airGrad, 1.0));
    float fres = pow(1.0 - clamp(n.z, 0.0, 1.0), 0.85);
    /* Afternoon sun from left-front matching Pexels 32109503 */
    vec3 Ld = normalize(vec3(-0.68, -0.52, 0.52));
    vec3 Hh = normalize(Ld + vec3(0.0, 0.0, 1.0));
    float spec = pow(max(dot(n, Hh), 0.0), 16.0) * smoothstep(0.02, 0.22, fres) * cov;

    vec2 rp = px + n.xy * 65.0;
    vec3 refl = (plateLin(rp) + plateLin(rp + vec2(8.0, 5.0)) + plateLin(rp - vec2(7.0, 8.0))) / 3.0;

    col = bent * (1.0 + uLift * cov);
    col += (refl * 1.15 + 0.008) * clamp(fres * uRefl, 0.0, 0.36) * 0.45 * cov;
    col += vec3(1.0, 0.97, 0.90) * uSpec * spec;

    /* Per-wavelength spectral rainbow prism glints catching sunlight on sonic wave crests! */
    vec3 glint = vec3(0.0);
    for (int j = 0; j < 12; j++){
      if (j >= uNS) break;
      vec3 nj = normalize(vec3((n.xy / max(n.z, 0.15)) * uSpecK[j], 1.0));
      float gSharp = pow(max(dot(nj, Hh), 0.0), 190.0) * 1.45;
      float gBand  = pow(max(dot(nj, normalize(vec3(0.42, -0.45, 0.78))), 0.0), 64.0) * 0.60;
      glint += uSpecW[j] * (gSharp + gBand);
    }
    float glintMask = clamp(fres * 1.50 + crest * 0.62 + ghostCov * 0.40, 0.0, 1.5) * smoothstep(0.03, 0.22, cov);
    col += glint * uGlint * glintMask;

    col = col / (1.0 + max(col - 0.92, vec3(0.0)) * 0.35);
    col = pow(clamp(col, 0.0, 1.0), vec3(1.0 / 2.2));
  } else {
    col = w0.rgb;
  }

  float cgMix = clamp(cov * 1.1, 0.0, 1.0);
  if (cgMix > 0.01){
    vec3 graded = max(col, vec3(uBlackFloor));
    vec2 p1 = gl_FragCoord.xy + vec2(uFrame * 17.3, uFrame * 31.7);
    vec2 p2 = floor(gl_FragCoord.xy * 0.5) + vec2(uFrame * 11.1, uFrame * 23.9);
    float gr = (hash21(p1) + hash21(p1 + 7.13) - 1.0) * 0.62 + (hash21(p2) + hash21(p2 + 5.71) - 1.0) * 0.38;
    float lum = dot(graded, vec3(0.299, 0.587, 0.114));
    graded += vec3(gr) * (uGrainAmt * (1.12 - 0.60 * smoothstep(0.55, 0.98, lum)));
    col = mix(col, graded, cgMix);
  }
  oCol = vec4(clamp(col, 0.0, 1.0), 1.0);
}
"""


def label(im, s):
    im = im.copy()
    cv2.putText(im, s, (12, im.shape[0] - 14), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 4, cv2.LINE_AA)
    cv2.putText(im, s, (12, im.shape[0] - 14), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 1, cv2.LINE_AA)
    return im


def encode(c, out_dir="out"):
    out = c["dir"] / out_dir
    out.mkdir(parents=True, exist_ok=True)
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


def build_moving_sound_waves(P_limbs, events_spec, fw0, seed=29):
    rng = np.random.default_rng(seed)
    waves = []
    for ev_f, delta_3d, power in events_spec:
        Q = P_limbs[ev_f].copy() + np.asarray(delta_3d, dtype=np.float64)[None, :]
        C = 0.25 * (Q[11] + Q[12] + Q[23] + Q[24])
        A = np.array([Q[s[0]] for s in SEGS], dtype=np.float64)
        B = np.array([Q[s[1]] for s in SEGS], dtype=np.float64)
        wgt = np.exp(-np.sum((0.5 * (A + B) - C) ** 2, 1) / (2.0 * 0.55 ** 2))
        lag = (1.0 - wgt) * 0.22
        lag[-1] -= 0.08

        lobes_imp = []
        for _ in range(2):
            az = float(rng.uniform(0, 2 * math.pi)); el = float(rng.uniform(-0.25, 0.35))
            lobes_imp.append((math.cos(el)*math.cos(az), math.cos(el)*math.sin(az), math.sin(el), 1.4))
        waves.append({
            "kind": "implode", "t0": ev_f - 11, "t_snap": ev_f,
            "C": C, "A": A, "B": B, "lag": lag, "wgt": wgt,
            "amp": 24.0 * power, "lobes": lobes_imp, "seed": float(rng.uniform(0, 50)),
        })

        for e_idx, (df, sc_a, spd, freq) in enumerate([(0, 1.00, 5.2, 2.75), (3, 0.72, 4.3, 3.05), (7, 0.48, 3.6, 3.25)]):
            lobes = []
            for l in range(2):
                if e_idx == 0:
                    base = math.atan2(-fw0[1], -fw0[0]) if l == 0 else math.atan2(fw0[1], fw0[0])
                    az = base + float(rng.uniform(-0.55, 0.55)); el = float(rng.uniform(-0.18, 0.32))
                else:
                    az = float(rng.uniform(0, 2 * math.pi)); el = float(rng.uniform(-0.28, 0.45))
                lobes.append((math.cos(el)*math.cos(az), math.cos(el)*math.sin(az), math.sin(el), float(rng.uniform(1.6, 3.6))))
            waves.append({
                "kind": "explode", "t0": ev_f + df, "t_snap": ev_f,
                "C": C, "A": A, "B": B, "lag": lag, "wgt": wgt,
                "amp": 36.0 * power * sc_a, "speed": spd * float(rng.uniform(0.92, 1.08)),
                "freq": freq, "lobes": lobes, "seed": float(rng.uniform(0, 50)),
            })
    return waves


def project_3d(P3, ro, R, f, W, H):
    q = R.T @ (P3 - ro)
    z = max(float(q[2]), 0.15)
    px = f * q[:2] / z + np.array([0.5 * W, 0.5 * H])
    return px, z


def main():
    args = sys.argv[1:]
    stills = None
    out_dir = "out"
    piece_arg = None
    k = 0
    while k < len(args):
        if args[k] == "--still":
            stills = [int(x) for x in args[k + 1].split(",")]; k += 2
        elif args[k] == "--out":
            out_dir = args[k + 1]; k += 2
        else:
            piece_arg = args[k]; k += 1

    c = load(piece_arg or (here / "piece_moving"))
    S = scene(c)
    L = np.load(work(c, "limbs.npz"))
    P_limbs = np.asarray(L["P"], np.float64)
    W, H, N = c["W"], c["H"], c["N"]
    fps = float(c.get("fps_num", 25)) / float(c.get("fps_den", 1))
    f = float(S["f"])
    cfg = c.get("moving_teleport", {})

    cam_path = np.asarray(S["cam"], np.float64)
    cam_rot = np.asarray(S["rot"], np.float64)
    t1_van = int(cfg.get("t1_vanish", 34))
    t1_arr = int(cfg.get("t1_arrive", 48))
    t2_van = int(cfg.get("t2_vanish", 96))
    t2_arr = int(cfg.get("t2_arrive", 110))
    d0 = np.array([0.0, 0.0, 0.0], dtype=np.float64)
    d1 = np.array(cfg.get("delta1_3d", [1.18, 0.42, 0.0]), dtype=np.float64)
    d2 = np.array(cfg.get("delta2_3d", [-1.08, -0.22, 0.0]), dtype=np.float64)

    clean80 = cv2.imread(str(work(c, "clean_plate_ref80.jpg")))
    Hs = np.load(work(c, "clean_homographies.npz"))["H"]  # (N, 3, 3) mapping ref80 -> frame i

    R0 = cam_rot[0]
    fw0 = np.array([R0[0, 2], R0[1, 2]]) / max(math.hypot(R0[0, 2], R0[1, 2]), 1e-6)

    # 4 world-locked 3D teleport sonic events on moving camera!
    events_spec = [
        (t1_van, d0, 1.12),
        (t1_arr, d1, 1.08),
        (t2_van, d1, 1.12),
        (t2_arr, d2, 1.10),
    ]
    sound_waves = build_moving_sound_waves(P_limbs, events_spec, fw0, seed=29)
    print(f"Moving-Cam Sonic Prism Teleport: {N}f | {len(sound_waves)} world-locked 3D sonic waves | 10-wavelength prism")

    # Pre-load departure ghost masks (in their departure frames' image space)
    gmask_1_raw = cv2.GaussianBlur(cv2.imread(str(work(c, "mask", f"{t1_van:04d}.png")), cv2.IMREAD_GRAYSCALE), (0, 0), 1.5)
    gmask_2_raw = cv2.GaussianBlur(cv2.imread(str(work(c, "mask", f"{t2_van:04d}.png")), cv2.IMREAD_GRAYSCALE), (0, 0), 1.5)

    pl0 = frame(c, 0)
    gray0 = cv2.cvtColor(pl0, cv2.COLOR_BGR2GRAY).astype(np.float32) / 255.0
    black_floor = float(np.clip(np.percentile(gray0, 1.0) * 0.85, 0.008, 0.045))
    hp0 = gray0 - cv2.GaussianBlur(gray0, (0, 0), 1.2)
    grain_amt = float(np.clip(float(np.median(np.abs(hp0 - np.median(hp0)))) * 1.4826 * 1.10, 0.008, 0.022))

    ctx = moderngl.create_standalone_context(require=330)
    world_prog = ctx.program(vertex_shader=FULL_VS, fragment_shader=WORLD_FS)
    trail_prog = ctx.program(vertex_shader=TRAIL_VS, fragment_shader=TRAIL_FS)
    blur_prog  = ctx.program(vertex_shader=FULL_VS, fragment_shader=BLUR_FS)
    ring_prog  = ctx.program(vertex_shader=FULL_VS, fragment_shader=RING_FS)
    comp_prog  = ctx.program(vertex_shader=FULL_VS, fragment_shader=COMP_FS)

    tri = ctx.buffer(np.array([-1, -1, 3, -1, -1, 3], "f4").tobytes())
    va_world = ctx.vertex_array(world_prog, [(tri, "2f", "p")])
    va_blur  = ctx.vertex_array(blur_prog,  [(tri, "2f", "p")])
    va_ring  = ctx.vertex_array(ring_prog,  [(tri, "2f", "p")])
    va_comp  = ctx.vertex_array(comp_prog,  [(tri, "2f", "p")])

    hw, hh = W // 2, H // 2
    t_world = ctx.texture((W, H), 4, dtype="f4"); t_world.filter = (moderngl.LINEAR, moderngl.LINEAR); t_world.repeat_x = t_world.repeat_y = False
    fbo_world = ctx.framebuffer([t_world])

    fA, fB, rT = (ctx.texture((hw, hh), 4, dtype="f4") for _ in range(3))
    for t in (fA, fB, rT):
        t.filter = (moderngl.LINEAR, moderngl.LINEAR); t.repeat_x = t.repeat_y = False
    fbA, fbB, fbR = ctx.framebuffer([fA]), ctx.framebuffer([fB]), ctx.framebuffer([rT])

    t_final = ctx.texture((W, H), 4, dtype="f4")
    fbo_final = ctx.framebuffer([t_final])

    live_t  = ctx.texture((W, H), 3); live_t.filter = (moderngl.LINEAR, moderngl.LINEAR); live_t.repeat_x = live_t.repeat_y = False
    clean_t = ctx.texture((W, H), 3); clean_t.filter = (moderngl.LINEAR, moderngl.LINEAR); clean_t.repeat_x = clean_t.repeat_y = False
    mask_t  = ctx.texture((W, H), 1); mask_t.filter = (moderngl.LINEAR, moderngl.LINEAR); mask_t.repeat_x = mask_t.repeat_y = False
    rem_t   = ctx.texture((W, H), 1, dtype="f4"); rem_t.filter = (moderngl.LINEAR, moderngl.LINEAR); rem_t.repeat_x = rem_t.repeat_y = False
    gmask_t = ctx.texture((W, H), 1); gmask_t.filter = (moderngl.LINEAR, moderngl.LINEAR); gmask_t.repeat_x = gmask_t.repeat_y = False
    dist_t  = ctx.texture((W, H), 1, dtype="f4"); dist_t.filter = (moderngl.LINEAR, moderngl.LINEAR); dist_t.repeat_x = dist_t.repeat_y = False

    K_s, W_s = spectral(abbe=1.9, ns=10)
    world_prog["uRes"].value = (float(W), float(H))
    trail_prog["uRes"].value = (float(W), float(H))
    trail_prog["uF"].value = f
    ring_prog["uRes"].value = (float(W), float(H))
    ring_prog["uScale"].value = (float(W) / hw, float(H) / hh)
    ring_prog["uF"].value = f
    comp_prog["uRes"].value = (float(W), float(H))
    comp_prog["uNS"].value = len(K_s)
    comp_prog["uSpecK"].value = list(K_s) + [1.0] * (12 - len(K_s))
    comp_prog["uSpecW"].value = [tuple(map(float, w)) for w in W_s] + [(0.0, 0.0, 0.0)] * (12 - len(W_s))
    comp_prog["uGlint"].value = 1.35
    comp_prog["uTilt"].value = 0.065
    comp_prog["uRefl"].value = 0.25
    comp_prog["uSpec"].value = 0.16
    comp_prog["uLift"].value = 0.04
    comp_prog["uBlackFloor"].value = black_floor
    comp_prog["uGrainAmt"].value = grain_amt

    limb_chains = [[13, 15, 19], [14, 16, 20], [11, 13, 15], [12, 14, 16]]
    limb_gains = [1.25, 1.25, 0.75, 0.75]

    def smooth01(x):
        x = max(0.0, min(1.0, x))
        return x * x * (3.0 - 2.0 * x)

    def at_pose(t, delta3):
        t = float(np.clip(t, 0, N - 1))
        i0 = min(int(t), N - 2); u = t - i0
        return P_limbs[i0] * (1.0 - u) + P_limbs[i0 + 1] * u + delta3[None, :]

    def chain_pts(Q, ids):
        pts = [Q[j] for j in ids]
        pts.append(pts[-1] + 0.28 * (pts[-1] - pts[-2]))
        seg = np.array([np.linalg.norm(pts[k + 1] - pts[k]) for k in range(len(pts) - 1)])
        cum = np.concatenate([[0.0], np.cumsum(seg)]) / max(seg.sum(), 1e-6)
        return np.array([[np.interp(u, cum, [p[a] for p in pts]) for a in range(3)] for u in np.linspace(0, 1, 9)])

    def build_trail_mesh(i, delta3):
        ages = np.arange(9 * 2 + 1) / 2.0
        verts = []
        for li, ids in enumerate(limb_chains):
            G = np.array([chain_pts(at_pose(i - a, delta3), ids) for a in ages])
            Gp = np.array([chain_pts(at_pose(i - a - 1.0, delta3), ids) for a in ages])
            for a in range(len(ages) - 1):
                for u in range(8):
                    for aa, uu in ((a, u), (a + 1, u), (a + 1, u + 1), (a, u), (a + 1, u + 1), (a, u + 1)):
                        verts.append([*G[aa, uu], *Gp[aa, uu], ages[aa] / ages[-1], uu / 8.0, limb_gains[li]])
        return np.array(verts, dtype=np.float32)

    def warp_mask_to_dest(m_u8, foot_o, foot_d, sc):
        M = np.array([[sc, 0.0, foot_d[0] - foot_o[0] * sc],
                      [0.0, sc, foot_d[1] - foot_o[1] * sc]], dtype=np.float32)
        return cv2.warpAffine(m_u8, M, (W, H), borderValue=0)

    def render_one(i):
        pl = frame(c, i)
        m_img = cv2.GaussianBlur(cv2.imread(str(work(c, "mask", f"{i:04d}.png")), cv2.IMREAD_GRAYSCALE), (0, 0), 1.0)
        t_sec = i / fps
        ro = cam_path[i]; R = cam_rot[i]
        Rt = tuple(R.astype(np.float32).T.flatten())
        p_depth = float(L["depth"][i])

        # Warp spotless AI clean plate into THIS exact moving-camera frame via static-background homography H_i!
        clean_i = cv2.warpPerspective(clean80, Hs[i], (W, H), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)

        # Build generous body + sunlit floor-shadow removal zone for THIS frame (so 0% shadow ghost remains!)
        rem_bin = cv2.dilate((m_img > 18).astype(np.uint8), cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (95, 95)))
        ys, xs = np.where(m_img > 40)
        if len(ys) > 0:
            rem_bin[max(0, int(ys.max()) - 95):min(H, int(ys.max()) + 100),
                    max(0, int(xs.min()) - 140):min(W, int(xs.max()) + 540)] = 1
        # Match subtle frame-to-frame exposure along clean border ring right outside rem_bin!
        ring = (cv2.dilate(rem_bin, np.ones((55, 55), np.uint8)) - rem_bin) > 0
        if ring.sum() > 200:
            for ch in range(3):
                cv_m = float(clean_i[:, :, ch][ring].mean())
                lv_m = float(pl[:, :, ch][ring].mean())
                clean_i[:, :, ch] = np.clip(clean_i[:, :, ch].astype(np.float32) + (lv_m - cv_m), 0, 255).astype(np.uint8)
        rem_f = cv2.GaussianBlur(rem_bin.astype(np.float32), (0, 0), 22.0)

        # State machine across 2 moving-camera 3D perspective teleports:
        # CENTER (d0) -> VANISH (t1_van=34) -> RIGHT-DEEP (d1, t1_arr=48) -> VANISH (t2_van=96) -> LEFT-CLOSE (d2, t2_arr=110)
        if i < t1_van - 2:
            visO, visD, cur_delta = 1.0, 0.0, d0
        elif i <= t1_van + 4:
            visO = 1.0 - smooth01((i - (t1_van - 2)) / 6.0); visD = 0.0; cur_delta = d0
        elif i < t1_arr - 2:
            visO, visD, cur_delta = 0.0, 0.0, d1
        elif i <= t1_arr + 4:
            visO = 0.0; visD = smooth01((i - (t1_arr - 2)) / 6.0); cur_delta = d1
        elif i < t2_van - 2:
            visO, visD, cur_delta = 0.0, 1.0, d1
        elif i <= t2_van + 4:
            visO = 0.0; visD = 1.0 - smooth01((i - (t2_van - 2)) / 6.0); cur_delta = d1
        elif i < t2_arr - 2:
            visO, visD, cur_delta = 0.0, 0.0, d2
        elif i <= t2_arr + 4:
            visO = 0.0; visD = smooth01((i - (t2_arr - 2)) / 6.0); cur_delta = d2
        else:
            visO, visD, cur_delta = 0.0, 1.0, d2

        # Exact 3D COLMAP projection of foot anchor at original & teleported 3D world destination!
        foot3_o = 0.5 * (P_limbs[i, 29] + P_limbs[i, 30])
        foot3_o[2] = 0.02
        foot3_d = foot3_o + cur_delta
        px_fo, z_fo = project_3d(foot3_o, ro, R, f, W, H)
        px_fd, z_fd = project_3d(foot3_d, ro, R, f, W, H)
        scale_d = float(np.clip(z_fo / max(z_fd, 0.5), 0.55, 1.55))

        # Active silhouette distance field (warped cleanly to current 3D destination!)
        if visO > 0.25:
            m_act = m_img
            trail_delta = d0
        elif visD > 0.25:
            m_act = warp_mask_to_dest(m_img, px_fo, px_fd, scale_d)
            trail_delta = cur_delta
        else:
            m_act = warp_mask_to_dest(m_img, px_fo, px_fd, scale_d)
            trail_delta = cur_delta
        mb = (cv2.GaussianBlur(m_act, (0, 0), 2.0) > 120).astype(np.uint8)
        dt_field = cv2.GaussianBlur(cv2.distanceTransform(1 - mb, cv2.DIST_L2, 5), (0, 0), 2.5).astype(np.float32)

        # World-locked vanished hot-air silhouette (re-projected via current moving camera homography!)
        g_alpha, g_age = 0.0, 0.0
        gmask_now = np.zeros((H, W), dtype=np.uint8)
        if t1_van <= i <= t1_van + 26:
            g_age = (i - t1_van) / fps
            g_alpha = max(0.0, 1.0 - (g_age / 1.0) ** 0.85) * 0.95
            # Warp departure mask from frame t1_van -> frame i using H_i @ inv(H_t1_van)!
            H_rel = Hs[i] @ np.linalg.inv(Hs[t1_van])
            gmask_now = cv2.warpPerspective(gmask_1_raw, H_rel, (W, H), borderValue=0)
        elif t2_van <= i <= t2_van + 26:
            g_age = (i - t2_van) / fps
            g_alpha = max(0.0, 1.0 - (g_age / 1.0) ** 0.85) * 0.95
            f_o2 = 0.5 * (P_limbs[t2_van, 29] + P_limbs[t2_van, 30]); f_o2[2] = 0.02
            p_o2, z_o2 = project_3d(f_o2, cam_path[t2_van], cam_rot[t2_van], f, W, H)
            p_d2, z_d2 = project_3d(f_o2 + d1, cam_path[t2_van], cam_rot[t2_van], f, W, H)
            g2_dest = warp_mask_to_dest(gmask_2_raw, p_o2, p_d2, z_o2 / max(z_d2, 0.5))
            H_rel = Hs[i] @ np.linalg.inv(Hs[t2_van])
            gmask_now = cv2.warpPerspective(g2_dest, H_rel, (W, H), borderValue=0)

        # Pass 0: Moving-camera spotless clean plate + 3D perspective teleported dancer!
        live_t.write(pl[:, :, ::-1].tobytes())
        clean_t.write(clean_i[:, :, ::-1].tobytes())
        mask_t.write(m_img.tobytes())
        rem_t.write(rem_f.astype(np.float32).tobytes())
        fbo_world.use()
        ctx.viewport = (0, 0, W, H)
        ctx.disable(moderngl.BLEND)
        live_t.use(0); clean_t.use(1); mask_t.use(2); rem_t.use(3)
        world_prog["uLive"].value = 0; world_prog["uClean"].value = 1
        world_prog["uMask"].value = 2; world_prog["uRemoveZone"].value = 3
        world_prog["uTime"].value = float(t_sec)
        world_prog["uVisOrig"].value = float(visO)
        world_prog["uVisDest"].value = float(visD)
        world_prog["uFootOrig"].value = (float(px_fo[0]), float(px_fo[1]))
        world_prog["uFootDest"].value = (float(px_fd[0]), float(px_fd[1]))
        world_prog["uScaleDest"].value = float(scale_d)
        va_world.render(moderngl.TRIANGLES)

        # Pass 1A: Live limb air trails
        fbA.use(); ctx.viewport = (0, 0, hw, hh); fbA.clear(0.0, 0.0, 0.0, 0.0)
        if max(visO, visD) > 0.15:
            trail_prog["uRo"].value = tuple(map(float, ro)); trail_prog["uCam"].value = Rt
            vbo = ctx.buffer(build_trail_mesh(i, trail_delta).tobytes())
            va_t = ctx.vertex_array(trail_prog, [(vbo, "3f 3f 3f", "pos", "prv", "attr")])
            ctx.enable(moderngl.BLEND); ctx.blend_func = (moderngl.ONE, moderngl.ONE)
            va_t.render(moderngl.TRIANGLES)
            ctx.disable(moderngl.BLEND)
            va_t.release(); vbo.release()
        fbB.use(); fA.use(0); blur_prog["uSrc"].value = 0; blur_prog["uDir"].value = (1.0, 0.0); va_blur.render(moderngl.TRIANGLES)
        fbA.use(); fB.use(0); blur_prog["uDir"].value = (0.0, 1.0); va_blur.render(moderngl.TRIANGLES)

        # Pass 1B: World-locked 3D body-conforming acoustic sound waves!
        act = []
        shake = np.zeros(2, dtype=np.float64)
        for sw in sound_waves:
            if sw["kind"] == "implode":
                if sw["t0"] <= i <= sw["t_snap"] + 1:
                    u = (i - sw["t0"]) / max(float(sw["t_snap"] - sw["t0"]), 1.0)
                    r_now = max(0.06, 1.60 * (1.0 - smooth01(u) * 0.94))
                    amp = sw["amp"] * math.sin(min(u, 1.0) * math.pi * 0.88)
                    act.append((amp, r_now, 0.24, -1.0, 2.5, 0.85, sw))
            else:
                age = (i - sw["t0"]) / fps
                life = 1.35
                if 0.0 <= age <= life:
                    r_now = 0.04 + sw["speed"] * (age ** 0.86)
                    amp = sw["amp"] * ((1.0 - age / life) ** 1.15) * min(1.0, age / 0.045)
                    width = 0.16 + 0.065 * r_now
                    act.append((amp, r_now, width, +1.0, sw["freq"], 0.0, sw))
                    if sw["t0"] == sw["t_snap"] and age < 0.26:
                        ka = 5.4 * math.exp(-age * 15.0)
                        shake[0] += ka * math.sin(age * 98.0 + sw["t_snap"])
                        shake[1] += ka * math.cos(age * 114.0 + sw["t_snap"] * 1.4) * 0.72

        act.sort(key=lambda x: -x[0]); act = act[:8]; n_act = len(act)
        ring_u, rp_u, rmode_u, lobe_u, sa_u, sb_u = [], [], [], [], [], []
        for amp, r_now, width, dsign, freq, rpull, sw in act:
            ring_u.append((*map(float, sw["C"]), float(r_now)))
            rp_u.append((float(width), float(amp), float(sw["seed"]), 0.24))
            rmode_u.append((float(dsign), float(freq), float(rpull), 0.0))
            lobe_u += [tuple(map(float, lb)) for lb in sw["lobes"]]
            sa_u += [(*map(float, sw["A"][s]), float(sw["lag"][s])) for s in range(len(SEGS))]
            sb_u += [(*map(float, sw["B"][s]), float(sw["wgt"][s])) for s in range(len(SEGS))]
        z4 = (0.0, 0.0, 0.0, 0.0)
        fw = np.array([R[0, 2], R[1, 2], 0.0]); fw /= max(float(np.linalg.norm(fw)), 1e-6)
        hip = 0.5 * (P_limbs[i, 23] + P_limbs[i, 24])
        ring_prog["uRo"].value = tuple(map(float, ro)); ring_prog["uCam"].value = Rt
        ring_prog["uZref"].value = p_depth
        ring_prog["uWallN"].value = tuple(map(float, fw))
        ring_prog["uWallD"].value = float(fw @ hip) + 0.55
        ring_prog["iTime"].value = float(t_sec)
        ring_prog["uNR"].value = n_act
        ring_prog["uRing"].value = ring_u + [z4] * (8 - n_act)
        ring_prog["uRingP"].value = rp_u + [(1.0, 0.0, 0.0, 0.0)] * (8 - n_act)
        ring_prog["uRingMode"].value = rmode_u + [(1.0, 2.8, 0.0, 0.0)] * (8 - n_act)
        ring_prog["uLobe"].value = lobe_u + [(1.0, 0.0, 0.0, 1.0)] * (16 - len(lobe_u))
        ring_prog["uSegA"].value = sa_u + [z4] * (120 - len(sa_u))
        ring_prog["uSegB"].value = sb_u + [z4] * (120 - len(sb_u))
        fbR.use(); ctx.viewport = (0, 0, hw, hh); ctx.disable(moderngl.BLEND); va_ring.render(moderngl.TRIANGLES)

        # Silhouette outline pulses projected dynamically through current moving camera!
        pulses, pulses_w = [], []
        for ev_f, delta3, power in events_spec:
            for e_i, df in enumerate((0, 4)):
                age = (i - (ev_f + df)) / fps
                if 0.0 <= age <= 0.60:
                    c3 = 0.25 * (P_limbs[ev_f, 11] + P_limbs[ev_f, 12] + P_limbs[ev_f, 23] + P_limbs[ev_f, 24]) + delta3
                    cpx, _ = project_3d(c3, ro, R, f, W, H)
                    front_px = 6.0 + 360.0 * (age ** 0.78)
                    amp_p = 0.85 * power * (0.65 ** e_i) * ((1.0 - age / 0.60) ** 1.1)
                    pulses.append((float(cpx[0]), float(cpx[1]), float(front_px), float(amp_p)))
                    pulses_w.append((22.0 + 38.0 * age, 480.0))
        n_p = min(len(pulses), 6)
        while len(pulses) < 6:
            pulses.append((0.0, 0.0, 0.0, 0.0)); pulses_w.append((20.0, 200.0))

        # Pass 2: Full-res 10-Wavelength Spectral Rainbow Prism Composite!
        dist_t.write(dt_field.tobytes())
        gmask_t.write(gmask_now.tobytes())
        fbo_final.use(); ctx.viewport = (0, 0, W, H); ctx.disable(moderngl.BLEND)
        t_world.use(0); fA.use(1); rT.use(2); dist_t.use(3); gmask_t.use(4)
        comp_prog["uWorld"].value = 0; comp_prog["uField"].value = 1
        comp_prog["uRingT"].value = 2; comp_prog["uDist"].value = 3; comp_prog["uGhostMask"].value = 4
        comp_prog["iTime"].value = float(t_sec)
        comp_prog["uNear"].value = float(f * 0.24 / max(p_depth, 0.8))
        comp_prog["uGhostAlpha"].value = float(g_alpha)
        comp_prog["uGhostAge"].value = float(g_age)
        comp_prog["uNPulse"].value = n_p
        comp_prog["uPulse"].value = pulses[:6]
        comp_prog["uPulseW"].value = pulses_w[:6]
        comp_prog["uShakePx"].value = (float(shake[0]), float(shake[1]))
        comp_prog["uFrame"].value = float(i)
        va_comp.render(moderngl.TRIANGLES)

        ctx.finish()
        arr = np.frombuffer(t_final.read(), "f4").reshape(H, W, 4)[::-1]
        img = (np.clip(arr[..., :3], 0, 1) * 255 + 0.5).astype(np.uint8)[:, :, ::-1]
        return pl, img

    if stills:
        rows = []
        hero = None
        for idx in stills:
            pl, comp = render_one(idx)
            if idx == stills[len(stills) // 2]:
                hero = comp
            t_im = lambda im, lb: label(cv2.resize(im, (960, 540)), lb)
            rows.append(np.hstack([t_im(pl, f"plate {idx}"), t_im(comp, f"moving teleport {idx}")]))
        out_p = work(c, "still_moving_sheet.jpg")
        cv2.imwrite(str(out_p), np.vstack(rows), [cv2.IMWRITE_JPEG_QUALITY, 92])
        if hero is not None:
            cv2.imwrite(str(work(c, "still_moving_hero.jpg")), hero, [cv2.IMWRITE_JPEG_QUALITY, 94])
        print("stills ->", out_p)
        return

    comp_dir = work(c, "comp")
    shutil.rmtree(comp_dir, ignore_errors=True)
    comp_dir.mkdir(parents=True)
    ts = time.time()
    for i in range(N):
        _, comp = render_one(i)
        cv2.imwrite(str(comp_dir / f"{i:04d}.png"), comp, [cv2.IMWRITE_PNG_COMPRESSION, 1])
        if i % 30 == 0 or i == N - 1:
            print(f"moving_teleport frame {i}/{N} ({time.time() - ts:.1f}s)", flush=True)
    encode(c, out_dir=out_dir)


if __name__ == "__main__":
    main()
