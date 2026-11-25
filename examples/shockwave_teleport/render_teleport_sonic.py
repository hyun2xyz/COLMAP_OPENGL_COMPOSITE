"""Sonic Wave + Spectral Rainbow Prism Air-Distortion Teleport (ModernGL, OpenGL 3.3).

    python render_teleport_sonic.py [piece_dir] --still 46,56,78,125,160,182   quick look -> work/still_sonic_sheet.jpg
    python render_teleport_sonic.py [piece_dir] --out out_sonic                full render -> out_sonic/Sonic_Prism_Teleport_*.mp4

Combines:
- Spotless clean-plate spatial teleportation (LEFT shutter -> VANISH -> RIGHT shutter +740px -> VANISH -> LEFT shutter)
- Exact space_distortion optical physics (ZERO liquid droplets / ZERO glass shards — pure refractive air!):
  1. Body-conforming 3D acoustic sound waves (15-capsule MediaPipe skeleton SDF on soft-joined floor+wall 3D proxy,
     multi-echo sin(2.8*x)*exp(-0.42*x^2) pressure crests + 3D fbm wobble + directional lobes; imploding before snap,
     exploding outward in 3 acoustic echoes on every departure & arrival!).
  2. Exact CIE 1931 10-wavelength Cauchy spectral prism dispersion (400-700nm, Wyman et al. 2013) across 7 smear taps.
  3. Per-wavelength tilted-air rainbow specular prism glints catching sunlight along every sonic wave crest!
  4. Expanding silhouette acoustic outline pulses + vanished-body shimmering hot-air vortex + live limb air trails.
  5. Impact camera micro-shake + matched plate sensor grain. Zero text/HUD overlays.
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
]  # 15 body capsules: arms, hands, legs, shoulders, hips, torso sides, head


def find_ffmpeg():
    tl = root / "tools.local.json"
    if tl.exists():
        d = json.loads(tl.read_text(encoding="utf-8"))
        p = d.get("ffmpeg") or (d.get("tools") or {}).get("ffmpeg")
        if p and Path(p).exists():
            return str(p)
    return shutil.which("ffmpeg") or "ffmpeg"


def spectral(abbe=2.3, ns=10):
    """Exact CIE 1931 Cauchy spectral prism weights (from examples/space_distortion/render.py)."""
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

# Pass 0: Clean-plate spatial teleportation base scene (renders spotless teleported person + contact shadow to FBO)
WORLD_FS = """#version 330
uniform sampler2D uLive;
uniform sampler2D uClean;
uniform sampler2D uMask;
uniform vec2 uRes;
uniform float uTime, uShiftPx, uVisLeft, uVisRight, uRightFootX, uRightFootY;

layout(location=0) out vec4 oWorld;  /* RGB = teleported scene (sRGB), A = active person matte */
""" + NOISE_GLSL + """
void main(){
  vec2 px = vec2(gl_FragCoord.x, uRes.y - gl_FragCoord.y);
  vec2 uv = clamp(px / uRes, vec2(0.001), vec2(0.999));
  vec3 cleanCol = texture(uClean, uv).rgb;
  vec3 liveCol  = texture(uLive, uv).rgb;
  float mL = texture(uMask, uv).r;

  float nTear = fbm(vec3(uv * vec2(22.0, 16.0), uTime * 3.8));
  /* Fix-cut full left-shutter + sidewalk shadow zone wipe when dancer leaves left side -> ZERO shadow or mask ghost! */
  float leftZone = smoothstep(195.0, 245.0, px.x) * smoothstep(945.0, 900.0, px.x) * smoothstep(20.0, 65.0, px.y);
  float cleanMix = clamp(max(smoothstep(0.01, 0.30, mL) * (1.0 - smoothstep(0.85, 1.0, uVisLeft)), leftZone * (1.0 - smoothstep(0.15, 0.92, uVisLeft))), 0.0, 1.0);
  vec3 col = mix(liveCol, cleanCol, cleanMix);
  float pMask = 0.0;

  /* Realistic ground contact shadow under teleported shoes on right shutter */
  if (uVisRight > 0.01){
    vec2 ds  = (px - vec2(uRightFootX, uRightFootY)) / vec2(88.0, 19.0);
    vec2 ds2 = (px - vec2(uRightFootX + 38.0, uRightFootY - 4.0)) / vec2(115.0, 22.0);
    float sh = max(exp(-dot(ds, ds) * 1.45) * 0.44, exp(-dot(ds2, ds2) * 1.35) * 0.26) * uVisRight;
    col *= (1.0 - sh);
  }

  /* 1. LEFT person visibility (organic air-dissolve without bright white water-edge outline) */
  if (uVisLeft > 0.005 && mL > 0.01){
    float tear = smoothstep(1.0 - uVisLeft - 0.16, 1.0 - uVisLeft + 0.12, nTear * 0.60 + 0.40 * mL);
    float aL = mL * mix(tear, 1.0, smoothstep(0.86, 1.0, uVisLeft)) * smoothstep(0.0, 0.14, uVisLeft);
    col = mix(col, liveCol, aL);
    pMask = max(pMask, aL);
  }

  /* 2. Teleported RIGHT person (shifted by +uShiftPx) */
  if (uVisRight > 0.005){
    vec2 uvSrc = uv - vec2(uShiftPx / uRes.x, 0.0);
    if (uvSrc.x > 0.01 && uvSrc.x < 0.99){
      float mR = texture(uMask, uvSrc).r;
      if (mR > 0.01){
        vec3 personR = texture(uLive, uvSrc).rgb * vec3(1.02, 1.02, 1.01);
        float nTearR = fbm(vec3(uvSrc * vec2(22.0, 16.0), -uTime * 3.8));
        float tearR = smoothstep(1.0 - uVisRight - 0.16, 1.0 - uVisRight + 0.12, nTearR * 0.60 + 0.40 * mR);
        float aR = smoothstep(0.06, 0.72, mR) * mix(tearR, 1.0, smoothstep(0.86, 1.0, uVisRight)) * smoothstep(0.0, 0.14, uVisRight);
        col = mix(col, personR, aR);
        pMask = max(pMask, aR);
      }
    }
  }
  oWorld = vec4(clamp(col, 0.0, 1.0), pMask);
}
"""

# Pass 1A: Live limb air trails (exact space_distortion swept limb ribbon -> half-res float target)
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

# Pass 1B: Exact space_distortion 3D Body-Conforming Acoustic Sound Waves (RING_FS, half-res!)
# Supports BOTH outward explosive teleport sonic booms (mode > 0) AND inward imploding pre-teleport pressure waves (mode < 0)!
RING_FS = """#version 330
uniform vec2 uRes, uScale;
uniform float iTime, uF, uZref, uWallD;
uniform vec3 uRo, uWallN;
uniform mat3 uCam;
uniform int uNR;
uniform vec4 uRing[8];      /* xyz = wave center, w = front radius_m */
uniform vec4 uRingP[8];     /* x = width_m, y = amp_px, z = seed, w = wobble */
uniform vec4 uRingMode[8];  /* x = dirSign (+1 explode, -1 implode), y = freq (2.6..3.2), z = radialPull, w = unused */
uniform vec4 uLobe[16];     /* 2 von Mises directional lobes per wave */
uniform vec4 uSegA[120];    /* 15 MediaPipe 3D capsules per wave (xyz + lag_m) */
uniform vec4 uSegB[120];    /* 15 MediaPipe 3D capsules per wave (xyz + weight) */
out vec4 o;
""" + NOISE_GLSL + """
/* Soft-joined 3D proxy: concrete sidewalk floor z=0 + upright shutter wall through the dancer */
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
      /* Multi-crest acoustic pressure wave: pushes/pulls air along body-conforming wavefront normal! */
      float osc = sin(freq * x) * dirSign;
      s.xy += dir * amp * osc * env;
      /* Optional inward implosion vortex pinch right before snap! */
      if (uRingMode[k].z > 0.01 && d0 < uRing[k].w + 0.55){
        float pull = exp(-(d0 * d0) / max(uRing[k].w * uRing[k].w * 0.7, 0.06)) * uRingMode[k].z;
        vec2 tang = vec2(-dir.y, dir.x);
        s.xy += (-dir * 1.15 + tang * 0.42) * amp * 0.65 * pull;
        s.z += amp * pull / 16.0;
      }
      s.z += amp * env / 16.0;
      /* Track sharp acoustic wave crest slope for vivid rainbow prism glint! */
      crest += abs(cos(freq * x)) * env * amp / 22.0;
    }
  }
  o = vec4(s, crest);
}
"""

# Pass 2: Exact space_distortion Spectral Prism + Rainbow Glint + Silhouette Outline Pulses + Acoustic Afterimage!
COMP_FS = """#version 330
uniform sampler2D uWorld;     /* Pass-0 teleported clean-plate scene (bottom-up FBO, A=active person matte) */
uniform sampler2D uField;     /* Pass-1A blurred limb air trail field (bottom-up) */
uniform sampler2D uRingT;     /* Pass-1B half-res body-conforming 3D sonic wave field (bottom-up) */
uniform sampler2D uDist;      /* distance (px) outside active/vanishing silhouette (top-down float) */
uniform sampler2D uGhostMask; /* frozen departure silhouette for shimmering hot-air acoustic afterimage (top-down) */
uniform vec2 uRes;
uniform float iTime, uNear;
uniform float uGhostAlpha, uGhostShiftPx, uGhostAge;
uniform vec4 uPulse[6];       /* silhouette outline sonic pulses: x=cx, y=cy, z=front_px, w=amp */
uniform vec2 uPulseW[6];      /* x=width_px, y=reach_px */
uniform int uNPulse;
uniform int uNS;
uniform float uSpecK[12];
uniform vec3 uSpecW[12];
uniform float uGlint, uTilt, uRefl, uSpec, uLift;
uniform vec2 uShakePx;
uniform float uShakeRoll, uShakeZoom;
uniform float uBlackFloor, uGrainAmt, uFrame;

layout(location=0) out vec4 oCol;
layout(location=1) out vec4 oClean;
layout(location=2) out vec4 oAux;
layout(location=3) out vec4 oPrism;
""" + NOISE_GLSL + """
vec3 toLin(vec3 c){ return pow(max(c, 0.0), vec3(2.2)); }
/* Read Pass-0 bottom-up FBO at top-down image coordinates px */
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
  /* Kinetic teleport impact camera shake (translation + aspect-correct roll + impact zoom punch; dynamic overscan prevents edge clamp!) */
  vec2 uvScr = gl_FragCoord.xy / uRes;
  vec2 dc = (uvScr - 0.5) * vec2(uRes.x / uRes.y, 1.0);
  float cr = cos(uShakeRoll), sr = sin(uShakeRoll);
  vec2 dr = vec2(dc.x * cr - dc.y * sr, dc.x * sr + dc.y * cr);
  vec2 uvRot = dr * vec2(uRes.y / uRes.x, 1.0) / max(uShakeZoom, 1.001) + 0.5;
  vec2 uvSh = clamp(uvRot + uShakePx / uRes, vec2(0.001), vec2(0.999));
  vec2 px = vec2(uvSh.x * uRes.x, (1.0 - uvSh.y) * uRes.y);

  vec4 w0 = sampleWorldPx(px);
  float activeMask = w0.a;
  oClean = vec4(w0.rgb, 1.0);
  /* Keep live person readable while letting strong teleport shockwaves & prism rims sweep right over! */
  float keep = 1.0 - 0.68 * smoothstep(0.35, 0.85, activeMask);

  /* 1. Silhouette outline sonic pulses ("pang!" rings hugging the body right at teleport snap!) */
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

  /* Gradient of pulse+trail air height field for hot-air lens bend */
  float aL = fieldPx(px - vec2(e,0)).z * exp(-pow(sdist(px - vec2(e,0))/max(uNear,20.0),2.0)) + pulseField(px - vec2(e,0));
  float aR = fieldPx(px + vec2(e,0)).z * exp(-pow(sdist(px + vec2(e,0))/max(uNear,20.0),2.0)) + pulseField(px + vec2(e,0));
  float aU = fieldPx(px - vec2(0,e)).z * exp(-pow(sdist(px - vec2(0,e))/max(uNear,20.0),2.0)) + pulseField(px - vec2(0,e));
  float aD = fieldPx(px + vec2(0,e)).z * exp(-pow(sdist(px + vec2(0,e))/max(uNear,20.0),2.0)) + pulseField(px + vec2(0,e));
  vec2 g = vec2(aR - aL, aD - aU) / (2.0 * e);

  /* 2. Vanished-body hollow hot-air shimmer vortex (pure air heat-distortion, ZERO milky liquid!) */
  vec2 ghostWarp = vec2(0.0);
  float ghostCov = 0.0;
  if (uGhostAlpha > 0.005){
    vec2 uvG = vec2(px.x - uGhostShiftPx, px.y) / uRes;
    /* Heat-haze buoyant rise + acoustic ripple vibration */
    vec2 drift = vec2((fbm(vec3(uvG * 16.0, uGhostAge * 3.5)) - 0.5) * 0.022, -0.016 * uGhostAge) * (0.35 + uGhostAge);
    uvG += drift;
    if (uvG.x > 0.01 && uvG.x < 0.99 && uvG.y > 0.01 && uvG.y < 0.99){
      float gm = texture(uGhostMask, uvG).r;
      if (gm > 0.015){
        vec2 eg = vec2(4.0 / uRes.x, 4.0 / uRes.y);
        float gx = texture(uGhostMask, uvG + vec2(eg.x, 0)).r - texture(uGhostMask, uvG - vec2(eg.x, 0)).r;
        float gy = texture(uGhostMask, uvG + vec2(0, eg.y)).r - texture(uGhostMask, uvG - vec2(0, eg.y)).r;
        vec3 nq = vec3(px * 0.018, iTime * 4.5);
        vec2 shimmer = vec2(fbm(nq) - 0.5, fbm(nq + vec3(7.3, -4.1, 2.9)) - 0.5);
        /* Strong heat-haze acoustic air turbulence right inside vanished silhouette! */
        ghostWarp = (vec2(gx, gy) * 34.0 + shimmer * 38.0) * gm * uGhostAlpha;
        ghostCov = gm * uGhostAlpha;
      }
    }
  }

  /* 3. Body-conforming 3D acoustic sound waves from Pass-1B (RING_FS) */
  vec4 rg = ringPx(px);
  vec2 rw = rg.xy * keep;
  float rcov = (1.0 - exp(-rg.z)) * keep;
  float crest = rg.w * keep;

  /* Fine rising hot-air shimmer inside distorted air */
  vec3 qh = vec3(px * 0.013 - mv * iTime * 0.7, iTime * 2.2);
  vec2 hz = vec2(fbm(qh), fbm(qh + vec3(5.2, 1.3, 2.7))) - 0.5;
  vec2 warp = -mv * 16.0 * covTrail
            - g * 3400.0 * covTrail
            + hz * 14.0 * max(covTrail, rcov * 0.55)
            + ghostWarp;

  vec2 totalWarp = warp + rw;
  float cov = clamp(max(max(covTrail, rcov), ghostCov * 0.85), 0.0, 1.0);

  /* 4. Exact space_distortion 7-tap motion smear x 10-wavelength CIE 1931 Cauchy Spectral Prism! */
  vec3 col;
  vec3 prismOut = vec3(0.0);
  if (length(totalWarp) > 0.25 || cov > 0.01){
    vec3 acc = vec3(0.0), monoAcc = vec3(0.0);
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
      monoAcc += wt * plateLin(px + o);
      tw += wt;
    }
    vec3 bent = acc / max(tw, 1e-4);
    vec3 mono = monoAcc / max(tw, 1e-4);

    /* Tilted air surface normal -> clear air refraction Fresnel + per-wavelength RAINBOW PRISM GLINT! */
    vec2 airGrad = -g * 52.0 * covTrail - (rw + ghostWarp * 0.65) * uTilt;
    vec3 n = normalize(vec3(airGrad, 1.0));
    float fres = pow(1.0 - clamp(n.z, 0.0, 1.0), 0.85);
    vec3 Ld = normalize(vec3(-0.42, -0.76, 0.50));
    vec3 Hh = normalize(Ld + vec3(0.0, 0.0, 1.0));
    float spec = pow(max(dot(n, Hh), 0.0), 16.0) * smoothstep(0.02, 0.22, fres) * cov;

    vec2 rp = px + n.xy * 65.0;
    vec3 refl = (plateLin(rp) + plateLin(rp + vec2(8.0, 5.0)) + plateLin(rp - vec2(7.0, 8.0))) / 3.0;

    col = bent * (1.0 + uLift * cov);
    col += (refl * 1.15 + 0.008) * clamp(fres * uRefl, 0.0, 0.36) * 0.45 * cov;
    col += vec3(1.0, 0.98, 0.94) * uSpec * spec;

    /* Per-wavelength spectral rainbow glint along sonic wave crests & tilted hot-air vortex!
       Exact space_distortion physics: each wavelength's tilted air normal bends by uSpecK[j] -> real prismatic rainbow band! */
    vec3 glint = vec3(0.0);
    for (int j = 0; j < 12; j++){
      if (j >= uNS) break;
      vec3 nj = normalize(vec3((n.xy / max(n.z, 0.15)) * uSpecK[j], 1.0));
      /* Combine sharp sun catchlight + broad wave-crest chromatic shimmer */
      float gSharp = pow(max(dot(nj, Hh), 0.0), 220.0) * 1.35;
      float gBand  = pow(max(dot(nj, normalize(vec3(0.35, -0.55, 0.76))), 0.0), 72.0) * 0.55;
      glint += uSpecW[j] * (gSharp + gBand);
    }
    /* Modulate rainbow glint tightly to real wave steepness so calm background air stays 100% untouched! */
    float glintMask = clamp(fres * 1.45 + crest * 0.55 + ghostCov * 0.35, 0.0, 1.4) * smoothstep(0.03, 0.22, cov);
    col += glint * uGlint * glintMask;
    prismOut = max(bent - mono, vec3(0.0)) + glint * uGlint * glintMask * 0.55;

    /* Convert linear -> sRGB with gentle filmic highlight shoulder */
    col = col / (1.0 + max(col - 0.92, vec3(0.0)) * 0.35);
    col = pow(clamp(col, 0.0, 1.0), vec3(1.0 / 2.2));
  } else {
    col = w0.rgb;
  }

  /* Plate-matched black floor & animated camera sensor grain ONLY on distorted air pixels */
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
  oAux = vec4(totalWarp, length(totalWarp), cov);
  oPrism = vec4(prismOut, 1.0);
}
"""


_CMAP = None
def thermal_bgr(x):
    global _CMAP
    if _CMAP is None:
        try:
            from matplotlib import colormaps
            _CMAP = (colormaps["inferno"](np.linspace(0, 1, 256))[:, :3] * 255).astype(np.uint8)
        except Exception:
            t = np.linspace(0, 1, 256)
            v4 = np.stack([np.ones_like(t), t, t * t, t ** 3], 1)
            v2 = np.stack([t ** 4, t ** 5], 1)
            k4 = np.array([[0.13572138, 4.61539260, -42.66032258, 132.13108234],
                           [0.09140261, 2.19418839, 4.84296658, -14.18503333],
                           [0.10667330, 12.64194608, -60.58204836, 110.36276771]])
            k2 = np.array([[-152.94239396, 59.28637943], [4.27729857, 2.82956604], [-89.90310912, 27.34824973]])
            _CMAP = (np.clip(v4 @ k4.T + v2 @ k2.T, 0, 1) * 255).astype(np.uint8)
    return _CMAP[(np.clip(x, 0.0, 1.0) * 255).astype(np.int32)][..., ::-1]


def make_sonic_passes(clean_bgr, aux_arr, prism_arr, out_wh=(1280, 720), cell=60.0):
    """Build real OpenGL intermediate breakdown passes (plate gain = 1.0 everywhere calm!)."""
    H, W = clean_bgr.shape[:2]
    clean_f = clean_bgr.astype(np.float32) / 255.0
    yy, xx = np.mgrid[0:H, 0:W].astype(np.float32)
    u = xx + aux_arr[..., 0]
    v = yy + aux_arr[..., 1]
    mag = aux_arr[..., 2]

    # 1. Warp grid over full-brightness clean plate (crisp white grid + thin dark outline so every bend reads on bright shutter!)
    du = np.abs(((u / cell) + 0.5) % 1.0 - 0.5) * cell
    dv = np.abs(((v / cell) + 0.5) % 1.0 - 0.5) * cell
    d_line = np.minimum(du, dv)
    g_core = np.clip(1.9 - d_line, 0.0, 1.0)
    g_halo = np.clip(3.2 - d_line, 0.0, 1.0) * (1.0 - g_core)
    warp_hl = np.clip(mag / 18.0, 0.0, 1.0)
    line_col = np.stack([np.ones_like(warp_hl), 1.0 - 0.15 * warp_hl, 1.0 - 0.72 * warp_hl], axis=-1)  # BGR: white -> vivid cyan-gold where bent!
    grid_f = clean_f * (1.0 - 0.42 * g_halo[..., None])
    grid_f = grid_f * (1.0 - 0.88 * g_core[..., None]) + line_col * (0.88 * g_core[..., None])

    # 2. Thermal distortion field (inferno colormap blended strictly where acoustic air warp > 0; calm plate stays gain=1.0!)
    t_val = np.sqrt(np.clip(mag / 36.0, 0.0, 1.0))
    heat_col = thermal_bgr(t_val).astype(np.float32) / 255.0
    h_alpha = np.clip((mag - 1.2) / 8.5, 0.0, 0.90)[..., None]
    heat_f = clean_f * (1.0 - h_alpha) + heat_col * h_alpha

    # 3. Spectral Prism dispersion pass (CIE 1931 10-wavelength chromatic split + rainbow wave glint over full-brightness plate!)
    pr_rgb = np.clip(np.maximum(prism_arr[..., :3], 0.0), 0.0, 1.0) ** (1.0 / 2.2)
    pr_bgr = pr_rgb[..., ::-1]
    pr_peak = np.max(pr_bgr, axis=-1, keepdims=True)
    # Normalize chromatic hue & saturate cleanly so vivid rainbow bands read strongly on bright daylight shutter!
    pr_sat = np.clip(pr_bgr * 3.6, 0.0, 1.0)
    p_alpha = np.clip((pr_peak - 0.015) * 4.2, 0.0, 0.92)
    prism_f = clean_f * (1.0 - p_alpha * 0.65) + pr_sat * p_alpha

    def pack(im_f):
        u8 = (np.clip(im_f, 0.0, 1.0) * 255.0 + 0.5).astype(np.uint8)
        return cv2.resize(u8, out_wh, interpolation=cv2.INTER_AREA)

    return {
        "clean": cv2.resize(clean_bgr, out_wh, interpolation=cv2.INTER_AREA),
        "heat": pack(heat_f),
        "grid": pack(grid_f),
        "prism": pack(prism_f),
    }


def label(im, s):
    im = im.copy()
    cv2.putText(im, s, (12, im.shape[0] - 14), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 4, cv2.LINE_AA)
    cv2.putText(im, s, (12, im.shape[0] - 14), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 1, cv2.LINE_AA)
    return im


def encode(c, out_dir="out_sonic", name="Sonic_Prism_Teleport"):
    out = c["dir"] / out_dir
    out.mkdir(parents=True, exist_ok=True)
    r = f"{c['fps_num']}/{c['fps_den']}"
    ff = find_ffmpeg()
    tags = ["-pix_fmt", "yuv420p", "-color_primaries", "bt709", "-color_trc", "bt709", "-colorspace", "bt709", "-movflags", "+faststart"]
    x264 = ["-c:v", "libx264", "-profile:v", "high", "-level", "4.1", "-crf", "14", "-preset", "slow"]
    run = lambda a: subprocess.run([ff, "-v", "error", "-y"] + a, check=True)
    run(["-framerate", r, "-i", str(work(c, "comp_sonic", "%04d.png"))] + x264 + tags + [str(out / f"{name}_Composite.mp4")])
    # Pure clean stacked compare with ZERO text overlays
    run([
        "-framerate", r, "-i", str(work(c, "comp_sonic", "%04d.png")),
        "-framerate", r, "-i", str(work(c, "frames", "%04d.jpg")),
        "-filter_complex",
        "[0]scale=1080:608[a];[1]scale=1080:608[b];[a][b]vstack,pad=1080:1350:0:67:color=black",
    ] + x264 + tags + [str(out / f"{name}_Compare_1080x1350.mp4")])
    print("encoded ->", out)


def build_teleport_sound_waves(P_limbs, right_u, shift_m, events_spec, fw0, seed=19):
    """Build body-conforming 3D acoustic sound waves (exact space_distortion format) for all 4 teleport events:
      - Pre-snap imploding compression wave (mode=-1, rushes INWARD into the body silhouette!)
      - 3 explosive outward multi-echo acoustic shockwaves (mode=+1, echoing 4 frames apart!)"""
    rng = np.random.default_rng(seed)
    waves = []
    for ev_f, is_right, power in events_spec:
        Q = P_limbs[ev_f].copy()
        if is_right:
            Q = Q + right_u[None, :] * shift_m
        C = 0.25 * (Q[11] + Q[12] + Q[23] + Q[24])
        A = np.array([Q[s[0]] for s in SEGS], dtype=np.float64)
        B = np.array([Q[s[1]] for s in SEGS], dtype=np.float64)
        # Body parts radiate cleanly from torso outward with slight natural limb lag
        wgt = np.exp(-np.sum((0.5 * (A + B) - C) ** 2, 1) / (2.0 * 0.55 ** 2))
        lag = (1.0 - wgt) * 0.22
        lag[-1] -= 0.08  # head ball

        # 1) Pre-snap inward implosion acoustic pull wave (-13 frames -> 0 frames)
        lobes_imp = []
        for l in range(2):
            az = float(rng.uniform(0, 2 * math.pi))
            el = float(rng.uniform(-0.25, 0.35))
            lobes_imp.append((math.cos(el)*math.cos(az), math.cos(el)*math.sin(az), math.sin(el), 1.4))
        waves.append({
            "kind": "implode",
            "t0": ev_f - 13,
            "t_snap": ev_f,
            "C": C, "A": A, "B": B, "lag": lag, "wgt": wgt,
            "amp": 24.0 * power,
            "lobes": lobes_imp,
            "seed": float(rng.uniform(0, 50)),
        })

        # 2) 3 Explosive outward acoustic echoes (echo 0 strongest, echoes 1 & 2 follow 4f & 9f later!)
        for e_idx, (df, sc_a, spd, freq) in enumerate([(0, 1.00, 4.9, 2.75), (4, 0.72, 4.1, 3.05), (9, 0.48, 3.5, 3.25)]):
            lobes = []
            for l in range(2):
                if e_idx == 0:
                    base = math.atan2(-fw0[1], -fw0[0]) if l == 0 else math.atan2(fw0[1], fw0[0])
                    az = base + float(rng.uniform(-0.55, 0.55))
                    el = float(rng.uniform(-0.18, 0.32))
                else:
                    az = float(rng.uniform(0, 2 * math.pi))
                    el = float(rng.uniform(-0.28, 0.45))
                lobes.append((math.cos(el)*math.cos(az), math.cos(el)*math.sin(az), math.sin(el), float(rng.uniform(1.6, 3.6))))
            waves.append({
                "kind": "explode",
                "t0": ev_f + df,
                "t_snap": ev_f,
                "C": C, "A": A, "B": B, "lag": lag, "wgt": wgt,
                "amp": 34.0 * power * sc_a,
                "speed": spd * float(rng.uniform(0.92, 1.08)),
                "freq": freq,
                "lobes": lobes,
                "seed": float(rng.uniform(0, 50)),
            })
    return waves


def main():
    args = sys.argv[1:]
    stills = None
    out_dir = "out_sonic"
    save_passes = False
    piece_arg = None
    k = 0
    while k < len(args):
        if args[k] == "--still":
            stills = [int(x) for x in args[k + 1].split(",")]; k += 2
        elif args[k] == "--out":
            out_dir = args[k + 1]; k += 2
        elif args[k] == "--passes":
            save_passes = True; k += 1
        else:
            piece_arg = args[k]; k += 1

    c = load(piece_arg or (here / "piece_teleport"))
    S = scene(c)
    L = np.load(work(c, "limbs.npz"))
    P_limbs = np.asarray(L["P"], np.float64)
    img_lm = np.asarray(L["img"], np.float64)
    W, H, N, fps = c["W"], c["H"], c["N"], c["fps"]
    f = float(S["f"])
    cfg = c.get("teleport", {})

    cam_path = np.asarray(S["cam"], np.float64)
    cam_rot = np.asarray(S["rot"], np.float64)
    shift_px = float(cfg.get("shift_px", 740.0))
    t1_van = int(cfg.get("t1_vanish", 52))
    t1_arr = int(cfg.get("t1_arrive", 75))
    t2_van = int(cfg.get("t2_vanish", 156))
    t2_arr = int(cfg.get("t2_arrive", 178))

    torso_L = 0.25 * (P_limbs[:, 11] + P_limbs[:, 12] + P_limbs[:, 23] + P_limbs[:, 24])
    mid = N // 2
    fwd_u = torso_L[mid] - cam_path[mid]; fwd_u[2] = 0.0; fwd_u /= max(float(np.linalg.norm(fwd_u)), 1e-4)
    right_u = np.array([fwd_u[1], -fwd_u[0], 0.0], np.float64); right_u /= max(float(np.linalg.norm(right_u)), 1e-4)
    p_depth_med = float(np.median(L["depth"]))
    shift_m = shift_px * p_depth_med / f

    R0 = cam_rot[0]
    fw0 = np.array([R0[0, 2], R0[1, 2]]) / max(math.hypot(R0[0, 2], R0[1, 2]), 1e-6)

    # 4 teleport sonic events (2 departures + 2 arrivals -> 16 body-conforming acoustic sound waves!)
    events_spec = [
        (t1_van, False, 1.10),
        (t1_arr, True,  1.05),
        (t2_van, True,  1.10),
        (t2_arr, False, 1.05),
    ]
    sound_waves = build_teleport_sound_waves(P_limbs, right_u, shift_m, events_spec, fw0, seed=19)
    print(f"Sonic Prism Teleport: {len(sound_waves)} body-conforming acoustic waves | 10-wavelength CIE 1931 prism (Abbe=2.3)")

    ai_cp = work(c, "clean_plate_ai.jpg")
    clean_bgr = cv2.imread(str(ai_cp if ai_cp.exists() else work(c, "clean_plate.jpg")))
    print("clean plate ->", ai_cp if ai_cp.exists() else work(c, "clean_plate.jpg"))
    ghost_mask_1 = cv2.GaussianBlur(cv2.imread(str(work(c, "mask", f"{t1_van:04d}.png")), cv2.IMREAD_GRAYSCALE), (0, 0), 1.5)
    ghost_mask_2 = cv2.GaussianBlur(cv2.imread(str(work(c, "mask", f"{t2_van:04d}.png")), cv2.IMREAD_GRAYSCALE), (0, 0), 1.5)

    # Measure plate black floor & grain cleanly
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
    t_clean_o = ctx.texture((W, H), 4, dtype="f4")
    t_aux_o   = ctx.texture((W, H), 4, dtype="f4")
    t_prism_o = ctx.texture((W, H), 4, dtype="f4")
    fbo_final = ctx.framebuffer([t_final, t_clean_o, t_aux_o, t_prism_o])

    live_t = ctx.texture((W, H), 3); live_t.filter = (moderngl.LINEAR, moderngl.LINEAR); live_t.repeat_x = live_t.repeat_y = False
    clean_t = ctx.texture((W, H), 3); clean_t.filter = (moderngl.LINEAR, moderngl.LINEAR); clean_t.repeat_x = clean_t.repeat_y = False
    clean_t.write(clean_bgr[:, :, ::-1].tobytes())
    mask_t = ctx.texture((W, H), 1); mask_t.filter = (moderngl.LINEAR, moderngl.LINEAR); mask_t.repeat_x = mask_t.repeat_y = False
    gmask_t = ctx.texture((W, H), 1); gmask_t.filter = (moderngl.LINEAR, moderngl.LINEAR); gmask_t.repeat_x = gmask_t.repeat_y = False
    dist_t = ctx.texture((W, H), 1, dtype="f4"); dist_t.filter = (moderngl.LINEAR, moderngl.LINEAR); dist_t.repeat_x = dist_t.repeat_y = False

    # Static uniforms (10-wavelength CIE 1931 Cauchy prism matching space_distortion!)
    K_s, W_s = spectral(abbe=1.9, ns=10)
    world_prog["uRes"].value = (float(W), float(H))
    world_prog["uShiftPx"].value = shift_px
    trail_prog["uRes"].value = (float(W), float(H))
    trail_prog["uF"].value = f
    ring_prog["uRes"].value = (float(W), float(H))
    ring_prog["uScale"].value = (float(W) / hw, float(H) / hh)
    ring_prog["uF"].value = f
    comp_prog["uRes"].value = (float(W), float(H))
    comp_prog["uNS"].value = len(K_s)
    comp_prog["uSpecK"].value = list(K_s) + [1.0] * (12 - len(K_s))
    comp_prog["uSpecW"].value = [tuple(map(float, w)) for w in W_s] + [(0.0, 0.0, 0.0)] * (12 - len(W_s))
    comp_prog["uGlint"].value = 1.25
    comp_prog["uTilt"].value = 0.064
    comp_prog["uRefl"].value = 0.24
    comp_prog["uSpec"].value = 0.14
    comp_prog["uLift"].value = 0.04
    comp_prog["uBlackFloor"].value = black_floor
    comp_prog["uGrainAmt"].value = grain_amt

    limb_chains = [[13, 15, 19], [14, 16, 20], [11, 13, 15], [12, 14, 16]]
    limb_gains = [1.25, 1.25, 0.75, 0.75]

    def smooth01(x):
        x = max(0.0, min(1.0, x))
        return x * x * (3.0 - 2.0 * x)

    def at_pose(t, shifted_right):
        t = float(np.clip(t, 0, N - 1))
        i0 = min(int(t), N - 2); u = t - i0
        Q = P_limbs[i0] * (1.0 - u) + P_limbs[i0 + 1] * u
        if shifted_right:
            Q = Q + right_u[None, :] * shift_m
        return Q

    def chain_pts(Q, ids):
        pts = [Q[j] for j in ids]
        pts.append(pts[-1] + 0.28 * (pts[-1] - pts[-2]))
        seg = np.array([np.linalg.norm(pts[k + 1] - pts[k]) for k in range(len(pts) - 1)])
        cum = np.concatenate([[0.0], np.cumsum(seg)]) / max(seg.sum(), 1e-6)
        return np.array([[np.interp(u, cum, [p[a] for p in pts]) for a in range(3)] for u in np.linspace(0, 1, 9)])

    def build_trail_mesh(i, shifted_right):
        ages = np.arange(9 * 2 + 1) / 2.0
        verts = []
        for li, ids in enumerate(limb_chains):
            G = np.array([chain_pts(at_pose(i - a, shifted_right), ids) for a in ages])
            Gp = np.array([chain_pts(at_pose(i - a - 1.0, shifted_right), ids) for a in ages])
            for a in range(len(ages) - 1):
                for u in range(8):
                    for aa, uu in ((a, u), (a + 1, u), (a + 1, u + 1), (a, u), (a + 1, u + 1), (a, u + 1)):
                        verts.append([*G[aa, uu], *Gp[aa, uu], ages[aa] / ages[-1], uu / 8.0, limb_gains[li]])
        return np.array(verts, dtype=np.float32)

    def render_one(i):
        pl = frame(c, i)
        m_img = cv2.GaussianBlur(cv2.imread(str(work(c, "mask", f"{i:04d}.png")), cv2.IMREAD_GRAYSCALE), (0, 0), 1.0)
        t_sec = i / fps
        ro = cam_path[i]; R = cam_rot[i]
        Rt = tuple(R.astype(np.float32).T.flatten())
        p_depth = float(L["depth"][i])

        # Exact 2-teleport visibility state machine (LEFT -> VANISH -> RIGHT -> VANISH -> LEFT)
        if i < t1_van - 2:
            visL, visR = 1.0, 0.0
        elif i <= t1_van + 4:
            visL = 1.0 - smooth01((i - (t1_van - 2)) / 6.0); visR = 0.0
        elif i < t1_arr - 2:
            visL, visR = 0.0, 0.0
        elif i <= t1_arr + 4:
            visL = 0.0; visR = smooth01((i - (t1_arr - 2)) / 6.0)
        elif i < t2_van - 2:
            visL, visR = 0.0, 1.0
        elif i <= t2_van + 4:
            visL = 0.0; visR = 1.0 - smooth01((i - (t2_van - 2)) / 6.0)
        elif i < t2_arr - 2:
            visL, visR = 0.0, 0.0
        elif i <= t2_arr + 4:
            visL = smooth01((i - (t2_arr - 2)) / 6.0); visR = 0.0
        else:
            visL, visR = 1.0, 0.0

        # Active silhouette distance field (for outline pulses + limb air-trail falloff!)
        if visR > 0.25:
            # Shift mask to right shutter coordinates cleanly
            M_sh = np.float32([[1, 0, shift_px], [0, 1, 0]])
            m_act = cv2.warpAffine(m_img, M_sh, (W, H), borderValue=0)
            trail_right = True
        elif visL > 0.25:
            m_act = m_img
            trail_right = False
        elif t1_van <= i < t1_arr:
            # During vanished void, keep distance field around departure or approaching arrival silhouette!
            if i < t1_arr - 10:
                m_act = ghost_mask_1
                trail_right = False
            else:
                M_sh = np.float32([[1, 0, shift_px], [0, 1, 0]])
                m_act = cv2.warpAffine(m_img, M_sh, (W, H), borderValue=0)
                trail_right = True
        else:
            if i < t2_arr - 10:
                M_sh = np.float32([[1, 0, shift_px], [0, 1, 0]])
                m_act = cv2.warpAffine(ghost_mask_2, M_sh, (W, H), borderValue=0)
                trail_right = True
            else:
                m_act = m_img
                trail_right = False

        mb = (cv2.GaussianBlur(m_act, (0, 0), 2.0) > 120).astype(np.uint8)
        dt_field = cv2.GaussianBlur(cv2.distanceTransform(1 - mb, cv2.DIST_L2, 5), (0, 0), 2.5).astype(np.float32)

        # Vanished-body hollow hot-air shimmer silhouette
        g_alpha, g_shift, g_age = 0.0, 0.0, 0.0
        if t1_van <= i <= t1_van + 32:
            g_age = (i - t1_van) / fps
            g_alpha = max(0.0, 1.0 - (g_age / 1.05) ** 0.85) * 0.95
            g_shift = 0.0
            gmask_t.write(ghost_mask_1.tobytes())
        elif t2_van <= i <= t2_van + 32:
            g_age = (i - t2_van) / fps
            g_alpha = max(0.0, 1.0 - (g_age / 1.05) ** 0.85) * 0.95
            g_shift = shift_px
            gmask_t.write(ghost_mask_2.tobytes())
        else:
            gmask_t.write(ghost_mask_1.tobytes())

        # Shoes contact shadow position on right shutter
        foot_xs = [img_lm[i, j, 0] for j in (29, 30, 31, 32)]
        foot_ys = [img_lm[i, j, 1] for j in (29, 30, 31, 32)]
        r_foot_x = float(np.mean(foot_xs)) + shift_px
        r_foot_y = float(np.max(foot_ys)) - 4.0

        # Pass 0: Render teleported clean-plate scene into t_world
        live_t.write(pl[:, :, ::-1].tobytes())
        mask_t.write(m_img.tobytes())
        fbo_world.use()
        ctx.viewport = (0, 0, W, H)
        ctx.disable(moderngl.BLEND)
        live_t.use(0); clean_t.use(1); mask_t.use(2)
        world_prog["uLive"].value = 0; world_prog["uClean"].value = 1; world_prog["uMask"].value = 2
        world_prog["uTime"].value = float(t_sec)
        world_prog["uVisLeft"].value = float(visL)
        world_prog["uVisRight"].value = float(visR)
        world_prog["uRightFootX"].value = r_foot_x
        world_prog["uRightFootY"].value = r_foot_y
        va_world.render(moderngl.TRIANGLES)

        # Pass 1A: Live limb air trails -> separable Gaussian blur (only when dancer is visible!)
        fbA.use()
        ctx.viewport = (0, 0, hw, hh)
        fbA.clear(0.0, 0.0, 0.0, 0.0)
        if max(visL, visR) > 0.15:
            trail_prog["uRo"].value = tuple(map(float, ro))
            trail_prog["uCam"].value = Rt
            vbo = ctx.buffer(build_trail_mesh(i, trail_right).tobytes())
            va_t = ctx.vertex_array(trail_prog, [(vbo, "3f 3f 3f", "pos", "prv", "attr")])
            ctx.enable(moderngl.BLEND); ctx.blend_func = (moderngl.ONE, moderngl.ONE)
            va_t.render(moderngl.TRIANGLES)
            ctx.disable(moderngl.BLEND)
            va_t.release(); vbo.release()
        fbB.use(); fA.use(0); blur_prog["uSrc"].value = 0; blur_prog["uDir"].value = (1.0, 0.0); va_blur.render(moderngl.TRIANGLES)
        fbA.use(); fB.use(0); blur_prog["uDir"].value = (0.0, 1.0); va_blur.render(moderngl.TRIANGLES)

        # Pass 1B: Evaluate active body-conforming 3D acoustic sound waves + punchy kinetic teleport camera shake!
        act = []
        shake = np.zeros(2, dtype=np.float64)
        shake_roll = 0.0
        zoom_punch = 0.0
        for sw in sound_waves:
            if sw["kind"] == "implode":
                if sw["t0"] <= i <= sw["t_snap"] + 1:
                    u = (i - sw["t0"]) / max(float(sw["t_snap"] - sw["t0"]), 1.0)
                    r_now = max(0.06, 1.65 * (1.0 - smooth01(u) * 0.94))
                    amp = sw["amp"] * math.sin(min(u, 1.0) * math.pi * 0.88)
                    act.append((amp, r_now, 0.24, -1.0, 2.5, 0.85, sw))
                    # Rising pre-snap micro-tension vibration right before detonation
                    if u > 0.35:
                        pre_a = 4.6 * ((u - 0.35) / 0.65) ** 1.6
                        shake[0] += pre_a * math.sin(i * 2.7 + sw["t_snap"])
                        shake[1] += pre_a * math.cos(i * 3.1 + sw["t_snap"]) * 0.85
            else:
                age = (i - sw["t0"]) / fps
                life = 1.45
                if 0.0 <= age <= life:
                    r_now = 0.04 + sw["speed"] * (age ** 0.86)
                    amp = sw["amp"] * ((1.0 - age / life) ** 1.15) * min(1.0, age / 0.045)
                    width = 0.16 + 0.065 * r_now
                    act.append((amp, r_now, width, +1.0, sw["freq"], 0.0, sw))
                    # Punchy kinetic impact camera shake (translation + whip-roll + shock-zoom!) on every teleport snap!
                    if sw["t0"] == sw["t_snap"] and age < 0.48:
                        sgn = -1.0 if sw["t_snap"] in (t1_van, t2_arr) else 1.0
                        # Initial sharp directional slam + multi-frequency damped handheld rattle
                        slam = 20.0 * math.exp(-age * 16.0)
                        rattle = 22.0 * math.exp(-age * 7.4)
                        shake[0] += sgn * slam * 0.95 + rattle * math.sin(age * 92.0 + sw["t_snap"] * 1.3)
                        shake[1] += -abs(slam) * 0.68 + rattle * math.cos(age * 116.0 + sw["t_snap"] * 2.1) * 0.78
                        shake_roll += sgn * 0.021 * math.exp(-age * 9.5) * math.cos(age * 68.0)
                        zoom_punch += 0.032 * math.exp(-age * 11.5)

        shake[0] = float(np.clip(shake[0], -30.0, 30.0))
        shake[1] = float(np.clip(shake[1], -20.0, 20.0))
        shake_roll = float(np.clip(shake_roll, -0.024, 0.024))
        # Dynamic safe overscan guarantees 0.0px black borders even at maximum shake + roll!
        min_zoom = 1.0 + 2.0 * max(abs(shake[0]) / W, abs(shake[1]) / H) + abs(shake_roll) * (W / H + H / W) * 0.54 + 0.010
        shake_zoom = max(1.044, min_zoom + zoom_punch)

        act.sort(key=lambda x: -x[0])
        act = act[:8]
        n_act = len(act)
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
        ring_prog["uRo"].value = tuple(map(float, ro))
        ring_prog["uCam"].value = Rt
        ring_prog["uZref"].value = p_depth
        ring_prog["uWallN"].value = tuple(map(float, fw))
        ring_prog["uWallD"].value = float(fw @ hip) + 0.28
        ring_prog["iTime"].value = float(t_sec)
        ring_prog["uNR"].value = n_act
        ring_prog["uRing"].value = ring_u + [z4] * (8 - n_act)
        ring_prog["uRingP"].value = rp_u + [(1.0, 0.0, 0.0, 0.0)] * (8 - n_act)
        ring_prog["uRingMode"].value = rmode_u + [(1.0, 2.8, 0.0, 0.0)] * (8 - n_act)
        ring_prog["uLobe"].value = lobe_u + [(1.0, 0.0, 0.0, 1.0)] * (16 - len(lobe_u))
        ring_prog["uSegA"].value = sa_u + [z4] * (120 - len(sa_u))
        ring_prog["uSegB"].value = sb_u + [z4] * (120 - len(sb_u))
        fbR.use()
        ctx.viewport = (0, 0, hw, hh)
        ctx.disable(moderngl.BLEND)
        va_ring.render(moderngl.TRIANGLES)

        # Silhouette outline sonic pulses ("pang!" rings hugging the dancer's silhouette on every snap!)
        pulses, pulses_w = [], []
        for ev_f, is_right, power in events_spec:
            for e_i, df in enumerate((0, 5)):
                age = (i - (ev_f + df)) / fps
                if 0.0 <= age <= 0.65:
                    cx = float(img_lm[ev_f, [11, 12, 23, 24], 0].mean()) + (shift_px if is_right else 0.0)
                    cy = float(img_lm[ev_f, [11, 12, 23, 24], 1].mean())
                    front_px = 6.0 + 360.0 * (age ** 0.78)
                    amp_p = 0.85 * power * (0.65 ** e_i) * ((1.0 - age / 0.65) ** 1.1)
                    pulses.append((cx, cy, front_px, amp_p))
                    pulses_w.append((22.0 + 38.0 * age, 480.0))
        n_p = min(len(pulses), 6)
        while len(pulses) < 6:
            pulses.append((0.0, 0.0, 0.0, 0.0)); pulses_w.append((20.0, 200.0))

        # Pass 2: Full-res 10-Wavelength Spectral Prism + Rainbow Glint Composite!
        dist_t.write(dt_field.tobytes())
        fbo_final.use()
        ctx.viewport = (0, 0, W, H)
        ctx.disable(moderngl.BLEND)
        t_world.use(0); fA.use(1); rT.use(2); dist_t.use(3); gmask_t.use(4)
        comp_prog["uWorld"].value = 0; comp_prog["uField"].value = 1
        comp_prog["uRingT"].value = 2; comp_prog["uDist"].value = 3; comp_prog["uGhostMask"].value = 4
        comp_prog["iTime"].value = float(t_sec)
        comp_prog["uNear"].value = float(f * 0.24 / max(p_depth, 0.8))
        comp_prog["uGhostAlpha"].value = float(g_alpha)
        comp_prog["uGhostShiftPx"].value = float(g_shift)
        comp_prog["uGhostAge"].value = float(g_age)
        comp_prog["uNPulse"].value = n_p
        comp_prog["uPulse"].value = pulses[:6]
        comp_prog["uPulseW"].value = pulses_w[:6]
        comp_prog["uShakePx"].value = (float(shake[0]), float(shake[1]))
        comp_prog["uShakeRoll"].value = float(shake_roll)
        comp_prog["uShakeZoom"].value = float(shake_zoom)
        comp_prog["uFrame"].value = float(i)
        va_comp.render(moderngl.TRIANGLES)

        ctx.finish()
        arr = np.frombuffer(t_final.read(), "f4").reshape(H, W, 4)[::-1]
        img = (np.clip(arr[..., :3], 0, 1) * 255 + 0.5).astype(np.uint8)[:, :, ::-1]
        pass_dict = None
        if save_passes:
            c_arr = np.frombuffer(t_clean_o.read(), "f4").reshape(H, W, 4)[::-1]
            clean_o = (np.clip(c_arr[..., :3], 0, 1) * 255 + 0.5).astype(np.uint8)[:, :, ::-1]
            a_arr = np.frombuffer(t_aux_o.read(), "f4").reshape(H, W, 4)[::-1]
            p_arr = np.frombuffer(t_prism_o.read(), "f4").reshape(H, W, 4)[::-1]
            pass_dict = make_sonic_passes(clean_o, a_arr, p_arr)
        return pl, img, pass_dict

    if stills:
        rows = []
        hero = None
        for idx in stills:
            pl, comp, _ = render_one(idx)
            if idx == stills[len(stills) // 2]:
                hero = comp
            t_im = lambda im, lb: label(cv2.resize(im, (960, 540)), lb)
            rows.append(np.hstack([t_im(pl, f"plate {idx}"), t_im(comp, f"sonic prism {idx}")]))
        out_p = work(c, "still_sonic_sheet.jpg")
        cv2.imwrite(str(out_p), np.vstack(rows), [cv2.IMWRITE_JPEG_QUALITY, 92])
        if hero is not None:
            cv2.imwrite(str(work(c, "still_sonic_hero.jpg")), hero, [cv2.IMWRITE_JPEG_QUALITY, 94])
        print("stills ->", out_p)
        return

    comp_dir = work(c, "comp_sonic")
    shutil.rmtree(comp_dir, ignore_errors=True)
    comp_dir.mkdir(parents=True)
    pass_root = work(c, "pass_sonic")
    if save_passes:
        shutil.rmtree(pass_root, ignore_errors=True)
        for pk in ("clean", "heat", "grid", "prism"):
            (pass_root / pk).mkdir(parents=True, exist_ok=True)
    ts = time.time()
    for i in range(N):
        _, comp, pdict = render_one(i)
        cv2.imwrite(str(comp_dir / f"{i:04d}.png"), comp, [cv2.IMWRITE_PNG_COMPRESSION, 1])
        if save_passes and pdict is not None:
            for pk, pim in pdict.items():
                cv2.imwrite(str(pass_root / pk / f"{i:04d}.jpg"), pim, [cv2.IMWRITE_JPEG_QUALITY, 93])
        if i % 40 == 0 or i == N - 1:
            print(f"sonic_teleport frame {i}/{N} ({time.time() - ts:.1f}s)", flush=True)
    encode(c, out_dir=out_dir, name="Sonic_Prism_Teleport")


if __name__ == "__main__":
    main()
