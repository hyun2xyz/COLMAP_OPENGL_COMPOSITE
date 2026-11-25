"""#1 Kinetic Shockwave Teleport — 3D Refractive Air Shockwaves + Spatial Teleportation + 1,200 Shards (ModernGL, OpenGL 3.3).

    python render_teleport.py [piece_dir] --still 46,56,78,125,168,184   quick look -> work/still_teleport_sheet.jpg
    python render_teleport.py [piece_dir] --out out                      full render -> out/*_Composite.mp4, *_Compare_1080x1350.mp4

Zero text/HUD overlays — 100% pure optical cinema VFX:
- Uses MediaPipe Pose + Segmentation matte + spotless reconstructed clean plate + COLMAP 3D camera.
- Act 1 (f=39..51): Air implosion lensing & swirling intake streaks pulling into the dancer on the left shutter.
- Act 2 (f=52): SNAP TELEPORT #1 — dancer shatters & vanishes cleanly (revealing real clean plate) inside a massive double
  3D refractive air shockwave dome, ghostly glass afterimage silhouette, and 600 bursting 3D refractive shards + mist vortex.
- Act 3 (f=63..75): Implosion vortex on the right shutter -> BANG! Dancer teleports physically to the right shutter with
  realistic floor contact shadow + secondary outward shockwave & 600 shards.
- Act 4 (f=156..178): SNAP TELEPORT #2 — vanishes from right shutter and snaps back to left shutter in a kinetic shockwave burst!
- Pass 3: Camera impulse micro-shake on impacts + thin-lens bokeh + OLPF softness + specular halation + animated sensor grain.
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

# Pass 1: Clean-plate teleport compositing + 3D Spherical Refractive Air Shockwaves + Ghost Glass Afterimage + Floor Ripple
TELEPORT_FS = """#version 330
uniform sampler2D uLive;        /* current frame live footage (top-down) */
uniform sampler2D uClean;       /* spotless clean plate (top-down) */
uniform sampler2D uMask;        /* current frame MediaPipe person matte (top-down) */
uniform sampler2D uGhostMask;   /* departure-frame frozen person silhouette matte for glass afterimage */
uniform vec2 uRes;
uniform float uF, uTime;
uniform vec3 uRo;
uniform mat3 uCam;

/* Person visibility & placement:
   uPersonMode: 0=real left, 1=vanished (clean plate), 2=teleported right (shifted by uShiftPx) */
uniform float uShiftPx;
uniform float uVisLeft;         /* 0..1 visibility of person at original left position */
uniform float uVisRight;        /* 0..1 visibility of person at teleported right position */
uniform float uDissolveSeed;    /* organic spatial-tear dissolve progress (0=solid, 1=fully torn) */
uniform float uRightFootY;      /* pixel Y of shoes for soft contact shadow on right curb */
uniform float uRightFootX;      /* pixel X center of shoes on right curb */

/* Ghost glass afterimage at departure position */
uniform float uGhostAlpha;      /* 0..1 fading ghost silhouette */
uniform float uGhostShiftPx;    /* 0 if left departure, uShiftPx if right departure */
uniform float uGhostAge;        /* seconds since departure for glass expansion/shatter ripple */

/* Up to 4 active 3D refractive shockwave / implosion events:
   uWavePos[i] = vec4(x, y, z, radius_m)
   uWavePar[i] = vec4(strength, thickness_m, mode (-1=implode, +1=explode), age_s) */
uniform vec4 uWavePos[4];
uniform vec4 uWavePar[4];
uniform int uNumWaves;
uniform float uPersonDepth;

layout(location=0) out vec4 oComp;
layout(location=1) out vec4 oDepth;   /* R: 1e4 (open air), G: active person depth for Pass-2 & Pass-3 */
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

float fbm2(vec2 p) {
  float a = 0.0, w = 0.5;
  for (int i = 0; i < 4; i++) {
    a += w * vnoise2(p);
    p = p * 2.07 + vec2(13.1, -7.4);
    w *= 0.5;
  }
  return a;
}

/* Sample composited live/teleported person + spotless clean plate at top-down uv */
vec3 sampleWorldPlate(vec2 uv, out float activePersonMask) {
  uv = clamp(uv, vec2(0.002), vec2(0.998));
  vec3 cleanCol = texture(uClean, uv).rgb;
  vec3 liveCol  = texture(uLive, uv).rgb;
  float mL = texture(uMask, uv).r;

  /* Organic high-frequency spatial-tear noise for teleport vanish/appear transitions */
  float nTear = fbm2(uv * vec2(26.0, 18.0) + vec2(0.0, uTime * 3.5));

  /* Start from spotless clean plate (with real background motion outside left person mask!) */
  vec3 bg = mix(liveCol, cleanCol, smoothstep(0.02, 0.25, mL));
  vec3 col = bg;
  activePersonMask = 0.0;

  /* Soft realistic concrete contact shadow under teleported shoes on right shutter */
  if (uVisRight > 0.01) {
    vec2 px = uv * uRes;
    vec2 ds = (px - vec2(uRightFootX, uRightFootY)) / vec2(88.0, 19.0);
    float sh = exp(-dot(ds, ds) * 1.45) * 0.44 * uVisRight;
    /* subtle elongated directional shadow to match left brick/shutter lighting */
    vec2 ds2 = (px - vec2(uRightFootX + 38.0, uRightFootY - 4.0)) / vec2(115.0, 22.0);
    sh = max(sh, exp(-dot(ds2, ds2) * 1.35) * 0.26 * uVisRight);
    col *= (1.0 - sh);
  }

  /* 1. Original LEFT person visibility */
  if (uVisLeft > 0.005 && mL > 0.01) {
    float tear = smoothstep(1.0 - uVisLeft - 0.18, 1.0 - uVisLeft + 0.12, nTear * 0.65 + 0.35 * mL);
    float aL = mL * mix(tear, 1.0, smoothstep(0.88, 1.0, uVisLeft)) * smoothstep(0.0, 0.15, uVisLeft);
    /* Crisp bright refractive rim right along dissolving edge */
    float edgeGlow = smoothstep(0.02, 0.25, aL) * smoothstep(0.85, 0.35, aL) * (1.0 - smoothstep(0.85, 1.0, uVisLeft));
    col = mix(col, liveCol, aL);
    col += vec3(0.88, 0.95, 1.0) * edgeGlow * 0.65;
    activePersonMask = max(activePersonMask, aL);
  }

  /* 2. Teleported RIGHT person (shifted by +uShiftPx from real live MediaPipe silhouette!) */
  if (uVisRight > 0.005) {
    vec2 uvSrc = uv - vec2(uShiftPx / uRes.x, 0.0);
    if (uvSrc.x > 0.01 && uvSrc.x < 0.99) {
      float mR = texture(uMask, uvSrc).r;
      if (mR > 0.01) {
        vec3 personR = texture(uLive, uvSrc).rgb;
        /* Tiny subtle ambient brightness match (+3%) for right shutter daylight */
        personR *= vec3(1.02, 1.02, 1.01);
        float nTearR = fbm2(uvSrc * vec2(26.0, 18.0) - vec2(0.0, uTime * 3.5));
        float tearR = smoothstep(1.0 - uVisRight - 0.18, 1.0 - uVisRight + 0.12, nTearR * 0.65 + 0.35 * mR);
        float aR = smoothstep(0.06, 0.72, mR) * mix(tearR, 1.0, smoothstep(0.88, 1.0, uVisRight)) * smoothstep(0.0, 0.15, uVisRight);
        float edgeGlowR = smoothstep(0.02, 0.25, aR) * smoothstep(0.85, 0.35, aR) * (1.0 - smoothstep(0.85, 1.0, uVisRight));
        col = mix(col, personR, aR);
        col += vec3(0.88, 0.95, 1.0) * edgeGlowR * 0.65;
        activePersonMask = max(activePersonMask, aR);
      }
    }
  }
  return col;
}

void main() {
  vec2 px0 = vec2(gl_FragCoord.x, uRes.y - gl_FragCoord.y);
  vec2 uv0 = px0 / uRes;
  vec3 rd = normalize(uCam * vec3((px0 - 0.5 * uRes) / uF, 1.0));

  /* Evaluate 3D spherical air-refraction shockwave domes + implosion gravitational lensing */
  vec2 dispG = vec2(0.0);
  vec2 dispCA = vec2(0.0);
  float waveSpec = 0.0;
  float waveCgA = 0.0;
  float minWaveT = 1e4;
  float floorPulse = 0.0;

  for (int i = 0; i < 4; i++) {
    if (i >= uNumWaves) break;
    vec3 cen = uWavePos[i].xyz;
    float R = uWavePos[i].w;
    float str = uWavePar[i].x;
    float thick = max(uWavePar[i].y, 0.05);
    float mode = uWavePar[i].z;
    float age = uWavePar[i].w;
    if (str <= 0.002 || R <= 0.02) continue;

    /* Ray-sphere closest approach & shell intersection in true 3D! */
    vec3 oc = cen - uRo;
    float b = dot(oc, rd);
    vec3 closest = uRo + rd * b;
    float rPerp = length(closest - cen);

    /* 1. Primary 3D refractive shockwave shell (bends light like compressed air / Mach cone!) */
    float shellDist = rPerp - R;
    float shellEnv = exp(-(shellDist * shellDist) / (thick * thick));
    /* Secondary trailing acoustic ring inside main shockwave */
    float shell2 = exp(-pow((rPerp - R * 0.72) / (thick * 0.65), 2.0)) * 0.55;
    float totalShell = shellEnv + shell2;

    if (totalShell > 0.004) {
      vec3 rad3 = (closest - cen) / max(rPerp, 1e-4);
      vec3 qRad = transpose(uCam) * rad3;
      vec2 dir2 = normalize(qRad.xy + vec2(1e-5));
      /* Turbulent fluid air ripple along the shockwave wavefront */
      float ang = atan(dir2.y, dir2.x);
      float turb = (fbm2(vec2(ang * 3.5, rPerp * 3.8 - age * 5.0)) - 0.5) * 0.65;
      /* Derivative-like S-wave radial push: compresses outside, stretches inside like real blast wave! */
      float sWave = -shellDist / thick * shellEnv + (-(rPerp - R * 0.72) / (thick * 0.65)) * shell2 * 0.6;
      float push = (sWave * mode + turb * totalShell * 0.55) * str;
      dispG += dir2 * push * 0.044;
      dispCA += dir2 * abs(push) * 0.0055;
      /* Crisp subtle sunlit Fresnel grazing rim on the compressed air dome */
      waveSpec += pow(clamp(totalShell, 0.0, 1.0), 2.8) * str * (0.24 + 0.30 * max(turb + 0.35, 0.0));
      waveCgA = max(waveCgA, clamp(totalShell * str * 1.4, 0.0, 1.0));
      minWaveT = min(minWaveT, max(b - R, 0.35));
    }

    /* 2. Implosion gravitational-lens pull inside R when mode < 0 */
    if (mode < 0.0 && rPerp < R * 1.65) {
      float pullEnv = exp(-(rPerp * rPerp) / max(R * R * 0.55, 0.04));
      vec3 rad3 = (closest - cen) / max(rPerp, 1e-4);
      vec2 dir2 = normalize((transpose(uCam) * rad3).xy + vec2(1e-5));
      /* Spiral vortex twist + inward pinch */
      vec2 tang = vec2(-dir2.y, dir2.x);
      float sw = pullEnv * str;
      dispG += (-dir2 * 0.042 + tang * 0.018) * sw;
      dispCA += dir2 * 0.0045 * sw;
      waveCgA = max(waveCgA, clamp(sw * 1.2, 0.0, 1.0));
      minWaveT = min(minWaveT, max(b - 0.3, 0.35));
    }

    /* 3. Ground shockwave dust/pressure ripple on z=0 asphalt/curb */
    float tFloor = (rd.z < -0.004) ? (-uRo.z / rd.z) : 1e4;
    if (tFloor > 0.2 && tFloor < 16.0) {
      vec3 Xf = uRo + rd * tFloor;
      float rf = length(Xf.xy - cen.xy);
      float fRing = exp(-pow((rf - R) / max(thick * 1.25, 0.14), 2.0));
      float fn = fbm2(Xf.xy * 4.5 + vec2(age * 2.0, -age * 1.5));
      floorPulse += fRing * str * (0.35 + 0.65 * fn);
    }
  }

  /* Ghostly refractive liquid-glass afterimage silhouette right where the dancer vanished! */
  float ghostGlow = 0.0;
  if (uGhostAlpha > 0.005) {
    vec2 uvG = uv0 - vec2(uGhostShiftPx / uRes.x, 0.0);
    /* Subtle outward glass expansion & upward vapor drift as the silhouette shatters */
    vec2 drift = vec2((fbm2(uv0 * 14.0 + uGhostAge * 3.0) - 0.5) * 0.018, -0.012 * uGhostAge) * uGhostAge;
    uvG += drift;
    if (uvG.x > 0.01 && uvG.x < 0.99 && uvG.y > 0.01 && uvG.y < 0.99) {
      float gm = texture(uGhostMask, uvG).r;
      if (gm > 0.02) {
        /* Compute silhouette normal gradient to bend background like hollow blown crystal glass! */
        vec2 e = vec2(3.5 / uRes.x, 3.5 / uRes.y);
        float gx = texture(uGhostMask, uvG + vec2(e.x, 0.0)).r - texture(uGhostMask, uvG - vec2(e.x, 0.0)).r;
        float gy = texture(uGhostMask, uvG + vec2(0.0, e.y)).r - texture(uGhostMask, uvG - vec2(0.0, e.y)).r;
        float edge = clamp(length(vec2(gx, gy)) * 2.6, 0.0, 1.0);
        float ripple = (fbm2(uv0 * 24.0 + vec2(uGhostAge * 4.0, -uGhostAge * 2.5)) - 0.5);
        dispG += (vec2(gx, gy) * 0.036 + vec2(ripple) * 0.014) * uGhostAlpha * gm;
        dispCA += vec2(gx, gy) * 0.0045 * uGhostAlpha * gm;
        ghostGlow = (edge * 0.38 + gm * 0.10 * (0.5 + 0.8 * abs(ripple))) * uGhostAlpha;
        waveCgA = max(waveCgA, gm * uGhostAlpha);
        minWaveT = min(minWaveT, uPersonDepth);
      }
    }
  }

  /* Sample refracted scene with realistic subtle crown-glass RGB chromatic separation */
  float pMask0;
  vec3 colG = sampleWorldPlate(uv0 + dispG, pMask0);
  vec3 col;
  if (length(dispCA) > 0.00015) {
    float dummy;
    float rC = sampleWorldPlate(uv0 + dispG + dispCA, dummy).r;
    float bC = sampleWorldPlate(uv0 + dispG - dispCA, dummy).b;
    col = vec3(rC, colG.g, bC);
  } else {
    col = colG;
  }

  /* Subtle bright optical Fresnel highlights on shockwave dome + hollow glass afterimage + floor dust ring */
  col += vec3(0.92, 0.96, 1.00) * waveSpec * 0.16;
  col += vec3(0.92, 0.97, 1.00) * ghostGlow * 0.28;
  col += vec3(0.96, 0.95, 0.92) * clamp(floorPulse, 0.0, 1.0) * 0.12;

  float effOcc = (pMask0 > 0.48) ? uPersonDepth : 1e4;
  oComp = vec4(clamp(col, 0.0, 1.0), 1.0);
  oDepth = vec4(1e4, effOcc, pMask0, 1.0);
  oMeta = vec4(waveCgA, (waveCgA > 0.01) ? (waveCgA / max(minWaveT, 0.25)) : 0.0, 1.0 / max(effOcc, 0.25), 1.0);
}
"""

# Pass 2: Instanced 3D Raymarched Kinetic Glass/Air Shards, Velocity Streaks & Sparkling Vortex Droplets
SHARD_VS = """#version 330
in vec2 inQuad;
in vec4 inPosRad;      /* xyz = world pos (m), w = radius (m) */
in vec4 inAxisAsp;     /* xyz = kinetic stretch axis (unit), w = aspect stretch (1.1..4.2) */
in vec4 inShape;       /* x = seed, y = wobble, z = facet/sharpness (0=drop, 1=sharp shard), w = opacity (0..1) */
in vec4 inSideDir;     /* xyz = secondary axis, w = tier */

uniform vec2 uRes;
uniform float uF;
uniform vec3 uRo;
uniform mat3 uCam;

out vec3 vCenter;
out float vRad;
out vec3 vAxis;
out float vAsp;
out vec4 vShape;
out vec4 vSide;
out float vBoundR;

void main() {
  vCenter = inPosRad.xyz;
  vRad = inPosRad.w;
  vAxis = normalize(inAxisAsp.xyz);
  vAsp = inAxisAsp.w;
  vShape = inShape;
  vSide = inSideDir;
  vBoundR = vRad * (max(vAsp, 1.0) * 1.45 + 0.35);

  vec3 q = transpose(uCam) * (vCenter - uRo);
  if (vRad <= 0.0004 || inShape.w <= 0.01 || q.z <= 0.06 + vBoundR * 0.15) {
    gl_Position = vec4(2.0, 2.0, 2.0, 1.0);
    return;
  }
  vec2 ctrPx = uF * q.xy / q.z + 0.5 * uRes;
  float pxR = clamp(uF * vBoundR / max(q.z - vBoundR * 0.78, 0.05) + 3.5, 3.5, 920.0);
  vec2 px = ctrPx + inQuad * pxR;
  vec2 ndc = vec2(px.x / uRes.x * 2.0 - 1.0, (1.0 - px.y / uRes.y) * 2.0 - 1.0);
  gl_Position = vec4(ndc, 0.0, 1.0);
}
"""

SHARD_FS = """#version 330
in vec3 vCenter;
in float vRad;
in vec3 vAxis;
in float vAsp;
in vec4 vShape;
in vec4 vSide;
in float vBoundR;

uniform sampler2D uBase;    /* Pass-1 FBO color (bottom-up) */
uniform sampler2D uDepth;   /* Pass-1 FBO depth: G = effOcc, B = active person mask (bottom-up) */
uniform vec2 uRes;
uniform float uF, uIor, uDisp, uProxyDist, uPersonDepth;
uniform vec3 uRo, uSun;
uniform mat3 uCam;

layout(location=0) out vec4 oCol;
layout(location=1) out vec4 oMeta;

/* Hybrid SDF: transitions cleanly between organic teardrop/streak and sharp faceted crystal shard! */
float sdShard(vec3 p) {
  vec3 q = p - vCenter;
  vec3 ax = vAxis;
  vec3 s1 = normalize(vSide.xyz - ax * dot(vSide.xyz, ax) + vec3(1e-4, 2e-4, 3e-4));
  vec3 s2 = normalize(cross(ax, s1));
  vec3 loc = vec3(dot(q, s1), dot(q, s2), dot(q, ax));

  /* Organic stretched liquid drop */
  float taper = 1.0 + 0.25 * clamp(loc.z / max(vRad * vAsp, 1e-4), -1.0, 1.0);
  vec3 qs = vec3(loc.xy / max(taper, 0.55), loc.z / max(vAsp, 1.0));
  float dDrop = (length(qs) - vRad) * min(1.0, 0.72 + 0.28 / max(vAsp, 1.0));

  /* Sharp faceted crystal shard (octahedral prism stretched along kinetic velocity) */
  vec3 qf = vec3(loc.x / 0.75, loc.y / 0.36, loc.z / max(vAsp * 1.15, 1.0));
  float dFacet = (abs(qf.x) + abs(qf.y) + abs(qf.z) - vRad * 1.32) * 0.46;

  float d = mix(dDrop, dFacet, vShape.z);
  float ph = vShape.x * 21.3;
  float wob = sin(loc.x / max(vRad, 1e-4) * 3.0 + ph) * cos(loc.z / max(vRad * vAsp, 1e-4) * 2.8 - ph);
  return d + wob * vRad * vShape.y;
}

vec3 shardNormal(vec3 p) {
  float h = max(vRad * 0.018, 0.0002);
  vec2 e = vec2(h, -h);
  return normalize(
    e.xyy * sdShard(p + e.xyy) +
    e.yyx * sdShard(p + e.yyx) +
    e.yxy * sdShard(p + e.yxy) +
    e.xxx * sdShard(p + e.xxx)
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
  vec3 dep = texture(uDepth, gl_FragCoord.xy / uRes).rgb;
  float effOcc = dep.g;
  float pMask  = dep.b;

  vec3 acc = vec3(0.0);
  float cov = 0.0;
  float minT = 1e4;
  vec2 offs[4] = vec2[4](vec2(-0.25, -0.25), vec2(0.25, -0.25), vec2(-0.25, 0.25), vec2(0.25, 0.25));
  vec3 L1 = normalize(uSun);
  vec3 L2 = normalize(vec3(-0.35, -0.25, 0.90));

  for (int k = 0; k < 4; k++) {
    vec2 px = px0 + offs[k];
    vec3 rd = normalize(uCam * vec3((px - 0.5 * uRes) / uF, 1.0));
    vec3 oc = vCenter - uRo;
    float b = dot(oc, rd);
    float c = dot(oc, oc) - vBoundR * vBoundR;
    float disc = b * b - c;
    if (disc < 0.0) continue;
    float sq = sqrt(disc);
    float tStart = max(b - sq, 0.04);
    float tEnd = b + sq;
    if (tStart >= tEnd) continue;

    float t = tStart;
    float tHit = -1.0;
    float eps = max(vRad * 0.008, 0.00015);
    for (int s = 0; s < 28; s++) {
      float d = sdShard(uRo + rd * t);
      if (d < eps) { tHit = t; break; }
      t += d * 0.88;
      if (t > tEnd) break;
    }
    if (tHit < 0.0) continue;

    /* Occlude shards cleanly BEHIND active person silhouette! */
    float visP = (tHit > uPersonDepth + 0.05) ? (1.0 - smoothstep(0.22, 0.72, pMask)) : 1.0;
    if (visP < 0.02) continue;
    minT = min(minT, tHit);

    vec3 p1 = uRo + rd * tHit;
    vec3 n1 = shardNormal(p1);
    vec3 rdIn = refract(rd, n1, 1.0 / uIor);
    if (length(rdIn) < 0.001) rdIn = reflect(rd, n1);
    vec3 p2 = p1 + rdIn * max(vRad * 0.85, 0.004);
    vec3 rdOutG = normalize(mix(rd, rdIn, 0.85));
    vec3 bend = rdOutG - rd;
    vec3 rdOutR = normalize(rd + bend * ((uIor - uDisp) - 1.0) / max(uIor - 1.0, 0.05));
    vec3 rdOutB = normalize(rd + bend * ((uIor + uDisp) - 1.0) / max(uIor - 1.0, 0.05));

    vec3 refrCol = vec3(sampleEnv(p2, rdOutR).r, sampleEnv(p2, rdOutG).g, sampleEnv(p2, rdOutB).b);
    float ndv = max(dot(-rd, n1), 0.0);
    float innerMeniscus = 1.0 - 0.24 * exp(-pow((ndv - 0.38) / 0.18, 2.0));
    refrCol *= 1.02 * innerMeniscus;
    float fresnel = 0.04 + 0.96 * pow(1.0 - ndv, 3.8);
    vec3 reflDir = reflect(rd, n1);
    vec3 envRefl = sampleEnv(p1 + n1 * 0.004, reflDir);
    float skyUp = smoothstep(-0.1, 0.85, reflDir.z);
    envRefl = mix(envRefl * 0.96, envRefl * 1.15 + vec3(0.18, 0.21, 0.24), skyUp * 0.45);

    float sp1 = pow(max(dot(n1, normalize(L1 - rd)), 0.0), 210.0) * 1.85 + pow(max(dot(n1, normalize(L1 - rd)), 0.0), 28.0) * 0.12;
    float sp2 = pow(max(dot(n1, normalize(L2 - rd)), 0.0), 150.0) * 0.85;
    float rim = pow(1.0 - ndv, 2.8) * (0.32 + 0.24 * skyUp);

    vec3 col = mix(refrCol, envRefl, clamp(fresnel * 0.48, 0.0, 0.78));
    col += vec3(0.96, 0.99, 1.0) * (sp1 + sp2 + rim);
    acc += col * visP;
    cov += visP;
  }

  if (cov < 0.05) discard;
  float a = clamp(cov * 0.25 * vShape.w, 0.0, 1.0);
  oCol = vec4(clamp(acc / cov, 0.0, 1.0), a);
  oMeta = vec4(1.0, 1.0 / max(minT, 0.20), 1.0 / max(effOcc, 0.20), a);
}
"""

# Pass 3: Camera Optical Finish (Impact Micro-Shake + Thin-Lens Bokeh + OLPF Softness + Halation + Edge CA + Sensor Grain)
POST_FS = """#version 330
uniform sampler2D uScene;
uniform sampler2D uMeta;
uniform vec2 uRes;
uniform vec2 uShakePx;
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
  /* Tiny damped optical impact micro-shake (scales in by 0.6% so zero black edges ever appear!) */
  vec2 uvRaw = gl_FragCoord.xy / uRes;
  vec2 uv = (uvRaw - 0.5) * 0.992 + 0.5 + uShakePx / uRes;
  uv = clamp(uv, vec2(0.001), vec2(0.999));

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
  vec3 res = mix(col0, cgCol, clamp(cgA0, 0.0, 1.0));
  float fgAlpha = clamp(fgHaloW * 1.15, 0.0, 0.92);
  if (fgHaloW > 0.001) {
    res = mix(res, fgHaloCol / fgHaloW, fgAlpha * (1.0 - cgA0 * 0.55));
  }

  float cgPresence = clamp(max(cgA0, fgAlpha), 0.0, 1.0);
  res += halation * vec3(0.98, 0.99, 1.02) * 0.055;

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
    # 100% pure clean Composite.mp4 (ZERO text!)
    run(["-framerate", r, "-i", str(work(c, "comp", "%04d.png"))] + x264 + tags + [str(out / f"{nm}_Composite.mp4")])
    # Clean stacked Compare_1080x1350.mp4 (ZERO intrusive text design!)
    run([
        "-framerate", r, "-i", str(work(c, "comp", "%04d.png")),
        "-framerate", r, "-i", str(work(c, "frames", "%04d.jpg")),
        "-filter_complex",
        "[0]scale=1080:608[a];[1]scale=1080:608[b];[a][b]vstack,pad=1080:1350:0:67:color=black",
    ] + x264 + tags + [str(out / f"{nm}_Compare_1080x1350.mp4")])
    print("encoded ->", out)


def build_burst_particles(origin, right_u, fwd_u, up_u, mode, count, seed):
    """Build dense 3D kinetic glass/air shards + swirling micro-beads for one teleport burst event (implode or explode)."""
    rng = np.random.default_rng(seed)
    origin = np.asarray(origin, np.float64)
    parts = []
    for k in range(count):
        # Asymmetric 3D burst distribution (wider across frame & forward toward camera for rich depth bokeh!)
        ang = float(rng.uniform(0.0, 2.0 * math.pi))
        el = float(rng.uniform(-0.85, 0.95))
        dir3 = right_u * (math.cos(ang) * math.cos(el) * 1.15) + fwd_u * (math.sin(ang) * math.cos(el) * 0.95) + up_u * (math.sin(el) * 0.85)
        dir3 /= max(float(np.linalg.norm(dir3)), 1e-4)

        # Spawn distributed across body height (feet to head!) so the whole body explodes/re-assembles!
        body_off = right_u * float(rng.uniform(-0.26, 0.26)) + fwd_u * float(rng.uniform(-0.20, 0.20)) + up_u * float(rng.uniform(-0.78, 0.68))
        p_body = origin + body_off
        p_body[2] = max(p_body[2], 0.14)

        speed = float(np.exp(rng.uniform(math.log(1.1), math.log(5.4))))
        swirl = float(rng.uniform(-2.6, 2.6))
        rad = float(np.exp(rng.uniform(math.log(0.005), math.log(0.042))))
        facet = 1.0 if rng.random() < 0.58 else 0.0   # mix of sharp crystal shards & kinetic liquid streaks!
        side = rng.normal(0, 1, 3); side /= max(float(np.linalg.norm(side)), 1e-4)
        parts.append({
            "p_body": p_body,
            "dir": dir3,
            "speed": speed,
            "swirl": swirl,
            "rad": rad,
            "facet": facet,
            "seed": float(rng.uniform(0, 10)),
            "wob": float(rng.uniform(0.03, 0.11)),
            "side": side,
            "life": float(rng.uniform(0.95, 2.35)),
            "delay": float(rng.uniform(0.0, 0.08)),
            "mode": mode,  # +1 explode out, -1 implode in
        })
    return parts


def main():
    c = load(sys.argv[1] if len(sys.argv) > 1 and not sys.argv[1].startswith("--") else here / "piece_teleport")
    S = scene(c)
    L = np.load(work(c, "limbs.npz"))
    P_limbs = np.asarray(L["P"], np.float64)
    img_lm = np.asarray(L["img"], np.float64)
    W, H, N, fps = c["W"], c["H"], c["N"], c["fps"]
    f = float(S["f"])
    cfg = c.get("teleport", {})
    stills = [int(x) for x in sys.argv[sys.argv.index("--still") + 1].split(",")] if "--still" in sys.argv else None
    out_dir = sys.argv[sys.argv.index("--out") + 1] if "--out" in sys.argv else "out"

    cam_path = np.asarray(S["cam"], np.float64)
    cam_rot = np.asarray(S["rot"], np.float64)
    shift_px = float(cfg.get("shift_px", 740.0))
    t1_van = int(cfg.get("t1_vanish", 52))
    t1_arr = int(cfg.get("t1_arrive", 75))
    t2_van = int(cfg.get("t2_vanish", 156))
    t2_arr = int(cfg.get("t2_arrive", 178))

    # Torso 3D trajectory at LEFT (real) and RIGHT (shifted in 3D along camera-right axis!)
    torso_L = 0.25 * (P_limbs[:, 11] + P_limbs[:, 12] + P_limbs[:, 23] + P_limbs[:, 24])
    mid = N // 2
    fwd_u = torso_L[mid] - cam_path[mid]; fwd_u[2] = 0.0; fwd_u /= max(float(np.linalg.norm(fwd_u)), 1e-4)
    right_u = np.array([fwd_u[1], -fwd_u[0], 0.0], np.float64); right_u /= max(float(np.linalg.norm(right_u)), 1e-4)
    up_u = np.array([0.0, 0.0, 1.0], np.float64)

    # Exact metric shift corresponding to shift_px at person's depth!
    p_depth_med = float(np.median(L["depth"]))
    shift_m = shift_px * p_depth_med / f
    torso_R = torso_L + right_u[None, :] * shift_m
    print(f"Teleport setup: depth={p_depth_med:.2f}m, shift={shift_px:.0f}px ({shift_m:.2f}m), events=[{t1_van}->{t1_arr}, {t2_van}->{t2_arr}]")

    # Pre-build dense 3D particle bursts for all 4 teleport events (2 departures + 2 arrivals -> ~2,200 total event shards!)
    events = [
        # (trigger_frame, origin_3d, mode (+1 explode, -1 implode), burst_particles)
        (t1_van, torso_L[t1_van], -1, build_burst_particles(torso_L[t1_van], right_u, fwd_u, up_u, -1, 320, 101)),
        (t1_van, torso_L[t1_van], +1, build_burst_particles(torso_L[t1_van], right_u, fwd_u, up_u, +1, 620, 102)),
        (t1_arr, torso_R[t1_arr], -1, build_burst_particles(torso_R[t1_arr], right_u, fwd_u, up_u, -1, 320, 103)),
        (t1_arr, torso_R[t1_arr], +1, build_burst_particles(torso_R[t1_arr], right_u, fwd_u, up_u, +1, 560, 104)),
        (t2_van, torso_R[t2_van], -1, build_burst_particles(torso_R[t2_van], right_u, fwd_u, up_u, -1, 300, 105)),
        (t2_van, torso_R[t2_van], +1, build_burst_particles(torso_R[t2_van], right_u, fwd_u, up_u, +1, 600, 106)),
        (t2_arr, torso_L[t2_arr], -1, build_burst_particles(torso_L[t2_arr], right_u, fwd_u, up_u, -1, 300, 107)),
        (t2_arr, torso_L[t2_arr], +1, build_burst_particles(torso_L[t2_arr], right_u, fwd_u, up_u, +1, 560, 108)),
    ]

    clean_bgr = cv2.imread(str(work(c, "clean_plate.jpg")))
    ghost_mask_1 = cv2.GaussianBlur(cv2.imread(str(work(c, "mask", f"{t1_van:04d}.png")), cv2.IMREAD_GRAYSCALE), (0, 0), 1.2)
    ghost_mask_2 = cv2.GaussianBlur(cv2.imread(str(work(c, "mask", f"{t2_van:04d}.png")), cv2.IMREAD_GRAYSCALE), (0, 0), 1.2)

    ctx = moderngl.create_standalone_context(require=330)
    tp_prog = ctx.program(vertex_shader=VS, fragment_shader=TELEPORT_FS)
    sh_prog = ctx.program(vertex_shader=SHARD_VS, fragment_shader=SHARD_FS)
    post_prog = ctx.program(vertex_shader=VS, fragment_shader=POST_FS)
    copy2_prog = ctx.program(
        vertex_shader=VS,
        fragment_shader="""#version 330
        uniform sampler2D uSrc0, uSrc1; uniform vec2 uRes;
        layout(location=0) out vec4 o0; layout(location=1) out vec4 o1;
        void main(){ vec2 uv = gl_FragCoord.xy / uRes; o0 = texture(uSrc0, uv); o1 = texture(uSrc1, uv); }""",
    )

    tri = ctx.buffer(np.array([-1, -1, 3, -1, -1, 3], "f4").tobytes())
    va_tp = ctx.vertex_array(tp_prog, [(tri, "2f", "p")])
    va_copy2 = ctx.vertex_array(copy2_prog, [(tri, "2f", "p")])
    va_post = ctx.vertex_array(post_prog, [(tri, "2f", "p")])

    quad_vbo = ctx.buffer(np.array([-1, -1, 1, -1, 1, 1, -1, -1, 1, 1, -1, 1], "f4").tobytes())
    inst_buf = ctx.buffer(reserve=2600 * 16 * 4)
    va_sh = ctx.vertex_array(sh_prog, [
        (quad_vbo, "2f", "inQuad"),
        (inst_buf, "4f 4f 4f 4f/i", "inPosRad", "inAxisAsp", "inShape", "inSideDir"),
    ])

    live_t = ctx.texture((W, H), 3); live_t.filter = (moderngl.LINEAR, moderngl.LINEAR); live_t.repeat_x = live_t.repeat_y = False
    clean_t = ctx.texture((W, H), 3); clean_t.filter = (moderngl.LINEAR, moderngl.LINEAR); clean_t.repeat_x = clean_t.repeat_y = False
    clean_t.write(clean_bgr[:, :, ::-1].tobytes())
    mask_t = ctx.texture((W, H), 1); mask_t.filter = (moderngl.LINEAR, moderngl.LINEAR); mask_t.repeat_x = mask_t.repeat_y = False
    gmask_t = ctx.texture((W, H), 1); gmask_t.filter = (moderngl.LINEAR, moderngl.LINEAR); gmask_t.repeat_x = gmask_t.repeat_y = False

    t_hero = ctx.texture((W, H), 4, dtype="f4"); t_hero.filter = (moderngl.LINEAR, moderngl.LINEAR); t_hero.repeat_x = t_hero.repeat_y = False
    t_depth = ctx.texture((W, H), 4, dtype="f4"); t_depth.filter = (moderngl.NEAREST, moderngl.NEAREST)
    t_meta0 = ctx.texture((W, H), 4, dtype="f4"); t_meta0.filter = (moderngl.NEAREST, moderngl.NEAREST)
    fbo_hero = ctx.framebuffer([t_hero, t_depth, t_meta0])

    t_scene = ctx.texture((W, H), 4, dtype="f4"); t_scene.filter = (moderngl.LINEAR, moderngl.LINEAR); t_scene.repeat_x = t_scene.repeat_y = False
    t_meta = ctx.texture((W, H), 4, dtype="f4"); t_meta.filter = (moderngl.NEAREST, moderngl.NEAREST)
    fbo_scene = ctx.framebuffer([t_scene, t_meta])

    t_final = ctx.texture((W, H), 4, dtype="f4")
    fbo_final = ctx.framebuffer([t_final])

    copy2_prog["uRes"].value = (float(W), float(H))
    tp_prog["uRes"].value = (float(W), float(H))
    tp_prog["uF"].value = f
    tp_prog["uShiftPx"].value = shift_px

    sh_prog["uRes"].value = (float(W), float(H))
    sh_prog["uF"].value = f
    sh_prog["uSun"].value = tuple(map(float, S["sun"]))
    sh_prog["uIor"].value = 1.46
    sh_prog["uDisp"].value = 0.020
    sh_prog["uProxyDist"].value = 6.5

    post_prog["uRes"].value = (float(W), float(H))
    post_prog["uFocusDist"].value = p_depth_med
    post_prog["uNearBlur"].value = 4.6
    post_prog["uFarBlur"].value = 4.5
    post_prog["uBaseSoft"].value = 0.68
    post_prog["uMaxCoc"].value = 10.5
    post_prog["uGrainAmt"].value = 0.012
    post_prog["uBlackFloor"].value = 0.016

    def smooth01(x):
        x = max(0.0, min(1.0, x))
        return x * x * (3.0 - 2.0 * x)

    def render_one(i):
        pl = frame(c, i)
        m_img = cv2.GaussianBlur(cv2.imread(str(work(c, "mask", f"{i:04d}.png")), cv2.IMREAD_GRAYSCALE), (0, 0), 1.0)
        t_sec = i / fps
        ro = cam_path[i]; R = cam_rot[i]
        Rt = tuple(R.astype(np.float32).T.flatten())
        p_depth = float(L["depth"][i])

        # Person visibility state machine across the 2 spatial teleports (LEFT -> VANISH -> RIGHT -> VANISH -> LEFT)
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

        # Active ghost glass silhouette afterimage right after each departure!
        g_alpha, g_shift, g_age = 0.0, 0.0, 0.0
        if t1_van <= i <= t1_van + 34:
            g_age = (i - t1_van) / fps
            g_alpha = max(0.0, 1.0 - (g_age / 1.10) ** 0.85) * 0.92
            g_shift = 0.0
            gmask_t.write(ghost_mask_1.tobytes())
        elif t2_van <= i <= t2_van + 34:
            g_age = (i - t2_van) / fps
            g_alpha = max(0.0, 1.0 - (g_age / 1.10) ** 0.85) * 0.92
            g_shift = shift_px
            gmask_t.write(ghost_mask_2.tobytes())
        else:
            gmask_t.write(ghost_mask_1.tobytes())

        # Compute up to 4 active 3D refractive shockwaves / implosion lenses + impact micro-shake!
        w_pos, w_par = [], []
        shake = np.zeros(2, np.float64)
        wave_specs = [
            (t1_van, torso_L[t1_van]),
            (t1_arr, torso_R[t1_arr]),
            (t2_van, torso_R[t2_van]),
            (t2_arr, torso_L[t2_arr]),
        ]
        for tf, cen in wave_specs:
            dt = (i - tf) / fps
            # Implosion phase (-0.42s .. +0.02s)
            if -0.42 <= dt <= 0.04:
                u = (dt + 0.42) / 0.42
                R_imp = max(0.18, 1.85 * (1.0 - smooth01(u) * 0.88))
                str_imp = math.sin(min(u, 1.0) * math.pi * 0.85) * 1.15
                w_pos.append((float(cen[0]), float(cen[1]), float(cen[2]), float(R_imp)))
                w_par.append((float(str_imp), 0.32, -1.0, float(dt + 0.42)))
            # Explosive outward double shockwave dome (0.0s .. 1.35s)
            if 0.0 <= dt <= 1.35:
                R_exp = 0.22 + 3.85 * (dt ** 0.72)
                str_exp = math.exp(-dt * 2.15) * (1.45 if dt < 0.25 else 1.15)
                thick = 0.22 + 0.32 * dt
                w_pos.append((float(cen[0]), float(cen[1]), float(cen[2]), float(R_exp)))
                w_par.append((float(str_exp), float(thick), +1.0, float(dt)))
                # Subtle cinema camera kick right on the impact snap (decays in 0.25s!)
                if dt < 0.28:
                    amp = 5.5 * math.exp(-dt * 14.0)
                    shake[0] += amp * math.sin(dt * 95.0 + tf)
                    shake[1] += amp * math.cos(dt * 112.0 + tf * 1.3) * 0.75

        n_waves = min(len(w_pos), 4)
        while len(w_pos) < 4:
            w_pos.append((0.0, 0.0, 0.0, 0.0))
            w_par.append((0.0, 0.2, 1.0, 0.0))

        # Shoes position for soft contact shadow on right shutter
        foot_xs = [img_lm[i, j, 0] for j in (29, 30, 31, 32)]
        foot_ys = [img_lm[i, j, 1] for j in (29, 30, 31, 32)]
        r_foot_x = float(np.mean(foot_xs)) + shift_px
        r_foot_y = float(np.max(foot_ys)) - 4.0

        # Pass 1: Teleport plate + 3D spherical air shockwaves + ghost glass afterimage
        live_t.write(pl[:, :, ::-1].tobytes())
        mask_t.write(m_img.tobytes())
        live_t.use(0); clean_t.use(1); mask_t.use(2); gmask_t.use(3)
        tp_prog["uLive"].value = 0; tp_prog["uClean"].value = 1
        tp_prog["uMask"].value = 2; tp_prog["uGhostMask"].value = 3
        tp_prog["uRo"].value = tuple(map(float, ro))
        tp_prog["uCam"].value = Rt
        tp_prog["uTime"].value = float(t_sec)
        tp_prog["uVisLeft"].value = float(visL)
        tp_prog["uVisRight"].value = float(visR)
        tp_prog["uRightFootX"].value = r_foot_x
        tp_prog["uRightFootY"].value = r_foot_y
        tp_prog["uGhostAlpha"].value = float(g_alpha)
        tp_prog["uGhostShiftPx"].value = float(g_shift)
        tp_prog["uGhostAge"].value = float(g_age)
        tp_prog["uWavePos"].value = w_pos[:4]
        tp_prog["uWavePar"].value = w_par[:4]
        tp_prog["uNumWaves"].value = n_waves
        tp_prog["uPersonDepth"].value = p_depth

        fbo_hero.use()
        ctx.disable(moderngl.BLEND)
        va_tp.render(moderngl.TRIANGLES)

        fbo_scene.use()
        t_hero.use(0); t_meta0.use(1)
        copy2_prog["uSrc0"].value = 0; copy2_prog["uSrc1"].value = 1
        va_copy2.render(moderngl.TRIANGLES)

        # Pass 2: Evaluate all active 3D kinetic glass shards & swirling vortex droplets across all 4 events!
        rows_s = []
        for tf, cen, mode, plist in events:
            dt = (i - tf) / fps
            if mode < 0:
                # Implosion runs from dt in [-0.46 .. +0.02]
                if dt < -0.48 or dt > 0.04:
                    continue
                for pt in plist:
                    u = (dt + 0.45 - pt["delay"]) / 0.45
                    if u <= 0.0 or u >= 1.02:
                        continue
                    # Accelerates violently inward from outer ring into the person's body!
                    rem = max(0.0, 1.0 - u) ** 1.65
                    ang = pt["swirl"] * (1.0 - rem) * 1.4
                    c_a, s_a = math.cos(ang), math.sin(ang)
                    d0 = pt["dir"]
                    # rotate direction gently around up_u for vortex swirl
                    d_rot = np.array([d0[0]*c_a - d0[1]*s_a, d0[0]*s_a + d0[1]*c_a, d0[2]])
                    p_now = pt["p_body"] + d_rot * (pt["speed"] * 0.42 * rem)
                    p_now[2] = max(p_now[2], 0.08)
                    op = smooth01(u * 3.5) * (1.0 - smooth01((u - 0.85) / 0.17))
                    asp = 1.6 + 2.4 * (u ** 1.2)
                    z_cam = float(((p_now - ro) @ R)[2])
                    if z_cam > 0.12 and op > 0.02:
                        rows_s.append((z_cam, [
                            float(p_now[0]), float(p_now[1]), float(p_now[2]), float(pt["rad"] * (0.65 + 0.5 * u)),
                            float(d_rot[0]), float(d_rot[1]), float(d_rot[2]), float(asp),
                            pt["seed"], pt["wob"], pt["facet"], float(op),
                            float(pt["side"][0]), float(pt["side"][1]), float(pt["side"][2]), 1.0,
                        ]))
            else:
                # Explosion runs from dt in [0.0 .. 2.4s]
                if dt <= 0.0 or dt > 2.45:
                    continue
                for pt in plist:
                    age = dt - pt["delay"]
                    if age <= 0.0 or age > pt["life"]:
                        continue
                    # Explosive radial burst with aerodynamic drag + swirling vortex drift + slight gravity settle!
                    tau = 0.34
                    dist = pt["speed"] * tau * (1.0 - math.exp(-age / tau)) + pt["speed"] * 0.10 * age
                    ang = pt["swirl"] * (1.0 - math.exp(-age * 1.8)) * 0.65
                    c_a, s_a = math.cos(ang), math.sin(ang)
                    d0 = pt["dir"]
                    d_rot = np.array([d0[0]*c_a - d0[1]*s_a, d0[0]*s_a + d0[1]*c_a, d0[2]])
                    p_now = pt["p_body"] + d_rot * dist
                    p_now[2] = max(p_now[2] - 0.42 * age * age, 0.06)
                    # Kinetic stretch highest right on explosion snap, relaxing into tumbling crystal shards
                    asp = 1.25 + 2.6 * math.exp(-age * 4.5)
                    fade = 1.0 - smooth01((age - pt["life"] * 0.58) / (pt["life"] * 0.42))
                    grow = min(1.0, 0.35 + 0.65 * (age / 0.06))
                    z_cam = float(((p_now - ro) @ R)[2])
                    if z_cam > 0.14 and fade > 0.02:
                        rows_s.append((z_cam, [
                            float(p_now[0]), float(p_now[1]), float(p_now[2]), float(pt["rad"] * grow),
                            float(d_rot[0]), float(d_rot[1]), float(d_rot[2]), float(asp),
                            pt["seed"], pt["wob"], pt["facet"], float(fade),
                            float(pt["side"][0]), float(pt["side"][1]), float(pt["side"][2]), 1.0,
                        ]))

        rows_s.sort(key=lambda x: -x[0])
        if rows_s:
            rows_s = rows_s[:2600]
            data = np.array([r[1] for r in rows_s], dtype=np.float32)
            inst_buf.write(data.tobytes())
            t_hero.use(0); t_depth.use(1)
            sh_prog["uBase"].value = 0; sh_prog["uDepth"].value = 1
            sh_prog["uRo"].value = tuple(map(float, ro))
            sh_prog["uCam"].value = Rt
            sh_prog["uPersonDepth"].value = p_depth
            ctx.enable(moderngl.BLEND)
            ctx.blend_func = (moderngl.SRC_ALPHA, moderngl.ONE_MINUS_SRC_ALPHA)
            va_sh.render(moderngl.TRIANGLES, instances=len(rows_s))
            ctx.disable(moderngl.BLEND)

        # Pass 3: Camera Optical Finish (impact shake + bokeh + halation + sensor grain)
        fbo_final.use()
        t_scene.use(0); t_meta.use(1)
        post_prog["uScene"].value = 0; post_prog["uMeta"].value = 1
        post_prog["uShakePx"].value = (float(shake[0]), float(shake[1]))
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
            if i == stills[1]:
                hero = comp
            t_im = lambda im, s: label(cv2.resize(im, (960, 540)), s)
            rows.append(np.hstack([t_im(pl, f"plate {i}"), t_im(comp, f"teleport {i}")]))
        out_p = work(c, "still_teleport_sheet.jpg")
        cv2.imwrite(str(out_p), np.vstack(rows), [cv2.IMWRITE_JPEG_QUALITY, 92])
        if hero is not None:
            cv2.imwrite(str(work(c, "still_teleport_hero.jpg")), hero, [cv2.IMWRITE_JPEG_QUALITY, 94])
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
            print(f"teleport frame {i}/{N} ({time.time() - ts:.1f}s)", flush=True)
    encode(c, out_dir=out_dir)


if __name__ == "__main__":
    main()
