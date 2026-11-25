"""Liquid-Glass Rain -> Abrupt Freeze Snap (AESPA 'Dirty Work' optical water/liquid-glass look).

    python render_rain_freeze.py [piece_dir] --still 20,64,74,135,210   quick look -> work/still_rain_sheet.jpg
    python render_rain_freeze.py [piece_dir] --out out                  full render -> out/*_Composite.mp4, *_Compare_1080x1350.mp4

3-Pass Native OpenGL 3.30 + COLMAP + MediaPipe Pipeline:
- Act 1 (f=0..63, 0.00..2.52s): ~680 liquid-glass raindrops pour down rapidly along -Z across 4 depth layers
  around the person & concrete pillar, aerodynamically stretched into sleek teardrops with shutter-streak blur
  and subtle z=0 floor splash rings.
- Act 2 (f=64..71, 2.56..2.84s): "훅!" Abrupt 7-frame bullet-time brake slams fall speed from 100% to ~0.8%,
  while damped surface-tension spring oscillation snaps stretched raindrops into plump wobbling water/glass beads.
- Act 3 (f=72..224, 2.88..9.00s): Pure colorless dielectric water/liquid-glass droplets (IOR 1.333, crisp inverted
  spherical lensing, dark TIR crescent edges, pinpoint overhead lamp highlights, zero fake tint) hang weightless
  with real camera parallax, MediaPipe person depth occlusion, concrete pillar holdout, and creamy foreground lens bokeh.
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


VS_FSQ = "#version 330\nin vec2 p; void main(){ gl_Position = vec4(p, 0.0, 1.0); }"

# Pass 1: Plate + 3D Cylindrical Pillar Holdout Depth + Floor Rain Splash Ripples (during active rain)
BASE_FS = """#version 330
uniform sampler2D uPlate;
uniform vec2 uRes;
uniform float uF, uRainActive, uTime;
uniform vec3 uRo;
uniform mat3 uCam;
uniform vec4 uCylOcc;       /* xy = cyl center, z = radius, w = maxZ */
uniform vec4 uSplashes[24]; /* xy = world splash pos on z=0, z = phase (0..1), w = radius (m) */
uniform int uNumSplashes;

layout(location=0) out vec4 oComp;
layout(location=1) out vec4 oDepth; /* R: tCylOcc, G: tCylOcc */
layout(location=2) out vec4 oMeta;  /* R: cgAlpha, G: cgInvDepth, B: occInvDepth, A: 1.0 */

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
  if (zh < -0.15 || zh > uCylOcc.w) return 1e4;
  return t;
}

void main() {
  vec2 px0 = vec2(gl_FragCoord.x, uRes.y - gl_FragCoord.y);
  vec2 uv0 = px0 / uRes;
  vec3 col = texture(uPlate, clamp(uv0, vec2(0.002), vec2(0.998))).rgb;
  vec3 rd = normalize(uCam * vec3((px0 - 0.5 * uRes) / uF, 1.0));
  float tOcc = hitCylOcc(uRo, rd);

  float cgA = 0.0;
  float cgInvD = 0.0;

  /* Delicate water ripple rings on z=0 platform floor while pouring rain impacts (fade gracefully on freeze) */
  if (uRainActive > 0.01 && rd.z < -0.006) {
    float tf = -uRo.z / rd.z;
    if (tf > 0.3 && tf < min(tOcc, 14.0)) {
      vec3 Xf = uRo + rd * tf;
      float ringSum = 0.0;
      for (int i = 0; i < 24; i++) {
        if (i >= uNumSplashes) break;
        vec4 sp = uSplashes[i];
        float ph = sp.z;
        if (ph <= 0.01 || ph >= 0.99) continue;
        float dist = length(Xf.xy - sp.xy);
        float rad = sp.w * (0.18 + 0.95 * ph);
        float w = 0.014 + 0.012 * ph;
        float ring = exp(-pow((dist - rad) / w, 2.0)) * (1.0 - ph) * (1.0 - ph);
        ringSum += ring;
      }
      if (ringSum > 0.005) {
        float fade = uRainActive * (1.0 - smoothstep(8.5, 13.5, tf));
        col += vec3(0.92, 0.95, 0.98) * clamp(ringSum, 0.0, 1.2) * 0.24 * fade;
        cgA = clamp(ringSum * 0.45 * fade, 0.0, 0.7);
        cgInvD = cgA / max(tf, 0.3);
      }
    }
  }

  oComp = vec4(clamp(col, 0.0, 1.0), 1.0);
  oDepth = vec4(tOcc, tOcc, 0.0, 1.0);
  oMeta = vec4(cgA, cgInvD, 1.0 / max(tOcc, 0.20), 1.0);
}
"""

COPY2_FS = """#version 330
uniform sampler2D uSrc0;
uniform sampler2D uSrc1;
uniform vec2 uRes;
layout(location=0) out vec4 o0;
layout(location=1) out vec4 o1;
void main() {
  vec2 uv = gl_FragCoord.xy / uRes;
  o0 = texture(uSrc0, uv);
  o1 = texture(uSrc1, uv);
}
"""

# Pass 2: Instanced Dual-Interface Ray-Traced Pure Water / Liquid-Glass Droplets
# Exact AESPA 'Dirty Work' aesthetic: crystal-clear colorless water (IOR 1.333),
# crisp inverted spherical lensing, dark TIR crescent rims, pinpoint overhead studio glints,
# velocity-stretched teardrops + 3-tap vertical motion blur while pouring -> plump wobbling spheres on freeze!
DROP_VS = """#version 330
in vec2 inQuad;
in vec4 inPosRad;     /* xyz = world center (m), w = base radius (m) */
in vec4 inShapeVel;   /* x = stretchZ (>=1 falling, ~1 frozen), y = wobbleAmp, z = blurM (m vertical smear), w = seed */
in vec4 inExtra;      /* x = opacity, y = teardropTaper (0=sphere, 1=falling teardrop), z = phase, w = reserved */

uniform vec2 uRes;
uniform float uF;
uniform vec3 uRo;
uniform mat3 uCam;

out vec3 vCenter;
out float vRad;
out float vStretchZ;
out float vWobble;
out float vBlurM;
out float vSeed;
out float vAlpha;
out float vTaper;
out float vPhase;

void main() {
  vCenter = inPosRad.xyz;
  vRad = inPosRad.w;
  vStretchZ = inShapeVel.x;
  vWobble = inShapeVel.y;
  vBlurM = inShapeVel.z;
  vSeed = inShapeVel.w;
  vAlpha = inExtra.x;
  vTaper = inExtra.y;
  vPhase = inExtra.z;

  vec3 q = transpose(uCam) * (vCenter - uRo);
  float boundR = vRad * max(vStretchZ, 1.0) * 1.28 + vBlurM * 0.55;
  if (vAlpha <= 0.01 || q.z <= 0.08 + vRad * 0.6) {
    gl_Position = vec4(2.0, 2.0, 2.0, 1.0);
    return;
  }
  vec2 ctrPx = uF * q.xy / q.z + 0.5 * uRes;
  float pxRx = clamp(uF * (vRad * 1.35) / max(q.z - vRad * 0.9, 0.05) + 4.0, 3.5, 520.0);
  float pxRy = clamp(uF * boundR / max(q.z - vRad * 0.9, 0.05) + 5.0, 4.0, 760.0);
  vec2 px = ctrPx + vec2(inQuad.x * pxRx, inQuad.y * pxRy);
  vec2 ndc = vec2(px.x / uRes.x * 2.0 - 1.0, (1.0 - px.y / uRes.y) * 2.0 - 1.0);
  gl_Position = vec4(ndc, 0.0, 1.0);
}
"""

DROP_FS = """#version 330
in vec3 vCenter;
in float vRad;
in float vStretchZ;
in float vWobble;
in float vBlurM;
in float vSeed;
in float vAlpha;
in float vTaper;
in float vPhase;

uniform sampler2D uBase;       /* Pass-1 scene color (bottom-up) */
uniform sampler2D uDepth;      /* Pass-1 cylindrical pillar holdout depth (bottom-up) */
uniform sampler2D uPersonMask; /* MediaPipe soft person matte (top-down) */
uniform vec2 uRes;
uniform float uF, uIor, uDisp, uPersonDepth, uProxyDist;
uniform vec3 uRo, uSun;
uniform mat3 uCam;

layout(location=0) out vec4 oCol;
layout(location=1) out vec4 oMeta;

/* Organic water/liquid-glass droplet SDF:
   - Falling: sleek aerodynamically stretched teardrop (plump bottom head, tapering upper tail)
   - Frozen: plump oscillating spherical/oval bead with smooth surface-tension quadrupole modes! */
float sdDrop(vec3 p, vec3 ctr) {
  vec3 q = p - ctr;
  /* Scale Z by vertical stretch (falling elongation OR damped freeze-snap oscillation) */
  float sz = max(vStretchZ, 0.65);
  float sxy = 1.0 / sqrt(sz); /* volume conservation! */
  vec3 qs = vec3(q.x / sxy, q.y / sxy, q.z / sz);

  /* Aerodynamic teardrop taper when falling rapidly (narrow top +z, fuller bottom -z) */
  if (vTaper > 0.01) {
    float topFrac = clamp(qs.z / max(vRad, 0.002), -1.0, 1.2);
    float pinch = 1.0 + vTaper * (0.42 * max(topFrac, 0.0) - 0.10 * max(-topFrac, 0.0));
    qs.xy *= pinch;
  }

  float r = length(qs);
  vec3 u = qs / max(r, 1e-6);
  /* Smooth organic quadrupole & octupole surface-tension oscillation (natural liquid water wobble) */
  float w1 = (3.0 * u.z * u.z - 1.0) * 0.5 * vWobble;
  float w2 = sin(u.x * 3.1 + vSeed * 19.3) * cos(u.y * 2.9 - vSeed * 13.7 + vPhase) * 0.085;
  float w3 = sin((u.x - u.z) * 4.7 + vSeed * 31.1) * 0.032;
  float effR = vRad * (1.0 + w1 + w2 + w3);
  return (r - effR) * min(sxy, 1.0) * 0.82;
}

vec3 dropNormal(vec3 p, vec3 ctr) {
  float h = max(vRad * 0.015, 0.00025);
  vec2 e = vec2(h, -h);
  return normalize(
    e.xyy * sdDrop(p + e.xyy, ctr) +
    e.yyx * sdDrop(p + e.yyx, ctr) +
    e.yxy * sdDrop(p + e.yxy, ctr) +
    e.xxx * sdDrop(p + e.xxx, ctr)
  );
}

/* Sample live scene plate cleanly along an arbitrary refracted or reflected 3D world ray! */
vec3 sampleSceneRay(vec3 ro, vec3 rd) {
  float tf = (rd.z < -0.015) ? (-ro.z / rd.z) : 1e4;
  if (tf < 0.05 || tf > 24.0) tf = 1e4;
  vec3 X = ro + rd * min(tf, uProxyDist);
  vec3 q = transpose(uCam) * (X - uRo);
  if (q.z <= 0.15) {
    /* Graceful dark moody ceiling/platform ambient when reflecting directly backward */
    float up = clamp(rd.z * 0.5 + 0.5, 0.0, 1.0);
    return mix(vec3(0.05, 0.05, 0.06), vec3(0.18, 0.17, 0.19), up);
  }
  vec2 px = uF * q.xy / q.z + 0.5 * uRes;
  vec2 uvTop = px / uRes;
  /* Single-fold clean clamp so strong inverted lensing never produces edge streaks */
  vec2 c = 1.0 - abs(1.0 - abs(clamp(uvTop, vec2(-0.65), vec2(1.65))));
  vec2 uvFbo = vec2(c.x, 1.0 - c.y);
  return texture(uBase, clamp(uvFbo, vec2(0.002), vec2(0.998))).rgb;
}

/* Shade one water/liquid-glass droplet sample centered at ctr (called 1x when frozen, 3x vertical shutter when pouring) */
bool shadeDropSample(vec3 ro, vec3 rd, vec3 ctr, float tMaxOcc, float pMask, out vec3 outCol, out float outT, out float outCov) {
  float boundR = vRad * max(vStretchZ, 1.0) * 1.32;
  vec3 oc = ro - ctr;
  float b = dot(oc, rd);
  float c = dot(oc, oc) - boundR * boundR;
  float disc = b * b - c;
  if (disc < 0.0) return false;
  float sq = sqrt(disc);
  float t0 = max(-b - sq, 0.04);
  float t1 = -b + sq;
  if (t1 <= t0 || t0 >= tMaxOcc) return false;

  float t = t0;
  float minD = 1e4;
  bool hit = false;
  for (int i = 0; i < 30; i++) {
    float d = sdDrop(ro + rd * t, ctr);
    minD = min(minD, d);
    if (d < max(vRad * 0.005, 0.00018)) { hit = true; break; }
    t += max(d * 0.92, vRad * 0.012);
    if (t > min(t1, tMaxOcc)) break;
  }
  if (!hit || t >= tMaxOcc) return false;

  /* MediaPipe person depth occlusion: drops behind the man vanish cleanly behind his silhouette! */
  float occPerson = (t > uPersonDepth + 0.04) ? pMask : 0.0;
  float vis = (1.0 - occPerson) * vAlpha;
  if (vis < 0.02) return false;

  vec3 p1 = ro + rd * t;
  vec3 N1 = dropNormal(p1, ctr);
  float cosI = clamp(dot(-rd, N1), 0.0, 1.0);

  /* Dual-interface physical water lensing (IOR 1.333):
     Ray enters droplet, marches through interior to exit face, and refracts back into world -> authentic inverted lensing! */
  vec3 rdInR = refract(rd, N1, 1.0 / (uIor - uDisp));
  vec3 rdInG = refract(rd, N1, 1.0 / uIor);
  vec3 rdInB = refract(rd, N1, 1.0 / (uIor + uDisp));

  /* March through water interior to exit surface (short accurate step) */
  float tIn = max(vRad * 0.35, 0.002);
  for (int j = 0; j < 14; j++) {
    float di = -sdDrop(p1 + rdInG * tIn, ctr);
    if (di < max(vRad * 0.012, 0.0003)) break;
    tIn += max(di * 0.85, vRad * 0.03);
    if (tIn > boundR * 2.4) break;
  }
  vec3 p2 = p1 + rdInG * tIn;
  vec3 N2 = -dropNormal(p2, ctr); /* inward normal at exit */

  vec3 rdOutR = refract(rdInR, N2, uIor - uDisp);
  vec3 rdOutG = refract(rdInG, N2, uIor);
  vec3 rdOutB = refract(rdInB, N2, uIor + uDisp);

  /* Detect Total Internal Reflection (TIR) at steep exit angles -> dark glossy rim band like AESPA Dirty Work! */
  bool isTir = (dot(rdOutG, rdOutG) < 0.01);
  vec3 refrCol;
  if (isTir) {
    vec3 rdTir = reflect(rdInG, N2);
    /* Dark moody internal reflection pocket */
    refrCol = sampleSceneRay(p2, rdTir) * 0.24;
  } else {
    if (dot(rdOutR, rdOutR) < 0.01) rdOutR = rdOutG;
    if (dot(rdOutB, rdOutB) < 0.01) rdOutB = rdOutG;
    refrCol = vec3(
      sampleSceneRay(p2, normalize(rdOutR)).r,
      sampleSceneRay(p2, normalize(rdOutG)).g,
      sampleSceneRay(p2, normalize(rdOutB)).b
    );
    /* Crisp internal optical magnification contrast + delicate dark grazing TIR meniscus band! */
    float tirEdge = smoothstep(0.46, 0.12, cosI);
    refrCol *= mix(1.06, 0.34, tirEdge * 0.72);
  }

  /* Pure colorless water Fresnel reflection (F0 = 0.02 for water, rising sharply at grazing rim) */
  float F0 = 0.022;
  float fres = F0 + (1.0 - F0) * pow(1.0 - cosI, 4.2);
  vec3 R = reflect(rd, N1);
  vec3 envRefl = sampleSceneRay(p1, R);
  float envLum = dot(envRefl, vec3(0.299, 0.587, 0.114));
  /* Boost bright ceiling lights in reflection while keeping dark subway walls deeply reflective */
  vec3 reflCol = envRefl * (0.45 + 0.95 * smoothstep(0.15, 0.80, envLum));

  /* Overhead platform studio lamp pinpoint specular glints (signature AESPA Dirty Work droplet sparkle!) */
  vec3 L1 = normalize(uSun);
  vec3 L2 = normalize(vec3(-0.35, 0.55, 0.76)); /* forward-left platform ceiling light */
  vec3 L3 = normalize(vec3(0.62, -0.20, 0.75)); /* right-side warm overhead lamp */
  float sp1 = pow(max(0.0, dot(R, L1)), 140.0) * 2.20 + pow(max(0.0, dot(R, L1)), 28.0) * 0.26;
  float sp2 = pow(max(0.0, dot(R, L2)), 110.0) * 1.45;
  float sp3 = pow(max(0.0, dot(R, L3)), 90.0)  * 0.85;
  /* Crisp subtle lower internal caustic glow inside bottom of droplet (classic real water drop lens focus!) */
  float botGlow = pow(max(0.0, -N1.z), 2.8) * smoothstep(0.15, 0.72, cosI) * 0.24;

  vec3 col = mix(refrCol, reflCol, clamp(fres * 1.15, 0.0, 0.92));
  col += vec3(1.00, 0.98, 0.95) * (sp1 + sp2) + vec3(1.00, 0.94, 0.86) * (sp3 + botGlow);
  /* Subtle filmic highlight rolloff */
  col = col / (1.0 + max(col - 0.88, vec3(0.0)) * 0.55);

  /* Crisp optical edge antialiasing */
  float edgeAA = smoothstep(0.02, 0.20, cosI);
  outCol = clamp(col, 0.0, 1.0);
  outT = t;
  outCov = vis * (0.28 + 0.72 * edgeAA);
  return true;
}

void main() {
  vec2 uvFbo = gl_FragCoord.xy / uRes;
  vec2 px = vec2(gl_FragCoord.x, uRes.y - gl_FragCoord.y);
  vec2 uvTop = px / uRes;
  vec3 rd = normalize(uCam * vec3((px - 0.5 * uRes) / uF, 1.0));
  float tOcc = texture(uDepth, uvFbo).r;
  float pMask = texture(uPersonMask, clamp(uvTop, vec2(0.001), vec2(0.999))).r;

  /* During fast pouring rain (vBlurM > 0.008m), 7-tap continuous vertical shutter motion blur (zero dotted gaps)!
     Once frozen (vBlurM ~ 0), single razor-sharp raymarch! */
  if (vBlurM > 0.008) {
    vec3 accCol = vec3(0.0);
    float accCov = 0.0;
    float bestT = 1e4;
    for (int s = -3; s <= 3; s++) {
      float u = float(s) / 3.0;
      vec3 cOff = vCenter + vec3(0.0, 0.0, u * vBlurM * 0.45);
      vec3 sc; float st, sv;
      if (shadeDropSample(uRo, rd, cOff, tOcc, pMask, sc, st, sv)) {
        float w = (1.0 - 0.35 * abs(u)) / 5.6;
        accCol += sc * sv * w;
        accCov += sv * w;
        bestT = min(bestT, st);
      }
    }
    if (accCov < 0.015) discard;
    vec3 finalC = accCol / max(accCov, 1e-4);
    float covOut = clamp(accCov * 1.25, 0.0, 1.0);
    oCol = vec4(finalC, covOut);
    oMeta = vec4(covOut, covOut / max(bestT, 0.15), 0.0, 1.0);
  } else {
    vec3 sc; float st, sv;
    if (!shadeDropSample(uRo, rd, vCenter, tOcc, pMask, sc, st, sv)) discard;
    oCol = vec4(sc, sv);
    oMeta = vec4(sv, sv / max(st, 0.15), 0.0, 1.0);
  }
}
"""

# Pass 3: Camera Optical Finish (Person-Locked Thin-Lens Creamy Bokeh + OLPF Softness + Halation + Plate Sensor Grain)
POST_FS = """#version 330
uniform sampler2D uScene;    /* bottom-up final composite */
uniform sampler2D uMeta;     /* bottom-up R=cgAlpha, G=cgInvDepth, B=occInvDepth */
uniform vec2 uRes;
uniform float uFocusDist, uBokehMax, uBlackFloor, uGrainAmt, uFrame;

out vec4 oCol;

float hash12(vec2 p) {
  vec3 p3 = fract(vec3(p.xyx) * 0.1031);
  p3 += dot(p3, p3.yzx + 33.33);
  return fract((p3.x + p3.y) * p3.z);
}

void main() {
  vec2 uv = gl_FragCoord.xy / uRes;
  vec2 tex = 1.0 / uRes;
  vec3 c0 = texture(uScene, uv).rgb;
  vec4 m0 = texture(uMeta, uv);

  float cgA = clamp(m0.r, 0.0, 1.0);
  float cgInvD = (cgA > 0.02) ? (m0.g / max(cgA, 0.02)) : 0.0;
  float occInvD = m0.b;
  float focusInvD = 1.0 / max(uFocusDist, 0.5);

  /* Gather neighbor max CoC so HUGE creamy near-lens foreground droplets bloom smoothly over background (AESPA Dirty Work!) */
  float maxNear = cgInvD;
  float maxNearA = cgA;
  const float PHI = 2.3999632;
  for (int i = 0; i < 12; i++) {
    float r = sqrt((float(i) + 0.5) / 12.0) * uBokehMax * 0.85;
    float a = float(i) * PHI;
    vec2 off = vec2(cos(a), sin(a)) * r * tex;
    vec4 mn = texture(uMeta, clamp(uv + off, vec2(0.002), vec2(0.998)));
    if (mn.r > 0.05) {
      float invN = mn.g / max(mn.r, 0.05);
      if (invN > maxNear && invN > focusInvD * 1.45) {
        maxNear = invN;
        maxNearA = max(maxNearA, mn.r);
      }
    }
  }

  /* Thin-lens Circle of Confusion (near-lens foreground drops get BIG creamy circular bokeh; person plane stays sharp!) */
  float invUse = (maxNear > focusInvD * 1.45) ? maxNear : cgInvD;
  float cocNear = max(0.0, invUse - focusInvD) * 14.0;      /* dramatic macro foreground bokeh! */
  float cocFar  = max(0.0, focusInvD - cgInvD) * 5.5;       /* gentle soft background bokeh */
  float coc = clamp((invUse > focusInvD) ? cocNear : cocFar, 0.0, uBokehMax);
  float cgPresence = max(cgA, maxNearA * smoothstep(1.5, 6.0, cocNear));

  vec3 res = c0;
  if (cgPresence > 0.005) {
    /* 48-tap jittered Vogel disk circular optical bokeh — 100% silky creamy lens bokeh with zero ringing! */
    if (coc > 0.85) {
      float jAng = hash12(gl_FragCoord.xy + vec2(uFrame * 3.7, uFrame * 5.3)) * 6.283185;
      vec3 acc = c0 * 0.35;
      float wsum = 0.35;
      for (int i = 0; i < 48; i++) {
        float rr = sqrt((float(i) + 0.5) / 48.0) * coc;
        float aa = float(i) * PHI + jAng;
        vec2 su = clamp(uv + vec2(cos(aa), sin(aa)) * rr * tex, vec2(0.002), vec2(0.998));
        vec3 sc = texture(uScene, su).rgb;
        vec4 sm = texture(uMeta, su);
        float w = (sm.r > 0.04 || maxNear > focusInvD * 1.45) ? 1.0 : 0.35;
        float hi = smoothstep(0.70, 1.0, max(sc.r, max(sc.g, sc.b)));
        w *= (1.0 + hi * 1.35);
        acc += sc * w;
        wsum += w;
      }
      vec3 bok = acc / max(wsum, 1e-4);
      res = mix(c0, bok, clamp(cgPresence * smoothstep(0.8, 3.2, coc), 0.0, 0.96));
    } else {
      /* Subtle OLPF lens softness on sharp midground droplets so edges match real lens resolution */
      vec3 soft = (
        c0 * 4.0 +
        texture(uScene, uv + vec2(tex.x, 0.0)).rgb +
        texture(uScene, uv - vec2(tex.x, 0.0)).rgb +
        texture(uScene, uv + vec2(0.0, tex.y)).rgb +
        texture(uScene, uv - vec2(0.0, tex.y)).rgb
      ) / 8.0;
      res = mix(c0, soft, cgA * 0.25);
    }

    /* Subtle warm optical halation around blazing droplet specular pinpoints */
    vec3 hal = vec3(0.0);
    for (int k = 0; k < 8; k++) {
      float a = float(k) * 0.785398;
      vec3 s = texture(uScene, clamp(uv + vec2(cos(a), sin(a)) * 3.0 * tex, vec2(0.002), vec2(0.998))).rgb;
      hal += s * smoothstep(0.84, 1.0, max(s.r, max(s.g, s.b)));
    }
    res += (hal / 8.0) * (0.20 * cgPresence);

    /* Match real camera black floor & animated sensor grain ONLY on CG pixels */
    vec3 graded = max(res, vec3(uBlackFloor));
    vec2 p1 = gl_FragCoord.xy + vec2(uFrame * 17.3, uFrame * 31.7);
    vec2 p2 = floor(gl_FragCoord.xy * 0.5) + vec2(uFrame * 11.1, uFrame * 23.9);
    float g = (hash12(p1) + hash12(p1 + 7.13) - 1.0) * 0.62 + (hash12(p2) + hash12(p2 + 5.71) - 1.0) * 0.38;
    float lum = dot(graded, vec3(0.299, 0.587, 0.114));
    graded += vec3(g) * (uGrainAmt * (1.15 - 0.65 * smoothstep(0.55, 0.98, lum)));
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
    # Pure clean stacked compare with ZERO text overlays
    run([
        "-framerate", r, "-i", str(work(c, "comp", "%04d.png")),
        "-framerate", r, "-i", str(work(c, "frames", "%04d.jpg")),
        "-filter_complex",
        "[0]scale=1080:608[a];[1]scale=1080:608[b];[a][b]vstack,pad=1080:1350:0:67:color=black",
    ] + x264 + tags + [str(out / f"{nm}_Compare_1080x1350.mp4")])
    print("encoded ->", out)


def build_rain_drops(cam_mean, R_mean, f_px, W, H, cyl_occ, cfg):
    """Spawn ~680 liquid-glass raindrops across 4 intentional Z-depth layers in camera frustum:
      - Layer 0 (n_near=24): Hero macro near-lens foreground drops (0.48..1.35m) -> huge creamy bokeh!
      - Layer 1 (n_hero=116): Crisp hero drops in front of person (1.7..3.8m) -> razor inverted lensing!
      - Layer 2 (n_mid=240): Dense curtain wrapping around & behind person (3.7..5.6m) -> MediaPipe occlusion!
      - Layer 3 (n_micro=300): Fine songgol-songgol background & atmospheric beads (1.5..8.4m)
    All drops fall smoothly in world -Z during Act 1, then brake abruptly at freeze_frame!"""
    rng = np.random.default_rng(cfg.get("seed", 2033))
    drops = []
    cx_occ, cy_occ, r_occ, _ = cyl_occ

    layers = [
        ("near",  cfg.get("n_near", 24),   (0.52, 1.38), (0.011, 0.025), (5.2, 7.6), 0.96),
        ("hero",  cfg.get("n_hero", 116),  (1.75, 3.85), (0.016, 0.044), (5.8, 8.6), 0.92),
        ("mid",   cfg.get("n_mid", 240),   (3.75, 5.75), (0.014, 0.040), (5.4, 8.2), 0.94),
        ("micro", cfg.get("n_micro", 300), (1.55, 8.20), (0.006, 0.018), (4.6, 7.4), 0.98),
    ]

    for lname, count, z_rng, r_rng, v_rng, span_scale in layers:
        created = 0
        tries = 0
        while created < count and tries < count * 12:
            tries += 1
            zc = float(rng.uniform(z_rng[0], z_rng[1]))
            # Place inside camera frustum with generous framing margin
            if lname == "near":
                # Keep huge foreground bokeh drops mostly framing the outer/mid zones (like AESPA Dirty Work!)
                ux = float(rng.uniform(-0.46, 0.46))
                uy = float(rng.uniform(-0.42, 0.42))
                if abs(ux) < 0.16 and abs(uy) < 0.16:
                    continue
            else:
                ux = float(rng.uniform(-0.50 * span_scale, 0.50 * span_scale))
                uy = float(rng.uniform(-0.48 * span_scale, 0.48 * span_scale))

            # Convert camera-local ray at zc to world coordinates (base frozen target position!)
            q_cam = np.array([ux * W / f_px * zc, uy * H / f_px * zc, zc], dtype=np.float64)
            p_freeze = cam_mean + R_mean @ q_cam
            # Keep safely above floor (z >= 0.08) and inside ceiling (z <= 3.6)
            if p_freeze[2] < 0.09 or p_freeze[2] > 3.55:
                continue
            # Avoid spawning directly inside the solid concrete pillar cylinder
            if r_occ > 0.01 and math.hypot(p_freeze[0] - cx_occ, p_freeze[1] - cy_occ) < r_occ + 0.08:
                continue

            rad = float(rng.uniform(r_rng[0], r_rng[1]))
            # Power-law size variety so drops feel naturally non-uniform ("비정형적으로 랜덤한 크기")
            if lname in ("hero", "mid"):
                rad = float(r_rng[0] + (r_rng[1] - r_rng[0]) * (rng.random() ** 1.65))
            vz = float(rng.uniform(v_rng[0], v_rng[1]))
            # Slight wind drift angle for organic rain feel
            vx = float(rng.uniform(-0.35, -0.05))
            vy = float(rng.uniform(-0.12, 0.12))
            drops.append({
                "layer": lname,
                "p_freeze": p_freeze,
                "rad": rad,
                "vel": np.array([vx, vy, -vz], dtype=np.float64),
                "seed": float(rng.random()),
                "osc_freq": float(rng.uniform(14.0, 26.0)),
                "drift_amp": float(rng.uniform(0.008, 0.022)),
                "z_wrap_lo": 0.04,
                "z_wrap_hi": max(3.65, float(p_freeze[2] + 1.8)),
            })
            created += 1

    # Also generate 24 floor splash ring slots on z=0 inside visible platform area
    splashes = []
    for _ in range(24):
        zc = float(rng.uniform(2.2, 7.8))
        ux = float(rng.uniform(-0.44, 0.44))
        q_cam = np.array([ux * W / f_px * zc, 0.25 * H / f_px * zc, zc], dtype=np.float64)
        pw = cam_mean + R_mean @ q_cam
        splashes.append({
            "xy": (float(pw[0]), float(pw[1])),
            "rate": float(rng.uniform(1.4, 2.8)),
            "phase0": float(rng.random()),
            "rad": float(rng.uniform(0.08, 0.22)),
        })
    return drops, splashes


def integrated_rain_time(t_sec, t_freeze, brake_dur):
    """Exact smooth analytical integral of rain fall speed v(t):
      - For t <= t_freeze: v(t) = 1.0 (full speed pouring rain!)
      - For t in [t_freeze, t_freeze + brake_dur]: v(t) slams abruptly from 1.0 -> 0.006 ('훅!' freeze snap!)
      - For t > t_freeze + brake_dur: v(t) = 0.006 (nearly dead-stopped weightless hover!)
    Returns (s_rel_to_freeze, speed_norm) where s_rel_to_freeze == 0 right when frozen!"""
    v_min = 0.006
    t_end = t_freeze + brake_dur
    # Compute total distance covered during brake window [t_freeze, t_end]
    # Using cubic smoothstep brake: v(u) = v_min + (1 - v_min) * (1 - u)^3  -> integral_0^1 = v_min + 0.25*(1-v_min)
    d_brake_total = brake_dur * (v_min + 0.25 * (1.0 - v_min))
    if t_sec <= t_freeze:
        # Negative time before freeze (so p = p_freeze + vel * s arrives cleanly at p_freeze right at t_end!)
        s = (t_sec - t_freeze) - d_brake_total
        return s, 1.0
    elif t_sec < t_end:
        u = (t_sec - t_freeze) / max(brake_dur, 1e-4)
        # Integral of (1 - u)^3 from 0 to u is 0.25 * (1 - (1 - u)^4)
        d_done = brake_dur * (v_min * u + 0.25 * (1.0 - v_min) * (1.0 - (1.0 - u) ** 4))
        s = d_done - d_brake_total
        spd = v_min + (1.0 - v_min) * ((1.0 - u) ** 3)
        return s, spd
    else:
        dt_post = t_sec - t_end
        s = dt_post * v_min
        return s, v_min


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

    c = load(piece_arg or (here / "piece_rain"))
    s = scene(c)
    C = np.array(s["cam"], dtype=np.float64)
    R_all = np.array(s["rot"], dtype=np.float64)
    f_px = float(s["f"])
    sun = np.array(s["sun"], dtype=np.float64)
    sun /= np.linalg.norm(sun) + 1e-9

    limbs = np.load(work(c, "limbs.npz"))
    person_depths = np.asarray(limbs["depth"], dtype=np.float64)

    N = len(C)
    fps = float(c.get("fps_num", 25)) / float(c.get("fps_den", 1))
    pl0 = frame(c, 0)
    H, W = pl0.shape[:2]

    # Measure plate black floor & sensor grain cleanly for Pass-3 camera optical match
    gray0 = cv2.cvtColor(pl0, cv2.COLOR_BGR2GRAY).astype(np.float32) / 255.0
    black_floor = float(np.clip(np.percentile(gray0, 1.0) * 0.85, 0.008, 0.055))
    hp0 = gray0 - cv2.GaussianBlur(gray0, (0, 0), 1.2)
    grain_amt = float(np.clip(float(np.median(np.abs(hp0 - np.median(hp0)))) * 1.4826 * 1.15, 0.008, 0.028))

    cfg = c.get("rain_freeze", {})
    freeze_f = int(cfg.get("freeze_frame", 64))
    brake_f = int(cfg.get("brake_frames", 7))
    t_freeze = freeze_f / fps
    brake_dur = brake_f / fps
    cyl_occ = tuple(float(x) for x in cfg.get("cyl_occ", [-1.726, 0.643, 0.513, 4.8]))

    # Build raindrops around the freeze-moment camera frustum so frozen composition is balanced & cinematic!
    drops, splashes = build_rain_drops(C[freeze_f], R_all[freeze_f], f_px, W, H, cyl_occ, cfg)
    print(f"Liquid-Glass Rain Freeze: {len(drops)} 3D droplets | freeze at f={freeze_f} ({t_freeze:.2f}s) over {brake_f}f")

    ctx = moderngl.create_standalone_context(require=330)
    fsq = ctx.buffer(np.array([-1, -1, 1, -1, -1, 1, -1, 1, 1, -1, 1, 1], dtype=np.float32).tobytes())
    q_inst = ctx.buffer(np.array([-1, -1, 1, -1, -1, 1, -1, 1, 1, -1, 1, 1], dtype=np.float32).tobytes())
    inst_buf = ctx.buffer(reserve=len(drops) * 12 * 4)

    base_prog = ctx.program(vertex_shader=VS_FSQ, fragment_shader=BASE_FS)
    copy2_prog = ctx.program(vertex_shader=VS_FSQ, fragment_shader=COPY2_FS)
    drop_prog = ctx.program(vertex_shader=DROP_VS, fragment_shader=DROP_FS)
    post_prog = ctx.program(vertex_shader=VS_FSQ, fragment_shader=POST_FS)

    va_base = ctx.vertex_array(base_prog, [(fsq, "2f", "p")])
    va_copy2 = ctx.vertex_array(copy2_prog, [(fsq, "2f", "p")])
    va_drop = ctx.vertex_array(drop_prog, [
        (q_inst, "2f", "inQuad"),
        (inst_buf, "4f 4f 4f/i", "inPosRad", "inShapeVel", "inExtra"),
    ])
    va_post = ctx.vertex_array(post_prog, [(fsq, "2f", "p")])

    plate_t = ctx.texture((W, H), 3)
    plate_t.filter = (moderngl.LINEAR, moderngl.LINEAR)
    plate_t.repeat_x = plate_t.repeat_y = False

    mask_t = ctx.texture((W, H), 1, dtype="f4")
    mask_t.filter = (moderngl.LINEAR, moderngl.LINEAR)
    mask_t.repeat_x = mask_t.repeat_y = False

    t_base = ctx.texture((W, H), 4, dtype="f4")
    t_depth = ctx.texture((W, H), 4, dtype="f4")
    t_meta0 = ctx.texture((W, H), 4, dtype="f4")
    t_base.filter = (moderngl.LINEAR, moderngl.LINEAR)
    t_base.repeat_x = t_base.repeat_y = False
    fbo_base = ctx.framebuffer([t_base, t_depth, t_meta0])

    t_scene = ctx.texture((W, H), 4, dtype="f4")
    t_meta = ctx.texture((W, H), 4, dtype="f4")
    t_scene.filter = (moderngl.LINEAR, moderngl.LINEAR)
    t_scene.repeat_x = t_scene.repeat_y = False
    t_meta.filter = (moderngl.LINEAR, moderngl.LINEAR)
    t_meta.repeat_x = t_meta.repeat_y = False
    fbo_scene = ctx.framebuffer([t_scene, t_meta])

    t_final = ctx.texture((W, H), 4, dtype="f4")
    fbo_final = ctx.framebuffer([t_final])

    # Static uniforms
    base_prog["uRes"].value = (float(W), float(H))
    base_prog["uF"].value = f_px
    base_prog["uCylOcc"].value = cyl_occ
    copy2_prog["uRes"].value = (float(W), float(H))
    drop_prog["uRes"].value = (float(W), float(H))
    drop_prog["uF"].value = f_px
    drop_prog["uIor"].value = float(cfg.get("ior", 1.333))
    drop_prog["uDisp"].value = float(cfg.get("dispersion", 0.009))
    drop_prog["uProxyDist"].value = 5.2
    if "uSun" in drop_prog:
        drop_prog["uSun"].value = tuple(map(float, sun))
    post_prog["uRes"].value = (float(W), float(H))
    post_prog["uBokehMax"].value = 18.0
    post_prog["uBlackFloor"].value = black_floor
    post_prog["uGrainAmt"].value = grain_amt

    def render_one(i):
        pl = frame(c, i)
        m_path = work(c, "mask", f"{i:04d}.png")
        m_raw = cv2.imread(str(m_path), cv2.IMREAD_GRAYSCALE)
        if m_raw is None:
            m_f = np.zeros((H, W), dtype=np.float32)
        else:
            if m_raw.shape[:2] != (H, W):
                m_raw = cv2.resize(m_raw, (W, H), interpolation=cv2.INTER_LINEAR)
            # Soft clean edge feather on MediaPipe mask
            m_f = cv2.GaussianBlur(m_raw.astype(np.float32) / 255.0, (0, 0), 1.0)
        ro = C[i]
        R = R_all[i]
        Rt = tuple(R.T.flatten().astype(np.float32))
        p_depth = float(person_depths[i])
        t_sec = i / fps

        s_rel, spd = integrated_rain_time(t_sec, t_freeze, brake_dur)
        # Rain active mask for floor splash ripples (active while falling, smoothly fades right as freeze hits!)
        rain_active = float(np.clip((spd - 0.05) / 0.95, 0.0, 1.0))
        dt_since_snap = max(0.0, t_sec - t_freeze)

        # Build 24 floor splash rings
        sp_arr = []
        for sp in splashes:
            ph = (sp["phase0"] + t_sec * sp["rate"]) % 1.0
            sp_arr.append((sp["xy"][0], sp["xy"][1], float(ph), sp["rad"]))

        # Update all ~680 liquid-glass drops
        rows = []
        for d in drops:
            p_f = d["p_freeze"]
            vel = d["vel"]
            if s_rel < 0.0:
                # Active falling rain: wrap smoothly in vertical curtain [z_lo, z_hi] so rain pours continuously
                # while guaranteeing exact arrival at p_freeze at s_rel == 0!
                z_lo, z_hi = d["z_wrap_lo"], d["z_wrap_hi"]
                span_z = max(z_hi - z_lo, 2.5)
                # Raw linear position along rain ray
                raw_z = p_f[2] + vel[2] * s_rel
                # Seamless modulo wrap in Z while keeping exact arrival at p_f[2] when s_rel == 0!
                wrapped_z = z_lo + ((raw_z - z_lo) % span_z)
                # Keep XY aligned along the tilted rain line for whichever wrapped drop is currently passing
                dt_local = (wrapped_z - p_f[2]) / vel[2]
                pos = np.array([p_f[0] + vel[0] * dt_local, p_f[1] + vel[1] * dt_local, wrapped_z], dtype=np.float64)
            else:
                # Frozen weightless suspension! Ultra-gentle zero-G hover drift around exact frozen coordinate!
                drift_x = math.sin(t_sec * 0.85 + d["seed"] * 19.0) * d["drift_amp"] * 0.45
                drift_y = math.cos(t_sec * 0.72 + d["seed"] * 27.0) * d["drift_amp"] * 0.45
                drift_z = math.sin(t_sec * 0.95 + d["seed"] * 31.0) * d["drift_amp"] * 0.35 + vel[2] * s_rel
                pos = p_f + np.array([drift_x, drift_y, drift_z], dtype=np.float64)

            # Aerodynamic elongation while pouring -> damped surface-tension spring oscillation right on '훅!' snap!
            snap_osc = 0.0
            if t_sec >= t_freeze:
                # Damped liquid jelly oscillation right after braking to a halt!
                snap_osc = 0.24 * math.exp(-dt_since_snap * 3.8) * math.cos(dt_since_snap * d["osc_freq"] + d["seed"] * 6.28)
            stretch_z = 1.0 + 2.15 * (spd ** 0.85) + snap_osc
            taper = float(np.clip(spd * 1.15, 0.0, 1.0))
            # Subtle organic jelly wobble when frozen, crisp teardrop while falling
            wobble_amp = (0.045 + 0.14 * abs(snap_osc)) * (1.0 - 0.65 * spd)
            # Continuous smooth shutter vertical motion smear during fast pouring rain -> 0 when frozen!
            blur_m = float(abs(vel[2]) * spd * (0.36 / fps))
            if blur_m < 0.008:
                blur_m = 0.0

            z_cam = float(((pos - ro) @ R)[2])
            if z_cam > 0.12:
                rows.append((z_cam, [
                    float(pos[0]), float(pos[1]), float(pos[2]), float(d["rad"]),
                    float(stretch_z), float(wobble_amp), float(blur_m), float(d["seed"]),
                    1.0, float(taper), float(t_sec * 2.2 + d["seed"] * 6.28), 0.0,
                ]))

        # Pass 1: Plate + Pillar Holdout Depth + Floor Rain Splash Ripples
        plate_t.write(pl[:, :, ::-1].tobytes())
        plate_t.use(0)
        base_prog["uPlate"].value = 0
        base_prog["uRo"].value = tuple(map(float, ro))
        base_prog["uCam"].value = Rt
        base_prog["uRainActive"].value = rain_active
        if "uTime" in base_prog:
            base_prog["uTime"].value = float(t_sec)
        base_prog["uSplashes"].value = sp_arr[:24]
        base_prog["uNumSplashes"].value = len(sp_arr[:24])
        fbo_base.use()
        ctx.disable(moderngl.BLEND)
        va_base.render(moderngl.TRIANGLES)

        fbo_scene.use()
        t_base.use(0); t_meta0.use(1)
        copy2_prog["uSrc0"].value = 0; copy2_prog["uSrc1"].value = 1
        va_copy2.render(moderngl.TRIANGLES)

        # Pass 2: Back-to-Front Ray-Traced Liquid-Glass Raindrops!
        rows.sort(key=lambda x: -x[0])
        if rows:
            mask_t.write(m_f.astype(np.float32).tobytes())
            data = np.array([r[1] for r in rows], dtype=np.float32)
            inst_buf.write(data.tobytes())
            t_base.use(0); t_depth.use(1); mask_t.use(2)
            drop_prog["uBase"].value = 0
            drop_prog["uDepth"].value = 1
            drop_prog["uPersonMask"].value = 2
            drop_prog["uRo"].value = tuple(map(float, ro))
            drop_prog["uCam"].value = Rt
            drop_prog["uPersonDepth"].value = p_depth
            ctx.enable(moderngl.BLEND)
            ctx.blend_func = (moderngl.SRC_ALPHA, moderngl.ONE_MINUS_SRC_ALPHA)
            va_drop.render(moderngl.TRIANGLES, instances=len(rows))
            ctx.disable(moderngl.BLEND)

        # Pass 3: Camera Optical Finish (Person-Locked Thin-Lens Creamy Bokeh + Halation + Plate Sensor Grain)
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
        rows_im = []
        hero = None
        for idx in stills:
            pl, comp = render_one(idx)
            if idx == stills[len(stills) // 2]:
                hero = comp
            t_im = lambda im, lb: label(cv2.resize(im, (960, 540)), lb)
            rows_im.append(np.hstack([t_im(pl, f"plate {idx}"), t_im(comp, f"rain freeze {idx}")]))
        out_p = work(c, "still_rain_sheet.jpg")
        cv2.imwrite(str(out_p), np.vstack(rows_im), [cv2.IMWRITE_JPEG_QUALITY, 92])
        if hero is not None:
            cv2.imwrite(str(work(c, "still_rain_hero.jpg")), hero, [cv2.IMWRITE_JPEG_QUALITY, 94])
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
            print(f"rain_freeze frame {i}/{N} ({time.time() - ts:.1f}s)", flush=True)
    encode(c, out_dir=out_dir)


if __name__ == "__main__":
    main()
