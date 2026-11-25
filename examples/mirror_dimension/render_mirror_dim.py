"""#4 Mirror Dimension Prism Fold — 3D Folding Crystal Mirror Slabs & Faceted Polyhedra (ModernGL, OpenGL 3.3).

    python render_mirror_dim.py [piece_dir] --still 15,85,155,235   quick look -> work/still_mirror_sheet.jpg
    python render_mirror_dim.py [piece_dir] --out out               full render -> out/*_Composite.mp4, *_Compare_1080x1350.mp4

Zero text/HUD overlays — 100% pure architectural kaleidoscope VFX:
- 76 world-locked 3D faceted crystal mirror-prism slabs, hexagonal columns & diamond polyhedra peeling and folding
  rhythmically along the 13.4m sunlit hall + 360 tumbling beveled micro-prism splinters.
- Every facet performs true 3D multi-face mirror reflection (reflecting tilted kaleidoscopic slices of the real house!)
  blended with flint-crystal Snell refraction (ior=1.54) and crisp beveled chamfer edge speculars.
- Exact 6-box 3D architectural wall/pillar holdout occlusion (left/right entry walls + 4 white structural pillars)
  + geometric prism floor caustics + Pass-3 thin-lens cinema bokeh, halation & animated sensor grain.
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

# Pass 1: 6-Box 3D Architectural Occlusion Depth + Geometric Prism Floor Caustics on z=0 oak planks
BASE_FS = """#version 330
uniform sampler2D uPlate;
uniform vec2 uRes;
uniform float uF, uTime;
uniform vec3 uRo, uSun;
uniform mat3 uCam;
uniform vec3 uOccMin[6];
uniform vec3 uOccMax[6];
uniform int uNumOcc;
uniform vec4 uCaustPrisms[32];  /* xyz = prism world pos, w = effective radius */
uniform int uNumCaust;

layout(location=0) out vec4 oComp;
layout(location=1) out vec4 oDepth;   /* R: tOcc, G: tOcc */
layout(location=2) out vec4 oMeta;    /* R: 0.0, G: 0.0, B: 1/tOcc, A: 1.0 */

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
    if (tFar > max(tNear, 0.05)) tBest = min(tBest, (tNear > 0.05) ? tNear : 0.05);
  }
  return tBest;
}

bool insideOccXY(vec2 xy) {
  for (int i = 0; i < 6; i++) {
    if (i >= uNumOcc) break;
    if (xy.x >= uOccMin[i].x && xy.x <= uOccMax[i].x && xy.y >= uOccMin[i].y && xy.y <= uOccMax[i].y) return true;
  }
  return false;
}

void main() {
  vec2 px0 = vec2(gl_FragCoord.x, uRes.y - gl_FragCoord.y);
  vec2 uv0 = px0 / uRes;
  vec3 plateCol = texture(uPlate, clamp(uv0, vec2(0.002), vec2(0.998))).rgb;
  vec3 rd = normalize(uCam * vec3((px0 - 0.5 * uRes) / uF, 1.0));
  float tOcc = hitOccluders(uRo, rd);

  vec3 col = plateCol;
  float tFloor = (rd.z < -0.004) ? (-uRo.z / rd.z) : 1e4;
  if (tFloor > 0.08 && tFloor < min(tOcc, 22.0)) {
    vec3 Xf = uRo + rd * tFloor;
    if (!insideOccXY(Xf.xy)) {
      vec3 sun = normalize(uSun);
      float invSz = 1.0 / max(sun.z, 0.22);
      vec3 caustRGB = vec3(0.0);
      float shad = 0.0;
      for (int i = 0; i < 32; i++) {
        if (i >= uNumCaust) break;
        vec4 pr = uCaustPrisms[i];
        if (pr.w <= 0.02) continue;
        vec2 sp = pr.xy - sun.xy * (pr.z * invSz);
        vec2 dv = Xf.xy - sp;
        float r = length(dv);
        float R = pr.w * 1.45;
        if (r > R * 2.6) continue;
        /* Crisp warm sunlit angular caustic reflection on oak planks (no magenta tint!) */
        float ang = float(i) * 0.85 + uTime * 0.25;
        float ca = cos(ang), sa = sin(ang);
        vec2 lr = vec2(dv.x * ca - dv.y * sa, dv.x * sa + dv.y * ca);
        float dFac = max(abs(lr.x) * 0.85 + abs(lr.y) * 0.50, abs(lr.y));
        float env = smoothstep(R * 1.25, R * 0.15, dFac);
        float beam = pow(max(0.0, 1.0 - abs(dFac - R * 0.35) / max(R * 0.45, 0.08)), 2.2);
        caustRGB += vec3(1.0, 0.95, 0.84) * (0.28 + 0.85 * beam) * env * 0.45;
        shad += smoothstep(R * 1.35, R * 0.15, dFac) * 0.10;
      }
      float luma = dot(plateCol, vec3(0.299, 0.587, 0.114));
      float fMask = smoothstep(0.08, 0.24, luma) * (1.0 - smoothstep(13.0, 21.0, tFloor));
      col = col * (1.0 - clamp(shad, 0.0, 0.22) * fMask);
      col += clamp(caustRGB, vec3(0.0), vec3(1.1)) * (0.24 * fMask) * (0.45 + 0.55 * col);
    }
  }

  oComp = vec4(clamp(col, 0.0, 1.0), 1.0);
  oDepth = vec4(tOcc, tOcc, 0.0, 1.0);
  oMeta = vec4(0.0, 0.0, 1.0 / max(tOcc, 0.20), 1.0);
}
"""

# Pass 2: Instanced 3D Raymarched Faceted Crystal Mirror-Prisms (436 instances: 76 architectural slabs/polyhedra + 360 micro-shards!)
PRISM_VS = """#version 330
in vec2 inQuad;
in vec4 inPosScale;   /* xyz = world pos (m), w = bound radius (m) */
in vec4 inSizeKind;   /* xyz = half-extents (hx, hy, hz), w = kind (0=beveled mirror slab, 1=hex prism, 2=octahedron diamond) */
in vec4 inRot0;       /* row 0 of local-from-world rotation matrix (xyz), w = mirrorMix (0.35..0.82) */
in vec4 inRot1;       /* row 1 of local-from-world rotation matrix (xyz), w = bevelRadius */
in vec4 inRot2;       /* row 2 of local-from-world rotation matrix (xyz), w = opacity */

uniform vec2 uRes;
uniform float uF;
uniform vec3 uRo;
uniform mat3 uCam;

out vec3 vCenter;
out float vBoundR;
out vec4 vSizeKind;
out mat3 vInvRot;
out float vMirrorMix;
out float vBevel;
out float vAlpha;

void main() {
  vCenter = inPosScale.xyz;
  vBoundR = inPosScale.w;
  vSizeKind = inSizeKind;
  /* rows of world->local matrix */
  vInvRot = transpose(mat3(inRot0.xyz, inRot1.xyz, inRot2.xyz));
  vMirrorMix = inRot0.w;
  vBevel = inRot1.w;
  vAlpha = inRot2.w;

  vec3 q = transpose(uCam) * (vCenter - uRo);
  if (vAlpha <= 0.01 || q.z <= 0.08 + vBoundR * 0.15) {
    gl_Position = vec4(2.0, 2.0, 2.0, 1.0);
    return;
  }
  vec2 ctrPx = uF * q.xy / q.z + 0.5 * uRes;
  float pxR = clamp(uF * vBoundR / max(q.z - vBoundR * 0.85, 0.05) + 4.0, 4.0, 1100.0);
  vec2 px = ctrPx + inQuad * pxR;
  vec2 ndc = vec2(px.x / uRes.x * 2.0 - 1.0, (1.0 - px.y / uRes.y) * 2.0 - 1.0);
  gl_Position = vec4(ndc, 0.0, 1.0);
}
"""

PRISM_FS = """#version 330
in vec3 vCenter;
in float vBoundR;
in vec4 vSizeKind;
in mat3 vInvRot;
in float vMirrorMix;
in float vBevel;
in float vAlpha;

uniform sampler2D uBase;    /* Pass-1 scene color (bottom-up) */
uniform sampler2D uDepth;   /* Pass-1 6-box architectural occluder depth (bottom-up) */
uniform vec2 uRes;
uniform float uF, uIor, uDisp, uProxyDist;
uniform vec3 uRo, uSun;
uniform mat3 uCam;

layout(location=0) out vec4 oCol;
layout(location=1) out vec4 oMeta;

/* 3 sharp architectural crystal prism SDFs in local space */
float sdPrismLocal(vec3 q) {
  vec3 b = vSizeKind.xyz;
  float kind = vSizeKind.w;
  if (kind < 0.5) {
    /* 0: Beveled architectural mirror-prism slab with crisp 45-deg optical chamfer */
    vec3 d = abs(q) - b;
    float box = length(max(d, 0.0)) + min(max(d.x, max(d.y, d.z)), 0.0);
    /* 45-degree chamfer cut along edges */
    float ch = (abs(q.x) - b.x + abs(q.y) - b.y + vBevel * 1.4) * 0.7071;
    float ch2 = (abs(q.x) - b.x + abs(q.z) - b.z + vBevel * 1.4) * 0.7071;
    float ch3 = (abs(q.y) - b.y + abs(q.z) - b.z + vBevel * 1.4) * 0.7071;
    return max(box, max(ch, max(ch2, ch3))) - vBevel * 0.18;
  } else if (kind < 1.5) {
    /* 1: Beveled hexagonal architectural crystal prism */
    const vec3 k = vec3(-0.8660254, 0.5, 0.57735);
    vec3 p = abs(q);
    p.xy -= 2.0 * min(dot(k.xy, p.xy), 0.0) * k.xy;
    vec2 d = vec2(
      length(p.xy - vec2(clamp(p.x, -k.z * b.x, k.z * b.x), b.x)) * sign(p.y - b.x),
      p.z - b.z
    );
    return min(max(d.x, d.y), 0.0) + length(max(d, 0.0)) - vBevel * 0.35;
  } else {
    /* 2: Razor-faceted diamond octahedron / asymmetric crystal shard */
    vec3 p = abs(q / max(b, vec3(0.005)));
    float m = p.x + p.y + p.z - 1.0;
    return m * min(b.x, min(b.y, b.z)) * 0.57735 - vBevel * 0.25;
  }
}

float sdPrism(vec3 p) {
  return sdPrismLocal(vInvRot * (p - vCenter));
}

vec3 prismNormal(vec3 p, out float edgeCurv) {
  float h = max(vBoundR * 0.006, 0.0008);
  vec2 e = vec2(h, -h);
  float d1 = sdPrism(p + e.xyy), d2 = sdPrism(p + e.yyx), d3 = sdPrism(p + e.yxy), d4 = sdPrism(p + e.xxx);
  vec3 n = normalize(e.xyy * d1 + e.yyx * d2 + e.yxy * d3 + e.xxx * d4);
  /* Measure razor bevel edge crease by comparing normals slightly offset along surface */
  float h2 = h * 4.5;
  vec3 n2 = normalize(
    vec3(h2,-h2,-h2)*sdPrism(p+vec3(h2,-h2,-h2)) + vec3(-h2,-h2,h2)*sdPrism(p+vec3(-h2,-h2,h2)) +
    vec3(-h2,h2,-h2)*sdPrism(p+vec3(-h2,h2,-h2)) + vec3(h2,h2,h2)*sdPrism(p+vec3(h2,h2,h2))
  );
  edgeCurv = clamp(length(n - n2) * 2.8, 0.0, 1.0);
  return n;
}

vec3 sampleEnv(vec3 ro, vec3 rd) {
  float tf = (rd.z < -0.012) ? (-ro.z / rd.z) : 1e4;
  if (tf < 0.05 || tf > 24.0) tf = 1e4;
  vec3 X = ro + rd * min(tf, uProxyDist);
  vec3 q = transpose(uCam) * (X - uRo);
  /* Smooth rear-hemisphere architectural interior fallback (eliminates zebra wrap artifacts on grazing mirror angles!) */
  float fwdVis = smoothstep(0.05, 0.75, q.z);
  vec2 uv = (uF * q.xy / max(q.z, 0.35) + 0.5 * uRes) / uRes;
  /* Soft single-fold mirror border instead of multi-wrap fract */
  uv = 1.0 - abs(1.0 - clamp(uv, vec2(-0.85), vec2(1.85)));
  uv = clamp(uv, vec2(0.010), vec2(0.990));
  vec3 plateS = texture(uBase, vec2(uv.x, 1.0 - uv.y)).rgb;
  /* Realistic sunlit interior bounce gradient for rays looking behind camera */
  vec3 rearRoom = mix(vec3(0.46, 0.37, 0.28), vec3(0.84, 0.86, 0.90), smoothstep(-0.35, 0.55, rd.z));
  return mix(rearRoom, plateS, fwdVis);
}

void main() {
  vec2 px0 = vec2(gl_FragCoord.x, uRes.y - gl_FragCoord.y);
  float tOcc = texture(uDepth, gl_FragCoord.xy / uRes).r;

  vec3 acc = vec3(0.0);
  float cov = 0.0;
  float minT = 1e4;
  vec2 offs[4] = vec2[4](vec2(-0.25, -0.25), vec2(0.25, -0.25), vec2(-0.25, 0.25), vec2(0.25, 0.25));
  vec3 L_sun = normalize(uSun);
  vec3 L_ceil = normalize(vec3(0.04, 0.14, 0.98));

  for (int k = 0; k < 4; k++) {
    vec2 px = px0 + offs[k];
    vec3 rd = normalize(uCam * vec3((px - 0.5 * uRes) / uF, 1.0));
    vec3 oc = vCenter - uRo;
    float b = dot(oc, rd);
    float disc = b * b - (dot(oc, oc) - vBoundR * vBoundR);
    if (disc < 0.0) continue;
    float sq = sqrt(disc);
    float tStart = max(b - sq, 0.04);
    float tEnd = min(b + sq, tOcc);
    if (tStart >= tEnd) continue;

    float t = tStart;
    float tHit = -1.0;
    float eps = max(vBoundR * 0.0025, 0.0004);
    for (int s = 0; s < 42; s++) {
      float d = sdPrism(uRo + rd * t);
      if (d < eps) { tHit = t; break; }
      t += max(d * 0.90, 0.0012);
      if (t > tEnd) break;
    }
    if (tHit < 0.0 || tHit >= tOcc) continue;
    minT = min(minT, tHit);

    vec3 p1 = uRo + rd * tHit;
    float edgeC;
    vec3 n1 = prismNormal(p1, edgeC);

    /* Dual-interface flint crystal refraction (ior=1.54) */
    vec3 rdIn = refract(rd, n1, 1.0 / uIor);
    if (length(rdIn) < 0.001) rdIn = reflect(rd, n1);
    float thick = vBoundR * 0.18;
    vec3 p2 = p1 + rdIn * max(vBoundR * 0.08, 0.004);
    for (int s = 0; s < 16; s++) {
      float dIn = -sdPrism(p2);
      if (dIn < eps * 2.0) break;
      float st = max(dIn * 0.88, vBoundR * 0.05);
      p2 += rdIn * st;
      thick += st;
      if (thick > vBoundR * 2.6) break;
    }
    float dummyE;
    vec3 n2 = -prismNormal(p2, dummyE);
    vec3 rdOutG = refract(rdIn, n2, uIor);
    if (length(rdOutG) < 0.001) rdOutG = reflect(rdIn, n2);
    vec3 bend = rdOutG - rd;
    vec3 rdOutR = normalize(rd + bend * ((uIor - uDisp) - 1.0) / max(uIor - 1.0, 0.05));
    vec3 rdOutB = normalize(rd + bend * ((uIor + uDisp) - 1.0) / max(uIor - 1.0, 0.05));

    vec3 refrCol = vec3(sampleEnv(p2, rdOutR).r, sampleEnv(p2, rdOutG).g, sampleEnv(p2, rdOutB).b);
    refrCol *= exp(-thick * vec3(0.055, 0.025, 0.018)) * 1.04;

    /* Crisp Mirror-Dimension architectural kaleidoscope reflection! */
    vec3 reflDir = reflect(rd, n1);
    vec3 mirCol = sampleEnv(p1 + n1 * 0.005, reflDir);
    float skyUp = smoothstep(-0.1, 0.85, reflDir.z);
    mirCol = mix(mirCol * 1.02, mirCol * 1.15 + vec3(0.16, 0.18, 0.21), skyUp * 0.44);

    float ndv = max(dot(-rd, n1), 0.0);
    float fresnel = 0.05 + 0.95 * pow(1.0 - ndv, 3.6);
    /* Mirror slabs reflect strongly like architectural kaleidoscope glass; octahedra blend mirror + prism refraction */
    float mFactor = clamp(mix(vMirrorMix * 0.55, 0.92, fresnel), 0.15, 0.92);
    vec3 col = mix(refrCol, mirCol, mFactor);

    /* Crisp sunlight speculars + blazing beveled chamfer crystal edge fire! */
    float spSun  = pow(max(dot(n1, normalize(L_sun  - rd)), 0.0), 190.0) * 1.9 + pow(max(dot(n1, normalize(L_sun  - rd)), 0.0), 26.0) * 0.14;
    float spCeil = pow(max(dot(n1, normalize(L_ceil - rd)), 0.0), 130.0) * 0.65;
    float rim = pow(1.0 - ndv, 3.0) * 0.30;
    /* Subtle prismatic fire right along the 45-degree beveled crystal seams! */
    vec3 edgeFire = mix(vec3(0.92, 0.97, 1.0), vec3(1.0, 0.94, 0.84), 0.5 + 0.5 * dot(n1, L_sun));
    col += vec3(1.0, 0.97, 0.90) * spSun + vec3(0.95, 0.98, 1.0) * (spCeil + rim);
    col += edgeFire * (edgeC * (0.34 + 0.45 * fresnel));

    acc += col;
    cov += 1.0;
  }

  if (cov < 0.5) discard;
  float a = clamp(cov * 0.25 * vAlpha, 0.0, 1.0);
  oCol = vec4(clamp(acc / cov, 0.0, 1.0), a);
  oMeta = vec4(1.0, 1.0 / max(minT, 0.20), 1.0 / max(tOcc, 0.20), a);
}
"""

# Pass 3: Camera Optical Finish (Depth-Aware Thin-Lens Bokeh + OLPF Softness + Halation + Edge CA + Sensor Grain)
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
  return clamp(uBaseSoft + uNearBlur * max(0.0, invD - invF) + uFarBlur * max(0.0, invF - invD), uBaseSoft, uMaxCoc);
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
  res += halation * vec3(1.02, 0.98, 0.93) * 0.055;

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


def euler_to_mat(rx, ry, rz):
    cx, sx = math.cos(rx), math.sin(rx)
    cy, sy = math.cos(ry), math.sin(ry)
    cz, sz = math.cos(rz), math.sin(rz)
    Rx = np.array([[1, 0, 0], [0, cx, -sx], [0, sx, cx]])
    Ry = np.array([[cy, 0, sy], [0, 1, 0], [-sy, 0, cy]])
    Rz = np.array([[cz, -sz, 0], [sz, cz, 0], [0, 0, 1]])
    return Rz @ Ry @ Rx


def build_mirror_dimension(cam_path, occ_boxes, n_slabs=42, n_poly=34, n_micro=360, seed=2030):
    """Build interlocking Doctor-Strange Mirror Dimension folding crystal slabs, faceted polyhedra & tumbling micro-prisms."""
    rng = np.random.default_rng(seed)
    items = []

    def clears_cam_and_walls(pos, bound_r):
        if np.linalg.norm(cam_path - pos[None, :], axis=1).min() < (0.38 + bound_r * 1.15):
            return False
        pad = bound_r * 0.55 + 0.08
        for (x0, x1, y0, y1, z0, z1) in occ_boxes:
            if (x0 - pad <= pos[0] <= x1 + pad and y0 - pad <= pos[1] <= y1 + pad and z0 <= pos[2] <= z1):
                return False
        return True

    # 1. Architectural Folding Kaleidoscope Ribs & Ceiling Mirror-Slabs cascading down the 13.4m corridor!
    for k in range(n_slabs):
        u = k / max(n_slabs - 1, 1)
        y = -8.4 + u * 14.2
        u_hall = np.clip((y + 11.0) / 16.0, 0.0, 1.0)
        side = -1.0 if (k % 2 == 0) else 1.0
        # Alternate flanking vertical mirror ribs & angled overhead kaleidoscope ceiling ribs!
        is_ceil = (k % 3 == 2)
        if is_ceil:
            x = float(side * rng.uniform(0.35, 1.35))
            z = float(2.25 + 0.70 * u_hall + rng.uniform(-0.18, 0.25))
            sz = (float(rng.uniform(0.28, 0.56)), float(rng.uniform(0.032, 0.055)), float(rng.uniform(0.26, 0.52)))
            base_rot = (math.radians(rng.uniform(25, 65)), math.radians(side * rng.uniform(18, 48)), math.radians(rng.uniform(-35, 35)))
        else:
            x = float(side * rng.uniform(1.05, 1.95))
            z = float(0.72 + 0.90 * u_hall + rng.uniform(-0.22, 0.35))
            sz = (float(rng.uniform(0.24, 0.44)), float(rng.uniform(0.035, 0.060)), float(rng.uniform(0.45, 0.92)))
            base_rot = (math.radians(rng.uniform(-16, 16)), math.radians(side * rng.uniform(-22, 22)), math.radians(side * rng.uniform(18, 52)))
        pos = np.array([x, y, z], np.float64)
        bound_r = float(np.linalg.norm(sz)) + 0.08
        if not clears_cam_and_walls(pos, bound_r):
            continue
        items.append({
            "pos": pos, "sz": sz, "bound_r": bound_r,
            "kind": 0.0 if rng.random() < 0.72 else 1.0,
            "rot0": base_rot,
            "fold_amp": (float(rng.uniform(-0.55, 0.55)), float(rng.uniform(-0.65, 0.65)), float(rng.uniform(-0.75, 0.75))),
            "shift_amp": np.array([side * -0.22, 0.0, float(rng.uniform(-0.18, 0.22))]),
            "trigger_y": y - 4.8,  # cascades open symmetrically as the camera glides toward it!
            "mirror_mix": float(rng.uniform(0.62, 0.88)),
            "bevel": 0.022,
            "phase": float(rng.uniform(0, 6.28)),
            "tier": 2,
        })

    # 2. Central Levitating Faceted Crystal Polyhedra & Diamond Prisms (flanking the corridor & climax hall!)
    for k in range(n_poly * 3):
        if sum(1 for it in items if it["tier"] == 1) >= n_poly:
            break
        y = float(rng.uniform(-9.4, 5.8))
        u_hall = np.clip((y + 11.0) / 16.0, 0.0, 1.0)
        if y < -8.50:
            x = float(rng.uniform(-0.35, 1.55))
        else:
            side = -1.0 if rng.random() < 0.5 else 1.0
            x = float(side * rng.uniform(0.55, 1.85)) if y < 2.2 else float(rng.uniform(-1.35, 1.35))
        z = float(rng.uniform(0.45 + 0.65 * u_hall, 1.85 + 0.85 * u_hall))
        kind = 2.0 if rng.random() < 0.62 else 1.0
        r0 = float(rng.uniform(0.14, 0.34))
        sz = (r0, r0 * float(rng.uniform(0.65, 1.0)), r0 * float(rng.uniform(1.15, 2.15)))
        pos = np.array([x, y, z], np.float64)
        bound_r = float(np.linalg.norm(sz)) + 0.06
        if not clears_cam_and_walls(pos, bound_r):
            continue
        items.append({
            "pos": pos, "sz": sz, "bound_r": bound_r, "kind": kind,
            "rot0": (float(rng.uniform(0, 6.28)), float(rng.uniform(0, 6.28)), float(rng.uniform(0, 6.28))),
            "fold_amp": (float(rng.uniform(-1.1, 1.1)), float(rng.uniform(-1.1, 1.1)), float(rng.uniform(-1.1, 1.1))),
            "shift_amp": np.array([0.0, 0.0, float(rng.uniform(0.10, 0.28))]),
            "trigger_y": y - 5.2,
            "mirror_mix": float(rng.uniform(0.42, 0.72)),
            "bevel": 0.015,
            "phase": float(rng.uniform(0, 6.28)),
            "tier": 1,
        })

    # 3. 360 Tumbling Faceted Crystal Micro-Shards peeling off the folding joints!
    anchors = [it["pos"] for it in items]
    for k in range(n_micro * 3):
        if sum(1 for it in items if it["tier"] == 0) >= n_micro:
            break
        if rng.random() < 0.68 and anchors:
            anc = anchors[int(rng.integers(0, len(anchors)))]
            pos = anc + rng.normal(0.0, [0.55, 0.65, 0.45])
            pos[2] = float(np.clip(pos[2], 0.22, 2.90))
        else:
            y = float(rng.uniform(-10.2, 5.6))
            u_hall = np.clip((y + 11.0) / 16.0, 0.0, 1.0)
            x = float(rng.uniform(-0.45, 1.65)) if y < -8.5 else float(rng.uniform(-2.0, 2.0))
            z = float(rng.uniform(0.28 + 0.55 * u_hall, 2.10 + 0.80 * u_hall))
            pos = np.array([x, y, z], np.float64)
        r0 = float(np.exp(rng.uniform(math.log(0.016), math.log(0.068))))
        sz = (r0, r0 * float(rng.uniform(0.35, 0.75)), r0 * float(rng.uniform(1.25, 2.40)))
        bound_r = float(np.linalg.norm(sz)) + 0.02
        if not clears_cam_and_walls(pos, bound_r):
            continue
        items.append({
            "pos": pos, "sz": sz, "bound_r": bound_r, "kind": 2.0 if rng.random() < 0.78 else 0.0,
            "rot0": (float(rng.uniform(0, 6.28)), float(rng.uniform(0, 6.28)), float(rng.uniform(0, 6.28))),
            "fold_amp": (float(rng.uniform(-1.6, 1.6)), float(rng.uniform(-1.6, 1.6)), float(rng.uniform(-1.6, 1.6))),
            "shift_amp": np.array([float(rng.uniform(-0.08, 0.08)), 0.0, float(rng.uniform(-0.12, 0.14))]),
            "trigger_y": pos[1] - 5.5,
            "mirror_mix": float(rng.uniform(0.45, 0.78)),
            "bevel": 0.005,
            "phase": float(rng.uniform(0, 6.28)),
            "tier": 0,
        })

    print(f"Mirror Dimension: {sum(1 for x in items if x['tier']==2)} slabs + {sum(1 for x in items if x['tier']==1)} polyhedra + {sum(1 for x in items if x['tier']==0)} micro-shards = {len(items)} prisms")
    return items


def main():
    c = load(sys.argv[1] if len(sys.argv) > 1 and not sys.argv[1].startswith("--") else here / "piece_mirror")
    S = scene(c)
    W, H, N, fps = c["W"], c["H"], c["N"], c["fps"]
    f = float(S["f"])
    cfg = c.get("mirror", {})
    stills = [int(x) for x in sys.argv[sys.argv.index("--still") + 1].split(",")] if "--still" in sys.argv else None
    out_dir = sys.argv[sys.argv.index("--out") + 1] if "--out" in sys.argv else "out"
    cam_path = np.asarray(S["cam"], np.float64)

    # Exact 6 COLMAP-triangulated & verified 3D architectural holdout boxes for Pexels 7578546!
    occ_boxes = [
        (-4.00, -0.71, -13.50, -8.56, 0.0, 5.0),
        ( 2.28,  4.50, -13.50, -8.85, 0.0, 5.0),
        (-1.50, -0.90,  -1.12, -0.57, 0.0, 5.0),
        ( 2.24,  2.82,  -1.00, -0.42, 0.0, 5.0),
        (-1.56, -1.07,   7.10,  7.65, 0.0, 5.0),
        ( 1.54,  2.05,   7.10,  7.65, 0.0, 5.0),
    ]

    items = build_mirror_dimension(
        cam_path, occ_boxes,
        n_slabs=int(cfg.get("n_slabs", 42)),
        n_poly=int(cfg.get("n_poly", 34)),
        n_micro=int(cfg.get("n_micro", 360)),
        seed=int(cfg.get("seed", 2030)),
    )
    caust_idx = sorted(range(len(items)), key=lambda idx: -items[idx]["bound_r"])[:32]

    ctx = moderngl.create_standalone_context(require=330)
    base_prog = ctx.program(vertex_shader=VS, fragment_shader=BASE_FS)
    prism_prog = ctx.program(vertex_shader=PRISM_VS, fragment_shader=PRISM_FS)
    post_prog = ctx.program(vertex_shader=VS, fragment_shader=POST_FS)
    copy2_prog = ctx.program(
        vertex_shader=VS,
        fragment_shader="""#version 330
        uniform sampler2D uSrc0, uSrc1; uniform vec2 uRes;
        layout(location=0) out vec4 o0; layout(location=1) out vec4 o1;
        void main(){ vec2 uv = gl_FragCoord.xy / uRes; o0 = texture(uSrc0, uv); o1 = texture(uSrc1, uv); }""",
    )

    tri = ctx.buffer(np.array([-1, -1, 3, -1, -1, 3], "f4").tobytes())
    va_base = ctx.vertex_array(base_prog, [(tri, "2f", "p")])
    va_copy2 = ctx.vertex_array(copy2_prog, [(tri, "2f", "p")])
    va_post = ctx.vertex_array(post_prog, [(tri, "2f", "p")])

    quad_vbo = ctx.buffer(np.array([-1, -1, 1, -1, 1, 1, -1, -1, 1, 1, -1, 1], "f4").tobytes())
    inst_buf = ctx.buffer(reserve=len(items) * 20 * 4)
    va_prism = ctx.vertex_array(prism_prog, [
        (quad_vbo, "2f", "inQuad"),
        (inst_buf, "4f 4f 4f 4f 4f/i", "inPosScale", "inSizeKind", "inRot0", "inRot1", "inRot2"),
    ])

    plate_t = ctx.texture((W, H), 3); plate_t.filter = (moderngl.LINEAR, moderngl.LINEAR); plate_t.repeat_x = plate_t.repeat_y = False
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
    base_prog["uRes"].value = (float(W), float(H)); base_prog["uF"].value = f
    base_prog["uSun"].value = tuple(map(float, S["sun"]))
    base_prog["uOccMin"].value = [(b[0], b[2], b[4]) for b in occ_boxes]
    base_prog["uOccMax"].value = [(b[1], b[3], b[5]) for b in occ_boxes]
    base_prog["uNumOcc"].value = len(occ_boxes)

    prism_prog["uRes"].value = (float(W), float(H)); prism_prog["uF"].value = f
    prism_prog["uSun"].value = tuple(map(float, S["sun"]))
    prism_prog["uIor"].value = float(cfg.get("ior", 1.54))
    prism_prog["uDisp"].value = float(cfg.get("dispersion", 0.022))
    prism_prog["uProxyDist"].value = 7.0

    post_prog["uRes"].value = (float(W), float(H))
    post_prog["uFocusDist"].value = 4.2
    post_prog["uNearBlur"].value = 4.4
    post_prog["uFarBlur"].value = 5.2
    post_prog["uBaseSoft"].value = 0.68
    post_prog["uMaxCoc"].value = 10.5
    post_prog["uGrainAmt"].value = 0.012
    post_prog["uBlackFloor"].value = 0.016

    def smooth01(x):
        x = max(0.0, min(1.0, x))
        return x * x * (3.0 - 2.0 * x)

    def render_one(i):
        pl = frame(c, i)
        t_sec = i / fps
        ro = cam_path[i]; R = np.asarray(S["rot"][i], np.float64)
        Rt = tuple(R.astype(np.float32).T.flatten())
        cam_y = float(ro[1])

        # Evaluate cascading Doctor-Strange Mirror Dimension geometric fold wave as camera glides down hall!
        rows_p = []
        caust_list = []
        for idx, it in enumerate(items):
            # Smooth wave fold triggered by camera approaching each station + continuous slow architectural drift
            fold_u = smooth01((cam_y - (it["trigger_y"] - 3.5)) / 5.5)
            # Crisp geometric quantized-feel ease (smooth acceleration into locked crystal angle + subtle zero-G breath)
            breath = 0.10 * math.sin(t_sec * 0.72 + it["phase"])
            ang_x = it["rot0"][0] + it["fold_amp"][0] * fold_u + breath * (1.5 if it["tier"] == 0 else 0.4)
            ang_y = it["rot0"][1] + it["fold_amp"][1] * fold_u + breath * 0.5
            ang_z = it["rot0"][2] + it["fold_amp"][2] * fold_u + (t_sec * 0.35 if it["tier"] == 0 else breath * 0.4)
            Rloc = euler_to_mat(ang_x, ang_y, ang_z)   # local -> world; transpose is world -> local
            p_now = it["pos"] + it["shift_amp"] * fold_u + np.array([0.0, 0.0, 0.04 * math.sin(t_sec * 0.8 + it["phase"])])
            if idx in caust_idx:
                caust_list.append((float(p_now[0]), float(p_now[1]), float(p_now[2]), float(it["bound_r"])))
            z_cam = float(((p_now - ro) @ R)[2])
            if z_cam > 0.08:
                Rt_loc = Rloc.T  # rows of world->local
                rows_p.append((z_cam, [
                    float(p_now[0]), float(p_now[1]), float(p_now[2]), float(it["bound_r"]),
                    float(it["sz"][0]), float(it["sz"][1]), float(it["sz"][2]), float(it["kind"]),
                    float(Rt_loc[0,0]), float(Rt_loc[0,1]), float(Rt_loc[0,2]), float(it["mirror_mix"]),
                    float(Rt_loc[1,0]), float(Rt_loc[1,1]), float(Rt_loc[1,2]), float(it["bevel"]),
                    float(Rt_loc[2,0]), float(Rt_loc[2,1]), float(Rt_loc[2,2]), 1.0,
                ]))

        n_caust = len(caust_list)
        while len(caust_list) < 32:
            caust_list.append((0.0, 0.0, 0.0, 0.0))

        plate_t.write(pl[:, :, ::-1].tobytes())
        plate_t.use(0)
        base_prog["uPlate"].value = 0
        base_prog["uRo"].value = tuple(map(float, ro)); base_prog["uCam"].value = Rt
        base_prog["uTime"].value = float(t_sec)
        base_prog["uCaustPrisms"].value = caust_list[:32]
        base_prog["uNumCaust"].value = n_caust
        fbo_hero.use()
        ctx.disable(moderngl.BLEND)
        va_base.render(moderngl.TRIANGLES)

        fbo_scene.use()
        t_hero.use(0); t_meta0.use(1)
        copy2_prog["uSrc0"].value = 0; copy2_prog["uSrc1"].value = 1
        va_copy2.render(moderngl.TRIANGLES)

        rows_p.sort(key=lambda x: -x[0])
        if rows_p:
            data = np.array([r[1] for r in rows_p], dtype=np.float32)
            inst_buf.write(data.tobytes())
            t_hero.use(0); t_depth.use(1)
            prism_prog["uBase"].value = 0; prism_prog["uDepth"].value = 1
            prism_prog["uRo"].value = tuple(map(float, ro)); prism_prog["uCam"].value = Rt
            ctx.enable(moderngl.BLEND)
            ctx.blend_func = (moderngl.SRC_ALPHA, moderngl.ONE_MINUS_SRC_ALPHA)
            va_prism.render(moderngl.TRIANGLES, instances=len(rows_p))
            ctx.disable(moderngl.BLEND)

        fbo_final.use()
        t_scene.use(0); t_meta.use(1)
        post_prog["uScene"].value = 0; post_prog["uMeta"].value = 1
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
            rows.append(np.hstack([t_im(pl, f"plate {i}"), t_im(comp, f"mirror dim {i}")]))
        out_p = work(c, "still_mirror_sheet.jpg")
        cv2.imwrite(str(out_p), np.vstack(rows), [cv2.IMWRITE_JPEG_QUALITY, 92])
        if hero is not None:
            cv2.imwrite(str(work(c, "still_mirror_hero.jpg")), hero, [cv2.IMWRITE_JPEG_QUALITY, 94])
        print("stills ->", out_p)
        return

    comp_dir = work(c, "comp")
    shutil.rmtree(comp_dir, ignore_errors=True)
    comp_dir.mkdir(parents=True)
    ts = time.time()
    for i in range(N):
        _, comp = render_one(i)
        cv2.imwrite(str(comp_dir / f"{i:04d}.png"), comp, [cv2.IMWRITE_PNG_COMPRESSION, 1])
        if i % 45 == 0 or i == N - 1:
            print(f"mirror dim frame {i}/{N} ({time.time() - ts:.1f}s)", flush=True)
    encode(c, out_dir=out_dir)


if __name__ == "__main__":
    main()
