"""#5 3D Air-Glass Shatter, Bullet-Time Freeze & Time Rewind (ModernGL, OpenGL 3.3).

    python render_air_shatter.py [piece_dir] --still 28,58,118,185,226   quick look -> work/still_shatter_sheet.jpg
    python render_air_shatter.py [piece_dir] --out out                  full render -> out/*_Composite.mp4, *_Compare_1080x1350.mp4

Zero text/HUD overlays — 100% pure 3-Act cinema shatter VFX:
- Act 1 (f=6..42): An invisible tempered glass wall suspended across the 3D space cracks radially with razor-sharp Voronoi
  spiderweb fractures that tilt-refract the live room behind every glass cell.
- Act 2 (f=43..153): SHATTER BANG! 840 dense sharp-edged 3D polygonal glass shards (280 beveled irregular 5-cut plates +
  560 diamond micro-splinters) explode straight toward & around the advancing camera and freeze dead-still in bullet-time
  while the real COLMAP camera glides right through the hovering shard storm.
- Act 3 (f=154..239): TIME REWIND! Subtle tremble -> all 840 tumbling 3D shards whip backward along their exact inverse
  trajectories & rotations, snapping back into the exact fracture plane right ahead of the camera and healing into pure air.
- 3 COLMAP-verified architectural wall holdouts + floor shard caustics + Pass-3 thin-lens bokeh, halation & sensor grain.
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

# Pass 1: 3D Cracking / Healing Tempered Air-Glass Pane + 3-Box Architectural Holdouts + Floor Shard Glints
PANE_FS = """#version 330
uniform sampler2D uPlate;
uniform vec2 uRes;
uniform float uF, uTime;
uniform vec3 uRo, uSun;
uniform mat3 uCam;
uniform vec3 uOccMin[4];
uniform vec3 uOccMax[4];
uniform int uNumOcc;

/* Cracking air-glass plane at y = uPaneCenter.y */
uniform vec3 uPaneCenter;
uniform float uPaneVis;       /* 0..1 visibility of intact cracking/healing glass wall */
uniform float uCrackRad;      /* radial propagation radius (m) of spiderweb fracture */
uniform float uCrackStr;      /* 0..1 intensity of fracture lines & cell tilt */
uniform vec4 uCaustShards[28];
uniform int uNumCaust;

layout(location=0) out vec4 oComp;
layout(location=1) out vec4 oDepth;   /* R: tOcc, G: tOcc */
layout(location=2) out vec4 oMeta;    /* R: cgAlpha, G: cgInvDepth, B: 1/tOcc, A: 1.0 */

float hash11(float p) { return fract(sin(p * 127.1) * 43758.5453); }

vec2 hash22(vec2 p) {
  p = vec2(dot(p, vec2(127.1, 311.7)), dot(p, vec2(269.5, 183.3)));
  return fract(sin(p) * 43758.5453);
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
    if (tFar > max(tNear, 0.05)) tBest = min(tBest, (tNear > 0.05) ? tNear : 0.05);
  }
  return tBest;
}

/* Voronoi distance to nearest cell edge (exact sharp straight tempered-glass fracture seams!) + cell id */
vec3 voronoiEdge(vec2 x) {
  vec2 n = floor(x);
  vec2 f = fract(x);
  vec2 mg, mr;
  float md = 8.0;
  vec2 bestId = vec2(0.0);
  for (int j = -1; j <= 1; j++) {
    for (int i = -1; i <= 1; i++) {
      vec2 g = vec2(float(i), float(j));
      vec2 o = hash22(n + g);
      vec2 r = g + o - f;
      float d = dot(r, r);
      if (d < md) { md = d; mr = r; mg = g; bestId = n + g; }
    }
  }
  md = 8.0;
  for (int j = -2; j <= 2; j++) {
    for (int i = -2; i <= 2; i++) {
      vec2 g = mg + vec2(float(i), float(j));
      vec2 o = hash22(n + g);
      vec2 r = g + o - f;
      if (dot(mr - r, mr - r) > 1e-5) {
        md = min(md, dot(0.5 * (mr + r), normalize(r - mr)));
      }
    }
  }
  return vec3(md, bestId);
}

void main() {
  vec2 px0 = vec2(gl_FragCoord.x, uRes.y - gl_FragCoord.y);
  vec2 uv0 = px0 / uRes;
  vec3 plateCol = texture(uPlate, clamp(uv0, vec2(0.002), vec2(0.998))).rgb;
  vec3 rd = normalize(uCam * vec3((px0 - 0.5 * uRes) / uF, 1.0));
  float tOcc = hitOccluders(uRo, rd);

  vec3 col = plateCol;
  float cgA = 0.0;
  float cgInvD = 0.0;

  /* Soft sunlit glass shard sparkles cast onto z=0 oak floor */
  float tFloor = (rd.z < -0.004) ? (-uRo.z / rd.z) : 1e4;
  if (tFloor > 0.10 && tFloor < min(tOcc, 18.0)) {
    vec3 Xf = uRo + rd * tFloor;
    float roomMask = smoothstep(3.4, 2.2, abs(Xf.x)) * smoothstep(2.2, 0.5, Xf.y) * smoothstep(-6.8, -5.5, Xf.y);
    if (roomMask > 0.01) {
      vec3 sun = normalize(uSun);
      float invSz = 1.0 / max(sun.z, 0.25);
      float caust = 0.0, shad = 0.0;
      for (int i = 0; i < 28; i++) {
        if (i >= uNumCaust) break;
        vec4 sh = uCaustShards[i];
        if (sh.w <= 0.01 || sh.z <= 0.05) continue;
        vec2 sp = sh.xy - sun.xy * (sh.z * invSz);
        float r = length(Xf.xy - sp);
        float R = sh.w * 1.9 + sh.z * 0.04;
        if (r > R * 2.8) continue;
        caust += exp(-(r * r) / max(R * R * 0.45, 0.002)) * 0.45;
        shad += exp(-(r * r) / max(R * R * 1.6, 0.005)) * 0.08;
      }
      float luma = dot(plateCol, vec3(0.299, 0.587, 0.114));
      float fMask = roomMask * smoothstep(0.08, 0.24, luma) * (1.0 - smoothstep(10.0, 16.5, tFloor));
      col = col * (1.0 - clamp(shad, 0.0, 0.25) * fMask);
      col += vec3(1.0, 0.96, 0.88) * (clamp(caust, 0.0, 1.2) * 0.22 * fMask) * (0.45 + 0.55 * col);
    }
  }

  /* Act 1 & Act 3: Ray-intersect the 3D vertical air-glass fracture plane y = uPaneCenter.y! */
  if (uPaneVis > 0.005 && abs(rd.y) > 1e-4) {
    float tP = (uPaneCenter.y - uRo.y) / rd.y;
    if (tP > 0.15 && tP < min(tOcc, 14.0)) {
      vec3 Xp = uRo + rd * tP;
      vec2 rel = vec2(Xp.x - uPaneCenter.x, Xp.z - uPaneCenter.z);
      float distC = length(rel);
      /* Radial spiderweb propagation envelope + organic fade into clear air */
      float crackEnv = smoothstep(uCrackRad + 0.12, uCrackRad - 0.24, distC) * smoothstep(2.55, 0.95, distC) * uPaneVis;
      /* Keep cleanly inside room bounds */
      float roomEnv = smoothstep(2.95, 2.10, abs(Xp.x)) * smoothstep(0.05, 0.25, Xp.z) * smoothstep(2.95, 2.55, Xp.z);
      float env = crackEnv * roomEnv;
      if (env > 0.004) {
        /* Combine coarse structural radial shards + fine splinter fracture near impact center! */
        float ang = atan(rel.y, rel.x);
        /* Distance-adaptive hairline width so close-up healing cracks stay razor-thin (~1.5-2.5 px), never chalky! */
        float wScale = clamp(tP / 4.2, 0.16, 1.0);
        /* Polar-warped + Cartesian Voronoi for authentic radial impact spiderweb + jagged straight-edged glass shards! */
        vec2 pCart = rel * 2.4;
        vec3 v1 = voronoiEdge(pCart);
        vec3 v2 = voronoiEdge(rel * 5.4 + vec2(3.7, -2.1));
        /* Crisp jagged straight radial stress fractures radiating from impact center (no wavy sine loops!) */
        float segIdx = floor(distC * 3.2);
        float angWarp = ang + (hash11(segIdx * 11.7 + floor(ang * 2.864789)) - 0.5) * 0.08;
        float raySpokes = abs(fract(angWarp * 2.864789) - 0.5) * max(distC * 2.2, 0.18);

        float edge1 = smoothstep(0.021 * wScale, 0.0022 * wScale, v1.x);
        float edge2 = smoothstep(0.030 * wScale, 0.0032 * wScale, v2.x) * smoothstep(1.60, 0.20, distC);
        float edgeRay = smoothstep(0.014 * wScale, 0.0018 * wScale, raySpokes) * smoothstep(2.25, 0.12, distC);
        float crackLine = clamp(max(max(edge1, edge2 * 0.85), edgeRay * 0.88), 0.0, 1.0);
        /* Anisotropic internal reflection glint along fracture facets (uneven catchlight like real broken glass!) */
        float seamGlint = 0.42 + 0.58 * pow(abs(sin(ang * 4.0 + v1.y * 7.3 + v1.z * 5.1)), 1.6);

        /* Each fractured Voronoi glass cell tilts slightly, refracting & shifting the room behind it! */
        vec2 cellTilt = (hash22(v1.yz) - 0.5) * 2.0 * uCrackStr * env;
        vec2 uvRefr = uv0 + cellTilt * 0.025;
        vec2 uvCA   = cellTilt * 0.0040;
        vec3 refrCol = vec3(
          texture(uPlate, clamp(uvRefr + uvCA, vec2(0.005), vec2(0.995))).r,
          texture(uPlate, clamp(uvRefr,        vec2(0.005), vec2(0.995))).g,
          texture(uPlate, clamp(uvRefr - uvCA, vec2(0.005), vec2(0.995))).b
        );
        /* Subtle cell-by-cell angular glass specular catchlight from left window & ceiling lights! */
        float cellGlint = pow(max(0.0, dot(normalize(vec3(cellTilt * 0.45, 1.0)), normalize(vec3(-0.35, 0.25, 0.90)))), 12.0) * 0.26 * uCrackStr;
        /* Crisp cool-silver internal total-reflection gleam right along razor hairline fracture seams! */
        vec3 seamCol = vec3(0.88, 0.96, 1.00) * crackLine * seamGlint * (0.56 + 0.36 * smoothstep(1.0, 0.0, distC)) * uCrackStr;
        vec3 glassCol = mix(col, refrCol * 1.03 + vec3(0.94, 0.98, 1.0) * cellGlint, env * 0.88) + seamCol * env;

        /* Subtle dark bevel micro-shadow right next to bright crack seam for physical glass thickness */
        float darkSeam = smoothstep(0.044 * wScale, 0.015 * wScale, v1.x) * (1.0 - edge1) * 0.22 * env * uCrackStr;
        col = glassCol * (1.0 - darkSeam);
        cgA = clamp(max(env * 0.38 * uCrackStr, crackLine * env), 0.0, 1.0);
        cgInvD = cgA / max(tP, 0.20);
      }
    }
  }

  oComp = vec4(clamp(col, 0.0, 1.0), 1.0);
  oDepth = vec4(tOcc, tOcc, 0.0, 1.0);
  oMeta = vec4(cgA, cgInvD, 1.0 / max(tOcc, 0.20), 1.0);
}
"""

# Pass 2: Instanced 3D Raymarched Sharp Irregular Polygonal Glass Shards (~840 shards: 280 beveled plates + 560 splinters!)
SHARD_VS = """#version 330
in vec2 inQuad;
in vec4 inPosBound;   /* xyz = world pos (m), w = bounding radius (m) */
in vec4 inSizeSeed;   /* x = radiusXY (m), y = halfThickZ (m), z = aspectXY, w = polySeed */
in vec4 inRot0;       /* row 0 of world->local rotation matrix, w = opacity */
in vec4 inRot1;       /* row 1 of world->local rotation matrix, w = isSplinter (0=plate, 1=needle splinter) */
in vec4 inRot2;       /* row 2 of world->local rotation matrix, w = edgeGlow */

uniform vec2 uRes;
uniform float uF;
uniform vec3 uRo;
uniform mat3 uCam;

out vec3 vCenter;
out float vBoundR;
out vec4 vSizeSeed;
out mat3 vInvRot;
out float vAlpha;
out float vIsSplint;

void main() {
  vCenter = inPosBound.xyz;
  vBoundR = inPosBound.w;
  vSizeSeed = inSizeSeed;
  vInvRot = transpose(mat3(inRot0.xyz, inRot1.xyz, inRot2.xyz));
  vAlpha = inRot0.w;
  vIsSplint = inRot1.w;

  vec3 q = transpose(uCam) * (vCenter - uRo);
  if (vAlpha <= 0.01 || q.z <= 0.06 + vBoundR * 0.12) {
    gl_Position = vec4(2.0, 2.0, 2.0, 1.0);
    return;
  }
  vec2 ctrPx = uF * q.xy / q.z + 0.5 * uRes;
  float pxR = clamp(uF * vBoundR / max(q.z - vBoundR * 0.85, 0.04) + 3.5, 3.5, 980.0);
  vec2 px = ctrPx + inQuad * pxR;
  vec2 ndc = vec2(px.x / uRes.x * 2.0 - 1.0, (1.0 - px.y / uRes.y) * 2.0 - 1.0);
  gl_Position = vec4(ndc, 0.0, 1.0);
}
"""

SHARD_FS = """#version 330
in vec3 vCenter;
in float vBoundR;
in vec4 vSizeSeed;
in mat3 vInvRot;
in float vAlpha;
in float vIsSplint;

uniform sampler2D uBase;    /* Pass-1 scene color (bottom-up) */
uniform sampler2D uDepth;   /* Pass-1 3-box wall holdout depth (bottom-up) */
uniform vec2 uRes;
uniform float uF, uIor, uDisp, uProxyDist;
uniform vec3 uRo, uSun;
uniform mat3 uCam;

layout(location=0) out vec4 oCol;
layout(location=1) out vec4 oMeta;

/* Exact razor-sharp irregular 5-sided convex polygonal glass shard plate with physical thickness hz! */
float sdShardLocal(vec3 q) {
  float R = vSizeSeed.x;
  float hz = vSizeSeed.y;
  float asp = vSizeSeed.z;
  float sd = vSizeSeed.w;

  vec2 p = vec2(q.x, q.y / max(asp, 0.4));
  /* 5 asymmetric half-plane cuts generating unique jagged triangular/quad/pentagonal shattered glass silhouettes! */
  float a0 = sd * 6.2831;
  float a1 = a0 + 1.15 + 0.45 * sin(sd * 17.3);
  float a2 = a1 + 1.25 + 0.42 * cos(sd * 23.1);
  float a3 = a2 + 1.30 + 0.40 * sin(sd * 31.7);
  float a4 = a3 + 1.35 + 0.35 * cos(sd * 41.9);

  float d2 = dot(p, vec2(cos(a0), sin(a0))) - R * (0.62 + 0.28 * fract(sd * 13.7));
  d2 = max(d2, dot(p, vec2(cos(a1), sin(a1))) - R * (0.58 + 0.32 * fract(sd * 29.3)));
  d2 = max(d2, dot(p, vec2(cos(a2), sin(a2))) - R * (0.56 + 0.34 * fract(sd * 43.1)));
  d2 = max(d2, dot(p, vec2(cos(a3), sin(a3))) - R * (0.60 + 0.30 * fract(sd * 59.9)));
  if (vIsSplint < 0.5) {
    d2 = max(d2, dot(p, vec2(cos(a4), sin(a4))) - R * (0.64 + 0.28 * fract(sd * 71.3)));
  }

  /* Extrude 2D irregular shard polygon to glass plate thickness 2*hz with crisp tiny bevel */
  float dz = abs(q.z) - hz;
  vec2 w = vec2(d2, dz);
  return min(max(w.x, w.y), 0.0) + length(max(w, 0.0)) - min(hz * 0.22, 0.0025);
}

float sdShard(vec3 p) {
  return sdShardLocal(vInvRot * (p - vCenter));
}

vec3 shardNormal(vec3 p, out float isCutEdge) {
  float h = max(vBoundR * 0.008, 0.0003);
  vec2 e = vec2(h, -h);
  vec3 nW = normalize(
    e.xyy * sdShard(p + e.xyy) +
    e.yyx * sdShard(p + e.yyx) +
    e.yxy * sdShard(p + e.yxy) +
    e.xxx * sdShard(p + e.xxx)
  );
  /* Detect sharp cut-glass rim edges (local normal perpendicular to plate Z axis!) */
  vec3 nLoc = vInvRot * nW;
  isCutEdge = 1.0 - smoothstep(0.25, 0.78, abs(nLoc.z));
  return nW;
}

vec3 sampleEnv(vec3 ro, vec3 rd) {
  float tf = (rd.z < -0.012) ? (-ro.z / rd.z) : 1e4;
  if (tf < 0.05 || tf > 22.0) tf = 1e4;
  vec3 X = ro + rd * min(tf, uProxyDist);
  vec3 q = transpose(uCam) * (X - uRo);
  float fwdVis = smoothstep(0.05, 0.70, q.z);
  vec2 uv = (uF * q.xy / max(q.z, 0.35) + 0.5 * uRes) / uRes;
  uv = 1.0 - abs(1.0 - clamp(uv, vec2(-0.85), vec2(1.85)));
  uv = clamp(uv, vec2(0.010), vec2(0.990));
  vec3 plateS = texture(uBase, vec2(uv.x, 1.0 - uv.y)).rgb;
  vec3 rearRoom = mix(vec3(0.45, 0.38, 0.32), vec3(0.84, 0.87, 0.91), smoothstep(-0.35, 0.55, rd.z));
  return mix(rearRoom, plateS, fwdVis);
}

void main() {
  vec2 px0 = vec2(gl_FragCoord.x, uRes.y - gl_FragCoord.y);
  float tOcc = texture(uDepth, gl_FragCoord.xy / uRes).r;

  vec3 acc = vec3(0.0);
  float cov = 0.0;
  float minT = 1e4;
  vec2 offs[4] = vec2[4](vec2(-0.25, -0.25), vec2(0.25, -0.25), vec2(-0.25, 0.25), vec2(0.25, 0.25));

  /* Real Pexels 7578547 room light directions: left daylight window, ceiling cans, right kitchen pendants */
  vec3 L_win  = normalize(vec3(-0.78, 0.25, 0.56));
  vec3 L_ceil = normalize(vec3( 0.05, 0.20, 0.98));
  vec3 L_pend = normalize(vec3( 0.68, 0.35, 0.64));

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
    float eps = max(vBoundR * 0.004, 0.0002);
    int steps = (vIsSplint > 0.5) ? 24 : 34;
    for (int s = 0; s < 34; s++) {
      if (s >= steps) break;
      float d = sdShard(uRo + rd * t);
      if (d < eps) { tHit = t; break; }
      t += max(d * 0.90, 0.0008);
      if (t > tEnd) break;
    }
    if (tHit < 0.0 || tHit >= tOcc) continue;
    minT = min(minT, tHit);

    vec3 p1 = uRo + rd * tHit;
    float isCutEdge;
    vec3 n1 = shardNormal(p1, isCutEdge);

    /* Dual-interface tempered plate glass refraction (flat slab shifts view + sharp internal reflection at cut edge!) */
    vec3 rdIn = refract(rd, n1, 1.0 / uIor);
    if (length(rdIn) < 0.001) rdIn = reflect(rd, n1);
    float thick = vSizeSeed.y * 2.2 / max(abs(dot(rdIn, n1)), 0.35);
    vec3 p2 = p1 + rdIn * min(thick, vBoundR * 1.4);
    /* Slight angular offset on thin plate exit for realistic broken-glass optical displacement */
    vec3 rdOutG = normalize(mix(rd, rdIn, isCutEdge > 0.4 ? 0.95 : 0.42));
    vec3 bend = rdOutG - rd;
    vec3 rdOutR = normalize(rd + bend * ((uIor - uDisp) - 1.0) / max(uIor - 1.0, 0.05));
    vec3 rdOutB = normalize(rd + bend * ((uIor + uDisp) - 1.0) / max(uIor - 1.0, 0.05));

    vec3 refrCol = vec3(sampleEnv(p2, rdOutR).r, sampleEnv(p2, rdOutG).g, sampleEnv(p2, rdOutB).b);
    /* Authentic tempered cut-glass green-aqua edge tint on broken rims! */
    vec3 edgeTint = mix(vec3(1.0, 1.0, 1.0), vec3(0.84, 0.98, 0.95), isCutEdge * 0.85);
    refrCol = refrCol * edgeTint * 1.05;

    float ndv = max(dot(-rd, n1), 0.0);
    float fresnel = 0.05 + 0.95 * pow(1.0 - ndv, 3.8);
    vec3 reflDir = reflect(rd, n1);
    vec3 envRefl = sampleEnv(p1 + n1 * 0.004, reflDir);
    float skyUp = smoothstep(-0.15, 0.85, reflDir.z);
    envRefl = mix(envRefl * 1.02 + vec3(0.04), envRefl * 1.20 + vec3(0.24, 0.27, 0.30), skyUp * 0.55);

    float spWin  = pow(max(dot(n1, normalize(L_win  - rd)), 0.0), 190.0) * 2.6 + pow(max(dot(n1, normalize(L_win  - rd)), 0.0), 22.0) * 0.16;
    float spCeil = pow(max(dot(n1, normalize(L_ceil - rd)), 0.0), 150.0) * 1.25;
    float spPend = pow(max(dot(n1, normalize(L_pend - rd)), 0.0), 160.0) * 0.95;
    /* Blazing bright razor-silver internal reflection right along broken cut-glass edges! */
    float cutGleam = isCutEdge * (0.45 + 0.55 * fresnel);

    vec3 col = mix(refrCol, envRefl, clamp(fresnel * 0.62 + isCutEdge * 0.18, 0.0, 0.86));
    col += vec3(0.95, 0.99, 1.00) * spWin + vec3(1.00, 0.96, 0.88) * spCeil + vec3(1.00, 0.92, 0.78) * spPend;
    col += vec3(0.88, 0.98, 0.97) * cutGleam;
    acc += col;
    cov += 1.0;
  }

  if (cov < 0.5) discard;
  float a = clamp(cov * 0.25 * vAlpha, 0.0, 1.0);
  oCol = vec4(clamp(acc / cov, 0.0, 1.0), a);
  oMeta = vec4(1.0, 1.0 / max(minT, 0.20), 1.0 / max(tOcc, 0.20), a);
}
"""

# Pass 3: Camera Optical Finish (Impact Micro-Shake + Depth-Aware Cinema Bokeh + Halation + Sensor Grain)
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
  return clamp(uBaseSoft + uNearBlur * max(0.0, invD - invF) + uFarBlur * max(0.0, invF - invD), uBaseSoft, uMaxCoc);
}

void main() {
  vec2 uvRaw = gl_FragCoord.xy / uRes;
  vec2 uv = clamp((uvRaw - 0.5) * 0.994 + 0.5 + uShakePx / uRes, vec2(0.001), vec2(0.999));
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
    if (lum > 0.78) halation += cj * (exp(-r * r / 26.0) * (lum - 0.78) * mj.r);
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
  res += halation * vec3(0.96, 0.99, 1.02) * 0.058;

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


def build_shatter_shards(cam_path, pane_center, occ_boxes, n_shards=280, n_micro=560, seed=2031):
    """Build 840 sharp 3D polygonal glass shards starting right on the fracture plane (y = pane_y) and exploding
    violently across the living room toward & around the advancing camera, freezing in bullet-time, then rewinding back!"""
    rng = np.random.default_rng(seed)
    pc = np.asarray(pane_center, np.float64)
    shards = []

    def clear_cam_and_walls(pos, bound_r):
        p = pos.copy()
        p[2] = float(np.clip(p[2], 0.14, 2.65))
        p[1] = float(np.clip(p[1], -5.75, 1.55))
        max_x = 2.15 if p[1] < 0.90 else 0.88
        p[0] = float(np.clip(p[0], -max_x, max_x))
        # Clear camera pinhole across entire glide so foreground shards sweep past lens in creamy bokeh!
        min_c = 0.18 + bound_r * 1.45
        dists = np.linalg.norm(p[None, :] - cam_path, axis=1)
        idx = int(np.argmin(dists))
        if dists[idx] < min_c:
            v = p - cam_path[idx]
            xz = np.array([v[0], v[2]])
            nrm = float(np.linalg.norm(xz))
            xz = xz / nrm if nrm > 1e-4 else np.array([math.cos(rng.uniform(0, 6.28)), math.sin(rng.uniform(0, 6.28))])
            push = (min_c - dists[idx]) + 0.05
            p[0] = float(np.clip(p[0] + xz[0] * push, -max_x, max_x))
            p[2] = float(np.clip(p[2] + xz[1] * push, 0.14, 2.65))
        return p

    # 1. 280 irregular sharp-edged Voronoi plate shards spanning the exact vertical fracture wall at y = pc[1]!
    # Note: local glass plate lies in XY plane (normal along +Z), so at rest on vertical wall y=const, rx = pi/2!
    for k in range(n_shards):
        # Radial jittered distribution across the 4.4m x 2.5m fracture wall (denser & finer near impact center!)
        r_u = float(rng.random() ** 0.72) * 2.15
        ang = float(rng.uniform(0.0, 2.0 * math.pi))
        sx = pc[0] + math.cos(ang) * r_u * 1.15
        sz = pc[2] + math.sin(ang) * r_u * 0.62
        sz = float(np.clip(sz, 0.22, 2.62))
        sx = float(np.clip(sx, -2.30, 2.30))
        p0 = np.array([sx, pc[1], sz], np.float64)

        # Smaller jagged shards near impact center, larger architectural plate shards further out!
        rad = float(np.exp(rng.uniform(math.log(0.045), math.log(0.165))) * (0.65 + 0.55 * min(r_u / 1.8, 1.0)))
        hz = float(rng.uniform(0.006, 0.014))
        asp = float(rng.uniform(0.62, 1.65))
        bound_r = rad * max(asp, 1.0) * 1.10 + hz

        # Explosive velocity shoots predominantly BACKWARD toward the advancing camera (-Y) + radially outward in XZ
        # so the entire 5m living room fills with suspended frozen 3D glass shards for the camera to fly through!
        out_xz = np.array([sx - pc[0], 0.0, sz - pc[2]], np.float64)
        out_xz /= max(float(np.linalg.norm(out_xz)), 1e-4)
        dy_burst = -float(rng.uniform(0.45, 4.95)) if rng.random() < 0.88 else float(rng.uniform(0.25, 1.45))
        p1_raw = p0 + np.array([out_xz[0] * rng.uniform(0.15, 0.95), dy_burst, out_xz[2] * rng.uniform(0.12, 0.75) - rng.uniform(0.05, 0.35)])
        p1 = clear_cam_and_walls(p1_raw, bound_r)

        shards.append({
            "p0": p0, "p1": p1, "rad": rad, "hz": hz, "asp": asp, "bound_r": bound_r,
            "seed": float(rng.uniform(0.0, 1.0)),
            "is_splint": 0.0,
            "spin": (float(rng.uniform(-4.8, 4.8)), float(rng.uniform(-4.8, 4.8)), float(rng.uniform(-4.8, 4.8))),
            "delay": float(min(r_u * 0.025, 0.07)),  # center shards shatter first, outer ring 1-2 frames later!
        })

    # 2. 560 glittering fine glass micro-splinters & diamond dust erupting from the fracture seams!
    for k in range(n_micro):
        r_u = float(rng.random() ** 0.80) * 2.10
        ang = float(rng.uniform(0.0, 2.0 * math.pi))
        sx = float(np.clip(pc[0] + math.cos(ang) * r_u * 1.15, -2.25, 2.25))
        sz = float(np.clip(pc[2] + math.sin(ang) * r_u * 0.62, 0.20, 2.60))
        p0 = np.array([sx, pc[1], sz], np.float64)

        rad = float(np.exp(rng.uniform(math.log(0.008), math.log(0.042))))
        hz = float(rng.uniform(0.003, 0.008))
        asp = float(rng.uniform(0.55, 2.20))
        bound_r = rad * max(asp, 1.0) * 1.12 + hz

        dy_burst = -float(rng.uniform(0.30, 5.15)) if rng.random() < 0.85 else float(rng.uniform(0.20, 1.65))
        p1_raw = p0 + np.array([rng.normal(0.0, 0.75), dy_burst, rng.normal(-0.08, 0.55)])
        p1 = clear_cam_and_walls(p1_raw, bound_r)

        shards.append({
            "p0": p0, "p1": p1, "rad": rad, "hz": hz, "asp": asp, "bound_r": bound_r,
            "seed": float(rng.uniform(0.0, 1.0)),
            "is_splint": 1.0,
            "spin": (float(rng.uniform(-7.5, 7.5)), float(rng.uniform(-7.5, 7.5)), float(rng.uniform(-7.5, 7.5))),
            "delay": float(min(r_u * 0.028, 0.08)),
        })

    print(f"Air-Glass Shatter: {n_shards} plate shards + {n_micro} micro-splinters = {len(shards)} 3D shards")
    return shards


def main():
    c = load(sys.argv[1] if len(sys.argv) > 1 and not sys.argv[1].startswith("--") else here / "piece_shatter")
    S = scene(c)
    W, H, N, fps = c["W"], c["H"], c["N"], c["fps"]
    f = float(S["f"])
    cfg = c.get("shatter", {})
    stills = [int(x) for x in sys.argv[sys.argv.index("--still") + 1].split(",")] if "--still" in sys.argv else None
    out_dir = sys.argv[sys.argv.index("--out") + 1] if "--out" in sys.argv else "out"
    cam_path = np.asarray(S["cam"], np.float64)

    pane_y = float(cfg.get("pane_y", -0.45))
    pane_center = np.array([-0.06, pane_y, 1.42], np.float64)
    crack_start = int(cfg.get("crack_start", 6))
    shatter_frame = int(cfg.get("shatter_frame", 43))
    freeze_dur_s = float(cfg.get("freeze_dur_s", 1.35))
    rewind_start = int(cfg.get("rewind_start", 154))
    reassemble_frame = int(cfg.get("reassemble_frame", 222))

    # Verified 3 architectural holdout boxes for Pexels 7578547
    occ_boxes = [
        (-3.50, -1.55, -8.00, -4.70, 0.0, 5.0),
        (-4.50, -1.08,  1.22,  3.80, 0.0, 5.0),
        ( 0.96,  4.50,  1.14,  3.80, 0.0, 5.0),
    ]

    shards = build_shatter_shards(
        cam_path, pane_center, occ_boxes,
        n_shards=int(cfg.get("n_shards", 280)),
        n_micro=int(cfg.get("n_micro", 560)),
        seed=int(cfg.get("seed", 2031)),
    )
    caust_idx = sorted(range(len(shards)), key=lambda idx: -shards[idx]["rad"])[:28]

    ctx = moderngl.create_standalone_context(require=330)
    pane_prog = ctx.program(vertex_shader=VS, fragment_shader=PANE_FS)
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
    va_pane = ctx.vertex_array(pane_prog, [(tri, "2f", "p")])
    va_copy2 = ctx.vertex_array(copy2_prog, [(tri, "2f", "p")])
    va_post = ctx.vertex_array(post_prog, [(tri, "2f", "p")])

    quad_vbo = ctx.buffer(np.array([-1, -1, 1, -1, 1, 1, -1, -1, 1, 1, -1, 1], "f4").tobytes())
    inst_buf = ctx.buffer(reserve=len(shards) * 20 * 4)
    va_sh = ctx.vertex_array(sh_prog, [
        (quad_vbo, "2f", "inQuad"),
        (inst_buf, "4f 4f 4f 4f 4f/i", "inPosBound", "inSizeSeed", "inRot0", "inRot1", "inRot2"),
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
    occ_mins = [(b[0], b[2], b[4]) for b in occ_boxes] + [(0.0, 0.0, 0.0)] * (4 - len(occ_boxes))
    occ_maxs = [(b[1], b[3], b[5]) for b in occ_boxes] + [(0.0, 0.0, 0.0)] * (4 - len(occ_boxes))
    pane_prog["uRes"].value = (float(W), float(H)); pane_prog["uF"].value = f
    pane_prog["uSun"].value = tuple(map(float, S["sun"]))
    pane_prog["uOccMin"].value = occ_mins[:4]; pane_prog["uOccMax"].value = occ_maxs[:4]; pane_prog["uNumOcc"].value = len(occ_boxes)
    pane_prog["uPaneCenter"].value = tuple(map(float, pane_center))

    sh_prog["uRes"].value = (float(W), float(H)); sh_prog["uF"].value = f
    if "uSun" in sh_prog:
        sh_prog["uSun"].value = tuple(map(float, S["sun"]))
    sh_prog["uIor"].value = float(cfg.get("ior", 1.52))
    sh_prog["uDisp"].value = float(cfg.get("dispersion", 0.020))
    sh_prog["uProxyDist"].value = 6.2

    post_prog["uRes"].value = (float(W), float(H))
    post_prog["uFocusDist"].value = 2.8
    post_prog["uNearBlur"].value = 4.6
    post_prog["uFarBlur"].value = 4.8
    post_prog["uBaseSoft"].value = 0.68
    post_prog["uMaxCoc"].value = 11.0
    post_prog["uGrainAmt"].value = 0.012
    post_prog["uBlackFloor"].value = 0.017

    def smooth01(x):
        x = max(0.0, min(1.0, x))
        return x * x * (3.0 - 2.0 * x)

    def render_one(i):
        pl = frame(c, i)
        t_sec = i / fps
        ro = cam_path[i]; R = np.asarray(S["rot"][i], np.float64)
        Rt = tuple(R.astype(np.float32).T.flatten())

        # Determine 3-Act timeline state:
        # Act 1 (6..42): Spiderweb fracture grows across vertical pane
        # Act 2 (43..153): Explosive burst into 840 3D shards -> bullet-time freeze (0 drift!)
        # Act 3 (154..222): Anticipation tremble -> Time Rewind whipping shards back into pane!
        # Healing (222..239): Spiderweb cracks zip closed and disappear cleanly into pure air!
        pane_vis, crack_rad, crack_str = 0.0, 0.0, 0.0
        shards_active = False
        shake = np.zeros(2, np.float64)

        if crack_start <= i < shatter_frame + 3:
            u_c = (i - crack_start) / max(float(shatter_frame - crack_start), 1.0)
            pane_vis = 1.0 if i < shatter_frame else max(0.0, 1.0 - (i - shatter_frame) / 3.0)
            crack_rad = 0.15 + 2.85 * (min(u_c, 1.0) ** 0.72)
            crack_str = min(1.0, 0.35 + 0.75 * u_c)
        elif reassemble_frame - 3 <= i <= N - 1:
            u_h = (i - reassemble_frame) / max(float(N - 1 - reassemble_frame), 1.0)
            if i < reassemble_frame:
                pane_vis = smooth01((i - (reassemble_frame - 3)) / 3.0)
                crack_rad = 2.95
                crack_str = 1.0
            else:
                pane_vis = max(0.0, 1.0 - smooth01(u_h * 1.15))
                crack_rad = max(0.0, 2.95 * (1.0 - smooth01(u_h)))
                crack_str = max(0.0, 1.0 - u_h)

        if shatter_frame <= i <= reassemble_frame + 1:
            shards_active = True

        # Micro camera kick on shatter impact (f=43) AND reassemble snap (f=222)!
        for tf, amp0 in [(shatter_frame, 5.2), (reassemble_frame, 3.8)]:
            dt = (i - tf) / fps
            if 0.0 <= dt < 0.26:
                amp = amp0 * math.exp(-dt * 15.0)
                shake[0] += amp * math.sin(dt * 96.0 + tf)
                shake[1] += amp * math.cos(dt * 114.0 + tf) * 0.7

        rows_s = []
        caust_list = []
        if shards_active:
            for idx, sh in enumerate(shards):
                if i < rewind_start:
                    # Act 2: Explosive burst -> bullet-time freeze (w: 0 -> 1.0, 0 drift once frozen!)
                    age = (i - shatter_frame) / fps - sh["delay"]
                    if age <= 0.0:
                        continue
                    x = min(max(age / freeze_dur_s, 0.0), 1.0)
                    w = 1.0 - (1.0 - x) ** 3.6
                    tremble = np.zeros(3)
                    op = min(1.0, age / 0.04)
                else:
                    # Act 3: Anticipation tremble (f=154..163) -> crisp reverse ballistic Time Rewind (164..222)!
                    if i < rewind_start + 10:
                        u_tr = (i - rewind_start) / 10.0
                        w = 1.0
                        amp_t = 0.012 * u_tr
                        tremble = np.array([
                            amp_t * math.sin(i * 3.1 + idx),
                            amp_t * math.cos(i * 2.7 + idx * 1.3),
                            amp_t * math.sin(i * 3.7 + idx * 0.7),
                        ])
                        op = 1.0
                    else:
                        u_rev = (i - (rewind_start + 10)) / max(float(reassemble_frame - (rewind_start + 10)), 1.0)
                        u_rev = min(max(u_rev, 0.0), 1.0)
                        # Accelerates smoothly backward and SNAPS crisply into locked wall!
                        w = max(0.0, 1.0 - (u_rev ** 2.35))
                        tremble = np.zeros(3)
                        op = 1.0 if i <= reassemble_frame - 2 else max(0.0, 1.0 - (i - (reassemble_frame - 2)) / 3.0)

                p_now = sh["p0"] + (sh["p1"] - sh["p0"]) * w + tremble
                # At w=0, shard lies flat in the vertical plane y=const (rx = pi/2); as w grows, it tumbles in 3D!
                rx = math.pi * 0.5 + sh["spin"][0] * w
                ry = sh["spin"][1] * w
                rz = sh["spin"][2] * w
                Rloc = euler_to_mat(rx, ry, rz)
                Rt_loc = Rloc.T
                if idx in caust_idx:
                    caust_list.append((float(p_now[0]), float(p_now[1]), float(p_now[2]), float(sh["rad"] * op)))
                z_cam = float(((p_now - ro) @ R)[2])
                if z_cam > 0.06 and op > 0.02:
                    rows_s.append((z_cam, [
                        float(p_now[0]), float(p_now[1]), float(p_now[2]), float(sh["bound_r"]),
                        float(sh["rad"]), float(sh["hz"]), float(sh["asp"]), float(sh["seed"]),
                        float(Rt_loc[0,0]), float(Rt_loc[0,1]), float(Rt_loc[0,2]), float(op),
                        float(Rt_loc[1,0]), float(Rt_loc[1,1]), float(Rt_loc[1,2]), float(sh["is_splint"]),
                        float(Rt_loc[2,0]), float(Rt_loc[2,1]), float(Rt_loc[2,2]), 1.0,
                    ]))

        n_caust = len(caust_list)
        while len(caust_list) < 28:
            caust_list.append((0.0, 0.0, 0.0, 0.0))

        # Pass 1: Cracking/Healing Air-Glass Pane + Floor Glints + 3-Box Architectural Holdouts
        plate_t.write(pl[:, :, ::-1].tobytes())
        plate_t.use(0)
        pane_prog["uPlate"].value = 0
        pane_prog["uRo"].value = tuple(map(float, ro)); pane_prog["uCam"].value = Rt
        if "uTime" in pane_prog:
            pane_prog["uTime"].value = float(t_sec)
        pane_prog["uPaneVis"].value = float(pane_vis)
        pane_prog["uCrackRad"].value = float(crack_rad)
        pane_prog["uCrackStr"].value = float(crack_str)
        pane_prog["uCaustShards"].value = caust_list[:28]
        pane_prog["uNumCaust"].value = n_caust
        fbo_hero.use()
        ctx.disable(moderngl.BLEND)
        va_pane.render(moderngl.TRIANGLES)

        fbo_scene.use()
        t_hero.use(0); t_meta0.use(1)
        copy2_prog["uSrc0"].value = 0; copy2_prog["uSrc1"].value = 1
        va_copy2.render(moderngl.TRIANGLES)

        # Pass 2: 840 Tumbling / Frozen / Rewinding 3D Polygonal Glass Shards!
        rows_s.sort(key=lambda x: -x[0])
        if rows_s:
            data = np.array([r[1] for r in rows_s], dtype=np.float32)
            inst_buf.write(data.tobytes())
            t_hero.use(0); t_depth.use(1)
            sh_prog["uBase"].value = 0; sh_prog["uDepth"].value = 1
            sh_prog["uRo"].value = tuple(map(float, ro)); sh_prog["uCam"].value = Rt
            ctx.enable(moderngl.BLEND)
            ctx.blend_func = (moderngl.SRC_ALPHA, moderngl.ONE_MINUS_SRC_ALPHA)
            va_sh.render(moderngl.TRIANGLES, instances=len(rows_s))
            ctx.disable(moderngl.BLEND)

        # Pass 3: Camera Optical Finish (impact shake + thin-lens bokeh + halation + sensor grain)
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
            if i == stills[len(stills) // 2]:
                hero = comp
            t_im = lambda im, s: label(cv2.resize(im, (960, 540)), s)
            rows.append(np.hstack([t_im(pl, f"plate {i}"), t_im(comp, f"air shatter {i}")]))
        out_p = work(c, "still_shatter_sheet.jpg")
        cv2.imwrite(str(out_p), np.vstack(rows), [cv2.IMWRITE_JPEG_QUALITY, 92])
        if hero is not None:
            cv2.imwrite(str(work(c, "still_shatter_hero.jpg")), hero, [cv2.IMWRITE_JPEG_QUALITY, 94])
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
            print(f"shatter frame {i}/{N} ({time.time() - ts:.1f}s)", flush=True)
    encode(c, out_dir=out_dir)


if __name__ == "__main__":
    main()
