"""3D Camera-Projection-Mapped Minecraft World Objects Attached to Real Nature & Architecture (Native OpenGL 3.3 PBR).

    python render_mc_world.py --scene park_fountain --still 0,108,215
    python render_mc_world.py --scene arch_hall --still 15,135,245
    python render_mc_world.py --scene metro_wall --still 0,112,224
    python render_mc_world.py --scene park_fountain --render

Uses real solved 6-DOF COLMAP / Homography cameras to projection-map & anchor authentic beveled 3D Minecraft
objects, structures, wall-mounted glowing torches, Nether Portals, voxel trees, & glowing beacons directly
onto real buildings, walls, pillars, and nature park plazas with:
- Real 2048x2048 Soft PCF Shadow Map + Ground Contact AO on z=0
- Real Emissive Glow & Local Point-Light Spill (warm torches/furnaces, purple Nether Portal, cyan Beacon)
- Live-Plate IBL + Linear PBR + ACES Filmic + Plate-Matched OLPF & Sensor Grain
- MediaPipe Person Depth Occlusion (when scene includes a foreground subject, e.g. metro_wall)
- ZERO text/HUD overlays on all Composite & Compare MP4 outputs.
"""
import argparse, json, math, shutil, subprocess, sys, time
from pathlib import Path
import numpy as np, cv2, moderngl

here = Path(__file__).resolve().parent
ROOT = here.parents[1]
sys.path.insert(0, str(here))
sys.path.insert(0, str(here.parent / "shader_live" / "pipeline"))
from slv import load, work, frame, scene  # noqa: E402
from mc_world_assets import build_all_world_assets  # noqa: E402


def find_ffmpeg():
    tl = ROOT / "tools.local.json"
    if tl.exists():
        d = json.loads(tl.read_text(encoding="utf-8"))
        p = d.get("ffmpeg") or (d.get("tools") or {}).get("ffmpeg")
        if p and Path(p).exists():
            return str(p)
    return shutil.which("ffmpeg") or "ffmpeg"


# ==============================================================================
#  3 CURATED NATURE & ARCHITECTURE MINECRAFT WORLD PROJECTION SCENES
# ==============================================================================
# Item tuple: (asset_key, x, y, z, scale, yaw_deg)

SCENES = {
    "park_fountain": {
        "title": "Minecraft_Proj_Nature_Park",
        "piece_dir": ROOT / "examples/turing_growth/piece_turing",
        "sun_dir": (-0.52, -0.45, 0.72),
        "sun_rgb": (1.14, 1.08, 0.96),
        "sun_gain": 3.05,
        "sky_rgb": (0.28, 0.36, 0.48),
        "gnd_rgb": (0.24, 0.19, 0.17),
        "shad_str": 0.62,
        "shad_center": (0.0, 0.0, 0.5),
        "shad_ext": 16.5,
        "wall_y": None,
        "use_person_mask": False,
        "items": [
            # Giant Glowing Diamond-Gold Beacon Monument rising out of real circular stone fountain center!
            ("beacon_pyramid",   0.0,  0.0,  0.10, 1.15,   0.0),
            # Classic Minecraft Village Oak & Cobblestone Cottage planted firmly on left red-brick plaza!
            ("village_cottage", -9.2, -1.6,  0.00, 0.88,  14.0),
            # 5 Full 3D Blocky Minecraft Oak Trees rooted right alongside real green park trees!
            ("oak_tree",        -7.4,  5.2,  0.00, 0.95,  15.0),
            ("oak_tree",         8.8,  3.8,  0.00, 1.00, -20.0),
            ("oak_tree",         9.4, -3.6,  0.00, 0.92,  35.0),
            ("oak_tree",        -6.0, -5.8,  0.00, 0.88, -10.0),
            ("oak_tree",         4.4,  7.4,  0.00, 0.90,  45.0),
            # Glowing Obsidian Nether Portal standing beside circular stone fountain rim!
            ("nether_portal",   -5.8,  2.0,  0.00, 0.98,  28.0),
            # Right fountain-rim crafting & ore camp!
            ("crafting_table",   5.8, -0.4,  0.00, 1.05, -18.0),
            ("chest",            6.9, -0.7,  0.00, 1.05, -105.0),
            ("furnace",          6.2, -1.6,  0.00, 1.05, -25.0),
            ("lantern",          5.8, -0.4,  1.05, 0.95,   0.0),
            # Stacked TNT + Diamond/Gold Ore blocks on northeast fountain rim!
            ("tnt_block",        5.4,  2.5,  0.00, 0.98,  12.0),
            ("tnt_block",        5.4,  2.5,  0.98, 0.98,  24.0),
            ("diamond_ore",      6.4,  2.8,  0.00, 0.98,  -8.0),
            ("gold_ore",         6.4,  1.8,  0.00, 0.98,  18.0),
            # Enchanting Table + Bookshelves + Grass/Emerald cluster on south-west rim!
            ("enchanting_table",-5.2, -2.8,  0.00, 1.05,  22.0),
            ("bookshelf",       -6.3, -3.3,  0.00, 0.98,  18.0),
            ("grass_block",     -4.4, -3.6,  0.00, 0.98,   5.0),
            ("emerald_ore",     -4.4, -3.6,  0.98, 0.98,  15.0),
            # Cardinal glowing lantern posts hugging the stone fountain rim!
            ("lantern",          4.9,  0.2,  0.00, 1.05,   0.0),
            ("lantern",         -4.9,  0.0,  0.00, 1.05,   0.0),
            ("lantern",          0.0,  4.9,  0.00, 1.05,   0.0),
            ("lantern",          0.0, -4.9,  0.00, 1.05,   0.0),
        ],
        "lights": [
            # (x, y, z, r, g, b, radius, intensity)
            ( 0.0,  0.0, 2.2, 0.45, 0.96, 1.00, 5.8, 0.55),  # Beacon cyan glow on fountain
            (-5.8,  2.0, 2.0, 0.72, 0.20, 1.00, 4.5, 0.48),  # Nether portal violet spill
            ( 6.2, -1.6, 0.6, 1.00, 0.58, 0.16, 3.8, 0.42),  # Furnace & lantern warm spill
        ],
    },

    "arch_hall": {
        "title": "Minecraft_Proj_Arch_Hall",
        "piece_dir": ROOT / "examples/liquid_glass/piece_glass",
        "sun_dir": (-0.62, -0.28, 0.73),
        "sun_rgb": (1.12, 1.04, 0.92),
        "sun_gain": 2.45,
        "sky_rgb": (0.42, 0.44, 0.46),
        "gnd_rgb": (0.36, 0.27, 0.18),   # warm oak hardwood bounce!
        "shad_str": 0.48,
        "shad_center": (0.2, -1.5, 1.0),
        "shad_ext": 9.5,
        "wall_y": None,
        "use_person_mask": False,
        "items": [
            # Foreground right hardwood hall cluster (camera flies right past in first 4s with huge 6-DOF parallax!)
            ("crafting_table",   1.38, -6.20, 0.00, 0.68, -165.0),
            ("chest",            1.42, -5.42, 0.00, 0.68,  -95.0),
            ("lantern",          1.38, -6.20, 0.68, 0.64,    0.0),
            # Mid-left corridor cluster along the sunlit hardwood floor!
            ("furnace",         -1.35, -4.10, 0.00, 0.68,  110.0),
            ("bookshelf",       -1.38, -3.38, 0.00, 0.68,   90.0),
            ("tnt_block",       -1.30, -2.66, 0.00, 0.66,   78.0),
            ("diamond_ore",     -1.35, -4.10, 0.68, 0.64,   98.0),
            # Left column base ore + glowing lantern resting solidly on mossy brick!
            ("emerald_ore",     -1.42, -0.45, 0.00, 0.56,    6.0),
            ("mossy_brick",     -1.45,  0.12, 0.00, 0.56,   -4.0),
            ("lantern",         -1.45,  0.12, 0.56, 0.58,    0.0),
            # Right enchanting nook + glowing lantern resting solidly on gold ore!
            ("enchanting_table", 1.55, -1.30, 0.00, 0.72, -140.0),
            ("bookshelf",        2.15, -0.60, 0.00, 0.66, -120.0),
            ("gold_ore",         2.15, -0.60, 0.66, 0.62, -108.0),
            ("lantern",          2.15, -0.60, 1.28, 0.56,    0.0),
            # Hero Centerpiece down the sunlit living hall: FULL Framed Glowing Obsidian Nether Portal + Potted Oak Tree!
            ("nether_portal",   -0.25,  5.60, 0.00, 0.66,  180.0),
            ("grass_block",     -2.15,  4.85, 0.00, 0.60,   12.0),
            ("oak_tree",        -2.15,  4.85, 0.60, 0.34,   25.0),
            ("chest",            1.35,  4.10, 0.00, 0.66, -165.0),
            ("tnt_block",        2.02,  4.30, 0.00, 0.64,  -15.0),
        ],
        "lights": [
            (-0.25,  5.35, 1.5, 0.76, 0.22, 1.00, 4.5, 0.62),  # Nether portal violet glow on hardwood!
            (-1.38, -0.75, 1.9, 1.00, 0.62, 0.18, 2.8, 0.45),  # Left pillar torch glow
            ( 2.05, -1.55, 1.9, 1.00, 0.62, 0.18, 2.8, 0.45),  # Right pillar torch glow
            ( 1.38, -6.20, 1.0, 1.00, 0.74, 0.28, 2.5, 0.40),  # Foreground lantern warm spill
        ],
    },

    "metro_wall": {
        "title": "Minecraft_Proj_Metro_Wall",
        "piece_dir": ROOT / "examples/liquid_glass/piece_human",
        "sun_dir": (0.18, -0.32, 0.93),  # crisp overhead metro station fluorescent strips!
        "sun_rgb": (0.98, 1.00, 1.06),
        "sun_gain": 2.30,
        "sky_rgb": (0.38, 0.38, 0.45),
        "gnd_rgb": (0.28, 0.28, 0.30),
        "shad_str": 0.56,
        "shad_center": (0.8, 1.2, 1.1),
        "shad_ext": 5.5,
        "wall_y": 2.98,
        "use_person_mask": True,
        "person_depth_m": 4.45,
        "items": [
            # STUCK FLUSH AGAINST PURPLE TILED METRO WALL (y = 2.85..2.98m, cleanly framed inside purple wall!)
            ("nether_portal",    1.35,  2.84, 0.06, 0.42,  180.0),
            ("wall_torch",       0.25,  2.90, 1.68, 0.65,  180.0),
            ("wall_torch",       2.48,  2.90, 1.68, 0.65,  180.0),
            # Protruding Minecraft ore & mossy brick blocks breaking straight out of the purple metro wall!
            ("diamond_ore",      0.18,  2.78, 0.88, 0.46,    4.0),
            ("mossy_brick",     -0.30,  2.80, 0.74, 0.44,   -5.0),
            ("gold_ore",         2.52,  2.78, 0.94, 0.46,    6.0),
            ("emerald_ore",      2.92,  2.80, 0.66, 0.42,   -4.0),
            # GROUNDED ON THE GRAY METRO PLATFORM FLOOR (inside yellow safety line, y in [-0.30, +0.20], z=0!)
            ("crafting_table",   1.22,  0.14, 0.00, 0.54, -168.0),
            ("chest",            1.80,  0.16, 0.00, 0.54, -162.0),
            ("lantern",          1.22,  0.14, 0.54, 0.50,    0.0),
            ("tnt_block",        2.36,  0.18, 0.00, 0.52,   12.0),
            ("furnace",          2.36,  0.18, 0.52, 0.50, -158.0),
            # Resting cleanly on platform tiles right beside the left concrete pillar!
            ("bookshelf",       -1.32, -0.32, 0.00, 0.52, -172.0),
            ("diamond_ore",     -1.32, -0.32, 0.52, 0.48,  -10.0),
            ("hanging_lantern", -0.62,  0.10, 1.88, 0.50,    0.0),
        ],
        "lights": [
            ( 1.35,  2.65, 1.55, 0.78, 0.22, 1.00, 3.6, 0.65),  # Nether portal purple glow on metro wall!
            ( 0.25,  2.70, 2.35, 1.00, 0.62, 0.16, 2.5, 0.55),  # Left wall torch glow
            ( 2.48,  2.70, 2.35, 1.00, 0.62, 0.16, 2.5, 0.55),  # Right wall torch glow
            ( 1.22,  0.14, 0.80, 1.00, 0.74, 0.28, 2.0, 0.42),  # Platform lantern glow
        ],
    },
}


# ==============================================================================
#  SHADERS: 2048x2048 SHADOW MAP + LINEAR PBR + EMISSIVE GLOW + WALL/FLOOR SPILL
# ==============================================================================

SHADOW_VS = """#version 330
in vec3 inPos;
uniform mat4 uLightVP;
uniform mat4 uModel;
void main() { gl_Position = uLightVP * uModel * vec4(inPos, 1.0); }
"""
SHADOW_FS = "#version 330\nvoid main() {}"

MESH_VS = """#version 330
in vec3 inPos;
in vec3 inNrm;
in vec2 inUv;
in vec3 inCol;
in vec3 inPbr;

uniform mat4 uProj, uView, uModel, uLightVP;
uniform mat3 uNormalMat;

out vec3 vWorldPos, vWorldNrm, vCol, vPbr;
out vec2 vUv;
out float vCamDepth;
out vec4 vLightSpace;

void main() {
  vec4 wp = uModel * vec4(inPos, 1.0);
  vWorldPos = wp.xyz;
  vWorldNrm = normalize(uNormalMat * inNrm);
  vUv = inUv; vCol = inCol; vPbr = inPbr;
  vec4 vp = uView * wp;
  vCamDepth = -vp.z;
  vLightSpace = uLightVP * wp;
  gl_Position = uProj * vp;
}
"""

MESH_FS = """#version 330
in vec3 vWorldPos, vWorldNrm, vCol, vPbr;
in vec2 vUv;
in float vCamDepth;
in vec4 vLightSpace;

uniform sampler2D uShadowMap, uPlate, uPersonMask;
uniform vec2 uRes;
uniform vec3 uRo, uSun, uSunCol, uSkyCol, uGndCol;
uniform float uTime, uPersonDepth;
uniform int uUseMask;

layout(location=0) out vec4 oCol;
layout(location=1) out vec4 oMeta;

float hash21(vec2 p){ p=fract(p*vec2(123.34,456.21)); p+=dot(p,p+45.32); return fract(p.x*p.y); }
float noise2(vec2 p){ vec2 i=floor(p),f=fract(p); f=f*f*(3.0-2.0*f);
  return mix(mix(hash21(i),hash21(i+vec2(1,0)),f.x), mix(hash21(i+vec2(0,1)),hash21(i+vec2(1,1)),f.x), f.y); }
vec3 acesFilm(vec3 x){ return clamp((x*(2.51*x+0.03))/(x*(2.43*x+0.59)+0.14),0.0,1.0); }

float calcShadow(vec4 lsPos, vec3 N, vec3 L) {
  vec3 proj = lsPos.xyz / max(lsPos.w, 1e-5) * 0.5 + 0.5;
  if (proj.z > 1.0 || proj.x < 0.01 || proj.x > 0.99 || proj.y < 0.01 || proj.y > 0.99) return 1.0;
  float bias = max(0.0025 * (1.0 - dot(N, L)), 0.0008);
  float lit = 0.0;
  vec2 tex = vec2(1.0 / 2048.0);
  for (int x = -1; x <= 1; x++) {
    for (int y = -1; y <= 1; y++) {
      float pcfD = texture(uShadowMap, proj.xy + vec2(float(x), float(y)) * tex * 1.25).r;
      lit += (proj.z - bias <= pcfD) ? 1.0 : 0.0;
    }
  }
  return lit / 9.0;
}

vec3 samplePlateIBL(vec3 R) {
  vec2 uvEnv = vec2(clamp(0.5 + R.x * 0.42, 0.02, 0.98), clamp(0.52 - R.z * 0.48, 0.02, 0.98));
  return pow(texture(uPlate, uvEnv).rgb, vec3(2.2)) * mix(0.45, 1.05, smoothstep(-0.5, 0.2, R.z));
}

void main() {
  float vis = 1.0;
  if (uUseMask > 0) {
    vec2 uvTop = vec2(gl_FragCoord.x / uRes.x, 1.0 - gl_FragCoord.y / uRes.y);
    float pMask = texture(uPersonMask, clamp(uvTop, vec2(0.001), vec2(0.999))).r;
    if (vCamDepth > uPersonDepth + 0.05 && pMask > 0.45) discard;
    vis = (vCamDepth > uPersonDepth + 0.05) ? (1.0 - smoothstep(0.12, 0.55, pMask)) : 1.0;
  }

  /* Emissive materials (pbr.z < 0: Nether Portal animated vortex, Torch flame, Furnace fire, Beacon beam!) */
  if (vPbr.z < -0.01) {
    float em = -vPbr.z;
    float pulse = 0.88 + 0.12 * sin(uTime * 6.5 + vWorldPos.x * 4.0 + vWorldPos.z * 5.0);
    vec3 eLin = pow(max(vCol, vec3(0.002)), vec3(2.2)) * em * pulse;
    vec3 mapped = acesFilm(eLin);
    vec3 srgb = pow(clamp(mapped, 0.0, 1.0), vec3(1.0 / 2.2));
    oCol = vec4(srgb, vis);
    oMeta = vec4(vis, vis / max(vCamDepth, 0.2), 0.0, 1.0);
    return;
  }

  vec3 N = normalize(vWorldNrm);
  vec3 V = normalize(uRo - vWorldPos);
  vec3 L = normalize(uSun);
  vec3 baseSrgb = vCol;
  float rough = vPbr.x, metal = vPbr.y, vAO = vPbr.z;

  float micro = noise2(vWorldPos.xy * 55.0 + vWorldPos.z * 48.0);
  baseSrgb *= (0.94 + 0.12 * micro);
  vec2 edgeD = min(vUv, 1.0 - vUv);
  vAO *= mix(0.88, 1.0, smoothstep(0.0, 0.07, min(edgeD.x, edgeD.y)));

  vec3 baseLin = pow(max(baseSrgb, vec3(0.002)), vec3(2.2));
  vec3 H = normalize(L + V);
  float NdotL = max(dot(N, L), 0.0);
  float NdotV = max(dot(N, V), 0.001);
  float NdotH = max(dot(N, H), 0.0);
  float shad = calcShadow(vLightSpace, N, L);

  vec3 hemi = mix(uGndCol, uSkyCol, clamp(N.z * 0.5 + 0.5, 0.0, 1.0));
  float gndAO = mix(0.36, 1.0, smoothstep(0.0, 0.35, vWorldPos.z));
  float totalAO = clamp(vAO * gndAO, 0.18, 1.0);

  vec3 F0 = mix(vec3(0.04), baseLin, metal);
  vec3 F = F0 + (1.0 - F0) * pow(1.0 - NdotV, 5.0);
  float a2 = max(rough * rough, 0.005);
  float denom = NdotH * NdotH * (a2 - 1.0) + 1.0;
  float specSun = min((a2 / (3.14159 * denom * denom + 1e-5)) * 0.24 * NdotL, 5.5);

  vec3 envRefl = samplePlateIBL(reflect(-V, N));
  float reflPow = pow(1.0 - rough, 1.85);
  vec3 diffAlb = baseLin * (1.0 - metal * 0.88);
  vec3 linCol = diffAlb * (uSunCol * NdotL * mix(0.18, 1.0, shad) + hemi * totalAO)
              + F * uSunCol * specSun * shad
              + F * envRefl * reflPow * (0.38 + 0.62 * totalAO);

  vec3 mapped = acesFilm(linCol * 0.88);
  vec3 srgb = pow(clamp(mapped, 0.0, 1.0), vec3(1.0 / 2.2));
  oCol = vec4(srgb, vis);
  oMeta = vec4(vis, vis / max(vCamDepth, 0.2), 0.0, 1.0);
}
"""

VS_FSQ = "#version 330\nin vec2 p; void main(){ gl_Position = vec4(p, 0.0, 1.0); }"

GROUND_FS = """#version 330
uniform sampler2D uPlate, uShadowMap, uPersonMask;
uniform vec2 uRes;
uniform float uF, uShadStr, uWallY, uTime;
uniform vec3 uRo;
uniform mat3 uCam;
uniform mat4 uLightVP;
uniform int uHasWall, uUseMask, uNumLights;
uniform vec4 uLightPosR[8];   /* xyz=world pos, w=radius */
uniform vec4 uLightColI[8];   /* rgb=color, w=intensity */

layout(location=0) out vec4 oCol;
layout(location=1) out vec4 oMeta;

void main() {
  vec2 px0 = vec2(gl_FragCoord.x, uRes.y - gl_FragCoord.y);
  vec2 uv0 = px0 / uRes;
  vec3 col = texture(uPlate, clamp(uv0, vec2(0.002), vec2(0.998))).rgb;
  vec3 rd = normalize(uCam * vec3((px0 - 0.5 * uRes) / uF, 1.0));
  float notPerson = 1.0;
  if (uUseMask > 0) {
    float pMask = texture(uPersonMask, clamp(uv0, vec2(0.001), vec2(0.999))).r;
    notPerson = 1.0 - smoothstep(0.15, 0.65, pMask);
  }
  float cgA = 0.0, cgInvD = 0.0;

  /* Ray intersect floor z=0 AND optional back architectural wall y=uWallY for real shadows & emissive light spill! */
  float tf = (rd.z < -0.004) ? (-uRo.z / rd.z) : 1e5;
  float tw = (uHasWall > 0 && rd.y > 0.004) ? ((uWallY - uRo.y) / rd.y) : 1e5;
  float tHit = min(tf, tw);
  bool isFloor = (tf <= tw);

  if (tHit > 0.2 && tHit < 90.0) {
    vec3 X = uRo + rd * tHit;
    if (isFloor) {
      /* Real 2048x2048 PCF floor shadow */
      vec4 ls = uLightVP * vec4(X, 1.0);
      vec3 proj = ls.xyz / max(ls.w, 1e-5) * 0.5 + 0.5;
      float occ = 0.0;
      if (proj.x > 0.01 && proj.x < 0.99 && proj.y > 0.01 && proj.y < 0.99 && proj.z < 1.0) {
        vec2 tex = vec2(1.0 / 2048.0);
        for (int i = -2; i <= 2; i++) {
          for (int j = -2; j <= 2; j++) {
            float dMap = texture(uShadowMap, proj.xy + vec2(float(i), float(j)) * tex * 1.4).r;
            if (proj.z - 0.0012 > dMap) occ += 1.0;
          }
        }
        occ /= 25.0;
      }
      float sAmt = clamp(occ * uShadStr, 0.0, 0.78) * notPerson;
      if (sAmt > 0.005) {
        col = mix(col, col * vec3(0.80, 0.85, 0.96) * 0.28, sAmt);
        cgA = max(cgA, sAmt * 0.7);
        cgInvD = cgA / max(tHit, 0.3);
      }
    }

    /* Real emissive point-light spill on both floor AND back architectural wall! */
    vec3 spill = vec3(0.0);
    for (int k = 0; k < 8; k++) {
      if (k >= uNumLights) break;
      vec3 lp = uLightPosR[k].xyz;
      float rad = max(uLightPosR[k].w, 0.2);
      float d = length(X - lp);
      if (d < rad) {
        float att = pow(1.0 - d / rad, 2.0) * uLightColI[k].w * (0.92 + 0.08 * sin(uTime * 7.5 + float(k) * 2.1));
        spill += uLightColI[k].rgb * att;
      }
    }
    if (length(spill) > 0.005) {
      vec3 lin = pow(max(col, vec3(0.002)), vec3(2.2));
      lin += spill * notPerson * 0.42;
      col = pow(clamp(lin, 0.0, 1.0), vec3(1.0 / 2.2));
      cgA = max(cgA, clamp(length(spill) * 0.6, 0.0, 0.6));
    }
  }
  oCol = vec4(clamp(col, 0.0, 1.0), 1.0);
  oMeta = vec4(cgA, cgInvD, 0.0, 1.0);
}
"""

POST_FS = """#version 330
uniform sampler2D uScene, uMeta; uniform vec2 uRes;
uniform float uBlackFloor, uGrainAmt, uFrame;
out vec4 oCol;
float hash12(vec2 p){ vec3 p3=fract(vec3(p.xyx)*0.1031); p3+=dot(p3,p3.yzx+33.33); return fract((p3.x+p3.y)*p3.z); }
void main() {
  vec2 uv = gl_FragCoord.xy / uRes;
  vec2 tex = 1.0 / uRes;
  vec3 c0 = texture(uScene, uv).rgb;
  float cgA = clamp(texture(uMeta, uv).r, 0.0, 1.0);
  vec3 res = c0;
  if (cgA > 0.005) {
    vec3 soft = (c0*4.0 + texture(uScene,uv+vec2(tex.x,0)).rgb + texture(uScene,uv-vec2(tex.x,0)).rgb +
                 texture(uScene,uv+vec2(0,tex.y)).rgb + texture(uScene,uv-vec2(0,tex.y)).rgb) / 8.0;
    res = mix(c0, soft, cgA * 0.24);
    vec3 graded = max(res, vec3(uBlackFloor));
    vec2 p1 = gl_FragCoord.xy + vec2(uFrame * 17.3, uFrame * 31.7);
    vec2 p2 = floor(gl_FragCoord.xy * 0.5) + vec2(uFrame * 11.1, uFrame * 23.9);
    float g = (hash12(p1)+hash12(p1+7.13)-1.0)*0.62 + (hash12(p2)+hash12(p2+5.71)-1.0)*0.38;
    float lum = dot(graded, vec3(0.299, 0.587, 0.114));
    graded += vec3(g) * (uGrainAmt * (1.12 - 0.55 * smoothstep(0.55, 0.98, lum)));
    res = mix(res, graded, cgA);
  }
  oCol = vec4(clamp(res, 0.0, 1.0), 1.0);
}
"""


def mat4_perspective_from_opencv(f_px, W, H, near=0.1, far=120.0):
    P = np.zeros((4, 4), dtype=np.float32)
    P[0, 0] = 2.0 * f_px / W
    P[1, 1] = 2.0 * f_px / H
    P[2, 2] = -(far + near) / (far - near)
    P[2, 3] = -2.0 * far * near / (far - near)
    P[3, 2] = -1.0
    return P


def mat4_view_from_colmap(ro, R_c2w):
    flip = np.diag([1.0, -1.0, -1.0])
    R_w2gl = flip @ R_c2w.T
    V = np.eye(4, dtype=np.float32)
    V[:3, :3] = R_w2gl
    V[:3, 3] = -R_w2gl @ ro
    return V


def mat4_ortho_light_vp(sun_dir, center, half_ext, near=0.5, far=45.0):
    L = np.asarray(sun_dir, dtype=np.float64); L /= np.linalg.norm(L) + 1e-9
    eye = np.asarray(center, dtype=np.float64) + L * 20.0
    z_ax = L
    up = np.array([0.0, 0.0, 1.0]) if abs(z_ax[2]) < 0.95 else np.array([0.0, 1.0, 0.0])
    x_ax = np.cross(up, z_ax); x_ax /= np.linalg.norm(x_ax)
    y_ax = np.cross(z_ax, x_ax)
    V = np.eye(4, dtype=np.float64)
    V[0, :3] = x_ax; V[1, :3] = y_ax; V[2, :3] = z_ax; V[:3, 3] = -V[:3, :3] @ eye
    O = np.zeros((4, 4), dtype=np.float64)
    O[0, 0] = 1.0 / half_ext; O[1, 1] = 1.0 / half_ext
    O[2, 2] = -2.0 / (far - near); O[2, 3] = -(far + near) / (far - near); O[3, 3] = 1.0
    return (O @ V).astype(np.float32)


def mat4_model(x, y, z, scale, yaw_deg):
    rad = math.radians(yaw_deg)
    c, s = math.cos(rad), math.sin(rad)
    R = np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]], dtype=np.float32)
    M = np.eye(4, dtype=np.float32)
    M[:3, :3] = R * float(scale)
    M[:3, 3] = (float(x), float(y), float(z))
    return M, R


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scene", required=True, choices=list(SCENES.keys()))
    ap.add_argument("--still", default=None)
    ap.add_argument("--render", action="store_true")
    args = ap.parse_args()

    cfg = SCENES[args.scene]
    c = load(cfg["piece_dir"])
    S = scene(c)
    C = np.asarray(S["cam"], dtype=np.float64)
    R_all = np.asarray(S["rot"], dtype=np.float64)
    f_px = float(S["f"])
    N = len(C)
    fps = float(c["fps"])
    pl0 = frame(c, 0)
    H, W = pl0.shape[:2]

    gray0 = cv2.cvtColor(pl0, cv2.COLOR_BGR2GRAY).astype(np.float32) / 255.0
    black_floor = float(np.clip(np.percentile(gray0, 1.0) * 0.85, 0.008, 0.050))
    hp0 = gray0 - cv2.GaussianBlur(gray0, (0, 0), 1.2)
    grain_amt = float(np.clip(float(np.median(np.abs(hp0 - np.median(hp0)))) * 1.4826 * 1.10, 0.008, 0.022))

    sun = np.asarray(cfg["sun_dir"], dtype=np.float64); sun /= np.linalg.norm(sun) + 1e-9
    light_vp = mat4_ortho_light_vp(sun, cfg["shad_center"], cfg["shad_ext"])
    proj_m = mat4_perspective_from_opencv(f_px, W, H, near=0.1, far=140.0)

    ctx = moderngl.create_standalone_context(require=330)
    shad_prog = ctx.program(vertex_shader=SHADOW_VS, fragment_shader=SHADOW_FS)
    mesh_prog = ctx.program(vertex_shader=MESH_VS, fragment_shader=MESH_FS)
    gnd_prog  = ctx.program(vertex_shader=VS_FSQ, fragment_shader=GROUND_FS)
    post_prog = ctx.program(vertex_shader=VS_FSQ, fragment_shader=POST_FS)

    fsq = ctx.buffer(np.array([-1, -1, 1, -1, -1, 1, -1, 1, 1, -1, 1, 1], dtype=np.float32).tobytes())
    va_gnd  = ctx.vertex_array(gnd_prog,  [(fsq, "2f", "p")])
    va_post = ctx.vertex_array(post_prog, [(fsq, "2f", "p")])

    assets = build_all_world_assets()
    gpu_meshes = {}
    for k, md in assets.items():
        vdata = np.column_stack([md["pos"], md["nrm"], md["uv"], md["col"], md["pbr"]]).astype(np.float32)
        vbo = ctx.buffer(vdata.tobytes())
        ibo = ctx.buffer(md["idx"].astype(np.uint32).tobytes())
        vao_m = ctx.vertex_array(mesh_prog, [(vbo, "3f 3f 2f 3f 3f", "inPos", "inNrm", "inUv", "inCol", "inPbr")], index_buffer=ibo)
        vao_s = ctx.vertex_array(shad_prog, [(vbo, "3f 44x", "inPos")], index_buffer=ibo)
        gpu_meshes[k] = (vao_m, vao_s)

    t_smap = ctx.depth_texture((2048, 2048)); t_smap.filter = (moderngl.LINEAR, moderngl.LINEAR); t_smap.repeat_x = t_smap.repeat_y = False
    fbo_shad = ctx.framebuffer(depth_attachment=t_smap)
    t_sub_col = ctx.texture((W, H), 4, dtype="f4"); t_sub_meta = ctx.texture((W, H), 4, dtype="f4")
    rbo_depth = ctx.depth_renderbuffer((W, H))
    fbo_sub = ctx.framebuffer([t_sub_col, t_sub_meta], depth_attachment=rbo_depth)
    t_final = ctx.texture((W, H), 4, dtype="f4")
    fbo_final = ctx.framebuffer([t_final])
    plate_t = ctx.texture((W, H), 3); plate_t.filter = (moderngl.LINEAR, moderngl.LINEAR); plate_t.repeat_x = plate_t.repeat_y = False
    mask_t = ctx.texture((W, H), 1, dtype="f4"); mask_t.filter = (moderngl.LINEAR, moderngl.LINEAR); mask_t.repeat_x = mask_t.repeat_y = False
    mask_t.write(np.zeros((H, W), dtype=np.float32).tobytes())

    # Pre-bake static 2048x2048 shadow map once (all projection-mapped world objects are rock-solid world-anchored!)
    fbo_shad.use(); ctx.viewport = (0, 0, 2048, 2048); ctx.enable(moderngl.DEPTH_TEST); ctx.disable(moderngl.BLEND)
    ctx.clear(depth=1.0)
    shad_prog["uLightVP"].write(light_vp.T.tobytes())
    for akey, x, y, z, sc, yaw in cfg["items"]:
        M, _ = mat4_model(x, y, z, sc, yaw)
        shad_prog["uModel"].write(M.T.tobytes())
        gpu_meshes[akey][1].render(moderngl.TRIANGLES)

    # Static uniforms
    mesh_prog["uProj"].write(proj_m.T.tobytes())
    mesh_prog["uLightVP"].write(light_vp.T.tobytes())
    mesh_prog["uRes"].value = (float(W), float(H))
    mesh_prog["uSun"].value = tuple(map(float, sun))
    mesh_prog["uSunCol"].value = tuple(float(v * cfg["sun_gain"]) for v in cfg["sun_rgb"])
    mesh_prog["uSkyCol"].value = tuple(map(float, cfg["sky_rgb"]))
    mesh_prog["uGndCol"].value = tuple(map(float, cfg["gnd_rgb"]))
    mesh_prog["uUseMask"].value = 1 if cfg["use_person_mask"] else 0
    mesh_prog["uPersonDepth"].value = float(cfg.get("person_depth_m", 4.5))

    gnd_prog["uRes"].value = (float(W), float(H))
    gnd_prog["uF"].value = f_px
    gnd_prog["uShadStr"].value = float(cfg["shad_str"])
    gnd_prog["uLightVP"].write(light_vp.T.tobytes())
    gnd_prog["uHasWall"].value = 1 if cfg["wall_y"] is not None else 0
    gnd_prog["uWallY"].value = float(cfg["wall_y"] or 0.0)
    gnd_prog["uUseMask"].value = 1 if cfg["use_person_mask"] else 0
    lpos = [(l[0], l[1], l[2], l[6]) for l in cfg["lights"]]
    lcol = [(l[3], l[4], l[5], l[7]) for l in cfg["lights"]]
    gnd_prog["uNumLights"].value = len(lpos)
    gnd_prog["uLightPosR"].value = lpos + [(0.0, 0.0, 0.0, 1.0)] * (8 - len(lpos))
    gnd_prog["uLightColI"].value = lcol + [(0.0, 0.0, 0.0, 0.0)] * (8 - len(lcol))

    post_prog["uRes"].value = (float(W), float(H))
    post_prog["uBlackFloor"].value = black_floor
    post_prog["uGrainAmt"].value = grain_amt

    def render_one(i):
        pl = frame(c, i)
        plate_t.write(pl[:, :, ::-1].tobytes())
        if cfg["use_person_mask"]:
            mp = work(c, "mask", f"{i:04d}.png")
            m_raw = cv2.imread(str(mp), cv2.IMREAD_GRAYSCALE) if mp.exists() else None
            m_f = cv2.GaussianBlur(m_raw.astype(np.float32) / 255.0, (0, 0), 0.9) if m_raw is not None else np.zeros((H, W), np.float32)
            mask_t.write(m_f.astype(np.float32).tobytes())

        ro = C[i]; R = R_all[i]
        Rt = tuple(R.T.flatten().astype(np.float32))
        view_m = mat4_view_from_colmap(ro, R)
        t_sec = i / fps

        # Pass 1: Live Plate + Floor Shadows + Emissive Light Spill on Floor/Wall
        fbo_sub.use(); ctx.viewport = (0, 0, W, H); ctx.disable(moderngl.DEPTH_TEST); ctx.disable(moderngl.BLEND)
        ctx.clear(depth=1.0)
        plate_t.use(0); t_smap.use(1); mask_t.use(2)
        gnd_prog["uPlate"].value = 0; gnd_prog["uShadowMap"].value = 1; gnd_prog["uPersonMask"].value = 2
        gnd_prog["uRo"].value = tuple(map(float, ro)); gnd_prog["uCam"].value = Rt
        gnd_prog["uTime"].value = float(t_sec)
        va_gnd.render(moderngl.TRIANGLES)

        # Pass 2: Projection-Mapped 3D Minecraft World Objects!
        ctx.enable(moderngl.DEPTH_TEST); ctx.enable(moderngl.BLEND)
        ctx.blend_func = (moderngl.SRC_ALPHA, moderngl.ONE_MINUS_SRC_ALPHA)
        t_smap.use(0); plate_t.use(1); mask_t.use(2)
        mesh_prog["uShadowMap"].value = 0; mesh_prog["uPlate"].value = 1; mesh_prog["uPersonMask"].value = 2
        mesh_prog["uView"].write(view_m.T.tobytes()); mesh_prog["uRo"].value = tuple(map(float, ro))
        mesh_prog["uTime"].value = float(t_sec)
        for akey, x, y, z, sc, yaw in cfg["items"]:
            M, N3 = mat4_model(x, y, z, sc, yaw)
            mesh_prog["uModel"].write(M.T.tobytes()); mesh_prog["uNormalMat"].write(N3.T.tobytes())
            gpu_meshes[akey][0].render(moderngl.TRIANGLES)

        # Pass 3: Optical Finish (OLPF + Black Floor + Sensor Grain)
        fbo_final.use(); ctx.viewport = (0, 0, W, H); ctx.disable(moderngl.DEPTH_TEST); ctx.disable(moderngl.BLEND)
        t_sub_col.use(0); t_sub_meta.use(1)
        post_prog["uScene"].value = 0; post_prog["uMeta"].value = 1; post_prog["uFrame"].value = float(i)
        va_post.render(moderngl.TRIANGLES)
        ctx.finish()
        arr = np.frombuffer(t_final.read(), "f4").reshape(H, W, 4)[::-1]
        img = (np.clip(arr[..., :3], 0, 1) * 255 + 0.5).astype(np.uint8)[:, :, ::-1]
        return pl, img

    out_root = here / "out" / args.scene
    work_root = here / "work" / args.scene
    work_root.mkdir(parents=True, exist_ok=True)

    if args.still:
        idxs = [int(x) for x in args.still.split(",")]
        rows = []
        for idx in idxs:
            pl, comp = render_one(idx)
            a = cv2.resize(pl, (640, 360))
            b = cv2.resize(comp, (640, 360))
            cv2.putText(a, f"plate {idx}", (10, 344), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 3)
            cv2.putText(a, f"plate {idx}", (10, 344), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 1)
            cv2.putText(b, f"{args.scene} {idx}", (10, 344), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 3)
            cv2.putText(b, f"{args.scene} {idx}", (10, 344), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (40, 255, 40), 1)
            rows.append(np.hstack([a, b]))
        out_p = work_root / f"still_{args.scene}.jpg"
        cv2.imwrite(str(out_p), np.vstack(rows), [cv2.IMWRITE_JPEG_QUALITY, 92])
        print("stills ->", out_p)
        return

    if args.render:
        comp_dir = work_root / "comp"
        shutil.rmtree(comp_dir, ignore_errors=True)
        comp_dir.mkdir(parents=True)
        ts = time.time()
        for i in range(N):
            _, comp = render_one(i)
            cv2.imwrite(str(comp_dir / f"{i:04d}.png"), comp, [cv2.IMWRITE_PNG_COMPRESSION, 1])
            if i % 50 == 0 or i == N - 1:
                print(f"{args.scene} frame {i}/{N} ({time.time() - ts:.1f}s)", flush=True)

        out_root.mkdir(parents=True, exist_ok=True)
        r = f"{c['fps_num']}/{c['fps_den']}"
        ff = find_ffmpeg()
        nm = cfg["title"]
        tags = ["-pix_fmt", "yuv420p", "-color_primaries", "bt709", "-color_trc", "bt709", "-colorspace", "bt709", "-movflags", "+faststart"]
        x264 = ["-c:v", "libx264", "-profile:v", "high", "-level", "4.1", "-crf", "14", "-preset", "slow"]
        run = lambda a: subprocess.run([ff, "-v", "error", "-y"] + a, check=True)
        run(["-framerate", r, "-i", str(comp_dir / "%04d.png")] + x264 + tags + [str(out_root / f"{nm}_Composite.mp4")])
        run([
            "-framerate", r, "-i", str(comp_dir / "%04d.png"),
            "-framerate", r, "-i", str(work(c, "frames", "%04d.jpg")),
            "-filter_complex",
            "[0]scale=1080:608[a];[1]scale=1080:608[b];[a][b]vstack,pad=1080:1350:0:67:color=black",
        ] + x264 + tags + [str(out_root / f"{nm}_Compare_1080x1350.mp4")])
        print("encoded ->", out_root)


if __name__ == "__main__":
    main()
