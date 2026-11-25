"""Real-World Motion-Captured & Articulated 3D Minecraft Character (Steve) Replacement (Native OpenGL 3.3 PBR).

    python render_steve_mocap.py [piece_dir] --still 0,56,110,165,215     quick look -> work/still_steve_sheet.jpg
    python render_steve_mocap.py [piece_dir] --out out --passes           full render + breakdown passes -> out/*.mp4

Pipeline:
1. Per-frame SIFT-homography-warped AI spotless clean plate removes original dancer & ground shadow cleanly.
2. MediaPipe 3D world landmarks -> Savitzky-Golay denoised, hemisphere-continuous 360-deg hierarchical FK rig
   (11 articulated Minecraft Steve parts with 2-layer 3D voxel relief & exact z=0 ground contact lock).
3. Native OpenGL 3.3 (ModernGL):
   - 2048x2048 Soft PCF Directional Daylight Shadow Map + Dynamic Dual-Foot Contact AO on concrete sidewalk
   - Linear Open-Shade Daylight PBR + Live-Plate IBL + ACES Filmic Tone Mapping
   - 9-Slice Quaternion-SLERP Cinema Motion Blur across fast arm swings, knee kicks, and 360-deg spins
   - Plate-Matched OLPF micro-softening, black floor & animated camera sensor grain (CG/shadow pixels only)
   - Zero text/HUD overlays on Composite & Compare outputs.
"""
import json, math, shutil, subprocess, sys, time
from pathlib import Path
import numpy as np, cv2, moderngl
from scipy.spatial.transform import Rotation as Rot, Slerp

here = Path(__file__).resolve().parent
ROOT = here.parents[1]
sys.path.insert(0, str(here))
sys.path.insert(0, str(here.parent / "shader_live" / "pipeline"))
from slv import load, work, frame, scene  # noqa: E402
from minecraft_rig import build_minecraft_character_meshes, solve_minecraft_rig, PX  # noqa: E402

SHUTTER = [0.0, 0.045, 0.09, 0.135, 0.18, 0.225, 0.27, 0.315, 0.36]  # 9-slice 180-deg cinema motion blur
PART_NAMES = [
    "pelvis", "chest", "head",
    "arm_upper_R", "arm_lower_R", "arm_upper_L", "arm_lower_L",
    "leg_upper_R", "leg_lower_R", "leg_upper_L", "leg_lower_L",
]


def find_ffmpeg():
    tl = ROOT / "tools.local.json"
    if tl.exists():
        d = json.loads(tl.read_text(encoding="utf-8"))
        p = d.get("ffmpeg") or (d.get("tools") or {}).get("ffmpeg")
        if p and Path(p).exists():
            return str(p)
    return shutil.which("ffmpeg") or "ffmpeg"


# ==============================================================================
#  SHADERS: 2048x2048 SHADOW MAP + LINEAR OPEN-SHADE DAYLIGHT PBR + FOOT AO + OLPF
# ==============================================================================

SHADOW_VS = """#version 330
in vec3 inPos;
uniform mat4 uLightVP;
uniform mat4 uModel;
void main() {
  gl_Position = uLightVP * uModel * vec4(inPos, 1.0);
}
"""
SHADOW_FS = "#version 330\nvoid main() {}"

MESH_VS = """#version 330
in vec3 inPos;
in vec3 inNrm;
in vec2 inUv;
in vec3 inCol;
in vec3 inPbr;

uniform mat4 uProj;
uniform mat4 uView;
uniform mat4 uModel;
uniform mat3 uNormalMat;
uniform mat4 uLightVP;

out vec3 vWorldPos;
out vec3 vWorldNrm;
out vec2 vUv;
out vec3 vCol;
out vec3 vPbr;
out float vCamDepth;
out vec4 vLightSpace;

void main() {
  vec4 wp = uModel * vec4(inPos, 1.0);
  vWorldPos = wp.xyz;
  vWorldNrm = normalize(uNormalMat * inNrm);
  vUv = inUv;
  vCol = inCol;
  vPbr = inPbr;
  vec4 vp = uView * wp;
  vCamDepth = -vp.z;
  vLightSpace = uLightVP * wp;
  gl_Position = uProj * vp;
}
"""

MESH_FS = """#version 330
in vec3 vWorldPos;
in vec3 vWorldNrm;
in vec2 vUv;
in vec3 vCol;
in vec3 vPbr;
in float vCamDepth;
in vec4 vLightSpace;

uniform sampler2D uShadowMap;
uniform sampler2D uPlate;
uniform vec3 uRo, uSun;
uniform int uRenderMode;  /* 0=full PBR, 1=wireframe, 2=clay+shadows */

layout(location=0) out vec4 oCol;
layout(location=1) out vec4 oMeta;

float hash21(vec2 p) {
  p = fract(p * vec2(123.34, 456.21));
  p += dot(p, p + 45.32);
  return fract(p.x * p.y);
}
float noise2(vec2 p) {
  vec2 i = floor(p), f = fract(p);
  f = f * f * (3.0 - 2.0 * f);
  return mix(mix(hash21(i), hash21(i + vec2(1,0)), f.x),
             mix(hash21(i + vec2(0,1)), hash21(i + vec2(1,1)), f.x), f.y);
}
vec3 acesFilm(vec3 x) {
  return clamp((x * (2.51 * x + 0.03)) / (x * (2.43 * x + 0.59) + 0.14), 0.0, 1.0);
}

float calcShadow(vec4 lsPos, vec3 N, vec3 L) {
  vec3 proj = lsPos.xyz / max(lsPos.w, 1e-5) * 0.5 + 0.5;
  if (proj.z > 1.0 || proj.x < 0.01 || proj.x > 0.99 || proj.y < 0.01 || proj.y > 0.99) return 1.0;
  float bias = max(0.0024 * (1.0 - dot(N, L)), 0.0008);
  float lit = 0.0;
  vec2 tex = vec2(1.0 / 2048.0);
  for (int x = -2; x <= 2; x++) {
    for (int y = -2; y <= 2; y++) {
      float pcfD = texture(uShadowMap, proj.xy + vec2(float(x), float(y)) * tex * 1.45).r;
      lit += (proj.z - bias <= pcfD) ? 1.0 : 0.0;
    }
  }
  return lit / 25.0;
}

vec3 samplePlateIBL(vec3 R) {
  vec2 uvEnv = vec2(clamp(0.5 + R.x * 0.42, 0.02, 0.98), clamp(0.52 - R.z * 0.48, 0.02, 0.98));
  vec3 pCol = pow(texture(uPlate, uvEnv).rgb, vec3(2.2));
  vec3 skyCol = vec3(0.58, 0.64, 0.72) * smoothstep(0.10, 0.85, R.z) * 0.90;
  float gndDim = mix(0.50, 1.0, smoothstep(-0.55, 0.15, R.z));
  return pCol * gndDim + skyCol;
}

void main() {
  if (uRenderMode == 1) {
    oCol = vec4(0.12, 0.96, 1.0, 0.95);
    oMeta = vec4(1.0, 1.0 / max(vCamDepth, 0.2), 0.0, 1.0);
    return;
  }

  vec3 N0 = normalize(vWorldNrm);
  vec3 V = normalize(uRo - vWorldPos);
  vec3 L = normalize(uSun);
  vec3 baseSrgb = vCol;
  float rough = vPbr.x;
  float metal = vPbr.y;
  float vAO   = vPbr.z;

  if (uRenderMode == 2) {
    baseSrgb = vec3(0.76, 0.77, 0.79);
    rough = 0.80;
    metal = 0.0;
  }

  /* Subtle organic fabric/vinyl micro-texture so Minecraft voxels integrate cleanly with live 1080p street grain */
  float micro = noise2(vWorldPos.xy * 78.0 + vWorldPos.z * 64.0);
  baseSrgb *= (0.955 + 0.09 * micro);
  /* Very subtle pixel-tile edge crease micro-AO for tactile Minecraft block definition */
  vec2 edgeD = min(vUv, 1.0 - vUv);
  float pixCrease = smoothstep(0.0, 0.07, min(edgeD.x, edgeD.y));
  vAO *= mix(0.90, 1.0, pixCrease);

  vec3 baseLin = pow(max(baseSrgb, vec3(0.002)), vec3(2.2));
  vec3 H = normalize(L + V);
  float NdotL = max(dot(N0, L), 0.0);
  float NdotV = max(dot(N0, V), 0.001);
  float NdotH = max(dot(N0, H), 0.0);
  float shad = calcShadow(vLightSpace, N0, L);

  /* Open-shade outdoor concrete street daylight matching Pexels_8928132 white balance & crisp 3D contrast! */
  vec3 sunCol    = vec3(1.05, 1.02, 0.96) * 2.42;   /* crisp soft overhead daylight key */
  vec3 skyFill   = vec3(0.43, 0.48, 0.52);          /* bright overcast sky ambient */
  vec3 wallBnc   = vec3(0.34, 0.33, 0.28);          /* warm concrete wall bounce behind character */
  vec3 gndBounce = vec3(0.24, 0.29, 0.29);          /* cool blue-gray sidewalk floor bounce */
  vec3 hemi = mix(gndBounce, skyFill, clamp(N0.z * 0.5 + 0.5, 0.0, 1.0));
  hemi += wallBnc * clamp(-N0.y * 0.5 + 0.5, 0.0, 1.0) * 0.30;
  /* Ground proximity AO under shoes/ankles */
  float gndAO = mix(0.40, 1.0, smoothstep(0.0, 0.30, vWorldPos.z));
  float totalAO = clamp(vAO * gndAO, 0.20, 1.0);

  vec3 F0 = mix(vec3(0.04), baseLin, metal);
  vec3 F = F0 + (1.0 - F0) * pow(1.0 - NdotV, 5.0);
  float a2 = max(rough * rough, 0.005);
  float denom = NdotH * NdotH * (a2 - 1.0) + 1.0;
  float D_ggx = a2 / (3.14159 * denom * denom + 1e-5);
  float specSun = min(D_ggx * 0.24 * NdotL, 4.8);

  vec3 R = reflect(-V, N0);
  vec3 envRefl = samplePlateIBL(R);
  float reflPow = pow(1.0 - rough, 1.9);

  /* Crisp open-shade shadow floor so faces & limbs sculpt strongly in 3D */
  float shadLit = mix(0.26, 1.0, shad);
  vec3 diffAlb = baseLin * (1.0 - metal * 0.88);
  vec3 linCol = diffAlb * (sunCol * NdotL * shadLit + hemi * totalAO)
              + F * sunCol * specSun * shad
              + F * envRefl * reflPow * (0.35 + 0.65 * totalAO);

  vec3 mapped = acesFilm(linCol * 0.86);
  vec3 srgb = pow(clamp(mapped, 0.0, 1.0), vec3(1.0 / 2.2));
  oCol = vec4(srgb, 1.0);
  oMeta = vec4(1.0, 1.0 / max(vCamDepth, 0.2), 0.0, 1.0);
}
"""

VS_FSQ = "#version 330\nin vec2 p; void main(){ gl_Position = vec4(p, 0.0, 1.0); }"

GROUND_SHAD_FS = """#version 330
uniform sampler2D uPlate;
uniform sampler2D uShadowMap;
uniform vec2 uRes;
uniform float uF, uWallY;
uniform vec3 uRo;
uniform mat3 uCam;
uniform mat4 uLightVP;
uniform vec4 uFootR;  /* xyz = right shoe bottom center, w = unused */
uniform vec4 uFootL;  /* xyz = left shoe bottom center, w = unused */
uniform int uSkipShadows;

layout(location=0) out vec4 oCol;
layout(location=1) out vec4 oMeta;

void main() {
  vec2 px0 = vec2(gl_FragCoord.x, uRes.y - gl_FragCoord.y);
  vec2 uv0 = px0 / uRes;
  vec3 col = texture(uPlate, clamp(uv0, vec2(0.002), vec2(0.998))).rgb;
  float cgA = 0.0;
  float cgInvD = 0.0;

  if (uSkipShadows == 0) {
    vec3 rd = normalize(uCam * vec3((px0 - 0.5 * uRes) / uF, 1.0));
    if (rd.z < -0.005) {
      float tf = -uRo.z / rd.z;
      if (tf > 0.2 && tf < 18.0) {
        vec3 X = uRo + rd * tf;
        if (X.y < uWallY + 0.12 && abs(X.x) < 4.5) {
          float wallFade = smoothstep(uWallY + 0.12, uWallY - 0.05, X.y);

          /* 1. Soft 5x5 PCF Daylight Shadow from Articulated Steve Mesh */
          float occ = 0.0;
          vec4 ls = uLightVP * vec4(X, 1.0);
          vec3 proj = ls.xyz / max(ls.w, 1e-5) * 0.5 + 0.5;
          if (proj.x > 0.01 && proj.x < 0.99 && proj.y > 0.01 && proj.y < 0.99 && proj.z < 1.0) {
            vec2 tex = vec2(1.0 / 2048.0);
            for (int i = -2; i <= 2; i++) {
              for (int j = -2; j <= 2; j++) {
                float dMap = texture(uShadowMap, proj.xy + vec2(float(i), float(j)) * tex * 2.8).r;
                if (proj.z - 0.0012 > dMap) occ += 1.0;
              }
            }
            occ /= 25.0;
          }

          /* 2. Grounded Dual-Shoe Contact AO (tight & dark when shoe rests on z=0, softly fades on knee lift/jump!) */
          float aoR = 0.0, aoL = 0.0;
          {
            float hR = clamp(uFootR.z, 0.0, 1.2);
            float rR = 0.15 + 0.22 * hR;
            float dR = length(X.xy - uFootR.xy);
            aoR = exp(-(dR * dR) / max(rR * rR * 0.52, 0.004)) * exp(-hR * 4.0) * 0.74;
          }
          {
            float hL = clamp(uFootL.z, 0.0, 1.2);
            float rL = 0.15 + 0.22 * hL;
            float dL = length(X.xy - uFootL.xy);
            aoL = exp(-(dL * dL) / max(rL * rL * 0.52, 0.004)) * exp(-hL * 4.0) * 0.74;
          }

          /* Soft ambient under-body grounding shade matching real overcast concrete sidewalk */
          vec2 midF = 0.5 * (uFootR.xy + uFootL.xy);
          float dB = length((X.xy - midF) / vec2(0.55, 0.40));
          float bodyAO = exp(-dB * dB) * 0.22;

          float sAmt = clamp(max(max(occ * 0.54, max(aoR, aoL)), bodyAO), 0.0, 0.74) * wallFade;
          if (sAmt > 0.005) {
            /* Match real blue-gray sidewalk shadow color (cool neutral dark concrete) */
            vec3 shadTint = vec3(0.84, 0.89, 0.95);
            col = mix(col, col * shadTint * 0.34, sAmt);
            cgA = clamp(sAmt * 0.70, 0.0, 0.75);
            cgInvD = cgA / max(tf, 0.3);
          }
        }
      }
    }
  }
  oCol = vec4(clamp(col, 0.0, 1.0), 1.0);
  oMeta = vec4(cgA, cgInvD, 0.0, 1.0);
}
"""

ACCUM_ADD_FS = """#version 330
uniform sampler2D uSrc0, uSrc1; uniform vec2 uRes; uniform float uWeight;
layout(location=0) out vec4 o0; layout(location=1) out vec4 o1;
void main() {
  vec2 uv = gl_FragCoord.xy / uRes;
  o0 = texture(uSrc0, uv) * uWeight;
  o1 = texture(uSrc1, uv) * uWeight;
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
    res = mix(c0, soft, cgA * 0.30);
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


def mat4_perspective_from_opencv(f_px, W, H, near=0.1, far=40.0):
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


def mat4_ortho_light_vp(sun_dir, center=(0.0, 0.6, 0.85), half_ext=2.6, near=0.5, far=14.0):
    L = np.asarray(sun_dir, dtype=np.float64); L /= np.linalg.norm(L) + 1e-9
    eye = np.asarray(center, dtype=np.float64) + L * 7.5
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


def mat4_from_pr(pos, R3, scale=1.0):
    M = np.eye(4, dtype=np.float32)
    M[:3, :3] = (R3 * float(scale)).astype(np.float32)
    M[:3, 3] = pos.astype(np.float32)
    return M, R3.astype(np.float32)


def interp_pose(rig_pos, rig_rot, i, frac):
    """Exact linear pos + SLERP rotation interpolation for sub-frame cinema motion blur!"""
    N = len(rig_pos["pelvis"])
    i0 = int(np.clip(i, 0, N - 1))
    i1 = int(np.clip(i + 1, 0, N - 1))
    pos_s, rot_s = {}, {}
    for k in PART_NAMES:
        p0, p1 = rig_pos[k][i0], rig_pos[k][i1]
        pos_s[k] = p0 * (1.0 - frac) + p1 * frac
        if frac <= 1e-4 or i0 == i1:
            rot_s[k] = rig_rot[k][i0]
        else:
            r0 = Rot.from_matrix(rig_rot[k][i0])
            r1 = Rot.from_matrix(rig_rot[k][i1])
            rot_s[k] = Slerp([0.0, 1.0], Rot.concatenate([r0, r1]))([frac])[0].as_matrix()
    return pos_s, rot_s


def draw_rig_pass(live_bgr, ro, R, f_px, P_hum, rig_pos, rig_rot, i, scale_char=1.0, out_wh=(1280, 720)):
    """Diagnostic Mocap Rig Pass (plate gain=1.0): green MediaPipe human skeleton + cyan/gold Minecraft FK rig bones!"""
    H, W = live_bgr.shape[:2]
    im = live_bgr.copy()
    def proj(X):
        q = (np.atleast_2d(X) - ro) @ R
        z = q[:, 2]
        uv = f_px * q[:, :2] / np.maximum(z[:, None], 0.05) + np.array([W * 0.5, H * 0.5])
        return uv, z

    # 1. MediaPipe 3D human capture skeleton (soft lime green)
    HBONES = [(11,12),(11,13),(13,15),(12,14),(14,16),(11,23),(12,24),(23,24),(23,25),(25,27),(24,26),(26,28)]
    huv, hz = proj(P_hum[i])
    for a, b in HBONES:
        if hz[a] > 0.1 and hz[b] > 0.1:
            cv2.line(im, (int(huv[a,0]), int(huv[a,1])), (int(huv[b,0]), int(huv[b,1])), (60, 235, 60), 3, cv2.LINE_AA)

    # 2. Solved Minecraft FK Articulated Rig Bones + Local Axes + Joint Pivots!
    u = PX * float(scale_char)
    p_pelv = rig_pos["pelvis"][i]; p_chst = rig_pos["chest"][i]; p_head = rig_pos["head"][i]
    p_head_t = p_head + rig_rot["head"][i] @ np.array([0.0, 0.0, 8.0 * u])
    p_sh_R = rig_pos["arm_upper_R"][i]; p_el_R = rig_pos["arm_lower_R"][i]
    p_wr_R = p_el_R + rig_rot["arm_lower_R"][i] @ np.array([0.0, 0.0, -6.0 * u])
    p_sh_L = rig_pos["arm_upper_L"][i]; p_el_L = rig_pos["arm_lower_L"][i]
    p_wr_L = p_el_L + rig_rot["arm_lower_L"][i] @ np.array([0.0, 0.0, -6.0 * u])
    p_hp_R = rig_pos["leg_upper_R"][i]; p_kn_R = rig_pos["leg_lower_R"][i]
    p_an_R = p_kn_R + rig_rot["leg_lower_R"][i] @ np.array([0.0, 0.0, -6.0 * u])
    p_hp_L = rig_pos["leg_upper_L"][i]; p_kn_L = rig_pos["leg_lower_L"][i]
    p_an_L = p_kn_L + rig_rot["leg_lower_L"][i] @ np.array([0.0, 0.0, -6.0 * u])

    rig_segs = [
        (p_pelv, p_chst), (p_chst, p_head), (p_head, p_head_t),
        (p_chst, p_sh_R), (p_sh_R, p_el_R), (p_el_R, p_wr_R),
        (p_chst, p_sh_L), (p_sh_L, p_el_L), (p_el_L, p_wr_L),
        (p_pelv, p_hp_R), (p_hp_R, p_kn_R), (p_kn_R, p_an_R),
        (p_pelv, p_hp_L), (p_hp_L, p_kn_L), (p_kn_L, p_an_L),
    ]
    for a, b in rig_segs:
        uv, z = proj(np.array([a, b]))
        if np.all(z > 0.1):
            cv2.line(im, (int(uv[0,0]), int(uv[0,1])), (int(uv[1,0]), int(uv[1,1])), (255, 225, 30), 5, cv2.LINE_AA)
    pivs = [p_pelv, p_chst, p_head, p_sh_R, p_el_R, p_wr_R, p_sh_L, p_el_L, p_wr_L, p_hp_R, p_kn_R, p_an_R, p_hp_L, p_kn_L, p_an_L]
    puv, pz = proj(np.array(pivs))
    for k in range(len(pivs)):
        if pz[k] > 0.1:
            cv2.circle(im, (int(puv[k,0]), int(puv[k,1])), 8, (0, 180, 255), -1, cv2.LINE_AA)
            cv2.circle(im, (int(puv[k,0]), int(puv[k,1])), 8, (255, 255, 255), 2, cv2.LINE_AA)
    # Draw character head facing arrow (+Y) so 360-deg head look reads crisply!
    p_look = p_head + rig_rot["head"][i] @ np.array([0.0, 0.35, 4.0 * u])
    p_hc   = p_head + rig_rot["head"][i] @ np.array([0.0, 0.0, 4.0 * u])
    luv, lz = proj(np.array([p_hc, p_look]))
    if np.all(lz > 0.1):
        cv2.arrowedLine(im, (int(luv[0,0]), int(luv[0,1])), (int(luv[1,0]), int(luv[1,1])), (40, 80, 255), 4, cv2.LINE_AA, tipLength=0.30)

    return cv2.resize(im, out_wh, interpolation=cv2.INTER_AREA)


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
    # Zero-text clean stacked Before/After comparison (top=Minecraft Steve, bottom=Original Dancer!)
    run([
        "-framerate", r, "-i", str(work(c, "comp", "%04d.png")),
        "-framerate", r, "-i", str(work(c, "frames", "%04d.jpg")),
        "-filter_complex",
        "[0]scale=1080:608[a];[1]scale=1080:608[b];[a][b]vstack,pad=1080:1350:0:67:color=black",
    ] + x264 + tags + [str(out / f"{nm}_Compare_1080x1350.mp4")])
    print("encoded ->", out)


def main():
    args = sys.argv[1:]
    stills = None
    out_dir = "out"
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

    c = load(piece_arg or (here / "piece_steve"))
    s = scene(c)
    C = np.array(s["cam"], dtype=np.float64)
    R_all = np.array(s["rot"], dtype=np.float64)
    f_px = float(s["f"])
    N = len(C)

    # Soft high front-left overhead street daylight matching Pexels_8928132 shadows!
    sun = np.array([-0.24, -0.45, 0.86], dtype=np.float64)
    sun /= np.linalg.norm(sun) + 1e-9

    limbs = np.load(work(c, "limbs.npz"))
    P_hum = np.asarray(limbs["P"], dtype=np.float64)
    img_lm = np.asarray(limbs["img"], dtype=np.float64)

    pl0 = cv2.imread(str(work(c, "clean_frames", "0000.jpg")))
    if pl0 is None:
        sys.exit("Missing clean_frames: run prepare_piece.py first!")
    H, W = pl0.shape[:2]

    # Match Steve's full height 1:1 with dancer's live screen height & stabilize depth from 2D stature!
    scale_char = 1.02
    rig_pos, rig_rot, rig_jnts = solve_minecraft_rig(
        P_hum, scale_char=scale_char, img_lm=img_lm, cam_C=C, cam_R=R_all, f_px=f_px, W=W, H=H
    )
    np.savez_compressed(
        work(c, "rig_solve.npz"),
        jnts=rig_jnts,
        **{f"pos_{k}": v for k, v in rig_pos.items()},
        **{f"rot_{k}": v for k, v in rig_rot.items()},
    )
    print(f"Solved 11-part Minecraft Steve FK rig for {N} frames (scale={scale_char})")

    # Measure plate black floor & live sensor grain cleanly
    gray0 = cv2.cvtColor(pl0, cv2.COLOR_BGR2GRAY).astype(np.float32) / 255.0
    black_floor = float(np.clip(np.percentile(gray0, 1.0) * 0.88, 0.010, 0.055))
    hp0 = gray0 - cv2.GaussianBlur(gray0, (0, 0), 1.2)
    grain_amt = float(np.clip(float(np.median(np.abs(hp0 - np.median(hp0)))) * 1.4826 * 1.10, 0.008, 0.022))

    # Estimate back concrete wall Y plane slightly behind dancer's furthest back heel (~+1.55m)
    wall_y = float(np.percentile(P_hum[:, 23, 1], 98.0)) + 0.42
    proj_m = mat4_perspective_from_opencv(f_px, W, H, near=0.1, far=40.0)

    ctx = moderngl.create_standalone_context(require=330)
    shad_prog = ctx.program(vertex_shader=SHADOW_VS, fragment_shader=SHADOW_FS)
    mesh_prog = ctx.program(vertex_shader=MESH_VS, fragment_shader=MESH_FS)
    gnd_prog  = ctx.program(vertex_shader=VS_FSQ, fragment_shader=GROUND_SHAD_FS)
    acc_prog  = ctx.program(vertex_shader=VS_FSQ, fragment_shader=ACCUM_ADD_FS)
    post_prog = ctx.program(vertex_shader=VS_FSQ, fragment_shader=POST_FS)

    fsq = ctx.buffer(np.array([-1, -1, 1, -1, -1, 1, -1, 1, 1, -1, 1, 1], dtype=np.float32).tobytes())
    va_gnd  = ctx.vertex_array(gnd_prog,  [(fsq, "2f", "p")])
    va_acc  = ctx.vertex_array(acc_prog,  [(fsq, "2f", "p")])
    va_post = ctx.vertex_array(post_prog, [(fsq, "2f", "p")])

    meshes = build_minecraft_character_meshes()
    gpu_meshes = {}
    for pname, md in meshes.items():
        vdata = np.column_stack([md["pos"], md["nrm"], md["uv"], md["col"], md["pbr"]]).astype(np.float32)
        vbo = ctx.buffer(vdata.tobytes())
        ibo = ctx.buffer(md["idx"].astype(np.uint32).tobytes())
        vao_mesh = ctx.vertex_array(mesh_prog, [(vbo, "3f 3f 2f 3f 3f", "inPos", "inNrm", "inUv", "inCol", "inPbr")], index_buffer=ibo)
        vao_shad = ctx.vertex_array(shad_prog, [(vbo, "3f 44x", "inPos")], index_buffer=ibo)
        gpu_meshes[pname] = (vao_mesh, vao_shad)

    t_smap = ctx.depth_texture((2048, 2048)); t_smap.filter = (moderngl.LINEAR, moderngl.LINEAR); t_smap.repeat_x = t_smap.repeat_y = False
    fbo_shad = ctx.framebuffer(depth_attachment=t_smap)
    t_sub_col = ctx.texture((W, H), 4, dtype="f4"); t_sub_meta = ctx.texture((W, H), 4, dtype="f4")
    rbo_depth = ctx.depth_renderbuffer((W, H))
    fbo_sub = ctx.framebuffer([t_sub_col, t_sub_meta], depth_attachment=rbo_depth)
    t_acc_col = ctx.texture((W, H), 4, dtype="f4"); t_acc_meta = ctx.texture((W, H), 4, dtype="f4")
    t_acc_col.filter = (moderngl.LINEAR, moderngl.LINEAR); t_acc_meta.filter = (moderngl.LINEAR, moderngl.LINEAR)
    fbo_acc = ctx.framebuffer([t_acc_col, t_acc_meta])
    t_final = ctx.texture((W, H), 4, dtype="f4")
    fbo_final = ctx.framebuffer([t_final])
    plate_t = ctx.texture((W, H), 3); plate_t.filter = (moderngl.LINEAR, moderngl.LINEAR); plate_t.repeat_x = plate_t.repeat_y = False

    mesh_prog["uProj"].write(proj_m.T.tobytes())
    mesh_prog["uSun"].value = tuple(map(float, sun))
    gnd_prog["uRes"].value = (float(W), float(H))
    gnd_prog["uF"].value = f_px
    gnd_prog["uWallY"].value = wall_y
    acc_prog["uRes"].value = (float(W), float(H))
    post_prog["uRes"].value = (float(W), float(H))
    post_prog["uBlackFloor"].value = black_floor
    post_prog["uGrainAmt"].value = grain_amt

    u = PX * scale_char

    def render_one(i):
        live_pl = frame(c, i)
        clean_pl = cv2.imread(str(work(c, "clean_frames", f"{i:04d}.jpg")))
        plate_t.write(clean_pl[:, :, ::-1].tobytes())

        ro = C[i]; R = R_all[i]
        Rt = tuple(R.T.flatten().astype(np.float32))
        view_m = mat4_view_from_colmap(ro, R)
        n_shut = len(SHUTTER)
        sub_w = 1.0 / float(n_shut)

        # Follow character center cleanly with ortho shadow map for maximum 2048x2048 shadow resolution!
        ctr = rig_pos["pelvis"][i]
        light_vp = mat4_ortho_light_vp(sun, center=(float(ctr[0]), float(ctr[1]), 0.80), half_ext=2.4)
        shad_prog["uLightVP"].write(light_vp.T.tobytes())
        mesh_prog["uLightVP"].write(light_vp.T.tobytes())
        gnd_prog["uLightVP"].write(light_vp.T.tobytes())

        fbo_acc.use(); ctx.viewport = (0, 0, W, H); ctx.disable(moderngl.DEPTH_TEST); ctx.disable(moderngl.BLEND)
        ctx.clear(0.0, 0.0, 0.0, 0.0)

        for si in range(n_shut):
            pos_s, rot_s = interp_pose(rig_pos, rig_rot, i, SHUTTER[si])
            foot_R = pos_s["leg_lower_R"] + rot_s["leg_lower_R"] @ np.array([0.0, 0.0, -6.0 * u])
            foot_L = pos_s["leg_lower_L"] + rot_s["leg_lower_L"] @ np.array([0.0, 0.0, -6.0 * u])

            # Pass 1: 2048x2048 Soft Daylight Shadow Map
            fbo_shad.use(); ctx.viewport = (0, 0, 2048, 2048); ctx.enable(moderngl.DEPTH_TEST); ctx.disable(moderngl.BLEND)
            ctx.clear(depth=1.0)
            for pname in PART_NAMES:
                M, _ = mat4_from_pr(pos_s[pname], rot_s[pname], scale=scale_char)
                shad_prog["uModel"].write(M.T.tobytes())
                gpu_meshes[pname][1].render(moderngl.TRIANGLES)

            # Pass 2A: Spotless Clean Plate + Soft PCF Ground Shadow + Dynamic Dual-Shoe Contact AO!
            fbo_sub.use(); ctx.viewport = (0, 0, W, H); ctx.disable(moderngl.DEPTH_TEST); ctx.disable(moderngl.BLEND)
            ctx.clear(depth=1.0)
            plate_t.use(0); t_smap.use(1)
            gnd_prog["uPlate"].value = 0; gnd_prog["uShadowMap"].value = 1
            gnd_prog["uRo"].value = tuple(map(float, ro)); gnd_prog["uCam"].value = Rt
            gnd_prog["uFootR"].value = (float(foot_R[0]), float(foot_R[1]), float(foot_R[2]), 1.0)
            gnd_prog["uFootL"].value = (float(foot_L[0]), float(foot_L[1]), float(foot_L[2]), 1.0)
            gnd_prog["uSkipShadows"].value = 0
            va_gnd.render(moderngl.TRIANGLES)

            # Pass 2B: Articulated 11-Part Beveled 3D Pixel Steve!
            ctx.enable(moderngl.DEPTH_TEST); ctx.disable(moderngl.BLEND)
            t_smap.use(0); plate_t.use(1)
            mesh_prog["uShadowMap"].value = 0; mesh_prog["uPlate"].value = 1
            mesh_prog["uView"].write(view_m.T.tobytes()); mesh_prog["uRo"].value = tuple(map(float, ro))
            mesh_prog["uRenderMode"].value = 0
            for pname in PART_NAMES:
                M, N3 = mat4_from_pr(pos_s[pname], rot_s[pname], scale=scale_char)
                mesh_prog["uModel"].write(M.T.tobytes()); mesh_prog["uNormalMat"].write(N3.T.tobytes())
                gpu_meshes[pname][0].render(moderngl.TRIANGLES)

            fbo_acc.use(); ctx.disable(moderngl.DEPTH_TEST); ctx.enable(moderngl.BLEND)
            ctx.blend_func = (moderngl.ONE, moderngl.ONE)
            t_sub_col.use(0); t_sub_meta.use(1)
            acc_prog["uSrc0"].value = 0; acc_prog["uSrc1"].value = 1; acc_prog["uWeight"].value = sub_w
            va_acc.render(moderngl.TRIANGLES)

        fbo_final.use(); ctx.viewport = (0, 0, W, H); ctx.disable(moderngl.DEPTH_TEST); ctx.disable(moderngl.BLEND)
        t_acc_col.use(0); t_acc_meta.use(1)
        post_prog["uScene"].value = 0; post_prog["uMeta"].value = 1; post_prog["uFrame"].value = float(i)
        va_post.render(moderngl.TRIANGLES)
        ctx.finish()
        arr = np.frombuffer(t_final.read(), "f4").reshape(H, W, 4)[::-1]
        img = (np.clip(arr[..., :3], 0, 1) * 255 + 0.5).astype(np.uint8)[:, :, ::-1]

        pass_dict = None
        if save_passes:
            pos_c, rot_c = interp_pose(rig_pos, rig_rot, i, SHUTTER[n_shut // 2])
            foot_R = pos_c["leg_lower_R"] + rot_c["leg_lower_R"] @ np.array([0.0, 0.0, -6.0 * u])
            foot_L = pos_c["leg_lower_L"] + rot_c["leg_lower_L"] @ np.array([0.0, 0.0, -6.0 * u])

            fbo_shad.use(); ctx.viewport = (0, 0, 2048, 2048); ctx.enable(moderngl.DEPTH_TEST); ctx.disable(moderngl.BLEND)
            ctx.clear(depth=1.0)
            for pname in PART_NAMES:
                M, _ = mat4_from_pr(pos_c[pname], rot_c[pname], scale=scale_char)
                shad_prog["uModel"].write(M.T.tobytes()); gpu_meshes[pname][1].render(moderngl.TRIANGLES)

            def render_diag(mode_id, skip_sh, wire_flag):
                fbo_sub.use(); ctx.viewport = (0, 0, W, H); ctx.disable(moderngl.DEPTH_TEST); ctx.disable(moderngl.BLEND)
                ctx.clear(depth=1.0)
                plate_t.use(0); t_smap.use(1)
                gnd_prog["uPlate"].value = 0; gnd_prog["uShadowMap"].value = 1
                gnd_prog["uRo"].value = tuple(map(float, ro)); gnd_prog["uCam"].value = Rt
                gnd_prog["uFootR"].value = (float(foot_R[0]), float(foot_R[1]), float(foot_R[2]), 1.0)
                gnd_prog["uFootL"].value = (float(foot_L[0]), float(foot_L[1]), float(foot_L[2]), 1.0)
                gnd_prog["uSkipShadows"].value = int(skip_sh)
                va_gnd.render(moderngl.TRIANGLES)

                ctx.enable(moderngl.DEPTH_TEST); ctx.disable(moderngl.BLEND)
                ctx.wireframe = bool(wire_flag)
                t_smap.use(0); plate_t.use(1)
                mesh_prog["uShadowMap"].value = 0; mesh_prog["uPlate"].value = 1
                mesh_prog["uView"].write(view_m.T.tobytes()); mesh_prog["uRo"].value = tuple(map(float, ro))
                mesh_prog["uRenderMode"].value = int(mode_id)
                for pname in PART_NAMES:
                    M, N3 = mat4_from_pr(pos_c[pname], rot_c[pname], scale=scale_char)
                    mesh_prog["uModel"].write(M.T.tobytes()); mesh_prog["uNormalMat"].write(N3.T.tobytes())
                    gpu_meshes[pname][0].render(moderngl.TRIANGLES)
                ctx.wireframe = False
                ctx.finish()
                a_d = np.frombuffer(t_sub_col.read(), "f4").reshape(H, W, 4)[::-1]
                u8 = (np.clip(a_d[..., :3], 0, 1) * 255 + 0.5).astype(np.uint8)[:, :, ::-1]
                return cv2.resize(u8, (1280, 720), interpolation=cv2.INTER_AREA)

            pass_dict = {
                "clean": cv2.resize(clean_pl, (1280, 720), interpolation=cv2.INTER_AREA),
                "rig":   draw_rig_pass(live_pl, ro, R, f_px, P_hum, rig_pos, rig_rot, i, scale_char=scale_char),
                "wire":  render_diag(mode_id=1, skip_sh=1, wire_flag=True),
                "clay":  render_diag(mode_id=2, skip_sh=0, wire_flag=False),
            }
        return live_pl, img, pass_dict

    if stills:
        rows_im = []
        hero = None
        for idx in stills:
            pl, comp, _ = render_one(idx)
            if idx == stills[len(stills) // 2]:
                hero = comp
            t_im = lambda im, lb: label(cv2.resize(im, (960, 540)), lb)
            rows_im.append(np.hstack([t_im(pl, f"original {idx}"), t_im(comp, f"steve mocap {idx}")]))
        out_p = work(c, "still_steve_sheet.jpg")
        cv2.imwrite(str(out_p), np.vstack(rows_im), [cv2.IMWRITE_JPEG_QUALITY, 92])
        if hero is not None:
            cv2.imwrite(str(work(c, "still_steve_hero.jpg")), hero, [cv2.IMWRITE_JPEG_QUALITY, 94])
        print("stills ->", out_p)
        return

    comp_dir = work(c, "comp")
    shutil.rmtree(comp_dir, ignore_errors=True)
    comp_dir.mkdir(parents=True)
    pass_root = work(c, "pass_steve")
    if save_passes:
        shutil.rmtree(pass_root, ignore_errors=True)
        for pk in ("clean", "rig", "wire", "clay"):
            (pass_root / pk).mkdir(parents=True, exist_ok=True)
    ts = time.time()
    for i in range(N):
        _, comp, pdict = render_one(i)
        cv2.imwrite(str(comp_dir / f"{i:04d}.png"), comp, [cv2.IMWRITE_PNG_COMPRESSION, 1])
        if save_passes and pdict is not None:
            for pk, pim in pdict.items():
                cv2.imwrite(str(pass_root / pk / f"{i:04d}.jpg"), pim, [cv2.IMWRITE_JPEG_QUALITY, 93])
        if i % 40 == 0 or i == N - 1:
            print(f"steve_mocap frame {i}/{N} ({time.time() - ts:.1f}s)", flush=True)
    encode(c, out_dir=out_dir)


if __name__ == "__main__":
    main()
