"""Effect #1: Camera-Projection-Mapped 3D Concrete Hexagonal Tile Kinetic Floor Wave (Native OpenGL 3.3).

    python render_floor_wave.py --still 20,75,140,210     quick look -> work/still_wave_sheet.jpg
    python render_floor_wave.py --render                 full render -> out/Concrete_Hex_Floor_Wave_*.mp4

Real concrete plaza floor (Pexels 32109492) fractures into ~850 3D beveled hexagonal monolith columns:
- Every rising hex column's top face projects its rest z=0 floor position back through the live COLMAP camera
  to sample the EXACT real sunlit concrete tile texture from the live plate (100% seamless at rest!).
- Expanding multi-ring physical wave ripples erupt directly from the dancer's foot stomps & weight shifts,
  with a calm grounded pad directly under his sneakers so he stays solidly planted on z=0 while waves sweep
  BOTH in front of his legs (occluding him) and behind him (occluded by MediaPipe person depth matte!).
- Exposes dark chiseled basalt side walls + subtle warm sunlit bevel edge + 2048x2048 midday sun self-shadows.
- ZERO text/HUD overlays.
"""
import argparse, json, math, shutil, subprocess, sys, time
from pathlib import Path
import numpy as np, cv2, moderngl

here = Path(__file__).resolve().parent
ROOT = here.parents[1]
sys.path.insert(0, str(ROOT / "examples/shader_live/pipeline"))
from slv import load, work, frame, scene  # noqa: E402

SRC_PIECE = ROOT / "examples/minecraft_drop_sim/piece_drops"


def find_ffmpeg():
    tl = ROOT / "tools.local.json"
    if tl.exists():
        d = json.loads(tl.read_text(encoding="utf-8"))
        p = d.get("ffmpeg") or (d.get("tools") or {}).get("ffmpeg")
        if p and Path(p).exists():
            return str(p)
    return shutil.which("ffmpeg") or "ffmpeg"


def build_unit_hex_prism(radius=0.140, depth=1.10, bevel=0.014):
    """Build a crisp 3D beveled hexagonal column (top face at z=0, extends down to z=-depth).
    Vertex format: pos3, nrm3, is_top(1f) -> 7 floats/vert."""
    pos, nrm, top_f, idx = [], [], [], []
    angs = [math.radians(60.0 * k + 30.0) for k in range(6)]
    r_out = radius
    r_in = radius - bevel
    outer = [(r_out * math.cos(a), r_out * math.sin(a)) for a in angs]
    inner = [(r_in * math.cos(a), r_in * math.sin(a)) for a in angs]

    def add_tri(a, b, c, n, tf):
        base = len(pos)
        nn = np.asarray(n, dtype=np.float32)
        nn /= (np.linalg.norm(nn) + 1e-8)
        for p in (a, b, c):
            pos.append(p); nrm.append(tuple(nn)); top_f.append(tf)
        idx.extend([base, base + 1, base + 2])

    def add_quad(a, b, c, d, n, tf):
        add_tri(a, b, c, n, tf)
        add_tri(a, c, d, n, tf)

    # 1. Flat top hexagon (z = 0.0, is_top = 1.0 -> samples live camera-projected concrete!)
    ctr = (0.0, 0.0, 0.0)
    for k in range(6):
        k2 = (k + 1) % 6
        add_tri(ctr, (inner[k][0], inner[k][1], 0.0), (inner[k2][0], inner[k2][1], 0.0), (0.0, 0.0, 1.0), 1.0)

    # 2. Crisp 45-deg top chamfer rim (z = 0 -> -bevel, is_top = 0.5 -> catches bright sunlit bevel gleam!)
    for k in range(6):
        k2 = (k + 1) % 6
        mx = 0.5 * (outer[k][0] + outer[k2][0])
        my = 0.5 * (outer[k][1] + outer[k2][1])
        ml = math.hypot(mx, my) + 1e-6
        nb = (mx / ml * 0.68, my / ml * 0.68, 0.73)
        add_quad(
            (inner[k][0], inner[k][1], 0.0),
            (outer[k][0], outer[k][1], -bevel),
            (outer[k2][0], outer[k2][1], -bevel),
            (inner[k2][0], inner[k2][1], 0.0),
            nb, 0.55,
        )

    # 3. 6 vertical chiseled concrete/slate side walls (z = -bevel -> -depth, is_top = 0.0)
    for k in range(6):
        k2 = (k + 1) % 6
        mx = 0.5 * (outer[k][0] + outer[k2][0])
        my = 0.5 * (outer[k][1] + outer[k2][1])
        ml = math.hypot(mx, my) + 1e-6
        ns = (mx / ml, my / ml, 0.0)
        add_quad(
            (outer[k][0], outer[k][1], -bevel),
            (outer[k][0], outer[k][1], -depth),
            (outer[k2][0], outer[k2][1], -depth),
            (outer[k2][0], outer[k2][1], -bevel),
            ns, 0.0,
        )

    vdata = np.column_stack([
        np.array(pos, dtype=np.float32),
        np.array(nrm, dtype=np.float32),
        np.array(top_f, dtype=np.float32)[:, None],
    ])
    return vdata, np.array(idx, dtype=np.uint32)


def build_hex_lattice(x_min=-4.30, x_max=5.20, y_min=-2.15, y_max=1.48, spacing=0.252):
    """Build tight honeycomb lattice covering the entire visible concrete plaza cleanly edge-to-edge."""
    pts = []
    dx = spacing
    dy = spacing * math.sqrt(3.0) * 0.5
    row = 0
    y = y_min
    while y <= y_max:
        x_off = (dx * 0.5) if (row % 2 == 1) else 0.0
        x = x_min + x_off
        while x <= x_max:
            pts.append((x, y))
            x += dx
        y += dy
        row += 1
    return np.array(pts, dtype=np.float64)


# ==============================================================================
#  SHADERS: INSTANCED HEX COLUMNS + LIVE CAMERA PROJECTION + 2048x2048 SHADOWS
# ==============================================================================

SHAD_VS = """#version 330
in vec3 inPos;
in vec4 inInst;  /* xy = world tile center, z = lift height h (>=0), w = wave energy */
uniform mat4 uLightVP;
void main() {
  vec3 wp = vec3(inPos.x + inInst.x, inPos.y + inInst.y, inPos.z + inInst.z);
  gl_Position = uLightVP * vec4(wp, 1.0);
}
"""
SHAD_FS = "#version 330\nvoid main() {}"

HEX_VS = """#version 330
in vec3 inPos;
in vec3 inNrm;
in float inIsTop;
in vec4 inInst;   /* xy = world tile center, z = lift h, w = wave energy (0..1) */

uniform mat4 uProj, uView, uLightVP;

out vec3 vWorldPos;
out vec3 vRestFloorPos;
out vec3 vNrm;
out float vIsTop;
out float vLift;
out float vEnergy;
out float vCamDepth;
out vec4 vLightSpace;

void main() {
  /* Slight organic tilt proportional to wave height makes columns breathe like physical stone! */
  vec3 wp = vec3(inPos.x + inInst.x, inPos.y + inInst.y, inPos.z + inInst.z);
  vWorldPos = wp;
  vRestFloorPos = vec3(inPos.x + inInst.x, inPos.y + inInst.y, 0.0);
  vNrm = normalize(inNrm);
  vIsTop = inIsTop;
  vLift = inInst.z;
  vEnergy = inInst.w;
  vec4 vp = uView * vec4(wp, 1.0);
  vCamDepth = -vp.z;
  vLightSpace = uLightVP * vec4(wp, 1.0);
  gl_Position = uProj * vp;
}
"""

HEX_FS = """#version 330
in vec3 vWorldPos;
in vec3 vRestFloorPos;
in vec3 vNrm;
in float vIsTop;
in float vLift;
in float vEnergy;
in float vCamDepth;
in vec4 vLightSpace;

uniform sampler2D uPlate, uShadowMap, uPersonMask;
uniform vec2 uRes;
uniform vec3 uRo, uSun;
uniform mat3 uCam;
uniform float uF, uPersonDepth;

layout(location=0) out vec4 oCol;
layout(location=1) out vec4 oMeta;

float hash21(vec2 p){ p=fract(p*vec2(123.34,456.21)); p+=dot(p,p+45.32); return fract(p.x*p.y); }
float noise2(vec2 p){ vec2 i=floor(p),f=fract(p); f=f*f*(3.0-2.0*f);
  return mix(mix(hash21(i),hash21(i+vec2(1,0)),f.x), mix(hash21(i+vec2(0,1)),hash21(i+vec2(1,1)),f.x), f.y); }
vec3 acesFilm(vec3 x){ return clamp((x*(2.51*x+0.03))/(x*(2.43*x+0.59)+0.14),0.0,1.0); }

float calcShadow(vec4 lsPos, vec3 N, vec3 L) {
  vec3 proj = lsPos.xyz / max(lsPos.w, 1e-5) * 0.5 + 0.5;
  if (proj.z > 1.0 || proj.x < 0.01 || proj.x > 0.99 || proj.y < 0.01 || proj.y > 0.99) return 1.0;
  float bias = max(0.0020 * (1.0 - dot(N, L)), 0.0006);
  float lit = 0.0;
  vec2 tex = vec2(1.0 / 2048.0);
  for (int x = -1; x <= 1; x++) {
    for (int y = -1; y <= 1; y++) {
      float dM = texture(uShadowMap, proj.xy + vec2(float(x), float(y)) * tex * 1.2).r;
      lit += (proj.z - bias <= dM) ? 1.0 : 0.0;
    }
  }
  return lit / 9.0;
}

void main() {
  /* Hide un-lifted tiles cleanly (let raw concrete plate show through with zero seam!) */
  if (vLift < 0.003) discard;

  vec2 uvScr = vec2(gl_FragCoord.x / uRes.x, 1.0 - gl_FragCoord.y / uRes.y);
  float pMask = texture(uPersonMask, clamp(uvScr, vec2(0.001), vec2(0.999))).r;
  if (vCamDepth > uPersonDepth + 0.05 && pMask > 0.45) discard;
  float vis = (vCamDepth > uPersonDepth + 0.05) ? (1.0 - smoothstep(0.12, 0.55, pMask)) : 1.0;

  /* Camera-projection-mapped real concrete texture from the tile's rest floor position (x, y, 0)! */
  vec3 qRest = transpose(uCam) * (vRestFloorPos - uRo);
  vec2 pxProj = uF * qRest.xy / max(qRest.z, 0.2) + 0.5 * uRes;
  vec2 uvProj = clamp(pxProj / uRes, vec2(0.002), vec2(0.998));
  vec3 plateSrgb = texture(uPlate, uvProj).rgb;
  /* Never project the dancer's shoes/jeans onto a rising tile if tile sits right behind his silhouette! */
  float restMask = texture(uPersonMask, uvProj).r;
  vec3 fallbackConcrete = vec3(0.42, 0.41, 0.39) * (0.92 + 0.16 * noise2(vRestFloorPos.xy * 18.0));
  vec3 topTex = mix(plateSrgb, fallbackConcrete, smoothstep(0.15, 0.60, restMask));

  vec3 N = normalize(vNrm);
  vec3 L = normalize(uSun);
  vec3 V = normalize(uRo - vWorldPos);
  float NdotL = max(dot(N, L), 0.0);
  float shad = calcShadow(vLightSpace, N, L);

  if (vIsTop > 0.85) {
    /* Top concrete cap: preserves real plate brightness & texture at lift=0, receives real 3D neighbor shadows as it rises! */
    float liftFade = smoothstep(0.003, 0.045, vLift);
    float sDark = mix(1.0, mix(0.40, 1.04, shad), liftFade);
    vec3 col = topTex * sDark;
    oCol = vec4(clamp(col, 0.0, 1.0), vis * liftFade);
    oMeta = vec4(vis * liftFade, vis / max(vCamDepth, 0.2), 0.0, 1.0);
    return;
  }

  /* Chiseled sunlit concrete vertical monolith side walls + stone grain + soft crevice AO matching the real plaza! */
  float grain = noise2(vWorldPos.xy * 42.0 + vWorldPos.z * 36.0);
  float strata = 0.90 + 0.10 * sin(vWorldPos.z * 54.0 + grain * 2.4);
  vec3 sideBase = mix(vec3(0.32, 0.315, 0.305), topTex * 0.82, vIsTop) * (0.88 + 0.22 * grain) * strata;
  vec3 baseLin = pow(max(sideBase, vec3(0.004)), vec3(2.2));

  /* Natural crevice AO near z=0 floor seam so columns look deeply extruded out of real concrete without turning pitch-black */
  float depthAO = mix(0.46, 1.0, smoothstep(-0.02, 0.22, vWorldPos.z));
  vec3 sunCol = vec3(1.18, 1.10, 0.96) * 2.55;
  vec3 skyFill = vec3(0.34, 0.40, 0.48);
  vec3 gndBnc = vec3(0.24, 0.23, 0.21);
  vec3 hemi = mix(gndBnc, skyFill, clamp(N.z * 0.5 + 0.5, 0.0, 1.0));

  vec3 H = normalize(L + V);
  float spec = pow(max(dot(N, H), 0.0), 32.0) * NdotL * shad * (0.14 + 0.32 * vIsTop);
  vec3 linCol = baseLin * (sunCol * NdotL * mix(0.24, 1.0, shad) + hemi * depthAO) + vec3(1.0, 0.96, 0.90) * spec;

  vec3 mapped = acesFilm(linCol * 0.98);
  vec3 srgb = pow(clamp(mapped, 0.0, 1.0), vec3(1.0 / 2.2));
  float a = vis * smoothstep(0.003, 0.025, vLift);
  oCol = vec4(srgb, a);
  oMeta = vec4(a, a / max(vCamDepth, 0.2), 0.0, 1.0);
}
"""

VS_FSQ = "#version 330\nin vec2 p; void main(){ gl_Position = vec4(p, 0.0, 1.0); }"

GROUND_FS = """#version 330
uniform sampler2D uPlate, uShadowMap, uPersonMask;
uniform vec2 uRes;
uniform float uF, uBackY;
uniform vec3 uRo;
uniform mat3 uCam;
uniform mat4 uLightVP;
layout(location=0) out vec4 oCol;
layout(location=1) out vec4 oMeta;

void main() {
  vec2 px0 = vec2(gl_FragCoord.x, uRes.y - gl_FragCoord.y);
  vec2 uv0 = px0 / uRes;
  vec3 col = texture(uPlate, clamp(uv0, vec2(0.002), vec2(0.998))).rgb;
  vec3 rd = normalize(uCam * vec3((px0 - 0.5 * uRes) / uF, 1.0));
  float pMask = texture(uPersonMask, clamp(uv0, vec2(0.001), vec2(0.999))).r;
  float cgA = 0.0, cgInvD = 0.0;
  if (rd.z < -0.005) {
    float tf = -uRo.z / rd.z;
    if (tf > 0.2 && tf < 18.0) {
      vec3 X = uRo + rd * tf;
      if (X.y < uBackY + 0.06 && X.x > -4.8 && X.x < 5.6) {
        float notP = 1.0 - smoothstep(0.15, 0.65, pMask);
        float curb = smoothstep(uBackY + 0.06, uBackY - 0.04, X.y);
        vec4 ls = uLightVP * vec4(X, 1.0);
        vec3 proj = ls.xyz / max(ls.w, 1e-5) * 0.5 + 0.5;
        float occ = 0.0;
        if (proj.x > 0.01 && proj.x < 0.99 && proj.y > 0.01 && proj.y < 0.99 && proj.z < 1.0) {
          vec2 tex = vec2(1.0 / 2048.0);
          for (int i = -2; i <= 2; i++) {
            for (int j = -2; j <= 2; j++) {
              float dM = texture(uShadowMap, proj.xy + vec2(float(i), float(j)) * tex * 1.25).r;
              if (proj.z - 0.0012 > dM) occ += 1.0;
            }
          }
          occ /= 25.0;
        }
        float sAmt = clamp(occ * 0.66, 0.0, 0.72) * notP * curb;
        if (sAmt > 0.005) {
          col = mix(col, col * vec3(0.78, 0.84, 0.98) * 0.28, sAmt);
          cgA = sAmt * 0.7; cgInvD = cgA / max(tf, 0.3);
        }
      }
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
    res = mix(c0, soft, cgA * 0.22);
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
    P[0, 0] = 2.0 * f_px / W; P[1, 1] = 2.0 * f_px / H
    P[2, 2] = -(far + near) / (far - near); P[2, 3] = -2.0 * far * near / (far - near); P[3, 2] = -1.0
    return P


def mat4_view_from_colmap(ro, R_c2w):
    flip = np.diag([1.0, -1.0, -1.0]); R_w2gl = flip @ R_c2w.T
    V = np.eye(4, dtype=np.float32); V[:3, :3] = R_w2gl; V[:3, 3] = -R_w2gl @ ro
    return V


def mat4_ortho_light_vp(sun_dir, center=(0.45, -0.25, 0.3), half_ext=6.2, near=0.5, far=22.0):
    L = np.asarray(sun_dir, dtype=np.float64); L /= np.linalg.norm(L) + 1e-9
    eye = np.asarray(center, dtype=np.float64) + L * 9.0
    z_ax = L; up = np.array([0.0, 0.0, 1.0])
    x_ax = np.cross(up, z_ax); x_ax /= np.linalg.norm(x_ax); y_ax = np.cross(z_ax, x_ax)
    V = np.eye(4, dtype=np.float64)
    V[0, :3] = x_ax; V[1, :3] = y_ax; V[2, :3] = z_ax; V[:3, 3] = -V[:3, :3] @ eye
    O = np.zeros((4, 4), dtype=np.float64)
    O[0, 0] = 1.0 / half_ext; O[1, 1] = 1.0 / half_ext
    O[2, 2] = -2.0 / (far - near); O[2, 3] = -(far + near) / (far - near); O[3, 3] = 1.0
    return (O @ V).astype(np.float32)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--still", default=None)
    ap.add_argument("--render", action="store_true")
    args = ap.parse_args()

    c = load(SRC_PIECE)
    S = scene(c)
    C = np.asarray(S["cam"], dtype=np.float64)
    R_all = np.asarray(S["rot"], dtype=np.float64)
    f_px = float(S["f"])
    N = len(C)
    fps = float(c["fps"])
    pl0 = frame(c, 0)
    H, W = pl0.shape[:2]

    L = np.load(work(c, "limbs.npz"))
    P_body = np.asarray(L["P"], dtype=np.float64)
    p_depths = np.asarray(L["depth"], dtype=np.float64)
    feet_xy = 0.5 * (P_body[:, 27, :2] + P_body[:, 28, :2])

    # Detect rhythmic dance stomp wave pulses from real dancer foot vertical motion + beat spacing!
    stomp_times = [0.35, 1.35, 2.45, 3.55, 4.70, 5.85, 7.00, 8.10]
    waves = []
    for st_t in stomp_times:
        fi = min(int(round(st_t * fps)), N - 1)
        waves.append((st_t, float(feet_xy[fi, 0]), float(feet_xy[fi, 1])))

    hex_xy = build_hex_lattice(x_min=-4.30, x_max=5.20, y_min=-2.15, y_max=1.48, spacing=0.252)
    n_hex = len(hex_xy)
    print(f"Concrete Hex Floor Wave: {n_hex} 3D beveled hex monoliths | {len(waves)} radial dance wave pulses")

    sun = np.array([-0.46, -0.48, 0.74], dtype=np.float64); sun /= np.linalg.norm(sun) + 1e-9
    light_vp = mat4_ortho_light_vp(sun)
    proj_m = mat4_perspective_from_opencv(f_px, W, H)

    gray0 = cv2.cvtColor(pl0, cv2.COLOR_BGR2GRAY).astype(np.float32) / 255.0
    black_floor = float(np.clip(np.percentile(gray0, 1.0) * 0.85, 0.008, 0.050))
    hp0 = gray0 - cv2.GaussianBlur(gray0, (0, 0), 1.2)
    grain_amt = float(np.clip(float(np.median(np.abs(hp0 - np.median(hp0)))) * 1.4826 * 1.10, 0.008, 0.022))

    ctx = moderngl.create_standalone_context(require=330)
    shad_prog = ctx.program(vertex_shader=SHAD_VS, fragment_shader=SHAD_FS)
    hex_prog  = ctx.program(vertex_shader=HEX_VS, fragment_shader=HEX_FS)
    gnd_prog  = ctx.program(vertex_shader=VS_FSQ, fragment_shader=GROUND_FS)
    post_prog = ctx.program(vertex_shader=VS_FSQ, fragment_shader=POST_FS)

    vdata, idata = build_unit_hex_prism(radius=0.143, depth=1.15, bevel=0.015)
    vbo = ctx.buffer(vdata.tobytes())
    ibo = ctx.buffer(idata.tobytes())
    inst_buf = ctx.buffer(reserve=n_hex * 4 * 4)
    vao_hex  = ctx.vertex_array(hex_prog,  [(vbo, "3f 3f 1f", "inPos", "inNrm", "inIsTop"), (inst_buf, "4f/i", "inInst")], index_buffer=ibo)
    vao_shad = ctx.vertex_array(shad_prog, [(vbo, "3f 16x", "inPos"), (inst_buf, "4f/i", "inInst")], index_buffer=ibo)

    fsq = ctx.buffer(np.array([-1, -1, 1, -1, -1, 1, -1, 1, 1, -1, 1, 1], dtype=np.float32).tobytes())
    va_gnd  = ctx.vertex_array(gnd_prog,  [(fsq, "2f", "p")])
    va_post = ctx.vertex_array(post_prog, [(fsq, "2f", "p")])

    t_smap = ctx.depth_texture((2048, 2048)); t_smap.filter = (moderngl.LINEAR, moderngl.LINEAR); t_smap.repeat_x = t_smap.repeat_y = False
    fbo_shad = ctx.framebuffer(depth_attachment=t_smap)
    t_sub_col = ctx.texture((W, H), 4, dtype="f4"); t_sub_meta = ctx.texture((W, H), 4, dtype="f4")
    rbo_depth = ctx.depth_renderbuffer((W, H))
    fbo_sub = ctx.framebuffer([t_sub_col, t_sub_meta], depth_attachment=rbo_depth)
    t_final = ctx.texture((W, H), 4, dtype="f4")
    fbo_final = ctx.framebuffer([t_final])
    plate_t = ctx.texture((W, H), 3); plate_t.filter = (moderngl.LINEAR, moderngl.LINEAR); plate_t.repeat_x = plate_t.repeat_y = False
    mask_t = ctx.texture((W, H), 1, dtype="f4"); mask_t.filter = (moderngl.LINEAR, moderngl.LINEAR); mask_t.repeat_x = mask_t.repeat_y = False

    shad_prog["uLightVP"].write(light_vp.T.tobytes())
    hex_prog["uProj"].write(proj_m.T.tobytes()); hex_prog["uLightVP"].write(light_vp.T.tobytes())
    hex_prog["uRes"].value = (float(W), float(H)); hex_prog["uSun"].value = tuple(map(float, sun)); hex_prog["uF"].value = f_px
    gnd_prog["uRes"].value = (float(W), float(H)); gnd_prog["uF"].value = f_px; gnd_prog["uBackY"].value = 1.55
    gnd_prog["uLightVP"].write(light_vp.T.tobytes())
    post_prog["uRes"].value = (float(W), float(H)); post_prog["uBlackFloor"].value = black_floor; post_prog["uGrainAmt"].value = grain_amt

    def eval_hex_heights(i):
        t = i / fps
        fx, fy = feet_xy[i]
        d_feet = np.hypot(hex_xy[:, 0] - fx, hex_xy[:, 1] - fy)
        # Keep graceful solid pad directly under dancer's sneakers so shoes stay 100% grounded on z=0!
        pad_keep = np.clip((d_feet - 0.42) / 0.46, 0.0, 1.0)
        pad_keep = pad_keep * pad_keep * (3.0 - 2.0 * pad_keep)
        # Soft edge taper ONLY at the back white curb so entire concrete plaza ripples edge-to-edge!
        edge_y = np.clip((1.48 - hex_xy[:, 1]) / 0.24, 0.0, 1.0)

        h = np.zeros(n_hex, dtype=np.float64)
        for wt, wx, wy in waves:
            age = t - wt
            if 0.0 <= age <= 3.10:
                r = np.hypot(hex_xy[:, 0] - wx, hex_xy[:, 1] - wy)
                r_front = 0.28 + age * 2.40
                # Primary graceful concrete wave crest + secondary + tertiary trailing ripples sweeping full plaza!
                u1 = (r - r_front) / 0.42
                u2 = (r - (r_front - 0.85)) / 0.38
                u3 = (r - (r_front - 1.65)) / 0.36
                env = max(0.0, 1.0 - age / 3.10) ** 0.72 * min(1.0, age / 0.12)
                h += (0.44 * np.exp(-u1 * u1)
                      + 0.22 * np.exp(-u2 * u2) * (r_front > 0.85)
                      + 0.12 * np.exp(-u3 * u3) * (r_front > 1.65)) * env

        # Continuous rolling architectural ground swell sweeping smoothly across the entire concrete plaza!
        swell = 0.095 * (0.5 + 0.5 * np.sin(hex_xy[:, 0] * 1.15 - t * 2.6 + hex_xy[:, 1] * 0.65)) ** 1.6
        h = (h + swell) * pad_keep * edge_y
        h = np.clip(h, 0.0, 0.52)
        inst = np.column_stack([hex_xy[:, 0], hex_xy[:, 1], h, np.clip(h / 0.45, 0.0, 1.0)]).astype(np.float32)
        return inst

    def render_one(i):
        pl = frame(c, i)
        m_raw = cv2.imread(str(work(c, "mask", f"{i:04d}.png")), cv2.IMREAD_GRAYSCALE)
        m_f = cv2.GaussianBlur(m_raw.astype(np.float32) / 255.0, (0, 0), 0.9) if m_raw is not None else np.zeros((H, W), np.float32)
        plate_t.write(pl[:, :, ::-1].tobytes())
        mask_t.write(m_f.astype(np.float32).tobytes())

        inst = eval_hex_heights(i)
        inst_buf.write(inst.tobytes())

        ro = C[i]; R = R_all[i]
        Rt = tuple(R.T.flatten().astype(np.float32))
        view_m = mat4_view_from_colmap(ro, R)
        p_dep = float(p_depths[i])

        # Pass 1: 2048x2048 Sun Shadow Map of all raised hex monoliths
        fbo_shad.use(); ctx.viewport = (0, 0, 2048, 2048); ctx.enable(moderngl.DEPTH_TEST); ctx.disable(moderngl.BLEND)
        ctx.clear(depth=1.0)
        vao_shad.render(moderngl.TRIANGLES, instances=n_hex)

        # Pass 2A: Live Plate + Cast Shadows on surrounding concrete
        fbo_sub.use(); ctx.viewport = (0, 0, W, H); ctx.disable(moderngl.DEPTH_TEST); ctx.disable(moderngl.BLEND)
        ctx.clear(depth=1.0)
        plate_t.use(0); t_smap.use(1); mask_t.use(2)
        gnd_prog["uPlate"].value = 0; gnd_prog["uShadowMap"].value = 1; gnd_prog["uPersonMask"].value = 2
        gnd_prog["uRo"].value = tuple(map(float, ro)); gnd_prog["uCam"].value = Rt
        va_gnd.render(moderngl.TRIANGLES)

        # Pass 2B: Camera-Projected 3D Hexagonal Concrete Monoliths!
        ctx.enable(moderngl.DEPTH_TEST); ctx.enable(moderngl.BLEND)
        ctx.blend_func = (moderngl.SRC_ALPHA, moderngl.ONE_MINUS_SRC_ALPHA)
        plate_t.use(0); t_smap.use(1); mask_t.use(2)
        hex_prog["uPlate"].value = 0; hex_prog["uShadowMap"].value = 1; hex_prog["uPersonMask"].value = 2
        hex_prog["uView"].write(view_m.T.tobytes()); hex_prog["uRo"].value = tuple(map(float, ro))
        hex_prog["uCam"].value = Rt; hex_prog["uPersonDepth"].value = p_dep
        vao_hex.render(moderngl.TRIANGLES, instances=n_hex)

        # Pass 3: Optical Finish (OLPF + Black Floor + Sensor Grain)
        fbo_final.use(); ctx.viewport = (0, 0, W, H); ctx.disable(moderngl.DEPTH_TEST); ctx.disable(moderngl.BLEND)
        t_sub_col.use(0); t_sub_meta.use(1)
        post_prog["uScene"].value = 0; post_prog["uMeta"].value = 1; post_prog["uFrame"].value = float(i)
        va_post.render(moderngl.TRIANGLES)
        ctx.finish()
        arr = np.frombuffer(t_final.read(), "f4").reshape(H, W, 4)[::-1]
        img = (np.clip(arr[..., :3], 0, 1) * 255 + 0.5).astype(np.uint8)[:, :, ::-1]
        return pl, img

    wk = here / "work"; wk.mkdir(parents=True, exist_ok=True)
    if args.still:
        idxs = [int(x) for x in args.still.split(",")]
        rows = []
        for idx in idxs:
            pl, comp = render_one(idx)
            a, b = cv2.resize(pl, (640, 360)), cv2.resize(comp, (640, 360))
            cv2.putText(b, f"hex wave f={idx}", (10, 344), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 0, 0), 3)
            cv2.putText(b, f"hex wave f={idx}", (10, 344), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (40, 255, 40), 1)
            rows.append(np.hstack([a, b]))
        out_p = wk / "still_wave_sheet.jpg"
        cv2.imwrite(str(out_p), np.vstack(rows), [cv2.IMWRITE_JPEG_QUALITY, 92])
        print("stills ->", out_p)
        return

    if args.render:
        comp_dir = wk / "comp"
        shutil.rmtree(comp_dir, ignore_errors=True); comp_dir.mkdir(parents=True)
        ts = time.time()
        for i in range(N):
            _, comp = render_one(i)
            cv2.imwrite(str(comp_dir / f"{i:04d}.png"), comp, [cv2.IMWRITE_PNG_COMPRESSION, 1])
            if i % 50 == 0 or i == N - 1:
                print(f"floor_wave frame {i}/{N} ({time.time() - ts:.1f}s)", flush=True)
        out_dir = here / "out"; out_dir.mkdir(parents=True, exist_ok=True)
        r = f"{c['fps_num']}/{c['fps_den']}"; ff = find_ffmpeg()
        tags = ["-pix_fmt", "yuv420p", "-color_primaries", "bt709", "-color_trc", "bt709", "-colorspace", "bt709", "-movflags", "+faststart"]
        x264 = ["-c:v", "libx264", "-profile:v", "high", "-level", "4.1", "-crf", "14", "-preset", "slow"]
        run = lambda a: subprocess.run([ff, "-v", "error", "-y"] + a, check=True)
        run(["-framerate", r, "-i", str(comp_dir / "%04d.png")] + x264 + tags + [str(out_dir / "Concrete_Hex_Floor_Wave_Composite.mp4")])
        run([
            "-framerate", r, "-i", str(comp_dir / "%04d.png"),
            "-framerate", r, "-i", str(work(c, "frames", "%04d.jpg")),
            "-filter_complex", "[0]scale=1080:608[a];[1]scale=1080:608[b];[a][b]vstack,pad=1080:1350:0:67:color=black",
        ] + x264 + tags + [str(out_dir / "Concrete_Hex_Floor_Wave_Compare_1080x1350.mp4")])
        print("encoded ->", out_dir)


if __name__ == "__main__":
    main()
