"""Minecraft-Style Falling Block Shatter + 3D Floating Spinning Pixel Loot Drops (MuJoCo 4.0 + Native OpenGL 3.3 PBR).

    python render_mc_drops.py [piece_dir] --still 25,65,115,170,225   quick look -> work/still_mc_sheet.jpg
    python render_mc_drops.py [piece_dir] --resim --still 25,115,225  re-run MuJoCo sim + stills
    python render_mc_drops.py [piece_dir] --out out                   full render -> out/Minecraft_Shatter_Pixel_Drops_*.mp4

Choreography & Physics:
- 11 beveled 3D pixel-textured blocks & wooden loot crates drop from the sky onto the sunlit concrete plaza (Pexels 32109492).
- Impact Break Motion: On slamming z=0, each block executes a snappy 4-frame impact squash + jagged pixelated crack overlay
  (Minecraft break stages 1..9) + camera impact micro-shake + sunlit concrete dust puff, then DETONATES into:
    1) 16 real MuJoCo 4.0.0 rigid-body 3D voxel debris cubes bouncing, colliding & skidding across the concrete tiles
       (before popping cleanly away after ~2.4s like authentic game break particles).
    2) 1 Hero 3D Extruded 16x16 Pixel-Art Item Drop (Diamond, Diamond Sword, Golden Apple, Diamond Pickaxe, Emerald,
       Ruby Heart, Crystal Potion, Gold Ingot, Golden Star) popping upward in a parabolic loot arc, catching gently at
       z=0.34m, and bobbing + spinning smoothly in mid-air with a breathing circular ground shadow + real sun shadow!
- Linear High-Contrast Midday Sun PBR + Live-Plate IBL + ACES Filmic + 2048x2048 PCF Shadow Map + 9-Slice Motion Blur
  + MediaPipe Person Depth Occlusion + OLPF + Plate Sensor Grain. ZERO text/HUD overlays.
"""
import json, math, shutil, subprocess, sys, time
from pathlib import Path
import numpy as np, cv2, moderngl, mujoco
from scipy.spatial.transform import Rotation as Rot

here = Path(__file__).resolve().parent
root = here.parents[1]
sys.path.insert(0, str(here))
sys.path.insert(0, str(here.parent / "shader_live" / "pipeline"))
from slv import load, work, frame, scene  # noqa: E402
from voxel_items import build_all_minecraft_meshes  # noqa: E402


SHUTTER = [0.0, 0.045, 0.09, 0.135, 0.18, 0.225, 0.27, 0.315, 0.36]  # 9 smooth cinema shutter slices!


def find_ffmpeg():
    tl = root / "tools.local.json"
    if tl.exists():
        d = json.loads(tl.read_text(encoding="utf-8"))
        p = d.get("ffmpeg") or (d.get("tools") or {}).get("ffmpeg")
        if p and Path(p).exists():
            return str(p)
    return shutil.which("ffmpeg") or "ffmpeg"


# 11 Choreographed Falling 3D Blocks -> 5 FOREGROUND (in front of dancer, y in [-1.02, -0.76], depth < person_depth!)
# + 6 MIDGROUND/BACKGROUND (behind dancer, y in [+0.42, +0.94]) for rich true 3D foreground/background depth layering!
DROP_SCHEDULE = [
    # (drop_f, block_kind, item_kind, x_drop, y_drop, z_start, vx, vy, vz0, hover_dx, hover_dy, z_hover)
    (  6, "diamond_ore",  "item_diamond", -0.82, -0.92, 3.8,  0.03, -0.02, -1.9,  0.04, -0.03, 0.31),  # FG front of dancer!
    ( 20, "wood_crate",   "item_sword",    0.72, -0.86, 3.9, -0.04, -0.02, -1.9,  0.04, -0.02, 0.35),  # FG center-right!
    ( 36, "gold_ore",     "item_gapple",   0.18,  0.65, 4.0,  0.03,  0.04, -1.9,  0.05,  0.04, 0.36),  # BG center
    ( 50, "emerald_ore",  "item_emerald", -1.42, -0.98, 3.9,  0.04, -0.02, -1.9,  0.04, -0.02, 0.31),  # FG front-left of dancer!
    ( 66, "wood_crate",   "item_pickaxe",  1.58,  0.42, 3.9, -0.05,  0.06, -1.9, -0.03,  0.04, 0.38),  # BG far right
    ( 82, "brick_block",  "item_heart",   -0.12, -0.96, 3.8, -0.03, -0.02, -1.9,  0.02, -0.02, 0.32),  # FG dead center!
    ( 98, "obsidian_box", "item_potion",  -1.72,  0.46, 3.9,  0.05,  0.05, -1.9,  0.04,  0.03, 0.35),  # BG far left
    (114, "gold_ore",     "item_ingot",    1.36, -0.76, 4.0, -0.04, -0.02, -1.9,  0.03, -0.02, 0.32),  # FG right!
    (130, "diamond_ore",  "item_star",    -0.55,  0.88, 3.9,  0.04,  0.04, -1.9, -0.03,  0.04, 0.37),  # BG behind dancer
    (146, "wood_crate",   "item_diamond",  0.88,  0.94, 3.9, -0.04,  0.03, -1.9,  0.04,  0.03, 0.36),  # BG right
    (162, "obsidian_box", "item_gapple",  -1.15,  0.76, 3.9,  0.03,  0.04, -1.9, -0.03,  0.04, 0.36),  # BG behind dancer
]

N_DEBRIS_PER_BLOCK = 16
CRACK_HOLD_S = 0.16   # Snappy 4-frame (0.16s) impact squash + Minecraft pixel-crack flash before explosive break!
# Debris stays permanently piled on the sunlit concrete plaza through the final frame (zero fade/disappear)!


def simulate_minecraft_drops(c, meshes, person_xy_track):
    """Run real MuJoCo 4.0.0 drop + crack-hitstop + voxel shatter + floating item pop-arc simulation."""
    cfg = c.get("drop_sim", {})
    seed = int(cfg.get("seed", 2041))
    back_y = float(cfg.get("back_wall_y", 1.55))
    grav = float(cfg.get("gravity", -9.81))
    substeps = int(cfg.get("substeps", 40))
    fps = float(c.get("fps_num", 25)) / float(c.get("fps_den", 1))
    N = int(c.get("frames", 240))
    dt = 1.0 / fps / substeps
    rng = np.random.default_rng(seed)

    # Register falling blocks (0..M-1) and their pre-allocated MuJoCo voxel debris cubes (M..B-1)
    bodies = []
    block_body_ids = []
    block_debris_ids = {}
    for bi, item in enumerate(DROP_SCHEDULE):
        _, bkind, _, _, _, _, _, _, _, _, _, _ = item
        he = meshes[bkind][1]
        block_body_ids.append(len(bodies))
        bodies.append({"name": f"blk{bi}", "mesh": bkind, "half_ext": he, "mass": 14.0, "is_block": True})
        d_ids = []
        for k in range(N_DEBRIS_PER_BLOCK):
            mname = f"deb_{bkind}_{k % 6}"
            dhe = meshes[mname][1]
            d_ids.append(len(bodies))
            bodies.append({"name": f"deb{bi}_{k}", "mesh": mname, "half_ext": dhe, "mass": 0.45, "is_block": False})
        block_debris_ids[bi] = d_ids

    n_bodies = len(bodies)
    print(f"MuJoCo 4.0.0 Minecraft Sim: {len(DROP_SCHEDULE)} falling blocks + {n_bodies - len(DROP_SCHEDULE)} voxel debris cubes + {len(DROP_SCHEDULE)} floating pixel items")

    xml = [
        f'<mujoco><option timestep="{dt:.7f}" gravity="0 0 {grav:.3f}" integrator="implicitfast" iterations="35"/>',
        '<default><geom solref="0.006 0.48" solimp="0.92 0.98 0.001" friction="0.88 0.05 0.012"/></default>',
        '<worldbody>',
        '<geom name="ground" type="box" size="12 12 0.25" pos="0 0 -0.25" contype="1" conaffinity="1" friction="0.96 0.08 0.025"/>',
        f'<geom name="back_wall" type="box" size="8 0.25 4.0" pos="0 {back_y + 0.25:.3f} 4.0" contype="1" conaffinity="1"/>',
        '<geom name="front_lim" type="box" size="8 0.25 0.35" pos="0 -1.78 0.35" contype="1" conaffinity="1"/>',
        '<geom name="left_lim" type="box" size="0.25 6.0 4.0" pos="-2.85 0.2 4.0" contype="1" conaffinity="1"/>',
        '<geom name="right_lim" type="box" size="0.25 6.0 4.0" pos="2.75 0.2 4.0" contype="1" conaffinity="1"/>',
        f'<body name="dancer" pos="{person_xy_track[0,0]:.3f} {person_xy_track[0,1]:.3f} 0.85" mocap="true">'
        '<geom type="cylinder" size="0.34 0.85" contype="1" conaffinity="1"/></body>',
    ]
    park_pos = []
    for idx_b, b in enumerate(bodies):
        hx, hy, hz = b["half_ext"]
        px = 40.0 + (idx_b % 16) * 2.0
        py = 40.0 + (idx_b // 16) * 2.0
        pz = 25.0
        park_pos.append((px, py, pz))
        xml.append(
            f'<body name="{b["name"]}" pos="{px:.3f} {py:.3f} {pz:.3f}"><freejoint name="j{idx_b}"/>'
            f'<geom name="g{idx_b}" type="box" size="{max(hx,0.015):.4f} {max(hy,0.015):.4f} {max(hz,0.015):.4f}" '
            f'mass="{b["mass"]:.3f}" contype="1" conaffinity="1"/></body>'
        )
    xml.append('</worldbody></mujoco>')
    model = mujoco.MjModel.from_xml_string("\n".join(xml))
    data = mujoco.MjData(model)

    qpos_adr = [model.jnt_qposadr[mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, f"j{i}")] for i in range(n_bodies)]
    qvel_adr = [model.jnt_dofadr[mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, f"j{i}")] for i in range(n_bodies)]
    dancer_mc = model.body_mocapid[mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "dancer")]

    active = np.zeros(n_bodies, dtype=bool)
    debris_born_t = np.full(n_bodies, -100.0)
    # Per-block state: 0=waiting, 1=falling, 2=cracking (impact squash + crack overlay!), 3=shattered
    blk_state = np.zeros(len(DROP_SCHEDULE), dtype=int)
    blk_hit_t = np.full(len(DROP_SCHEDULE), -100.0)
    blk_hit_pos = np.zeros((len(DROP_SCHEDULE), 3), dtype=np.float64)
    blk_hit_quat = np.tile(np.array([1.0, 0.0, 0.0, 0.0]), (len(DROP_SCHEDULE), 1))
    impacts = []       # [t_sec, x, y, energy, is_break]
    loot_events = []   # dict per popped floating 3D pixel item!

    def activate_body(bi, pos, quat, linvel, angvel, t_now):
        qa = qpos_adr[bi]; va = qvel_adr[bi]
        data.qpos[qa:qa + 3] = pos
        data.qpos[qa + 3:qa + 7] = quat
        data.qvel[va:va + 3] = linvel
        data.qvel[va + 3:va + 6] = angvel
        active[bi] = True
        debris_born_t[bi] = t_now

    def deactivate_body(bi):
        qa = qpos_adr[bi]; va = qvel_adr[bi]
        data.qpos[qa:qa + 3] = park_pos[bi]
        data.qpos[qa + 3:qa + 7] = (1.0, 0.0, 0.0, 0.0)
        data.qvel[va:va + 6] = 0.0
        active[bi] = False

    shutter_steps = [int(round(s * substeps)) for s in SHUTTER]
    n_shut = len(SHUTTER)
    out_pos = np.zeros((N, n_shut, n_bodies, 3), dtype=np.float32)
    out_quat = np.zeros((N, n_shut, n_bodies, 4), dtype=np.float32)
    out_scale = np.zeros((N, n_shut, n_bodies, 3), dtype=np.float32)  # xyz scale (for impact squash & pop-out shrink!)
    out_crack = np.zeros((N, n_shut, n_bodies), dtype=np.float32)     # 0..1 pixelated Minecraft crack stage!

    t0 = time.time()
    for f_idx in range(N):
        for hi, item in enumerate(DROP_SCHEDULE):
            drop_f, bkind, ikind, xd, yd, zs, vx, vy, vz0, _, _, _ = item
            if f_idx == drop_f:
                hbi = block_body_ids[hi]
                # Gentle natural tumble as block drops from sky (looks heavy & readable!)
                q_init = Rot.from_euler("xyz", [rng.uniform(-0.35, 0.35), rng.uniform(-0.35, 0.35), rng.uniform(0, 6.28)]).as_quat()
                activate_body(hbi, np.array([xd, yd, zs]), np.array([q_init[3], q_init[0], q_init[1], q_init[2]]),
                              np.array([vx, vy, vz0]), rng.uniform(-1.6, 1.6, 3), f_idx / fps)
                blk_state[hi] = 1

        pxy0 = person_xy_track[f_idx]
        pxy1 = person_xy_track[min(f_idx + 1, N - 1)]

        for sub in range(substeps):
            u = sub / float(substeps)
            t_now = (f_idx + u) / fps
            pxy = pxy0 * (1.0 - u) + pxy1 * u
            data.mocap_pos[dancer_mc] = [pxy[0], pxy[1], 0.85]

            # Record shutter slice
            if sub in shutter_steps:
                si = shutter_steps.index(sub)
                for bi in range(n_bodies):
                    if not active[bi]:
                        continue
                    qa = qpos_adr[bi]
                    out_pos[f_idx, si, bi] = data.qpos[qa:qa + 3]
                    out_quat[f_idx, si, bi] = data.qpos[qa + 3:qa + 7]
                    if bodies[bi]["is_block"]:
                        hi_m = block_body_ids.index(bi)
                        if blk_state[hi_m] == 2:
                            # IMPACT SQUASH + PIXELATED CRACK STAGE (0.25 -> 1.0) + micro tremble!
                            cu = np.clip((t_now - blk_hit_t[hi_m]) / CRACK_HOLD_S, 0.0, 1.0)
                            # Elastic impact squash right on contact, then swells right before POP!
                            sq_z = 0.78 + 0.14 * math.sin(cu * math.pi) + 0.06 * cu
                            sq_xy = 1.0 / math.sqrt(max(sq_z, 0.5))
                            # Tiny rapid high-freq tremble right before bursting
                            tr = np.array([math.sin(t_now * 180.0 + hi_m), math.cos(t_now * 210.0 + hi_m), 0.0]) * 0.012 * cu
                            out_pos[f_idx, si, bi] = blk_hit_pos[hi_m] + tr
                            out_pos[f_idx, si, bi, 2] = bodies[bi]["half_ext"][2] * sq_z
                            out_quat[f_idx, si, bi] = blk_hit_quat[hi_m]
                            out_scale[f_idx, si, bi] = (sq_xy, sq_xy, sq_z)
                            out_crack[f_idx, si, bi] = 0.28 + 0.72 * cu
                        else:
                            out_scale[f_idx, si, bi] = (1.0, 1.0, 1.0)
                            out_crack[f_idx, si, bi] = 0.0
                    else:
                        # Permanent voxel debris cube: remains at full size resting & piling across the concrete plaza!
                        out_scale[f_idx, si, bi] = (1.0, 1.0, 1.0)

            # Pin parked bodies AND cracking hitstop blocks cleanly (never deactivate resting debris!)
            for bi in range(n_bodies):
                if not active[bi]:
                    qa = qpos_adr[bi]; va = qvel_adr[bi]
                    data.qpos[qa:qa + 3] = park_pos[bi]
                    data.qvel[va:va + 6] = 0.0

            for hi in range(len(DROP_SCHEDULE)):
                if blk_state[hi] == 2:
                    hbi = block_body_ids[hi]
                    qa = qpos_adr[hbi]; va = qvel_adr[hbi]
                    data.qpos[qa:qa + 3] = blk_hit_pos[hi]
                    data.qpos[qa + 3:qa + 7] = blk_hit_quat[hi]
                    data.qvel[va:va + 6] = 0.0

            pre_vz = [float(data.qvel[qvel_adr[block_body_ids[hi]] + 2]) for hi in range(len(DROP_SCHEDULE))]
            mujoco.mj_step(model, data)

            # Check falling blocks hitting z=0 -> enter 4-frame Crack Hitstop -> then DETONATE into voxels + floating pixel item!
            for hi, item in enumerate(DROP_SCHEDULE):
                hbi = block_body_ids[hi]
                if blk_state[hi] == 1 and active[hbi]:
                    qa = qpos_adr[hbi]; va = qvel_adr[hbi]
                    pos = data.qpos[qa:qa + 3].copy()
                    vz_in = min(float(data.qvel[va + 2]), pre_vz[hi])
                    hz = bodies[hbi]["half_ext"][2]
                    # Trigger on ANY ground or pileup slam (including rotated cube corner bounce at z up to sqrt(3)*hz!)
                    dv_up = float(data.qvel[va + 2]) - pre_vz[hi]
                    hit_floor = (pos[2] <= hz * 1.88 and pre_vz[hi] < -2.0 and dv_up > 1.2) or (pos[2] <= hz + 0.14 and vz_in < -1.5)
                    if hit_floor:
                        blk_state[hi] = 2
                        blk_hit_t[hi] = t_now
                        blk_hit_pos[hi] = np.array([pos[0], min(back_y - 0.35, pos[1]), hz], dtype=np.float64)
                        # Snap gently upright-ish on concrete slam so crack faces look super crisp & grounded!
                        r_cur = Rot.from_quat([data.qpos[qa+4], data.qpos[qa+5], data.qpos[qa+6], data.qpos[qa+3]]).as_euler("xyz")
                        q_land = Rot.from_euler("xyz", [r_cur[0] * 0.15, r_cur[1] * 0.15, r_cur[2]]).as_quat()
                        blk_hit_quat[hi] = np.array([q_land[3], q_land[0], q_land[1], q_land[2]])
                        impacts.append([float(t_now), float(pos[0]), float(pos[1]), 1.15, 0.0])

                elif blk_state[hi] == 2 and (t_now - blk_hit_t[hi]) >= CRACK_HOLD_S:
                    # DETONATE!! Hide cracked block, burst 16 MuJoCo voxel debris cubes, & pop out 3D floating pixel item!
                    blk_state[hi] = 3
                    deactivate_body(hbi)
                    p0 = blk_hit_pos[hi].copy()
                    impacts.append([float(t_now), float(p0[0]), float(p0[1]), 0.95, 1.0])

                    # 1) Burst 16 non-overlapping chunky 3D voxel break cubes in 2 tiers!
                    d_ids = block_debris_ids[hi]
                    half_n = len(d_ids) // 2
                    for k_i, dbi in enumerate(d_ids):
                        tier = 0 if k_i < half_n else 1
                        ri = k_i if tier == 0 else (k_i - half_n)
                        rn = half_n if tier == 0 else (len(d_ids) - half_n)
                        ang = (ri / float(rn)) * 2.0 * math.pi + tier * 0.40 + float(rng.uniform(-0.14, 0.14))
                        r_off = (0.24 if tier == 0 else 0.16) + float(rng.uniform(0.0, 0.08))
                        z_off = (0.14 if tier == 0 else 0.38) + float(rng.uniform(0.0, 0.10))
                        sp = np.array([p0[0] + math.cos(ang) * r_off, np.clip(p0[1] + math.sin(ang) * r_off, -1.24, back_y - 0.16), z_off])
                        br = float(rng.uniform(0.85, 2.25))
                        bz = float(rng.uniform(1.65, 3.70))
                        sv = np.array([math.cos(ang) * br, math.sin(ang) * br * 0.62, bz])
                        sw = rng.uniform(-14.0, 14.0, 3)
                        sq = Rot.from_euler("xyz", rng.uniform(0, 2 * math.pi, 3)).as_quat()
                        activate_body(dbi, sp, np.array([sq[3], sq[0], sq[1], sq[2]]), sv, sw, t_now)

                    # 2) Spawn Iconic Floating Spinning 3D Pixel Loot Item!
                    _, _, ikind, _, _, _, _, _, _, hdx, hdy, z_hov = item
                    loot_events.append({
                        "t_pop": float(t_now),
                        "item": ikind,
                        "p_start": [float(p0[0]), float(p0[1]), 0.28],
                        "p_hover": [float(p0[0] + hdx), float(min(back_y - 0.25, p0[1] + hdy)), float(z_hov)],
                        "yaw0": float(rng.uniform(0.0, 6.28)),
                        "spin_spd": float(rng.uniform(1.95, 2.55)) * (1.0 if hi % 2 == 0 else -1.0),
                        "bob_phase": float(rng.uniform(0.0, 6.28)),
                    })

    print(f"MuJoCo Minecraft sim done in {time.time() - t0:.2f}s | {len(loot_events)} floating pixel loot items spawned")
    sim_path = work(c, "sim_mc.npz")
    np.savez_compressed(
        sim_path,
        pos=out_pos,
        quat=out_quat,
        scale=out_scale,
        crack=out_crack,
        mesh_names=np.array([b["mesh"] for b in bodies]),
        impacts=np.array(impacts, dtype=np.float32) if impacts else np.zeros((0, 5), dtype=np.float32),
        loot_json=json.dumps(loot_events),
    )
    return sim_path


# ==============================================================================
#  4-PASS LINEAR HIGH-CONTRAST PBR + MINECRAFT PIXEL CRACK + BREATHING HOVER SHADOW
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
in vec3 inPbr;   /* x=roughness, y=metallic, z=ao */

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
uniform sampler2D uPersonMask;
uniform sampler2D uPlate;
uniform vec2 uRes;
uniform vec3 uRo, uSun;
uniform float uPersonDepth;
uniform float uCrackStage;   /* 0=intact, 0.25..1.0=jagged pixelated Minecraft break cracks! */
uniform float uIsLoot;       /* 1.0 for floating pixel loot item (subtle crisp sunlight gleam!) */
uniform int uRenderMode;     /* 0=full PBR, 1=wireframe overlay, 2=clay+shadows */

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
  float bias = max(0.0022 * (1.0 - dot(N, L)), 0.0007);
  float lit = 0.0;
  vec2 tex = vec2(1.0 / 2048.0);
  for (int x = -1; x <= 1; x++) {
    for (int y = -1; y <= 1; y++) {
      float pcfD = texture(uShadowMap, proj.xy + vec2(float(x), float(y)) * tex * 1.15).r;
      lit += (proj.z - bias <= pcfD) ? 1.0 : 0.0;
    }
  }
  return lit / 9.0;
}

vec3 samplePlateIBL(vec3 R) {
  vec2 uvEnv = vec2(clamp(0.5 + R.x * 0.42, 0.02, 0.98), clamp(0.52 - R.z * 0.48, 0.02, 0.98));
  vec3 pCol = pow(texture(uPlate, uvEnv).rgb, vec3(2.2));
  vec3 skyBoost = vec3(0.52, 0.68, 0.88) * smoothstep(0.15, 0.85, R.z) * 1.15;
  float gndDim = mix(0.42, 1.0, smoothstep(-0.55, 0.15, R.z));
  return pCol * gndDim + skyBoost;
}

/* Authentic pixelated 16x16 jagged Minecraft break-stage crack pattern! */
float pixelCrackMask(vec2 uv, float stage) {
  if (stage <= 0.01) return 0.0;
  vec2 p = floor(clamp(uv, 0.0, 0.999) * 16.0) / 16.0;
  vec2 c = p - vec2(0.47, 0.47);
  float r = length(c);
  float ang = atan(c.y, c.x);
  /* 6 jagged radial pixel crack branches radiating from center of block face! */
  float branch = abs(fract((ang + 0.35 * sin(r * 14.0)) * 0.955) - 0.5);
  float branch2 = abs(fract((ang - 0.42 * cos(r * 11.0) + 0.5) * 1.43) - 0.5);
  float reach = stage * 0.68;
  float w = 0.045 + 0.075 * stage;
  float m1 = step(r, reach) * step(branch, w);
  float m2 = step(r, reach * 0.78) * step(branch2, w * 0.85);
  float core = step(r, stage * 0.14);
  return clamp(max(max(m1, m2), core), 0.0, 1.0);
}

void main() {
  vec2 uvTop = vec2(gl_FragCoord.x / uRes.x, 1.0 - gl_FragCoord.y / uRes.y);
  float pMask = texture(uPersonMask, clamp(uvTop, vec2(0.001), vec2(0.999))).r;
  if (vCamDepth > uPersonDepth + 0.06 && pMask > 0.45) discard;
  float vis = (vCamDepth > uPersonDepth + 0.06) ? (1.0 - smoothstep(0.12, 0.55, pMask)) : 1.0;

  if (uRenderMode == 1) {
    vec3 wCol = (uIsLoot > 0.5) ? vec3(1.0, 0.86, 0.18) : vec3(0.12, 0.96, 1.0);
    oCol = vec4(wCol, vis * 0.95);
    oMeta = vec4(vis, vis / max(vCamDepth, 0.2), 0.0, 1.0);
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
    rough = 0.78;
    metal = 0.0;
  }

  /* Crisp subtle surface micro-grain so blocks & floating voxel items feel physically solid in real sunlight */
  float micro = noise2(vWorldPos.xy * 65.0 + vWorldPos.z * 48.0);
  baseSrgb *= (0.94 + 0.12 * micro);

  /* Jagged pixelated Minecraft crack overlay right before explosive shatter! */
  float crack = pixelCrackMask(vUv, uCrackStage);
  if (crack > 0.5) {
    baseSrgb *= 0.10;
    rough = 0.95;
    metal = 0.0;
    vAO *= 0.25;
  }

  /* Subtle micro-bump perturbation for gritty sunlit realism on stone/wood blocks */
  vec3 N = N0;
  if (uIsLoot < 0.5 && rough > 0.5) {
    vec3 dPdx = dFdx(vWorldPos);
    vec3 dPdy = dFdy(vWorldPos);
    float dHdx = dFdx(micro);
    float dHdy = dFdy(micro);
    vec3 surfGrad = cross(dPdy, N0) * dHdx + cross(N0, dPdx) * dHdy;
    float det = dot(dPdx, cross(dPdy, N0));
    if (abs(det) > 1e-8) {
      N = normalize(N0 - sign(det) * clamp(surfGrad / max(abs(det), 1e-5), vec3(-1.2), vec3(1.2)) * 0.045);
    }
  }

  vec3 baseLin = pow(max(baseSrgb, vec3(0.002)), vec3(2.2));
  vec3 H = normalize(L + V);
  float NdotL = max(dot(N, L), 0.0);
  float NdotV = max(dot(N, V), 0.001);
  float NdotH = max(dot(N, H), 0.0);
  float shad = calcShadow(vLightSpace, N0, L);

  /* High-contrast 7:1 outdoor midday sunlight matching the live dancer! */
  vec3 sunCol = vec3(1.20, 1.10, 0.94) * 3.45;
  vec3 skyFill = vec3(0.18, 0.27, 0.39);
  vec3 gndBounce = vec3(0.16, 0.14, 0.12);
  vec3 hemi = mix(gndBounce, skyFill, clamp(N.z * 0.5 + 0.5, 0.0, 1.0));
  /* Floating loot gets clean air ambient (no dark ground floor crush), resting blocks get tight ground AO! */
  float gndAO = (uIsLoot > 0.5) ? 0.92 : mix(0.30, 1.0, smoothstep(0.0, 0.25, vWorldPos.z));
  float totalAO = clamp(vAO * gndAO, 0.14, 1.0);

  vec3 F0 = mix(vec3(0.04), baseLin, metal);
  vec3 F = F0 + (1.0 - F0) * pow(1.0 - NdotV, 5.0);
  float a2 = max(rough * rough, 0.004);
  float denom = NdotH * NdotH * (a2 - 1.0) + 1.0;
  float D_ggx = a2 / (3.14159 * denom * denom + 1e-5);
  float specSun = min(D_ggx * 0.26 * NdotL, 6.5);

  vec3 R = reflect(-V, N);
  vec3 envRefl = samplePlateIBL(R);
  float reflPow = pow(1.0 - rough, 1.85);

  vec3 diffAlb = baseLin * (1.0 - metal * 0.90);
  vec3 linCol = diffAlb * (sunCol * NdotL * shad + hemi * totalAO)
              + F * sunCol * specSun * shad
              + F * envRefl * reflPow * (0.40 + 0.60 * totalAO);

  /* Crisp subtle gemstone / enchanted-gold Fresnel gleam on floating pixel loot items! */
  if (uIsLoot > 0.5 && rough < 0.22) {
    float rim = pow(1.0 - NdotV, 3.2);
    linCol += baseLin * (0.14 + 0.28 * rim);
  }

  vec3 mapped = acesFilm(linCol * 0.90);
  vec3 srgb = pow(clamp(mapped, 0.0, 1.0), vec3(1.0 / 2.2));
  oCol = vec4(srgb, vis);
  oMeta = vec4(vis, vis / max(vCamDepth, 0.2), 0.0, 1.0);
}
"""

VS_FSQ = "#version 330\nin vec2 p; void main(){ gl_Position = vec4(p, 0.0, 1.0); }"

# Ground Shadow Catcher: real 2048x2048 sun shadows + Iconic Minecraft Breathing Circular Hover Shadow under every floating item!
GROUND_SHAD_FS = """#version 330
uniform sampler2D uPlate;
uniform sampler2D uShadowMap;
uniform sampler2D uPersonMask;
uniform vec2 uRes;
uniform float uF, uBackY;
uniform vec3 uRo;
uniform mat3 uCam;
uniform mat4 uLightVP;
uniform vec4 uHoverShad[16];  /* xy = world item XY, z = height z above floor (m), w = alpha (0..1) */
uniform int uNumHover;
uniform int uSkipShadows;     /* 1 = raw plate (gain=1.0) for wireframe pass */

layout(location=0) out vec4 oCol;
layout(location=1) out vec4 oMeta;

void main() {
  vec2 px0 = vec2(gl_FragCoord.x, uRes.y - gl_FragCoord.y);
  vec2 uv0 = px0 / uRes;
  vec3 col = texture(uPlate, clamp(uv0, vec2(0.002), vec2(0.998))).rgb;
  vec3 rd = normalize(uCam * vec3((px0 - 0.5 * uRes) / uF, 1.0));
  float pMask = texture(uPersonMask, clamp(uv0, vec2(0.001), vec2(0.999))).r;

  float cgA = 0.0;
  float cgInvD = 0.0;

  if (uSkipShadows == 0 && rd.z < -0.005) {
    float tf = -uRo.z / rd.z;
    if (tf > 0.2 && tf < 18.0) {
      vec3 X = uRo + rd * tf;
      if (X.y < uBackY + 0.08 && abs(X.x) < 4.5) {
        float notPerson = 1.0 - smoothstep(0.15, 0.65, pMask);
        float curbFade = smoothstep(uBackY + 0.08, uBackY - 0.04, X.y);
        float occ = 0.0;

        /* 1. Real 2048x2048 Directional Midday Sun Shadow Map */
        vec4 ls = uLightVP * vec4(X, 1.0);
        vec3 proj = ls.xyz / max(ls.w, 1e-5) * 0.5 + 0.5;
        if (proj.x > 0.01 && proj.x < 0.99 && proj.y > 0.01 && proj.y < 0.99 && proj.z < 1.0) {
          vec2 tex = vec2(1.0 / 2048.0);
          for (int i = -2; i <= 2; i++) {
            for (int j = -2; j <= 2; j++) {
              float dMap = texture(uShadowMap, proj.xy + vec2(float(i), float(j)) * tex * 1.25).r;
              if (proj.z - 0.0010 > dMap) occ += 1.0;
            }
          }
          occ /= 25.0;
        }

        /* 2. Iconic Minecraft Breathing Circular Hover Shadow directly beneath every floating pixel icon!
           Shrinks & softens when item bobs UP, tightens & darkens when item bobs DOWN -> instant floating depth cue! */
        float hovShad = 0.0;
        for (int k = 0; k < 16; k++) {
          if (k >= uNumHover) break;
          vec4 hs = uHoverShad[k];
          if (hs.w <= 0.01) continue;
          float h = clamp(hs.z, 0.05, 1.4);
          float rad = 0.16 + 0.10 * h;
          float d = length(X.xy - hs.xy);
          float str = hs.w * clamp(1.0 - h * 0.55, 0.18, 0.88);
          hovShad = max(hovShad, exp(-(d * d) / max(rad * rad * 0.45, 0.004)) * str * 0.56);
        }

        float sAmt = clamp(max(occ * 0.72, hovShad), 0.0, 0.80) * notPerson * curbFade;
        if (sAmt > 0.005) {
          vec3 shadTint = vec3(0.76, 0.84, 1.00);
          col = mix(col, col * shadTint * 0.26, sAmt);
          cgA = clamp(sAmt * 0.65, 0.0, 0.7);
          cgInvD = cgA / max(tf, 0.3);
        }
      }
    }
  }
  oCol = vec4(clamp(col, 0.0, 1.0), 1.0);
  oMeta = vec4(cgA, cgInvD, 0.0, 1.0);
}
"""

DUST_VS = """#version 330
in vec2 inQuad;
in vec4 inPosRad;
in vec4 inInfo;
uniform vec2 uRes; uniform float uF; uniform vec3 uRo; uniform mat3 uCam;
out vec2 vLocal; out float vAlpha; out float vSeed; out float vCamDepth;
void main() {
  vLocal = inQuad; vAlpha = inInfo.x; vSeed = inInfo.z;
  vec3 q = transpose(uCam) * (inPosRad.xyz - uRo);
  vCamDepth = q.z;
  if (vAlpha <= 0.008 || q.z <= 0.25) { gl_Position = vec4(2.0, 2.0, 2.0, 1.0); return; }
  float pxR = clamp(uF * inPosRad.w / q.z, 2.0, 320.0);
  float c = cos(inInfo.y), s = sin(inInfo.y);
  vec2 rq = vec2(inQuad.x * c - inQuad.y * s, inQuad.x * s + inQuad.y * c);
  vec2 ctrPx = uF * q.xy / q.z + 0.5 * uRes;
  vec2 px = ctrPx + rq * pxR;
  gl_Position = vec4(px.x / uRes.x * 2.0 - 1.0, (1.0 - px.y / uRes.y) * 2.0 - 1.0, 0.0, 1.0);
}
"""
DUST_FS = """#version 330
in vec2 vLocal; in float vAlpha; in float vSeed; in float vCamDepth;
uniform sampler2D uPersonMask; uniform vec2 uRes; uniform float uPersonDepth;
layout(location=0) out vec4 oCol; layout(location=1) out vec4 oMeta;
float hash21(vec2 p){ p=fract(p*vec2(123.34,456.21)); p+=dot(p,p+45.32); return fract(p.x*p.y); }
float noise2(vec2 p){ vec2 i=floor(p),f=fract(p); f=f*f*(3.0-2.0*f);
  return mix(mix(hash21(i),hash21(i+vec2(1,0)),f.x), mix(hash21(i+vec2(0,1)),hash21(i+vec2(1,1)),f.x), f.y); }
float fbm(vec2 p){ float v=0.0,a=0.5; for(int k=0;k<4;k++){ v+=a*noise2(p); p*=2.05; a*=0.5; } return v; }
void main() {
  float r = length(vLocal); if (r > 1.0) discard;
  vec2 uvTop = vec2(gl_FragCoord.x / uRes.x, 1.0 - gl_FragCoord.y / uRes.y);
  float pMask = texture(uPersonMask, clamp(uvTop, vec2(0.001), vec2(0.999))).r;
  float occ = (vCamDepth > uPersonDepth + 0.06) ? (1.0 - smoothstep(0.15, 0.60, pMask)) : 1.0;
  float n = fbm(vLocal * 2.2 + vec2(vSeed * 13.7, -vSeed * 9.3));
  float env = smoothstep(1.0, 0.15, r) * smoothstep(0.24, 0.78, n + (1.0 - r) * 0.42);
  float a = env * vAlpha * occ;
  if (a < 0.006) discard;
  float sunLit = clamp(0.55 - 0.35 * vLocal.x + 0.45 * vLocal.y, 0.0, 1.0);
  vec3 dustCol = mix(vec3(0.52, 0.48, 0.44), vec3(0.84, 0.80, 0.74), sunLit);
  oCol = vec4(dustCol, a);
  oMeta = vec4(a * 0.6, a * 0.6 / max(vCamDepth, 0.2), 0.0, 1.0);
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
uniform sampler2D uScene, uMeta; uniform vec2 uRes, uShakePx;
uniform float uBlackFloor, uGrainAmt, uFrame;
out vec4 oCol;
float hash12(vec2 p){ vec3 p3=fract(vec3(p.xyx)*0.1031); p3+=dot(p3,p3.yzx+33.33); return fract((p3.x+p3.y)*p3.z); }
void main() {
  vec2 uv0 = gl_FragCoord.xy / uRes;
  vec2 uv = clamp((uv0 - 0.5) / 1.008 + 0.5 + uShakePx / uRes, vec2(0.001), vec2(0.999));
  vec2 tex = 1.0 / uRes;
  vec3 c0 = texture(uScene, uv).rgb;
  float cgA = clamp(texture(uMeta, uv).r, 0.0, 1.0);
  vec3 res = c0;
  if (cgA > 0.005) {
    vec3 soft = (c0*4.0 + texture(uScene,uv+vec2(tex.x,0)).rgb + texture(uScene,uv-vec2(tex.x,0)).rgb +
                 texture(uScene,uv+vec2(0,tex.y)).rgb + texture(uScene,uv-vec2(0,tex.y)).rgb) / 8.0;
    res = mix(c0, soft, cgA * 0.26);
    vec3 graded = max(res, vec3(uBlackFloor));
    vec2 p1 = gl_FragCoord.xy + vec2(uFrame * 17.3, uFrame * 31.7);
    vec2 p2 = floor(gl_FragCoord.xy * 0.5) + vec2(uFrame * 11.1, uFrame * 23.9);
    float g = (hash12(p1)+hash12(p1+7.13)-1.0)*0.62 + (hash12(p2)+hash12(p2+5.71)-1.0)*0.38;
    float lum = dot(graded, vec3(0.299, 0.587, 0.114));
    graded += vec3(g) * (uGrainAmt * (1.12 - 0.58 * smoothstep(0.55, 0.98, lum)));
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


def mat4_ortho_light_vp(sun_dir, center=(0.1, 0.35, 0.5), half_ext=4.2, near=0.5, far=18.0):
    L = np.asarray(sun_dir, dtype=np.float64); L /= np.linalg.norm(L) + 1e-9
    eye = np.asarray(center, dtype=np.float64) + L * 9.0
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


def mat4_trs(pos, quat_wxyz, scale_xyz=(1.0, 1.0, 1.0)):
    w, x, y, z = float(quat_wxyz[0]), float(quat_wxyz[1]), float(quat_wxyz[2]), float(quat_wxyz[3])
    n = math.sqrt(w*w + x*x + y*y + z*z)
    if n < 1e-6:
        R = np.eye(3, dtype=np.float32)
    else:
        w /= n; x /= n; y /= n; z /= n
        R = np.array([
            [1 - 2*(y*y + z*z), 2*(x*y - z*w),     2*(x*z + y*w)],
            [2*(x*y + z*w),     1 - 2*(x*x + z*z), 2*(y*z - x*w)],
            [2*(x*z - y*w),     2*(y*z + x*w),     1 - 2*(x*x + y*y)],
        ], dtype=np.float32)
    S = np.diag([float(scale_xyz[0]), float(scale_xyz[1]), float(scale_xyz[2])]).astype(np.float32)
    M = np.eye(4, dtype=np.float32)
    M[:3, :3] = R @ S
    M[:3, 3] = pos
    # Inverse-transpose for normals under non-uniform impact squash!
    invS = np.diag([1.0/max(float(scale_xyz[0]), 0.05), 1.0/max(float(scale_xyz[1]), 0.05), 1.0/max(float(scale_xyz[2]), 0.05)]).astype(np.float32)
    N3 = R @ invS
    return M, N3


def eval_floating_loot(ev, t_now):
    """Evaluate exact Minecraft item-drop pop-out parabolic arc -> perpetual smooth sinusoidal bob & 360 turntable spin!"""
    dt = t_now - ev["t_pop"]
    if dt < 0.0:
        return None
    p0 = np.array(ev["p_start"], dtype=np.float64)
    ph = np.array(ev["p_hover"], dtype=np.float64)
    pop_dur = 0.54
    if dt < pop_dur:
        u = dt / pop_dur
        # Smooth ease-out horizontal hop + snappy parabolic pop-up arc peaking ~0.88m above plaza!
        uh = 1.0 - (1.0 - u) ** 2.2
        xy = p0[:2] * (1.0 - uh) + ph[:2] * uh
        # Parabolic arc starting at p0[2]=0.28, peaking ~0.88m, smoothly meeting bob trough at u=1!
        z_base = p0[2] * (1.0 - u) + ph[2] * u
        z_arc = 0.56 * math.sin(u * math.pi)
        pos = np.array([xy[0], xy[1], z_base + z_arc], dtype=np.float64)
        # Playful elastic pop-in scale (0.35 -> 1.12 -> 0.96) — crisp jewel-like floating pixel icon size!
        if u < 0.25:
            sc = 0.35 + 0.77 * (u / 0.25)
        elif u < 0.50:
            sc = 1.12 - 0.16 * ((u - 0.25) / 0.25)
        else:
            sc = 0.96
        # Fast playful initial spin settling into iconic steady Minecraft turntable rotation!
        yaw = ev["yaw0"] + ev["spin_spd"] * dt * 1.45
        tilt = 0.10 * math.sin(u * math.pi)
        op = min(1.0, u * 6.0)
    else:
        dt_h = dt - pop_dur
        # Iconic Minecraft steady sinusoidal hover bob (+-7.5cm) + continuous steady turntable spin!
        bob = 0.075 * math.sin(dt_h * 3.4 + ev["bob_phase"])
        pos = np.array([ph[0], ph[1], ph[2] + bob], dtype=np.float64)
        sc = 0.96
        yaw = ev["yaw0"] + ev["spin_spd"] * (pop_dur * 1.45 + dt_h)
        tilt = 0.09 * math.sin(dt_h * 1.7 + ev["bob_phase"])
        op = 1.0

    q_xyzw = Rot.from_euler("xyz", [tilt, 0.06 * math.cos(yaw), yaw]).as_quat()
    quat_wxyz = np.array([q_xyzw[3], q_xyzw[0], q_xyzw[1], q_xyzw[2]], dtype=np.float32)
    return pos.astype(np.float32), quat_wxyz, float(sc), float(op)


def build_dust_catalog(impacts, seed=55):
    rng = np.random.default_rng(seed)
    pts = []
    for t_imp, ix, iy, energy, is_brk in impacts:
        n_p = 8 if is_brk > 0.5 else 6
        for k in range(n_p):
            ang = (k / float(n_p)) * 2 * math.pi + float(rng.uniform(-0.25, 0.25))
            v_r = float(rng.uniform(0.55, 1.65))
            v_z = float(rng.uniform(0.10, 0.42))
            pts.append({
                "t0": float(t_imp), "life": float(rng.uniform(0.50, 0.95)),
                "p0": np.array([ix + math.cos(ang) * 0.14, iy + math.sin(ang) * 0.14, 0.06]),
                "vel": np.array([math.cos(ang) * v_r + 0.10, math.sin(ang) * v_r * 0.65, v_z]),
                "r0": float(rng.uniform(0.10, 0.18)), "r1": float(rng.uniform(0.36, 0.68)),
                "alpha": float(rng.uniform(0.10, 0.18)),
                "rot0": float(rng.uniform(0, 6.28)), "rotv": float(rng.uniform(-1.0, 1.0)), "seed": float(rng.random()),
            })
    return pts


def draw_physics_pass(pl_bgr, ro, R, f_px, active_ids, pos_s, quat_s, sc_s, mesh_names, meshes, loot_events, t_sub, shake_px, out_wh=(1280, 720)):
    """Real MuJoCo 4.0 rigid-body 3D oriented bounding boxes + parabolic loot arcs over full-brightness plate (gain=1.0)."""
    H, W = pl_bgr.shape[:2]
    im = pl_bgr.copy()
    def proj(pts):
        q = (np.atleast_2d(pts) - ro) @ R
        z = q[:, 2]
        uv = f_px * q[:, :2] / np.maximum(z[:, None], 0.05) + np.array([W * 0.5, H * 0.5])
        return uv, z

    # 12 edges of unit cube [-1, +1]^3
    C8 = np.array([[sx, sy, sz] for sx in (-1, 1) for sy in (-1, 1) for sz in (-1, 1)], dtype=np.float64)
    EDGES = [(0,1),(2,3),(4,5),(6,7),(0,2),(1,3),(4,6),(5,7),(0,4),(1,5),(2,6),(3,7)]
    for bi in active_ids:
        he = np.asarray(meshes[mesh_names[bi]][1], dtype=np.float64) * np.asarray(sc_s[bi], dtype=np.float64)
        M, _ = mat4_trs(pos_s[bi], quat_s[bi], (1.0, 1.0, 1.0))
        Rm = M[:3, :3].astype(np.float64)
        t3 = np.asarray(pos_s[bi], dtype=np.float64)
        corners = (C8 * he[None, :]) @ Rm.T + t3[None, :]
        uv, z = proj(corners)
        if np.all(z > 0.2):
            is_blk = not mesh_names[bi].startswith("deb_")
            col = (20, 185, 255) if is_blk else (55, 255, 110)  # BGR: vibrant amber for blocks, lime for permanent voxel debris
            th = 3 if is_blk else 2
            for a, b in EDGES:
                cv2.line(im, (int(uv[a, 0]), int(uv[a, 1])), (int(uv[b, 0]), int(uv[b, 1])), col, th, cv2.LINE_AA)
            if is_blk:
                # Draw 3D drop plumbline to z=0 floor
                g_uv, gz = proj(np.array([t3, [t3[0], t3[1], 0.0]]))
                if np.all(gz > 0.2):
                    cv2.line(im, (int(g_uv[0, 0]), int(g_uv[0, 1])), (int(g_uv[1, 0]), int(g_uv[1, 1])), (0, 110, 255), 2, cv2.LINE_AA)
                    cv2.circle(im, (int(g_uv[1, 0]), int(g_uv[1, 1])), 7, (0, 110, 255), 2, cv2.LINE_AA)

    # Parabolic loot pop arcs + floating hover markers!
    for ev in loot_events:
        res_l = eval_floating_loot(ev, t_sub)
        if res_l is None:
            continue
        lpos, _, _, _ = res_l
        # Draw pop-arc curve from p_start -> p_hover
        arc_pts = []
        p0 = np.array(ev["p_start"], dtype=np.float64)
        ph = np.array(ev["p_hover"], dtype=np.float64)
        for s_k in range(15):
            u = s_k / 14.0
            uh = 1.0 - (1.0 - u) ** 2.2
            xy = p0[:2] * (1.0 - uh) + ph[:2] * uh
            arc_pts.append([xy[0], xy[1], p0[2] * (1.0 - u) + ph[2] * u + 0.56 * math.sin(u * math.pi)])
        auv, az = proj(np.array(arc_pts))
        if np.all(az > 0.2):
            for k in range(len(auv) - 1):
                cv2.line(im, (int(auv[k, 0]), int(auv[k, 1])), (int(auv[k + 1, 0]), int(auv[k + 1, 1])), (255, 225, 45), 2, cv2.LINE_AA)
        # Ground hover circle + plumb stem to live floating loot center
        c_pts = np.array([[lpos[0] + 0.16 * math.cos(th), lpos[1] + 0.16 * math.sin(th), 0.01] for th in np.linspace(0, 2 * math.pi, 17)])
        cuv, cz = proj(c_pts)
        if np.all(cz > 0.2):
            for k in range(len(cuv) - 1):
                cv2.line(im, (int(cuv[k, 0]), int(cuv[k, 1])), (int(cuv[k + 1, 0]), int(cuv[k + 1, 1])), (255, 210, 0), 2, cv2.LINE_AA)
        luv, lz = proj(np.array([[lpos[0], lpos[1], 0.0], lpos]))
        if np.all(lz > 0.2):
            cv2.line(im, (int(luv[0, 0]), int(luv[0, 1])), (int(luv[1, 0]), int(luv[1, 1])), (255, 220, 60), 2, cv2.LINE_AA)
            cv2.circle(im, (int(luv[1, 0]), int(luv[1, 1])), 8, (0, 235, 255), -1, cv2.LINE_AA)

    # Match exact POST_FS 1.008 zoom + uShakePx so wipe boundary aligns 0.0px with wire/shadow/comp!
    # POST_FS: uv_src = (uv_dst - 0.5) / 1.008 + 0.5 + (sx/W, sy/H), where uv_y is bottom-up (+sy shifts sampled image UP -> dst moves DOWN!)
    sx, sy = float(shake_px[0]), float(shake_px[1])
    z_sc = 1.008
    M_aff = np.array([
        [z_sc, 0.0, W * 0.5 * (1.0 - z_sc) - sx * z_sc],
        [0.0, z_sc, H * 0.5 * (1.0 - z_sc) + sy * z_sc],
    ], dtype=np.float32)
    im = cv2.warpAffine(im, M_aff, (W, H), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
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
    resim = False
    save_passes = False
    piece_arg = None
    k = 0
    while k < len(args):
        if args[k] == "--still":
            stills = [int(x) for x in args[k + 1].split(",")]; k += 2
        elif args[k] == "--out":
            out_dir = args[k + 1]; k += 2
        elif args[k] == "--resim":
            resim = True; k += 1
        elif args[k] == "--passes":
            save_passes = True; k += 1
        else:
            piece_arg = args[k]; k += 1

    c = load(piece_arg or (here / "piece_drops"))
    s = scene(c)
    C = np.array(s["cam"], dtype=np.float64)
    R_all = np.array(s["rot"], dtype=np.float64)
    f_px = float(s["f"])
    sun = np.array([-0.46, -0.48, 0.74], dtype=np.float64)
    sun /= np.linalg.norm(sun) + 1e-9

    limbs = np.load(work(c, "limbs.npz"))
    P_body = np.asarray(limbs["P"], dtype=np.float64)
    person_depths = np.asarray(limbs["depth"], dtype=np.float64)
    person_xy = P_body[:, [11, 12, 23, 24], :2].mean(axis=1)

    N = len(C)
    fps = float(c.get("fps_num", 25)) / float(c.get("fps_den", 1))
    pl0 = frame(c, 0)
    H, W = pl0.shape[:2]

    gray0 = cv2.cvtColor(pl0, cv2.COLOR_BGR2GRAY).astype(np.float32) / 255.0
    black_floor = float(np.clip(np.percentile(gray0, 1.0) * 0.85, 0.008, 0.050))
    hp0 = gray0 - cv2.GaussianBlur(gray0, (0, 0), 1.2)
    grain_amt = float(np.clip(float(np.median(np.abs(hp0 - np.median(hp0)))) * 1.4826 * 1.15, 0.008, 0.024))

    meshes = build_all_minecraft_meshes()
    sim_path = work(c, "sim_mc.npz")
    if resim or not sim_path.exists():
        simulate_minecraft_drops(c, meshes, person_xy)
    sim = np.load(sim_path, allow_pickle=False)
    sim_pos = sim["pos"]
    sim_quat = sim["quat"]
    sim_scale = sim["scale"]
    sim_crack = sim["crack"]
    mesh_names = [str(x) for x in sim["mesh_names"]]
    impacts = sim["impacts"]
    loot_events = json.loads(str(sim["loot_json"]))
    dust_particles = build_dust_catalog(impacts)

    back_y = float(c.get("drop_sim", {}).get("back_wall_y", 1.55))
    light_vp = mat4_ortho_light_vp(sun, center=(0.0, 0.05, 0.5), half_ext=4.6)
    proj_m = mat4_perspective_from_opencv(f_px, W, H, near=0.1, far=40.0)

    ctx = moderngl.create_standalone_context(require=330)
    shad_prog = ctx.program(vertex_shader=SHADOW_VS, fragment_shader=SHADOW_FS)
    mesh_prog = ctx.program(vertex_shader=MESH_VS, fragment_shader=MESH_FS)
    gnd_prog  = ctx.program(vertex_shader=VS_FSQ, fragment_shader=GROUND_SHAD_FS)
    dust_prog = ctx.program(vertex_shader=DUST_VS, fragment_shader=DUST_FS)
    acc_prog  = ctx.program(vertex_shader=VS_FSQ, fragment_shader=ACCUM_ADD_FS)
    post_prog = ctx.program(vertex_shader=VS_FSQ, fragment_shader=POST_FS)

    fsq = ctx.buffer(np.array([-1, -1, 1, -1, -1, 1, -1, 1, 1, -1, 1, 1], dtype=np.float32).tobytes())
    va_gnd  = ctx.vertex_array(gnd_prog,  [(fsq, "2f", "p")])
    va_acc  = ctx.vertex_array(acc_prog,  [(fsq, "2f", "p")])
    va_post = ctx.vertex_array(post_prog, [(fsq, "2f", "p")])

    # Upload all blocks, voxel debris cubes, and 9 extruded 3D pixel-art loot meshes (14 floats per vertex!)
    gpu_meshes = {}
    for mname, (md, _) in meshes.items():
        vdata = np.column_stack([md["pos"], md["nrm"], md["uv"], md["col"], md["pbr"]]).astype(np.float32)
        vbo = ctx.buffer(vdata.tobytes())
        ibo = ctx.buffer(md["idx"].astype(np.uint32).tobytes())
        vao_mesh = ctx.vertex_array(mesh_prog, [(vbo, "3f 3f 2f 3f 3f", "inPos", "inNrm", "inUv", "inCol", "inPbr")], index_buffer=ibo)
        vao_shad = ctx.vertex_array(shad_prog, [(vbo, "3f 44x", "inPos")], index_buffer=ibo)
        gpu_meshes[mname] = (vao_mesh, vao_shad)

    dust_q = ctx.buffer(np.array([-1, -1, 1, -1, -1, 1, -1, 1, 1, -1, 1, 1], dtype=np.float32).tobytes())
    dust_inst = ctx.buffer(reserve=max(len(dust_particles), 16) * 8 * 4)
    va_dust = ctx.vertex_array(dust_prog, [(dust_q, "2f", "inQuad"), (dust_inst, "4f 4f/i", "inPosRad", "inInfo")])

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
    mask_t = ctx.texture((W, H), 1, dtype="f4"); mask_t.filter = (moderngl.LINEAR, moderngl.LINEAR); mask_t.repeat_x = mask_t.repeat_y = False

    shad_prog["uLightVP"].write(light_vp.T.tobytes())
    mesh_prog["uProj"].write(proj_m.T.tobytes())
    mesh_prog["uLightVP"].write(light_vp.T.tobytes())
    mesh_prog["uRes"].value = (float(W), float(H))
    mesh_prog["uSun"].value = tuple(map(float, sun))
    gnd_prog["uRes"].value = (float(W), float(H))
    gnd_prog["uF"].value = f_px
    gnd_prog["uBackY"].value = back_y
    gnd_prog["uLightVP"].write(light_vp.T.tobytes())
    dust_prog["uRes"].value = (float(W), float(H))
    dust_prog["uF"].value = f_px
    acc_prog["uRes"].value = (float(W), float(H))
    post_prog["uRes"].value = (float(W), float(H))
    post_prog["uBlackFloor"].value = black_floor
    post_prog["uGrainAmt"].value = grain_amt

    def compute_shake(t_sec):
        sx, sy = 0.0, 0.0
        for t_imp, _, _, energy, _ in impacts:
            dt = t_sec - float(t_imp)
            if 0.0 <= dt < 0.22:
                amp = float(energy) * 4.5 * math.exp(-dt * 18.0)
                sx += amp * math.sin(dt * 95.0 + float(t_imp) * 19.0) * 0.40
                sy += amp * math.cos(dt * 110.0 + float(t_imp) * 13.0)
        return (float(np.clip(sx, -7.5, 7.5)), float(np.clip(sy, -9.0, 9.0)))

    def render_one(i):
        pl = frame(c, i)
        m_raw = cv2.imread(str(work(c, "mask", f"{i:04d}.png")), cv2.IMREAD_GRAYSCALE)
        m_f = cv2.GaussianBlur(m_raw.astype(np.float32) / 255.0, (0, 0), 0.9) if m_raw is not None else np.zeros((H, W), np.float32)
        plate_t.write(pl[:, :, ::-1].tobytes())
        mask_t.write(m_f.astype(np.float32).tobytes())

        ro = C[i]; R = R_all[i]
        Rt = tuple(R.T.flatten().astype(np.float32))
        view_m = mat4_view_from_colmap(ro, R)
        p_depth = float(person_depths[i])
        n_shut = len(SHUTTER)
        sub_w = 1.0 / float(n_shut)

        fbo_acc.use(); ctx.viewport = (0, 0, W, H); ctx.disable(moderngl.DEPTH_TEST); ctx.disable(moderngl.BLEND)
        ctx.clear(0.0, 0.0, 0.0, 0.0)

        for si in range(n_shut):
            t_sub = (i + SHUTTER[si]) / fps
            pos_s = sim_pos[i, si]
            quat_s = sim_quat[i, si]
            sc_s = sim_scale[i, si]
            cr_s = sim_crack[i, si]
            active_ids = np.where(sc_s[:, 0] > 0.02)[0]

            # Evaluate all active hovering 3D pixel-art loot drops at t_sub!
            loot_draws = []
            hover_shads = []
            for ev in loot_events:
                res_l = eval_floating_loot(ev, t_sub)
                if res_l is not None:
                    lpos, lquat, lsc, lop = res_l
                    loot_draws.append((ev["item"], lpos, lquat, lsc))
                    hover_shads.append((float(lpos[0]), float(lpos[1]), float(lpos[2]), float(lop)))

            # Pass 1: 2048x2048 Sun Shadow Map (blocks + voxel debris + floating pixel loot icons!)
            fbo_shad.use(); ctx.viewport = (0, 0, 2048, 2048); ctx.enable(moderngl.DEPTH_TEST); ctx.disable(moderngl.BLEND)
            ctx.clear(depth=1.0)
            for bi in active_ids:
                M, _ = mat4_trs(pos_s[bi], quat_s[bi], sc_s[bi])
                shad_prog["uModel"].write(M.T.tobytes())
                gpu_meshes[mesh_names[bi]][1].render(moderngl.TRIANGLES)
            for iname, lpos, lquat, lsc in loot_draws:
                M, _ = mat4_trs(lpos, lquat, (lsc, lsc, lsc))
                shad_prog["uModel"].write(M.T.tobytes())
                gpu_meshes[iname][1].render(moderngl.TRIANGLES)

            # Pass 2A: Plate + Directional Sun Shadows + Breathing Minecraft Hover Shadows on Concrete Plaza!
            fbo_sub.use(); ctx.viewport = (0, 0, W, H); ctx.disable(moderngl.DEPTH_TEST); ctx.disable(moderngl.BLEND)
            ctx.clear(depth=1.0)
            plate_t.use(0); t_smap.use(1); mask_t.use(2)
            gnd_prog["uPlate"].value = 0; gnd_prog["uShadowMap"].value = 1; gnd_prog["uPersonMask"].value = 2
            gnd_prog["uRo"].value = tuple(map(float, ro)); gnd_prog["uCam"].value = Rt
            gnd_prog["uSkipShadows"].value = 0
            n_hov = min(len(hover_shads), 16)
            gnd_prog["uNumHover"].value = n_hov
            gnd_prog["uHoverShad"].value = hover_shads[:16] + [(0.0, 0.0, 0.0, 0.0)] * (16 - n_hov)
            va_gnd.render(moderngl.TRIANGLES)

            # Pass 2B: Linear High-Contrast PBR 3D Blocks + Voxel Debris + Floating 3D Pixel Loot Icons!
            ctx.enable(moderngl.DEPTH_TEST); ctx.enable(moderngl.BLEND)
            ctx.blend_func = (moderngl.SRC_ALPHA, moderngl.ONE_MINUS_SRC_ALPHA)
            t_smap.use(0); mask_t.use(1); plate_t.use(2)
            mesh_prog["uShadowMap"].value = 0; mesh_prog["uPersonMask"].value = 1; mesh_prog["uPlate"].value = 2
            mesh_prog["uView"].write(view_m.T.tobytes()); mesh_prog["uRo"].value = tuple(map(float, ro))
            mesh_prog["uPersonDepth"].value = p_depth
            mesh_prog["uRenderMode"].value = 0

            mesh_prog["uIsLoot"].value = 0.0
            for bi in active_ids:
                M, N3 = mat4_trs(pos_s[bi], quat_s[bi], sc_s[bi])
                mesh_prog["uModel"].write(M.T.tobytes()); mesh_prog["uNormalMat"].write(N3.T.tobytes())
                mesh_prog["uCrackStage"].value = float(cr_s[bi])
                gpu_meshes[mesh_names[bi]][0].render(moderngl.TRIANGLES)

            mesh_prog["uCrackStage"].value = 0.0
            mesh_prog["uIsLoot"].value = 1.0
            for iname, lpos, lquat, lsc in loot_draws:
                M, N3 = mat4_trs(lpos, lquat, (lsc, lsc, lsc))
                mesh_prog["uModel"].write(M.T.tobytes()); mesh_prog["uNormalMat"].write(N3.T.tobytes())
                gpu_meshes[iname][0].render(moderngl.TRIANGLES)

            # Pass 3: Subtle Sunlit Impact Concrete Dust Puffs
            d_rows = []
            for dp in dust_particles:
                dt_p = t_sub - dp["t0"]
                if 0.0 <= dt_p < dp["life"]:
                    u = dt_p / dp["life"]
                    drag_t = (1.0 - math.exp(-dt_p * 2.5)) / 2.5
                    pp = dp["p0"] + dp["vel"] * drag_t + np.array([0.12 * dt_p, 0.0, 0.20 * dt_p])
                    rr = dp["r0"] + (dp["r1"] - dp["r0"]) * (u ** 0.55)
                    aa = dp["alpha"] * math.sin(min(1.0, u * 4.5) * math.pi * 0.5) * ((1.0 - u) ** 1.45)
                    z_cam = float(((pp - ro) @ R)[2])
                    if z_cam > 0.35 and aa > 0.008:
                        d_rows.append((z_cam, [float(pp[0]), float(pp[1]), float(pp[2]), float(rr),
                                               float(aa), float(dp["rot0"] + dp["rotv"] * dt_p), float(dp["seed"]), 0.0]))
            if d_rows:
                d_rows.sort(key=lambda x: -x[0])
                dust_inst.write(np.array([r[1] for r in d_rows], dtype=np.float32).tobytes())
                ctx.disable(moderngl.DEPTH_TEST); ctx.enable(moderngl.BLEND)
                ctx.blend_func = (moderngl.SRC_ALPHA, moderngl.ONE_MINUS_SRC_ALPHA)
                mask_t.use(0); dust_prog["uPersonMask"].value = 0
                dust_prog["uRo"].value = tuple(map(float, ro)); dust_prog["uCam"].value = Rt
                dust_prog["uPersonDepth"].value = p_depth
                va_dust.render(moderngl.TRIANGLES, instances=len(d_rows))

            fbo_acc.use(); ctx.disable(moderngl.DEPTH_TEST); ctx.enable(moderngl.BLEND)
            ctx.blend_func = (moderngl.ONE, moderngl.ONE)
            t_sub_col.use(0); t_sub_meta.use(1)
            acc_prog["uSrc0"].value = 0; acc_prog["uSrc1"].value = 1; acc_prog["uWeight"].value = sub_w
            va_acc.render(moderngl.TRIANGLES)

        shake_xy = compute_shake(i / fps)
        fbo_final.use(); ctx.viewport = (0, 0, W, H); ctx.disable(moderngl.DEPTH_TEST); ctx.disable(moderngl.BLEND)
        t_acc_col.use(0); t_acc_meta.use(1)
        post_prog["uScene"].value = 0; post_prog["uMeta"].value = 1
        post_prog["uShakePx"].value = shake_xy; post_prog["uFrame"].value = float(i)
        va_post.render(moderngl.TRIANGLES)

        ctx.finish()
        arr = np.frombuffer(t_final.read(), "f4").reshape(H, W, 4)[::-1]
        img = (np.clip(arr[..., :3], 0, 1) * 255 + 0.5).astype(np.uint8)[:, :, ::-1]

        pass_dict = None
        if save_passes:
            # Render real breakdown passes at center shutter slice si=4 with IDENTICAL uShakePx (0.0px wipe seam!)
            si_c = n_shut // 2
            t_sub_c = (i + SHUTTER[si_c]) / fps
            pos_c = sim_pos[i, si_c]; quat_c = sim_quat[i, si_c]; sc_c = sim_scale[i, si_c]; cr_c = sim_crack[i, si_c]
            act_c = np.where(sc_c[:, 0] > 0.02)[0]
            loot_c, hover_c = [], []
            for ev in loot_events:
                res_l = eval_floating_loot(ev, t_sub_c)
                if res_l is not None:
                    lpos, lquat, lsc, lop = res_l
                    loot_c.append((ev["item"], lpos, lquat, lsc))
                    hover_c.append((float(lpos[0]), float(lpos[1]), float(lpos[2]), float(lop)))

            # Recompute shadow map for center slice si_c
            fbo_shad.use(); ctx.viewport = (0, 0, 2048, 2048); ctx.enable(moderngl.DEPTH_TEST); ctx.disable(moderngl.BLEND)
            ctx.clear(depth=1.0)
            for bi in act_c:
                M, _ = mat4_trs(pos_c[bi], quat_c[bi], sc_c[bi])
                shad_prog["uModel"].write(M.T.tobytes()); gpu_meshes[mesh_names[bi]][1].render(moderngl.TRIANGLES)
            for iname, lpos, lquat, lsc in loot_c:
                M, _ = mat4_trs(lpos, lquat, (lsc, lsc, lsc))
                shad_prog["uModel"].write(M.T.tobytes()); gpu_meshes[iname][1].render(moderngl.TRIANGLES)

            def render_diag_mode(mode_id, skip_shad, wire_flag):
                fbo_sub.use(); ctx.viewport = (0, 0, W, H); ctx.disable(moderngl.DEPTH_TEST); ctx.disable(moderngl.BLEND)
                ctx.clear(depth=1.0)
                plate_t.use(0); t_smap.use(1); mask_t.use(2)
                gnd_prog["uPlate"].value = 0; gnd_prog["uShadowMap"].value = 1; gnd_prog["uPersonMask"].value = 2
                gnd_prog["uRo"].value = tuple(map(float, ro)); gnd_prog["uCam"].value = Rt
                gnd_prog["uSkipShadows"].value = int(skip_shad)
                n_h = min(len(hover_c), 16)
                gnd_prog["uNumHover"].value = n_h
                gnd_prog["uHoverShad"].value = hover_c[:16] + [(0.0, 0.0, 0.0, 0.0)] * (16 - n_h)
                va_gnd.render(moderngl.TRIANGLES)

                ctx.enable(moderngl.DEPTH_TEST); ctx.enable(moderngl.BLEND)
                ctx.blend_func = (moderngl.SRC_ALPHA, moderngl.ONE_MINUS_SRC_ALPHA)
                ctx.wireframe = bool(wire_flag)
                t_smap.use(0); mask_t.use(1); plate_t.use(2)
                mesh_prog["uShadowMap"].value = 0; mesh_prog["uPersonMask"].value = 1; mesh_prog["uPlate"].value = 2
                mesh_prog["uView"].write(view_m.T.tobytes()); mesh_prog["uRo"].value = tuple(map(float, ro))
                mesh_prog["uPersonDepth"].value = p_depth
                mesh_prog["uRenderMode"].value = int(mode_id)
                mesh_prog["uIsLoot"].value = 0.0
                for bi in act_c:
                    M, N3 = mat4_trs(pos_c[bi], quat_c[bi], sc_c[bi])
                    mesh_prog["uModel"].write(M.T.tobytes()); mesh_prog["uNormalMat"].write(N3.T.tobytes())
                    mesh_prog["uCrackStage"].value = float(cr_c[bi])
                    gpu_meshes[mesh_names[bi]][0].render(moderngl.TRIANGLES)
                mesh_prog["uCrackStage"].value = 0.0
                mesh_prog["uIsLoot"].value = 1.0
                for iname, lpos, lquat, lsc in loot_c:
                    M, N3 = mat4_trs(lpos, lquat, (lsc, lsc, lsc))
                    mesh_prog["uModel"].write(M.T.tobytes()); mesh_prog["uNormalMat"].write(N3.T.tobytes())
                    gpu_meshes[iname][0].render(moderngl.TRIANGLES)
                ctx.wireframe = False

                fbo_final.use(); ctx.viewport = (0, 0, W, H); ctx.disable(moderngl.DEPTH_TEST); ctx.disable(moderngl.BLEND)
                t_sub_col.use(0); t_sub_meta.use(1)
                post_prog["uScene"].value = 0; post_prog["uMeta"].value = 1
                post_prog["uShakePx"].value = shake_xy; post_prog["uFrame"].value = float(i)
                va_post.render(moderngl.TRIANGLES)
                ctx.finish()
                a_d = np.frombuffer(t_final.read(), "f4").reshape(H, W, 4)[::-1]
                u8 = (np.clip(a_d[..., :3], 0, 1) * 255 + 0.5).astype(np.uint8)[:, :, ::-1]
                return cv2.resize(u8, (1280, 720), interpolation=cv2.INTER_AREA)

            pass_dict = {
                "physics": draw_physics_pass(pl, ro, R, f_px, act_c, pos_c, quat_c, sc_c, mesh_names, meshes, loot_events, t_sub_c, shake_xy),
                "wire":    render_diag_mode(mode_id=1, skip_shad=1, wire_flag=True),
                "shadow":  render_diag_mode(mode_id=2, skip_shad=0, wire_flag=False),
            }

        return pl, img, pass_dict

    if stills:
        rows_im = []
        hero = None
        for idx in stills:
            pl, comp, _ = render_one(idx)
            if idx == stills[len(stills) // 2]:
                hero = comp
            t_im = lambda im, lb: label(cv2.resize(im, (960, 540)), lb)
            rows_im.append(np.hstack([t_im(pl, f"plate {idx}"), t_im(comp, f"mc drop {idx}")]))
        out_p = work(c, "still_mc_sheet.jpg")
        cv2.imwrite(str(out_p), np.vstack(rows_im), [cv2.IMWRITE_JPEG_QUALITY, 92])
        if hero is not None:
            cv2.imwrite(str(work(c, "still_mc_hero.jpg")), hero, [cv2.IMWRITE_JPEG_QUALITY, 94])
        print("stills ->", out_p)
        return

    comp_dir = work(c, "comp")
    shutil.rmtree(comp_dir, ignore_errors=True)
    comp_dir.mkdir(parents=True)
    pass_root = work(c, "pass_mc")
    if save_passes:
        shutil.rmtree(pass_root, ignore_errors=True)
        for pk in ("physics", "wire", "shadow"):
            (pass_root / pk).mkdir(parents=True, exist_ok=True)
    ts = time.time()
    for i in range(N):
        _, comp, pdict = render_one(i)
        cv2.imwrite(str(comp_dir / f"{i:04d}.png"), comp, [cv2.IMWRITE_PNG_COMPRESSION, 1])
        if save_passes and pdict is not None:
            for pk, pim in pdict.items():
                cv2.imwrite(str(pass_root / pk / f"{i:04d}.jpg"), pim, [cv2.IMWRITE_JPEG_QUALITY, 93])
        if i % 40 == 0 or i == N - 1:
            print(f"mc_drops frame {i}/{N} ({time.time() - ts:.1f}s)", flush=True)
    encode(c, out_dir=out_dir)


if __name__ == "__main__":
    main()
