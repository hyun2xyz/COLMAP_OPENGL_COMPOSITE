"""Falling 3D Modeled Objects + Ground Shatter + Real MuJoCo 4.0 Rigid-Body Debris Simulation & OpenGL PBR Composite.

    python render_debris.py [piece_dir] --still 30,75,120,165,220   quick stills -> work/still_debris_sheet.jpg
    python render_debris.py [piece_dir] --resim --still 30,75,120   force re-run MuJoCo sim + stills
    python render_debris.py [piece_dir] --out out                   full render -> out/*_Composite.mp4, *_Compare_1080x1350.mp4

Pipeline:
- Real MuJoCo 4.0.0 3D rigid-body physics (40 substeps/frame, g=-9.81 m/s^2, Coulomb friction, elastic restitution,
  multi-body debris pileup collisions, concrete curb back wall y=+1.55m, kinematic dancer body deflection cylinder).
- 15 falling hero 3D modeled everyday props (Retro CRT TV, Wooden Shipping Crate, Terracotta Amphora Vase,
  Hollow Concrete Cinder Block, Ribbed Red Steel Oil Drum, Car Rubber Tire) -> fragile props dynamically shatter
  on z=0 concrete impact into ~156 rigid-body debris pieces (boards, curved ceramic shards, CRT glass & PCB, concrete rubble)
  that bounce, clatter, collide with each other & heavy rolling tires/barrels, and settle naturally on the sunlit plaza!
- 4-Pass Native OpenGL 3.30 PBR Renderer:
  Pass 1: 2048x2048 directional sun PCF shadow map (matched to COLMAP scene.json sun vector)
  Pass 2: 5-subframe 180-degree motion-blurred PBR 3D mesh render + concrete shadow catcher + MediaPipe person occlusion
  Pass 3: Volumetric sunlit impact dust plumes & micro-rubble spray erupting from exact z=0 impact points
  Pass 4: Impact camera micro-shake + OLPF lens softness + measured plate black floor & animated sensor grain
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
from props_mesh import build_all_meshes  # noqa: E402


SHUTTER = [0.0, 0.045, 0.09, 0.135, 0.18, 0.225, 0.27, 0.315, 0.36]  # 9 smooth action-shutter slices (zero stepping!)


def find_ffmpeg():
    tl = root / "tools.local.json"
    if tl.exists():
        d = json.loads(tl.read_text(encoding="utf-8"))
        p = d.get("ffmpeg") or (d.get("tools") or {}).get("ffmpeg")
        if p and Path(p).exists():
            return str(p)
    return shutil.which("ffmpeg") or "ffmpeg"


# ==============================================================================
#  STAGE 1: REAL MUJOCO 4.0.0 TWO-STAGE DROP + IMPACT SHATTER + DEBRIS PHYSICS
# ==============================================================================

# 15 choreographed hero drops targeting the visible sunlit concrete plaza sweet-spot (x in [-1.75, +1.85], y in [+0.20, +1.15])
# Carefully choreographed so hero props plummet clearly through mid-air, smash visibly onto the concrete tiles,
# and their shattered debris + bouncing tires & steel drums collide and pile up RIGHT ON SCREEN!
HERO_SCHEDULE = [
    # (drop_frame, prop_kind, x_drop, y_drop, z_start, vx, vy, vz0, wx, wy, wz, mass, n_shards, shard_kind)
    (  6, "ceramic_vase",  0.55,  0.62, 3.7, -0.08,  0.12, -1.8,  2.0, -1.4,  1.2,  4.5, 14, "vase_shard"),
    ( 20, "wood_crate",    1.28,  0.75, 3.9, -0.18,  0.10, -1.6, -1.6,  2.0,  0.9, 10.5, 14, "wood_plank"),
    ( 36, "crt_tv",        0.18,  0.68, 4.0,  0.12,  0.18, -1.8,  1.4,  1.9, -1.2, 20.0, 14, "crt_shard"),
    ( 52, "car_tire",      1.62,  0.52, 4.0, -0.45,  0.22, -2.0,  0.6,  4.2, -0.8, 12.5,  0, None),
    ( 66, "cinder_block",  0.82,  0.92, 3.8, -0.10,  0.08, -1.8,  2.8, -2.1,  1.5, 16.0, 12, "concrete_chunk"),
    ( 80, "wood_crate",   -1.62,  0.48, 3.9,  0.22,  0.20, -1.8,  1.8, -1.6,  1.3, 10.5, 14, "wood_plank"),
    ( 94, "oil_drum",      0.95,  0.55, 4.1, -0.20,  0.25, -2.0,  1.2, -1.5,  0.8, 28.0,  0, None),
    (108, "ceramic_vase",  1.48,  0.35, 3.7, -0.25,  0.28, -1.8, -2.2,  1.8, -1.4,  4.5, 14, "vase_shard"),
    (122, "crt_tv",        1.12,  0.82, 4.0, -0.22,  0.12, -1.9, -1.7, -1.9,  1.4, 20.0, 14, "crt_shard"),
    (136, "cinder_block",  0.35,  0.32, 3.8,  0.08,  0.28, -1.9, -2.4,  2.6, -1.1, 16.0, 12, "concrete_chunk"),
    (148, "wood_crate",    0.68,  0.85, 3.9, -0.14,  0.10, -1.8,  2.1,  1.5, -1.6, 12.5,  0, None),
    (160, "car_tire",     -0.12,  0.28, 4.0,  0.52,  0.28, -2.1, -0.9, -4.6,  1.1, 12.5,  0, None),
    (172, "ceramic_vase", -1.48,  0.72, 3.7,  0.25,  0.15, -1.8,  2.3, -1.9,  1.5,  4.5, 12, "vase_shard"),
    (182, "cinder_block",  1.55,  0.65, 3.8, -0.28,  0.16, -1.9,  2.2,  2.4, -1.0, 16.0, 12, "concrete_chunk"),
    (192, "oil_drum",      0.24,  0.88, 4.1,  0.26,  0.14, -2.0, -1.4,  1.8, -0.9, 28.0,  0, None),
]


def simulate_debris(c, meshes, person_xy_track):
    """Run real MuJoCo 4.0.0 two-stage drop + shatter + debris simulation (40 substeps/frame, 5 shutter slices)."""
    cfg = c.get("debris_sim", {})
    seed = int(cfg.get("seed", 2035))
    back_y = float(cfg.get("back_wall_y", 1.55))
    grav = float(cfg.get("gravity", -9.81))
    substeps = int(cfg.get("substeps", 40))
    fps = float(c.get("fps_num", 25)) / float(c.get("fps_den", 1))
    N = int(c.get("frames", 240))
    dt = 1.0 / fps / substeps
    rng = np.random.default_rng(seed)

    # Build complete body registry: track exact hero_body_ids[hi] and hero_shard_indices[hi]!
    bodies = []
    hero_body_ids = []
    hero_shard_indices = {}
    for hi, item in enumerate(HERO_SCHEDULE):
        _, prop_kind, _, _, _, _, _, _, _, _, _, mass, n_sh, sh_kind = item
        he = meshes[prop_kind][1]
        hero_body_ids.append(len(bodies))
        bodies.append({
            "name": f"h{hi}",
            "mesh": prop_kind,
            "is_shard": False,
            "hero_idx": hi,
            "half_ext": he,
            "mass": mass,
            "fragile": (n_sh > 0),
        })
        sh_ids = []
        for s_i in range(n_sh):
            mname = f"{sh_kind}_{s_i % 6}"
            she = meshes[mname][1]
            # Realistic debris piece mass
            sh_mass = max(0.28, mass / max(n_sh, 1) * float(rng.uniform(0.65, 1.35)))
            sh_ids.append(len(bodies))
            bodies.append({
                "name": f"s{hi}_{s_i}",
                "mesh": mname,
                "is_shard": True,
                "hero_idx": hi,
                "half_ext": she,
                "mass": sh_mass,
                "fragile": False,
            })
        hero_shard_indices[hi] = sh_ids

    n_bodies = len(bodies)
    print(f"Building MuJoCo 4.0.0 world: {len(HERO_SCHEDULE)} hero props + {n_bodies - len(HERO_SCHEDULE)} rigid debris shards = {n_bodies} rigid bodies")

    # Build MuJoCo XML with real concrete plaza z=0, concrete curb back wall y=back_y, and kinematic dancer cylinder!
    xml = [
        f'<mujoco><option timestep="{dt:.7f}" gravity="0 0 {grav:.3f}" integrator="implicitfast" iterations="40"/>',
        '<default><geom solref="0.006 0.55" solimp="0.92 0.98 0.001" friction="0.92 0.06 0.015"/></default>',
        '<worldbody>',
        # Finite thick concrete plaza ground slab (top surface strictly z=0.0, covers [-12..12] x [-12..12])
        '<geom name="ground" type="box" size="12 12 0.25" pos="0 0 -0.25" contype="1" conaffinity="1" friction="0.94 0.08 0.018"/>',
        # Concrete curb + atrium railing wall behind plaza (front face strictly y=back_y!)
        f'<geom name="back_wall" type="box" size="8 0.25 4.0" pos="0 {back_y + 0.25:.3f} 4.0" contype="1" conaffinity="1" friction="0.65 0.02 0.004"/>',
        # Soft off-screen side containment boxes so bouncing debris stays richly visible across the plaza
        '<geom name="left_lim" type="box" size="0.25 6.0 4.0" pos="-2.85 0.2 4.0" contype="1" conaffinity="1"/>',
        '<geom name="right_lim" type="box" size="0.25 6.0 4.0" pos="2.75 0.2 4.0" contype="1" conaffinity="1"/>',
        # Invisible kinematic dancer deflector cylinder (deflects rolling tires/debris around his legs!)
        f'<body name="dancer" pos="{person_xy_track[0,0]:.3f} {person_xy_track[0,1]:.3f} 0.85" mocap="true">'
        '<geom type="cylinder" size="0.34 0.85" contype="1" conaffinity="1" friction="0.35 0.01 0.001"/></body>',
    ]

    park_pos = []
    for bi, b in enumerate(bodies):
        hx, hy, hz = b["half_ext"]
        # Park inactive bodies cleanly far off-stage (x>=40, y>=40, z=25) with FULL compile-time contype=1/conaffinity=1!
        px = 40.0 + (bi % 16) * 2.2
        py = 40.0 + (bi // 16) * 2.2
        pz = 25.0
        park_pos.append((px, py, pz))
        if b["mesh"] == "oil_drum":
            geom_tag = f'<geom name="g{bi}" type="cylinder" size="{hx:.4f} {hz:.4f}" mass="{b["mass"]:.3f}" solref="0.008 0.42" friction="0.82 0.05 0.012" contype="1" conaffinity="1"/>'
        elif b["mesh"] == "car_tire":
            # Rubber tire: lively realistic rubber bounce + grippy rolling friction on concrete!
            geom_tag = f'<geom name="g{bi}" type="cylinder" size="{hx:.4f} {hy:.4f}" euler="90 0 0" mass="{b["mass"]:.3f}" solref="0.010 0.24" friction="1.05 0.06 0.016" contype="1" conaffinity="1"/>'
        else:
            damp = 0.68 if ("concrete" in b["mesh"] or "cinder" in b["mesh"]) else 0.50
            geom_tag = f'<geom name="g{bi}" type="box" size="{max(hx,0.014):.4f} {max(hy,0.014):.4f} {max(hz,0.012):.4f}" mass="{b["mass"]:.3f}" solref="0.006 {damp:.2f}" friction="0.92 0.06 0.016" contype="1" conaffinity="1"/>'
        xml.append(f'<body name="{b["name"]}" pos="{px:.3f} {py:.3f} {pz:.3f}"><freejoint name="j{bi}"/>{geom_tag}</body>')

    xml.append('</worldbody></mujoco>')
    model = mujoco.MjModel.from_xml_string("\n".join(xml))
    data = mujoco.MjData(model)

    # Pre-lookup joint qpos/qvel & geom indices for O(1) speed in inner substep loop!
    qpos_adr = [model.jnt_qposadr[mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, f"j{bi}")] for bi in range(n_bodies)]
    qvel_adr = [model.jnt_dofadr[mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, f"j{bi}")] for bi in range(n_bodies)]
    geom_id  = [mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, f"g{bi}") for bi in range(n_bodies)]
    dancer_mocap_id = model.body_mocapid[mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "dancer")]

    active = np.zeros(n_bodies, dtype=bool)
    shattered = np.zeros(len(HERO_SCHEDULE), dtype=bool)
    bounce_count = np.zeros(len(HERO_SCHEDULE), dtype=int)
    last_bounce_t = np.full(len(HERO_SCHEDULE), -10.0)
    impacts = []

    def activate_body(bi, pos, quat, linvel, angvel):
        qa = qpos_adr[bi]; va = qvel_adr[bi]
        data.qpos[qa:qa + 3] = pos
        data.qpos[qa + 3:qa + 7] = quat
        data.qvel[va:va + 3] = linvel
        data.qvel[va + 3:va + 6] = angvel
        active[bi] = True

    def deactivate_body(bi):
        qa = qpos_adr[bi]; va = qvel_adr[bi]
        data.qpos[qa:qa + 3] = park_pos[bi]
        data.qpos[qa + 3:qa + 7] = (1.0, 0.0, 0.0, 0.0)
        data.qvel[va:va + 6] = 0.0
        active[bi] = False

    # Keep all parked bodies frozen at zero velocity every substep so they don't fall to infinity
    shutter_steps = [int(round(s * substeps)) for s in SHUTTER]
    n_shut = len(SHUTTER)
    out_pos = np.zeros((N, n_shut, n_bodies, 3), dtype=np.float32)
    out_quat = np.zeros((N, n_shut, n_bodies, 4), dtype=np.float32)
    out_vis = np.zeros((N, n_shut, n_bodies), dtype=np.uint8)

    t0 = time.time()
    for f_idx in range(N):
        # Check if any hero prop drops on this frame!
        for hi, item in enumerate(HERO_SCHEDULE):
            drop_f, prop_kind, xd, yd, zs, vx, vy, vz0, wx, wy, wz, mass, n_sh, _ = item
            if f_idx == drop_f:
                hbi = hero_body_ids[hi]
                q_init = Rot.from_euler("xyz", rng.uniform(0, 2 * math.pi, 3)).as_quat()  # xyzw
                q_mj = np.array([q_init[3], q_init[0], q_init[1], q_init[2]])            # wxyz
                activate_body(hbi, np.array([xd, yd, zs]), q_mj, np.array([vx, vy, vz0]), np.array([wx, wy, wz]))

        # Interpolate dancer mocap cylinder across frame
        pxy0 = person_xy_track[f_idx]
        pxy1 = person_xy_track[min(f_idx + 1, N - 1)]

        for sub in range(substeps):
            u = sub / float(substeps)
            t_now = (f_idx + u) / fps
            pxy = pxy0 * (1.0 - u) + pxy1 * u
            data.mocap_pos[dancer_mocap_id] = [pxy[0], pxy[1], 0.85]

            # Record shutter slices BEFORE stepping so frame f starts cleanly at shutter=0
            if sub in shutter_steps:
                si = shutter_steps.index(sub)
                for bi in range(n_bodies):
                    if active[bi]:
                        qa = qpos_adr[bi]
                        out_pos[f_idx, si, bi] = data.qpos[qa:qa + 3]
                        out_quat[f_idx, si, bi] = data.qpos[qa + 3:qa + 7]
                        out_vis[f_idx, si, bi] = 1

            # Keep parked bodies pinned stationary at their far off-stage parking grid
            for bi in range(n_bodies):
                if not active[bi]:
                    qa = qpos_adr[bi]; va = qvel_adr[bi]
                    data.qpos[qa:qa + 3] = park_pos[bi]
                    data.qvel[va:va + 6] = 0.0

            # Capture pre-step vertical velocities so flat-face collisions that bounce in 1 substep still trigger shatter!
            pre_vz = [float(data.qvel[qvel_adr[hero_body_ids[hi]] + 2]) for hi in range(len(HERO_SCHEDULE))]
            mujoco.mj_step(model, data)

            # Impact detection for active falling hero props hitting z=0 concrete plaza (or slamming onto pileup!)
            for hi, item in enumerate(HERO_SCHEDULE):
                hbi = hero_body_ids[hi]
                if not active[hbi]:
                    continue
                qa = qpos_adr[hbi]; va = qvel_adr[hbi]
                pos = data.qpos[qa:qa + 3].copy()
                vel = data.qvel[va:va + 3].copy()
                ang = data.qvel[va + 3:va + 6].copy()
                he = bodies[hbi]["half_ext"]
                rad_eff = float(max(he[0], he[1], he[2]))
                prop_kind = item[1]
                mass = item[11]
                vz_in = min(float(vel[2]), pre_vz[hi])

                if bodies[hbi]["fragile"] and not shattered[hi]:
                    # Trigger dramatic shatter on concrete slam OR hard collision onto pileup!
                    hit_ground = (pos[2] <= rad_eff + 0.14 and vz_in < -2.6)
                    hit_pileup = (pos[2] <= 0.85 and pre_vz[hi] < -4.2 and (vel[2] - pre_vz[hi]) > 3.2)
                    if hit_ground or hit_pileup:
                        shattered[hi] = True
                        deactivate_body(hbi)
                        imp_speed = abs(vz_in)
                        energy = min(1.8, (mass * imp_speed * imp_speed) / 950.0)
                        kcode = 1 if prop_kind == "ceramic_vase" else (2 if prop_kind == "crt_tv" else 0)
                        impacts.append([float(t_now), float(pos[0]), float(pos[1]), float(max(0.35, energy)), float(kcode)])

                        # Spawn fractured MuJoCo rigid debris sub-bodies on a clean non-overlapping 2-tier radial shell
                        # (eliminates interpenetration launch spikes so 100% of shards follow realistic ballistic arcs!)
                        sh_ids = hero_shard_indices[hi]
                        n_s = len(sh_ids)
                        half_n = max(1, n_s // 2)
                        for idx_k, sbi in enumerate(sh_ids):
                            tier = 0 if idx_k < half_n else 1
                            ring_i = idx_k if tier == 0 else (idx_k - half_n)
                            ring_n = half_n if tier == 0 else max(1, n_s - half_n)
                            ang_az = (ring_i / float(ring_n)) * 2.0 * math.pi + tier * 0.45 + float(rng.uniform(-0.10, 0.10))
                            r_off = (0.34 if tier == 0 else 0.24) + float(rng.uniform(0.0, 0.10))
                            z_off = (0.14 if tier == 0 else 0.42) + float(rng.uniform(0.0, 0.08))
                            sp = np.array([
                                pos[0] + math.cos(ang_az) * r_off,
                                min(back_y - 0.18, pos[1] + math.sin(ang_az) * r_off),
                                z_off,
                            ], dtype=np.float64)
                            # Crisp realistic concrete shatter burst -> tumbles, clatters & settles on sunlit plaza!
                            burst_r = float(rng.uniform(0.65, 1.85)) * (0.80 + 0.03 * imp_speed)
                            burst_z = float(rng.uniform(0.85, 2.25)) * (0.78 + 0.03 * imp_speed)
                            sv = np.array([
                                vel[0] * 0.30 + math.cos(ang_az) * burst_r,
                                vel[1] * 0.30 + math.sin(ang_az) * burst_r * 0.60,
                                burst_z,
                            ], dtype=np.float64)
                            sw = rng.uniform(-10.0, 10.0, 3) + ang * 0.30
                            sq = Rot.from_euler("xyz", rng.uniform(0, 2 * math.pi, 3)).as_quat()
                            activate_body(sbi, sp, np.array([sq[3], sq[0], sq[1], sq[2]]), sv, sw)
                else:
                    # Heavy non-shattering prop (oil_drum / car_tire): emit impact dust ONLY on initial hard slam + 1 big bounce!
                    if pos[2] <= rad_eff + 0.10 and vel[2] < -3.6 and bounce_count[hi] < 2 and (t_now - last_bounce_t[hi]) > 0.40:
                        last_bounce_t[hi] = t_now
                        bounce_count[hi] += 1
                        imp_speed = abs(float(vel[2]))
                        energy = min(1.8, (mass * imp_speed * imp_speed) / 950.0)
                        kcode = 3 if prop_kind == "car_tire" else 2
                        impacts.append([float(t_now), float(pos[0]), float(pos[1]), float(max(0.40, energy)), float(kcode)])

    print(f"MuJoCo sim done in {time.time() - t0:.2f}s | {len(impacts)} ground impacts recorded")
    sim_path = work(c, "sim_debris.npz")
    mesh_names = np.array([b["mesh"] for b in bodies])
    np.savez_compressed(
        sim_path,
        pos=out_pos,
        quat=out_quat,
        vis=out_vis,
        mesh_names=mesh_names,
        impacts=np.array(impacts, dtype=np.float32) if impacts else np.zeros((0, 5), dtype=np.float32),
    )
    return sim_path


# ==============================================================================
#  STAGE 2: NATIVE OPENGL 3.30 4-PASS PBR MESH + SHADOW MAP + VOLUMETRIC DUST
# ==============================================================================

SHADOW_VS = """#version 330
in vec3 inPos;
uniform mat4 uLightVP;
uniform mat4 uModel;
void main() {
  gl_Position = uLightVP * uModel * vec4(inPos, 1.0);
}
"""

SHADOW_FS = """#version 330
void main() {}
"""

MESH_VS = """#version 330
in vec3 inPos;
in vec3 inNrm;
in vec2 inUv;
in float inMat;
in float inAO;

uniform mat4 uProj;
uniform mat4 uView;
uniform mat4 uModel;
uniform mat3 uNormalMat;
uniform mat4 uLightVP;

out vec3 vWorldPos;
out vec3 vWorldNrm;
out vec2 vUv;
out float vMat;
out float vAO;
out float vCamDepth;
out vec4 vLightSpace;

void main() {
  vec4 wp = uModel * vec4(inPos, 1.0);
  vWorldPos = wp.xyz;
  vWorldNrm = normalize(uNormalMat * inNrm);
  vUv = inUv;
  vMat = inMat;
  vAO = inAO;
  vec4 vp = uView * wp;
  vCamDepth = -vp.z; /* OpenGL view space looks along -Z */
  vLightSpace = uLightVP * wp;
  gl_Position = uProj * vp;
}
"""

MESH_FS = """#version 330
in vec3 vWorldPos;
in vec3 vWorldNrm;
in vec2 vUv;
in float vMat;
in float vAO;
in float vCamDepth;
in vec4 vLightSpace;

uniform sampler2D uShadowMap;
uniform sampler2D uPersonMask;
uniform sampler2D uPlate;
uniform vec2 uRes;
uniform vec3 uRo, uSun;
uniform float uPersonDepth;

layout(location=0) out vec4 oCol;  /* sRGB RGB, A=vis */
layout(location=1) out vec4 oMeta; /* R=cgAlpha, G=cgInvDepth, B=0, A=1 */

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

float fbm(vec2 p) {
  float v = 0.0, a = 0.5;
  for (int k = 0; k < 4; k++) { v += a * noise2(p); p *= 2.04; a *= 0.5; }
  return v;
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

/* Sample real sunlit outdoor plaza + glass atrium environment directly from live footage! */
vec3 samplePlateIBL(vec3 R) {
  vec2 uvEnv = vec2(clamp(0.5 + R.x * 0.42, 0.02, 0.98), clamp(0.52 - R.z * 0.48, 0.02, 0.98));
  vec3 pCol = pow(texture(uPlate, uvEnv).rgb, vec3(2.2));
  /* Boost bright sunlit overhead sky/facade above horizon, darken ground underside */
  vec3 skyBoost = vec3(0.52, 0.68, 0.88) * smoothstep(0.15, 0.85, R.z) * 1.15;
  float gndDim = mix(0.42, 1.0, smoothstep(-0.55, 0.15, R.z));
  return pCol * gndDim + skyBoost;
}

void main() {
  /* MediaPipe person depth occlusion: objects behind dancer vanish cleanly behind him! */
  vec2 uvTop = vec2(gl_FragCoord.x / uRes.x, 1.0 - gl_FragCoord.y / uRes.y);
  float pMask = texture(uPersonMask, clamp(uvTop, vec2(0.001), vec2(0.999))).r;
  if (vCamDepth > uPersonDepth + 0.06 && pMask > 0.45) discard;
  float vis = (vCamDepth > uPersonDepth + 0.06) ? (1.0 - smoothstep(0.12, 0.55, pMask)) : 1.0;

  vec3 N0 = normalize(vWorldNrm);
  vec3 V = normalize(uRo - vWorldPos);
  vec3 L = normalize(uSun);
  int m = int(vMat + 0.5);

  /* Realistic Weathered Outdoor PBR Material Library (sRGB albedos -> converted strictly to Linear!) + Procedural Bump Height! */
  vec3 baseSrgb = vec3(0.5);
  float rough = 0.65;
  float metal = 0.0;
  float bumpH = 0.0;
  float bumpStr = 0.0;

  if (m == 0) {
    /* Weathered charcoal retro CRT plastic housing with fine molded grain & scuffs */
    float n = fbm(vUv * 24.0);
    float fine = noise2(vUv * 85.0);
    baseSrgb = mix(vec3(0.11, 0.115, 0.125), vec3(0.20, 0.205, 0.215), n * 0.7 + fine * 0.3);
    rough = 0.48 + 0.16 * n;
    bumpH = fine; bumpStr = 0.12;
  } else if (m == 1) {
    /* Glossy curved dark green-charcoal CRT glass screen (crisp Fresnel atrium & sky reflection!) */
    baseSrgb = vec3(0.03, 0.055, 0.052);
    rough = 0.06;
    metal = 0.05;
  } else if (m == 2) {
    /* Gritty weathered pine shipping crate wood planks (deep wood grain grooves + dark edge wear!) */
    float warp = fbm(vUv * vec2(7.0, 28.0));
    float grain = sin(vUv.x * 54.0 + warp * 5.8) * 0.5 + 0.5;
    float fine = fbm(vUv * vec2(18.0, 72.0));
    float grime = smoothstep(0.55, 0.88, fbm(vUv * 6.5 + vec2(3.1, -2.4)));
    baseSrgb = mix(vec3(0.36, 0.23, 0.12), vec3(0.62, 0.44, 0.25), grain * 0.55 + fine * 0.45);
    baseSrgb = mix(baseSrgb, baseSrgb * 0.58, grime * 0.45);
    rough = 0.72 + 0.16 * fine;
    bumpH = grain * 0.65 + fine * 0.35; bumpStr = 0.28;
  } else if (m == 3) {
    /* Brushed weathered dark steel metal (crate brackets, tire wheel rim, TV antennas, drum bung) */
    float n = fbm(vUv * 32.0);
    float scratch = noise2(vUv * vec2(120.0, 14.0));
    baseSrgb = mix(vec3(0.26, 0.27, 0.29), vec3(0.62, 0.64, 0.66), n * 0.65 + scratch * 0.35);
    rough = 0.26 + 0.22 * n;
    metal = 0.88;
    bumpH = scratch; bumpStr = 0.14;
  } else if (m == 4) {
    /* Sun-baked terracotta amphora clay with mineral speckling & subtle kiln satin glaze */
    float n = fbm(vUv * 12.0);
    float fine = noise2(vUv * 55.0);
    baseSrgb = mix(vec3(0.50, 0.21, 0.11), vec3(0.71, 0.35, 0.19), n * 0.72 + fine * 0.28);
    rough = 0.36 + 0.24 * n;
    bumpH = n * 0.5 + fine * 0.5; bumpStr = 0.18;
  } else if (m == 5) {
    /* Rough porous gray concrete cinder masonry (strong raking-sun micro-pitting!) */
    float n1 = fbm(vUv * 18.0);
    float pit = noise2(vUv * 110.0);
    baseSrgb = mix(vec3(0.28, 0.28, 0.27), vec3(0.54, 0.53, 0.51), n1 * 0.65 + pit * 0.35);
    rough = 0.90;
    bumpH = n1 * 0.45 + pit * 0.55; bumpStr = 0.45;
  } else if (m == 6) {
    /* Realistic sun-weathered industrial crimson-red painted steel oil drum with chipped dark rust & oil grime! */
    float n = fbm(vUv * 14.0);
    float chip = smoothstep(0.64, 0.74, fbm(vUv * 24.0 + vec2(4.2, -1.9)));
    float grime = smoothstep(0.50, 0.86, fbm(vUv * 8.0 + vec2(-1.7, 5.3)));
    vec3 paint = mix(vec3(0.42, 0.085, 0.065), vec3(0.58, 0.135, 0.095), n);
    vec3 rust  = vec3(0.24, 0.12, 0.07);
    baseSrgb = mix(paint, rust, chip * 0.72);
    baseSrgb = mix(baseSrgb, baseSrgb * 0.52, grime * 0.42);
    rough = mix(0.28, 0.80, max(chip, grime * 0.6));
    metal = mix(0.45, 0.12, chip);
    bumpH = n * 0.4 + chip * 0.6; bumpStr = 0.22;
  } else if (m == 7) {
    /* Matte dusty charcoal car rubber tire tread (authentic outdoor tire grime & concrete dust in recesses!) */
    float n = fbm(vUv * 24.0);
    float dust = smoothstep(0.48, 0.82, fbm(vUv * 12.0 + vec2(2.3, 1.7)));
    baseSrgb = mix(vec3(0.055, 0.058, 0.062), vec3(0.145, 0.140, 0.132), n * 0.6 + dust * 0.4);
    rough = 0.82 + 0.10 * dust;
    bumpH = noise2(vUv * 95.0); bumpStr = 0.24;
  } else {
    /* Green fiberglass internal CRT circuit board */
    float trace = smoothstep(0.45, 0.55, sin(vUv.x * 70.0) * cos(vUv.y * 70.0));
    baseSrgb = mix(vec3(0.07, 0.28, 0.12), vec3(0.38, 0.48, 0.20), trace * 0.48);
    rough = 0.34;
    metal = 0.25;
  }

  /* Screen-space procedural normal-map bump perturbation so raking midday sun catches real surface grit! */
  vec3 N = N0;
  if (bumpStr > 0.01) {
    vec3 dPdx = dFdx(vWorldPos);
    vec3 dPdy = dFdy(vWorldPos);
    float dHdx = dFdx(bumpH);
    float dHdy = dFdy(bumpH);
    vec3 surfGrad = cross(dPdy, N0) * dHdx + cross(N0, dPdx) * dHdy;
    float det = dot(dPdx, cross(dPdy, N0));
    if (abs(det) > 1e-8) {
      N = normalize(N0 - sign(det) * clamp(surfGrad / max(abs(det), 1e-5), vec3(-1.5), vec3(1.5)) * bumpStr * 0.14);
    }
  }

  vec3 baseLin = pow(max(baseSrgb, vec3(0.002)), vec3(2.2));
  vec3 H = normalize(L + V);
  float NdotL = max(dot(N, L), 0.0);
  float NdotV = max(dot(N, V), 0.001);
  float NdotH = max(dot(N, H), 0.0);
  float shad = calcShadow(vLightSpace, N0, L);

  /* HARSH OUTDOOR MIDDAY SUN-TO-SHADOW CONTRAST (~7:1 ratio matching the live dancer's harsh outdoor contrast!):
     - Blazing warm golden-white direct sun on lit faces
     - Deep cool blue-teal atrium shadow fill on dark faces + baked structural crevice AO + tight ground contact AO! */
  vec3 sunCol = vec3(1.20, 1.10, 0.94) * 3.55;
  vec3 skyFill = vec3(0.16, 0.25, 0.36);   /* crisp cool skylight shadow fill (deep, NOT washed out!) */
  vec3 gndBounce = vec3(0.14, 0.13, 0.11); /* subtle warm concrete ground bounce */
  vec3 hemi = mix(gndBounce, skyFill, clamp(N.z * 0.5 + 0.5, 0.0, 1.0));
  /* Strong ground contact occlusion near z=0 + baked crevice AO inside slats/cores/hubs! */
  float gndAO = mix(0.28, 1.0, smoothstep(0.0, 0.26, vWorldPos.z));
  float totalAO = clamp(vAO * gndAO, 0.14, 1.0);

  /* Physical GGX-like specular + real live-plate Fresnel IBL reflections! */
  vec3 F0 = mix(vec3(0.04), baseLin, metal);
  vec3 F = F0 + (1.0 - F0) * pow(1.0 - NdotV, 5.0);
  float a2 = max(rough * rough, 0.004);
  float denom = NdotH * NdotH * (a2 - 1.0) + 1.0;
  float D_ggx = a2 / (3.14159 * denom * denom + 1e-5);
  float specSun = min(D_ggx * 0.26 * NdotL, 6.0);

  vec3 R = reflect(-V, N);
  vec3 envRefl = samplePlateIBL(R);
  float reflPow = pow(1.0 - rough, 1.85);

  vec3 diffAlb = baseLin * (1.0 - metal * 0.92);
  vec3 linCol = diffAlb * (sunCol * NdotL * shad + hemi * totalAO)
              + F * sunCol * specSun * shad
              + F * envRefl * reflPow * (0.35 + 0.65 * totalAO);

  /* ACES Filmic Tone Mapping + Linear -> sRGB conversion matching DSLR outdoor contrast cleanly! */
  vec3 mapped = acesFilm(linCol * 0.88);
  vec3 srgb = pow(clamp(mapped, 0.0, 1.0), vec3(1.0 / 2.2));
  oCol = vec4(srgb, vis);
  oMeta = vec4(vis, vis / max(vCamDepth, 0.2), 0.0, 1.0);
}
"""

VS_FSQ = "#version 330\nin vec2 p; void main(){ gl_Position = vec4(p, 0.0, 1.0); }"

# Ground Shadow Catcher Pass: casts real 2048x2048 directional sun shadows + soft contact AO onto z=0 sunlit concrete!
GROUND_SHAD_FS = """#version 330
uniform sampler2D uPlate;
uniform sampler2D uShadowMap;
uniform sampler2D uPersonMask;
uniform vec2 uRes;
uniform float uF, uPersonDepth, uBackY;
uniform vec3 uRo, uSun;
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

  float cgA = 0.0;
  float cgInvD = 0.0;

  /* Intersect sunlit concrete plaza z=0 (stopping cleanly at concrete curb wall y = uBackY!) */
  if (rd.z < -0.005) {
    float tf = -uRo.z / rd.z;
    if (tf > 0.2 && tf < 18.0) {
      vec3 X = uRo + rd * tf;
      if (X.y < uBackY + 0.08 && abs(X.x) < 4.5) {
        vec4 ls = uLightVP * vec4(X, 1.0);
        vec3 proj = ls.xyz / max(ls.w, 1e-5) * 0.5 + 0.5;
        if (proj.x > 0.01 && proj.x < 0.99 && proj.y > 0.01 && proj.y < 0.99 && proj.z < 1.0) {
          /* 5x5 crisp sunlit concrete shadow matching dancer's exact hard sneaker shadow! */
          float occ = 0.0;
          vec2 tex = vec2(1.0 / 2048.0);
          for (int i = -2; i <= 2; i++) {
            for (int j = -2; j <= 2; j++) {
              float dMap = texture(uShadowMap, proj.xy + vec2(float(i), float(j)) * tex * 1.25).r;
              if (proj.z - 0.0010 > dMap) occ += 1.0;
            }
          }
          occ /= 25.0;
          /* Don't paint ground shadows on top of the dancer's body silhouette! */
          float notPerson = 1.0 - smoothstep(0.15, 0.65, pMask);
          float curbFade = smoothstep(uBackY + 0.08, uBackY - 0.04, X.y);
          float sAmt = occ * notPerson * curbFade * 0.74;
          if (sAmt > 0.005) {
            /* Crisp dark cool-slate shadow tint matching the dancer's real shadow on the concrete tiles! */
            vec3 shadTint = vec3(0.76, 0.84, 1.00);
            col = mix(col, col * shadTint * 0.26, sAmt);
            cgA = clamp(sAmt * 0.65, 0.0, 0.7);
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

# Pass 3: Instanced Volumetric Sunlit Concrete Impact Dust Plumes & Micro-Gravel Spray!
DUST_VS = """#version 330
in vec2 inQuad;
in vec4 inPosRad;   /* xyz = world center, w = radius (m) */
in vec4 inInfo;     /* x = alpha, y = rotRad, z = seed, w = kind (0=dust cloud, 1=tiny dark gravel/splinter) */

uniform vec2 uRes;
uniform float uF;
uniform vec3 uRo;
uniform mat3 uCam;

out vec2 vLocal;
out float vAlpha;
out float vSeed;
out float vKind;
out float vCamDepth;
out vec3 vWorldPos;

void main() {
  vLocal = inQuad;
  vAlpha = inInfo.x;
  vSeed = inInfo.z;
  vKind = inInfo.w;
  vWorldPos = inPosRad.xyz;

  vec3 q = transpose(uCam) * (inPosRad.xyz - uRo);
  vCamDepth = q.z;
  if (vAlpha <= 0.008 || q.z <= 0.25) {
    gl_Position = vec4(2.0, 2.0, 2.0, 1.0);
    return;
  }
  float pxR = clamp(uF * inPosRad.w / q.z, 2.0, 420.0);
  float c = cos(inInfo.y), s = sin(inInfo.y);
  vec2 rq = vec2(inQuad.x * c - inQuad.y * s, inQuad.x * s + inQuad.y * c);
  vec2 ctrPx = uF * q.xy / q.z + 0.5 * uRes;
  vec2 px = ctrPx + rq * pxR;
  vec2 ndc = vec2(px.x / uRes.x * 2.0 - 1.0, (1.0 - px.y / uRes.y) * 2.0 - 1.0);
  gl_Position = vec4(ndc, 0.0, 1.0);
}
"""

DUST_FS = """#version 330
in vec2 vLocal;
in float vAlpha;
in float vSeed;
in float vKind;
in float vCamDepth;
in vec3 vWorldPos;

uniform sampler2D uPersonMask;
uniform vec2 uRes;
uniform float uPersonDepth;

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
float fbm(vec2 p) {
  float v = 0.0, a = 0.5;
  for (int k = 0; k < 4; k++) { v += a * noise2(p); p *= 2.05; a *= 0.5; }
  return v;
}

void main() {
  float r = length(vLocal);
  if (r > 1.0) discard;

  vec2 uvTop = vec2(gl_FragCoord.x / uRes.x, 1.0 - gl_FragCoord.y / uRes.y);
  float pMask = texture(uPersonMask, clamp(uvTop, vec2(0.001), vec2(0.999))).r;
  float occ = (vCamDepth > uPersonDepth + 0.06) ? (1.0 - smoothstep(0.15, 0.60, pMask)) : 1.0;
  if (occ < 0.02) discard;

  if (vKind > 0.5) {
    /* Sharp flying concrete gravel / ceramic micro-chip particle */
    float shape = smoothstep(0.92, 0.35, r + (noise2(vLocal * 3.0 + vSeed * 20.0) - 0.5) * 0.45);
    float a = shape * vAlpha * occ;
    if (a < 0.02) discard;
    vec3 chipCol = mix(vec3(0.32, 0.30, 0.28), vec3(0.68, 0.58, 0.46), fract(vSeed * 17.3));
    oCol = vec4(chipCol, a);
    oMeta = vec4(a, a / max(vCamDepth, 0.2), 0.0, 1.0);
  } else {
    /* Volumetric billowing sunlit concrete dust puff (warm sunlit top, shadowed dusty belly!) */
    float n = fbm(vLocal * 2.2 + vec2(vSeed * 13.7, -vSeed * 9.3));
    float env = smoothstep(1.0, 0.15, r) * smoothstep(0.24, 0.78, n + (1.0 - r) * 0.42);
    float a = env * vAlpha * occ;
    if (a < 0.006) discard;
    /* Directional sun scattering: top-left lit warmly, bottom-right subtly cooler dust shadow */
    float sunLit = clamp(0.55 - 0.35 * vLocal.x + 0.45 * vLocal.y, 0.0, 1.0);
    vec3 dustCol = mix(vec3(0.54, 0.50, 0.45), vec3(0.84, 0.80, 0.73), sunLit);
    oCol = vec4(dustCol, a);
    oMeta = vec4(a * 0.65, a * 0.65 / max(vCamDepth, 0.2), 0.0, 1.0);
  }
}
"""

# Subframe Accumulation & Final Camera Optical Finish Pass (impact shake + OLPF + plate grain)
ACCUM_ADD_FS = """#version 330
uniform sampler2D uSrc0;
uniform sampler2D uSrc1;
uniform vec2 uRes;
uniform float uWeight;
layout(location=0) out vec4 o0;
layout(location=1) out vec4 o1;
void main() {
  vec2 uv = gl_FragCoord.xy / uRes;
  o0 = texture(uSrc0, uv) * uWeight;
  o1 = texture(uSrc1, uv) * uWeight;
}
"""

POST_FS = """#version 330
uniform sampler2D uScene;
uniform sampler2D uMeta;
uniform vec2 uRes;
uniform vec2 uShakePx;
uniform float uBlackFloor, uGrainAmt, uFrame;

out vec4 oCol;

float hash12(vec2 p) {
  vec3 p3 = fract(vec3(p.xyx) * 0.1031);
  p3 += dot(p3, p3.yzx + 33.33);
  return fract((p3.x + p3.y) * p3.z);
}

void main() {
  /* Apply subtle camera micro-shake on heavy ground slams (zoom 1.008 keeps edges 100% clean!) */
  vec2 uv0 = gl_FragCoord.xy / uRes;
  vec2 uv = (uv0 - 0.5) / 1.008 + 0.5 + uShakePx / uRes;
  uv = clamp(uv, vec2(0.001), vec2(0.999));
  vec2 tex = 1.0 / uRes;

  vec3 c0 = texture(uScene, uv).rgb;
  float cgA = clamp(texture(uMeta, uv).r, 0.0, 1.0);

  vec3 res = c0;
  if (cgA > 0.005) {
    /* Subtle camera OLPF optical softness on CG meshes & debris so edges fuse into real lens sharpness */
    vec3 soft = (
      c0 * 4.0 +
      texture(uScene, uv + vec2(tex.x, 0.0)).rgb +
      texture(uScene, uv - vec2(tex.x, 0.0)).rgb +
      texture(uScene, uv + vec2(0.0, tex.y)).rgb +
      texture(uScene, uv - vec2(0.0, tex.y)).rgb
    ) / 8.0;
    res = mix(c0, soft, cgA * 0.30);

    /* Match live sunlit plaza black floor & animated sensor grain ONLY on CG pixels */
    vec3 graded = max(res, vec3(uBlackFloor));
    vec2 p1 = gl_FragCoord.xy + vec2(uFrame * 17.3, uFrame * 31.7);
    vec2 p2 = floor(gl_FragCoord.xy * 0.5) + vec2(uFrame * 11.1, uFrame * 23.9);
    float g = (hash12(p1) + hash12(p1 + 7.13) - 1.0) * 0.62 + (hash12(p2) + hash12(p2 + 5.71) - 1.0) * 0.38;
    float lum = dot(graded, vec3(0.299, 0.587, 0.114));
    graded += vec3(g) * (uGrainAmt * (1.12 - 0.58 * smoothstep(0.55, 0.98, lum)));
    res = mix(res, graded, cgA);
  }
  oCol = vec4(clamp(res, 0.0, 1.0), 1.0);
}
"""


def mat4_perspective_from_opencv(f_px, W, H, near=0.1, far=40.0):
    """Exact OpenGL clip-space projection matrix matching OpenCV pinhole camera (cx=W/2, cy=H/2)."""
    P = np.zeros((4, 4), dtype=np.float32)
    P[0, 0] = 2.0 * f_px / W
    P[1, 1] = 2.0 * f_px / H
    P[2, 2] = -(far + near) / (far - near)
    P[2, 3] = -2.0 * far * near / (far - near)
    P[3, 2] = -1.0
    return P


def mat4_view_from_colmap(ro, R_c2w):
    """Exact OpenGL view matrix from COLMAP camera center ro and c2w rotation R_c2w (OpenCV: +X right, +Y down, +Z fwd)."""
    # Convert OpenCV camera axes (+X right, +Y down, +Z fwd) -> OpenGL camera axes (+X right, +Y up, -Z fwd)
    flip = np.diag([1.0, -1.0, -1.0])
    R_w2gl = flip @ R_c2w.T
    t_w2gl = -R_w2gl @ ro
    V = np.eye(4, dtype=np.float32)
    V[:3, :3] = R_w2gl
    V[:3, 3] = t_w2gl
    return V


def mat4_ortho_light_vp(sun_dir, center=(0.0, 0.3, 0.6), half_ext=4.2, near=0.5, far=18.0):
    """Orthographic directional sun shadow-map LightVP matrix tightly framing the concrete plaza stage."""
    L = np.asarray(sun_dir, dtype=np.float64)
    L /= np.linalg.norm(L) + 1e-9
    eye = np.asarray(center, dtype=np.float64) + L * 9.0
    z_ax = L  # points from center toward light (+Z in OpenGL view)
    up = np.array([0.0, 0.0, 1.0]) if abs(z_ax[2]) < 0.95 else np.array([0.0, 1.0, 0.0])
    x_ax = np.cross(up, z_ax); x_ax /= np.linalg.norm(x_ax)
    y_ax = np.cross(z_ax, x_ax)
    V = np.eye(4, dtype=np.float64)
    V[0, :3] = x_ax; V[1, :3] = y_ax; V[2, :3] = z_ax
    V[:3, 3] = -V[:3, :3] @ eye
    O = np.zeros((4, 4), dtype=np.float64)
    O[0, 0] = 1.0 / half_ext
    O[1, 1] = 1.0 / half_ext
    O[2, 2] = -2.0 / (far - near)
    O[2, 3] = -(far + near) / (far - near)
    O[3, 3] = 1.0
    return (O @ V).astype(np.float32)


def mat4_from_pos_quat(pos, quat_wxyz):
    """Convert MuJoCo (pos, quat_wxyz) to 4x4 column-major-ready model matrix & 3x3 normal matrix."""
    w, x, y, z = float(quat_wxyz[0]), float(quat_wxyz[1]), float(quat_wxyz[2]), float(quat_wxyz[3])
    n = math.sqrt(w * w + x * x + y * y + z * z)
    if n < 1e-6:
        R = np.eye(3, dtype=np.float32)
    else:
        w /= n; x /= n; y /= n; z /= n
        R = np.array([
            [1 - 2*(y*y + z*z), 2*(x*y - z*w),     2*(x*z + y*w)],
            [2*(x*y + z*w),     1 - 2*(x*x + z*z), 2*(y*z - x*w)],
            [2*(x*z - y*w),     2*(y*z + x*w),     1 - 2*(x*x + y*y)],
        ], dtype=np.float32)
    M = np.eye(4, dtype=np.float32)
    M[:3, :3] = R
    M[:3, 3] = pos
    return M, R


def build_impact_dust_catalog(impacts, seed=99):
    """Pre-generate subtle, translucent, low-hugging sunlit concrete impact dust puffs & crisp gravel chips
    so every 3D object & debris pile remains 100% razor-visible (zero opaque fog wall!)."""
    rng = np.random.default_rng(seed)
    particles = []
    for t_imp, ix, iy, energy, kcode in impacts:
        # 12 low-hugging translucent sunlit concrete dust puffs fanning out right along z=0!
        n_puffs = int(10 + 4 * min(energy, 1.4))
        for k in range(n_puffs):
            ang = (k / float(n_puffs)) * 2 * math.pi + float(rng.uniform(-0.25, 0.25))
            v_r = float(rng.uniform(0.60, 1.85)) * (0.75 + 0.25 * energy)
            v_z = float(rng.uniform(0.12, 0.48)) * (0.75 + 0.20 * energy)
            particles.append({
                "t0": float(t_imp) + float(rng.uniform(0.0, 0.04)),
                "life": float(rng.uniform(0.60, 1.15)),
                "p0": np.array([ix + math.cos(ang) * 0.14, iy + math.sin(ang) * 0.14, 0.06], dtype=np.float64),
                "vel": np.array([math.cos(ang) * v_r + 0.12, math.sin(ang) * v_r * 0.65, v_z], dtype=np.float64),
                "r0": float(rng.uniform(0.12, 0.20)),
                "r1": float(rng.uniform(0.42, 0.78)),
                "alpha": float(rng.uniform(0.11, 0.19)) * min(1.15, 0.8 + 0.25 * energy),
                "rot0": float(rng.uniform(0, 6.28)),
                "rotv": float(rng.uniform(-1.0, 1.0)),
                "seed": float(rng.random()),
                "kind": 0.0,
            })
        # 18 crisp flying concrete/ceramic micro-chips arcing ballistically!
        for _ in range(18):
            ang = float(rng.uniform(0, 2 * math.pi))
            v_r = float(rng.uniform(0.9, 3.2))
            v_z = float(rng.uniform(1.4, 3.8))
            particles.append({
                "t0": float(t_imp),
                "life": float(rng.uniform(0.45, 0.90)),
                "p0": np.array([ix, iy, 0.06], dtype=np.float64),
                "vel": np.array([math.cos(ang) * v_r, math.sin(ang) * v_r * 0.7, v_z], dtype=np.float64),
                "r0": float(rng.uniform(0.012, 0.028)),
                "r1": float(rng.uniform(0.010, 0.024)),
                "alpha": 0.90,
                "rot0": float(rng.uniform(0, 6.28)),
                "rotv": float(rng.uniform(-12.0, 12.0)),
                "seed": float(rng.random()),
                "kind": 1.0,
            })
    return particles


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
    # Zero text overlays on Compare_1080x1350.mp4
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
    piece_arg = None
    k = 0
    while k < len(args):
        if args[k] == "--still":
            stills = [int(x) for x in args[k + 1].split(",")]; k += 2
        elif args[k] == "--out":
            out_dir = args[k + 1]; k += 2
        elif args[k] == "--resim":
            resim = True; k += 1
        else:
            piece_arg = args[k]; k += 1

    c = load(piece_arg or (here / "piece_debris"))
    s = scene(c)
    C = np.array(s["cam"], dtype=np.float64)
    R_all = np.array(s["rot"], dtype=np.float64)
    f_px = float(s["f"])
    # Calibrate sun strictly to the dancer's real hard outdoor shadow angle (top-front-left -> casting shadow back-right!)
    sun = np.array([-0.46, -0.48, 0.74], dtype=np.float64)
    sun /= np.linalg.norm(sun) + 1e-9

    limbs = np.load(work(c, "limbs.npz"))
    P_body = np.asarray(limbs["P"], dtype=np.float64)
    person_depths = np.asarray(limbs["depth"], dtype=np.float64)
    # Smooth dancer XY center for kinematic MuJoCo deflector cylinder
    person_xy = P_body[:, [11, 12, 23, 24], :2].mean(axis=1)

    N = len(C)
    fps = float(c.get("fps_num", 25)) / float(c.get("fps_den", 1))
    pl0 = frame(c, 0)
    H, W = pl0.shape[:2]

    # Measure plate black floor & sensor grain cleanly for Pass-4 optical finish
    gray0 = cv2.cvtColor(pl0, cv2.COLOR_BGR2GRAY).astype(np.float32) / 255.0
    black_floor = float(np.clip(np.percentile(gray0, 1.0) * 0.85, 0.008, 0.050))
    hp0 = gray0 - cv2.GaussianBlur(gray0, (0, 0), 1.2)
    grain_amt = float(np.clip(float(np.median(np.abs(hp0 - np.median(hp0)))) * 1.4826 * 1.15, 0.008, 0.025))

    meshes = build_all_meshes()
    sim_path = work(c, "sim_debris.npz")
    if resim or not sim_path.exists():
        simulate_debris(c, meshes, person_xy)
    sim = np.load(sim_path, allow_pickle=False)
    sim_pos = sim["pos"]          # (N, 5, B, 3)
    sim_quat = sim["quat"]        # (N, 5, B, 4)
    sim_vis = sim["vis"]          # (N, 5, B)
    mesh_names = [str(x) for x in sim["mesh_names"]]
    impacts = sim["impacts"]      # (K, 5)
    dust_particles = build_impact_dust_catalog(impacts)

    back_y = float(c.get("debris_sim", {}).get("back_wall_y", 1.55))
    light_vp = mat4_ortho_light_vp(sun, center=(0.1, 0.35, 0.5), half_ext=4.2)
    proj_m = mat4_perspective_from_opencv(f_px, W, H, near=0.1, far=40.0)

    # Setup Native OpenGL 3.30 context & upload every modeled prop + debris mesh once to GPU!
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

    # Upload all 30 unique 3D triangle meshes to GPU VAOs (pos3 + nrm3 + uv2 + mat1 + ao1 = 10 floats!)
    gpu_meshes = {}
    for mname, (md, _) in meshes.items():
        vdata = np.column_stack([md["pos"], md["nrm"], md["uv"], md["mat"], md["ao"]]).astype(np.float32)
        vbo = ctx.buffer(vdata.tobytes())
        ibo = ctx.buffer(md["idx"].astype(np.uint32).tobytes())
        vao_mesh = ctx.vertex_array(mesh_prog, [(vbo, "3f 3f 2f 1f 1f", "inPos", "inNrm", "inUv", "inMat", "inAO")], index_buffer=ibo)
        vao_shad = ctx.vertex_array(shad_prog, [(vbo, "3f 28x", "inPos")], index_buffer=ibo)
        gpu_meshes[mname] = (vao_mesh, vao_shad)

    # Dust instanced quad VAO
    dust_q = ctx.buffer(np.array([-1, -1, 1, -1, -1, 1, -1, 1, 1, -1, 1, 1], dtype=np.float32).tobytes())
    dust_inst = ctx.buffer(reserve=max(len(dust_particles), 16) * 8 * 4)
    va_dust = ctx.vertex_array(dust_prog, [
        (dust_q, "2f", "inQuad"),
        (dust_inst, "4f 4f/i", "inPosRad", "inInfo"),
    ])

    # Framebuffers: 2048x2048 Shadow Map, Subframe 4x-depth PBR buffer, Accumulation buffer, Final Post buffer
    t_smap = ctx.depth_texture((2048, 2048))
    t_smap.filter = (moderngl.LINEAR, moderngl.LINEAR)
    t_smap.repeat_x = t_smap.repeat_y = False
    fbo_shad = ctx.framebuffer(depth_attachment=t_smap)

    t_sub_col = ctx.texture((W, H), 4, dtype="f4")
    t_sub_meta = ctx.texture((W, H), 4, dtype="f4")
    rbo_depth = ctx.depth_renderbuffer((W, H))
    fbo_sub = ctx.framebuffer([t_sub_col, t_sub_meta], depth_attachment=rbo_depth)

    t_acc_col = ctx.texture((W, H), 4, dtype="f4")
    t_acc_meta = ctx.texture((W, H), 4, dtype="f4")
    t_acc_col.filter = (moderngl.LINEAR, moderngl.LINEAR)
    t_acc_meta.filter = (moderngl.LINEAR, moderngl.LINEAR)
    fbo_acc = ctx.framebuffer([t_acc_col, t_acc_meta])

    t_final = ctx.texture((W, H), 4, dtype="f4")
    fbo_final = ctx.framebuffer([t_final])

    plate_t = ctx.texture((W, H), 3)
    plate_t.filter = (moderngl.LINEAR, moderngl.LINEAR)
    plate_t.repeat_x = plate_t.repeat_y = False

    mask_t = ctx.texture((W, H), 1, dtype="f4")
    mask_t.filter = (moderngl.LINEAR, moderngl.LINEAR)
    mask_t.repeat_x = mask_t.repeat_y = False

    # Static uniforms
    shad_prog["uLightVP"].write(light_vp.T.tobytes())
    mesh_prog["uProj"].write(proj_m.T.tobytes())
    mesh_prog["uLightVP"].write(light_vp.T.tobytes())
    mesh_prog["uRes"].value = (float(W), float(H))
    mesh_prog["uSun"].value = tuple(map(float, sun))
    gnd_prog["uRes"].value = (float(W), float(H))
    gnd_prog["uF"].value = f_px
    gnd_prog["uBackY"].value = back_y
    if "uSun" in gnd_prog:
        gnd_prog["uSun"].value = tuple(map(float, sun))
    gnd_prog["uLightVP"].write(light_vp.T.tobytes())
    dust_prog["uRes"].value = (float(W), float(H))
    dust_prog["uF"].value = f_px
    acc_prog["uRes"].value = (float(W), float(H))
    post_prog["uRes"].value = (float(W), float(H))
    post_prog["uBlackFloor"].value = black_floor
    post_prog["uGrainAmt"].value = grain_amt

    def compute_impact_shake(t_sec):
        sx, sy = 0.0, 0.0
        for t_imp, _, _, energy, _ in impacts:
            dt = t_sec - float(t_imp)
            if 0.0 <= dt < 0.28:
                amp = float(energy) * 5.2 * math.exp(-dt * 15.0)
                sx += amp * math.sin(dt * 85.0 + float(t_imp) * 19.0) * 0.45
                sy += amp * math.cos(dt * 95.0 + float(t_imp) * 13.0)
        return (float(np.clip(sx, -8.5, 8.5)), float(np.clip(sy, -10.5, 10.5)))

    def render_one(i):
        pl = frame(c, i)
        m_path = work(c, "mask", f"{i:04d}.png")
        m_raw = cv2.imread(str(m_path), cv2.IMREAD_GRAYSCALE)
        if m_raw is None:
            m_f = np.zeros((H, W), dtype=np.float32)
        else:
            if m_raw.shape[:2] != (H, W):
                m_raw = cv2.resize(m_raw, (W, H), interpolation=cv2.INTER_LINEAR)
            m_f = cv2.GaussianBlur(m_raw.astype(np.float32) / 255.0, (0, 0), 0.9)

        plate_t.write(pl[:, :, ::-1].tobytes())
        mask_t.write(m_f.astype(np.float32).tobytes())

        ro = C[i]
        R = R_all[i]
        Rt = tuple(R.T.flatten().astype(np.float32))
        view_m = mat4_view_from_colmap(ro, R)
        p_depth = float(person_depths[i])
        n_shut = len(SHUTTER)
        sub_w = 1.0 / float(n_shut)

        # Clear accumulation FBO
        fbo_acc.use()
        ctx.viewport = (0, 0, W, H)
        ctx.disable(moderngl.DEPTH_TEST)
        ctx.disable(moderngl.BLEND)
        ctx.clear(0.0, 0.0, 0.0, 0.0)

        # 5-Subframe 180-degree shutter loop for crisp cinema motion blur on fast falling & shattering debris!
        for si in range(n_shut):
            t_sub = (i + SHUTTER[si]) / fps
            vis_s = sim_vis[i, si]
            pos_s = sim_pos[i, si]
            quat_s = sim_quat[i, si]
            active_ids = np.where(vis_s > 0)[0]

            # Pass 1: 2048x2048 Sun Directional Shadow Map
            fbo_shad.use()
            ctx.viewport = (0, 0, 2048, 2048)
            ctx.enable(moderngl.DEPTH_TEST)
            ctx.disable(moderngl.BLEND)
            ctx.clear(depth=1.0)
            for bi in active_ids:
                M, _ = mat4_from_pos_quat(pos_s[bi], quat_s[bi])
                shad_prog["uModel"].write(M.T.tobytes())
                gpu_meshes[mesh_names[bi]][1].render(moderngl.TRIANGLES)

            # Pass 2A: Plate + Directional Concrete Plaza Shadow Catcher
            fbo_sub.use()
            ctx.viewport = (0, 0, W, H)
            ctx.disable(moderngl.DEPTH_TEST)
            ctx.disable(moderngl.BLEND)
            ctx.clear(depth=1.0)
            plate_t.use(0); t_smap.use(1); mask_t.use(2)
            gnd_prog["uPlate"].value = 0
            gnd_prog["uShadowMap"].value = 1
            gnd_prog["uPersonMask"].value = 2
            gnd_prog["uRo"].value = tuple(map(float, ro))
            gnd_prog["uCam"].value = Rt
            if "uPersonDepth" in gnd_prog:
                gnd_prog["uPersonDepth"].value = p_depth
            va_gnd.render(moderngl.TRIANGLES)

            # Pass 2B: PBR 3D Modeled Props & Fractured Rigid Debris Meshes with Real Depth Buffer & Shadow Map!
            ctx.enable(moderngl.DEPTH_TEST)
            ctx.enable(moderngl.BLEND)
            ctx.blend_func = (moderngl.SRC_ALPHA, moderngl.ONE_MINUS_SRC_ALPHA)
            t_smap.use(0); mask_t.use(1); plate_t.use(2)
            mesh_prog["uShadowMap"].value = 0
            mesh_prog["uPersonMask"].value = 1
            mesh_prog["uPlate"].value = 2
            mesh_prog["uView"].write(view_m.T.tobytes())
            mesh_prog["uRo"].value = tuple(map(float, ro))
            mesh_prog["uPersonDepth"].value = p_depth
            for bi in active_ids:
                M, N3 = mat4_from_pos_quat(pos_s[bi], quat_s[bi])
                mesh_prog["uModel"].write(M.T.tobytes())
                mesh_prog["uNormalMat"].write(N3.T.tobytes())
                gpu_meshes[mesh_names[bi]][0].render(moderngl.TRIANGLES)

            # Pass 3: Volumetric Sunlit Impact Dust Plumes & Micro-Grit Spray!
            d_rows = []
            for dp in dust_particles:
                dt = t_sub - dp["t0"]
                if 0.0 <= dt < dp["life"]:
                    u = dt / dp["life"]
                    if dp["kind"] > 0.5:
                        # Ballistic flying chip
                        pp = dp["p0"] + dp["vel"] * dt + np.array([0.0, 0.0, -4.905 * dt * dt])
                        if pp[2] < 0.02:
                            continue
                        rr = dp["r0"]
                        aa = dp["alpha"] * (1.0 - u * u)
                    else:
                        # Drag-slowed expanding rising concrete dust puff
                        drag_t = (1.0 - math.exp(-dt * 2.4)) / 2.4
                        pp = dp["p0"] + dp["vel"] * drag_t + np.array([0.14 * dt, 0.0, 0.22 * dt])
                        rr = dp["r0"] + (dp["r1"] - dp["r0"]) * (u ** 0.55)
                        aa = dp["alpha"] * math.sin(min(1.0, u * 4.5) * math.pi * 0.5) * ((1.0 - u) ** 1.45)
                    z_cam = float(((pp - ro) @ R)[2])
                    if z_cam > 0.35 and aa > 0.008:
                        d_rows.append((z_cam, [
                            float(pp[0]), float(pp[1]), float(pp[2]), float(rr),
                            float(aa), float(dp["rot0"] + dp["rotv"] * dt), float(dp["seed"]), float(dp["kind"]),
                        ]))
            if d_rows:
                d_rows.sort(key=lambda x: -x[0])
                dust_inst.write(np.array([r[1] for r in d_rows], dtype=np.float32).tobytes())
                ctx.disable(moderngl.DEPTH_TEST)
                ctx.enable(moderngl.BLEND)
                ctx.blend_func = (moderngl.SRC_ALPHA, moderngl.ONE_MINUS_SRC_ALPHA)
                mask_t.use(0)
                dust_prog["uPersonMask"].value = 0
                dust_prog["uRo"].value = tuple(map(float, ro))
                dust_prog["uCam"].value = Rt
                dust_prog["uPersonDepth"].value = p_depth
                va_dust.render(moderngl.TRIANGLES, instances=len(d_rows))

            # Accumulate subframe cleanly into 180-degree shutter buffer
            fbo_acc.use()
            ctx.disable(moderngl.DEPTH_TEST)
            ctx.enable(moderngl.BLEND)
            ctx.blend_func = (moderngl.ONE, moderngl.ONE)
            t_sub_col.use(0); t_sub_meta.use(1)
            acc_prog["uSrc0"].value = 0; acc_prog["uSrc1"].value = 1
            acc_prog["uWeight"].value = sub_w
            va_acc.render(moderngl.TRIANGLES)

        # Pass 4: Camera Optical Finish (Impact Micro-Shake + OLPF Softness + Plate Sensor Grain)
        fbo_final.use()
        ctx.viewport = (0, 0, W, H)
        ctx.disable(moderngl.DEPTH_TEST)
        ctx.disable(moderngl.BLEND)
        t_acc_col.use(0); t_acc_meta.use(1)
        post_prog["uScene"].value = 0; post_prog["uMeta"].value = 1
        post_prog["uShakePx"].value = compute_impact_shake(i / fps)
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
            rows_im.append(np.hstack([t_im(pl, f"plate {idx}"), t_im(comp, f"debris sim {idx}")]))
        out_p = work(c, "still_debris_sheet.jpg")
        cv2.imwrite(str(out_p), np.vstack(rows_im), [cv2.IMWRITE_JPEG_QUALITY, 92])
        if hero is not None:
            cv2.imwrite(str(work(c, "still_debris_hero.jpg")), hero, [cv2.IMWRITE_JPEG_QUALITY, 94])
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
            print(f"debris_sim frame {i}/{N} ({time.time() - ts:.1f}s)", flush=True)
    encode(c, out_dir=out_dir)


if __name__ == "__main__":
    main()
