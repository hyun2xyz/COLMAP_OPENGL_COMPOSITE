"""Effect #5: 3D Bioluminescent Morpho Butterfly Boids Flock & Ethereal Silk Ribbon Aura (Native OpenGL 3.3).

    python render_butterfly_silk.py --still 25,75,130,185,225   quick QA -> work/still_butterfly_sheet.jpg
    python render_butterfly_silk.py --render                    full render -> out/Butterfly_Silk_Aura_*.mp4

Real A24 misty winter woodland walk (Pexels 31808102, 240f @ 29.97fps = 8.01s):
- 320 articulated 3D dual-wing flapping Morpho butterflies (GPU-instanced 3D hinge-wing mesh, 11-15Hz realistic
  flutter, 3D flight-tangent orientation, structural iridescent sapphire-cyan <-> pearl-gold wing scales +
  botanical vein alpha silhouette) driven by true 3D Boids flocking + spiral vortex ascent around the walker!
- 3 gossamer translucent 3D anisotropic satin silk ribbons (dynamic 3D triangle-strip meshes with traveling
  cloth flutter waves) physically spiraling BEHIND and IN FRONT OF the walking figure via depth-tested
  MediaPipe silhouette occlusion!
- ~4,500 delicate shimmering wing-dust light motes drifting through the cold morning mist + dual-radius 1:1
  ethereal optical bloom + matched camera film grain. ZERO text/HUD overlays.
"""
import argparse, json, math, shutil, subprocess, sys, time
from pathlib import Path
import numpy as np, cv2, moderngl

here = Path(__file__).resolve().parent
ROOT = here.parents[1]
WORK = ROOT / "examples" / "particle_disintegration" / "piece_forest" / "work"


def find_ffmpeg():
    tl = ROOT / "tools.local.json"
    if tl.exists():
        d = json.loads(tl.read_text(encoding="utf-8"))
        p = d.get("ffmpeg") or (d.get("tools") or {}).get("ffmpeg")
        if p and Path(p).exists():
            return str(p)
    return shutil.which("ffmpeg") or "ffmpeg"


def smoothstep(e0, e1, x):
    t = np.clip((x - e0) / max(e1 - e0, 1e-6), 0.0, 1.0)
    return t * t * (3.0 - 2.0 * t)


def build_butterfly_unit_mesh():
    """Build a crisp articulated 3D dual-wing butterfly mesh in local space:
    - Thorax along +Y (flight forward), wings extending along -X (left, side=-1) and +X (right, side=+1)!
    - Vertex attributes: pos3, uv2 (u in [0,1] root->tip, v in [0,1] hind->fore), side(1f: -1 left, +1 right, 0 body).
    """
    verts = []
    idxs = []

    def add_quad(p0, p1, p2, p3, uv0, uv1, uv2, uv3, side):
        base = len(verts)
        for p, uv in ((p0, uv0), (p1, uv1), (p2, uv2), (p3, uv3)):
            verts.append([p[0], p[1], p[2], uv[0], uv[1], side])
        idxs.extend([base, base + 1, base + 2, base, base + 2, base + 3])

    # Right wing (x: 0 -> +1.0, y: -0.95 -> +0.95) subdivided into 2 curved panels for gentle wing camber!
    add_quad((0.02, -0.95, 0.0), (0.52, -0.95, 0.03), (0.52, 0.95, 0.03), (0.02, 0.95, 0.0),
             (0.0, 0.0), (0.5, 0.0), (0.5, 1.0), (0.0, 1.0), 1.0)
    add_quad((0.52, -0.95, 0.03), (1.05, -0.95, -0.02), (1.05, 0.95, -0.02), (0.52, 0.95, 0.03),
             (0.5, 0.0), (1.0, 0.0), (1.0, 1.0), (0.5, 1.0), 1.0)

    # Left wing (x: 0 -> -1.05, mirrored identically with side = -1.0!)
    add_quad((-0.02, -0.95, 0.0), (-0.52, -0.95, 0.03), (-0.52, 0.95, 0.03), (-0.02, 0.95, 0.0),
             (0.0, 0.0), (0.5, 0.0), (0.5, 1.0), (0.0, 1.0), -1.0)
    add_quad((-0.52, -0.95, 0.03), (-1.05, -0.95, -0.02), (-1.05, 0.95, -0.02), (-0.52, 0.95, 0.03),
             (0.5, 0.0), (1.0, 0.0), (1.0, 1.0), (0.5, 1.0), -1.0)

    # Slender thorax/abdomen body (side = 0.0)
    add_quad((-0.04, -0.50, 0.0), (0.04, -0.50, 0.0), (0.04, 0.55, 0.0), (-0.04, 0.55, 0.0),
             (0.0, 0.0), (1.0, 0.0), (1.0, 1.0), (0.0, 1.0), 0.0)

    return np.array(verts, dtype=np.float32), np.array(idxs, dtype=np.uint32)


def compute_walker_3d_center(bboxes, depths, ro0, R0, f_px, W, H):
    """Fit smooth 3D world center trajectory (pelvis/torso) of the walking figure down the misty trail."""
    N = len(bboxes)
    pts = np.zeros((N, 3), dtype=np.float64)
    for i in range(N):
        cx = 0.5 * (bboxes[i, 0] + bboxes[i, 2])
        cy = 0.52 * bboxes[i, 1] + 0.48 * bboxes[i, 3]
        ray = R0 @ np.array([(cx - 0.5 * W) / f_px, (cy - 0.5 * H) / f_px, 1.0], dtype=np.float64)
        pts[i] = ro0 + ray * depths[i]
    ts = np.arange(N, dtype=np.float64)
    smooth = np.zeros_like(pts)
    for c in range(3):
        med = np.array([np.median(pts[max(0, i - 6):min(N, i + 7), c]) for i in range(N)])
        smooth[:, c] = np.poly1d(np.polyfit(ts, med, deg=2))(ts)
    return smooth


def simulate_butterfly_flock_and_dust(walker_pos, fps, n_bfly=185):
    """Simulate 185 articulated 3D jewel-sapphire & gold Morpho butterflies sweeping across the entire misty woodland
    (opening cleanly around the walker's silhouette + weaving through the gnarled oak branches on the right!).
    """
    N = len(walker_pos)
    dt = 1.0 / fps
    rng = np.random.default_rng(505)

    spawn_f = np.concatenate([
        rng.integers(4, 24, size=18),
        rng.integers(18, 95, size=115),
        rng.integers(75, 165, size=n_bfly - 133),
    ])
    scale    = rng.uniform(0.050, 0.122, n_bfly).astype(np.float64)
    flap_hz  = rng.uniform(9.5, 15.0, n_bfly).astype(np.float64)
    phase    = rng.uniform(0.0, 6.28318, n_bfly).astype(np.float64)
    hue_seed = rng.uniform(0.0, 1.0, n_bfly).astype(np.float64)
    # Wider graceful orbit (0.58..2.45m) so butterflies frame the walker cleanly instead of crowding his chest!
    orb_r    = rng.uniform(0.58, 2.45, n_bfly).astype(np.float64)
    orb_spd  = rng.uniform(0.95, 2.15, n_bfly).astype(np.float64) * rng.choice([-1.0, 1.0], n_bfly)
    orb_ang0 = rng.uniform(0.0, 6.28318, n_bfly).astype(np.float64)
    climb_v  = rng.uniform(0.16, 0.62, n_bfly).astype(np.float64)
    # Sweeping rightward forest breeze carrying the flock gracefully across the gnarled oak canopy!
    drift_x  = rng.uniform(-0.45, 2.65, n_bfly).astype(np.float64)
    is_hero  = np.arange(n_bfly) < 34

    # Initial spawn positions tucked organically behind/around his coat & shoulders
    pos = np.zeros((n_bfly, 3), dtype=np.float64)
    vel = np.zeros((n_bfly, 3), dtype=np.float64)
    for k in range(n_bfly):
        fi = min(int(spawn_f[k]), N - 1)
        wc = walker_pos[fi]
        pos[k] = wc + np.array([rng.normal(0.0, 0.28), rng.uniform(0.06, 0.45), rng.uniform(-0.65, 0.55)])
        vel[k] = np.array([rng.normal(0.15, 0.35), rng.normal(0.0, 0.25), rng.uniform(0.20, 0.70)])

    bfly_frames = []
    # Dust pool (ring buffer of 4,200 delicate glowing wing motes!)
    M_dust = 4200
    d_pos = np.zeros((M_dust, 3), dtype=np.float64)
    d_vel = np.zeros((M_dust, 3), dtype=np.float64)
    d_rgb = np.zeros((M_dust, 3), dtype=np.float32)
    d_age = np.full(M_dust, 99.0, dtype=np.float64)
    d_life = rng.uniform(1.2, 2.8, M_dust)
    d_size = rng.uniform(0.004, 0.012, M_dust).astype(np.float32)
    d_cursor = 0
    dust_frames = []

    for i in range(N):
        t = i * dt
        wc = walker_pos[i]
        alive = (spawn_f <= i)
        idx = np.where(alive)[0]

        if len(idx) > 0:
            age = (i - spawn_f[idx]).astype(np.float64) * dt
            # Target 3D spiral-orbit around the walking figure + gradual Boids canopy drift!
            ang = orb_ang0[idx] + orb_spd[idx] * age
            # Spiral starts tight around body (passing BOTH behind +y and in front -y!), then widens majestically!
            rad = orb_r[idx] * (0.42 + 0.58 * np.clip(age / 2.2, 0.0, 1.0)) + 0.35 * np.maximum(age - 2.6, 0.0)
            # Hero close-pass butterflies swoop gently closer to camera (-Y) for jaw-dropping 3D depth parallax!
            y_bias = np.where(is_hero[idx], -1.35 * np.sin(np.clip(age / 3.8, 0.0, 1.0) * math.pi), 0.15 * age)
            tgt = np.stack([
                wc[0] + rad * np.cos(ang) + drift_x[idx] * np.maximum(age - 1.2, 0.0) * 0.42,
                wc[1] + rad * 0.78 * np.sin(ang) + y_bias,
                np.clip(wc[2] - 0.35 + climb_v[idx] * age + 0.35 * np.sin(ang * 1.5 + phase[idx]), 0.18, 3.65),
            ], axis=1)

            # Lightweight Boids alignment + curl flutter jitter (smooth organic butterfly flight bobbing!)
            flutter_bob = np.stack([
                0.28 * np.sin(t * 3.8 + phase[idx]),
                0.22 * np.cos(t * 3.2 + phase[idx] * 1.3),
                0.42 * np.sin(t * flap_hz[idx] * 0.5 + phase[idx]),
            ], axis=1)
            desired_v = (tgt - pos[idx]) * 1.95 + flutter_bob
            # Cap realistic butterfly speed (0.4..2.4 m/s)
            spd = np.linalg.norm(desired_v, axis=1, keepdims=True)
            desired_v = desired_v / np.maximum(spd, 1e-4) * np.clip(spd, 0.35, 2.35)

            # Smooth steering integration
            moving = (spawn_f[idx] < i)
            if np.any(moving):
                mi = idx[moving]
                vel[mi] += (desired_v[moving] - vel[mi]) * (dt * 4.2)
                pos[mi] += vel[mi] * dt
                pos[mi, 1] = np.clip(pos[mi, 1], 1.55, 13.5)
                pos[mi, 2] = np.maximum(pos[mi, 2], 0.14)

            fwd = vel[idx] / (np.linalg.norm(vel[idx], axis=1, keepdims=True) + 1e-5)
            # Wing flap angle in radians (-0.35 rad up-glide to +1.28 rad full upstroke!)
            flap = 0.25 + 1.05 * np.sin(2.0 * math.pi * flap_hz[idx] * t + phase[idx])
            # Smooth birth scale-in (0.3s) & gentle dispersal fade near end of clip
            a_in = np.clip(age / 0.32, 0.0, 1.0)
            a_out = np.clip(1.0 - (i - 224) / 16.0, 0.35, 1.0) if i > 224 else np.ones(len(idx))
            alpha = (a_in * a_out).astype(np.float32)
            glow = (0.75 + 0.35 * np.sin(t * 4.0 + phase[idx])).astype(np.float32)

            batch = np.column_stack([
                pos[idx].astype(np.float32),
                fwd.astype(np.float32),
                (scale[idx] * (0.45 + 0.55 * a_in)).astype(np.float32),
                flap.astype(np.float32),
                hue_seed[idx].astype(np.float32),
                alpha,
                glow,
                (phase[idx] + t * 2.0).astype(np.float32),
            ]).astype(np.float32)
            bfly_frames.append(batch)

            # Emit ~34 shimmering wing-dust motes per frame directly from active flapping butterflies!
            k_emit = min(38, len(idx))
            emit_sel = rng.choice(idx, size=k_emit, replace=False)
            for ek in emit_sel:
                d_pos[d_cursor] = pos[ek] + rng.normal(0.0, 0.035, 3)
                d_vel[d_cursor] = vel[ek] * 0.25 + np.array([rng.normal(0.0, 0.08), rng.normal(0.0, 0.06), rng.uniform(-0.08, 0.14)])
                if hue_seed[ek] < 0.72:
                    d_rgb[d_cursor] = np.array([0.35, 0.88, 1.00], dtype=np.float32)
                else:
                    d_rgb[d_cursor] = np.array([1.00, 0.82, 0.42], dtype=np.float32)
                d_age[d_cursor] = 0.0
                d_cursor = (d_cursor + 1) % M_dust
        else:
            bfly_frames.append(np.zeros((0, 12), dtype=np.float32))

        # Step shimmering dust motes
        d_alive = d_age < d_life
        if np.any(d_alive):
            d_age[d_alive] += dt
            d_pos[d_alive] += d_vel[d_alive] * dt
            d_vel[d_alive] *= 0.985
            da = np.where(d_alive)[0]
            vis = np.clip(1.0 - d_age[da] / d_life[da], 0.0, 1.0) * np.clip(d_age[da] / 0.15, 0.0, 1.0)
            twk = 0.55 + 0.45 * np.sin(t * 9.0 + da * 1.7)
            dust_frames.append(np.column_stack([
                d_pos[da].astype(np.float32),
                d_rgb[da],
                d_size[da],
                (vis * twk).astype(np.float32),
            ]).astype(np.float32))
        else:
            dust_frames.append(np.zeros((0, 8), dtype=np.float32))

    return bfly_frames, dust_frames


def build_silk_ribbon_geometry(walker_pos, i, fps, n_seg=150, n_cross=5):
    """Build 3 flowing 3D Vellum-style translucent silk ribbons spiraling gracefully BEHIND and IN FRONT OF
    the walking figure with traveling cloth flutter waves!
    Returns interleaved float32 vertex buffer (pos3, nrm3, uv2, ribbon_id1, alpha1) and uint32 index buffer.
    """
    t = i / fps
    wc = walker_pos[i]
    # Smooth global emergence & lingering glide across clip
    env = float(smoothstep(14.0, 48.0, float(i))) * (1.0 - 0.55 * float(smoothstep(218.0, 239.0, float(i))))
    if env < 0.005:
        return np.zeros((0, 10), dtype=np.float32), np.zeros((0,), dtype=np.uint32)

    specs = [
        # 2 sweeping couture translucent silk ribbons wrapping gracefully around walker & trailing far into oak branches!
        (0.64, 1.30, -0.72, 1.12, 0.115, 0.2,  1.25, 0.0),
        (0.95, 1.05, -0.35, 1.48, 0.092, 2.8, -1.05, 1.0),
    ]

    all_v = []
    all_idx = []
    v_off = 0

    for rad_b, turns, z_lo, z_hi, width, ph, spd, rid in specs:
        s = np.linspace(0.0, 1.0, n_seg, dtype=np.float64)
        grow = float(smoothstep(16.0 + rid * 14.0, 92.0 + rid * 16.0, float(i)))
        s_eff = s * grow
        ang = ph + spd * t + s_eff * turns * 2.0 * math.pi
        # Graceful sweeping couture windward tail curving right toward the gnarled frosted oak branches!
        flare = 1.15 * (s_eff ** 2.2)
        rad = rad_b * (0.86 + 0.14 * np.sin(s_eff * 5.0 - t * 2.5 + ph)) + flare
        wave_z = 0.13 * np.sin(s_eff * 9.0 - t * 4.0 + ph)
        wave_r = 0.09 * np.cos(s_eff * 7.5 - t * 3.4 + ph)

        cx = wc[0] + (rad + wave_r) * np.cos(ang) + 1.55 * (s_eff ** 2.1)
        cy = wc[1] + (rad + wave_r) * 0.85 * np.sin(ang)
        cz = np.clip(wc[2] + z_lo + (z_hi - z_lo) * s_eff + wave_z, 0.16, 3.50)
        spine = np.stack([cx, cy, cz], axis=1)

        # Smooth Frenet tangent + silky twisting binormal for natural cloth ribbon curl!
        tang = np.gradient(spine, axis=0)
        tang /= (np.linalg.norm(tang, axis=1, keepdims=True) + 1e-6)
        up_ref = np.array([0.0, 0.0, 1.0], dtype=np.float64)
        side0 = np.cross(tang, up_ref)
        side0 /= (np.linalg.norm(side0, axis=1, keepdims=True) + 1e-6)
        nrm0 = np.cross(side0, tang)
        nrm0 /= (np.linalg.norm(nrm0, axis=1, keepdims=True) + 1e-6)

        twist = 0.75 * np.sin(s_eff * 7.5 - t * 3.1 + ph) + 0.35 * s_eff * math.pi
        ct, st = np.cos(twist)[:, None], np.sin(twist)[:, None]
        b_dir = side0 * ct + nrm0 * st
        n_dir = -side0 * st + nrm0 * ct

        # Taper ribbon gracefully to silky pointed ends (u=0 and u=1)
        w_prof = width * np.sin(np.clip(s, 0.0, 1.0) * math.pi) ** 0.65 * grow
        # Smooth visibility fade at ends
        a_prof = (np.sin(np.clip(s, 0.0, 1.0) * math.pi) ** 0.55) * env * grow

        vs = np.linspace(-1.0, 1.0, n_cross, dtype=np.float64)
        # Build (n_seg, n_cross) ribbon surface with gentle cross-sectional S-curl cloth camber!
        pts = np.zeros((n_seg, n_cross, 3), dtype=np.float32)
        nrms = np.zeros((n_seg, n_cross, 3), dtype=np.float32)
        uvs = np.zeros((n_seg, n_cross, 2), dtype=np.float32)
        rids = np.full((n_seg, n_cross, 1), rid, dtype=np.float32)
        alps = np.zeros((n_seg, n_cross, 1), dtype=np.float32)

        for ci, vv in enumerate(vs):
            camber = 0.22 * (vv * vv - 0.33) * w_prof
            pts[:, ci, :] = (spine + b_dir * (vv * w_prof[:, None]) + n_dir * camber[:, None]).astype(np.float32)
            # Tilt normal slightly across curved silk width for luxurious anisotropic highlights!
            nn = n_dir - b_dir * (0.45 * vv)
            nn /= (np.linalg.norm(nn, axis=1, keepdims=True) + 1e-6)
            nrms[:, ci, :] = nn.astype(np.float32)
            uvs[:, ci, 0] = s.astype(np.float32)
            uvs[:, ci, 1] = float(0.5 * (vv + 1.0))
            alps[:, ci, 0] = a_prof.astype(np.float32)

        v_block = np.concatenate([pts, nrms, uvs, rids, alps], axis=2).reshape(-1, 10)
        all_v.append(v_block)

        # Triangle indices
        tris = []
        for si in range(n_seg - 1):
            for ci in range(n_cross - 1):
                i00 = v_off + si * n_cross + ci
                i10 = v_off + (si + 1) * n_cross + ci
                i11 = v_off + (si + 1) * n_cross + (ci + 1)
                i01 = v_off + si * n_cross + (ci + 1)
                tris.extend([i00, i10, i11, i00, i11, i01])
        all_idx.append(np.array(tris, dtype=np.uint32))
        v_off += len(v_block)

    return np.concatenate(all_v, axis=0), np.concatenate(all_idx, axis=0)


# ==============================================================================
#  SHADERS: SILK RIBBON (ANISOTROPIC SATIN), 3D FLAPPING MORPHO BUTTERFLIES,
#           SHIMMERING WING-DUST MOTES, DUAL-RADIUS 1:1 ETHEREAL BLOOM
# ==============================================================================

VS_FSQ = "#version 330\nin vec2 p; void main(){ gl_Position = vec4(p, 0.0, 1.0); }"

PLATE_FS = """#version 330
uniform sampler2D uPlate;
uniform vec2 uRes;
layout(location=0) out vec4 oScene;
layout(location=1) out vec4 oEmiss;
void main() {
  vec2 uv = vec2(gl_FragCoord.x / uRes.x, 1.0 - gl_FragCoord.y / uRes.y);
  vec3 lin = pow(max(texture(uPlate, clamp(uv, vec2(0.001), vec2(0.999))).rgb, vec3(0.0005)), vec3(2.2));
  oScene = vec4(lin, 1.0);
  oEmiss = vec4(0.0);
}
"""

SILK_VS = """#version 330
in vec3 inPos;
in vec3 inNrm;
in vec2 inUv;
in float inRid;
in float inAlpha;

uniform mat4 uProj, uView;

out vec3 vWorldPos;
out vec3 vNrm;
out vec2 vUv;
out float vRid;
out float vAlpha;
out float vDepth;

void main() {
  vWorldPos = inPos;
  vNrm      = normalize(inNrm);
  vUv       = inUv;
  vRid      = inRid;
  vAlpha    = inAlpha;
  vec4 vp = uView * vec4(inPos, 1.0);
  vDepth = -vp.z;
  gl_Position = uProj * vp;
}
"""

SILK_FS = """#version 330
in vec3 vWorldPos;
in vec3 vNrm;
in vec2 vUv;
in float vRid;
in float vAlpha;
in float vDepth;

uniform sampler2D uPersonMask;
uniform vec2 uRes;
uniform vec3 uRo;
uniform float uPersonDepth, uTime;

layout(location=0) out vec4 oScene;
layout(location=1) out vec4 oEmiss;

void main() {
  if (vAlpha < 0.01) discard;
  /* True 3D MediaPipe depth occlusion: ribbon behind walker tucks cleanly BEHIND his coat, front sweeps IN FRONT! */
  vec2 uvScr = vec2(gl_FragCoord.x / uRes.x, 1.0 - gl_FragCoord.y / uRes.y);
  float pMask = texture(uPersonMask, clamp(uvScr, vec2(0.001), vec2(0.999))).r;
  if (vDepth > uPersonDepth + 0.06 && pMask > 0.45) discard;
  float occ = (vDepth > uPersonDepth + 0.06) ? (1.0 - smoothstep(0.14, 0.52, pMask)) : 1.0;

  vec3 N = normalize(vNrm);
  vec3 V = normalize(uRo - vWorldPos);
  if (dot(N, V) < 0.0) N = -N;
  float NdotV = max(dot(N, V), 0.0);

  /* Luxury iridescent satin silk palette: pearl-cyan / warm gold-champagne / ice-aurora */
  vec3 cBase = (vRid < 0.5) ? vec3(0.24, 0.78, 1.00) :
               ((vRid < 1.5) ? vec3(1.00, 0.80, 0.42) : vec3(0.62, 0.86, 1.00));
  vec3 cIrr  = (vRid < 0.5) ? vec3(0.68, 0.94, 1.00) :
               ((vRid < 1.5) ? vec3(1.00, 0.94, 0.76) : vec3(0.84, 0.72, 1.00));

  /* Anisotropic silk sheen + fine luminous wave shimmer along cloth weave! */
  vec3 Lback = normalize(vec3(0.10, 0.85, 0.52));  /* misty glowing woodland backlight! */
  vec3 H = normalize(Lback + V);
  float sheen = pow(max(dot(N, H), 0.0), 18.0) * 1.45 + pow(1.0 - NdotV, 2.4) * 1.20;
  float weave = 0.86 + 0.14 * sin(vUv.x * 160.0 - uTime * 3.5 + vUv.y * 18.0);

  /* Gossamer sheer center + luminous thicker satin hem along both edges (vUv.y near 0 or 1)! */
  float hem = pow(abs(vUv.y * 2.0 - 1.0), 2.2);
  float silkAlpha = clamp((0.20 + 0.48 * hem + 0.34 * pow(1.0 - NdotV, 2.0)) * vAlpha * occ, 0.0, 0.86);

  vec3 silkLin = mix(cBase, cIrr, clamp(pow(1.0 - NdotV, 1.5) + hem * 0.4, 0.0, 1.0)) * (0.65 + 0.85 * sheen) * weave;
  vec3 emiss   = silkLin * (0.42 + 0.75 * hem + 0.45 * sheen) * silkAlpha;

  oScene = vec4(silkLin * 1.25, silkAlpha);
  oEmiss = vec4(emiss, silkAlpha);
}
"""

BFLY_VS = """#version 330
in vec3 inPos;     /* unit butterfly vertex */
in vec2 inUv;      /* wing UV */
in float inSide;   /* -1 left wing, +1 right wing, 0 body */

/* Per-instance 3D flight state */
in vec3 iPos;
in vec3 iFwd;
in float iScale;
in float iFlap;    /* live wing hinge angle in radians! */
in float iHue;
in float iAlpha;
in float iGlow;
in float iPhase;

uniform mat4 uProj, uView;

out vec2 vUv;
out float vSide;
out float vHue;
out float vAlpha;
out float vGlow;
out float vDepth;
out vec3 vNrm;
out vec3 vWorldPos;

void main() {
  /* 1. Articulate left/right wings around thorax Y-spine by live hinge angle iFlap! */
  vec3 p = inPos;
  vec3 nLocal = vec3(0.0, 0.0, 1.0);
  if (abs(inSide) > 0.5) {
    float a = iFlap * inSide; /* symmetric V-flap upward/downward */
    float ca = cos(a), sa = sin(a);
    p = vec3(inPos.x * ca - inPos.z * sa, inPos.y, inPos.x * sa + inPos.z * ca);
    nLocal = vec3(-sa, 0.0, ca);
  }
  p *= iScale;

  /* 2. Build true 3D orthonormal flight frame aligned with butterfly's live velocity vector iFwd! */
  vec3 fwd = normalize(iFwd + vec3(0.0, 0.001, 0.0));
  vec3 upRef = abs(fwd.z) < 0.92 ? vec3(0.0, 0.0, 1.0) : vec3(1.0, 0.0, 0.0);
  vec3 right = normalize(cross(fwd, upRef));
  vec3 up = cross(right, fwd);
  mat3 basis = mat3(right, fwd, up);

  vec3 wp = iPos + basis * p;
  vWorldPos = wp;
  vNrm      = normalize(basis * nLocal);
  vUv       = inUv;
  vSide     = inSide;
  vHue      = iHue;
  vAlpha    = iAlpha;
  vGlow     = iGlow;

  vec4 vp = uView * vec4(wp, 1.0);
  vDepth = -vp.z;
  gl_Position = uProj * vp;
}
"""

BFLY_FS = """#version 330
in vec2 vUv;
in float vSide;
in float vHue;
in float vAlpha;
in float vGlow;
in float vDepth;
in vec3 vNrm;
in vec3 vWorldPos;

uniform sampler2D uPersonMask;
uniform vec2 uRes;
uniform vec3 uRo;
uniform float uPersonDepth;

layout(location=0) out vec4 oScene;
layout(location=1) out vec4 oEmiss;

void main() {
  if (vAlpha < 0.02 || vDepth < 0.30) discard;

  /* Depth-tested MediaPipe occlusion -> butterflies physically orbit BEHIND & IN FRONT OF the walking man! */
  vec2 uvScr = vec2(gl_FragCoord.x / uRes.x, 1.0 - gl_FragCoord.y / uRes.y);
  float pMask = texture(uPersonMask, clamp(uvScr, vec2(0.001), vec2(0.999))).r;
  if (vDepth > uPersonDepth + 0.06 && pMask > 0.45) discard;
  float occ = (vDepth > uPersonDepth + 0.06) ? (1.0 - smoothstep(0.14, 0.52, pMask)) : 1.0;

  if (abs(vSide) < 0.5) {
    /* Slender dark velvety butterfly thorax */
    float r = length((vUv - vec2(0.5)) * vec2(2.0, 1.0));
    if (r > 0.95) discard;
    oScene = vec4(vec3(0.02, 0.03, 0.05), vAlpha * occ);
    oEmiss = vec4(0.0);
    return;
  }

  /* Procedural Botanical Swallowtail / Morpho Butterfly Wing Silhouette (forewing + scalloped hindwing!) */
  float u = vUv.x; /* 0=body root -> 1=outer wingtip */
  float v = vUv.y; /* 0=hindwing bottom -> 1=forewing top */
  vec2 pF = vec2((u - 0.46) / 0.54, (v - 0.68) / 0.30);
  /* Swept pointed Morpho forewing lobe */
  pF.y -= 0.22 * u;
  float dFore = length(pF * vec2(1.0, 1.18)) - 0.88;
  /* Rounded scalloped hindwing lobe */
  vec2 pH = vec2((u - 0.35) / 0.42, (v - 0.28) / 0.27);
  float scallop = 0.04 * sin(atan(pH.y, pH.x) * 7.0);
  float dHind = length(pH) - (0.86 + scallop);
  float dWing = min(dFore, dHind);
  float wingMask = smoothstep(0.04, -0.06, dWing);
  if (wingMask < 0.04) discard;

  vec3 N = normalize(vNrm);
  vec3 V = normalize(uRo - vWorldPos);
  if (dot(N, V) < 0.0) N = -N;
  float NdotV = max(dot(N, V), 0.0);

  /* Deep jewel-tone Structural Iridescence -> photorealistic sapphire Morpho & warm amber gold against silver mist! */
  vec3 cCore, cRim;
  if (vHue < 0.72) {
    cCore = mix(vec3(0.015, 0.12, 0.68), vec3(0.04, 0.52, 0.96), pow(NdotV, 0.80));
    cRim  = vec3(0.45, 0.88, 1.00);
  } else {
    cCore = mix(vec3(0.72, 0.28, 0.03), vec3(0.98, 0.72, 0.22), pow(NdotV, 0.80));
    cRim  = vec3(1.00, 0.92, 0.65);
  }

  /* Crisp dark botanical wing veins + velvety obsidian margin -> zero washed-out neon blobs! */
  float radialAng = atan(v - 0.50, u + 0.08);
  float veins = smoothstep(0.80, 0.97, abs(sin(radialAng * 13.0 + u * 2.2)));
  float borderDark = smoothstep(-0.24, -0.02, dWing);
  float marginSpots = smoothstep(0.82, 0.96, sin(v * 38.0)) * smoothstep(-0.14, -0.04, dWing) * smoothstep(0.01, -0.05, dWing);

  vec3 wingLin = mix(cCore * (1.0 - 0.62 * veins), vec3(0.008, 0.012, 0.022), borderDark * 0.92);
  wingLin += cRim * marginSpots * 1.25;
  float spec = pow(1.0 - NdotV, 2.4) * 0.75 + 0.28;
  vec3 colOut = wingLin * (0.90 + 0.65 * spec * vGlow);
  vec3 emOut  = cCore * (1.0 - borderDark * 0.85) * (0.36 + 0.42 * vGlow) + cRim * marginSpots * 0.7;

  float aOut = wingMask * vAlpha * occ;
  oScene = vec4(colOut, aOut);
  oEmiss = vec4(emOut * aOut, aOut);
}
"""

DUST_VS = """#version 330
in vec2 inQuad;
in vec3 iPos;
in vec3 iRgb;
in float iSize;
in float iAlpha;

uniform vec2 uRes;
uniform float uF;
uniform vec3 uRo;
uniform mat3 uCam;

out vec2 vLocal;
out vec3 vRgb;
out float vAlpha;
out float vDepth;

void main() {
  vec3 q = transpose(uCam) * (iPos - uRo);
  if (q.z < 0.35) { gl_Position = vec4(-10.0); return; }
  vec2 px0 = uF * q.xy / q.z + 0.5 * uRes;
  float rPx = clamp(uF * iSize / q.z, 1.2, 6.5);
  vec2 px = px0 + inQuad * rPx;
  vec2 ndc = vec2(px.x / uRes.x * 2.0 - 1.0, (1.0 - px.y / uRes.y) * 2.0 - 1.0);
  gl_Position = vec4(ndc, clamp((q.z - 0.35) / 25.0, 0.0, 0.999) * 2.0 - 1.0, 1.0);
  vLocal = inQuad; vRgb = iRgb; vAlpha = iAlpha; vDepth = q.z;
}
"""

DUST_FS = """#version 330
in vec2 vLocal;
in vec3 vRgb;
in float vAlpha;
in float vDepth;

uniform sampler2D uPersonMask;
uniform vec2 uRes;
uniform float uPersonDepth;

layout(location=0) out vec4 oScene;
layout(location=1) out vec4 oEmiss;

void main() {
  vec2 uvScr = vec2(gl_FragCoord.x / uRes.x, 1.0 - gl_FragCoord.y / uRes.y);
  if (vDepth > uPersonDepth + 0.06 && texture(uPersonMask, uvScr).r > 0.52) discard;
  float r2 = dot(vLocal, vLocal);
  if (r2 > 1.0) discard;
  float g = exp(-r2 * 4.6) * vAlpha;
  vec3 em = vRgb * g * 2.2;
  oScene = vec4(em * 0.65, 1.0);
  oEmiss = vec4(em * 0.95, 1.0);
}
"""

BLUR_FS = """#version 330
uniform sampler2D uTex;
uniform vec2 uOutRes, uDir;
out vec4 oCol;
void main() {
  vec2 uv = gl_FragCoord.xy / uOutRes;
  const float w[5] = float[](0.227027, 0.1945946, 0.1216216, 0.054054, 0.016216);
  vec3 sum = texture(uTex, uv).rgb * w[0];
  for (int k = 1; k < 5; k++) {
    vec2 off = uDir * float(k);
    sum += texture(uTex, clamp(uv + off, vec2(0.001), vec2(0.999))).rgb * w[k];
    sum += texture(uTex, clamp(uv - off, vec2(0.001), vec2(0.999))).rgb * w[k];
  }
  oCol = vec4(sum, 1.0);
}
"""

COMP_FS = """#version 330
uniform sampler2D uScene, uBloom1, uBloom2;
uniform vec2 uRes;
uniform float uBlackFloor, uGrainAmt, uFrame;
out vec4 oCol;
float hash21(vec2 p){ vec3 q=fract(vec3(p.xyx)*0.1031); q+=dot(q,q.yzx+33.33); return fract((q.x+q.y)*q.z); }

void main() {
  vec2 uv = gl_FragCoord.xy / uRes;
  vec3 sc = texture(uScene, uv).rgb;
  vec3 b1 = texture(uBloom1, uv).rgb;
  vec3 b2 = texture(uBloom2, uv).rgb;
  vec3 bloom = b1 * 0.28 + b2 * 0.22;

  /* Preserve 100% untouched misty woodland exposure; add dreamy ethereal bioluminescent bloom cleanly! */
  vec3 baseC = min(sc, vec3(1.0));
  vec3 hdr = max(sc - vec3(1.0), vec3(0.0)) + bloom;
  vec3 emSoft = hdr / (1.0 + hdr * 0.30);
  vec3 totalLin = baseC + emSoft * (1.0 - baseC * 0.38);

  vec3 col = pow(clamp(totalLin, 0.0, 1.0), vec3(1.0 / 2.2));
  float act = clamp(length(bloom) * 2.2 + length(hdr) * 1.8, 0.0, 1.0);
  if (act > 0.01) {
    vec3 graded = max(col, vec3(uBlackFloor));
    vec2 p1 = gl_FragCoord.xy + vec2(uFrame * 17.3, uFrame * 31.7);
    vec2 p2 = floor(gl_FragCoord.xy * 0.5) + vec2(uFrame * 11.1, uFrame * 23.9);
    float g = (hash21(p1)+hash21(p1+7.13)-1.0)*0.62 + (hash21(p2)+hash21(p2+5.71)-1.0)*0.38;
    float lum = dot(graded, vec3(0.299, 0.587, 0.114));
    graded += vec3(g) * (uGrainAmt * (1.10 - 0.55 * smoothstep(0.55, 0.98, lum)));
    col = mix(col, graded, act);
  }
  oCol = vec4(clamp(col, 0.0, 1.0), 1.0);
}
"""


def mat4_perspective_from_opencv(f_px, W, H, near=0.1, far=60.0):
    P = np.zeros((4, 4), dtype=np.float32)
    P[0, 0] = 2.0 * f_px / W; P[1, 1] = 2.0 * f_px / H
    P[2, 2] = -(far + near) / (far - near); P[2, 3] = -2.0 * far * near / (far - near); P[3, 2] = -1.0
    return P


def mat4_view_from_colmap(ro, R_c2w):
    flip = np.diag([1.0, -1.0, -1.0]); R_w2gl = flip @ R_c2w.T
    V = np.eye(4, dtype=np.float32); V[:3, :3] = R_w2gl; V[:3, 3] = -R_w2gl @ ro
    return V


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--still", default=None)
    ap.add_argument("--render", action="store_true")
    args = ap.parse_args()

    meta = json.loads((WORK / "meta.json").read_text())
    S    = json.loads((WORK / "scene.json").read_text())
    T    = np.load(WORK / "track.npz")
    W, H, N = int(meta["W"]), int(meta["H"]), int(meta["N"])
    fps = float(meta["fps"])
    f_px = float(S["f"])
    ro0 = np.asarray(S["cam"][0], dtype=np.float64)
    R0  = np.asarray(S["rot"][0], dtype=np.float64)
    depths = np.asarray(T["depth"], dtype=np.float64)
    bboxes = np.asarray(T["bboxes"], dtype=np.float64)

    walker_pos = compute_walker_3d_center(bboxes, depths, ro0, R0, f_px, W, H)
    bfly_frames, dust_frames = simulate_butterfly_flock_and_dust(walker_pos, fps, n_bfly=185)
    print(f"Butterfly & Silk Aura: {N}f @ {fps:.2f}fps | 185 articulated 3D jewel butterflies + 2 couture silk ribbons + pixie dust")

    pl0 = cv2.imread(str(WORK / "frames" / "0000.jpg"))
    gray0 = cv2.cvtColor(pl0, cv2.COLOR_BGR2GRAY).astype(np.float32) / 255.0
    black_floor = float(np.clip(np.percentile(gray0, 1.0) * 0.85, 0.008, 0.055))
    hp0 = gray0 - cv2.GaussianBlur(gray0, (0, 0), 1.2)
    grain_amt = float(np.clip(float(np.median(np.abs(hp0 - np.median(hp0)))) * 1.4826 * 1.15, 0.008, 0.025))

    ctx = moderngl.create_standalone_context(require=330)
    plate_prog = ctx.program(vertex_shader=VS_FSQ,  fragment_shader=PLATE_FS)
    silk_prog  = ctx.program(vertex_shader=SILK_VS, fragment_shader=SILK_FS)
    bfly_prog  = ctx.program(vertex_shader=BFLY_VS, fragment_shader=BFLY_FS)
    dust_prog  = ctx.program(vertex_shader=DUST_VS, fragment_shader=DUST_FS)
    blur_prog  = ctx.program(vertex_shader=VS_FSQ,  fragment_shader=BLUR_FS)
    comp_prog  = ctx.program(vertex_shader=VS_FSQ,  fragment_shader=COMP_FS)

    fsq = ctx.buffer(np.array([-1, -1, 1, -1, -1, 1, -1, 1, 1, -1, 1, 1], dtype=np.float32).tobytes())
    va_plate = ctx.vertex_array(plate_prog, [(fsq, "2f", "p")])
    va_blur  = ctx.vertex_array(blur_prog,  [(fsq, "2f", "p")])
    va_comp  = ctx.vertex_array(comp_prog,  [(fsq, "2f", "p")])

    # 3D Articulated Butterfly mesh + instance buffer
    bf_v, bf_i = build_butterfly_unit_mesh()
    bf_vbo = ctx.buffer(bf_v.tobytes())
    bf_ibo = ctx.buffer(bf_i.tobytes())
    bf_inst = ctx.buffer(reserve=400 * 12 * 4)
    va_bfly = ctx.vertex_array(
        bfly_prog,
        [(bf_vbo, "3f 2f 1f", "inPos", "inUv", "inSide"),
         (bf_inst, "3f 3f 1f 1f 1f 1f 1f 4x/i", "iPos", "iFwd", "iScale", "iFlap", "iHue", "iAlpha", "iGlow")],
        index_buffer=bf_ibo,
    )

    # Dynamic 3D Silk Ribbon mesh buffers
    silk_vbo = ctx.buffer(reserve=3 * 150 * 5 * 10 * 4)
    silk_ibo = ctx.buffer(reserve=3 * 149 * 4 * 6 * 4)
    va_silk = ctx.vertex_array(silk_prog, [(silk_vbo, "3f 3f 2f 1f 1f", "inPos", "inNrm", "inUv", "inRid", "inAlpha")], index_buffer=silk_ibo)

    # Shimmering Wing-Dust Motes
    quad_vbo = ctx.buffer(np.array([-1, -1, 1, -1, -1, 1, -1, 1, 1, -1, 1, 1], dtype=np.float32).tobytes())
    dust_inst = ctx.buffer(reserve=4500 * 8 * 4)
    va_dust = ctx.vertex_array(dust_prog, [(quad_vbo, "2f", "inQuad"), (dust_inst, "3f 3f 1f 1f/i", "iPos", "iRgb", "iSize", "iAlpha")])

    t_scene = ctx.texture((W, H), 4, dtype="f4")
    t_emiss = ctx.texture((W, H), 4, dtype="f4")
    rbo_dep = ctx.depth_renderbuffer((W, H))
    fbo_scene = ctx.framebuffer([t_scene, t_emiss], depth_attachment=rbo_dep)

    hw, hh = W // 2, H // 2
    qw, qh = W // 4, H // 4
    t_b1a = ctx.texture((hw, hh), 4, dtype="f4"); t_b1b = ctx.texture((hw, hh), 4, dtype="f4")
    t_b2a = ctx.texture((qw, qh), 4, dtype="f4"); t_b2b = ctx.texture((qw, qh), 4, dtype="f4")
    for t in (t_scene, t_emiss, t_b1a, t_b1b, t_b2a, t_b2b):
        t.filter = (moderngl.LINEAR, moderngl.LINEAR); t.repeat_x = t.repeat_y = False
    fb_b1a, fb_b1b = ctx.framebuffer([t_b1a]), ctx.framebuffer([t_b1b])
    fb_b2a, fb_b2b = ctx.framebuffer([t_b2a]), ctx.framebuffer([t_b2b])

    t_final = ctx.texture((W, H), 4, dtype="f4")
    fbo_final = ctx.framebuffer([t_final])
    plate_t = ctx.texture((W, H), 3); plate_t.filter = (moderngl.LINEAR, moderngl.LINEAR); plate_t.repeat_x = plate_t.repeat_y = False
    mask_t  = ctx.texture((W, H), 1, dtype="f4"); mask_t.filter = (moderngl.LINEAR, moderngl.LINEAR); mask_t.repeat_x = mask_t.repeat_y = False

    proj_m = mat4_perspective_from_opencv(f_px, W, H)
    view_m = mat4_view_from_colmap(ro0, R0)
    Rt = tuple(R0.T.flatten().astype(np.float32))

    plate_prog["uRes"].value = (float(W), float(H))
    silk_prog["uProj"].write(proj_m.T.tobytes()); silk_prog["uView"].write(view_m.T.tobytes())
    silk_prog["uRes"].value = (float(W), float(H)); silk_prog["uRo"].value = tuple(map(float, ro0))
    bfly_prog["uProj"].write(proj_m.T.tobytes()); bfly_prog["uView"].write(view_m.T.tobytes())
    bfly_prog["uRes"].value = (float(W), float(H)); bfly_prog["uRo"].value = tuple(map(float, ro0))
    dust_prog["uRes"].value = (float(W), float(H)); dust_prog["uF"].value = f_px
    dust_prog["uRo"].value = tuple(map(float, ro0)); dust_prog["uCam"].value = Rt
    comp_prog["uRes"].value = (float(W), float(H))
    comp_prog["uBlackFloor"].value = black_floor; comp_prog["uGrainAmt"].value = grain_amt

    def render_one(i):
        pl = cv2.imread(str(WORK / "frames" / f"{i:04d}.jpg"))
        m_u8 = cv2.imread(str(WORK / "mask" / f"{i:04d}.png"), cv2.IMREAD_GRAYSCALE)
        m_f = m_u8.astype(np.float32) / 255.0 if m_u8 is not None else np.zeros((H, W), np.float32)
        plate_t.write(pl[:, :, ::-1].tobytes())
        mask_t.write(m_f.astype(np.float32).tobytes())
        p_dep = float(depths[i])

        # Pass 1: Live Misty Woodland Plate
        fbo_scene.use(); ctx.viewport = (0, 0, W, H); ctx.disable(moderngl.DEPTH_TEST); ctx.disable(moderngl.BLEND)
        fbo_scene.clear(0.0, 0.0, 0.0, 0.0, depth=1.0)
        plate_t.use(0); plate_prog["uPlate"].value = 0
        va_plate.render(moderngl.TRIANGLES)

        # Pass 2A: 320 Articulated 3D Flapping Morpho Butterflies (depth-tested with walker depth occlusion!)
        b_arr = bfly_frames[i]
        if len(b_arr) > 0:
            bf_inst.write(b_arr.tobytes())
            ctx.enable(moderngl.DEPTH_TEST); ctx.depth_mask = True
            ctx.enable(moderngl.BLEND); ctx.blend_func = (moderngl.SRC_ALPHA, moderngl.ONE_MINUS_SRC_ALPHA)
            mask_t.use(0)
            bfly_prog["uPersonMask"].value = 0
            bfly_prog["uPersonDepth"].value = p_dep
            va_bfly.render(moderngl.TRIANGLES, instances=len(b_arr))

        # Pass 2B: 3 Flowing Translucent 3D Satin Silk Ribbons wrapping BEHIND & IN FRONT OF his body!
        sv, si_arr = build_silk_ribbon_geometry(walker_pos, i, fps)
        if len(si_arr) > 0:
            silk_vbo.write(sv.tobytes())
            silk_ibo.write(si_arr.tobytes())
            ctx.enable(moderngl.DEPTH_TEST); ctx.depth_mask = False
            ctx.enable(moderngl.BLEND); ctx.blend_func = (moderngl.SRC_ALPHA, moderngl.ONE_MINUS_SRC_ALPHA)
            mask_t.use(0)
            silk_prog["uPersonMask"].value = 0
            silk_prog["uPersonDepth"].value = p_dep
            silk_prog["uTime"].value = float(i / fps)
            va_silk.render(moderngl.TRIANGLES, vertices=len(si_arr))

        # Pass 2C: Shimmering Wing-Dust Light Motes (additive HDR glow!)
        d_arr = dust_frames[i]
        if len(d_arr) > 0:
            dust_inst.write(d_arr.tobytes())
            ctx.enable(moderngl.DEPTH_TEST); ctx.depth_mask = False
            ctx.enable(moderngl.BLEND); ctx.blend_func = (moderngl.ONE, moderngl.ONE)
            mask_t.use(0)
            dust_prog["uPersonMask"].value = 0
            dust_prog["uPersonDepth"].value = p_dep
            va_dust.render(moderngl.TRIANGLES, instances=len(d_arr))
        ctx.depth_mask = True

        # Pass 3: Dual-Radius 1:1 Ethereal Bioluminescent Optical Bloom
        ctx.disable(moderngl.DEPTH_TEST); ctx.disable(moderngl.BLEND)
        fb_b1a.use(); ctx.viewport = (0, 0, hw, hh)
        t_emiss.use(0); blur_prog["uTex"].value = 0; blur_prog["uOutRes"].value = (float(hw), float(hh)); blur_prog["uDir"].value = (2.2 / hw, 0.0)
        va_blur.render(moderngl.TRIANGLES)
        fb_b1b.use(); ctx.viewport = (0, 0, hw, hh)
        t_b1a.use(0); blur_prog["uTex"].value = 0; blur_prog["uOutRes"].value = (float(hw), float(hh)); blur_prog["uDir"].value = (0.0, 2.2 / hh)
        va_blur.render(moderngl.TRIANGLES)
        fb_b2a.use(); ctx.viewport = (0, 0, qw, qh)
        t_b1b.use(0); blur_prog["uTex"].value = 0; blur_prog["uOutRes"].value = (float(qw), float(qh)); blur_prog["uDir"].value = (3.8 / qw, 0.0)
        va_blur.render(moderngl.TRIANGLES)
        fb_b2b.use(); ctx.viewport = (0, 0, qw, qh)
        t_b2a.use(0); blur_prog["uTex"].value = 0; blur_prog["uOutRes"].value = (float(qw), float(qh)); blur_prog["uDir"].value = (0.0, 3.8 / qh)
        va_blur.render(moderngl.TRIANGLES)

        # Pass 4: Final A24 Fantasy Composite
        fbo_final.use(); ctx.viewport = (0, 0, W, H)
        t_scene.use(0); t_b1b.use(1); t_b2b.use(2)
        comp_prog["uScene"].value = 0; comp_prog["uBloom1"].value = 1; comp_prog["uBloom2"].value = 2
        comp_prog["uFrame"].value = float(i)
        va_comp.render(moderngl.TRIANGLES)
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
            cv2.putText(b, f"butterfly+silk f={idx}", (10, 344), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 0, 0), 3)
            cv2.putText(b, f"butterfly+silk f={idx}", (10, 344), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (140, 235, 255), 1)
            rows.append(np.hstack([a, b]))
        out_p = wk / "still_butterfly_sheet.jpg"
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
            if i % 40 == 0 or i == N - 1:
                print(f"butterfly_silk frame {i}/{N} ({time.time() - ts:.1f}s)", flush=True)
        out_dir = here / "out"; out_dir.mkdir(parents=True, exist_ok=True)
        r = f"{meta['fps_num']}/{meta['fps_den']}"; ff = find_ffmpeg()
        tags = ["-pix_fmt", "yuv420p", "-color_primaries", "bt709", "-color_trc", "bt709", "-colorspace", "bt709", "-movflags", "+faststart"]
        x264 = ["-c:v", "libx264", "-profile:v", "high", "-level", "4.1", "-crf", "14", "-preset", "slow"]
        run = lambda a: subprocess.run([ff, "-v", "error", "-y"] + a, check=True)
        run(["-framerate", r, "-i", str(comp_dir / "%04d.png")] + x264 + tags + [str(out_dir / "Butterfly_Silk_Aura_Composite.mp4")])
        run([
            "-framerate", r, "-i", str(comp_dir / "%04d.png"),
            "-framerate", r, "-i", str(WORK / "frames" / "%04d.jpg"),
            "-filter_complex", "[0]scale=1080:608[a];[1]scale=1080:608[b];[a][b]vstack,pad=1080:1350:0:67:color=black",
        ] + x264 + tags + [str(out_dir / "Butterfly_Silk_Aura_Compare_1080x1350.mp4")])
        print("encoded ->", out_dir)


if __name__ == "__main__":
    main()
