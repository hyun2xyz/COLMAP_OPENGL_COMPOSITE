"""Effect #5: 3D Ember/Ash Particle Disintegration & Reverse-Time Reassembly (Native OpenGL 3.3).

    python render_disintegration.py --still 28,48,76,102,126   quick look -> work/still_disint_sheet.jpg
    python render_disintegration.py --render                   full render -> out/Particle_Disintegration_Reassembly_*.mp4

Moving-camera contemporary dance footage (Pexels 32109503, 163f @ 25fps, dark architectural glass facade):
- Spotless moving-camera homography AI clean plate (0.50px mean err + floor shadow auto-cleaned) reveals the
  empty plaza cleanly as the dancer completely disintegrates and fades his sunlit concrete floor shadow away.
- Phase 1 (f=18..64): Organic 3D body-locked fiery burn front eats diagonally from raised fingertips -> torso -> feet
  with molten white-orange-gold incandescent rim + charred soot border, spawning ~30,000 true 3D tumbling ash/cloth
  flakes (sampling exact live white-shirt/skin/black-pant RGB!) & glowing motion-streaked embers into a 3D Curl-Noise
  wind vortex sweeping across the dark glass facade.
- Phase 2 (f=65..86): Suspended mid-air 3D ember & ash storm hangs over the empty plaza, then reverses momentum!
- Phase 3 (f=87..138): ~30,000 backward-integrated 3D particles whip out of the mid-air storm along swirling curl
  streamlines and SNAP with pixel-exact COLMAP 3D precision back into his live moving body along a rising fiery seam!
- Phase 4 (f=139..162): Solid live-action dance outro with lingering golden embers drifting past in sunlight.
- Dual-pass HDR optical bloom + velocity-stretched spark motion blur + matched plate sensor grain. ZERO text/HUD.
"""
import argparse, json, math, shutil, subprocess, sys, time
from pathlib import Path
import numpy as np, cv2, moderngl

here = Path(__file__).resolve().parent
ROOT = here.parents[1]
sys.path.insert(0, str(ROOT / "examples/shader_live/pipeline"))
from slv import load, work, frame, scene  # noqa: E402

SRC_PIECE = ROOT / "examples/shockwave_teleport/piece_moving"


def find_ffmpeg():
    tl = ROOT / "tools.local.json"
    if tl.exists():
        d = json.loads(tl.read_text(encoding="utf-8"))
        p = d.get("ffmpeg") or (d.get("tools") or {}).get("ffmpeg")
        if p and Path(p).exists():
            return str(p)
    return shutil.which("ffmpeg") or "ffmpeg"


def clean_residual_floor_shadow(clean80):
    """Remove the tiny residual dark floor shadow patch (y=845..925, x=1065..1455) from clean_plate_ref80.jpg
    by seamlessly cloning adjacent spotless sunlit concrete tile texture from the same row."""
    out = clean80.copy()
    H, W = out.shape[:2]
    y0, y1, x0, x1 = 842, 928, 1060, 1465
    # Clone exact sunlit concrete slab 440px to the left (identical perspective row & paving tone!)
    src_x0 = x0 - 440
    patch_src = out[y0:y1, src_x0:src_x0 + (x1 - x0)].astype(np.float32)
    patch_dst = out[y0:y1, x0:x1].astype(np.float32)
    # Match subtle horizontal brightness gradient across the border of the patch
    border_top = out[y0 - 8:y0, x0:x1].mean(axis=(0, 1)) - out[y0 - 8:y0, src_x0:src_x0 + (x1 - x0)].mean(axis=(0, 1))
    patch_src = np.clip(patch_src + border_top[None, None, :], 0, 255)
    # Smooth elliptical feather mask centered strictly on the dark patch
    yy, xx = np.mgrid[0:(y1 - y0), 0:(x1 - x0)].astype(np.float32)
    cy, cx = (y1 - y0) * 0.5, (x1 - x0) * 0.5
    r = np.hypot((yy - cy) / (cy * 0.85), (xx - cx) / (cx * 0.88))
    alpha = np.clip(1.0 - (r - 0.55) / 0.45, 0.0, 1.0)
    alpha = alpha * alpha * (3.0 - 2.0 * alpha)
    out[y0:y1, x0:x1] = np.clip(patch_dst * (1.0 - alpha[:, :, None]) + patch_src * alpha[:, :, None], 0, 255).astype(np.uint8)
    return out


def smoothstep_np(e0, e1, x):
    t = np.clip((x - e0) / max(e1 - e0, 1e-6), 0.0, 1.0)
    return t * t * (3.0 - 2.0 * t)


def curl_wind_3d(pos, t_val):
    """Exact divergence-free 3D analytical Curl Noise (curl of 3-component trig vector potential)
    for sweeping cinematic Houdini-style wind turbulence in world meters!"""
    x, y, z = pos[:, 0], pos[:, 1], pos[:, 2]
    # Octave 1: broad majestic architectural wind eddies (scale ~1.35 rad/m)
    k1 = 1.45
    ph1 = t_val * 0.72
    sx1, cx1 = np.sin(k1 * x + ph1), np.cos(k1 * x + ph1)
    sy1, cy1 = np.sin(k1 * y - ph1 * 0.6), np.cos(k1 * y - ph1 * 0.6)
    sz1, cz1 = np.sin(k1 * z + ph1 * 0.5), np.cos(k1 * z + ph1 * 0.5)
    # Potential Psi = (sin(ky)*cos(kz), sin(kz)*cos(kx), sin(kx)*cos(ky)) -> exact curl:
    vx1 = k1 * (sx1 * (-sy1) - (-sz1) * cx1)  # dPsi_z/dy - dPsi_y/dz
    vy1 = k1 * (sy1 * (-sz1) - (-sx1) * cy1)  # dPsi_x/dz - dPsi_z/dx
    vz1 = k1 * (sz1 * (-sx1) - (-sy1) * cz1)  # dPsi_y/dx - dPsi_x/dy

    # Octave 2: crisp swirling micro-vortices (scale ~3.4 rad/m)
    k2 = 3.45
    ph2 = t_val * 1.35 + 2.1
    sx2, cx2 = np.sin(k2 * x - ph2), np.cos(k2 * x - ph2)
    sy2, cy2 = np.sin(k2 * y + ph2 * 0.8), np.cos(k2 * y + ph2 * 0.8)
    sz2, cz2 = np.sin(k2 * z - ph2 * 0.7), np.cos(k2 * z - ph2 * 0.7)
    vx2 = 0.42 * k2 * (sx2 * (-sy2) - (-sz2) * cx2)
    vy2 = 0.42 * k2 * (sy2 * (-sz2) - (-sx2) * cy2)
    vz2 = 0.42 * k2 * (sz2 * (-sx2) - (-sy2) * cz2)

    return np.stack([vx1 + vx2, vy1 + vy2, vz1 + vz2], axis=1) * 0.58


def build_body_burn_noise(H, W, seed=42):
    """Build a crisp multi-octave seamless organic noise texture (tileable in body UV) for jagged ember edges."""
    rng = np.random.default_rng(seed)
    acc = np.zeros((H, W), dtype=np.float32)
    amp = 0.52
    for sz in (12, 26, 58, 128):
        g = rng.random((sz, sz), dtype=np.float32)
        up = cv2.resize(g, (W, H), interpolation=cv2.INTER_CUBIC)
        acc += amp * up
        amp *= 0.50
    acc = (acc - acc.min()) / max(acc.max() - acc.min(), 1e-5)
    return acc.astype(np.float32)


def eval_body_field(mask_u8, noise_tex, mode="disint"):
    """Compute exact normalized organic burn-order scalar field S in [0, 1] inside dancer silhouette.
    - mode='disint': eats diagonally from upper-left raised fingertips (0.0) -> torso -> feet (1.0)
    - mode='reasm' : heals upward organically from feet (0.0) -> legs -> torso -> raised fingertips (1.0)
    Returns float32 (H, W) with values in [0, 1] inside mask (and smooth extrapolated outside)."""
    H, W = mask_u8.shape
    ys, xs = np.where(mask_u8 > 55)
    if len(ys) < 20:
        return np.zeros((H, W), dtype=np.float32)
    y_min, y_max = float(ys.min()), float(ys.max())
    x_min, x_max = float(xs.min()), float(xs.max())
    bh = max(y_max - y_min, 40.0)
    bw = max(x_max - x_min, 40.0)

    # Normalized body-local coordinates (u in [0,1] left->right, v in [0,1] top->bottom)
    yy = np.linspace(0, H - 1, H, dtype=np.float32)[:, None]
    xx = np.linspace(0, W - 1, W, dtype=np.float32)[None, :]
    v = np.clip((yy - y_min) / bh, -0.2, 1.2)
    u = np.clip((xx - x_min) / bw, -0.2, 1.2)

    # Map body-local (u, v) cleanly into the precomputed organic noise texture so fire cracks STICK to his body!
    nh, nw = noise_tex.shape
    ix = np.clip((u[0, :] * (nw - 1)).astype(np.int32), 0, nw - 1)
    iy = np.clip((v[:, 0] * (nh - 1)).astype(np.int32), 0, nh - 1)
    n_body = noise_tex[iy[:, None], ix[None, :]]

    if mode == "disint":
        # Diagonal sweep from upper-left fingertips -> torso -> bare feet (organic edge, zero detached islands!)
        base = 0.82 * v + 0.18 * u
    else:
        # Reverse-time rising sweep from feet -> legs -> torso -> head & fingertips!
        base = 0.88 * (1.0 - v) + 0.12 * (1.0 - u)

    raw = base + 0.18 * (n_body - 0.5)
    # Normalize strictly across actual silhouette pixels so 0.0 -> 1.0 covers 100% of body with crisp margin
    vals = raw[mask_u8 > 80]
    lo, hi = float(np.percentile(vals, 0.5)), float(np.percentile(vals, 99.5))
    field = np.clip((raw - lo) / max(hi - lo, 1e-4), 0.0, 1.0)
    return field.astype(np.float32)


def timeline_state(i):
    """Master choreography timeline over N=163 frames (25fps = 6.52s):
    Returns (phase_mode, burn_p, shadow_vis):
      - f=0..15  : solid live action ('disint', burn_p=-0.10, shadow=1.0)
      - f=16..64 : full diagonal fiery disintegration ('disint', burn_p: 0.0 -> 1.14, shadow: 1.0 -> 0.0)
      - f=65..85 : empty plaza + suspended 3D ember vortex ('empty', burn_p=1.20, shadow=0.0)
      - f=86..136: dramatic reverse-time magnetic reassembly ('reasm', burn_p: -0.05 -> 1.14, shadow: 0.0 -> 1.0)
      - f=137..162: solid live-action outro ('solid', burn_p=1.20, shadow=1.0)
    """
    if i < 16:
        # Subtle pre-ignition hint at f=12..15
        pre = max(0.0, (i - 12) / 4.0) * 0.04
        return "disint", -0.08 + pre, 1.0
    elif i <= 64:
        u = (i - 16) / 48.0
        # Smooth cinematic S-curve acceleration through torso & flowing black pants
        p = -0.02 + 1.16 * (u * u * (2.6 - 1.6 * u))
        sh = float(np.clip(1.0 - max(0.0, (p - 0.15) / 0.85), 0.0, 1.0))
        return "disint", float(p), sh
    elif i < 86:
        return "empty", 1.25, 0.0
    elif i <= 136:
        u = (i - 86) / 50.0
        p = -0.02 + 1.16 * (u * u * (2.5 - 1.5 * u))
        sh = float(np.clip(p / 0.72, 0.0, 1.0))
        return "reasm", float(p), sh
    else:
        return "solid", 1.25, 1.0


def smootherstep_np(e0, e1, x):
    t = np.clip((x - e0) / max(e1 - e0, 1e-6), 0.0, 1.0)
    return t * t * t * (t * (t * 6.0 - 15.0) + 10.0)


def simulate_all_particles(c, cam_C, cam_R, f_px, p_depths, noise_tex, W, H, fps):
    """Simulate ONE strictly conserved population (~24,200 3D particles) from start to finish:
    - ZERO particles ever spawn or dissolve into existence in mid-air!
    - 100% of particles are born exclusively along the burning body seam during f=16..64.
    - During f=64..86, those exact particles swirl in mid-air over the empty plaza and execute a smooth C2
      turnaround along 3D Curl-Noise streamlines (spatially matched in 12 height bands for natural U-turns).
    - During f=86..136, every single particle whips back and snaps pixel-exact into his live reassembling body.
    - At f=137..162, 0 stray mid-air sparks remain (clean live-action dance outro).
    Returns list-of-arrays `frame_batches[i]` -> (K_i, 13) float32 ready for direct GPU upload!
    """
    N = len(cam_C)
    dt = 1.0 / fps
    rng = np.random.default_rng(2026)

    def unproject_pixels(xs, ys, i, z_jitter):
        ro = cam_C[i]
        R = cam_R[i]
        d_cam = p_depths[i] + z_jitter
        xc = (xs - 0.5 * W) / f_px * d_cam
        yc = (ys - 0.5 * H) / f_px * d_cam
        zc = d_cam
        q_cam = np.stack([xc, yc, zc], axis=1)
        return ro[None, :] + q_cam @ R.T

    # ------------------------------------------------------------------
    # 1. SAMPLE EXACT DISINTEGRATION BIRTHS ON BODY (f=16..64)
    # ------------------------------------------------------------------
    d_pos_list, d_rgb_list, d_spawn_list = [], [], []
    prev_p = -0.08
    for i in range(16, 65):
        _, cur_p, _ = timeline_state(i)
        m_u8 = cv2.imread(str(work(c, "mask", f"{i:04d}.png")), cv2.IMREAD_GRAYSCALE)
        if m_u8 is None:
            prev_p = cur_p
            continue
        fld = eval_body_field(m_u8, noise_tex, mode="disint")
        band = (m_u8 > 120) & (fld >= prev_p - 0.012) & (fld <= cur_p + 0.014)
        ys, xs = np.where(band)
        prev_p = cur_p
        if len(xs) == 0:
            continue
        k_take = min(len(xs), 620)
        sel = rng.choice(len(xs), size=k_take, replace=False)
        sx = xs[sel].astype(np.float64) + rng.uniform(-0.5, 0.5, k_take)
        sy = ys[sel].astype(np.float64) + rng.uniform(-0.5, 0.5, k_take)
        pl = frame(c, i)
        rgb = pl[ys[sel], xs[sel], ::-1].astype(np.float32) / 255.0
        z_jit = rng.normal(0.0, 0.075, k_take)
        pos3 = unproject_pixels(sx, sy, i, z_jit)
        d_pos_list.append(pos3.astype(np.float32))
        d_rgb_list.append(rgb)
        d_spawn_list.append(np.full(k_take, i, dtype=np.int32))

    d_pos0 = np.concatenate(d_pos_list, axis=0)
    d_rgb0 = np.concatenate(d_rgb_list, axis=0)
    d_spawn = np.concatenate(d_spawn_list, axis=0)
    M = len(d_spawn)

    # ------------------------------------------------------------------
    # 2. SAMPLE EXACT REASSEMBLY TARGETS ON LIVE BODY (f=86..136)
    # ------------------------------------------------------------------
    r_pos_list, r_rgb_list, r_arr_list = [], [], []
    prev_p = -0.05
    for i in range(86, 137):
        _, cur_p, _ = timeline_state(i)
        m_u8 = cv2.imread(str(work(c, "mask", f"{i:04d}.png")), cv2.IMREAD_GRAYSCALE)
        if m_u8 is None:
            prev_p = cur_p
            continue
        fld = eval_body_field(m_u8, noise_tex, mode="reasm")
        band = (m_u8 > 120) & (fld >= prev_p - 0.012) & (fld <= cur_p + 0.014)
        ys, xs = np.where(band)
        prev_p = cur_p
        if len(xs) == 0:
            continue
        k_take = min(len(xs), 620)
        sel = rng.choice(len(xs), size=k_take, replace=False)
        sx = xs[sel].astype(np.float64) + rng.uniform(-0.5, 0.5, k_take)
        sy = ys[sel].astype(np.float64) + rng.uniform(-0.5, 0.5, k_take)
        pl = frame(c, i)
        rgb = pl[ys[sel], xs[sel], ::-1].astype(np.float32) / 255.0
        z_jit = rng.normal(0.0, 0.065, k_take)
        pos3 = unproject_pixels(sx, sy, i, z_jit)
        r_pos_list.append(pos3.astype(np.float32))
        r_rgb_list.append(rgb)
        r_arr_list.append(np.full(k_take, i, dtype=np.int32))

    r_pos_all = np.concatenate(r_pos_list, axis=0)
    r_rgb_all = np.concatenate(r_rgb_list, axis=0)
    r_arr_all = np.concatenate(r_arr_list, axis=0)
    # Resample to exact 1-to-1 count M so 100% of shed particles map directly to 100% of reassembled particles!
    pick = rng.choice(len(r_arr_all), size=M, replace=(len(r_arr_all) < M))
    r_pos1 = r_pos_all[pick]
    r_rgb1 = r_rgb_all[pick]
    r_arr = r_arr_all[pick]

    # Shared particle identities across their entire continuous life (f=16 -> f=136)
    ptype = (rng.random(M) < 0.28).astype(np.float32)
    psize = np.where(ptype > 0.5, rng.uniform(0.0038, 0.0102, M), rng.uniform(0.0055, 0.0142, M)).astype(np.float32)
    phase = rng.uniform(0.0, 6.28318, M).astype(np.float32)
    drag  = rng.uniform(0.58, 1.48, M).astype(np.float32)
    cool  = rng.uniform(0.75, 1.65, M).astype(np.float32)
    wmul  = rng.uniform(0.55, 1.45, M).astype(np.float32)

    # ------------------------------------------------------------------
    # 3. INTEGRATE FORWARD BRANCH P_fwd[16..92] & BACKWARD BRANCH P_bwd[58..136]
    # ------------------------------------------------------------------
    f_lo, f_hi = 16, 90
    b_lo, b_hi = 58, 136
    P_fwd = np.zeros((f_hi - f_lo + 1, M, 3), dtype=np.float32)
    cur_p = d_pos0.copy()
    cur_v = np.stack([
        rng.uniform(0.18, 1.15, M),
        rng.normal(0.0, 0.22, M),
        rng.uniform(-0.08, 0.88, M),
    ], axis=1).astype(np.float32)

    for i in range(f_lo, f_hi + 1):
        P_fwd[i - f_lo] = cur_p
        moving = (d_spawn <= i)
        if np.any(moving):
            m_idx = np.where(moving)[0]
            p_m = cur_p[m_idx]
            v_m = cur_v[m_idx]
            cw = curl_wind_3d(p_m, i * dt)
            hang = 1.0 - 0.78 * float(smoothstep_np(60.0, 75.0, float(i)))
            wind_tgt = (np.array([0.82, 0.0, 0.34], dtype=np.float32)[None, :] * wmul[m_idx, None]) * hang + cw * (1.35 * hang + 0.32)
            v_m = v_m + (wind_tgt - v_m) * (dt * 2.0 * drag[m_idx, None])
            v_m *= (0.94 + 0.06 * hang)
            p_m = p_m + v_m * dt
            p_m[:, 1] = np.clip(p_m[:, 1], -0.95, 1.35)
            below = p_m[:, 2] < 0.08
            if np.any(below):
                p_m[below, 2] = 0.08
                v_m[below, 2] = np.abs(v_m[below, 2]) * 0.45 + 0.18
            cur_p[m_idx] = p_m
            cur_v[m_idx] = v_m

    P_bwd = np.zeros((b_hi - b_lo + 1, M, 3), dtype=np.float32)
    cur_p = r_pos1.copy()
    cur_v = np.stack([
        rng.uniform(-1.20, -0.25, M),
        rng.normal(0.0, 0.22, M),
        rng.uniform(-0.78, 0.12, M),
    ], axis=1).astype(np.float32)

    for i in range(b_hi, b_lo - 1, -1):
        P_bwd[i - b_lo] = cur_p
        active = (r_arr >= i)
        if np.any(active):
            s_idx = np.where(active)[0]
            p_s = cur_p[s_idx]
            v_s = cur_v[s_idx]
            cw = curl_wind_3d(p_s, i * dt + 1.4)
            pull = 0.22 + 0.78 * float(smoothstep_np(72.0, 90.0, float(i)))
            fwd_vel_tgt = (np.array([-0.86, 0.0, -0.32], dtype=np.float32)[None, :] * wmul[s_idx, None]) * pull + cw * (1.30 * pull + 0.32)
            p_s = p_s - v_s * dt
            v_s = v_s + (fwd_vel_tgt - v_s) * (dt * 2.0 * drag[s_idx, None])
            p_s[:, 1] = np.clip(p_s[:, 1], -0.95, 1.35)
            p_s[:, 2] = np.clip(p_s[:, 2], 0.14, 2.30)
            cur_p[s_idx] = p_s
            cur_v[s_idx] = v_s

    # ------------------------------------------------------------------
    # 4. SPATIAL CORRIDOR PAIRING AT MID-AIR TURNAROUND (f=74)
    #    Matches each forward floating particle to a nearby return trajectory
    #    so every particle executes a graceful local mid-air U-turn!
    # ------------------------------------------------------------------
    f_mid = 74
    pf_mid = P_fwd[f_mid - f_lo]
    pb_mid = P_bwd[f_mid - b_lo]
    # Bucket into 14 height (z) bands and sort by x within each band for tight spatial matching
    def spatial_order(pts):
        z_rank = np.argsort(np.argsort(pts[:, 2]))
        z_bin = (z_rank * 14) // len(pts)
        return np.lexsort((pts[:, 0], z_bin))

    ord_f = spatial_order(pf_mid)
    ord_b = spatial_order(pb_mid)
    # Map forward index -> paired backward index
    pair_b = np.empty(M, dtype=np.int32)
    pair_b[ord_f] = ord_b

    # Re-index backward arrays to align 1-to-1 with forward particle indices 0..M-1!
    P_bwd = P_bwd[:, pair_b, :]
    r_rgb1 = r_rgb1[pair_b]
    r_arr = r_arr[pair_b]

    # ------------------------------------------------------------------
    # 5. EVALUATE SINGLE CONTINUOUS TRAJECTORY P(i) FOR EVERY FRAME (16..136)
    #    Exact finite-difference velocity -> authentic motion-blur streaks!
    #    ZERO mid-air spawns, ZERO mid-air fades!
    # ------------------------------------------------------------------
    frame_batches = [np.zeros((0, 13), dtype=np.float32) for _ in range(N)]
    prev_pos_global = d_pos0.copy()
    max_k = 0

    for i in range(16, 137):
        alive = (d_spawn <= i) & (i <= r_arr)
        if not np.any(alive):
            continue
        idx = np.where(alive)[0]
        # Global C2 smootherstep turnaround window across empty-plaza hover (f=62..86)
        w = float(smootherstep_np(62.0, 86.0, float(i)))
        if i <= 62:
            pos_i = P_fwd[i - f_lo, idx]
        elif i >= 86:
            pos_i = P_bwd[i - b_lo, idx]
        else:
            pos_i = (1.0 - w) * P_fwd[i - f_lo, idx] + w * P_bwd[i - b_lo, idx]

        # True 3D frame-to-frame velocity (for newly born particles use small initial lift velocity)
        newly_born = (d_spawn[idx] == i)
        vel_i = (pos_i - prev_pos_global[idx]) / dt
        if np.any(newly_born):
            vel_i[newly_born] = np.array([0.45, 0.0, 0.30], dtype=np.float32)
        prev_pos_global[idx] = pos_i

        age_birth = (i - d_spawn[idx]).astype(np.float32) * dt
        rem_arr   = (r_arr[idx] - i).astype(np.float32) * dt

        # Smooth RGB blend from shed pixel color -> arriving live body pixel color during turnaround
        rgb_i = (1.0 - w) * d_rgb0[idx] + w * r_rgb1[idx]

        # Physical thermal curve: hot off burning seam -> cools to amber/ash in mid-air -> re-ignites on magnetic snap-in!
        heat_birth = np.exp(-age_birth * cool[idx] * np.where(ptype[idx] > 0.5, 0.55, 2.0))
        heat_snap  = np.exp(-rem_arr * np.where(ptype[idx] > 0.5, 0.65, 2.2)) * float(smoothstep_np(72.0, 92.0, float(i)))
        ambient_ember = 0.28 * ptype[idx]
        temp_i = np.clip(np.maximum(heat_birth, heat_snap * 1.15) + ambient_ember, 0.0, 1.30)

        # Gentle birth & snap-in scale ramp ONLY right at body contact (0.06s), 100% solid in mid-air!
        end_ramp = np.clip(np.minimum(age_birth, rem_arr) / 0.06 + 0.40, 0.40, 1.0)
        sz_i = psize[idx] * end_ramp
        ph_i = phase[idx] + age_birth * 4.6

        batch = np.column_stack([
            pos_i,
            vel_i,
            rgb_i,
            temp_i.astype(np.float32),
            sz_i.astype(np.float32),
            ph_i.astype(np.float32),
            ptype[idx],
        ]).astype(np.float32)
        frame_batches[i] = batch
        max_k = max(max_k, len(batch))

    print(f"Conserved 3D Particle Sim: M={M} particles (0 mid-air spawns!) | peak active/frame={max_k}")
    return frame_batches


# ==============================================================================
#  SHADERS: PASS 1 (CLEAN PLATE + FIERY BURN FRONT), PASS 2 (3D INSTANCED FLAKES/EMBERS),
#           PASS 3 (DUAL-PASS OPTICAL BLOOM + SENSOR GRAIN FINISH)
# ==============================================================================

VS_FSQ = "#version 330\nin vec2 p; void main(){ gl_Position = vec4(p, 0.0, 1.0); }"

BODY_FS = """#version 330
uniform sampler2D uLive, uClean, uMask, uRemZone, uBurnField;
uniform vec2 uRes;
uniform float uBurnProg, uShadowVis, uTime;
uniform int uMode; /* 0=disint, 1=empty, 2=reasm, 3=solid */

layout(location=0) out vec4 oScene; /* linear HDR scene color */
layout(location=1) out vec4 oEmiss; /* emissive fire rim for optical bloom! */

float hash21(vec2 p){ vec3 q=fract(vec3(p.xyx)*0.1031); q+=dot(q,q.yzx+33.33); return fract((q.x+q.y)*q.z); }

void main() {
  vec2 px = vec2(gl_FragCoord.x, uRes.y - gl_FragCoord.y);
  vec2 uv = clamp(px / uRes, vec2(0.001), vec2(0.999));

  vec3 liveSrgb  = texture(uLive, uv).rgb;
  vec3 cleanSrgb = texture(uClean, uv).rgb;
  vec3 liveLin   = pow(max(liveSrgb, vec3(0.0005)), vec3(2.2));
  vec3 cleanLin  = pow(max(cleanSrgb, vec3(0.0005)), vec3(2.2));
  float m        = texture(uMask, uv).r;
  float remZ     = texture(uRemZone, uv).r;
  float bf       = texture(uBurnField, uv).r;

  /* Base background with sunlit floor shadow smoothly fading out as body disintegrates (and returning on reassembly!) */
  float shadowWipe = clamp(remZ * (1.0 - uShadowVis), 0.0, 1.0);
  vec3 bgLin = mix(liveLin, cleanLin, shadowWipe);

  if (uMode == 1) {
    /* Completely vanished empty plaza */
    vec3 emptyCol = mix(liveLin, cleanLin, clamp(max(smoothstep(0.01, 0.25, m), remZ), 0.0, 1.0));
    oScene = vec4(emptyCol, 0.0);
    oEmiss = vec4(0.0);
    return;
  }
  if (uMode == 3) {
    oScene = vec4(liveLin, m);
    oEmiss = vec4(0.0);
    return;
  }

  /* Signed distance to active organic burn seam:
     disint (uMode==0): keep pixels where bf > uBurnProg; eaten where bf < uBurnProg
     reasm  (uMode==2): keep pixels where bf < uBurnProg; not-yet-arrived where bf > uBurnProg */
  float dEdge = (uMode == 0) ? (bf - uBurnProg) : (uBurnProg - bf);

  /* Crisp jagged live-body opacity with glowing molten rim right at dEdge ~ 0 */
  float bodyKeep = smoothstep(-0.006, 0.014, dEdge);
  float personAlpha = m * bodyKeep;

  /* Ensure eaten person pixels cleanly reveal spotless clean plate underneath! */
  float personWipe = smoothstep(0.01, 0.22, m) * (1.0 - bodyKeep);
  vec3 baseLin = mix(bgLin, cleanLin, personWipe);

  /* Charred dark soot band eating slightly ahead into his white shirt/skin right before it burns to ash! */
  float charBand = smoothstep(0.075, 0.012, dEdge) * smoothstep(-0.010, 0.012, dEdge) * m;
  vec3 liveCharred = mix(liveLin, liveLin * vec3(0.14, 0.07, 0.04), charBand * 0.78);
  vec3 col = mix(baseLin, liveCharred, personAlpha);

  /* Blazing incandescent molten-orange & white-gold ember seam right along dEdge in [-0.018, +0.042]! */
  float flicker = 0.85 + 0.30 * hash21(floor(px * 0.35) + vec2(uTime * 19.0, -uTime * 13.0));
  float hotCore = exp(-pow((dEdge - 0.008) / 0.012, 2.0)) * smoothstep(0.06, 0.35, m) * flicker;
  float hotHalo = exp(-pow((dEdge - 0.016) / 0.036, 2.0)) * smoothstep(0.03, 0.28, m);

  /* Slightly hotter golden-white during magnetic reassembly, fiery orange-amber during disintegration */
  vec3 seamCore = (uMode == 0) ? vec3(4.2, 1.95, 0.42) : vec3(4.6, 2.55, 0.72);
  vec3 seamHalo = (uMode == 0) ? vec3(2.2, 0.48, 0.04) : vec3(2.4, 0.72, 0.10);
  vec3 emiss = seamCore * hotCore + seamHalo * hotHalo * 0.85;

  oScene = vec4(col + emiss * 0.55, personAlpha);
  oEmiss = vec4(emiss, 1.0);
}
"""

PART_VS = """#version 330
in vec2 inQuad;     /* unit quad corner in [-1, 1]^2 */
in vec3 inPos;      /* world 3D pos */
in vec3 inVel;      /* world 3D velocity */
in vec3 inRgb;      /* sampled live shirt/skin/pants RGB */
in float inTemp;    /* ember heat (0..1.4) */
in float inSize;    /* world radius (meters) */
in float inPhase;   /* spin / shard seed */
in float inType;    /* 0=torn cloth/ash flake, 1=incandescent ember/spark */

uniform vec2 uRes;
uniform float uF;
uniform vec3 uRo;
uniform mat3 uCam;

out vec2 vLocal;
out vec3 vRgb;
out float vTemp;
out float vPhase;
out float vType;
out float vDepth;
out float vLight;

void main() {
  /* Project 3D particle center through exact COLMAP camera */
  vec3 q = transpose(uCam) * (inPos - uRo);
  if (q.z < 0.25) {
    gl_Position = vec4(-10.0, -10.0, -10.0, 1.0);
    return;
  }
  vec2 px0 = uF * q.xy / q.z + 0.5 * uRes;
  /* Project velocity tail for authentic 3D screen-space shutter motion blur streaks! */
  vec3 qPrev = transpose(uCam) * (inPos - inVel * 0.034 - uRo);
  vec2 pxPrev = uF * qPrev.xy / max(qPrev.z, 0.25) + 0.5 * uRes;
  vec2 velPx = px0 - pxPrev;
  float spdPx = length(velPx);

  /* World-space radius -> crisp 3D perspective pixel size (never giant blurry blobs!) */
  float radPx = clamp(uF * inSize / q.z, 1.15, 9.5);

  vec2 axisX, axisY;
  if (inType > 0.5 && spdPx > 1.2) {
    /* Glowing embers & sparks stretch sleekly along their 3D velocity vector (motion-blurred fire streaks!) */
    vec2 dir = velPx / spdPx;
    vec2 perp = vec2(-dir.y, dir.x);
    float stretch = min(spdPx * 0.62 + radPx, radPx * 4.5);
    axisX = dir * stretch;
    axisY = perp * (radPx * 0.54);
  } else {
    /* Torn cloth & charred ash flakes tumble organically in 3D with perspective foreshortening! */
    float ca = cos(inPhase * 0.7), sa = sin(inPhase * 0.7);
    float tumble = 0.32 + 0.68 * abs(cos(inPhase * 1.35));
    axisX = vec2(ca, sa) * radPx;
    axisY = vec2(-sa, ca) * (radPx * tumble);
  }

  vec2 px = px0 + axisX * inQuad.x + axisY * inQuad.y;
  /* Convert OpenCV top-down pixel coordinates to OpenGL NDC (y-up) */
  vec2 ndc = vec2(px.x / uRes.x * 2.0 - 1.0, (1.0 - px.y / uRes.y) * 2.0 - 1.0);
  /* Depth in [0, 1] for proper 3D front-to-back sorting */
  float zNdc = clamp((q.z - 0.25) / 25.0, 0.0, 0.999) * 2.0 - 1.0;
  gl_Position = vec4(ndc, zNdc, 1.0);

  vLocal = inQuad;
  vRgb   = inRgb;
  vTemp  = inTemp;
  vPhase = inPhase;
  vType  = inType;
  vDepth = q.z;
  /* Sunlight glint as each 3D flake tumbles in the afternoon golden sun! */
  vLight = 0.55 + 0.65 * abs(sin(inPhase * 1.35 + 0.6));
}
"""

FLAKE_FS = """#version 330
in vec2 vLocal;
in vec3 vRgb;
in float vTemp;
in float vPhase;
in float vType;
in float vDepth;
in float vLight;

uniform sampler2D uPersonAlpha;
uniform vec2 uRes;
uniform float uPersonDepth;

layout(location=0) out vec4 oScene;
layout(location=1) out vec4 oEmiss;

void main() {
  /* Only draw solid tumbling cloth/ash flakes in Pass 2A (additive glowing embers render in Pass 2B!) */
  if (vType > 0.5) discard;

  /* Depth occlusion against live solid dancer: flakes behind his remaining body tuck cleanly BEHIND him! */
  vec2 uvScr = gl_FragCoord.xy / uRes;
  float pA = texture(uPersonAlpha, uvScr).a;
  if (vDepth > uPersonDepth + 0.06 && pA > 0.45) discard;

  /* Irregular angular torn-cloth / burnt-paper shard silhouette (never boring circles!) */
  float ang = atan(vLocal.y, vLocal.x);
  float r = length(vLocal);
  float jag = 0.66 + 0.22 * cos(ang * 3.0 + vPhase * 2.1) + 0.14 * sin(ang * 5.0 - vPhase * 3.7);
  float a = smoothstep(jag, jag - 0.22, r);
  if (a < 0.05) discard;

  /* Sampled live white-shirt / skin / black-pants color -> chars gradually into warm ash as it drifts,
     with glowing red-orange molten edge while freshly torn off the burning seam! */
  vec3 liveLin = pow(max(vRgb, vec3(0.004)), vec3(2.2));
  vec3 ashCol = mix(liveLin * 0.48, vec3(0.06, 0.055, 0.05), 0.48);
  float fresh = smoothstep(0.08, 0.80, vTemp);
  vec3 bodyLin = mix(ashCol, liveLin, 0.48 + 0.52 * fresh) * vLight;

  /* Glowing molten edge on hot freshly-shed flakes */
  float edgeHot = smoothstep(jag * 0.35, jag * 0.92, r) * smoothstep(0.18, 0.95, vTemp);
  vec3 emberCol = mix(vec3(1.8, 0.32, 0.02), vec3(3.6, 1.55, 0.28), clamp(vTemp, 0.0, 1.0)) * edgeHot;

  oScene = vec4(bodyLin + emberCol * 0.7, a * 0.94);
  oEmiss = vec4(emberCol * a, a);
}
"""

EMBER_FS = """#version 330
in vec2 vLocal;
in vec3 vRgb;
in float vTemp;
in float vPhase;
in float vType;
in float vDepth;
in float vLight;

uniform sampler2D uPersonAlpha;
uniform vec2 uRes;
uniform float uPersonDepth;

layout(location=0) out vec4 oScene;
layout(location=1) out vec4 oEmiss;

void main() {
  /* Pass 2B: Incandescent velocity-streaked molten embers & sparks (additive HDR glow!) */
  if (vType <= 0.5) discard;

  vec2 uvScr = gl_FragCoord.xy / uRes;
  float pA = texture(uPersonAlpha, uvScr).a;
  if (vDepth > uPersonDepth + 0.06 && pA > 0.55) discard;

  float r2 = dot(vLocal, vLocal);
  if (r2 > 1.0) discard;
  float core = exp(-r2 * 5.2);
  float glow = pow(max(1.0 - r2, 0.0), 2.2);

  /* Sparkle twinkle as embers tumble through air */
  float twk = 0.72 + 0.42 * sin(vPhase * 3.1);
  float tHot = clamp(vTemp, 0.0, 1.35);

  /* Physical blackbody fire palette: balanced crisp fiery orange-gold embers (never over-whiteout!) */
  vec3 cRuby   = vec3(0.75, 0.08, 0.008);
  vec3 cOrange = vec3(1.65, 0.42, 0.03);
  vec3 cGold   = vec3(2.55, 1.38, 0.32);
  vec3 cWhite  = vec3(3.40, 2.75, 1.55);
  vec3 col = mix(cRuby, cOrange, smoothstep(0.05, 0.55, tHot));
  col = mix(col, cGold, smoothstep(0.55, 1.05, tHot));
  col = mix(col, cWhite, core * smoothstep(0.50, 1.20, tHot));

  vec3 outEm = col * (core * 0.95 + glow * 0.28) * twk;
  oScene = vec4(outEm * 0.52, 1.0);
  oEmiss = vec4(outEm * 0.78, 1.0);
}
"""

BLUR_FS = """#version 330
uniform sampler2D uTex;
uniform vec2 uOutRes; /* destination FBO size -> guarantees exact 1:1 full-frame UV mapping across mip levels! */
uniform vec2 uDir;    /* (1/w, 0) or (0, 1/h) scaled by radius */
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
  vec3 sceneLin = texture(uScene, uv).rgb;
  vec3 b1 = texture(uBloom1, uv).rgb;
  vec3 b2 = texture(uBloom2, uv).rgb;
  vec3 bloom = b1 * 0.55 + b2 * 0.42;

  /* Soft HDR highlight rolloff ONLY on excess luminous energy (> 1.0) so untouched dark glass & plaza keep
     100.0% EXACT original camera exposure & contrast! */
  vec3 baseClamped = min(sceneLin, vec3(1.0));
  vec3 hdrExcess = max(sceneLin - vec3(1.0), vec3(0.0)) + bloom;
  vec3 emSoft = hdrExcess / (1.0 + hdrExcess * 0.24);
  vec3 totalLin = baseClamped + emSoft * (1.0 - baseClamped * 0.22);

  vec3 col = pow(clamp(totalLin, 0.0, 1.0), vec3(1.0 / 2.2));
  /* Subtle film sensor grain matched to camera plate on glowing/active pixels */
  float act = clamp(length(bloom) * 2.5 + length(hdrExcess) * 2.0, 0.0, 1.0);
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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--still", default=None)
    ap.add_argument("--render", action="store_true")
    ap.add_argument("--out", default="out_v2")
    args = ap.parse_args()

    c = load(SRC_PIECE)
    S = scene(c)
    cam_C = np.asarray(S["cam"], dtype=np.float64)
    cam_R = np.asarray(S["rot"], dtype=np.float64)
    f_px = float(S["f"])
    N = len(cam_C)
    fps = float(c["fps"])
    pl0 = frame(c, 0)
    H, W = pl0.shape[:2]

    L = np.load(work(c, "limbs.npz"))
    p_depths = np.asarray(L["depth"], dtype=np.float64)
    Hs = np.load(work(c, "clean_homographies.npz"))["H"]
    clean80_raw = cv2.imread(str(work(c, "clean_plate_ref80.jpg")))
    clean80 = clean_residual_floor_shadow(clean80_raw)

    noise_tex = build_body_burn_noise(512, 512, seed=42)
    t0 = time.time()
    frame_batches = simulate_all_particles(c, cam_C, cam_R, f_px, p_depths, noise_tex, W, H, fps)
    print(f"Simulation finished in {time.time() - t0:.2f}s")

    gray0 = cv2.cvtColor(pl0, cv2.COLOR_BGR2GRAY).astype(np.float32) / 255.0
    black_floor = float(np.clip(np.percentile(gray0, 1.0) * 0.85, 0.008, 0.045))
    hp0 = gray0 - cv2.GaussianBlur(gray0, (0, 0), 1.2)
    grain_amt = float(np.clip(float(np.median(np.abs(hp0 - np.median(hp0)))) * 1.4826 * 1.10, 0.008, 0.022))

    ctx = moderngl.create_standalone_context(require=330)
    body_prog  = ctx.program(vertex_shader=VS_FSQ, fragment_shader=BODY_FS)
    flake_prog = ctx.program(vertex_shader=PART_VS, fragment_shader=FLAKE_FS)
    ember_prog = ctx.program(vertex_shader=PART_VS, fragment_shader=EMBER_FS)
    blur_prog  = ctx.program(vertex_shader=VS_FSQ, fragment_shader=BLUR_FS)
    comp_prog  = ctx.program(vertex_shader=VS_FSQ, fragment_shader=COMP_FS)

    fsq = ctx.buffer(np.array([-1, -1, 1, -1, -1, 1, -1, 1, 1, -1, 1, 1], dtype=np.float32).tobytes())
    va_body = ctx.vertex_array(body_prog, [(fsq, "2f", "p")])
    va_blur = ctx.vertex_array(blur_prog, [(fsq, "2f", "p")])
    va_comp = ctx.vertex_array(comp_prog, [(fsq, "2f", "p")])

    quad_vbo = ctx.buffer(np.array([-1, -1, 1, -1, -1, 1, -1, 1, 1, -1, 1, 1], dtype=np.float32).tobytes())
    max_inst = max(max(len(b) for b in frame_batches), 1)
    inst_buf = ctx.buffer(reserve=max_inst * 13 * 4)
    fmt = "3f 3f 3f 1f 1f 1f 1f/i"
    attrs = ("inPos", "inVel", "inRgb", "inTemp", "inSize", "inPhase", "inType")
    va_flake = ctx.vertex_array(flake_prog, [(quad_vbo, "2f", "inQuad"), (inst_buf, fmt, *attrs)])
    va_ember = ctx.vertex_array(ember_prog, [(quad_vbo, "2f", "inQuad"), (inst_buf, "3f 3f 12x 1f 1f 1f 1f/i", "inPos", "inVel", "inTemp", "inSize", "inPhase", "inType")])

    # Framebuffers & Textures
    t_body_col = ctx.texture((W, H), 4, dtype="f4")
    t_body_em  = ctx.texture((W, H), 4, dtype="f4")
    fbo_body   = ctx.framebuffer([t_body_col, t_body_em])

    t_scene = ctx.texture((W, H), 4, dtype="f4")
    t_emiss = ctx.texture((W, H), 4, dtype="f4")
    rbo_dep = ctx.depth_renderbuffer((W, H))
    fbo_scene = ctx.framebuffer([t_scene, t_emiss], depth_attachment=rbo_dep)

    hw, hh = W // 2, H // 2
    qw, qh = W // 4, H // 4
    t_b1a = ctx.texture((hw, hh), 4, dtype="f4"); t_b1b = ctx.texture((hw, hh), 4, dtype="f4")
    t_b2a = ctx.texture((qw, qh), 4, dtype="f4"); t_b2b = ctx.texture((qw, qh), 4, dtype="f4")
    for t in (t_body_col, t_body_em, t_scene, t_emiss, t_b1a, t_b1b, t_b2a, t_b2b):
        t.filter = (moderngl.LINEAR, moderngl.LINEAR); t.repeat_x = t.repeat_y = False
    fb_b1a, fb_b1b = ctx.framebuffer([t_b1a]), ctx.framebuffer([t_b1b])
    fb_b2a, fb_b2b = ctx.framebuffer([t_b2a]), ctx.framebuffer([t_b2b])

    t_final = ctx.texture((W, H), 4, dtype="f4")
    fbo_final = ctx.framebuffer([t_final])

    live_t  = ctx.texture((W, H), 3); live_t.filter = (moderngl.LINEAR, moderngl.LINEAR); live_t.repeat_x = live_t.repeat_y = False
    clean_t = ctx.texture((W, H), 3); clean_t.filter = (moderngl.LINEAR, moderngl.LINEAR); clean_t.repeat_x = clean_t.repeat_y = False
    mask_t  = ctx.texture((W, H), 1, dtype="f4"); mask_t.filter = (moderngl.LINEAR, moderngl.LINEAR); mask_t.repeat_x = mask_t.repeat_y = False
    rem_t   = ctx.texture((W, H), 1, dtype="f4"); rem_t.filter = (moderngl.LINEAR, moderngl.LINEAR); rem_t.repeat_x = rem_t.repeat_y = False
    burn_t  = ctx.texture((W, H), 1, dtype="f4"); burn_t.filter = (moderngl.LINEAR, moderngl.LINEAR); burn_t.repeat_x = burn_t.repeat_y = False

    body_prog["uRes"].value = (float(W), float(H))
    flake_prog["uRes"].value = (float(W), float(H)); flake_prog["uF"].value = f_px
    ember_prog["uRes"].value = (float(W), float(H)); ember_prog["uF"].value = f_px
    comp_prog["uRes"].value = (float(W), float(H))
    comp_prog["uBlackFloor"].value = black_floor; comp_prog["uGrainAmt"].value = grain_amt

    mode_map = {"disint": 0, "empty": 1, "reasm": 2, "solid": 3}

    def render_one(i):
        pl = frame(c, i)
        m_raw = cv2.imread(str(work(c, "mask", f"{i:04d}.png")), cv2.IMREAD_GRAYSCALE)
        m_u8 = cv2.GaussianBlur(m_raw, (0, 0), 1.0) if m_raw is not None else np.zeros((H, W), np.uint8)
        m_f = m_u8.astype(np.float32) / 255.0

        # Warp spotless clean plate into exact moving-camera frame via homography H_i + exposure border match
        clean_i = cv2.warpPerspective(clean80, Hs[i], (W, H), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
        rem_bin = cv2.dilate((m_u8 > 18).astype(np.uint8), cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (95, 95)))
        ys, xs = np.where(m_u8 > 40)
        if len(ys) > 0:
            rem_bin[max(0, int(ys.max()) - 95):min(H, int(ys.max()) + 100),
                    max(0, int(xs.min()) - 140):min(W, int(xs.max()) + 540)] = 1
        ring = (cv2.dilate(rem_bin, np.ones((55, 55), np.uint8)) - rem_bin) > 0
        if ring.sum() > 200:
            for ch in range(3):
                cv_m = float(clean_i[:, :, ch][ring].mean())
                lv_m = float(pl[:, :, ch][ring].mean())
                clean_i[:, :, ch] = np.clip(clean_i[:, :, ch].astype(np.float32) + (lv_m - cv_m), 0, 255).astype(np.uint8)
        rem_f = cv2.GaussianBlur(rem_bin.astype(np.float32), (0, 0), 22.0)

        mode_str, burn_p, sh_vis = timeline_state(i)
        if mode_str in ("disint", "reasm"):
            bf = eval_body_field(m_u8, noise_tex, mode=mode_str)
        else:
            bf = np.zeros((H, W), dtype=np.float32)

        live_t.write(pl[:, :, ::-1].tobytes())
        clean_t.write(clean_i[:, :, ::-1].tobytes())
        mask_t.write(m_f.astype(np.float32).tobytes())
        rem_t.write(rem_f.astype(np.float32).tobytes())
        burn_t.write(bf.astype(np.float32).tobytes())

        # Pass 1A: Save body alpha matte into fbo_body (for depth occlusion behind remaining solid body)
        fbo_body.use(); ctx.viewport = (0, 0, W, H); ctx.disable(moderngl.DEPTH_TEST); ctx.disable(moderngl.BLEND)
        live_t.use(0); clean_t.use(1); mask_t.use(2); rem_t.use(3); burn_t.use(4)
        body_prog["uLive"].value = 0; body_prog["uClean"].value = 1; body_prog["uMask"].value = 2
        body_prog["uRemZone"].value = 3; body_prog["uBurnField"].value = 4
        body_prog["uBurnProg"].value = float(burn_p)
        body_prog["uShadowVis"].value = float(sh_vis)
        body_prog["uTime"].value = float(i / fps)
        body_prog["uMode"].value = mode_map[mode_str]
        va_body.render(moderngl.TRIANGLES)

        # Pass 1B: Render live clean plate + burning body DIRECTLY into fbo_scene AFTER clearing depth!
        fbo_scene.use(); ctx.viewport = (0, 0, W, H); ctx.disable(moderngl.DEPTH_TEST); ctx.disable(moderngl.BLEND)
        fbo_scene.clear(0.0, 0.0, 0.0, 0.0, depth=1.0)
        va_body.render(moderngl.TRIANGLES)

        # Pass 2: 3D GPU-Instanced Ash/Cloth Flakes (depth-tested) + Velocity-Streaked Molten Embers (additive!)
        batch = frame_batches[i]
        if len(batch) > 0:
            inst_buf.write(batch.tobytes())
            ro = cam_C[i]; R = cam_R[i]
            Rt = tuple(R.T.flatten().astype(np.float32))
            p_dep = float(p_depths[i])

            # 2A: Solid tumbling cloth & charred ash flakes (alpha blend + depth write)
            ctx.enable(moderngl.DEPTH_TEST); ctx.depth_mask = True
            ctx.enable(moderngl.BLEND)
            ctx.blend_func = (moderngl.SRC_ALPHA, moderngl.ONE_MINUS_SRC_ALPHA)
            t_body_col.use(0)
            flake_prog["uPersonAlpha"].value = 0
            flake_prog["uPersonDepth"].value = p_dep
            flake_prog["uRo"].value = tuple(map(float, ro))
            flake_prog["uCam"].value = Rt
            va_flake.render(moderngl.TRIANGLES, instances=len(batch))

            # 2B: Blazing incandescent embers & motion-blurred sparks (additive HDR glow, depth test ON, mask OFF)
            ctx.depth_mask = False
            ctx.blend_func = (moderngl.ONE, moderngl.ONE)
            ember_prog["uPersonAlpha"].value = 0
            ember_prog["uPersonDepth"].value = p_dep
            ember_prog["uRo"].value = tuple(map(float, ro))
            ember_prog["uCam"].value = Rt
            va_ember.render(moderngl.TRIANGLES, instances=len(batch))
            ctx.depth_mask = True

        # Pass 3: Dual-Radius Separable Optical Bloom (exact 1:1 full-frame UV alignment across mip levels!)
        ctx.disable(moderngl.DEPTH_TEST); ctx.disable(moderngl.BLEND)
        # Tight bloom (half-res)
        fb_b1a.use(); ctx.viewport = (0, 0, hw, hh)
        t_emiss.use(0); blur_prog["uTex"].value = 0; blur_prog["uOutRes"].value = (float(hw), float(hh)); blur_prog["uDir"].value = (2.0 / hw, 0.0)
        va_blur.render(moderngl.TRIANGLES)
        fb_b1b.use(); ctx.viewport = (0, 0, hw, hh)
        t_b1a.use(0); blur_prog["uTex"].value = 0; blur_prog["uOutRes"].value = (float(hw), float(hh)); blur_prog["uDir"].value = (0.0, 2.0 / hh)
        va_blur.render(moderngl.TRIANGLES)
        # Wide glow bloom (quarter-res)
        fb_b2a.use(); ctx.viewport = (0, 0, qw, qh)
        t_b1b.use(0); blur_prog["uTex"].value = 0; blur_prog["uOutRes"].value = (float(qw), float(qh)); blur_prog["uDir"].value = (3.4 / qw, 0.0)
        va_blur.render(moderngl.TRIANGLES)
        fb_b2b.use(); ctx.viewport = (0, 0, qw, qh)
        t_b2a.use(0); blur_prog["uTex"].value = 0; blur_prog["uOutRes"].value = (float(qw), float(qh)); blur_prog["uDir"].value = (0.0, 3.4 / qh)
        va_blur.render(moderngl.TRIANGLES)

        # Pass 4: Final Optical Composite (preserves untouched raw plate exposure + HDR bloom + plate sensor grain)
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
            cv2.putText(b, f"disint f={idx}", (10, 344), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 0, 0), 3)
            cv2.putText(b, f"disint f={idx}", (10, 344), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (80, 195, 255), 1)
            rows.append(np.hstack([a, b]))
        out_p = wk / "still_disint_sheet.jpg"
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
                print(f"disintegration frame {i}/{N} ({time.time() - ts:.1f}s)", flush=True)
        out_dir = here / args.out; out_dir.mkdir(parents=True, exist_ok=True)
        r = f"{c['fps_num']}/{c['fps_den']}"; ff = find_ffmpeg()
        tags = ["-pix_fmt", "yuv420p", "-color_primaries", "bt709", "-color_trc", "bt709", "-colorspace", "bt709", "-movflags", "+faststart"]
        x264 = ["-c:v", "libx264", "-profile:v", "high", "-level", "4.1", "-crf", "14", "-preset", "slow"]
        run = lambda a: subprocess.run([ff, "-v", "error", "-y"] + a, check=True)
        run(["-framerate", r, "-i", str(comp_dir / "%04d.png")] + x264 + tags + [str(out_dir / "Particle_Disintegration_Reassembly_Composite.mp4")])
        run([
            "-framerate", r, "-i", str(comp_dir / "%04d.png"),
            "-framerate", r, "-i", str(work(c, "frames", "%04d.jpg")),
            "-filter_complex", "[0]scale=1080:608[a];[1]scale=1080:608[b];[a][b]vstack,pad=1080:1350:0:67:color=black",
        ] + x264 + tags + [str(out_dir / "Particle_Disintegration_Reassembly_Compare_1080x1350.mp4")])
        print("encoded ->", out_dir)


if __name__ == "__main__":
    main()
