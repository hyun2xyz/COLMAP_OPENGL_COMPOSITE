"""Effect #5 (New Cinema Night-Scene Edition): 3D Ember/Ash Particle Disintegration & Reverse Reassembly
on Pexels 11839713 (Foggy Snowy Night Walk — 100% Real Clean Plate + Conserved 3D Particle Physics).

    python render_disintegration_night.py --still 40,65,94,132,155,185   quick QA -> piece_night/work/still_night_sheet.jpg
    python render_disintegration_night.py --render                       full render -> out_night/Night_Walk_Particle_Disintegration_*.mp4

Cinematic movie-scene showcase (Pexels 11839713, 200f @ 25fps = 8.00s, locked tripod night walk):
- 100% REAL zero-AI temporal-median clean plate from the empty street tail (0.00px tripod error!).
- Hybrid clean-plate difference + MediaPipe silhouette matte with smooth 3D road perspective depth (9.5m -> 5.3m).
- ONE strictly conserved population (~26,000 3D particles) — ZERO mid-air particle spawns or dissolve-ins!
  1) f=22..80 : Fiery molten burn seam sweeps head->coat->boots, shedding 26,000 3D ash flakes & glowing
                velocity-streaked embers into the foggy night breeze toward the streetlamps.
  2) f=81..108: Eerily empty snowy road + suspended glowing ember storm curving smoothly in mid-air (C2 turnaround).
  3) f=109..168: All 26,000 embers rocket FORWARD in 3D depth toward his closer walking position and snap
                 pixel-exact into his reassembling foreground silhouette (boots->coat->head)!
  4) f=169..199: Clean, solid live-action movie climax stride toward camera (0 stray particles).
- Interactive fiery orange volumetric light spill onto the wet snowy asphalt & cold night fog + dual-radius
  1:1 optical bloom + matched night-cinema film grain. ZERO text/HUD overlays.
"""
import argparse, json, math, shutil, subprocess, sys, time
from pathlib import Path
import numpy as np, cv2, moderngl

here = Path(__file__).resolve().parent
ROOT = here.parents[1]
WORK = here / "piece_night" / "work"


def find_ffmpeg():
    tl = ROOT / "tools.local.json"
    if tl.exists():
        d = json.loads(tl.read_text(encoding="utf-8"))
        p = d.get("ffmpeg") or (d.get("tools") or {}).get("ffmpeg")
        if p and Path(p).exists():
            return str(p)
    return shutil.which("ffmpeg") or "ffmpeg"


def smoothstep_np(e0, e1, x):
    t = np.clip((x - e0) / max(e1 - e0, 1e-6), 0.0, 1.0)
    return t * t * (3.0 - 2.0 * t)


def smootherstep_np(e0, e1, x):
    t = np.clip((x - e0) / max(e1 - e0, 1e-6), 0.0, 1.0)
    return t * t * t * (t * (t * 6.0 - 15.0) + 10.0)


def curl_wind_3d(pos, t_val):
    """Exact divergence-free 3D analytical Curl Noise for sweeping foggy-night winter wind eddies."""
    x, y, z = pos[:, 0], pos[:, 1], pos[:, 2]
    k1 = 1.15
    ph1 = t_val * 0.65
    sx1, cx1 = np.sin(k1 * x + ph1), np.cos(k1 * x + ph1)
    sy1, cy1 = np.sin(k1 * y - ph1 * 0.6), np.cos(k1 * y - ph1 * 0.6)
    sz1, cz1 = np.sin(k1 * z + ph1 * 0.5), np.cos(k1 * z + ph1 * 0.5)
    vx1 = k1 * (sx1 * (-sy1) - (-sz1) * cx1)
    vy1 = k1 * (sy1 * (-sz1) - (-sx1) * cy1)
    vz1 = k1 * (sz1 * (-sx1) - (-sy1) * cz1)

    k2 = 2.85
    ph2 = t_val * 1.25 + 1.9
    sx2, cx2 = np.sin(k2 * x - ph2), np.cos(k2 * x - ph2)
    sy2, cy2 = np.sin(k2 * y + ph2 * 0.8), np.cos(k2 * y + ph2 * 0.8)
    sz2, cz2 = np.sin(k2 * z - ph2 * 0.7), np.cos(k2 * z - ph2 * 0.7)
    vx2 = 0.44 * k2 * (sx2 * (-sy2) - (-sz2) * cx2)
    vy2 = 0.44 * k2 * (sy2 * (-sz2) - (-sx2) * cy2)
    vz2 = 0.44 * k2 * (sz2 * (-sx2) - (-sy2) * cz2)

    return np.stack([vx1 + vx2, vy1 + vy2, vz1 + vz2], axis=1) * 0.88


def build_body_burn_noise(H, W, seed=77):
    rng = np.random.default_rng(seed)
    acc = np.zeros((H, W), dtype=np.float32)
    amp = 0.52
    for sz in (12, 26, 56, 124):
        g = rng.random((sz, sz), dtype=np.float32)
        up = cv2.resize(g, (W, H), interpolation=cv2.INTER_CUBIC)
        acc += amp * up
        amp *= 0.50
    acc = (acc - acc.min()) / max(acc.max() - acc.min(), 1e-5)
    return acc.astype(np.float32)


def compute_smooth_boxes(bboxes):
    """Fit jitter-free quadratic trajectories to person bounding boxes (rejecting +/-57px boot-shadow spikes!)."""
    N = len(bboxes)
    ts = np.arange(N, dtype=np.float64)
    out = np.zeros_like(bboxes, dtype=np.float64)
    for col in range(4):
        raw = bboxes[:, col].copy()
        # Running median window (15 frames) kills transient boot-shadow flicker spikes before polynomial fit
        med = np.array([np.median(raw[max(0, i - 7):min(N, i + 8)]) for i in range(N)], dtype=np.float64)
        poly = np.poly1d(np.polyfit(ts, med, deg=2))
        out[:, col] = poly(ts)
    return out


def eval_body_field(smooth_box, noise_tex, H, W, mode="disint"):
    """Temporally butter-smooth body-locked scalar burn order S in [0, 1] (zero frame-to-frame jump!)."""
    x_min, y_min, x_max, y_max = [float(v) for v in smooth_box]
    bh = max(y_max - y_min, 40.0)
    bw = max(x_max - x_min, 40.0)

    yy = np.linspace(0, H - 1, H, dtype=np.float32)[:, None]
    xx = np.linspace(0, W - 1, W, dtype=np.float32)[None, :]
    v = np.clip((yy - y_min) / bh, -0.25, 1.25)
    u = np.clip((xx - x_min) / bw, -0.25, 1.25)

    nh, nw = noise_tex.shape
    ix = np.clip((u[0, :] * (nw - 1)).astype(np.int32), 0, nw - 1)
    iy = np.clip((v[:, 0] * (nh - 1)).astype(np.int32), 0, nh - 1)
    n_body = noise_tex[iy[:, None], ix[None, :]]

    if mode == "disint":
        # Sweeps smoothly head & shoulders -> winter coat -> boots
        base = 0.86 * v + 0.14 * u
    else:
        # Sweeps smoothly boots -> legs -> winter coat -> head & shoulders
        base = 0.88 * (1.0 - v) + 0.12 * (1.0 - u)

    raw = base + 0.16 * (n_body - 0.5)
    # Fixed deterministic range [-0.04, 1.04] -> ZERO per-frame percentile jump!
    field = np.clip((raw + 0.04) / 1.08, 0.0, 1.0)
    return field.astype(np.float32)


def timeline_state(i):
    """Master 200-frame (8.00s @ 25fps) cinema arc:
      - f=0..21  : moody night walk establishing shot (subtle spark pre-ignition f=18..21)
      - f=22..80 : full fiery head-to-boot 3D disintegration (2.32s)
      - f=81..108: empty foggy snowy street + suspended 3D ember turnaround (1.12s)
      - f=109..168: dramatic closer foreground boot-to-head reverse reassembly (2.40s)
      - f=169..199: pure solid live-action movie finale stride (1.24s, 0 stray particles!)
    """
    if i < 18:
        return "solid", -0.15
    elif i < 22:
        pre = ((i - 18) / 4.0) * 0.06
        return "disint", -0.06 + pre
    elif i <= 80:
        u = (i - 22) / 58.0
        p = -0.02 + 1.16 * (u * u * (2.5 - 1.5 * u))
        return "disint", float(p)
    elif i < 109:
        return "empty", 1.25
    elif i <= 168:
        u = (i - 109) / 59.0
        p = -0.02 + 1.16 * (u * u * (2.5 - 1.5 * u))
        return "reasm", float(p)
    else:
        return "solid", 1.25


def simulate_conserved_particles(cam_C, cam_R, f_px, depths, smooth_boxes, noise_tex, W, H, fps):
    """Simulate ONE strictly conserved population (~25,000 3D particles) over 200 frames:
    - ZERO mid-air particle spawns or dissolve-ins!
    - Temporally smooth body burn field -> zero frame-to-frame burn seam stutter!
    - 3D spatially-matched C2 turnaround -> butter-smooth mid-air curve!
    """
    N = len(cam_C)
    dt = 1.0 / fps
    rng = np.random.default_rng(707)

    def unproject(xs, ys, i, z_jit):
        ro = cam_C[i]; R = cam_R[i]
        d_cam = depths[i] + z_jit
        xc = (xs - 0.5 * W) / f_px * d_cam
        yc = (ys - 0.5 * H) / f_px * d_cam
        zc = d_cam
        q = np.stack([xc, yc, zc], axis=1)
        return ro[None, :] + q @ R.T

    # 1. Sample exact disintegration births on his body (f=22..80) using smooth_boxes!
    d_pos_l, d_rgb_l, d_sp_l = [], [], []
    prev_p = -0.06
    for i in range(22, 81):
        _, cur_p = timeline_state(i)
        m_u8 = cv2.imread(str(WORK / "mask" / f"{i:04d}.png"), cv2.IMREAD_GRAYSCALE)
        if m_u8 is None:
            prev_p = cur_p
            continue
        fld = eval_body_field(smooth_boxes[i], noise_tex, H, W, mode="disint")
        band = (m_u8 > 110) & (fld >= prev_p - 0.012) & (fld <= cur_p + 0.014)
        ys, xs = np.where(band)
        prev_p = cur_p
        if len(xs) == 0:
            continue
        k = min(len(xs), 480)
        sel = rng.choice(len(xs), size=k, replace=False)
        sx = xs[sel].astype(np.float64) + rng.uniform(-0.5, 0.5, k)
        sy = ys[sel].astype(np.float64) + rng.uniform(-0.5, 0.5, k)
        im = cv2.imread(str(WORK / "frames" / f"{i:04d}.jpg"))
        rgb = im[ys[sel], xs[sel], ::-1].astype(np.float32) / 255.0
        pos3 = unproject(sx, sy, i, rng.normal(0.0, 0.09, k))
        d_pos_l.append(pos3.astype(np.float32))
        d_rgb_l.append(rgb)
        d_sp_l.append(np.full(k, i, dtype=np.int32))

    d_pos0 = np.concatenate(d_pos_l, axis=0)
    d_rgb0 = np.concatenate(d_rgb_l, axis=0)
    d_spawn = np.concatenate(d_sp_l, axis=0)
    M = len(d_spawn)

    # 2. Sample exact reassembly targets on his CLOSER walking body (f=109..168) using smooth_boxes!
    r_pos_l, r_rgb_l, r_arr_l = [], [], []
    prev_p = -0.05
    for i in range(109, 169):
        _, cur_p = timeline_state(i)
        m_u8 = cv2.imread(str(WORK / "mask" / f"{i:04d}.png"), cv2.IMREAD_GRAYSCALE)
        if m_u8 is None:
            prev_p = cur_p
            continue
        fld = eval_body_field(smooth_boxes[i], noise_tex, H, W, mode="reasm")
        band = (m_u8 > 110) & (fld >= prev_p - 0.012) & (fld <= cur_p + 0.014)
        ys, xs = np.where(band)
        prev_p = cur_p
        if len(xs) == 0:
            continue
        k = min(len(xs), 480)
        sel = rng.choice(len(xs), size=k, replace=False)
        sx = xs[sel].astype(np.float64) + rng.uniform(-0.5, 0.5, k)
        sy = ys[sel].astype(np.float64) + rng.uniform(-0.5, 0.5, k)
        im = cv2.imread(str(WORK / "frames" / f"{i:04d}.jpg"))
        rgb = im[ys[sel], xs[sel], ::-1].astype(np.float32) / 255.0
        pos3 = unproject(sx, sy, i, rng.normal(0.0, 0.08, k))
        r_pos_l.append(pos3.astype(np.float32))
        r_rgb_l.append(rgb)
        r_arr_l.append(np.full(k, i, dtype=np.int32))

    r_pos_all = np.concatenate(r_pos_l, axis=0)
    r_rgb_all = np.concatenate(r_rgb_l, axis=0)
    r_arr_all = np.concatenate(r_arr_l, axis=0)
    pick = rng.choice(len(r_arr_all), size=M, replace=(len(r_arr_all) < M))
    r_pos1 = r_pos_all[pick]
    r_rgb1 = r_rgb_all[pick]
    r_arr  = r_arr_all[pick]

    # Particle personalities: 73% dark charred ash flakes, 27% razor-crisp molten orange-gold embers/sparks
    ptype = (rng.random(M) < 0.27).astype(np.float32)
    # Fine Houdini-grade particle scale (crisp 1.2..6.5px on 1080p -> silky wispy ribbons, never dense yellow wall!)
    psize = np.where(ptype > 0.5, rng.uniform(0.0046, 0.0128, M), rng.uniform(0.0070, 0.0175, M)).astype(np.float32)
    phase = rng.uniform(0.0, 6.28318, M).astype(np.float32)
    drag  = rng.uniform(0.50, 1.55, M).astype(np.float32)
    cool  = rng.uniform(0.65, 1.55, M).astype(np.float32)
    wmul  = rng.uniform(0.42, 1.78, M).astype(np.float32)

    # 3. Integrate Forward Branch (f=22..112) & Backward Branch (f=72..168)
    #    Notice: wind blows slightly LEFT-UPWARD (-X, +Z) across the dark dark snowy forest/fog,
    #    creating MAXIMUM contrast against the deep dark night left-center AND glowing lamps on the right!
    f_lo, f_hi = 22, 112
    b_lo, b_hi = 72, 168
    P_fwd = np.zeros((f_hi - f_lo + 1, M, 3), dtype=np.float32)
    cur_p = d_pos0.copy()
    cur_v = np.stack([
        rng.uniform(-1.10, 0.45, M),
        rng.uniform(-0.95, 0.15, M),  # gently drifts slightly forward along the snowy road toward camera!
        rng.uniform( 0.10, 0.95, M),
    ], axis=1).astype(np.float32)

    for i in range(f_lo, f_hi + 1):
        P_fwd[i - f_lo] = cur_p
        moving = (d_spawn <= i)
        if np.any(moving):
            m_i = np.where(moving)[0]
            p_m = cur_p[m_i]; v_m = cur_v[m_i]
            cw = curl_wind_3d(p_m, i * dt)
            hang = 1.0 - 0.80 * float(smoothstep_np(76.0, 94.0, float(i)))
            # Sweeps dramatically slightly left-upward & forward through the misty night air!
            wind_tgt = (np.array([-0.52, -0.55, 0.48], dtype=np.float32)[None, :] * wmul[m_i, None]) * hang + cw * (1.35 * hang + 0.35)
            v_m = v_m + (wind_tgt - v_m) * (dt * 1.85 * drag[m_i, None])
            v_m *= (0.94 + 0.06 * hang)
            p_m = p_m + v_m * dt
            p_m[:, 1] = np.clip(p_m[:, 1], 2.8, 13.5)
            below = p_m[:, 2] < 0.08
            if np.any(below):
                p_m[below, 2] = 0.08
                v_m[below, 2] = np.abs(v_m[below, 2]) * 0.45 + 0.22
            cur_p[m_i] = p_m; cur_v[m_i] = v_m

    P_bwd = np.zeros((b_hi - b_lo + 1, M, 3), dtype=np.float32)
    cur_p = r_pos1.copy()
    cur_v = np.stack([
        rng.uniform(-0.35, 1.15, M),
        rng.uniform(-0.85, 0.25, M),
        rng.uniform(-0.95, 0.05, M),
    ], axis=1).astype(np.float32)

    for i in range(b_hi, b_lo - 1, -1):
        P_bwd[i - b_lo] = cur_p
        active = (r_arr >= i)
        if np.any(active):
            s_i = np.where(active)[0]
            p_s = cur_p[s_i]; v_s = cur_v[s_i]
            cw = curl_wind_3d(p_s, i * dt + 1.5)
            pull = 0.22 + 0.78 * float(smoothstep_np(92.0, 114.0, float(i)))
            # Forward velocity during reassembly sweeps right-downward & forward from upper-left fog into his body!
            fwd_tgt = (np.array([0.55, -0.45, -0.46], dtype=np.float32)[None, :] * wmul[s_i, None]) * pull + cw * (1.30 * pull + 0.35)
            p_s = p_s - v_s * dt
            v_s = v_s + (fwd_tgt - v_s) * (dt * 1.85 * drag[s_i, None])
            p_s[:, 1] = np.clip(p_s[:, 1], 2.8, 13.5)
            p_s[:, 2] = np.clip(p_s[:, 2], 0.14, 3.60)
            cur_p[s_i] = p_s; cur_v[s_i] = v_s

    # 4. 3D Spatial Corridor Pairing at Mid-Air Turnaround (f=94) -> ultra-smooth low-jerk U-turns!
    f_mid = 94
    pf_mid = P_fwd[f_mid - f_lo]
    pb_mid = P_bwd[f_mid - b_lo]
    def spatial_order(pts):
        z_bin = (np.argsort(np.argsort(pts[:, 2])) * 10) // len(pts)
        x_bin = (np.argsort(np.argsort(pts[:, 0])) * 8) // len(pts)
        return np.lexsort((pts[:, 1], x_bin, z_bin))

    ord_f = spatial_order(pf_mid)
    ord_b = spatial_order(pb_mid)
    pair_b = np.empty(M, dtype=np.int32)
    pair_b[ord_f] = ord_b
    P_bwd = P_bwd[:, pair_b, :]
    r_rgb1 = r_rgb1[pair_b]
    r_arr  = r_arr[pair_b]

    # 5. Evaluate Single Conserved Continuous Trajectory P(i) (f=22..168, 0 particles outside!)
    frame_batches = [np.zeros((0, 13), dtype=np.float32) for _ in range(N)]
    glow_centers  = np.zeros((N, 4), dtype=np.float32)  # (cx_px, cy_px, radius_px, intensity) for snowy-road fire spill!
    prev_pos = d_pos0.copy()
    max_k = 0

    for i in range(22, 169):
        alive = (d_spawn <= i) & (i <= r_arr)
        if not np.any(alive):
            continue
        idx = np.where(alive)[0]
        w = float(smootherstep_np(72.0, 112.0, float(i)))
        if i <= 72:
            pos_i = P_fwd[i - f_lo, idx]
        elif i >= 112:
            pos_i = P_bwd[i - b_lo, idx]
        else:
            pos_i = (1.0 - w) * P_fwd[i - f_lo, idx] + w * P_bwd[i - b_lo, idx]

        newly_born = (d_spawn[idx] == i)
        vel_i = (pos_i - prev_pos[idx]) / dt
        if np.any(newly_born):
            vel_i[newly_born] = np.array([-0.25, -0.35, 0.35], dtype=np.float32)
        prev_pos[idx] = pos_i

        age_b = (i - d_spawn[idx]).astype(np.float32) * dt
        rem_a = (r_arr[idx] - i).astype(np.float32) * dt
        rgb_i = (1.0 - w) * d_rgb0[idx] + w * r_rgb1[idx]

        # Rich fiery thermal curve: blazing at burn/reassembly seams, warm glowing amber in dark night fog!
        heat_b = np.exp(-age_b * cool[idx] * np.where(ptype[idx] > 0.5, 0.48, 1.80))
        heat_s = np.exp(-rem_a * np.where(ptype[idx] > 0.5, 0.56, 2.00)) * float(smoothstep_np(92.0, 116.0, float(i)))
        ambient_ember = 0.34 * ptype[idx]
        temp_i = np.clip(np.maximum(heat_b, heat_s * 1.18) + ambient_ember, 0.0, 1.35)

        end_ramp = np.clip(np.minimum(age_b, rem_a) / 0.06 + 0.42, 0.42, 1.0)
        sz_i = psize[idx] * end_ramp
        ph_i = phase[idx] + age_b * 4.4

        batch = np.column_stack([
            pos_i, vel_i, rgb_i,
            temp_i.astype(np.float32),
            sz_i.astype(np.float32),
            ph_i.astype(np.float32),
            ptype[idx],
        ]).astype(np.float32)
        frame_batches[i] = batch
        max_k = max(max_k, len(batch))

        # Project centroid of hottest embers to drive REAL warm fiery light spill onto the wet snowy road & night fog!
        hot_w = (temp_i ** 2) + 1e-4
        c3 = (pos_i * hot_w[:, None]).sum(axis=0) / hot_w.sum()
        q = cam_R[i].T @ (c3 - cam_C[i])
        if q[2] > 0.5:
            cx_px = float(f_px * q[0] / q[2] + 0.5 * W)
            cy_px = float(f_px * q[1] / q[2] + 0.5 * H)
            rad_px = float(np.clip(f_px * 1.85 / q[2], 180.0, 520.0))
            inten = float(np.clip(len(idx) / 14000.0, 0.0, 1.0) * (0.45 + 0.55 * float(temp_i.mean())))
            glow_centers[i] = [cx_px, cy_px, rad_px, inten]

    # Smooth glow center envelope temporally for silky cinema lighting
    for ch in range(4):
        glow_centers[:, ch] = cv2.GaussianBlur(glow_centers[:, ch:ch + 1], (1, 9), 2.2)[:, 0]

    print(f"Conserved Night-Walk 3D Particle Sim: M={M} particles (0 mid-air spawns!) | peak active/frame={max_k}")
    return frame_batches, glow_centers


# ==============================================================================
#  SHADERS: PASS 1 (CLEAN PLATE + FIERY BURN FRONT + WET SNOW/FOG ORANGE LIGHT SPILL),
#           PASS 2 (3D INSTANCED ASH FLAKES + MOTION-STREAKED EMBERS),
#           PASS 3 (DUAL-RADIUS 1:1 OPTICAL BLOOM + NIGHT FILM GRAIN)
# ==============================================================================

VS_FSQ = "#version 330\nin vec2 p; void main(){ gl_Position = vec4(p, 0.0, 1.0); }"

BODY_FS = """#version 330
uniform sampler2D uLive, uClean, uMask, uBurnField;
uniform vec2 uRes;
uniform float uBurnProg, uTime;
uniform int uMode;       /* 0=disint, 1=empty, 2=reasm, 3=solid */
uniform vec4 uGlow;      /* xy=fire centroid px, z=radius px, w=intensity (0..1) */

layout(location=0) out vec4 oScene;
layout(location=1) out vec4 oEmiss;

float hash21(vec2 p){ vec3 q=fract(vec3(p.xyx)*0.1031); q+=dot(q,q.yzx+33.33); return fract((q.x+q.y)*q.z); }

void main() {
  vec2 px = vec2(gl_FragCoord.x, uRes.y - gl_FragCoord.y);
  vec2 uv = clamp(px / uRes, vec2(0.001), vec2(0.999));

  vec3 liveLin  = pow(max(texture(uLive, uv).rgb, vec3(0.0004)), vec3(2.2));
  vec3 cleanLin = pow(max(texture(uClean, uv).rgb, vec3(0.0004)), vec3(2.2));
  float m       = texture(uMask, uv).r;
  float bf      = texture(uBurnField, uv).r;

  /* Interactive warm fiery orange light spill onto wet snowy asphalt & cold night fog! */
  vec3 nightSpill = vec3(0.0);
  if (uGlow.w > 0.005) {
    /* 1. Soft atmospheric fog scatter around burning silhouette / ember cloud */
    vec2 dg = (px - uGlow.xy) / max(uGlow.z, 120.0);
    float fogGlow = exp(-dot(dg, dg) * 1.15) * uGlow.w * 0.11;
    /* 2. Ground bounce onto wet snowy sidewalk & crosswalk directly beneath fire (y > 680px)! */
    float roadMask = smoothstep(580.0, 840.0, px.y);
    vec2 dr = vec2((px.x - uGlow.x) / (uGlow.z * 1.35), (px.y - max(uGlow.y + 180.0, 860.0)) / 210.0);
    float roadLit = exp(-dot(dr, dr) * 1.35) * roadMask * uGlow.w;
    float roadSpec = dot(cleanLin, vec3(0.3, 0.5, 0.2)) * 2.4 + 0.08;
    float flicker = 0.94 + 0.06 * sin(uTime * 17.0 + sin(uTime * 9.3));
    nightSpill = vec3(1.00, 0.36, 0.06) * (fogGlow + roadLit * roadSpec * 0.45) * flicker;
  }

  if (uMode == 1) {
    /* 100% spotless real empty foggy snowy night street + warm ember light spill! */
    oScene = vec4(cleanLin + nightSpill, 0.0);
    oEmiss = vec4(0.0);
    return;
  }
  if (uMode == 3) {
    oScene = vec4(liveLin + nightSpill, m);
    oEmiss = vec4(0.0);
    return;
  }

  float dEdge = (uMode == 0) ? (bf - uBurnProg) : (uBurnProg - bf);
  float bodyKeep = smoothstep(-0.006, 0.015, dEdge);
  float personAlpha = m * bodyKeep;

  /* Eaten pixels cleanly reveal the 100%-real locked-tripod empty night street! */
  float personWipe = smoothstep(0.01, 0.20, m) * (1.0 - bodyKeep);
  vec3 baseLin = mix(liveLin, cleanLin, personWipe) + nightSpill;

  /* Charred dark ember rim eating into his winter coat right ahead of the flame seam */
  float charBand = smoothstep(0.080, 0.012, dEdge) * smoothstep(-0.010, 0.012, dEdge) * m;
  vec3 liveCharred = mix(liveLin + nightSpill * 0.45, liveLin * vec3(0.18, 0.08, 0.04), charBand * 0.75);
  vec3 col = mix(baseLin, liveCharred, personAlpha);

  /* Scorching movie-grade incandescent molten orange-white ember rim along the active burn edge! */
  float flk = 0.84 + 0.32 * hash21(floor(px * 0.35) + vec2(uTime * 21.0, -uTime * 15.0));
  float hotCore = exp(-pow((dEdge - 0.008) / 0.013, 2.0)) * smoothstep(0.06, 0.35, m) * flk;
  float hotHalo = exp(-pow((dEdge - 0.016) / 0.040, 2.0)) * smoothstep(0.03, 0.28, m);

  float seamEnv = smoothstep(-0.05, 0.015, uBurnProg) * smoothstep(1.15, 1.02, uBurnProg);
  vec3 seamCore = (uMode == 0) ? vec3(4.6, 1.95, 0.38) : vec3(5.0, 2.55, 0.68);
  vec3 seamHalo = (uMode == 0) ? vec3(2.4, 0.45, 0.03) : vec3(2.6, 0.68, 0.08);
  vec3 emiss = (seamCore * hotCore + seamHalo * hotHalo * 0.90) * seamEnv;

  oScene = vec4(col + emiss * 0.58, personAlpha);
  oEmiss = vec4(emiss, 1.0);
}
"""

PART_VS = """#version 330
in vec2 inQuad;
in vec3 inPos;
in vec3 inVel;
in vec3 inRgb;
in float inTemp;
in float inSize;
in float inPhase;
in float inType;

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
  vec3 q = transpose(uCam) * (inPos - uRo);
  if (q.z < 0.35) {
    gl_Position = vec4(-10.0, -10.0, -10.0, 1.0);
    return;
  }
  vec2 px0 = uF * q.xy / q.z + 0.5 * uRes;
  vec3 qPrev = transpose(uCam) * (inPos - inVel * 0.034 - uRo);
  vec2 pxPrev = uF * qPrev.xy / max(qPrev.z, 0.35) + 0.5 * uRes;
  vec2 velPx = px0 - pxPrev;
  float spdPx = length(velPx);

  float radPx = clamp(uF * inSize / q.z, 1.15, 9.5);

  vec2 axisX, axisY;
  if (inType > 0.5 && spdPx > 1.0) {
    vec2 dir = velPx / spdPx;
    vec2 perp = vec2(-dir.y, dir.x);
    float stretch = min(spdPx * 0.65 + radPx, radPx * 4.8);
    axisX = dir * stretch;
    axisY = perp * (radPx * 0.54);
  } else {
    float ca = cos(inPhase * 0.7), sa = sin(inPhase * 0.7);
    float tumble = 0.34 + 0.66 * abs(cos(inPhase * 1.35));
    axisX = vec2(ca, sa) * radPx;
    axisY = vec2(-sa, ca) * (radPx * tumble);
  }

  vec2 px = px0 + axisX * inQuad.x + axisY * inQuad.y;
  vec2 ndc = vec2(px.x / uRes.x * 2.0 - 1.0, (1.0 - px.y / uRes.y) * 2.0 - 1.0);
  float zNdc = clamp((q.z - 0.35) / 25.0, 0.0, 0.999) * 2.0 - 1.0;
  gl_Position = vec4(ndc, zNdc, 1.0);

  vLocal = inQuad;
  vRgb   = inRgb;
  vTemp  = inTemp;
  vPhase = inPhase;
  vType  = inType;
  vDepth = q.z;
  vLight = 0.55 + 0.65 * abs(sin(inPhase * 1.35 + 0.5));
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
  if (vType > 0.5) discard;
  vec2 uvScr = gl_FragCoord.xy / uRes;
  float pA = texture(uPersonAlpha, uvScr).a;
  if (vDepth > uPersonDepth + 0.08 && pA > 0.45) discard;

  float ang = atan(vLocal.y, vLocal.x);
  float r = length(vLocal);
  float jag = 0.66 + 0.22 * cos(ang * 3.0 + vPhase * 2.1) + 0.14 * sin(ang * 5.0 - vPhase * 3.7);
  float a = smoothstep(jag, jag - 0.22, r);
  if (a < 0.05) discard;

  /* Dark winter coat / charred ash flake with warm streetlamp & ember rim catch so flakes read cleanly in dark fog! */
  vec3 liveLin = pow(max(vRgb, vec3(0.004)), vec3(2.2));
  vec3 ashCol = mix(liveLin * 0.55, vec3(0.045, 0.042, 0.036), 0.55);
  vec3 lampCatch = vec3(0.14, 0.13, 0.09) * smoothstep(0.4, 1.1, vLight) * 0.45;
  vec3 bodyLin = ashCol * vLight + lampCatch;

  float edgeHot = smoothstep(jag * 0.28, jag * 0.92, r) * smoothstep(0.14, 0.92, vTemp);
  vec3 emberCol = mix(vec3(1.85, 0.32, 0.02), vec3(3.8, 1.55, 0.26), clamp(vTemp, 0.0, 1.0)) * edgeHot;

  oScene = vec4(bodyLin + emberCol * 0.72, a * 0.94);
  oEmiss = vec4(emberCol * a * 0.85, a);
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
  if (vType <= 0.5) discard;
  vec2 uvScr = gl_FragCoord.xy / uRes;
  float pA = texture(uPersonAlpha, uvScr).a;
  if (vDepth > uPersonDepth + 0.08 && pA > 0.55) discard;

  float r2 = dot(vLocal, vLocal);
  if (r2 > 1.0) discard;
  float core = exp(-r2 * 5.0);
  float glow = pow(max(1.0 - r2, 0.0), 2.2);

  float twk = 0.72 + 0.42 * sin(vPhase * 3.1);
  float tHot = clamp(vTemp, 0.0, 1.35);

  /* Deep cinematic molten orange-red-gold fire palette (rich cinema contrast vs foggy night, never pale yellow!) */
  vec3 cRuby   = vec3(0.75, 0.06, 0.005);
  vec3 cOrange = vec3(1.68, 0.32, 0.015);
  vec3 cGold   = vec3(2.45, 1.05, 0.16);
  vec3 cWhite  = vec3(3.25, 2.35, 1.00);
  vec3 col = mix(cRuby, cOrange, smoothstep(0.05, 0.55, tHot));
  col = mix(col, cGold, smoothstep(0.55, 1.05, tHot));
  col = mix(col, cWhite, core * smoothstep(0.55, 1.25, tHot));

  vec3 outEm = col * (core * 0.88 + glow * 0.24) * twk;
  oScene = vec4(outEm * 0.48, 1.0);
  oEmiss = vec4(outEm * 0.72, 1.0);
}
"""

BLUR_FS = """#version 330
uniform sampler2D uTex;
uniform vec2 uOutRes;
uniform vec2 uDir;
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
  vec3 bloom = b1 * 0.60 + b2 * 0.48;

  /* Preserve 100% exact dark night-cinema exposure on untouched pixels; soft HDR rolloff ONLY on bright fire/bloom! */
  vec3 baseClamped = min(sceneLin, vec3(1.0));
  vec3 hdrExcess = max(sceneLin - vec3(1.0), vec3(0.0)) + bloom;
  vec3 emSoft = hdrExcess / (1.0 + hdrExcess * 0.24);
  vec3 totalLin = baseClamped + emSoft * (1.0 - baseClamped * 0.18);

  vec3 col = pow(clamp(totalLin, 0.0, 1.0), vec3(1.0 / 2.2));
  float act = clamp(length(bloom) * 2.5 + length(hdrExcess) * 2.0, 0.0, 1.0);
  if (act > 0.01) {
    vec3 graded = max(col, vec3(uBlackFloor));
    vec2 p1 = gl_FragCoord.xy + vec2(uFrame * 17.3, uFrame * 31.7);
    vec2 p2 = floor(gl_FragCoord.xy * 0.5) + vec2(uFrame * 11.1, uFrame * 23.9);
    float g = (hash21(p1)+hash21(p1+7.13)-1.0)*0.62 + (hash21(p2)+hash21(p2+5.71)-1.0)*0.38;
    float lum = dot(graded, vec3(0.299, 0.587, 0.114));
    graded += vec3(g) * (uGrainAmt * (1.12 - 0.55 * smoothstep(0.55, 0.98, lum)));
    col = mix(col, graded, act);
  }
  oCol = vec4(clamp(col, 0.0, 1.0), 1.0);
}
"""


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--still", default=None)
    ap.add_argument("--render", action="store_true")
    ap.add_argument("--passes", action="store_true")
    ap.add_argument("--out", default="out_night")
    args = ap.parse_args()

    meta = json.loads((WORK / "meta.json").read_text())
    S    = json.loads((WORK / "scene.json").read_text())
    T    = np.load(WORK / "track.npz")
    W, H, N = int(meta["W"]), int(meta["H"]), int(meta["N"])
    fps = float(meta["fps"])
    f_px = float(S["f"])
    cam_C = np.asarray(S["cam"], dtype=np.float64)
    cam_R = np.asarray(S["rot"], dtype=np.float64)
    depths = np.asarray(T["depth"], dtype=np.float64)
    smooth_boxes = compute_smooth_boxes(np.asarray(T["bboxes"], dtype=np.float64))

    clean_bgr = cv2.imread(str(WORK / "clean_plate.png"))
    pl0 = cv2.imread(str(WORK / "frames" / "0000.jpg"))
    noise_tex = build_body_burn_noise(512, 512, seed=77)

    t0 = time.time()
    frame_batches, glow_centers = simulate_conserved_particles(cam_C, cam_R, f_px, depths, smooth_boxes, noise_tex, W, H, fps)
    print(f"Night sim finished in {time.time() - t0:.2f}s")

    gray0 = cv2.cvtColor(pl0, cv2.COLOR_BGR2GRAY).astype(np.float32) / 255.0
    black_floor = float(np.clip(np.percentile(gray0, 1.0) * 0.85, 0.008, 0.055))
    hp0 = gray0 - cv2.GaussianBlur(gray0, (0, 0), 1.2)
    grain_amt = float(np.clip(float(np.median(np.abs(hp0 - np.median(hp0)))) * 1.4826 * 1.15, 0.008, 0.026))

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
    va_flake = ctx.vertex_array(flake_prog, [(quad_vbo, "2f", "inQuad"), (inst_buf, "3f 3f 3f 1f 1f 1f 1f/i", "inPos", "inVel", "inRgb", "inTemp", "inSize", "inPhase", "inType")])
    va_ember = ctx.vertex_array(ember_prog, [(quad_vbo, "2f", "inQuad"), (inst_buf, "3f 3f 12x 1f 1f 1f 1f/i", "inPos", "inVel", "inTemp", "inSize", "inPhase", "inType")])

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
    burn_t  = ctx.texture((W, H), 1, dtype="f4"); burn_t.filter = (moderngl.LINEAR, moderngl.LINEAR); burn_t.repeat_x = burn_t.repeat_y = False
    clean_t.write(clean_bgr[:, :, ::-1].tobytes())

    body_prog["uRes"].value = (float(W), float(H))
    flake_prog["uRes"].value = (float(W), float(H)); flake_prog["uF"].value = f_px
    ember_prog["uRes"].value = (float(W), float(H)); ember_prog["uF"].value = f_px
    comp_prog["uRes"].value = (float(W), float(H))
    comp_prog["uBlackFloor"].value = black_floor; comp_prog["uGrainAmt"].value = grain_amt

    mode_map = {"disint": 0, "empty": 1, "reasm": 2, "solid": 3}

    def render_one(i):
        pl = cv2.imread(str(WORK / "frames" / f"{i:04d}.jpg"))
        m_u8 = cv2.imread(str(WORK / "mask" / f"{i:04d}.png"), cv2.IMREAD_GRAYSCALE)
        m_f = m_u8.astype(np.float32) / 255.0 if m_u8 is not None else np.zeros((H, W), np.float32)

        mode_str, burn_p = timeline_state(i)
        bf = eval_body_field(smooth_boxes[i], noise_tex, H, W, mode=mode_str) if mode_str in ("disint", "reasm") else np.zeros((H, W), np.float32)

        live_t.write(pl[:, :, ::-1].tobytes())
        mask_t.write(m_f.astype(np.float32).tobytes())
        burn_t.write(bf.astype(np.float32).tobytes())

        # Pass 1A & 1B: Clean Plate + Fiery Burn Front + Interactive Wet Snowy-Road Orange Light Spill!
        fbo_body.use(); ctx.viewport = (0, 0, W, H); ctx.disable(moderngl.DEPTH_TEST); ctx.disable(moderngl.BLEND)
        live_t.use(0); clean_t.use(1); mask_t.use(2); burn_t.use(3)
        body_prog["uLive"].value = 0; body_prog["uClean"].value = 1; body_prog["uMask"].value = 2; body_prog["uBurnField"].value = 3
        body_prog["uBurnProg"].value = float(burn_p)
        body_prog["uTime"].value = float(i / fps)
        body_prog["uMode"].value = mode_map[mode_str]
        body_prog["uGlow"].value = tuple(map(float, glow_centers[i]))
        va_body.render(moderngl.TRIANGLES)

        fbo_scene.use(); ctx.viewport = (0, 0, W, H); ctx.disable(moderngl.DEPTH_TEST); ctx.disable(moderngl.BLEND)
        fbo_scene.clear(0.0, 0.0, 0.0, 0.0, depth=1.0)
        va_body.render(moderngl.TRIANGLES)

        # Pass 2: Conserved 3D Instanced Ash Flakes + Motion-Streaked Night Embers!
        batch = frame_batches[i]
        if len(batch) > 0:
            inst_buf.write(batch.tobytes())
            ro = cam_C[i]; R = cam_R[i]
            Rt = tuple(R.T.flatten().astype(np.float32))
            p_dep = float(depths[i])

            ctx.enable(moderngl.DEPTH_TEST); ctx.depth_mask = True
            ctx.enable(moderngl.BLEND); ctx.blend_func = (moderngl.SRC_ALPHA, moderngl.ONE_MINUS_SRC_ALPHA)
            t_body_col.use(0)
            flake_prog["uPersonAlpha"].value = 0; flake_prog["uPersonDepth"].value = p_dep
            flake_prog["uRo"].value = tuple(map(float, ro)); flake_prog["uCam"].value = Rt
            va_flake.render(moderngl.TRIANGLES, instances=len(batch))

            ctx.depth_mask = False
            ctx.blend_func = (moderngl.ONE, moderngl.ONE)
            ember_prog["uPersonAlpha"].value = 0; ember_prog["uPersonDepth"].value = p_dep
            ember_prog["uRo"].value = tuple(map(float, ro)); ember_prog["uCam"].value = Rt
            va_ember.render(moderngl.TRIANGLES, instances=len(batch))
            ctx.depth_mask = True

        # Pass 3: Dual-Radius 1:1 Optical Bloom
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

        # Pass 4: Final Night-Cinema Composite
        fbo_final.use(); ctx.viewport = (0, 0, W, H)
        t_scene.use(0); t_b1b.use(1); t_b2b.use(2)
        comp_prog["uScene"].value = 0; comp_prog["uBloom1"].value = 1; comp_prog["uBloom2"].value = 2
        comp_prog["uFrame"].value = float(i)
        va_comp.render(moderngl.TRIANGLES)
        ctx.finish()

        arr = np.frombuffer(t_final.read(), "f4").reshape(H, W, 4)[::-1]
        img = (np.clip(arr[..., :3], 0, 1) * 255 + 0.5).astype(np.uint8)[:, :, ::-1]
        return pl, img

    if args.passes:
        p_root = WORK / "pass_night"
        shutil.rmtree(p_root, ignore_errors=True)
        for sub in ("matte_clean", "burn_seam", "particles", "bloom_spill"):
            (p_root / sub).mkdir(parents=True, exist_ok=True)
        # Save a compact subsampled 3D particle cloud for the v7 3D diagnostic orbit view!
        cloud_frames = {}
        for i in range(0, N):
            b = frame_batches[i]
            if len(b) > 0:
                step = max(1, len(b) // 1800)
                cloud_frames[f"p_{i:04d}"] = b[::step, :3].astype(np.float32)
                cloud_frames[f"t_{i:04d}"] = b[::step, 9].astype(np.float32)
        np.savez_compressed(WORK / "pass_night" / "cloud3d_sample.npz", **cloud_frames)

        PW, PH = 1280, 720
        clean_dim = cv2.resize((clean_bgr.astype(np.float32) * 0.42).astype(np.uint8), (PW, PH))
        for i in range(25, 145):
            pl = cv2.imread(str(WORK / "frames" / f"{i:04d}.jpg"))
            m_u8 = cv2.imread(str(WORK / "mask" / f"{i:04d}.png"), cv2.IMREAD_GRAYSCALE)
            m_f = m_u8.astype(np.float32) / 255.0
            mode_str, burn_p = timeline_state(i)
            bf = eval_body_field(smooth_boxes[i], noise_tex, H, W, mode=mode_str) if mode_str in ("disint", "reasm") else np.zeros((H, W), np.float32)

            # Pass A: Real Clean Plate + Hybrid MediaPipe/Diff Matte Visualization
            mc = clean_bgr.copy()
            # Subtle green diagnostic tint inside person matte + crisp white/green contour showing exact extraction!
            tint = np.zeros_like(mc); tint[:, :, 1] = 215; tint[:, :, 0] = 60
            mc = np.clip(mc.astype(np.float32) * (1.0 - 0.42 * m_f[:, :, None]) + tint.astype(np.float32) * (0.42 * m_f[:, :, None]), 0, 255).astype(np.uint8)
            edge = cv2.dilate(cv2.Canny(m_u8, 60, 160), np.ones((3, 3), np.uint8))
            mc[edge > 0] = (90, 255, 120)
            cv2.imwrite(str(p_root / "matte_clean" / f"{i:04d}.jpg"), cv2.resize(mc, (PW, PH)), [cv2.IMWRITE_JPEG_QUALITY, 90])

            # Render full GPU passes for this frame to harvest exact GPU buffers!
            _, _ = render_one(i)

            # Pass B: Body-Locked Organic Burn Field + Molten Seam Pass (directly from fbo_body!)
            b_arr = np.frombuffer(t_body_col.read(), "f4").reshape(H, W, 4)[::-1]
            b_em  = np.frombuffer(t_body_em.read(), "f4").reshape(H, W, 4)[::-1]
            b_lin = np.clip(b_arr[..., :3] + b_em[..., :3] * 0.85, 0.0, 1.0)
            # Overlay subtle magma heat colormap of bf inside remaining solid body so organic burn order is crystal clear!
            bf_u8 = np.clip(bf * 255.0, 0, 255).astype(np.uint8)
            bf_cm = cv2.applyColorMap(bf_u8, cv2.COLORMAP_INFERNO)[:, :, ::-1].astype(np.float32) / 255.0
            body_vis = np.clip(b_arr[..., 3:4], 0.0, 1.0)
            b_srgb = np.power(b_lin, 1.0 / 2.2)
            b_mix = np.clip(b_srgb * (1.0 - 0.55 * body_vis) + bf_cm * (0.55 * body_vis) + np.clip(b_em[..., :3] * 0.65, 0, 1), 0, 1)
            cv2.imwrite(str(p_root / "burn_seam" / f"{i:04d}.jpg"), cv2.resize((b_mix * 255).astype(np.uint8)[:, :, ::-1], (PW, PH)), [cv2.IMWRITE_JPEG_QUALITY, 90])

            # Pass C: Isolated 3D OpenGL Conserved Particles (ash flakes + velocity-streaked embers over dimmed road!)
            # Render particles ONLY onto clean_dim backdrop using exact GPU pipeline
            fbo_scene.use(); ctx.viewport = (0, 0, W, H); ctx.disable(moderngl.DEPTH_TEST); ctx.disable(moderngl.BLEND)
            fbo_scene.clear(0.0, 0.0, 0.0, 0.0, depth=1.0)
            batch = frame_batches[i]
            if len(batch) > 0:
                inst_buf.write(batch.tobytes())
                ro = cam_C[i]; R = cam_R[i]; Rt = tuple(R.T.flatten().astype(np.float32))
                ctx.enable(moderngl.DEPTH_TEST); ctx.depth_mask = True
                ctx.enable(moderngl.BLEND); ctx.blend_func = (moderngl.SRC_ALPHA, moderngl.ONE_MINUS_SRC_ALPHA)
                t_body_col.use(0)
                flake_prog["uPersonAlpha"].value = 0; flake_prog["uPersonDepth"].value = 999.0
                flake_prog["uRo"].value = tuple(map(float, ro)); flake_prog["uCam"].value = Rt
                va_flake.render(moderngl.TRIANGLES, instances=len(batch))
                ctx.depth_mask = False; ctx.blend_func = (moderngl.ONE, moderngl.ONE)
                ember_prog["uPersonAlpha"].value = 0; ember_prog["uPersonDepth"].value = 999.0
                ember_prog["uRo"].value = tuple(map(float, ro)); ember_prog["uCam"].value = Rt
                va_ember.render(moderngl.TRIANGLES, instances=len(batch))
                ctx.depth_mask = True
            ctx.finish()
            p_arr = np.frombuffer(t_scene.read(), "f4").reshape(H, W, 4)[::-1]
            p_srgb = (np.power(np.clip(p_arr[..., :3] / (1.0 + p_arr[..., :3] * 0.25), 0.0, 1.0), 1.0 / 2.2) * 255).astype(np.uint8)[:, :, ::-1]
            p_vis = cv2.add(clean_dim, cv2.resize(p_srgb, (PW, PH)))
            cv2.imwrite(str(p_root / "particles" / f"{i:04d}.jpg"), p_vis, [cv2.IMWRITE_JPEG_QUALITY, 90])

            # Pass D: Isolated Optical Dual-Radius Bloom + Interactive Wet-Road Light Spill!
            b1 = np.frombuffer(t_b1b.read(), "f4").reshape(hh, hw, 4)[::-1][..., :3]
            b2 = np.frombuffer(t_b2b.read(), "f4").reshape(qh, qw, 4)[::-1][..., :3]
            bl_lin = cv2.resize(b1, (PW, PH)) * 0.85 + cv2.resize(b2, (PW, PH)) * 0.70
            # Add road light spill contribution directly from glow_centers[i]
            gx, gy, gr, gi = glow_centers[i]
            yy_p, xx_p = np.mgrid[0:PH, 0:PW].astype(np.float32)
            sx_r, sy_r = PW / float(W), PH / float(H)
            dg2 = ((xx_p - gx * sx_r) / max(gr * sx_r, 80.0)) ** 2 + ((yy_p - gy * sy_r) / max(gr * sy_r, 80.0)) ** 2
            road_m = np.clip((yy_p - 580 * sy_r) / (260 * sy_r), 0.0, 1.0)
            dr2 = ((xx_p - gx * sx_r) / max(gr * sx_r * 1.35, 100.0)) ** 2 + ((yy_p - max(gy + 180, 860) * sy_r) / (210 * sy_r)) ** 2
            spill = (np.exp(-dg2 * 1.15) * 0.20 + np.exp(-dr2 * 1.35) * road_m * 0.52) * gi
            bl_lin += spill[:, :, None] * np.array([1.0, 0.36, 0.06], dtype=np.float32)[None, None, :]
            bl_srgb = (np.power(np.clip(bl_lin / (1.0 + bl_lin * 0.25), 0.0, 1.0), 1.0 / 2.2) * 255).astype(np.uint8)[:, :, ::-1]
            bl_vis = cv2.add(clean_dim, bl_srgb)
            cv2.imwrite(str(p_root / "bloom_spill" / f"{i:04d}.jpg"), bl_vis, [cv2.IMWRITE_JPEG_QUALITY, 90])
        print("passes exported ->", p_root)
        return

    if args.still:
        idxs = [int(x) for x in args.still.split(",")]
        rows = []
        for idx in idxs:
            pl, comp = render_one(idx)
            a, b = cv2.resize(pl, (640, 360)), cv2.resize(comp, (640, 360))
            cv2.putText(b, f"night f={idx}", (10, 344), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 0, 0), 3)
            cv2.putText(b, f"night f={idx}", (10, 344), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (80, 195, 255), 1)
            rows.append(np.hstack([a, b]))
        out_p = WORK / "still_night_sheet.jpg"
        cv2.imwrite(str(out_p), np.vstack(rows), [cv2.IMWRITE_JPEG_QUALITY, 92])
        print("stills ->", out_p)
        return

    if args.render:
        comp_dir = WORK / "comp"
        shutil.rmtree(comp_dir, ignore_errors=True); comp_dir.mkdir(parents=True)
        ts = time.time()
        for i in range(N):
            _, comp = render_one(i)
            cv2.imwrite(str(comp_dir / f"{i:04d}.png"), comp, [cv2.IMWRITE_PNG_COMPRESSION, 1])
            if i % 40 == 0 or i == N - 1:
                print(f"night_disint frame {i}/{N} ({time.time() - ts:.1f}s)", flush=True)
        out_dir = here / args.out; out_dir.mkdir(parents=True, exist_ok=True)
        r = f"{meta['fps_num']}/{meta['fps_den']}"; ff = find_ffmpeg()
        tags = ["-pix_fmt", "yuv420p", "-color_primaries", "bt709", "-color_trc", "bt709", "-colorspace", "bt709", "-movflags", "+faststart"]
        x264 = ["-c:v", "libx264", "-profile:v", "high", "-level", "4.1", "-crf", "14", "-preset", "slow"]
        run = lambda a: subprocess.run([ff, "-v", "error", "-y"] + a, check=True)
        run(["-framerate", r, "-i", str(comp_dir / "%04d.png")] + x264 + tags + [str(out_dir / "Night_Walk_Particle_Disintegration_Composite.mp4")])
        run([
            "-framerate", r, "-i", str(comp_dir / "%04d.png"),
            "-framerate", r, "-i", str(WORK / "frames" / "%04d.jpg"),
            "-filter_complex", "[0]scale=1080:608[a];[1]scale=1080:608[b];[a][b]vstack,pad=1080:1350:0:67:color=black",
        ] + x264 + tags + [str(out_dir / "Night_Walk_Particle_Disintegration_Compare_1080x1350.mp4")])
        print("encoded ->", out_dir)


if __name__ == "__main__":
    main()
