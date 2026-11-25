"""Effect #5 (Cinema Forest Edition): 3D Ember/Ash Particle Disintegration & Deep-Mist Reverse Reassembly
on Pexels 31808102 (Misty Frosted Woodland Walk — SIFT+RANSAC Real Clean Plate + Conserved 3D Physics).

    python render_disintegration_forest.py --still 45,78,112,155,188,224  quick QA -> piece_forest/work/still_forest_sheet.jpg
    python render_disintegration_forest.py --render                       full render -> out_forest/Misty_Forest_Particle_Disintegration_*.mp4

Cinematic A24 woodland scene (Pexels 31808102, 240f @ 29.97fps = 8.01s):
- 100% REAL zero-AI empty misty forest clean plate aligned per-frame via SIFT+RANSAC static-background homographies,
  removing BOTH the hiker and his diagonal morning meadow shadow seamlessly.
- ONE strictly conserved population (~25,000 3D particles) — ZERO mid-air particle spawns or dissolve-ins!
  1) f=0..23   : Pure untouched A24 misty woodland opening.
  2) f=24..96  : Fiery burn front sweeps head/backpack->coat->boots in the foreground (~5.2m), shedding 25,000
                 crisp dark charred ash flakes & deep molten orange embers into a 3D Curl-Noise breeze sweeping
                 rightward and deeper through the frosted oak branches!
  3) f=97..131 : Eerily empty misty woodland trail + suspended ember/ash storm turning smoothly deeper down trail.
  4) f=132..204: All 25,000 conserved embers funnel DEEPER into the silver fog (~8.5m) with true 3D perspective
                 foreshortening and snap pixel-exact back into his receding silhouette (boots->coat->head)!
  5) f=205..239: Pure clean live-action A24 cinema outro striding into the fog (0 stray particles).
- Warm orange interactive light bounce onto the frosted winter grass + dual-radius 1:1 bloom + film grain. ZERO text/HUD.
"""
import argparse, json, math, shutil, subprocess, sys, time
from pathlib import Path
import numpy as np, cv2, moderngl

here = Path(__file__).resolve().parent
ROOT = here.parents[1]
WORK = here / "piece_forest" / "work"


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
    """Divergence-free 3D analytical Curl Noise for silky woodland mist eddies."""
    x, y, z = pos[:, 0], pos[:, 1], pos[:, 2]
    k1 = 1.15; ph1 = t_val * 0.65
    sx1, cx1 = np.sin(k1 * x + ph1), np.cos(k1 * x + ph1)
    sy1, cy1 = np.sin(k1 * y - ph1 * 0.6), np.cos(k1 * y - ph1 * 0.6)
    sz1, cz1 = np.sin(k1 * z + ph1 * 0.5), np.cos(k1 * z + ph1 * 0.5)
    vx1 = k1 * (sx1 * (-sy1) - (-sz1) * cx1)
    vy1 = k1 * (sy1 * (-sz1) - (-sx1) * cy1)
    vz1 = k1 * (sz1 * (-sx1) - (-sy1) * cz1)

    k2 = 2.90; ph2 = t_val * 1.30 + 2.3
    sx2, cx2 = np.sin(k2 * x - ph2), np.cos(k2 * x - ph2)
    sy2, cy2 = np.sin(k2 * y + ph2 * 0.8), np.cos(k2 * y + ph2 * 0.8)
    sz2, cz2 = np.sin(k2 * z - ph2 * 0.7), np.cos(k2 * z - ph2 * 0.7)
    vx2 = 0.44 * k2 * (sx2 * (-sy2) - (-sz2) * cx2)
    vy2 = 0.44 * k2 * (sy2 * (-sz2) - (-sx2) * cy2)
    vz2 = 0.44 * k2 * (sz2 * (-sx2) - (-sy2) * cz2)

    return np.stack([vx1 + vx2, vy1 + vy2, vz1 + vz2], axis=1) * 0.85


def build_body_burn_noise(H, W, seed=91):
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


def eval_body_field(mask_u8, noise_tex, mode="disint"):
    H, W = mask_u8.shape
    ys, xs = np.where(mask_u8 > 55)
    if len(ys) < 20:
        return np.zeros((H, W), dtype=np.float32)
    y_min, y_max = float(ys.min()), float(ys.max())
    x_min, x_max = float(xs.min()), float(xs.max())
    bh = max(y_max - y_min, 40.0)
    bw = max(x_max - x_min, 40.0)

    yy = np.linspace(0, H - 1, H, dtype=np.float32)[:, None]
    xx = np.linspace(0, W - 1, W, dtype=np.float32)[None, :]
    v = np.clip((yy - y_min) / bh, -0.2, 1.2)
    u = np.clip((xx - x_min) / bw, -0.2, 1.2)

    nh, nw = noise_tex.shape
    ix = np.clip((u[0, :] * (nw - 1)).astype(np.int32), 0, nw - 1)
    iy = np.clip((v[:, 0] * (nh - 1)).astype(np.int32), 0, nh - 1)
    n_body = noise_tex[iy[:, None], ix[None, :]]

    if mode == "disint":
        base = 0.84 * v + 0.16 * u
    else:
        base = 0.88 * (1.0 - v) + 0.12 * (1.0 - u)

    raw = base + 0.18 * (n_body - 0.5)
    vals = raw[mask_u8 > 80]
    lo, hi = float(np.percentile(vals, 0.5)), float(np.percentile(vals, 99.5))
    return np.clip((raw - lo) / max(hi - lo, 1e-4), 0.0, 1.0).astype(np.float32)


def timeline_state(i):
    """Master 240-frame (8.01s @ 29.97fps) A24 woodland arc:
      - f=0..21  : 100% pure untouched live-action opening (0 glow!)
      - f=22..26 : subtle smooth flame ignition along head/shoulders
      - f=27..96 : full fiery head-to-boot 3D disintegration (foreground -> mist)
      - f=97..131: empty frosted forest trail + suspended 3D ember & ash turnaround
      - f=132..204: dramatic deep-mist reverse reassembly further down the trail
      - f=205..239: pure clean live-action finale striding into silver fog (0 stray sparks!)
    Returns (mode, burn_p, shadow_vis).
    """
    if i < 22:
        return "solid", -0.15, 1.0
    elif i < 27:
        pre = ((i - 22) / 5.0) * 0.06
        return "disint", -0.06 + pre, 1.0
    elif i <= 96:
        u = (i - 27) / 69.0
        p = -0.02 + 1.16 * (u * u * (2.5 - 1.5 * u))
        sh = float(np.clip(1.0 - max(0.0, (p - 0.15) / 0.85), 0.0, 1.0))
        return "disint", float(p), sh
    elif i < 132:
        return "empty", 1.25, 0.0
    elif i <= 204:
        u = (i - 132) / 72.0
        p = -0.02 + 1.16 * (u * u * (2.5 - 1.5 * u))
        sh = float(np.clip(p / 0.72, 0.0, 1.0))
        return "reasm", float(p), sh
    else:
        return "solid", 1.25, 1.0


def simulate_conserved_particles(cam_C, cam_R, f_px, depths, noise_tex, W, H, fps):
    """Simulate ONE strictly conserved population (~25,000 3D particles) over 240 frames:
    - ZERO mid-air particle spawns!
    - Sheds from foreground walker (f=27..96, ~5.5m), sweeps rightward & DEEPER into the glowing misty clearing
      through the oak branches, turns smoothly in mid-air (f=90..132), and snaps pixel-exact into his deeper
      receding silhouette (f=132..204, ~8.8m)!
    """
    N = len(cam_C)
    dt = 1.0 / fps
    rng = np.random.default_rng(318)

    def unproject(xs, ys, i, z_jit):
        ro = cam_C[i]; R = cam_R[i]
        d_cam = depths[i] + z_jit
        xc = (xs - 0.5 * W) / f_px * d_cam
        yc = (ys - 0.5 * H) / f_px * d_cam
        zc = d_cam
        return ro[None, :] + np.stack([xc, yc, zc], axis=1) @ R.T

    # 1. Sample exact disintegration births on foreground body (f=27..96)
    d_pos_l, d_rgb_l, d_sp_l = [], [], []
    prev_p = -0.06
    for i in range(27, 97):
        _, cur_p, _ = timeline_state(i)
        m_u8 = cv2.imread(str(WORK / "mask" / f"{i:04d}.png"), cv2.IMREAD_GRAYSCALE)
        if m_u8 is None:
            prev_p = cur_p; continue
        fld = eval_body_field(m_u8, noise_tex, mode="disint")
        band = (m_u8 > 110) & (fld >= prev_p - 0.012) & (fld <= cur_p + 0.014)
        ys, xs = np.where(band)
        prev_p = cur_p
        if len(xs) == 0:
            continue
        k = min(len(xs), 420)
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

    # 2. Sample exact reassembly targets on his DEEPER receding silhouette (f=132..204)
    r_pos_l, r_rgb_l, r_arr_l = [], [], []
    prev_p = -0.05
    for i in range(132, 205):
        _, cur_p, _ = timeline_state(i)
        m_u8 = cv2.imread(str(WORK / "mask" / f"{i:04d}.png"), cv2.IMREAD_GRAYSCALE)
        if m_u8 is None:
            prev_p = cur_p; continue
        fld = eval_body_field(m_u8, noise_tex, mode="reasm")
        band = (m_u8 > 110) & (fld >= prev_p - 0.012) & (fld <= cur_p + 0.014)
        ys, xs = np.where(band)
        prev_p = cur_p
        if len(xs) == 0:
            continue
        k = min(len(xs), 420)
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
    r_pos1 = r_pos_all[pick]; r_rgb1 = r_rgb_all[pick]; r_arr = r_arr_all[pick]

    # 70% dark torn charcoal/fabric ash flakes (incredible dark contrast against bright silver morning mist!)
    # 30% rich deep fiery molten orange-gold embers!
    ptype = (rng.random(M) < 0.30).astype(np.float32)
    psize = np.where(ptype > 0.5, rng.uniform(0.0055, 0.0155, M), rng.uniform(0.0095, 0.0235, M)).astype(np.float32)
    phase = rng.uniform(0.0, 6.28318, M).astype(np.float32)
    drag  = rng.uniform(0.50, 1.55, M).astype(np.float32)
    cool  = rng.uniform(0.60, 1.50, M).astype(np.float32)
    wmul  = rng.uniform(0.45, 1.72, M).astype(np.float32)

    # 3. Integrate Forward Branch (f=27..136) & Backward Branch (f=86..204)
    #    Wind carries ash & embers RIGHTWARD (+X), UPWARD (+Z), and DEEPER into the misty clearing (+Y)!
    f_lo, f_hi = 27, 136
    b_lo, b_hi = 86, 204
    P_fwd = np.zeros((f_hi - f_lo + 1, M, 3), dtype=np.float32)
    cur_p = d_pos0.copy()
    cur_v = np.stack([
        rng.uniform(0.25, 1.35, M),   # +X rightward across misty woodland clearing!
        rng.uniform(0.30, 1.15, M),   # +Y deeper down the trail into the fog!
        rng.uniform(0.05, 0.95, M),   # +Z buoyant updraft toward oak branches!
    ], axis=1).astype(np.float32)

    for i in range(f_lo, f_hi + 1):
        P_fwd[i - f_lo] = cur_p
        moving = (d_spawn <= i)
        if np.any(moving):
            m_i = np.where(moving)[0]
            p_m = cur_p[m_i]; v_m = cur_v[m_i]
            cw = curl_wind_3d(p_m, i * dt)
            hang = 1.0 - 0.78 * float(smoothstep_np(92.0, 114.0, float(i)))
            wind_tgt = (np.array([0.85, 0.75, 0.42], dtype=np.float32)[None, :] * wmul[m_i, None]) * hang + cw * (1.35 * hang + 0.35)
            v_m = v_m + (wind_tgt - v_m) * (dt * 1.85 * drag[m_i, None])
            v_m *= (0.94 + 0.06 * hang)
            p_m = p_m + v_m * dt
            p_m[:, 1] = np.clip(p_m[:, 1], 2.8, 15.0)
            below = p_m[:, 2] < 0.08
            if np.any(below):
                p_m[below, 2] = 0.08
                v_m[below, 2] = np.abs(v_m[below, 2]) * 0.45 + 0.20
            cur_p[m_i] = p_m; cur_v[m_i] = v_m

    P_bwd = np.zeros((b_hi - b_lo + 1, M, 3), dtype=np.float32)
    cur_p = r_pos1.copy()
    cur_v = np.stack([
        rng.uniform(-1.15, 0.25, M),
        rng.uniform( 0.20, 1.15, M),  # particles continue funneling DEEPER (+Y) into his receding body!
        rng.uniform(-0.90, 0.08, M),
    ], axis=1).astype(np.float32)

    for i in range(b_hi, b_lo - 1, -1):
        P_bwd[i - b_lo] = cur_p
        active = (r_arr >= i)
        if np.any(active):
            s_i = np.where(active)[0]
            p_s = cur_p[s_i]; v_s = cur_v[s_i]
            cw = curl_wind_3d(p_s, i * dt + 1.6)
            pull = 0.22 + 0.78 * float(smoothstep_np(112.0, 136.0, float(i)))
            fwd_tgt = (np.array([-0.55, 0.65, -0.44], dtype=np.float32)[None, :] * wmul[s_i, None]) * pull + cw * (1.30 * pull + 0.35)
            p_s = p_s - v_s * dt
            v_s = v_s + (fwd_tgt - v_s) * (dt * 1.85 * drag[s_i, None])
            p_s[:, 1] = np.clip(p_s[:, 1], 2.8, 15.0)
            p_s[:, 2] = np.clip(p_s[:, 2], 0.14, 3.60)
            cur_p[s_i] = p_s; cur_v[s_i] = v_s

    # 4. Spatial Corridor Pairing at Mid-Air Turnaround (f=114)
    f_mid = 114
    pf_mid = P_fwd[f_mid - f_lo]; pb_mid = P_bwd[f_mid - b_lo]
    def spatial_order(pts):
        z_rank = np.argsort(np.argsort(pts[:, 2]))
        z_bin = (z_rank * 14) // len(pts)
        return np.lexsort((pts[:, 0], z_bin))

    ord_f = spatial_order(pf_mid); ord_b = spatial_order(pb_mid)
    pair_b = np.empty(M, dtype=np.int32)
    pair_b[ord_f] = ord_b
    P_bwd = P_bwd[:, pair_b, :]; r_rgb1 = r_rgb1[pair_b]; r_arr = r_arr[pair_b]

    # 5. Evaluate Single Conserved Continuous Trajectory P(i) (f=27..204, 0 particles outside!)
    frame_batches = [np.zeros((0, 13), dtype=np.float32) for _ in range(N)]
    glow_centers  = np.zeros((N, 4), dtype=np.float32)
    prev_pos = d_pos0.copy()
    max_k = 0

    for i in range(27, 205):
        alive = (d_spawn <= i) & (i <= r_arr)
        if not np.any(alive):
            continue
        idx = np.where(alive)[0]
        w = float(smootherstep_np(92.0, 132.0, float(i)))
        if i <= 92:
            pos_i = P_fwd[i - f_lo, idx]
        elif i >= 132:
            pos_i = P_bwd[i - b_lo, idx]
        else:
            pos_i = (1.0 - w) * P_fwd[i - f_lo, idx] + w * P_bwd[i - b_lo, idx]

        newly_born = (d_spawn[idx] == i)
        vel_i = (pos_i - prev_pos[idx]) / dt
        if np.any(newly_born):
            vel_i[newly_born] = np.array([0.45, 0.35, 0.35], dtype=np.float32)
        prev_pos[idx] = pos_i

        age_b = (i - d_spawn[idx]).astype(np.float32) * dt
        rem_a = (r_arr[idx] - i).astype(np.float32) * dt
        rgb_i = (1.0 - w) * d_rgb0[idx] + w * r_rgb1[idx]

        heat_b = np.exp(-age_b * cool[idx] * np.where(ptype[idx] > 0.5, 0.45, 1.90))
        heat_s = np.exp(-rem_a * np.where(ptype[idx] > 0.5, 0.52, 2.10)) * float(smoothstep_np(112.0, 138.0, float(i)))
        ambient_ember = 0.36 * ptype[idx]
        temp_i = np.clip(np.maximum(heat_b, heat_s * 1.18) + ambient_ember, 0.0, 1.35)

        end_ramp = np.clip(np.minimum(age_b, rem_a) / 0.06 + 0.42, 0.42, 1.0)
        sz_i = psize[idx] * end_ramp
        ph_i = phase[idx] + age_b * 4.5

        batch = np.column_stack([
            pos_i, vel_i, rgb_i,
            temp_i.astype(np.float32), sz_i.astype(np.float32),
            ph_i.astype(np.float32), ptype[idx],
        ]).astype(np.float32)
        frame_batches[i] = batch
        max_k = max(max_k, len(batch))

        hot_w = (temp_i ** 2) + 1e-4
        c3 = (pos_i * hot_w[:, None]).sum(axis=0) / hot_w.sum()
        q = cam_R[i].T @ (c3 - cam_C[i])
        if q[2] > 0.5:
            cx_px = float(f_px * q[0] / q[2] + 0.5 * W)
            cy_px = float(f_px * q[1] / q[2] + 0.5 * H)
            rad_px = float(np.clip(f_px * 1.65 / q[2], 160.0, 480.0))
            inten = float(np.clip(len(idx) / 15000.0, 0.0, 1.0) * (0.45 + 0.55 * float(temp_i.mean())))
            glow_centers[i] = [cx_px, cy_px, rad_px, inten]

    for ch in range(4):
        glow_centers[:, ch] = cv2.GaussianBlur(glow_centers[:, ch:ch + 1], (1, 9), 2.2)[:, 0]

    print(f"Conserved Forest 3D Particle Sim: M={M} particles (0 mid-air spawns!) | peak active/frame={max_k}")
    return frame_batches, glow_centers


VS_FSQ = "#version 330\nin vec2 p; void main(){ gl_Position = vec4(p, 0.0, 1.0); }"

BODY_FS = """#version 330
uniform sampler2D uLive, uClean, uMask, uRemZone, uBurnField;
uniform vec2 uRes;
uniform float uBurnProg, uShadowVis, uTime;
uniform int uMode;
uniform vec4 uGlow;

layout(location=0) out vec4 oScene;
layout(location=1) out vec4 oEmiss;

float hash21(vec2 p){ vec3 q=fract(vec3(p.xyx)*0.1031); q+=dot(q,q.yzx+33.33); return fract((q.x+q.y)*q.z); }

void main() {
  vec2 px = vec2(gl_FragCoord.x, uRes.y - gl_FragCoord.y);
  vec2 uv = clamp(px / uRes, vec2(0.001), vec2(0.999));

  vec3 liveLin  = pow(max(texture(uLive, uv).rgb, vec3(0.0005)), vec3(2.2));
  vec3 cleanLin = pow(max(texture(uClean, uv).rgb, vec3(0.0005)), vec3(2.2));
  float m       = texture(uMask, uv).r;
  float remZ    = texture(uRemZone, uv).r;
  float bf      = texture(uBurnField, uv).r;

  /* Warm fiery orange light bounce onto frosted morning meadow grass beneath burning embers! */
  vec3 meadowSpill = vec3(0.0);
  if (uGlow.w > 0.005) {
    float groundMask = smoothstep(660.0, 880.0, px.y);
    vec2 dr = vec2((px.x - uGlow.x) / max(uGlow.z * 1.4, 140.0), (px.y - max(uGlow.y + 180.0, 850.0)) / 220.0);
    float gLit = exp(-dot(dr, dr) * 1.35) * groundMask * uGlow.w;
    float flk = 0.94 + 0.06 * sin(uTime * 16.0 + sin(uTime * 8.7));
    meadowSpill = vec3(1.00, 0.34, 0.04) * gLit * 0.24 * flk;
  }

  float shadowWipe = clamp(remZ * (1.0 - uShadowVis), 0.0, 1.0);
  vec3 bgLin = mix(liveLin, cleanLin, shadowWipe) + meadowSpill;

  if (uMode == 1) {
    vec3 emptyCol = mix(liveLin, cleanLin, clamp(max(smoothstep(0.01, 0.25, m), remZ), 0.0, 1.0));
    oScene = vec4(emptyCol + meadowSpill, 0.0);
    oEmiss = vec4(0.0);
    return;
  }
  if (uMode == 3) {
    oScene = vec4(liveLin + meadowSpill, m);
    oEmiss = vec4(0.0);
    return;
  }

  float dEdge = (uMode == 0) ? (bf - uBurnProg) : (uBurnProg - bf);
  float bodyKeep = smoothstep(-0.006, 0.015, dEdge);
  float personAlpha = m * bodyKeep;

  float personWipe = smoothstep(0.01, 0.22, m) * (1.0 - bodyKeep);
  vec3 baseLin = mix(bgLin, cleanLin + meadowSpill, personWipe);

  float charBand = smoothstep(0.080, 0.012, dEdge) * smoothstep(-0.010, 0.012, dEdge) * m;
  vec3 liveCharred = mix(liveLin, liveLin * vec3(0.16, 0.07, 0.03), charBand * 0.78);
  vec3 col = mix(baseLin, liveCharred, personAlpha);

  float flk = 0.84 + 0.32 * hash21(floor(px * 0.35) + vec2(uTime * 21.0, -uTime * 15.0));
  float hotCore = exp(-pow((dEdge - 0.008) / 0.013, 2.0)) * smoothstep(0.06, 0.35, m) * flk;
  float hotHalo = exp(-pow((dEdge - 0.016) / 0.038, 2.0)) * smoothstep(0.03, 0.28, m);
  float seamEnv = smoothstep(-0.05, 0.015, uBurnProg) * smoothstep(1.15, 1.02, uBurnProg);

  /* Rich saturated volcanic orange-red seam that cuts boldly across bright silver morning mist! */
  vec3 seamCore = (uMode == 0) ? vec3(5.0, 1.85, 0.28) : vec3(5.4, 2.40, 0.52);
  vec3 seamHalo = (uMode == 0) ? vec3(2.8, 0.42, 0.02) : vec3(3.0, 0.62, 0.06);
  vec3 emiss = (seamCore * hotCore + seamHalo * hotHalo * 0.90) * seamEnv;

  oScene = vec4(col + emiss * 0.62, personAlpha);
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
  vec3 qPrev = transpose(uCam) * (inPos - inVel * 0.030 - uRo);
  vec2 pxPrev = uF * qPrev.xy / max(qPrev.z, 0.35) + 0.5 * uRes;
  vec2 velPx = px0 - pxPrev;
  float spdPx = length(velPx);

  float radPx = clamp(uF * inSize / q.z, 1.20, 9.5);

  vec2 axisX, axisY;
  if (inType > 0.5 && spdPx > 1.0) {
    vec2 dir = velPx / spdPx;
    vec2 perp = vec2(-dir.y, dir.x);
    float stretch = min(spdPx * 0.65 + radPx, radPx * 4.8);
    axisX = dir * stretch;
    axisY = perp * (radPx * 0.56);
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
  vLight = 0.55 + 0.60 * abs(sin(inPhase * 1.35 + 0.5));
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
  float jag = 0.68 + 0.22 * cos(ang * 3.0 + vPhase * 2.1) + 0.14 * sin(ang * 5.0 - vPhase * 3.7);
  float a = smoothstep(jag, jag - 0.20, r);
  if (a < 0.05) discard;

  /* Crisp pitch-dark charred winter fabric & ash flakes -> dramatic A24 dark-ink contrast against silver morning mist! */
  vec3 liveLin = pow(max(vRgb, vec3(0.003)), vec3(2.2));
  vec3 ashCol = mix(liveLin * 0.45, vec3(0.018, 0.019, 0.021), 0.65);
  vec3 bodyLin = ashCol * (0.65 + 0.50 * vLight);

  /* Fiery glowing red-orange rim along freshly-shed charred edges */
  float edgeHot = smoothstep(jag * 0.28, jag * 0.92, r) * smoothstep(0.14, 0.90, vTemp);
  vec3 emberCol = mix(vec3(2.2, 0.26, 0.01), vec3(4.2, 1.35, 0.16), clamp(vTemp, 0.0, 1.0)) * edgeHot;

  oScene = vec4(bodyLin + emberCol * 0.72, a * 0.96);
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

  /* Deep volcanic fiery crimson-orange-gold palette -> pops intensely even against bright misty woodland fog! */
  vec3 cRuby   = vec3(1.10, 0.07, 0.005);
  vec3 cOrange = vec3(2.15, 0.36, 0.015);
  vec3 cGold   = vec3(2.85, 1.15, 0.15);
  vec3 cWhite  = vec3(3.60, 2.45, 0.95);
  vec3 col = mix(cRuby, cOrange, smoothstep(0.05, 0.55, tHot));
  col = mix(col, cGold, smoothstep(0.55, 1.05, tHot));
  col = mix(col, cWhite, core * smoothstep(0.55, 1.25, tHot));

  vec3 outEm = col * (core * 0.92 + glow * 0.28) * twk;
  oScene = vec4(outEm * 0.55, 1.0);
  oEmiss = vec4(outEm * 0.78, 1.0);
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
  vec3 bloom = b1 * 0.52 + b2 * 0.38;

  /* Preserve 100% untouched misty woodland exposure; roll off glowing orange fire smoothly without washing out fog! */
  vec3 baseClamped = min(sceneLin, vec3(1.0));
  vec3 hdrExcess = max(sceneLin - vec3(1.0), vec3(0.0)) + bloom;
  vec3 emSoft = hdrExcess / (1.0 + hdrExcess * 0.28);
  vec3 totalLin = baseClamped + emSoft * (1.0 - baseClamped * 0.35);

  vec3 col = pow(clamp(totalLin, 0.0, 1.0), vec3(1.0 / 2.2));
  float act = clamp(length(bloom) * 2.2 + length(hdrExcess) * 1.8, 0.0, 1.0);
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
    ap.add_argument("--out", default="out_forest")
    args = ap.parse_args()

    meta = json.loads((WORK / "meta.json").read_text())
    S    = json.loads((WORK / "scene.json").read_text())
    T    = np.load(WORK / "track.npz")
    Hs   = np.load(WORK / "clean_homographies.npz")["H"]
    W, H, N = int(meta["W"]), int(meta["H"]), int(meta["N"])
    fps = float(meta["fps"])
    f_px = float(S["f"])
    cam_C = np.asarray(S["cam"], dtype=np.float64)
    cam_R = np.asarray(S["rot"], dtype=np.float64)
    depths = np.asarray(T["depth"], dtype=np.float64)

    clean_ref = cv2.imread(str(WORK / "clean_plate_ref.png"))
    pl0 = cv2.imread(str(WORK / "frames" / "0000.jpg"))
    noise_tex = build_body_burn_noise(512, 512, seed=91)

    t0 = time.time()
    frame_batches, glow_centers = simulate_conserved_particles(cam_C, cam_R, f_px, depths, noise_tex, W, H, fps)
    print(f"Forest sim finished in {time.time() - t0:.2f}s")

    gray0 = cv2.cvtColor(pl0, cv2.COLOR_BGR2GRAY).astype(np.float32) / 255.0
    black_floor = float(np.clip(np.percentile(gray0, 1.0) * 0.85, 0.008, 0.055))
    hp0 = gray0 - cv2.GaussianBlur(gray0, (0, 0), 1.2)
    grain_amt = float(np.clip(float(np.median(np.abs(hp0 - np.median(hp0)))) * 1.4826 * 1.15, 0.008, 0.025))

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
    rem_t   = ctx.texture((W, H), 1, dtype="f4"); rem_t.filter = (moderngl.LINEAR, moderngl.LINEAR); rem_t.repeat_x = rem_t.repeat_y = False
    burn_t  = ctx.texture((W, H), 1, dtype="f4"); burn_t.filter = (moderngl.LINEAR, moderngl.LINEAR); burn_t.repeat_x = burn_t.repeat_y = False

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

        # Warp real misty-forest clean plate into exact frame i via SIFT+RANSAC homography H[i]!
        clean_i = cv2.warpPerspective(clean_ref, Hs[i], (W, H), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
        # Generous body + left-stretching morning shadow removal zone!
        rem_bin = cv2.dilate((m_u8 > 18).astype(np.uint8), cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (75, 75)))
        ys, xs = np.where(m_u8 > 40)
        if len(ys) > 0:
            rem_bin[max(0, int(ys.max()) - 75):min(H, int(ys.max()) + 140),
                    max(0, int(xs.min()) - 420):min(W, int(xs.max()) + 120)] = 1
        ring = (cv2.dilate(rem_bin, np.ones((55, 55), np.uint8)) - rem_bin) > 0
        if ring.sum() > 200:
            for ch in range(3):
                delta = float(pl[:, :, ch][ring].mean()) - float(clean_i[:, :, ch][ring].mean())
                clean_i[:, :, ch] = np.clip(clean_i[:, :, ch].astype(np.float32) + delta, 0, 255).astype(np.uint8)
        rem_f = cv2.GaussianBlur(rem_bin.astype(np.float32), (0, 0), 20.0)

        mode_str, burn_p, sh_vis = timeline_state(i)
        bf = eval_body_field(m_u8, noise_tex, mode=mode_str) if mode_str in ("disint", "reasm") else np.zeros((H, W), np.float32)

        live_t.write(pl[:, :, ::-1].tobytes())
        clean_t.write(clean_i[:, :, ::-1].tobytes())
        mask_t.write(m_f.astype(np.float32).tobytes())
        rem_t.write(rem_f.astype(np.float32).tobytes())
        burn_t.write(bf.astype(np.float32).tobytes())

        fbo_body.use(); ctx.viewport = (0, 0, W, H); ctx.disable(moderngl.DEPTH_TEST); ctx.disable(moderngl.BLEND)
        live_t.use(0); clean_t.use(1); mask_t.use(2); rem_t.use(3); burn_t.use(4)
        body_prog["uLive"].value = 0; body_prog["uClean"].value = 1; body_prog["uMask"].value = 2
        body_prog["uRemZone"].value = 3; body_prog["uBurnField"].value = 4
        body_prog["uBurnProg"].value = float(burn_p)
        body_prog["uShadowVis"].value = float(sh_vis)
        body_prog["uTime"].value = float(i / fps)
        body_prog["uMode"].value = mode_map[mode_str]
        body_prog["uGlow"].value = tuple(map(float, glow_centers[i]))
        va_body.render(moderngl.TRIANGLES)

        fbo_scene.use(); ctx.viewport = (0, 0, W, H); ctx.disable(moderngl.DEPTH_TEST); ctx.disable(moderngl.BLEND)
        fbo_scene.clear(0.0, 0.0, 0.0, 0.0, depth=1.0)
        va_body.render(moderngl.TRIANGLES)

        batch = frame_batches[i]
        if len(batch) > 0:
            inst_buf.write(batch.tobytes())
            ro = cam_C[i]; R = cam_R[i]; Rt = tuple(R.T.flatten().astype(np.float32))
            p_dep = float(depths[i])

            ctx.enable(moderngl.DEPTH_TEST); ctx.depth_mask = True
            ctx.enable(moderngl.BLEND); ctx.blend_func = (moderngl.SRC_ALPHA, moderngl.ONE_MINUS_SRC_ALPHA)
            t_body_col.use(0)
            flake_prog["uPersonAlpha"].value = 0; flake_prog["uPersonDepth"].value = p_dep
            flake_prog["uRo"].value = tuple(map(float, ro)); flake_prog["uCam"].value = Rt
            va_flake.render(moderngl.TRIANGLES, instances=len(batch))

            ctx.depth_mask = False; ctx.blend_func = (moderngl.ONE, moderngl.ONE)
            ember_prog["uPersonAlpha"].value = 0; ember_prog["uPersonDepth"].value = p_dep
            ember_prog["uRo"].value = tuple(map(float, ro)); ember_prog["uCam"].value = Rt
            va_ember.render(moderngl.TRIANGLES, instances=len(batch))
            ctx.depth_mask = True

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

        fbo_final.use(); ctx.viewport = (0, 0, W, H)
        t_scene.use(0); t_b1b.use(1); t_b2b.use(2)
        comp_prog["uScene"].value = 0; comp_prog["uBloom1"].value = 1; comp_prog["uBloom2"].value = 2
        comp_prog["uFrame"].value = float(i)
        va_comp.render(moderngl.TRIANGLES)
        ctx.finish()

        arr = np.frombuffer(t_final.read(), "f4").reshape(H, W, 4)[::-1]
        img = (np.clip(arr[..., :3], 0, 1) * 255 + 0.5).astype(np.uint8)[:, :, ::-1]
        return pl, img

    if args.still:
        idxs = [int(x) for x in args.still.split(",")]
        rows = []
        for idx in idxs:
            pl, comp = render_one(idx)
            a, b = cv2.resize(pl, (640, 360)), cv2.resize(comp, (640, 360))
            cv2.putText(b, f"forest f={idx}", (10, 344), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 0, 0), 3)
            cv2.putText(b, f"forest f={idx}", (10, 344), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (80, 195, 255), 1)
            rows.append(np.hstack([a, b]))
        out_p = WORK / "still_forest_sheet.jpg"
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
                print(f"forest_disint frame {i}/{N} ({time.time() - ts:.1f}s)", flush=True)
        out_dir = here / args.out; out_dir.mkdir(parents=True, exist_ok=True)
        r = f"{meta['fps_num']}/{meta['fps_den']}"; ff = find_ffmpeg()
        tags = ["-pix_fmt", "yuv420p", "-color_primaries", "bt709", "-color_trc", "bt709", "-colorspace", "bt709", "-movflags", "+faststart"]
        x264 = ["-c:v", "libx264", "-profile:v", "high", "-level", "4.1", "-crf", "14", "-preset", "slow"]
        run = lambda a: subprocess.run([ff, "-v", "error", "-y"] + a, check=True)
        run(["-framerate", r, "-i", str(comp_dir / "%04d.png")] + x264 + tags + [str(out_dir / "Misty_Forest_Particle_Disintegration_Composite.mp4")])
        run([
            "-framerate", r, "-i", str(comp_dir / "%04d.png"),
            "-framerate", r, "-i", str(WORK / "frames" / "%04d.jpg"),
            "-filter_complex", "[0]scale=1080:608[a];[1]scale=1080:608[b];[a][b]vstack,pad=1080:1350:0:67:color=black",
        ] + x264 + tags + [str(out_dir / "Misty_Forest_Particle_Disintegration_Compare_1080x1350.mp4")])
        print("encoded ->", out_dir)


if __name__ == "__main__":
    main()
