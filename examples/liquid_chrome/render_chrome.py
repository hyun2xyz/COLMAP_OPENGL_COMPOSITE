"""3D Liquid Chrome / Mercury Sculpting & Gravity Floor Puddles on a tracked dancer (ModernGL, OpenGL 3.3).

    python render_chrome.py [piece_dir] --still 55,115,190    quick look -> work/still_sheet.jpg
    python render_chrome.py [piece_dir]                        full render -> out/*_Composite.mp4, *_Compare_1080x1350.mp4

Raymarches silky smooth 3D liquid-mercury ribbons trailing off the dancer's MediaPipe-tracked 3D hands in plane
coordinates, sheds ballistic gravity droplets (g = -9.8 m/s^2) on fast arm whips, and merges landed droplets into
mirror-smooth liquid-mercury pools resting strictly on the open concrete tiles (z = 0, clamped in front of the
railing). Shaded with real live-action plate reflections, smooth studio/sky metallic gradients, sunlight speculars,
3D floor cast shadows, and depth-aware MediaPipe body occlusion.
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

FS = """#version 330
uniform sampler2D uPlate;
uniform sampler2D uMask;
uniform vec2 uRes;
uniform float uF, iTime, uBodyDepth, uShadowGain, uSminK;
uniform vec3 uRo, uSun;
uniform mat3 uCam;
uniform vec4 uTrailL[22];     /* xyz = 3D plane pos (m), w = radius (m); w <= 0 unused */
uniform vec4 uTrailR[22];
uniform vec4 uDrops[20];      /* xyz = falling 3D droplet pos, w = radius */
uniform vec4 uPuddles[20];    /* xy = floor puddle pos, z = radius (m), w = height (m) */
uniform vec4 uBBoxMin;
uniform vec4 uBBoxMax;
uniform int uMode;            /* 0 = composite, 1 = diagnostic chrome normal + shadow */

out vec4 oOut;

float smin(float a, float b, float k) {
  float h = max(k - abs(a - b), 0.0) / max(k, 1e-5);
  return min(a, b) - h * h * k * 0.25;
}

float sdCapsule(vec3 p, vec3 a, vec3 b, float ra, float rb) {
  vec3 pa = p - a, ba = b - a;
  float h = clamp(dot(pa, ba) / max(dot(ba, ba), 1e-6), 0.0, 1.0);
  return length(pa - ba * h) - mix(ra, rb, h);
}

/* silky smooth liquid-mercury SDF (hand ribbons + gravity drops + floor meniscus puddles) */
float mapChrome(vec3 p) {
  float d = 1e4;
  float k = uSminK;

  float dL = 1e4;
  for (int i = 0; i < 21; i++) {
    vec4 a = uTrailL[i], b = uTrailL[i + 1];
    if (a.w <= 0.002 || b.w <= 0.002) continue;
    float seg = sdCapsule(p, a.xyz, b.xyz, a.w, b.w);
    dL = (dL > 9e3) ? seg : smin(dL, seg, k * 0.70);
  }
  d = min(d, dL);

  float dR = 1e4;
  for (int i = 0; i < 21; i++) {
    vec4 a = uTrailR[i], b = uTrailR[i + 1];
    if (a.w <= 0.002 || b.w <= 0.002) continue;
    float seg = sdCapsule(p, a.xyz, b.xyz, a.w, b.w);
    dR = (dR > 9e3) ? seg : smin(dR, seg, k * 0.70);
  }
  d = smin(d, dR, k);

  for (int i = 0; i < 20; i++) {
    vec4 dr = uDrops[i];
    if (dr.w <= 0.002) continue;
    vec3 q = p - dr.xyz;
    q.z *= 0.82;
    float dd = (length(q) - dr.w) * 0.88;
    d = smin(d, dd, k * 0.65);
  }

  /* sleek liquid-mercury meniscus puddles resting flat on concrete z = 0 */
  if (p.z < 0.25 && p.z > -0.15) {
    float dP = 1e4;
    for (int i = 0; i < 20; i++) {
      vec4 pu = uPuddles[i];
      if (pu.z <= 0.005) continue;
      /* smooth convex mercury bead/pool sitting flush on z = 0 (no harsh plane cut!) */
      vec3 q = p - vec3(pu.x, pu.y, pu.w * 0.35);
      float h = max(pu.w, 0.022);
      float r = pu.z;
      float ell = (length(vec3(q.xy / r, q.z / h)) - 1.0) * min(r, h);
      float dp = max(ell, -p.z - 0.002);
      dP = (dP > 9e3) ? dp : smin(dP, dp, 0.08);
    }
    d = smin(d, dP, 0.08);
  }

  /* ultra-gentle low-frequency fluid surface tension swell (silky T-1000 mercury, zero foil crinkle!) */
  float wob = sin(p.x * 4.5 + iTime * 2.2) * cos(p.y * 4.2 - iTime * 1.8) * sin(p.z * 4.8 + iTime * 2.0);
  return d + wob * 0.0012;
}

vec3 calcNormal(vec3 p) {
  vec2 e = vec2(0.0028, -0.0028);
  return normalize(
    e.xyy * mapChrome(p + e.xyy) +
    e.yyx * mapChrome(p + e.yyx) +
    e.yxy * mapChrome(p + e.yxy) +
    e.xxx * mapChrome(p + e.xxx)
  );
}

bool hitAABB(vec3 ro, vec3 rd, vec3 bmin, vec3 bmax, out float t0, out float t1) {
  vec3 inv = 1.0 / (rd + vec3(1e-7));
  vec3 a = (bmin - ro) * inv;
  vec3 b = (bmax - ro) * inv;
  vec3 mn = min(a, b), mx = max(a, b);
  t0 = max(max(mn.x, mn.y), mn.z);
  t1 = min(min(mx.x, mx.y), mx.z);
  return t1 >= max(t0, 0.05);
}

/* true 360-deg spherical mirror-chrome environment (zero multi-wrap zebra bands!) */
vec3 sampleChromeEnv(vec3 p, vec3 reflDir) {
  /* map full 360-deg azimuth ONCE around the horizon onto the real architectural facade plate */
  float az = atan(reflDir.x, reflDir.y) / 6.2831853 + 0.5;   /* [0..1] once per 360 deg */
  float uMir = 1.0 - abs(fract(az) * 2.0 - 1.0);             /* seamless mirror wrap */
  float vEnv = clamp(0.54 - reflDir.z * 0.58, 0.03, 0.97);
  vec2 uv = vec2(clamp(uMir, 0.02, 0.98), vEnv);

  /* smooth 5-tap reflection prefilter for buttery liquid mercury */
  vec2 tx = 4.0 / uRes;
  vec3 plateEnv = texture(uPlate, uv).rgb * 0.36;
  plateEnv += texture(uPlate, clamp(uv + vec2( tx.x,  tx.y), 0.01, 0.99)).rgb * 0.16;
  plateEnv += texture(uPlate, clamp(uv + vec2(-tx.x,  tx.y), 0.01, 0.99)).rgb * 0.16;
  plateEnv += texture(uPlate, clamp(uv + vec2( tx.x, -tx.y), 0.01, 0.99)).rgb * 0.16;
  plateEnv += texture(uPlate, clamp(uv + vec2(-tx.x, -tx.y), 0.01, 0.99)).rgb * 0.16;

  /* pure silver-mercury contrast while keeping subtle architectural teal & warm gold roof reflections */
  float plLuma = dot(plateEnv, vec3(0.299, 0.587, 0.114));
  plateEnv = mix(vec3(plLuma), plateEnv, 0.62);
  plateEnv = clamp((plateEnv - 0.44) * 1.55 + 0.46, 0.0, 1.0);

  /* crisp physical sky + dark equatorial chrome horizon band */
  float elev = reflDir.z;
  vec3 skyCol = mix(vec3(0.96, 0.98, 1.0), vec3(0.55, 0.72, 0.86), smoothstep(0.06, 0.85, elev));
  vec3 gndCol = vec3(0.08, 0.09, 0.11);
  vec3 dome = mix(gndCol, skyCol, smoothstep(-0.07, 0.10, elev));
  float horBand = exp(-pow((elev + 0.01) / 0.09, 2.0));
  dome *= (1.0 - 0.62 * horBand);
  float panel = pow(max(dot(reflDir, normalize(vec3(-0.30, -0.35, 0.88))), 0.0), 16.0);
  dome += vec3(0.55, 0.58, 0.62) * panel;

  vec3 env = mix(plateEnv, dome, 0.44 + 0.28 * smoothstep(0.12, 0.85, elev));
  return env * vec3(0.96, 0.98, 1.02);
}

float floorShadow(vec3 Xf) {
  vec3 sun = normalize(uSun);
  float t = 0.025;
  float res = 1.0;
  for (int i = 0; i < 36; i++) {
    vec3 p = Xf + sun * t;
    if (p.z > 2.6) break;
    float h = mapChrome(p);
    if (h < 0.004) return 0.0;
    res = min(res, 6.5 * h / t);
    t += clamp(h, 0.018, 0.16);
  }
  return clamp(res, 0.0, 1.0);
}

vec3 shadePixel(vec2 px, out vec3 diagCol) {
  vec2 uv = px / uRes;
  vec3 plate = texture(uPlate, uv).rgb;
  float mask = texture(uMask, uv).r;
  diagCol = plate * 0.32;

  vec3 rd = normalize(uCam * vec3((px - 0.5 * uRes) / uF, 1.0));
  float tFloor = (rd.z < -0.004) ? (-uRo.z / rd.z) : 1e4;
  if (tFloor < 0.05 || tFloor > 18.0) tFloor = 1e4;

  vec3 bg = plate;
  if (tFloor < 14.0 && mask < 0.5) {
    vec3 Xf = uRo + rd * tFloor;
    if (Xf.x > uBBoxMin.x - 1.5 && Xf.x < uBBoxMax.x + 1.5 && Xf.y > uBBoxMin.y - 1.5 && Xf.y < uBBoxMax.y + 1.5) {
      float sh = floorShadow(Xf);
      float shadAmt = (1.0 - sh) * uShadowGain * (1.0 - smoothstep(0.15, 0.5, mask));
      bg *= (1.0 - shadAmt);
      diagCol = mix(diagCol, vec3(0.05, 0.06, 0.08), shadAmt);
    }
  }

  float t0, t1;
  if (!hitAABB(uRo, rd, uBBoxMin.xyz, uBBoxMax.xyz, t0, t1)) {
    return bg;
  }

  float t = max(t0, 0.08);
  float tMax = min(t1, tFloor + 0.14);
  float tHit = -1.0;
  for (int i = 0; i < 88; i++) {
    vec3 p = uRo + rd * t;
    float h = mapChrome(p);
    if (h < 0.0016) {
      tHit = t;
      break;
    }
    t += max(h * 0.85, 0.004);
    if (t > tMax) break;
  }

  if (tHit < 0.0) {
    return bg;
  }

  vec3 P = uRo + rd * tHit;
  float zCam = (transpose(uCam) * (P - uRo)).z;
  /* body occlusion: ribbons behind dancer pass behind him; puddles on floor stay cleanly beneath his shoes */
  float behindBody = smoothstep(-0.06, 0.14, zCam - uBodyDepth) * smoothstep(0.25, 0.72, mask);
  if (P.z < 0.08) {
    /* shoes/legs stand ON TOP of the floor mercury puddle! */
    behindBody = max(behindBody, smoothstep(0.30, 0.70, mask));
  }
  if (behindBody > 0.96) {
    return bg;
  }

  vec3 N = calcNormal(P);
  diagCol = mix(0.5 + 0.5 * N, diagCol, behindBody);

  vec3 V = -rd;
  vec3 R = reflect(rd, N);
  float ndv = max(dot(N, V), 0.0);
  float fresnel = 0.92 + 0.08 * pow(1.0 - ndv, 5.0);
  vec3 env = sampleChromeEnv(P, R);

  vec3 sun = normalize(uSun);
  vec3 H = normalize(sun + V);
  float specSun = pow(max(dot(N, H), 0.0), 280.0) * 2.1;
  float specBroad = pow(max(dot(N, H), 0.0), 28.0) * 0.22;

  /* grounded dark contact rim right where mercury touches the concrete tiles */
  float floorAO = smoothstep(0.0, 0.025, P.z) * 0.28 + 0.72;
  float rimDark = 1.0 - 0.16 * pow(1.0 - ndv, 2.0);

  vec3 chromeCol = env * fresnel * floorAO * rimDark;
  chromeCol += vec3(1.0, 0.99, 0.96) * (specSun + specBroad);
  chromeCol += vec3(0.90, 0.95, 1.0) * pow(1.0 - ndv, 4.2) * 0.24;

  return mix(chromeCol, bg, behindBody);
}

void main() {
  vec2 px0 = vec2(gl_FragCoord.x, uRes.y - gl_FragCoord.y);
  vec3 acc = vec3(0.0), dAcc = vec3(0.0);
  vec2 offs[4] = vec2[4](vec2(-0.25, -0.25), vec2(0.25, -0.25), vec2(-0.25, 0.25), vec2(0.25, 0.25));
  for (int k = 0; k < 4; k++) {
    vec3 dc;
    acc += shadePixel(px0 + offs[k], dc);
    dAcc += dc;
  }
  vec3 finalCol = (uMode == 0) ? (acc * 0.25) : (dAcc * 0.25);
  oOut = vec4(clamp(finalCol, 0.0, 1.0), 1.0);
}
"""


def label(im, s):
    im = im.copy()
    cv2.putText(im, s, (12, im.shape[0] - 14), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 4, cv2.LINE_AA)
    cv2.putText(im, s, (12, im.shape[0] - 14), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 1, cv2.LINE_AA)
    return im


def encode(c, title_label="3D Liquid Chrome Sculpting"):
    out = c["dir"] / "out"
    out.mkdir(exist_ok=True)
    nm = c["name"]
    r = f"{c['fps_num']}/{c['fps_den']}"
    ff = find_ffmpeg()
    tags = ["-pix_fmt", "yuv420p", "-color_primaries", "bt709", "-color_trc", "bt709", "-colorspace", "bt709", "-movflags", "+faststart"]
    x264 = ["-c:v", "libx264", "-profile:v", "high", "-level", "4.1", "-crf", "14", "-preset", "slow"]
    run = lambda a: subprocess.run([ff, "-v", "error", "-y"] + a, check=True)
    run(["-framerate", r, "-i", str(work(c, "comp", "%04d.png"))] + x264 + tags + [str(out / f"{nm}_Composite.mp4")])
    lab = lambda t: f"drawtext=fontfile='C\\:/Windows/Fonts/arialbd.ttf':text='{t}':x=w-tw-28:y=h-th-24:fontsize=44:fontcolor=white:shadowcolor=black@0.6:shadowx=2:shadowy=2"
    run([
        "-framerate", r, "-i", str(work(c, "comp", "%04d.png")),
        "-framerate", r, "-i", str(work(c, "frames", "%04d.jpg")),
        "-filter_complex",
        f"[0]scale=1080:608,{lab(title_label)}[a];[1]scale=1080:608,{lab('Before')}[b];[a][b]vstack,pad=1080:1350:0:67:color=black",
    ] + x264 + tags + [str(out / f"{nm}_Compare_1080x1350.mp4")])
    print("encoded ->", out)


def simulate_mercury(P, fps, cc):
    """Simulate silky 3D liquid-mercury hand ribbons, gravity drops, and floor pools clamped strictly on concrete tiles."""
    N = len(P)
    dt = 1.0 / fps
    hl = P[:, 15] * 0.52 + P[:, 19] * 0.48
    hr = P[:, 16] * 0.52 + P[:, 20] * 0.48
    vl = np.gradient(hl, dt, axis=0)
    vr = np.gradient(hr, dt, axis=0)
    sl = np.linalg.norm(vl, axis=1)
    sr = np.linalg.norm(vr, axis=1)

    # Determine safe back-railing Y boundary from the dancer's real floor foot Y coordinates!
    foot_y = np.median(P[:, [27, 28, 31, 32], 1])
    y_back_max = float(foot_y + 0.32)   # strictly in front of the glass railing!
    y_front_min = float(foot_y - 1.35)

    rng = np.random.default_rng(11)
    vel_thr = float(cc.get("drop_vel_thresh", 1.85))
    trail_len = int(cc.get("trail_frames", 20))
    base_r = float(cc.get("ribbon_radius", 0.090))

    drops = []
    puddles = []
    frame_data = []

    for i in range(N):
        def make_trail(h_arr, s_arr):
            pts = []
            for k in range(22):
                idx = i - k
                if idx < 0 or k >= trail_len:
                    pts.append((0.0, 0.0, -5.0, 0.0))
                    continue
                u = k / max(trail_len - 1, 1)
                spd_boost = float(np.clip(s_arr[idx] / 3.0, 0.50, 1.48))
                rad = base_r * spd_boost * math.sin(math.pi * (0.14 + 0.86 * (1.0 - u))) ** 0.72 * (1.0 - 0.60 * u)
                pos = h_arr[idx].copy()
                pos[2] = max(0.08, pos[2] - 0.12 * (u * u))
                pts.append((float(pos[0]), float(pos[1]), float(pos[2]), float(max(rad, 0.010))))
            return pts

        tL = make_trail(hl, sl)
        tR = make_trail(hr, sr)

        for pos_h, vel_h, spd_h in ((hl[i], vl[i], sl[i]), (hr[i], vr[i], sr[i])):
            if spd_h > vel_thr and (i % 4 == 0) and pos_h[2] > 0.40:
                v0 = vel_h * 0.32 + rng.uniform(-0.22, 0.22, 3)
                # Keep outward drop velocity slightly toward camera/plaza center so drops land on concrete tiles, not past railing
                v0[1] = min(v0[1], 0.15)
                v0[2] = min(v0[2], 0.9)
                drops.append({
                    "pos": pos_h.copy(),
                    "vel": v0,
                    "rad": float(rng.uniform(0.048, 0.072)),
                })

        kept_drops = []
        for dr in drops:
            dr["vel"][2] -= 9.8 * dt
            dr["pos"] += dr["vel"] * dt
            if dr["pos"][2] <= dr["rad"] * 0.45:
                px_l = float(dr["pos"][0])
                py_l = float(np.clip(dr["pos"][1], y_front_min, y_back_max))
                puddles.append({
                    "x": px_l,
                    "y": py_l,
                    "rad": float(dr["rad"] * 1.8),
                    "target_rad": float(dr["rad"] * 2.85),
                    "h": 0.048,
                    "age": 0.0,
                })
            else:
                kept_drops.append(dr)
        drops = kept_drops[-20:]

        for pu in puddles:
            pu["age"] += dt
            pu["rad"] += (pu["target_rad"] - pu["rad"]) * min(1.0, dt * 4.0)
            pu["h"] = max(0.026, 0.052 * math.exp(-pu["age"] * 0.45) + 0.024)
        puddles = puddles[-20:]

        d_uni = [(float(d["pos"][0]), float(d["pos"][1]), float(d["pos"][2]), float(d["rad"])) for d in drops]
        d_uni += [(0.0, 0.0, -5.0, 0.0)] * (20 - len(d_uni))
        p_uni = [(float(p["x"]), float(p["y"]), float(p["rad"]), float(p["h"])) for p in puddles]
        p_uni += [(0.0, 0.0, 0.0, 0.0)] * (20 - len(p_uni))

        pts_all = [x[:3] for x in (tL + tR + d_uni) if x[3] > 0.002]
        for p in puddles:
            pts_all.append((p["x"] - p["rad"] - 0.1, p["y"] - p["rad"] - 0.1, -0.05))
            pts_all.append((p["x"] + p["rad"] + 0.1, p["y"] + p["rad"] + 0.1, p["h"] + 0.15))
        arr_all = np.array(pts_all, np.float32) if pts_all else np.zeros((1, 3), np.float32)
        bmin = arr_all.min(axis=0) - 0.35
        bmax = arr_all.max(axis=0) + 0.35
        bmin[2] = min(bmin[2], -0.08)

        frame_data.append((tL, tR, d_uni, p_uni, bmin, bmax))
    return frame_data


def main():
    c = load(sys.argv[1] if len(sys.argv) > 1 and not sys.argv[1].startswith("--") else here / "piece_chrome")
    S = scene(c)
    W, H, N, fps = c["W"], c["H"], c["N"], c["fps"]
    f = float(S["f"])
    cc = c.get("chrome", {})
    stills = [int(x) for x in sys.argv[sys.argv.index("--still") + 1].split(",")] if "--still" in sys.argv else None

    L = np.load(work(c, "limbs.npz"))
    P = L["P"].astype(np.float64)
    depths = L["depth"].astype(np.float64)
    sim_frames = simulate_mercury(P, fps, cc)

    ctx = moderngl.create_standalone_context(require=330)
    prog = ctx.program(vertex_shader=VS, fragment_shader=FS)
    quad = ctx.buffer(np.array([-1, -1, 3, -1, -1, 3], "f4").tobytes())
    va = ctx.vertex_array(prog, [(quad, "2f", "p")])
    plate_t = ctx.texture((W, H), 3)
    plate_t.filter = (moderngl.LINEAR, moderngl.LINEAR)
    plate_t.repeat_x = plate_t.repeat_y = False
    mask_t = ctx.texture((W, H), 1)
    mask_t.filter = (moderngl.LINEAR, moderngl.LINEAR)
    mask_t.repeat_x = mask_t.repeat_y = False
    out_t = ctx.texture((W, H), 4, dtype="f4")
    fbo = ctx.framebuffer([out_t])

    prog["uRes"].value = (float(W), float(H))
    prog["uF"].value = f
    prog["uSun"].value = tuple(map(float, S["sun"]))
    prog["uShadowGain"].value = float(cc.get("shadow_gain", 0.54))
    prog["uSminK"].value = float(cc.get("smin_k", 0.16))

    def render_one(i, mode=0):
        pl = frame(c, i)
        m8 = cv2.imread(str(work(c, "mask", f"{i:04d}.png")), cv2.IMREAD_GRAYSCALE)
        plate_t.write(pl[:, :, ::-1].tobytes())
        mask_t.write(m8.tobytes())
        plate_t.use(0)
        mask_t.use(1)
        prog["uPlate"].value = 0
        prog["uMask"].value = 1
        prog["uRo"].value = tuple(map(float, S["cam"][i]))
        prog["uCam"].value = tuple(np.asarray(S["rot"][i], np.float32).T.flatten())
        prog["uBodyDepth"].value = float(depths[i])
        prog["iTime"].value = i / fps
        tL, tR, d_uni, p_uni, bmin, bmax = sim_frames[i]
        prog["uTrailL"].value = tL
        prog["uTrailR"].value = tR
        prog["uDrops"].value = d_uni
        prog["uPuddles"].value = p_uni
        prog["uBBoxMin"].value = (*map(float, bmin), 0.0)
        prog["uBBoxMax"].value = (*map(float, bmax), 0.0)
        prog["uMode"].value = mode
        fbo.use()
        va.render(moderngl.TRIANGLES)
        ctx.finish()
        arr = np.frombuffer(out_t.read(), "f4").reshape(H, W, 4)[::-1]
        img = (np.clip(arr[..., :3], 0, 1) * 255 + 0.5).astype(np.uint8)[:, :, ::-1]
        return pl, img

    if stills:
        rows = []
        hero = None
        for i in stills:
            pl, comp = render_one(i, mode=0)
            _, diag = render_one(i, mode=1)
            if i == stills[len(stills) // 2]:
                hero = comp
            t_im = lambda im, s: label(cv2.resize(im, (640, 360)), s)
            rows.append(np.hstack([t_im(pl, f"plate {i}"), t_im(diag, "3D chrome normal + shadow"), t_im(comp, "3D liquid chrome")]))
        out_p = work(c, "still_sheet.jpg")
        cv2.imwrite(str(out_p), np.vstack(rows), [cv2.IMWRITE_JPEG_QUALITY, 92])
        if hero is not None:
            cv2.imwrite(str(work(c, "still_hero.jpg")), hero, [cv2.IMWRITE_JPEG_QUALITY, 94])
        print("stills ->", out_p)
        return

    comp_dir = work(c, "comp")
    shutil.rmtree(comp_dir, ignore_errors=True)
    comp_dir.mkdir(parents=True)
    ts = time.time()
    for i in range(N):
        _, comp = render_one(i, mode=0)
        cv2.imwrite(str(comp_dir / f"{i:04d}.png"), comp, [cv2.IMWRITE_PNG_COMPRESSION, 1])
        if i % 40 == 0 or i == N - 1:
            print(f"frame {i}/{N} ({time.time() - ts:.1f}s)", flush=True)
    encode(c)


if __name__ == "__main__":
    main()
