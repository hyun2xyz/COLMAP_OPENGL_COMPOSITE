"""3D World-Locked Reaction-Diffusion (Gray-Scott Turing Growth) on a tracked live-action plaza (ModernGL, OpenGL 3.3).

    python render_turing.py [piece_dir] --still 20,105,205    quick look -> work/still_sheet.jpg
    python render_turing.py [piece_dir]                        full render -> out/*_Composite.mp4, *_Compare_1080x1350.mp4

Runs a real-time 1024x1024 floating-point GPU Gray-Scott PDE simulation directly in solved world-locked ground plane
coordinates (X, Y in metres), seeded radially from the circular fountain rim and organic plaza nodes.
Composite projects every live-action camera ray onto z = 0, samples the height field + finite-difference 3D surface
normals, and integrates tactile raised organic stone/mineral ridges with sunlit specular sheen, micro-shadows, and
automatic vegetation/water masking so growth creeps strictly over the real stone pavement.
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

# Real GPU Gray-Scott Reaction-Diffusion PDE step (1024x1024 RG32F world-space field)
SIM_FS = """#version 330
uniform sampler2D uState;
uniform vec2 uTexel;
uniform float uF, uK, uDu, uDv, uDt;
uniform float uWorldExt;    /* half-width of simulated world square in metres */
uniform float uFountainR;   /* circular fountain outer rim radius in metres */
uniform float iTime;
uniform int uSeedPulse;

out vec4 oState;

float hash21(vec2 p) {
  p = fract(p * vec2(127.1, 311.7));
  p += dot(p, p + 34.5);
  return fract(p.x * p.y);
}

void main() {
  vec2 uv = gl_FragCoord.xy * uTexel;
  vec2 ab = texture(uState, uv).rg;
  float a = ab.r;
  float b = ab.g;

  /* 9-point isotropic Laplacian stencil (Karl Sims weights: 0.2 orthogonal, 0.05 diagonal) */
  vec2 lap = -ab;
  lap += 0.20 * texture(uState, uv + vec2( uTexel.x, 0.0)).rg;
  lap += 0.20 * texture(uState, uv + vec2(-uTexel.x, 0.0)).rg;
  lap += 0.20 * texture(uState, uv + vec2(0.0,  uTexel.y)).rg;
  lap += 0.20 * texture(uState, uv + vec2(0.0, -uTexel.y)).rg;
  lap += 0.05 * texture(uState, uv + vec2( uTexel.x,  uTexel.y)).rg;
  lap += 0.05 * texture(uState, uv + vec2(-uTexel.x,  uTexel.y)).rg;
  lap += 0.05 * texture(uState, uv + vec2( uTexel.x, -uTexel.y)).rg;
  lap += 0.05 * texture(uState, uv + vec2(-uTexel.x, -uTexel.y)).rg;

  /* gentle spatial parameter modulation kept cleanly inside crisp labyrinth/coral Turing regime */
  vec2 worldXY = (uv * 2.0 - 1.0) * uWorldExt;
  float r = length(worldXY);
  float ang = atan(worldXY.y, worldXY.x);
  float F = uF + 0.0018 * sin(ang * 3.0 + r * 0.16);
  float K = uK + 0.0009 * cos(ang * 4.0 - r * 0.14);

  float abb = a * b * b;
  float da = uDu * lap.r - abb + F * (1.0 - a);
  float db = uDv * lap.g + abb - (F + K) * b;

  a = clamp(a + da * uDt, 0.0, 1.0);
  b = clamp(b + db * uDt, 0.0, 1.0);

  /* keep inside the circular fountain basin clear so growth erupts outward from the stone rim */
  if (r < uFountainR - 0.15) {
    a = 1.0;
    b = 0.0;
  }

  /* periodic organic spore reinforcement along the fountain rim during early frames */
  if (uSeedPulse == 1) {
    float rim = abs(r - (uFountainR + 0.55));
    if (rim < 0.35 && sin(ang * 14.0 + iTime * 2.0) > 0.35) {
      b = max(b, 0.55);
      a = min(a, 0.50);
    }
  }

  oState = vec4(a, b, 0.0, 1.0);
}
"""

# World-locked 3D tactile relief + sunlit wet specular composite onto real stone plaza
COMP_FS = """#version 330
uniform sampler2D uPlate;
uniform sampler2D uState;
uniform vec2 uRes;
uniform float uF, uWorldExt, uFountainR, uReliefGain, iTime;
uniform vec3 uRo, uSun;
uniform mat3 uCam;
uniform int uMode;   /* 0 = composite, 1 = diagnostic height + normal map */

out vec4 oOut;

float sampleH(vec2 worldXY) {
  vec2 uv = worldXY / (2.0 * uWorldExt) + 0.5;
  if (uv.x < 0.002 || uv.x > 0.998 || uv.y < 0.002 || uv.y > 0.998) return 0.0;
  float b = texture(uState, uv).g;
  /* crisp sculpted height curve from Turing morphogen B */
  return smoothstep(0.11, 0.38, b);
}

/* automatic stone-pavement mask from live-action plate RGB (excludes green canopy trees & cyan fountain water!) */
float pavementMask(vec3 rgb, float rWorld) {
  /* green foliage detector: G clearly exceeds R and B */
  float greenExcess = rgb.g - max(rgb.r, rgb.b);
  float treeMask = 1.0 - smoothstep(0.012, 0.055, greenExcess);
  /* dark tree canopy shadow & deep dark foliage */
  float luma = dot(rgb, vec3(0.299, 0.587, 0.114));
  float darkFoliage = smoothstep(0.12, 0.26, luma);
  /* warm terracotta stone affinity (R >= G - 0.02 and R > B) */
  float warmStone = smoothstep(-0.02, 0.045, rgb.r - rgb.b) * smoothstep(-0.04, 0.02, rgb.r - rgb.g * 0.95);
  /* clean exclusion of fountain basin interior */
  float outFountain = smoothstep(uFountainR - 0.15, uFountainR + 0.35, rWorld);
  /* keep growth inside the plaza bounds (exclude outer street asphalt & parked white cars) */
  float plazaBound = 1.0 - smoothstep(18.5, 23.0, rWorld);
  return clamp(treeMask * darkFoliage * (0.25 + 0.75 * warmStone) * outFountain * plazaBound, 0.0, 1.0);
}

void main() {
  vec2 px = vec2(gl_FragCoord.x, uRes.y - gl_FragCoord.y);
  vec2 uv = px / uRes;
  vec3 plate = texture(uPlate, uv).rgb;

  /* exact 3D camera ray intersection with the solved world ground plane z = 0 */
  vec3 rd = normalize(uCam * vec3((px - 0.5 * uRes) / uF, 1.0));
  if (rd.z >= -1e-4) {
    oOut = vec4(plate, 1.0);
    return;
  }
  float t = -uRo.z / rd.z;
  vec3 X = uRo + t * rd;
  float rWorld = length(X.xy);

  /* compute Turing morphogen height & world-space 3D normal via finite differences */
  float eps = 0.08;  /* 8 cm world-space sample step */
  float h0 = sampleH(X.xy);
  float hx = sampleH(X.xy + vec2(eps, 0.0));
  float hy = sampleH(X.xy + vec2(0.0, eps));
  float hL = sampleH(X.xy - vec2(eps, 0.0));
  float hD = sampleH(X.xy - vec2(0.0, eps));

  /* tactile raised coral/travertine normal */
  vec3 N = normalize(vec3((hL - hx) * 1.85 * uReliefGain, (hD - hy) * 1.85 * uReliefGain, 1.0));
  float pMask = pavementMask(plate, rWorld);
  float growth = h0 * pMask;

  if (uMode == 1) {
    /* diagnostic pass: world-space Turing height + tactile normal field over dimmed plate */
    vec3 dbg = plate * 0.30;
    vec3 nCol = 0.5 + 0.5 * N;
    dbg = mix(dbg, nCol, clamp(growth * 1.2, 0.0, 1.0));
    oOut = vec4(dbg, 1.0);
    return;
  }

  if (growth < 0.002 && max(max(hx, hy), max(hL, hD)) * pMask < 0.002) {
    oOut = vec4(plate, 1.0);
    return;
  }

  /* directional sunlight + contact shadow along -uSun across the raised organic ridges! */
  vec3 sun = normalize(uSun);
  vec2 shadStep = sun.xy / max(sun.z, 0.25);
  float hSun1 = sampleH(X.xy + shadStep * 0.22);
  float hSun2 = sampleH(X.xy + shadStep * 0.44);
  float ridgeShadow = clamp((max(hSun1, hSun2) - h0) * 1.45, 0.0, 0.72) * pMask;

  /* tactile mineral/wet-stone material response:
     - valleys between ridges darken subtly (wet etched stone erosion + AO)
     - raised ridges catch warm sunlight diffuse + crisp wet specular gloss! */
  float ndl = max(dot(N, sun), 0.0);
  vec3 H = normalize(sun - rd);
  float spec = pow(max(dot(N, H), 0.0), 48.0) * 0.52;
  float fresnel = pow(1.0 - max(dot(N, -rd), 0.0), 3.0) * 0.18;

  /* subtle micro-refraction / parallax displacement of the underlying real terracotta stone texture */
  vec2 uvDisp = clamp(uv + N.xy * (0.0022 * growth), 0.002, 0.998);
  vec3 stoneTex = texture(uPlate, uvDisp).rgb;

  /* blend natural bleached travertine/mineral ridge crest + damp dark etched valley */
  vec3 crestCol = mix(stoneTex * 1.24 + vec3(0.08, 0.075, 0.065), vec3(0.92, 0.89, 0.84), 0.34);
  vec3 troughCol = stoneTex * vec3(0.62, 0.58, 0.56);

  /* growth boundary wet-ring halo just ahead of the advancing organic front */
  float frontHalo = smoothstep(0.01, 0.14, h0) * (1.0 - smoothstep(0.14, 0.38, h0)) * pMask;

  vec3 mat = mix(troughCol, crestCol, smoothstep(0.18, 0.85, h0));
  /* light by actual sun direction */
  float lit = 0.52 + 0.62 * ndl;
  mat = mat * lit + vec3(1.0, 0.96, 0.90) * (spec + fresnel) * smoothstep(0.25, 0.85, h0);

  /* apply ridge cast-shadow onto the flat stone right beside each raised ridge */
  vec3 bgShaded = plate * (1.0 - ridgeShadow * 0.55) * (1.0 - frontHalo * 0.22);
  vec3 col = mix(bgShaded, mat, smoothstep(0.04, 0.45, growth));
  oOut = vec4(clamp(col, 0.0, 1.0), 1.0);
}
"""


def label(im, s):
    im = im.copy()
    cv2.putText(im, s, (12, im.shape[0] - 14), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 4, cv2.LINE_AA)
    cv2.putText(im, s, (12, im.shape[0] - 14), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 1, cv2.LINE_AA)
    return im


def encode(c, title_label="3D World-Locked Turing Growth"):
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


def make_initial_state(res, world_ext, fountain_r):
    """Seed organic Gray-Scott morphogen B colonies tightly around the circular fountain stone rim + 4 outer plaza sprouts."""
    yy, xx = np.mgrid[0:res, 0:res].astype(np.float32)
    wx = (xx / res * 2.0 - 1.0) * world_ext
    wy = (yy / res * 2.0 - 1.0) * world_ext
    r = np.hypot(wx, wy)
    ang = np.arctan2(wy, wx)
    A = np.ones((res, res), np.float32)
    B = np.zeros((res, res), np.float32)

    rng = np.random.default_rng(42)
    # 16 organic spore nodules ringing the circular fountain outer edge
    for k in range(18):
        th = (k / 18.0) * 2 * math.pi + rng.uniform(-0.08, 0.08)
        rr = fountain_r + rng.uniform(0.45, 0.95)
        sx, sy = rr * math.cos(th), rr * math.sin(th)
        rad = rng.uniform(0.32, 0.58)
        m = ((wx - sx) ** 2 + (wy - sy) ** 2) < rad ** 2
        B[m] = 0.85
        A[m] = 0.25

    # 6 secondary outer courtyard colonies so growth also bridges across the wide stone plaza as the drone rises
    for sx, sy in [(-9.5, -2.5), (9.5, 2.2), (-6.5, 7.8), (7.2, -7.4), (-11.0, 5.0), (11.2, -3.0)]:
        rad = rng.uniform(0.35, 0.55)
        m = ((wx - sx) ** 2 + (wy - sy) ** 2) < rad ** 2
        B[m] = 0.78
        A[m] = 0.30

    # slight organic noise perturbation
    B = np.clip(B + rng.uniform(0, 0.02, B.shape).astype(np.float32) * (B > 0.1), 0.0, 1.0)
    return np.dstack([A, B, np.zeros_like(A), np.ones_like(A)]).astype(np.float32)


def main():
    c = load(sys.argv[1] if len(sys.argv) > 1 and not sys.argv[1].startswith("--") else here / "piece_turing")
    S = scene(c)
    W, H, N, fps = c["W"], c["H"], c["N"], c["fps"]
    f = float(S["f"])
    tc = c.get("turing", {})
    stills = [int(x) for x in sys.argv[sys.argv.index("--still") + 1].split(",")] if "--still" in sys.argv else None

    SIM_RES = 1024
    world_ext = float(tc.get("world_extent_m", 26.0))
    fountain_r = float(tc.get("fountain_r_m", 4.9))
    steps_pf = int(tc.get("steps_per_frame", 22))

    ctx = moderngl.create_standalone_context(require=330)
    sim_prog = ctx.program(vertex_shader=VS, fragment_shader=SIM_FS)
    comp_prog = ctx.program(vertex_shader=VS, fragment_shader=COMP_FS)
    quad = ctx.buffer(np.array([-1, -1, 3, -1, -1, 3], "f4").tobytes())
    va_sim = ctx.vertex_array(sim_prog, [(quad, "2f", "p")])
    va_comp = ctx.vertex_array(comp_prog, [(quad, "2f", "p")])

    # Ping-pong RG/RGBA32F world-space PDE buffers
    init_arr = make_initial_state(SIM_RES, world_ext, fountain_r)
    tA = ctx.texture((SIM_RES, SIM_RES), 4, data=init_arr.tobytes(), dtype="f4")
    tB = ctx.texture((SIM_RES, SIM_RES), 4, dtype="f4")
    for t in (tA, tB):
        t.filter = (moderngl.LINEAR, moderngl.LINEAR)
        t.repeat_x = t.repeat_y = False
    fbA, fbB = ctx.framebuffer([tA]), ctx.framebuffer([tB])

    plate_t = ctx.texture((W, H), 3)
    plate_t.filter = (moderngl.LINEAR, moderngl.LINEAR)
    plate_t.repeat_x = plate_t.repeat_y = False
    out_t = ctx.texture((W, H), 4, dtype="f4")
    fbo_out = ctx.framebuffer([out_t])

    sim_prog["uTexel"].value = (1.0 / SIM_RES, 1.0 / SIM_RES)
    sim_prog["uF"].value = float(tc.get("feed", 0.037))
    sim_prog["uK"].value = float(tc.get("kill", 0.060))
    sim_prog["uDu"].value = 1.0
    sim_prog["uDv"].value = 0.5
    sim_prog["uDt"].value = 1.0
    sim_prog["uWorldExt"].value = world_ext
    sim_prog["uFountainR"].value = fountain_r

    comp_prog["uRes"].value = (float(W), float(H))
    comp_prog["uF"].value = f
    comp_prog["uSun"].value = tuple(map(float, S["sun"]))
    comp_prog["uWorldExt"].value = world_ext
    comp_prog["uFountainR"].value = fountain_r
    comp_prog["uReliefGain"].value = float(tc.get("relief_gain", 1.25))

    ping = [True]

    def step_sim(n_steps, t_sec, pulse=False):
        sim_prog["iTime"].value = t_sec
        sim_prog["uSeedPulse"].value = 1 if pulse else 0
        for _ in range(n_steps):
            src_t, dst_fb = (tA, fbB) if ping[0] else (tB, fbA)
            src_t.use(0)
            sim_prog["uState"].value = 0
            dst_fb.use()
            va_sim.render(moderngl.TRIANGLES)
            ping[0] = not ping[0]

    # Warm up 180 PDE steps so frame 0 opens with crisp organic coral rings already emerging at the fountain rim
    step_sim(180, 0.0, pulse=False)

    def render_one(i, mode=0):
        pl = frame(c, i)
        plate_t.write(pl[:, :, ::-1].tobytes())
        cur_t = tA if ping[0] else tB
        plate_t.use(0)
        cur_t.use(1)
        comp_prog["uPlate"].value = 0
        comp_prog["uState"].value = 1
        comp_prog["uRo"].value = tuple(map(float, S["cam"][i]))
        comp_prog["uCam"].value = tuple(np.asarray(S["rot"][i], np.float32).T.flatten())
        comp_prog["uMode"].value = mode
        fbo_out.use()
        va_comp.render(moderngl.TRIANGLES)
        ctx.finish()
        arr = np.frombuffer(out_t.read(), "f4").reshape(H, W, 4)[::-1]
        img = (np.clip(arr[..., :3], 0, 1) * 255 + 0.5).astype(np.uint8)[:, :, ::-1]
        return pl, img

    if stills:
        rows = []
        todo = set(stills)
        hero = None
        for i in range(max(stills) + 1):
            if i > 0:
                step_sim(steps_pf, i / fps, pulse=(i < 45 and i % 15 == 0))
            if i in todo:
                pl, comp = render_one(i, mode=0)
                _, diag = render_one(i, mode=1)
                if i == stills[len(stills) // 2]:
                    hero = comp
                t_im = lambda im, s: label(cv2.resize(im, (640, 360)), s)
                rows.append(np.hstack([t_im(pl, f"plate {i}"), t_im(diag, "3D Turing normal field"), t_im(comp, "3D Turing growth")]))
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
        if i > 0:
            step_sim(steps_pf, i / fps, pulse=(i < 45 and i % 15 == 0))
        _, comp = render_one(i, mode=0)
        cv2.imwrite(str(comp_dir / f"{i:04d}.png"), comp, [cv2.IMWRITE_PNG_COMPRESSION, 1])
        if i % 40 == 0 or i == N - 1:
            print(f"frame {i}/{N} ({time.time() - ts:.1f}s)", flush=True)
    encode(c)


if __name__ == "__main__":
    main()
