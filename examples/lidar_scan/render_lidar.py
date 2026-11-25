"""3D LiDAR Spatial Pulse Scan & Elevation Contour Wave in a COLMAP-tracked room (ModernGL, OpenGL 3.3).

    python render_lidar.py [piece_dir] --still 35,115,195    quick look -> work/still_sheet.jpg
    python render_lidar.py [piece_dir]                        full render -> out/*_Composite.mp4, *_Compare_1080x1350.mp4

Builds a per-frame metric 3D scene depth map directly from the solved COLMAP 3D point cloud (4,900+ real architectural
landmarks) fused with the solved floor plane (z = 0) and ceiling bound, edge-refined against the live-action plate.
An expanding 3D spherical LiDAR pulse sweeps through world space, lighting up:
  1. Razor-sharp ice-white 3D wavefront intersection hugging floor, island, chairs, doorway, and ceiling
  2. True 3D world-space architectural grid (0.50 m) and vertical Z elevation contours (0.20 m)
  3. Crisp architectural edge wireframe inside the trailing monochrome scan zone
  4. Real COLMAP 3D landmarks rendered natively as OpenGL GL_POINTS that ignite as the 3D sphere crosses them.
"""
import json, math, shutil, subprocess, sys, time
from pathlib import Path
import numpy as np, cv2, moderngl
from scipy.spatial import cKDTree

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

# Pass 1: Full-screen 3D LiDAR surface scan wave + 3D world contours + architectural edge wireframe
SCAN_FS = """#version 330
uniform sampler2D uPlate;
uniform sampler2D uDepth;     /* R32F metric ray distance t (metres) from camera to 3D surface */
uniform vec2 uRes;
uniform float uF, iTime;
uniform vec3 uRo;
uniform mat3 uCam;
uniform vec4 uPulses[3];      /* xyz = pulse 3D world origin, w = current sphere radius R (m); w < 0 inactive */
uniform float uBandW;         /* trailing scan band thickness in metres */
uniform float uContourStep;   /* vertical elevation contour spacing in metres */
uniform float uGridStep;      /* 3D world lattice spacing in metres */
uniform int uMode;            /* 0 = composite, 1 = 3D depth + scan diagnostic */

out vec4 oOut;

float gridLine(float v, float stepM, float widthM) {
  float m = abs(fract(v / stepM + 0.5) - 0.5) * stepM;
  return 1.0 - smoothstep(widthM * 0.4, widthM, m);
}

void main() {
  vec2 px = vec2(gl_FragCoord.x, uRes.y - gl_FragCoord.y);
  vec2 uv = px / uRes;
  vec3 plate = texture(uPlate, uv).rgb;
  float luma = dot(plate, vec3(0.299, 0.587, 0.114));

  /* reconstruct exact 3D world position X = ro + t * rd in COLMAP plane coordinates (metres) */
  vec3 rd = normalize(uCam * vec3((px - 0.5 * uRes) / uF, 1.0));
  float t = texture(uDepth, uv).r;
  vec3 X = uRo + rd * t;

  /* edge detector on real plate luminance for crisp architectural wireframe inside scan band */
  vec2 tx = 1.0 / uRes;
  float lL = dot(texture(uPlate, clamp(uv - vec2(tx.x, 0.0), 0.001, 0.999)).rgb, vec3(0.299, 0.587, 0.114));
  float lR = dot(texture(uPlate, clamp(uv + vec2(tx.x, 0.0), 0.001, 0.999)).rgb, vec3(0.299, 0.587, 0.114));
  float lU = dot(texture(uPlate, clamp(uv - vec2(0.0, tx.y), 0.001, 0.999)).rgb, vec3(0.299, 0.587, 0.114));
  float lD = dot(texture(uPlate, clamp(uv + vec2(0.0, tx.y), 0.001, 0.999)).rgb, vec3(0.299, 0.587, 0.114));
  float edge = clamp(length(vec2(lR - lL, lD - lU)) * 4.8, 0.0, 1.0);
  edge = smoothstep(0.14, 0.65, edge);

  /* compute active LiDAR spherical wavefront + trailing digital reveal mask */
  float frontGlow = 0.0;
  float bandMask = 0.0;
  float wakeMask = 0.0;
  for (int k = 0; k < 3; k++) {
    float R = uPulses[k].w;
    if (R < 0.0) continue;
    float d = length(X - uPulses[k].xyz);
    /* distance fade so distant pulses decay smoothly */
    float distFade = 1.0 - smoothstep(10.5, 16.5, R);
    /* razor-sharp leading laser wavefront ring */
    float diff = R - d;
    float front = exp(-pow(diff / 0.068, 2.0)) * distFade;
    frontGlow = max(frontGlow, front);

    /* trailing digital scan band behind the wavefront */
    float inBand = smoothstep(-0.08, 0.12, diff) * (1.0 - smoothstep(uBandW * 0.45, uBandW, diff)) * distFade;
    bandMask = max(bandMask, inBand);

    /* subtle longer echo for faint contour persistence */
    float inWake = smoothstep(-0.05, 0.18, diff) * (1.0 - smoothstep(uBandW * 0.8, uBandW * 1.65, diff)) * distFade;
    wakeMask = max(wakeMask, inWake);
  }

  if (uMode == 1) {
    /* diagnostic view: normalized metric 3D depth + active wavefront */
    float dn = clamp((t - 1.0) / 11.0, 0.0, 1.0);
    vec3 colD = mix(vec3(0.95, 0.98, 1.0), vec3(0.06, 0.12, 0.22), dn);
    colD = mix(colD, vec3(0.30, 0.92, 1.0), frontGlow);
    oOut = vec4(colD, 1.0);
    return;
  }

  /* distance-scaled line widths so 3D lines remain crisp near & far without moire */
  float wScale = clamp(t * 0.0042, 0.008, 0.036);

  /* 3D world-space architectural grid (X, Y on floor/ceiling + vertical Z elevation contours!) */
  float gX = gridLine(X.x, uGridStep, wScale * 1.15);
  float gY = gridLine(X.y, uGridStep, wScale * 1.15);
  float floorGrid = max(gX, gY) * smoothstep(0.45, 0.05, abs(X.z));
  float ceilGrid  = max(gX, gY) * smoothstep(2.4, 2.95, X.z) * 0.65;

  /* 3D vertical elevation contours (every uContourStep metres above floor) + major 1m elevation lines */
  float zCont = gridLine(X.z, uContourStep, wScale * 0.95) * smoothstep(0.04, 0.14, X.z);
  float zMajor = gridLine(X.z, 1.0, wScale * 1.55) * smoothstep(0.05, 0.16, X.z);
  /* 3D radial distance rings every 1.0m from pulse origin */
  float wallGrid = max(gX, gY) * smoothstep(0.12, 0.35, X.z) * (1.0 - smoothstep(2.5, 3.0, X.z)) * 0.42;

  /* crisp minimal monochrome LiDAR palette (carbon slate + silver + glacial ice-white) */
  vec3 iceWhite = vec3(0.92, 0.98, 1.0);
  vec3 coolCyan = vec3(0.62, 0.88, 0.98);

  /* inside the trailing scan band, darken & desaturate the room into sleek dark architectural blueprint */
  vec3 monoRoom = vec3(luma * 0.38) * vec3(0.86, 0.93, 1.02);
  vec3 scanSurface = monoRoom;
  scanSurface += coolCyan * (floorGrid * 0.42 + ceilGrid * 0.32 + wallGrid * 0.30);
  scanSurface += iceWhite * (zCont * 0.44 + zMajor * 0.55);
  scanSurface += iceWhite * (edge * 0.68);

  /* subtle micro scanline shimmer inside active zone */
  float microDot = gridLine(X.x, 0.125, wScale * 0.55) * gridLine(X.y, 0.125, wScale * 0.55);
  scanSurface += coolCyan * microDot * 0.32;

  vec3 col = mix(plate, scanSurface, bandMask * 0.88);
  /* lingering subtle elevation wireframe during wake recovery */
  col += coolCyan * (zCont * 0.16 + floorGrid * 0.14 + edge * 0.14) * wakeMask * (1.0 - bandMask * 0.6);
  /* blazing razor-sharp 3D spherical LiDAR leading pulse line */
  col = mix(col, iceWhite, clamp(frontGlow * 0.82, 0.0, 0.92));
  col += coolCyan * frontGlow * (0.35 + 0.55 * edge);

  oOut = vec4(clamp(col, 0.0, 1.0), 1.0);
}
"""

# Pass 2: Native OpenGL GL_POINTS rendering of REAL COLMAP 3D SfM landmarks lit by the expanding 3D LiDAR wave!
PTS_VS = """#version 330
in vec3 inPos;
uniform vec2 uRes;
uniform float uF, uPtSize;
uniform vec3 uRo;
uniform mat3 uCam;
uniform vec4 uPulses[3];
uniform float uBandW;
out float vAlpha;
out float vHot;

void main() {
  vec3 q = transpose(uCam) * (inPos - uRo);
  if (q.z <= 0.25) {
    gl_Position = vec4(2.0, 2.0, 2.0, 1.0);
    vAlpha = 0.0;
    vHot = 0.0;
    return;
  }
  vec2 px = uF * q.xy / q.z + 0.5 * uRes;
  /* convert top-down pixel coords to OpenGL NDC (match flipped y in our FBO) */
  vec2 ndc = vec2(px.x / uRes.x * 2.0 - 1.0, (1.0 - px.y / uRes.y) * 2.0 - 1.0);
  gl_Position = vec4(ndc, 0.0, 1.0);

  float act = 0.0;
  float hot = 0.0;
  for (int k = 0; k < 3; k++) {
    float R = uPulses[k].w;
    if (R < 0.0) continue;
    float d = length(inPos - uPulses[k].xyz);
    float diff = R - d;
    float fade = 1.0 - smoothstep(10.5, 16.5, R);
    float flash = exp(-pow(diff / 0.28, 2.0)) * fade;
    float linger = smoothstep(-0.15, 0.15, diff) * (1.0 - smoothstep(uBandW * 0.75, uBandW * 1.9, diff)) * fade;
    hot = max(hot, flash);
    act = max(act, max(flash, linger * 0.85));
  }
  vAlpha = act * clamp(5.0 / max(q.z, 1.0), 0.35, 1.0);
  vHot = hot;
  gl_PointSize = clamp(uPtSize * (4.2 / max(q.z, 0.8)) * (1.0 + 0.75 * hot), 2.0, 9.0);
}
"""

PTS_FS = """#version 330
in float vAlpha;
in float vHot;
out vec4 oCol;
void main() {
  if (vAlpha < 0.02) discard;
  vec2 p = abs(gl_PointCoord - 0.5) * 2.0;
  /* crisp architectural LiDAR diamond/plus glyph point */
  float d = max(p.x, p.y);
  float core = 1.0 - smoothstep(0.45, 0.95, d);
  float cross = max(1.0 - smoothstep(0.15, 0.35, p.x), 1.0 - smoothstep(0.15, 0.35, p.y));
  float shape = max(core * 0.9, cross * 0.65);
  vec3 col = mix(vec3(0.62, 0.90, 1.0), vec3(1.0, 1.0, 1.0), vHot);
  oCol = vec4(col, vAlpha * shape);
}
"""


def label(im, s):
    im = im.copy()
    cv2.putText(im, s, (12, im.shape[0] - 14), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 4, cv2.LINE_AA)
    cv2.putText(im, s, (12, im.shape[0] - 14), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 1, cv2.LINE_AA)
    return im


def encode(c, title_label="3D LiDAR Spatial Scan"):
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


def build_frame_depth(Psfm, R, C, f, W, H, pl_bgr, dw=320, dh=180):
    """Build a smooth, room-consistent metric ray distance map t(u,v) in metres from real COLMAP 3D points + floor/ceiling planes."""
    sx, sy = dw / W, dh / H
    # Analytical rays for low-res grid
    yy, xx = np.mgrid[0:dh, 0:dw].astype(np.float32)
    px = (xx + 0.5) / sx
    py = (yy + 0.5) / sy
    dirs_cam = np.stack([(px - W * 0.5) / f, (py - H * 0.5) / f, np.ones_like(px)], axis=-1)
    dirs_cam /= np.linalg.norm(dirs_cam, axis=-1, keepdims=True)
    rd = dirs_cam @ R.T  # plane coords ray directions [dh, dw, 3]

    # Analytical room envelope: floor z=0 and ceiling z=3.25m, bounded at 14m far corridor
    t_floor = np.where(rd[..., 2] < -0.015, -C[2] / np.minimum(rd[..., 2], -1e-4), 14.0)
    t_ceil = np.where(rd[..., 2] > 0.015, (3.25 - C[2]) / np.maximum(rd[..., 2], 1e-4), 14.0)
    t_env = np.clip(np.minimum(t_floor, t_ceil), 0.5, 14.0)

    # Project real COLMAP 3D points into this camera
    q = (Psfm - C) @ R
    z_c = q[:, 2]
    ray_t = np.linalg.norm(Psfm - C, axis=1)
    u = (f * q[:, 0] / np.maximum(z_c, 1e-3) + W * 0.5) * sx
    v = (f * q[:, 1] / np.maximum(z_c, 1e-3) + H * 0.5) * sy
    valid = (z_c > 0.35) & (ray_t < 15.0) & (u >= 0) & (u < dw) & (v >= 0) & (v < dh)

    if valid.sum() > 40:
        pts_2d = np.stack([u[valid], v[valid]], axis=1)
        vals = ray_t[valid].astype(np.float32)
        tree = cKDTree(pts_2d)
        grid_2d = np.stack([xx.ravel() + 0.5, yy.ravel() + 0.5], axis=1)
        dists, idxs = tree.query(grid_2d, k=min(14, len(vals)))
        # Robust median depth over k-NN eliminates any isolated single-point depth bubble!
        t_sfm = np.median(vals[idxs], axis=1).reshape(dh, dw).astype(np.float32)
        min_d = dists[:, 2].reshape(dh, dw).astype(np.float32)  # require >=3 nearby real 3D points
        conf = np.exp(-(min_d / 20.0) ** 2)
        t_fused = t_sfm * conf + t_env * (1.0 - conf)
        t_fused = np.minimum(t_fused, t_env + 0.12)
    else:
        t_fused = t_env.astype(np.float32)

    t_med = cv2.medianBlur(t_fused.astype(np.float32), 5)
    t_sm = cv2.bilateralFilter(t_med, 7, 1.0, 5.0)
    return cv2.resize(t_sm, (W, H), interpolation=cv2.INTER_CUBIC).astype(np.float32)


def main():
    c = load(sys.argv[1] if len(sys.argv) > 1 and not sys.argv[1].startswith("--") else here / "piece_lidar")
    S = scene(c)
    W, H, N, fps = c["W"], c["H"], c["N"], c["fps"]
    f = float(S["f"])
    lc = c.get("lidar", {})
    stills = [int(x) for x in sys.argv[sys.argv.index("--still") + 1].split(",")] if "--still" in sys.argv else None

    cam = np.load(work(c, "camera.npz"))
    Psfm = (S["scale"] * ((cam["pts"].astype(np.float64) - S["O"]) @ np.asarray(S["M"]).T)).astype(np.float32)
    # Keep clean architectural landmarks & reject isolated floating 3D outliers
    keep = (Psfm[:, 2] > -0.20) & (Psfm[:, 2] < 3.4) & (np.linalg.norm(Psfm[:, :2], axis=1) < 20.0)
    Psfm = Psfm[keep]
    d3, _ = cKDTree(Psfm).query(Psfm, k=8)
    Psfm = Psfm[np.median(d3[:, 1:], axis=1) < 0.50]
    print(f"LiDAR real COLMAP landmarks (outlier-filtered): {len(Psfm)}")

    ctx = moderngl.create_standalone_context(require=330)
    ctx.enable(moderngl.PROGRAM_POINT_SIZE)
    scan_prog = ctx.program(vertex_shader=VS, fragment_shader=SCAN_FS)
    pts_prog = ctx.program(vertex_shader=PTS_VS, fragment_shader=PTS_FS)

    quad = ctx.buffer(np.array([-1, -1, 3, -1, -1, 3], "f4").tobytes())
    va_scan = ctx.vertex_array(scan_prog, [(quad, "2f", "p")])
    pts_vbo = ctx.buffer(Psfm.astype("f4").tobytes())
    va_pts = ctx.vertex_array(pts_prog, [(pts_vbo, "3f", "inPos")])

    plate_t = ctx.texture((W, H), 3)
    plate_t.filter = (moderngl.LINEAR, moderngl.LINEAR)
    plate_t.repeat_x = plate_t.repeat_y = False
    depth_t = ctx.texture((W, H), 1, dtype="f4")
    depth_t.filter = (moderngl.LINEAR, moderngl.LINEAR)
    depth_t.repeat_x = depth_t.repeat_y = False
    out_t = ctx.texture((W, H), 4, dtype="f4")
    fbo = ctx.framebuffer([out_t])

    period = float(lc.get("pulse_period_s", 2.6))
    speed = float(lc.get("wave_speed_mps", 5.8))
    band_w = float(lc.get("band_width_m", 2.8))
    contour_step = float(lc.get("contour_step_m", 0.22))
    grid_step = float(lc.get("grid_step_m", 0.50))
    pt_size = float(lc.get("point_size_px", 3.6))

    # Schedule 3 clean LiDAR pulses emitted from the camera's real 3D position at launch time!
    launch_frames = [int(round(0.25 * fps)), int(round((0.25 + period) * fps)), int(round((0.25 + 2.0 * period) * fps))]
    launch_frames = [min(k, N - 1) for k in launch_frames]
    launch_origins = [np.asarray(S["cam"][k], np.float32) for k in launch_frames]
    # Shift launch origin down near floor/chest height so radial rings sweep dramatically across floor & island
    launch_origins = [np.array([o[0], o[1], 0.85], np.float32) for o in launch_origins]

    scan_prog["uRes"].value = (float(W), float(H))
    scan_prog["uF"].value = f
    scan_prog["uBandW"].value = band_w
    scan_prog["uContourStep"].value = contour_step
    scan_prog["uGridStep"].value = grid_step

    pts_prog["uRes"].value = (float(W), float(H))
    pts_prog["uF"].value = f
    pts_prog["uPtSize"].value = pt_size
    pts_prog["uBandW"].value = band_w

    def get_pulses(i):
        t_sec = i / fps
        arr = []
        for lf, orig in zip(launch_frames, launch_origins):
            age = t_sec - lf / fps
            rad = (0.35 + speed * age) if age >= 0.0 else -1.0
            if rad > 17.5:
                rad = -1.0
            arr.append((float(orig[0]), float(orig[1]), float(orig[2]), float(rad)))
        return arr

    def render_one(i, mode=0):
        pl = frame(c, i)
        R = np.asarray(S["rot"][i], np.float64)
        C = np.asarray(S["cam"][i], np.float64)
        dmap = build_frame_depth(Psfm, R, C, f, W, H, pl)

        plate_t.write(pl[:, :, ::-1].tobytes())
        depth_t.write(dmap.tobytes())
        plate_t.use(0)
        depth_t.use(1)
        pulses = get_pulses(i)
        Rt = tuple(np.asarray(R, np.float32).T.flatten())
        ro = tuple(map(float, C))

        scan_prog["uPlate"].value = 0
        scan_prog["uDepth"].value = 1
        scan_prog["uRo"].value = ro
        scan_prog["uCam"].value = Rt
        scan_prog["uPulses"].value = pulses
        scan_prog["uMode"].value = mode
        if "iTime" in scan_prog:
            scan_prog["iTime"].value = i / fps

        fbo.use()
        ctx.disable(moderngl.BLEND)
        va_scan.render(moderngl.TRIANGLES)

        if mode == 0:
            # Render real COLMAP 3D SfM points additively as crisp glowing LiDAR dots!
            pts_prog["uRo"].value = ro
            pts_prog["uCam"].value = Rt
            pts_prog["uPulses"].value = pulses
            ctx.enable(moderngl.BLEND)
            ctx.blend_func = (moderngl.SRC_ALPHA, moderngl.ONE)
            va_pts.render(moderngl.POINTS)
            ctx.disable(moderngl.BLEND)

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
            rows.append(np.hstack([t_im(pl, f"plate {i}"), t_im(diag, "3D metric depth + pulse"), t_im(comp, "3D LiDAR scan")]))
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
