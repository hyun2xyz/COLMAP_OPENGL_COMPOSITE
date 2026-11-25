"""Cartoon slit-scan: a 1920s public-domain Felix the Cat close-up, his head puffs up and wobbles through time (ModernGL, GL 3.3).

    python cartoon_slit.py [piece_dir] --still 40,120,200      quick look -> work/still_sheet.jpg
    python cartoon_slit.py [piece_dir]                         all frames -> out/*_Composite_1440x1080.mp4, *_Compare_1080x1350.mp4

Per frame:
  head     Felix is the biggest black shape; his head is the top part of it. Its centre and size are tracked from the black
           mask (threshold, largest component, top head_frac of its height), smoothed over frames.
  bloat    a lens around the head magnifies it (centre up to 1 / (1 - amp) times), and the amount springs in "boing" pulses
           (bloat.base + decaying sine kicks every bloat.every_s), with squash and stretch (x and y scaled oppositely).
  jelly    slit-scan inside the head: each row of the face comes from a moment jelly.rate frames per px further back from
           the top of the head, so the face wobbles whenever he moves or turns.
  halo     outside the head the picture gets older with distance (halo.rate frames per px), so old heads trail around him.
Frames sit in a GPU texture-array ring buffer (last K), read between neighbouring frames. No colour is added.
Source: "Felix the Cat in Two-Lip Time" (Pat Sullivan, 1926), public domain (US, published before 1930), Internet Archive."""
import json, math, shutil, subprocess, sys, time
from pathlib import Path
import numpy as np, cv2, moderngl

here = Path(__file__).resolve().parent
VS = "#version 330\nin vec2 p; void main(){ gl_Position = vec4(p, 0.0, 1.0); }"
FS = """#version 330
uniform sampler2DArray uHist; uniform vec2 uRes; uniform int uI, uK;
uniform vec2 uC; uniform float uR, uAmp, uSq, uJelly, uHalo;
out vec4 o;
float hist(vec2 uv, float tau){
  tau = clamp(tau, 0.0, float(min(uI, uK - 1))); float t0 = floor(tau), a = tau - t0;
  int l0 = int(mod(float(uI) - t0, float(uK))), l1 = int(mod(float(uI) - t0 - 1.0, float(uK)));
  if (t0 + 1.0 > float(min(uI, uK - 1))) l1 = l0;
  return mix(texture(uHist, vec3(uv, float(l0))).r, texture(uHist, vec3(uv, float(l1))).r, a);
}
void main(){
  vec2 px = vec2(gl_FragCoord.x, uRes.y - gl_FragCoord.y);
  vec2 sq = vec2(1.0 + uSq, 1.0 - uSq);                                            /* squash and stretch */
  vec2 d = (px - uC) / sq; float r = length(d) / uR;
  vec2 src = px;
  if (r < 1.0){ float k = 1.0 - uAmp * pow(1.0 - r * r, 2.0); src = uC + d * k; } /* bloat: pull the sample towards the centre */
  float inside = 1.0 - smoothstep(0.75, 1.05, r);
  float tau = inside * max(src.y - (uC.y - uR), 0.0) * uJelly                     /* jelly: lower rows of the head are older */
            + max(r - 1.0, 0.0) * uR * uHalo;                                      /* halo: older with distance from the head */
  o = vec4(vec3(hist(clamp(src, vec2(0.5), uRes - 0.5) / uRes, tau)), 1.0);
}"""

def track_heads(frames, cfg):
    """centre (x, y) and radius of Felix's head per frame from the black silhouette, smoothed"""
    out, prev = [], None
    for g in frames:
        blk = (cv2.GaussianBlur(g, (0, 0), 2) < cfg.get("black", 70)).astype(np.uint8)
        blk = cv2.morphologyEx(blk, cv2.MORPH_OPEN, np.ones((7, 7), np.uint8))
        n, lab, st, cen = cv2.connectedComponentsWithStats(blk)
        cand = [k for k in range(1, n) if st[k, cv2.CC_STAT_AREA] > cfg.get("min_area", 15000)]
        if cand:
            k = max(cand, key=lambda k: st[k, cv2.CC_STAT_AREA]); x0, y0, w, h = st[k, :4]
            top = (lab == k) & (np.arange(g.shape[0])[:, None] < y0 + cfg.get("band_frac", 0.3) * h)   # ears and crown: always black
            ys, xs = np.nonzero(top)
            hc = cfg.get("head_frac", 0.45) * h                                   # the face itself can be white: place it from the crown down
            c = np.array([xs.mean(), y0 + 0.55 * hc, cfg.get("radius_scale", 0.6) * hc])
        else:
            c = prev if prev is not None else np.array([g.shape[1] / 2, g.shape[0] / 2, 200.0])
        prev = c if prev is None else cfg.get("smooth", 0.35) * c + (1 - cfg.get("smooth", 0.35)) * prev
        out.append(prev.copy())
    return np.array(out)

def bloat_curve(n, fps, b):
    """base + decaying sine kicks ("boing") every bloat.every_s; squash follows the kick's velocity"""
    t = np.arange(n) / fps; amp = np.full(n, b.get("base", 0.25)); sq = np.zeros(n)
    for t0 in np.arange(b.get("first_s", 0.3), t[-1], b.get("every_s", 1.1)):
        a = t - t0; on = a >= 0; env = np.exp(-a[on] / b.get("decay_s", 0.35))
        amp[on] += b.get("kick", 0.35) * env * np.abs(np.sin(2 * math.pi * b.get("hz", 3.0) * a[on]))
        sq[on] += b.get("squash", 0.12) * env * np.cos(2 * math.pi * b.get("hz", 3.0) * a[on])
    return np.clip(amp, 0, 0.85), sq

def main():
    d = Path(sys.argv[1] if len(sys.argv) > 1 and not sys.argv[1].startswith("--") else here / "piece_cartoon").resolve()
    c = json.loads((d / "piece.json").read_text(encoding="utf-8")); W, H = c.get("size", [1440, 1080])
    stills = [int(x) for x in sys.argv[sys.argv.index("--still") + 1].split(",")] if "--still" in sys.argv else None
    wk = d / "work"; fr = wk / "frames"
    if not (fr / "0000.png").exists():                                            # deinterlace, square pixels, 1440x1080 grey
        fr.mkdir(parents=True, exist_ok=True)
        subprocess.run(["ffmpeg", "-v", "error", "-y", "-ss", str(c["start_s"]), "-i", str((d / c["source"]).resolve()), "-t", str(c["dur_s"]),
                        "-vf", f"yadif,scale={W}:{H}:flags=lanczos,setsar=1,format=gray", "-start_number", "0", str(fr / "%04d.png")], check=True)
    files = sorted(fr.glob("*.png")); N = len(files); fps = float(c.get("fps", 30000 / 1001))
    frames = [cv2.imread(str(p), cv2.IMREAD_GRAYSCALE) for p in files]
    if c.get("keys"):                                                             # hand keys [frame, x, y, r] beat the tracker on a cartoon
        kk = np.array(c["keys"], float); heads = np.stack([np.interp(np.arange(N), kk[:, 0], kk[:, j]) for j in (1, 2, 3)], 1)
    else:
        heads = track_heads(frames, c.get("head", {}))
    amp, sq = bloat_curve(N, fps, c.get("bloat", {}))
    K = int(c.get("K", 48)); ctx = moderngl.create_standalone_context(require=330); prog = ctx.program(vertex_shader=VS, fragment_shader=FS)
    quad = ctx.buffer(np.array([-1, -1, 3, -1, -1, 3], "f4").tobytes()); va = ctx.vertex_array(prog, [(quad, "2f", "p")])
    hist = ctx.texture_array((W, H, K), 1); hist.filter = (moderngl.LINEAR, moderngl.LINEAR); hist.repeat_x = hist.repeat_y = False
    out_t = ctx.texture((W, H), 4); fbo = ctx.framebuffer([out_t])
    prog["uRes"].value = (W, H); prog["uK"].value = K; prog["uJelly"].value = c.get("jelly", {}).get("rate", 0.06); prog["uHalo"].value = c.get("halo", {}).get("rate", 0.03)
    def render(i):
        hist.write(frames[i].tobytes(), viewport=(0, 0, i % K, W, H, 1))
        x, y, r = heads[i]; prog["uI"].value = i; prog["uC"].value = (float(x), float(y)); prog["uR"].value = float(r * c.get("lens_scale", 1.6))
        prog["uAmp"].value = float(amp[i]); prog["uSq"].value = float(sq[i])
        hist.use(0); prog["uHist"].value = 0; fbo.use(); va.render(moderngl.TRIANGLES)
        return np.frombuffer(out_t.read(), np.uint8).reshape(H, W, 4)[::-1, :, 0].copy()
    if stills:
        rows = []
        for i in range(max(0, min(stills) - K), max(stills) + 1):
            o = render(i)
            if i in stills:
                a = cv2.cvtColor(frames[i], cv2.COLOR_GRAY2BGR); x, y, r = heads[i]; cv2.circle(a, (int(x), int(y)), int(r * c.get("lens_scale", 1.6)), (0, 200, 255), 3)
                rows.append(np.hstack([cv2.resize(a, (640, 480)), cv2.resize(cv2.cvtColor(o, cv2.COLOR_GRAY2BGR), (640, 480))]))
        cv2.imwrite(str(wk / "still_sheet.jpg"), np.vstack(rows), [cv2.IMWRITE_JPEG_QUALITY, 90]); print("stills ->", wk / "still_sheet.jpg"); return
    cp = wk / "comp"; shutil.rmtree(cp, ignore_errors=True); cp.mkdir()
    ts = time.time()
    for i in range(N):
        cv2.imwrite(str(cp / f"{i:04d}.png"), render(i), [cv2.IMWRITE_PNG_COMPRESSION, 1])
        if i % 90 == 0: print("frame", i, "%.0fs" % (time.time() - ts), flush=True)
    out = d / "out"; out.mkdir(exist_ok=True); nm = c["name"]; rate = "30000/1001"
    tags = ["-c:v", "libx264", "-crf", "15", "-preset", "slow", "-pix_fmt", "yuv420p", "-color_primaries", "bt709", "-color_trc", "bt709", "-colorspace", "bt709", "-movflags", "+faststart"]
    run = lambda a: subprocess.run(["ffmpeg", "-v", "error", "-y"] + a + tags, check=True)
    run(["-framerate", rate, "-i", str(cp / "%04d.png"), str(out / f"{nm}_Composite_1440x1080.mp4")])
    run(["-framerate", rate, "-i", str(cp / "%04d.png"), "-framerate", rate, "-i", str(fr / "%04d.png"), "-filter_complex",
         "[0]scale=1080:810[a];[1]scale=720:540[b];[b]pad=1080:540:180:0:color=black[bb];[a][bb]vstack", str(out / f"{nm}_Compare_1080x1350.mp4")])
    print("encoded ->", out)

if __name__ == "__main__":
    main()
