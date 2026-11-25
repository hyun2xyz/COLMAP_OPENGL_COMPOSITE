"""Body slit-scan, the classic streak kind: one slit line, laid out over time (ModernGL, OpenGL 3.3).

    python render_slit.py [piece_dir] --still 80,150,230      quick look -> work/still_sheet.jpg
    python render_slit.py [piece_dir]                         all frames -> work/comp, work/pass/timemap, out/*.mp4

Like strip / slit-scan photography (Trumbull's 2001 corridor, the "3D slitscan" TouchDesigner tutorials): pick a slit,
and lay what passed through it out over time. Here the slit is a line that hugs the dancer's silhouette on one side
(slit.side, +1 = right): per image row it sits slit.offset_m outside the outermost body pixel (MediaPipe mask), smoothed
over rows and frames. Left of the slit the plate is untouched. Right of it, a pixel at distance d from the slit shows the
slit's own column from d * slit.rate frames ago (last slit.K frames are kept in a GPU texture array, blended between
neighbouring frames). So the moving body leaves long horizontal streaks of itself behind, and the still wall turns into
stripes, the slit-scan look. slit.rows: "all" streaks every row; "body" only the rows near the body (soft falloff).
No colour is added: every pixel is the plate's own colour from some moment."""
import json, shutil, sys, time
from pathlib import Path
import numpy as np, cv2, moderngl

here = Path(__file__).resolve().parent
sys.path.insert(0, str(here.parent / "shader_live" / "pipeline")); sys.path.insert(0, str(here.parent / "space_distortion"))
from slv import load, work, frame, scene   # noqa: E402
import render as sd                        # noqa: E402  (thermal, label, encode)

VS = "#version 330\nin vec2 p; void main(){ gl_Position = vec4(p, 0.0, 1.0); }"
FS = """#version 330
uniform sampler2DArray uHist; uniform sampler2D uSlit; uniform vec2 uRes; uniform int uI, uK;
uniform float uRate, uFeather, uSide;
layout(location=0) out vec4 oComp; layout(location=1) out vec4 oTau;
vec3 hist(vec2 uv, float tau){                                                    /* plate at frame i - tau, blended */
  tau = clamp(tau, 0.0, float(min(uI, uK - 1))); float t0 = floor(tau), a = tau - t0;
  int l0 = int(mod(float(uI) - t0, float(uK))), l1 = int(mod(float(uI) - t0 - 1.0, float(uK)));
  if (t0 + 1.0 > float(min(uI, uK - 1))) l1 = l0;
  return mix(texture(uHist, vec3(uv, float(l0))).rgb, texture(uHist, vec3(uv, float(l1))).rgb, a);
}
void main(){
  vec2 px = vec2(gl_FragCoord.x, uRes.y - gl_FragCoord.y); vec2 uv = px / uRes;
  vec2 s = texture(uSlit, vec2(0.5, uv.y)).rg;                                    /* slit x (px) for this row, how much this row streaks */
  float d = (px.x - s.x) * uSide;                                                 /* distance past the slit */
  vec3 now = hist(uv, 0.0);
  float tau = max(d, 0.0) * uRate;
  vec3 streak = hist(vec2((s.x + 0.5) / uRes.x, uv.y), tau);                     /* the slit column, tau frames ago */
  float w = smoothstep(0.0, uFeather, d) * s.y;
  oComp = vec4(mix(now, streak, w), 1.0); oTau = vec4(tau * w, s.x, w, 1.0);
}"""

def slit_line(m, side, off_px, rows, prev, alpha):
    """per row: x of the slit, just outside the outermost body pixel on `side`; rows without body interpolated; smoothed"""
    H, W = m.shape; body = m > 127; has = body.any(1)
    if side > 0: edge = np.where(has, W - 1 - np.argmax(body[:, ::-1], 1), np.nan)
    else: edge = np.where(has, np.argmax(body, 1), np.nan)
    y = np.arange(H, dtype=np.float64)
    if not has.any(): xs, wgt = np.full(H, W * 0.5), np.zeros(H)
    else:
        xs = np.interp(y, y[has], edge[has]); xs = cv2.GaussianBlur(xs.reshape(-1, 1), (1, 0), sigmaX=0.1, sigmaY=25).ravel() + side * off_px
        if rows == "all": wgt = np.ones(H)
        else:
            dist = np.min(np.abs(y[:, None] - y[has][None, ::8]), 1); wgt = np.exp(-(dist / 60.0) ** 2)
    if prev is not None: xs = alpha * xs + (1 - alpha) * prev                    # temporal smoothing: the slit must not jitter
    return np.clip(xs, 0, W - 1), wgt

def main():
    c = load(sys.argv[1] if len(sys.argv) > 1 and not sys.argv[1].startswith("--") else here / "piece"); S = scene(c)
    W, H, N, fps = c["W"], c["H"], c["N"], c["fps"]; f = float(S["f"]); sc = c["slit"]; K = int(sc.get("K", 64))
    stills = [int(x) for x in sys.argv[sys.argv.index("--still") + 1].split(",")] if "--still" in sys.argv else None
    L = np.load(work(c, "limbs.npz")); zbody = float(np.median(L["depth"]))
    side = float(sc.get("side", 1)); off_px = f * sc.get("offset_m", 0.12) / zbody
    ctx = moderngl.create_standalone_context(require=330); prog = ctx.program(vertex_shader=VS, fragment_shader=FS)
    quad = ctx.buffer(np.array([-1, -1, 3, -1, -1, 3], "f4").tobytes()); va = ctx.vertex_array(prog, [(quad, "2f", "p")])
    hist = ctx.texture_array((W, H, K), 3); hist.filter = (moderngl.LINEAR, moderngl.LINEAR); hist.repeat_x = hist.repeat_y = False
    slit_t = ctx.texture((1, H), 2, dtype="f4"); slit_t.filter = (moderngl.LINEAR, moderngl.LINEAR)
    oc, ot = ctx.texture((W, H), 4, dtype="f4"), ctx.texture((W, H), 4, dtype="f4"); fbo = ctx.framebuffer([oc, ot])
    for k, v in (("uRes", (W, H)), ("uK", K), ("uRate", sc.get("rate", 0.05)), ("uFeather", sc.get("feather_px", 24.0)), ("uSide", side)): prog[k].value = v
    state = {"prev": None}

    def render(i):
        pl = frame(c, i); hist.write(pl[:, :, ::-1].tobytes(), viewport=(0, 0, i % K, W, H, 1))
        m = cv2.imread(str(work(c, "mask", f"{i:04d}.png")), cv2.IMREAD_GRAYSCALE)
        xs, wgt = slit_line(m, side, off_px, sc.get("rows", "all"), state["prev"], sc.get("smooth", 0.35)); state["prev"] = xs
        slit_t.write(np.stack([xs, wgt], 1).astype("f4").tobytes())
        prog["uI"].value = i; hist.use(0); slit_t.use(1); prog["uHist"].value = 0; prog["uSlit"].value = 1
        fbo.use(); va.render(moderngl.TRIANGLES); ctx.finish()
        rd = lambda t: np.frombuffer(t.read(), "f4").reshape(H, W, 4)[::-1]
        co, ta = rd(oc), rd(ot)
        o8 = (np.clip(co[..., :3], 0, 1) * 255 + 0.5).astype(np.uint8)[:, :, ::-1]
        gray = cv2.cvtColor(pl, cv2.COLOR_BGR2GRAY).astype(np.float32) / 255.0
        heat = sd.thermal(np.clip(ta[..., 0] / (K - 1), 0, 1)).astype(np.float32) / 255.0
        tm = (np.clip(np.maximum(heat, (gray * 0.25)[..., None]), 0, 1) * 255 + 0.5).astype(np.uint8)
        for y in range(0, H, 2): cv2.circle(tm, (int(xs[y]), y), 1, (255, 255, 255), -1)          # the slit line
        return pl, o8, tm
    if stills:
        rows = []; todo = set(stills)
        for i in range(max(0, min(stills) - K), max(stills) + 1):
            pl, o, tm = render(i)
            if i in todo:
                t = lambda im, s: sd.label(cv2.resize(im, (640, 360)), s)
                rows.append(np.hstack([t(pl, f"plate {i}"), t(tm, "time map + slit"), t(o, "slit-scan")]))
        cv2.imwrite(str(work(c, "still_sheet.jpg")), np.vstack(rows), [cv2.IMWRITE_JPEG_QUALITY, 90]); print("stills ->", work(c, "still_sheet.jpg")); return
    for dd in ("comp", "pass/timemap"): shutil.rmtree(work(c, dd), ignore_errors=True); work(c, dd).mkdir(parents=True)
    shutil.rmtree(work(c, "pass/slits"), ignore_errors=True)
    ts = time.time()
    for i in range(N):
        pl, o, tm = render(i)
        cv2.imwrite(str(work(c, "comp", f"{i:04d}.png")), o, [cv2.IMWRITE_PNG_COMPRESSION, 1])
        cv2.imwrite(str(work(c, "pass", "timemap", f"{i:04d}.jpg")), tm, [cv2.IMWRITE_JPEG_QUALITY, 93])
        if i % 60 == 0: print("frame", i, "%.0fs" % (time.time() - ts), flush=True)
    sd.encode(c)

if __name__ == "__main__":
    main()
