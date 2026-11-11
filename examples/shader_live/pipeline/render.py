"""World-locked GLSL effect over the tracked plate (moderngl, OpenGL 3.3).

    python render.py <piece_dir> --still 0,90,179     quick look: work/still_sheet.jpg (1 sample per pixel)
    python render.py <piece_dir>                      all frames -> work/comp, work/fx, work/ae, then out/*.mp4 / .mov

The piece's effect.glsl defines   void effect(in vec3 ro, in vec3 rd, in vec2 px, inout FX fx)
  ro, rd   camera position and pixel ray in PLANE coordinates (metres; z = 0 is the fitted plane, +z towards the camera side)
  px       pixel (origin top-left)
  fx.over, fx.a   opaque layer over the plate (linear colour, coverage)      fx.add   added light (linear)
  fx.mul          multiplies the plate (shadows, tints)                      fx.warp  plate lookup offset in pixels (refraction)
composite = mix(plate(px + warp) * mul, over, a) + add, in linear light.
Each frame averages `sub` samples over a centred half-frame shutter with sub-pixel jitter (anti-aliasing + motion blur of the
effect against the moving camera). The plate itself is never resampled where the effect is absent."""
import json, shutil, subprocess, sys
from pathlib import Path
import numpy as np, cv2, moderngl
from scipy.spatial.transform import Rotation as Rot, Slerp
from slv import load, work, frame, scene

HEADER = """#version 330
uniform sampler2D uPlate; uniform vec2 uRes; uniform float uF; uniform vec3 uRo; uniform mat3 uCam; uniform vec3 uSun;
uniform float iTime; uniform float iFrame; uniform float uDur; uniform vec2 uJit; uniform float uW; uniform float uP[16];
layout(location=0) out vec4 oComp; layout(location=1) out vec4 oFx;
struct FX { vec3 add; float a; vec3 over; vec3 mul; vec2 warp; };
const float PI = 3.14159265;
vec3 toLin(vec3 c){ return pow(max(c, 0.0), vec3(2.2)); }
vec3 plate(vec2 px){ return toLin(texture(uPlate, px / uRes).rgb); }                 // linear plate colour at a pixel
float luma(vec3 c){ return dot(c, vec3(0.2126, 0.7152, 0.0722)); }
vec3 toCam(vec3 P){ return transpose(uCam) * (P - uRo); }                            // plane coords -> camera (z forward)
vec2 project(vec3 P){ vec3 q = toCam(P); return uF * q.xy / q.z + 0.5 * uRes; }      // only valid when toCam(P).z > 0
float hitPlane(vec3 ro, vec3 rd, float h){ return (h - ro.z) / rd.z; }               // ray parameter at height h (negative: behind)
float hash11(float p){ p = fract(p * 0.1031); p *= p + 33.33; p *= p + p; return fract(p); }
float hash21(vec2 p){ vec3 q = fract(vec3(p.xyx) * 0.1031); q += dot(q, q.yzx + 33.33); return fract((q.x + q.y) * q.z); }
vec2 hash22(vec2 p){ vec3 q = fract(vec3(p.xyx) * vec3(0.1031, 0.1030, 0.0973)); q += dot(q, q.yzx + 33.33); return fract((q.xx + q.yz) * q.zy); }
float hash31(vec3 p){ p = fract(p * 0.1031); p += dot(p, p.zyx + 31.32); return fract((p.x + p.y) * p.z); }
vec3 hash33(vec3 p){ p = fract(p * vec3(0.1031, 0.1030, 0.0973)); p += dot(p, p.yxz + 33.33); return fract((p.xxy + p.yxx) * p.zyx); }
float vnoise(vec2 p){ vec2 i = floor(p), f = fract(p); f = f * f * (3.0 - 2.0 * f);
  return mix(mix(hash21(i), hash21(i + vec2(1, 0)), f.x), mix(hash21(i + vec2(0, 1)), hash21(i + vec2(1, 1)), f.x), f.y); }
float vnoise(vec3 p){ vec3 i = floor(p), f = fract(p); f = f * f * (3.0 - 2.0 * f);
  return mix(mix(mix(hash31(i), hash31(i + vec3(1, 0, 0)), f.x), mix(hash31(i + vec3(0, 1, 0)), hash31(i + vec3(1, 1, 0)), f.x), f.y),
             mix(mix(hash31(i + vec3(0, 0, 1)), hash31(i + vec3(1, 0, 1)), f.x), mix(hash31(i + vec3(0, 1, 1)), hash31(i + vec3(1, 1, 1)), f.x), f.y), f.z); }
float fbm(vec2 p){ float a = 0.5, s = 0.0; for (int i = 0; i < 5; i++){ s += a * vnoise(p); p = p * 2.03 + vec2(17.1, 5.3); a *= 0.5; } return s; }
float fbm(vec3 p){ float a = 0.5, s = 0.0; for (int i = 0; i < 5; i++){ s += a * vnoise(p); p = p * 2.03 + vec3(17.1, 5.3, 9.7); a *= 0.5; } return s; }
#line 1
"""
FOOTER = """
void main(){
  vec2 px0 = vec2(gl_FragCoord.x, uRes.y - gl_FragCoord.y); vec2 px = px0 + uJit;
  vec3 rd = normalize(uCam * vec3((px - 0.5 * uRes) / uF, 1.0));
  FX fx; fx.add = vec3(0.0); fx.a = 0.0; fx.over = vec3(0.0); fx.mul = vec3(1.0); fx.warp = vec2(0.0);
  effect(uRo, rd, px, fx);
  fx.a = clamp(fx.a, 0.0, 1.0); fx.add = max(fx.add, 0.0); fx.mul = max(fx.mul, 0.0);
  vec3 bg = plate(px0 + fx.warp) * fx.mul;
  oComp = vec4(mix(bg, fx.over, fx.a) + fx.add, 1.0) * uW;
  // effect on its own (for the breakdown): layer + light, the darkening as black coverage, refraction as the bent plate
  float dk = 1.0 - min(min(fx.mul.r, fx.mul.g), fx.mul.b); float aw = clamp(length(fx.warp) / 6.0, 0.0, 1.0) * (1.0 - fx.a);
  float al = clamp(max(fx.a, max(fx.add.r, max(fx.add.g, fx.add.b))), 0.0, 1.0);
  float A = 1.0 - (1.0 - al) * (1.0 - dk * (1.0 - fx.a)) * (1.0 - aw);
  oFx = vec4(fx.over * fx.a + fx.add + bg * aw * (1.0 - al), A) * uW;
}
"""
VS = "#version 330\nin vec2 p; void main(){ gl_Position = vec4(p, 0.0, 1.0); }"

def halton(i, b):
    f, r = 1.0, 0.0
    while i > 0: f /= b; r += f * (i % b); i //= b
    return r
def srgb(x): return np.clip(x, 0, 1) ** (1 / 2.2)

def main():
    c = load(sys.argv[1]); S = scene(c); W, H, N = c["W"], c["H"], c["N"]; fps = c["fps"]
    arg = lambda k, d=None: (sys.argv[sys.argv.index(k) + 1] if k in sys.argv else d)
    stills = [int(x) for x in arg("--still").split(",")] if "--still" in sys.argv else None
    sub = int(arg("--sub", 1 if stills else c.get("sub", 6))); a, b = [int(x) for x in arg("--frames", f"0:{N}").split(":")]
    ctx = moderngl.create_standalone_context(require=330)
    src = (c["dir"] / "effect.glsl").read_text(encoding="utf-8")
    try: prog = ctx.program(vertex_shader=VS, fragment_shader=HEADER + src + FOOTER)
    except Exception as e: print("SHADER ERROR (line numbers are effect.glsl's)\n", e); sys.exit(1)
    vao = ctx.vertex_array(prog, [(ctx.buffer(np.array([-1, -1, 3, -1, -1, 3], "f4").tobytes()), "2f", "p")])
    tex = ctx.texture((W, H), 3); tex.filter = (moderngl.LINEAR, moderngl.LINEAR); tex.repeat_x = tex.repeat_y = False
    t0, t1 = ctx.texture((W, H), 4, dtype="f4"), ctx.texture((W, H), 4, dtype="f4"); fbo = ctx.framebuffer([t0, t1])
    def U(k, v):
        if k in prog: prog[k].value = v
    U("uRes", (W, H)); U("uF", float(S["f"])); U("uSun", tuple(S["sun"])); U("uDur", N / fps)
    par = list(c.get("params", [])) + [0.0] * 16; U("uP", tuple(float(x) for x in par[:16])); U("uPlate", 0)
    Cp, Rcp = S["cam"], S["rot"]; sl = Slerp(np.arange(N), Rot.from_matrix(Rcp))
    def pose(t):
        t = float(np.clip(t, 0, N - 1)); i = min(int(t), N - 2); u = t - i
        return Cp[i] * (1 - u) + Cp[i + 1] * u, sl(t).as_matrix()
    def render(i):
        pl = frame(c, i); tex.write(pl[:, :, ::-1].tobytes()); tex.use(0)
        fbo.use(); fbo.clear(0, 0, 0, 0); ctx.enable(moderngl.BLEND); ctx.blend_func = (moderngl.ONE, moderngl.ONE); U("uW", 1.0 / sub)
        for s in range(sub):
            t = i if sub == 1 else i + ((s + 0.5) / sub - 0.5) * 0.5
            ro, R = pose(t); U("uRo", tuple(ro)); U("uCam", tuple(R.T.flatten())); U("iTime", t / fps); U("iFrame", float(t))
            U("uJit", (0.0, 0.0) if sub == 1 else (halton(s + 1, 2) - 0.5, halton(s + 1, 3) - 0.5)); vao.render(moderngl.TRIANGLES)
        ctx.finish()
        comp = np.frombuffer(t0.read(), "f4").reshape(H, W, 4)[::-1, :, :3]; fx = np.frombuffer(t1.read(), "f4").reshape(H, W, 4)[::-1]
        return pl, comp, fx
    def finish(pl, comp, fx, i):
        p = pl[:, :, ::-1].astype(np.float32) / 255; o = srgb(comp); d = o - p
        touched = np.abs(d).max(2) > 0.5 / 255
        if touched.any():
            sg = float(c.get("soften", 0.6))
            if sg > 0: d = cv2.GaussianBlur(d, (0, 0), sg)
            g = float(c.get("grain", 0.010))
            if g > 0:                                                       # plate-like grain only where the effect changed the picture
                rng = np.random.default_rng(1000 + i); n = cv2.GaussianBlur(rng.normal(0, 1, (H, W)).astype(np.float32), (0, 0), 0.7) * 1.6
                m = np.clip(cv2.GaussianBlur(np.abs(d).max(2), (0, 0), 2.0) * 8, 0, 1); d += (n * g * m)[..., None]
        o = np.clip(p + d, 0, 1); o8 = (o * 255 + 0.5).astype(np.uint8); of = o8.astype(np.float32) / 255
        # exact straight-alpha layer for After Effects: plate * (1 - A) + L * A == composite (in display space)
        A = np.maximum((of - p) / np.maximum(1 - p, 1e-4), (p - of) / np.maximum(p, 1e-4)).max(2).clip(0, 1); A[np.abs(of - p).max(2) < 0.75 / 255] = 0
        L = np.where(A[..., None] > 1e-4, (of - p * (1 - A[..., None])) / np.maximum(A[..., None], 1e-4), 0).clip(0, 1)
        ae = np.dstack([(L[:, :, ::-1] * 255 + 0.5).astype(np.uint8), (A * 255 + 0.5).astype(np.uint8)])
        al = fx[..., 3:4].clip(0, 1); col = srgb(np.where(al > 1e-4, fx[..., :3] / np.maximum(al, 1e-4), 0))
        fxo = np.dstack([(col[:, :, ::-1] * 255 + 0.5).astype(np.uint8), (al[..., 0] * 255 + 0.5).astype(np.uint8)])
        return o8[:, :, ::-1], ae, fxo
    if stills:
        rows = []
        for i in stills:
            pl, comp, fx = render(i); o, ae, fxo = finish(pl, comp, fx, i); a_ = fxo[..., 3:4].astype(np.float32) / 255
            grey = (fxo[..., :3] * a_ + 90 * (1 - a_)).astype(np.uint8)
            cv2.imwrite(str(work(c, f"still_{i:04d}.png")), o); rows.append(np.hstack([cv2.resize(o, (1280, 720)), np.vstack([cv2.resize(pl, (640, 360)), cv2.resize(grey, (640, 360))])]))
        cv2.imwrite(str(work(c, "still_sheet.jpg")), np.vstack(rows), [cv2.IMWRITE_JPEG_QUALITY, 90]); print("stills", stills, "->", work(c, "still_sheet.jpg")); return
    for d in ("comp", "fx", "ae"):
        if a == 0 and b == N: shutil.rmtree(work(c, d), ignore_errors=True)
        work(c, d).mkdir(exist_ok=True)
    import time; ts = time.time()
    for i in range(a, b):
        pl, comp, fx = render(i); o, ae, fxo = finish(pl, comp, fx, i)
        cv2.imwrite(str(work(c, "comp", f"{i:04d}.png")), o, [cv2.IMWRITE_PNG_COMPRESSION, 1])
        cv2.imwrite(str(work(c, "ae", f"{i:04d}.png")), ae, [cv2.IMWRITE_PNG_COMPRESSION, 1]); cv2.imwrite(str(work(c, "fx", f"{i:04d}.png")), fxo, [cv2.IMWRITE_PNG_COMPRESSION, 1])
        if i % 30 == 0: print("frame", i, "%.0fs" % (time.time() - ts), flush=True)
    if a == 0 and b == N: encode(c)

def encode(c):
    out = c["dir"] / "out"; out.mkdir(exist_ok=True); nm = c["name"]; r = f"{c['fps_num']}/{c['fps_den']}"
    tags = ["-pix_fmt", "yuv420p", "-color_primaries", "bt709", "-color_trc", "bt709", "-colorspace", "bt709", "-movflags", "+faststart"]
    run = lambda a: subprocess.run(["ffmpeg", "-v", "error", "-y"] + a, check=True)
    run(["-framerate", r, "-i", str(work(c, "comp", "%04d.png")), "-c:v", "libx264", "-crf", "14", "-preset", "slow"] + tags + [str(out / f"{nm}_Composite.mp4")])
    run(["-framerate", r, "-i", str(work(c, "frames", "%04d.jpg")), "-c:v", "libx264", "-crf", "14", "-preset", "slow"] + tags + [str(out / f"{nm}_Plate.mp4")])
    run(["-framerate", r, "-i", str(work(c, "ae", "%04d.png")), "-c:v", "prores_ks", "-profile:v", "4444", "-pix_fmt", "yuva444p10le", "-vendor", "apl0",
         "-color_primaries", "bt709", "-color_trc", "bt709", "-colorspace", "bt709", str(out / f"{nm}_Straight_Alpha.mov")])
    print("encoded ->", out)

if __name__ == "__main__":
    main()
