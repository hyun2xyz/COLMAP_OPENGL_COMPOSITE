"""OpenGL (moderngl, native GL 3.3) render + composite of the simulated fruit onto the plate.

Per frame:
 1. person depth: the invisible body capsules are rasterised (camera-space z) and the MediaPipe mask
    (edge-refined with a guided filter on the plate) is filled with the nearest capsule depth.
 2. fruit: procedural meshes + procedural albedo/normal detail, hemisphere + key light, gloss.
    Each fragment compares its depth with the person depth under the mask -> hidden behind her.
    5 shutter samples are averaged (180-degree motion blur).
 3. shadows/AO of every fruit (analytic sphere occlusion along the key light) on the person
    (her 3D surface from the depth fill), the table/counter plane and the back wall.
usage: python render.py out.mp4 [--frames a,b,c] [--layers]   (--layers also writes debug passes)
"""
import json, sys, math, subprocess
from pathlib import Path
import numpy as np, cv2, moderngl

root = Path(__file__).parent
S = json.loads((root / "work" / "sim.json").read_text())
SRC_W, SRC_H, F0 = S["W"], S["H"], S["f"]
OW, OH = 1080, 1920; SC = OW / SRC_W; F = F0 * SC; CX, CY = OW / 2, OH / 2
NEAR, FAR = 0.2, 10.0
KEY = np.array([-0.38, -0.72, -0.58]); KEY /= np.linalg.norm(KEY)   # direction TO the light (camera coords)
TABLE_P = np.array(S["table_point"]); UP = np.array(S["up"]); WALL_Z = 2.3
args = sys.argv[1:]; out_path = Path(args[0])
only = [int(x) for x in args[args.index("--frames") + 1].split(",")] if "--frames" in args else None
layers = "--layers" in args

# ---------------------------------------------------------------- meshes
def sphere_grid(nlat, nlon):
    th = np.linspace(0, math.pi, nlat); ph = np.linspace(0, 2 * math.pi, nlon)
    T, Pp = np.meshgrid(th, ph, indexing="ij")
    d = np.stack([np.sin(T) * np.cos(Pp), np.sin(T) * np.sin(Pp), np.cos(T)], -1)
    return T, Pp, d
def tri_mesh(V, nlat, nlon):
    idx = []
    for i in range(nlat - 1):
        for j in range(nlon - 1):
            a = i * nlon + j; b = a + nlon
            idx += [a, b, a + 1, a + 1, b, b + 1]
    V = V.reshape(-1, 3); I = np.array(idx, np.int32).reshape(-1, 3)
    fn = np.cross(V[I[:, 1]] - V[I[:, 0]], V[I[:, 2]] - V[I[:, 0]])
    N = np.zeros_like(V); [np.add.at(N, I[:, k], fn) for k in range(3)]
    # seam/pole vertices are duplicated -> average normals of coincident points
    key = np.round(V, 5); _, inv = np.unique(key, axis=0, return_inverse=True)
    acc = np.zeros((inv.max() + 1, 3)); np.add.at(acc, inv.ravel(), N); N = acc[inv.ravel()]
    N /= np.linalg.norm(N, axis=1, keepdims=True) + 1e-12
    return V.astype("f4"), N.astype("f4"), I
def fruit_mesh(kind):
    nlat, nlon = 72, 96
    T, Pp, d = sphere_grid(nlat, nlon)
    if kind == "apple":
        r = 1 - 0.30 * np.exp(-(T / 0.36) ** 2) - 0.17 * np.exp(-((math.pi - T) / 0.42) ** 2) + 0.06 * np.cos(T) \
            + 0.015 * np.cos(5 * Pp) * np.sin(T) ** 2
        V = d * r[..., None]
    elif kind == "lemon":
        r = 1 + 0.16 * np.exp(-(T / 0.16) ** 2) + 0.13 * np.exp(-((math.pi - T) / 0.16) ** 2)
        V = d * r[..., None] * [0.78, 0.78, 1.12]
    else:
        V = d * [1.0, 1.0, 1.08]
    return tri_mesh(V, nlat, nlon)
def cylinder_mesh(n=24):
    ph = np.linspace(0, 2 * math.pi, n); z = np.array([0.0, 1.0])
    Pz, Zz = np.meshgrid(ph, z, indexing="ij")
    V = np.stack([np.cos(Pz), np.sin(Pz), Zz], -1)
    idx = []
    for j in range(n - 1):
        a = j * 2; idx += [a, a + 2, a + 1, a + 1, a + 2, a + 3]
    V = V.reshape(-1, 3); N = V * [1, 1, 0]
    return V.astype("f4"), N.astype("f4"), np.array(idx, np.int32).reshape(-1, 3)
def leaf_mesh():
    u = np.linspace(0, 1, 12); v = np.linspace(-1, 1, 7); U, Vv = np.meshgrid(u, v, indexing="ij")
    w = np.sin(np.pi * U) ** 0.8 * 0.38
    V = np.stack([Vv * w, U, 0.18 * (Vv * w) ** 2 * 6 + 0.12 * U ** 2], -1)
    idx = []
    for i in range(11):
        for j in range(6):
            a = i * 7 + j; b = a + 7; idx += [a, b, a + 1, a + 1, b, b + 1]
    V = V.reshape(-1, 3); I = np.array(idx, np.int32).reshape(-1, 3)
    fn = np.cross(V[I[:, 1]] - V[I[:, 0]], V[I[:, 2]] - V[I[:, 0]]); N = np.zeros_like(V)
    [np.add.at(N, I[:, k], fn) for k in range(3)]; N /= np.linalg.norm(N, axis=1, keepdims=True) + 1e-12
    return V.astype("f4"), N.astype("f4"), I

ctx = moderngl.create_standalone_context(require=330)
VS = """#version 330
uniform mat4 M; uniform mat3 Nm; uniform mat4 P;
in vec3 in_pos; in vec3 in_n;
out vec3 vP; out vec3 vN; out vec3 vL;
void main(){ vec4 p = M*vec4(in_pos,1.0); vP = p.xyz; vN = Nm*in_n; vL = in_pos; gl_Position = P*p; }"""
FS = """#version 330
uniform int kind; uniform float seed; uniform vec3 KEY; uniform float subw;
uniform sampler2D maskT; uniform sampler2D pdT;
in vec3 vP; in vec3 vN; in vec3 vL; out vec4 o;
float h(vec3 p){ p = fract(p*0.3183099+.1); p*=17.0; return fract(p.x*p.y*p.z*(p.x+p.y+p.z)); }
float n3(vec3 x){ vec3 i=floor(x), f=fract(x); f=f*f*(3.0-2.0*f);
  return mix(mix(mix(h(i),h(i+vec3(1,0,0)),f.x),mix(h(i+vec3(0,1,0)),h(i+vec3(1,1,0)),f.x),f.y),
             mix(mix(h(i+vec3(0,0,1)),h(i+vec3(1,0,1)),f.x),mix(h(i+vec3(0,1,1)),h(i+vec3(1,1,1)),f.x),f.y),f.z); }
float fbm(vec3 p){ float a=0.5,s=0.0; for(int i=0;i<5;i++){ s+=a*n3(p); p*=2.03; a*=0.5; } return s; }
vec3 lin(vec3 c){ return pow(c, vec3(2.2)); }
void main(){
  ivec2 px = ivec2(gl_FragCoord.xy);
  float m = texelFetch(maskT, px, 0).r; float pd = texelFetch(pdT, px, 0).r;
  float hidden = m * smoothstep(-0.015, 0.015, vP.z - pd);
  if (hidden > 0.995) discard;
  vec3 N = normalize(vN); vec3 V = normalize(-vP);
  vec3 lp = normalize(vL); float th = acos(clamp(lp.z,-1.0,1.0)); float ph = atan(lp.y, lp.x);
  vec3 alb; float spec, shin, bump;
  vec3 q = vL*3.0 + seed*7.1;
  if (kind == 0) {            // apple: red over yellow-green, meridian streaks, lenticels, green cavity
    float streak = fbm(vec3(ph*5.0, lp.z*1.2, seed));
    float blush = smoothstep(0.10, 0.45, fbm(q*0.8) + 0.25*lp.z + 0.25*fract(seed*3.7));
    vec3 red = mix(lin(vec3(0.62,0.05,0.06)), lin(vec3(0.80,0.12,0.08)), streak);
    vec3 yel = lin(vec3(0.86,0.70,0.22));
    alb = mix(yel, red, clamp(blush*0.9 + streak*0.35, 0.0, 1.0));
    alb = mix(alb, lin(vec3(0.55,0.62,0.18)), smoothstep(0.55,0.0,th)*0.8);
    float dots = smoothstep(0.83, 0.9, n3(vL*55.0 + seed*13.0)); alb = mix(alb, lin(vec3(0.95,0.85,0.55)), dots*0.6);
    spec = 0.45; shin = 70.0; bump = 0.04;
  } else if (kind == 1) {     // lemon: yellow, greenish tips, pitted peel
    alb = lin(vec3(0.97,0.80,0.10)) * (0.9 + 0.15*fbm(q*2.0));
    alb = mix(alb, lin(vec3(0.70,0.72,0.15)), smoothstep(0.65,1.0,abs(lp.z))*0.5);
    spec = 0.25; shin = 28.0; bump = 0.22;
  } else if (kind == 3) {     // apple stem
    alb = lin(vec3(0.30,0.20,0.10)); spec = 0.1; shin = 10.0; bump = 0.0;
  } else if (kind == 4) {     // apple leaf
    alb = lin(vec3(0.20,0.45,0.10)) * (0.85 + 0.3*fbm(vL*20.0)); spec = 0.3; shin = 30.0; bump = 0.0;
  } else {                    // watermelon: dark/light green wavy stripes
    float w = sin(ph*9.0 + 2.2*fbm(vec3(lp*3.0) + seed) + 1.5*sin(th*3.0));
    float stripe = smoothstep(-0.15, 0.35, w);
    alb = mix(lin(vec3(0.48,0.64,0.30)), lin(vec3(0.06,0.20,0.05)), stripe);
    alb *= 0.85 + 0.25*fbm(q*3.0);
    alb = mix(alb, lin(vec3(0.75,0.72,0.35)), smoothstep(0.35,0.0,abs(th-2.6))*0.0);
    spec = 0.30; shin = 45.0; bump = 0.05;
  }
  // micro normal detail
  float e = 0.004; vec3 g = vec3(n3(vL*90.0+vec3(e,0,0))-n3(vL*90.0-vec3(e,0,0)), n3(vL*90.0+vec3(0,e,0))-n3(vL*90.0-vec3(0,e,0)), n3(vL*90.0+vec3(0,0,e))-n3(vL*90.0-vec3(0,0,e)));
  N = normalize(N - bump*30.0*g);
  vec3 L = normalize(KEY);
  float ndl = max(dot(N,L),0.0); float wrap = max((dot(N,L)+0.35)/1.35, 0.0);
  float up = 0.5 - 0.5*N.y;                                   // camera y is down
  vec3 amb = mix(lin(vec3(0.80,0.74,0.70))*0.55, lin(vec3(0.97,0.97,1.0))*0.95, up);
  vec3 col = alb * (amb*0.95 + lin(vec3(1.0,0.97,0.92))*1.05*wrap);
  if (kind == 1) col += alb*lin(vec3(1.0,0.75,0.2))*0.25*pow(max(dot(-N,L),0.0),1.5);   // peel translucency
  vec3 Hh = normalize(L+V); float fres = pow(1.0-max(dot(N,V),0.0), 5.0);
  col += spec*pow(max(dot(N,Hh),0.0), shin)*ndl*1.3 + (0.04+0.5*fres)*spec*amb*0.6;
  // soft window reflection
  vec3 R = reflect(-V, N); col += spec*0.35*smoothstep(0.75,0.95,dot(R, normalize(vec3(-0.5,-0.6,-0.6))));
  col = pow(col/(1.0+col*0.15), vec3(1.0/2.2));
  float a = (1.0 - hidden);
  o = vec4(col*a, a) * subw;
}"""
fruit_prog = ctx.program(vertex_shader=VS, fragment_shader=FS)
CAP_VS = """#version 330
uniform mat4 M; uniform mat4 P; in vec3 in_pos; out float z;
void main(){ vec4 p = M*vec4(in_pos,1.0); z = p.z; gl_Position = P*p; }"""
CAP_FS = """#version 330
in float z; out vec4 o; void main(){ o = vec4(z,0,0,1); }"""
cap_prog = ctx.program(vertex_shader=CAP_VS, fragment_shader=CAP_FS)

def vao_for(prog, V, N, I, with_n=True):
    vb = ctx.buffer(np.hstack([V, N]).astype("f4").tobytes()); ib = ctx.buffer(I.astype("i4").tobytes())
    if with_n: return ctx.vertex_array(prog, [(vb, "3f 3f", "in_pos", "in_n")], ib)
    return ctx.vertex_array(prog, [(vb, "3f 12x", "in_pos")], ib)
MESH = {k: vao_for(fruit_prog, *fruit_mesh(k)) for k in ["apple", "lemon", "melon"]}
STEM = vao_for(fruit_prog, *cylinder_mesh()); LEAF = vao_for(fruit_prog, *leaf_mesh())
SPH_CAP = vao_for(cap_prog, *tri_mesh(sphere_grid(24, 32)[2], 24, 32), with_n=False)
CYL_CAP = vao_for(cap_prog, *cylinder_mesh(32), with_n=False)

Pm = np.zeros((4, 4), "f4")
Pm[0, 0] = 2 * F / OW; Pm[0, 2] = 2 * CX / OW - 1
Pm[1, 1] = -2 * F / OH; Pm[1, 2] = -(2 * CY / OH - 1)
Pm[2, 2] = (FAR + NEAR) / (FAR - NEAR); Pm[2, 3] = -2 * FAR * NEAR / (FAR - NEAR); Pm[3, 2] = 1
def U(prog, name, val):
    if prog.get(name, None) is not None: prog[name].write(np.ascontiguousarray(val, "f4").T.tobytes() if np.ndim(val) == 2 else np.float32(val).tobytes())
for p in (fruit_prog, cap_prog): U(p, "P", Pm)
fruit_prog["KEY"].value = tuple(KEY)
def quat_m(q):  # MuJoCo w x y z
    w, x, y, z = q
    return np.array([[1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
                     [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
                     [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)]])
def TRS(t, R, s):
    M = np.eye(4); M[:3, :3] = R * np.asarray(s, float); M[:3, 3] = t; return M

MS = 4
fbo_c = ctx.framebuffer(ctx.renderbuffer((OW, OH), 4, samples=MS, dtype="f4"), ctx.depth_renderbuffer((OW, OH), samples=MS))
fbo_r = ctx.framebuffer(ctx.renderbuffer((OW, OH), 4, dtype="f4"))
acc_tex = ctx.texture((OW, OH), 4, dtype="f4"); fbo_acc = ctx.framebuffer(acc_tex)
cap_fbo = ctx.framebuffer(ctx.renderbuffer((OW, OH), 1, dtype="f4"), ctx.depth_renderbuffer((OW, OH)))
maskT = ctx.texture((OW, OH), 1, dtype="f4"); pdT = ctx.texture((OW, OH), 1, dtype="f4")

def guided(I, p, r=8, eps=2e-3):
    bf = lambda x: cv2.boxFilter(x, -1, (2 * r + 1, 2 * r + 1))
    mI, mp_, cIp, vI = bf(I), bf(p), bf(I * p), bf(I * I) - bf(I) ** 2
    a = (cIp - mI * mp_) / (vI + eps); b = mp_ - a * mI
    return bf(a) * I + bf(b)

def person_depth(caps):
    cap_fbo.use(); ctx.enable(moderngl.DEPTH_TEST); ctx.disable(moderngl.CULL_FACE)
    cap_fbo.clear(1e4, 0, 0, 1, depth=1.0)
    for c in caps:
        a, b, r = np.array(c[:3]), np.array(c[3:6]), c[6]
        for pt in ([a] if np.allclose(a, b) else [a, b]):
            U(cap_prog, "M", TRS(pt, np.eye(3), [r, r, r])); SPH_CAP.render()
        if not np.allclose(a, b):
            d = b - a; L = np.linalg.norm(d); z = d / L
            x = np.cross(z, [0, 0, 1.0]) if abs(z[2]) < 0.9 else np.cross(z, [1.0, 0, 0]); x /= np.linalg.norm(x); y = np.cross(z, x)
            U(cap_prog, "M", TRS(a, np.stack([x, y, z], 1), [r, r, L])); CYL_CAP.render()
    d = np.frombuffer(cap_fbo.read(components=1, dtype="f4"), "f4").reshape(OH, OW)[::-1].copy()
    small = cv2.resize(d, (OW // 4, OH // 4), interpolation=cv2.INTER_NEAREST)
    have = small < 1e3
    _, lab = cv2.distanceTransformWithLabels((~have).astype(np.uint8), cv2.DIST_L2, 5, labelType=cv2.DIST_LABEL_PIXEL)
    vals = np.zeros(lab.max() + 1, "f4"); vals[lab[have]] = small[have]
    filled = vals[lab]
    filled = cv2.GaussianBlur(filled, (0, 0), 3)
    filled = cv2.resize(filled, (OW, OH), interpolation=cv2.INTER_LINEAR)
    return np.where(d < 1e3, np.minimum(d, filled + 0.05), filled).astype("f4")

def shade_plate(plate_lin, mask, pd, fruit_states):
    """analytic soft shadows + AO from the fruit spheres on: person (mask), table plane, back wall"""
    ys, xs = np.mgrid[0:OH:4, 0:OW:4].astype("f4")
    ray = np.stack([(xs - CX) / F, (ys - CY) / F, np.ones_like(xs)], -1)
    m = mask[::4, ::4]
    t_tab = ((TABLE_P @ UP)) / np.maximum(ray @ UP, 1e-6)   # ray hits plane where up.(t r - P0)=0
    t_tab = np.where(ray @ UP > 1e-6, t_tab, 1e9)
    t_tab = np.where(ray @ UP < -1e-6, ((TABLE_P @ UP)) / (ray @ UP), t_tab)
    P_tab = ray * np.where(t_tab > 0, t_tab, 1e9)[..., None]
    P_wall = ray * (WALL_Z / ray[..., 2])[..., None]
    use_tab = (P_tab[..., 2] < WALL_Z) & (t_tab > 0)
    P_env = np.where(use_tab[..., None], P_tab, P_wall)
    P_per = ray * pd[::4, ::4][..., None]
    shade = np.ones(m.shape, "f4")
    for P, wgt in ((P_per, m), (P_env, 1 - m)):
        s = np.ones(m.shape, "f4")
        for c, r in fruit_states:
            v = c - P; t = v @ KEY; dperp = np.linalg.norm(v - t[..., None] * KEY, axis=-1)
            w = 0.25 * r + 0.18 * np.maximum(t, 0)
            sh = (1 - np.clip((dperp - (r - w)) / (2 * w + 1e-6), 0, 1)) * (t > 0) * 0.55 / (1 + 1.2 * np.maximum(t, 0))
            dist = np.linalg.norm(v, axis=-1); ao = np.clip((r / np.maximum(dist, 1e-3)) ** 2, 0, 1) * 0.6
            s *= (1 - sh) * (1 - ao)
        shade = shade * (1 - wgt) + shade * s * wgt
    shade = cv2.resize(cv2.GaussianBlur(shade, (0, 0), 1.2), (OW, OH), interpolation=cv2.INTER_LINEAR)
    return plate_lin * shade[..., None], shade

def render_frame(i, plate_src):
    fr = S["frames"][i]
    plate = cv2.resize(plate_src, (OW, OH), interpolation=cv2.INTER_LANCZOS4)
    gray = cv2.cvtColor(plate_src, cv2.COLOR_BGR2GRAY).astype("f4") / 255
    m = cv2.imread(str(root / "work" / "mask" / f"{i:04d}.png"), 0).astype("f4") / 255
    m = np.clip(guided(gray, m), 0, 1); m = np.clip((m - 0.5) * 1.6 + 0.5, 0, 1)
    mask = cv2.resize(m, (OW, OH), interpolation=cv2.INTER_LINEAR)
    pd = person_depth(fr["caps"])
    maskT.write(np.ascontiguousarray(mask[::-1]).tobytes()); pdT.write(np.ascontiguousarray(pd[::-1]).tobytes())
    maskT.use(0); pdT.use(1); fruit_prog["maskT"].value = 0; fruit_prog["pdT"].value = 1
    # fruit, averaged over the shutter samples
    fbo_acc.use(); fbo_acc.clear(0, 0, 0, 0)
    nsub = len(fr["fruit"])
    for samp in fr["fruit"]:
        fbo_c.use(); fbo_c.clear(0, 0, 0, 0, depth=1.0); ctx.enable(moderngl.DEPTH_TEST); ctx.disable(moderngl.BLEND)
        fruit_prog["subw"].value = 1.0
        for k, ((kind, s), st) in enumerate(zip(S["fruit_kinds"], samp)):
            R = quat_m(st[3:]); t = np.array(st[:3])
            fruit_prog["kind"].value = {"apple": 0, "lemon": 1, "melon": 2}[kind]; fruit_prog["seed"].value = k * 0.137 + 0.3
            M = TRS(t, R, [s, s, s]); U(fruit_prog, "M", M); U(fruit_prog, "Nm", R); MESH[kind].render()
            if kind == "apple":     # stem + leaf (shaded with the apple program, green/brown via kind trick)
                top = t + R @ np.array([0, 0, s * 0.66])
                fruit_prog["kind"].value = 3
                U(fruit_prog, "M", TRS(top, R, [0.0055, 0.0055, s * 0.40])); STEM.render()
                fruit_prog["kind"].value = 4
                if k % 2 == 0:
                    Rl = R @ quat_m([math.cos(0.75), math.sin(0.75), 0, 0])
                    U(fruit_prog, "M", TRS(top + R @ np.array([0, 0, s * 0.22]), Rl, [s * 0.5, s * 0.5, s * 0.5])); U(fruit_prog, "Nm", Rl); LEAF.render()
        ctx.copy_framebuffer(fbo_r, fbo_c)
        rgba = np.frombuffer(fbo_r.read(components=4, dtype="f4"), "f4").reshape(OH, OW, 4)
        acc = rgba if samp is fr["fruit"][0] else acc + rgba
    rgba = (acc / nsub)[::-1]
    # composite: plate * shadows, then premultiplied fruit over
    states = [(np.array(st[:3]), s * (1.05 if kind != "lemon" else 0.9)) for (kind, s), st in zip(S["fruit_kinds"], fr["fruit"][len(fr["fruit"]) // 2])]
    plate_lin = (plate[..., ::-1].astype("f4") / 255) ** 2.2
    shaded, shade = shade_plate(plate_lin, mask, pd, states)
    base = shaded ** (1 / 2.2)
    outc = base * (1 - rgba[..., 3:4]) + rgba[..., :3]
    out = np.clip(outc * 255 + 0.5, 0, 255).astype(np.uint8)
    dbg = None
    if layers:
        pdv = np.clip((pd - 0.8) / 1.4, 0, 1)
        dbg = {"mask": (mask * 255).astype(np.uint8), "pd": (np.where(mask > 0.5, 1 - pdv, 0) * 255).astype(np.uint8),
               "fruit": np.clip(rgba[..., :3] * 255, 0, 255).astype(np.uint8), "alpha": (rgba[..., 3] * 255).astype(np.uint8),
               "shade": (np.clip(shade, 0, 1) * 255).astype(np.uint8)}
    return out, dbg

frames = only or list(range(len(S["frames"])))
if only:
    for i in frames:
        o, dbg = render_frame(i, cv2.imread(str(root / "work" / "frames" / f"{i:04d}.png")))
        cv2.imwrite(str(out_path.parent / f"still_{i:04d}.png"), o[..., ::-1])
        if dbg:
            for k, v in dbg.items(): cv2.imwrite(str(out_path.parent / f"still_{i:04d}_{k}.png"), v[..., ::-1] if v.ndim == 3 else v)
    sys.exit()
ff = subprocess.Popen(["ffmpeg", "-v", "error", "-y", "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{OW}x{OH}", "-r", "25", "-i", "-",
                       "-c:v", "libx264", "-crf", "16", "-preset", "slow", "-pix_fmt", "yuv420p", "-color_primaries", "bt709",
                       "-color_trc", "bt709", "-colorspace", "bt709", "-movflags", "+faststart", str(out_path)], stdin=subprocess.PIPE)
for i in frames:
    o, _ = render_frame(i, cv2.imread(str(root / "work" / "frames" / f"{i:04d}.png")))
    ff.stdin.write(o.tobytes())
    if i % 50 == 0: print("frame", i, flush=True)
ff.stdin.close(); ff.wait(); print("done", ff.returncode)
