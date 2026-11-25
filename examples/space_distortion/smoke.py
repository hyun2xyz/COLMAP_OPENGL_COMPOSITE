"""3D smoke in the solved scene (ModernGL, OpenGL 3.3), stirred by the dancer and blown out by the sound waves.

Solver: Stam, "Stable Fluids" (SIGGRAPH 1999) as done on the GPU in Harris, "Fast Fluid Dynamics Simulation on the GPU"
(GPU Gems, ch. 38, 2004): forces -> vorticity confinement -> divergence -> Jacobi pressure -> subtract gradient -> advect.
The pass structure and the curl / vorticity / divergence / pressure / gradient / advection shaders follow Pavel Dobryakov's
WebGL-Fluid-Simulation (MIT License, Copyright (c) 2017 Pavel Dobryakov, https://github.com/PavelDoGreat/WebGL-Fluid-Simulation),
taken from 2D to 3D and rewritten for GLSL 3.30. The forces and the renderer are ours.

The grid is a box in PLANE coordinates (metres, z up), aligned with the first frame's camera: u to the right, v up, w away
from the camera, from smoke.near_m in front of the lens to smoke.back_m behind the dancer. OpenGL 3.3 (macOS too) has no
compute shaders, so the volume lives in a 2D atlas: slice k of the w axis is tile (k % tiles_x, k // tiles_x); every pass is
one full-screen draw over the atlas and finds its cell from gl_FragCoord. Velocities are in cells per video frame.

Per frame:
  arms     each fast limb joint (MediaPipe, lifted to 3D by trails.py) drags the air around it and sheds a little smoke
  waves    the same sound-wave fronts as render.py (distance to the body at the hit), but in 3D: the air behind a front is
           blown outward along the front's normal, so smoke flies towards the lens and away into the back of the scene
  body     capsules between the joints are obstacles: smoke is pushed out of them
  floor    patchy ground smoke near z = 0 across the box, lifted by buoyancy
Render: a ray per pixel (half resolution) from the solved camera through the box, density and sun transmittance read with
trilinear filtering, front-to-back. The ray stops at the floor and, inside the dancer's mask, at the dancer's depth, so smoke
behind the dancer is hidden and smoke in front covers them. Near the lens the smoke is big and soft, far away it is small."""
import numpy as np, moderngl

VS = "#version 330\nin vec2 p; void main(){ gl_Position = vec4(p, 0.0, 1.0); }"
COMMON = """#version 330
uniform ivec3 uN; uniform int uTX; uniform vec3 uO, uEu, uEv, uEw, uCs; uniform float dt;
ivec2 atl(ivec3 c){ return ivec2((c.z % uTX) * uN.x + c.x, (c.z / uTX) * uN.y + c.y); }
ivec3 cell(){ ivec2 f = ivec2(gl_FragCoord.xy); int tx = f.x / uN.x, ty = f.y / uN.y; return ivec3(f.x - tx * uN.x, f.y - ty * uN.y, ty * uTX + tx); }
vec4 F(sampler2D s, ivec3 c){ return texelFetch(s, atl(clamp(c, ivec3(0), uN - 1)), 0); }
vec4 S3(sampler2D s, vec3 p){                                                    /* trilinear, p in cell units (centres at i + 0.5) */
  p = clamp(p, vec3(0.5), vec3(uN) - 0.5); float zf = p.z - 0.5; int z0 = int(zf); float t = zf - float(z0); int z1 = min(z0 + 1, uN.z - 1);
  vec2 sz = vec2(textureSize(s, 0));
  vec4 a = texture(s, (vec2(float((z0 % uTX) * uN.x), float((z0 / uTX) * uN.y)) + p.xy) / sz);
  vec4 b = texture(s, (vec2(float((z1 % uTX) * uN.x), float((z1 / uTX) * uN.y)) + p.xy) / sz);
  return mix(a, b, t); }
vec3 world(vec3 q){ return uO + uEu * q.x * uCs.x + uEv * q.y * uCs.y + uEw * q.z * uCs.z; }
vec3 toCell(vec3 v){ return vec3(dot(v, uEu) / uCs.x, dot(v, uEv) / uCs.y, dot(v, uEw) / uCs.z); }
"""
CURL = COMMON + """uniform sampler2D uVel; out vec4 o;
void main(){ ivec3 c = cell(); ivec3 X = ivec3(1, 0, 0), Y = ivec3(0, 1, 0), Z = ivec3(0, 0, 1);
  vec3 dx = 0.5 * (F(uVel, c + X).xyz - F(uVel, c - X).xyz), dy = 0.5 * (F(uVel, c + Y).xyz - F(uVel, c - Y).xyz), dz = 0.5 * (F(uVel, c + Z).xyz - F(uVel, c - Z).xyz);
  o = vec4(dy.z - dz.y, dz.x - dx.z, dx.y - dy.x, 0.0); }"""
VORT = COMMON + """uniform sampler2D uVel, uCurl; uniform float uK; out vec4 o;
void main(){ ivec3 c = cell(); ivec3 X = ivec3(1, 0, 0), Y = ivec3(0, 1, 0), Z = ivec3(0, 0, 1);
  vec3 w = F(uCurl, c).xyz;
  vec3 eta = 0.5 * vec3(length(F(uCurl, c + X).xyz) - length(F(uCurl, c - X).xyz), length(F(uCurl, c + Y).xyz) - length(F(uCurl, c - Y).xyz), length(F(uCurl, c + Z).xyz) - length(F(uCurl, c - Z).xyz));
  vec3 N = eta / (length(eta) + 1e-5);
  o = vec4(clamp(F(uVel, c).xyz + uK * cross(N, w) * dt, -40.0, 40.0), 0.0); }"""
DIV = COMMON + """uniform sampler2D uVel; out vec4 o;
void main(){ ivec3 c = cell(); ivec3 X = ivec3(1, 0, 0), Y = ivec3(0, 1, 0), Z = ivec3(0, 0, 1);
  o = vec4(0.5 * (F(uVel, c + X).x - F(uVel, c - X).x + F(uVel, c + Y).y - F(uVel, c - Y).y + F(uVel, c + Z).z - F(uVel, c - Z).z), 0, 0, 0); }"""
SCALE = COMMON + """uniform sampler2D uSrc; uniform float uK; out vec4 o; void main(){ o = uK * F(uSrc, cell()); }"""
PRES = COMMON + """uniform sampler2D uP, uDiv; out vec4 o;
void main(){ ivec3 c = cell(); ivec3 X = ivec3(1, 0, 0), Y = ivec3(0, 1, 0), Z = ivec3(0, 0, 1);
  o = vec4((F(uP, c + X).x + F(uP, c - X).x + F(uP, c + Y).x + F(uP, c - Y).x + F(uP, c + Z).x + F(uP, c - Z).x - F(uDiv, c).x) / 6.0, 0, 0, 0); }"""
GRAD = COMMON + """uniform sampler2D uP, uVel; out vec4 o;
void main(){ ivec3 c = cell(); ivec3 X = ivec3(1, 0, 0), Y = ivec3(0, 1, 0), Z = ivec3(0, 0, 1);
  vec3 g = 0.5 * vec3(F(uP, c + X).x - F(uP, c - X).x, F(uP, c + Y).x - F(uP, c - Y).x, F(uP, c + Z).x - F(uP, c - Z).x);
  o = vec4(F(uVel, c).xyz - g, 0.0); }"""
ADV = COMMON + """uniform sampler2D uVel, uSrc; uniform float uDiss; out vec4 o;
void main(){ ivec3 c = cell(); vec3 q = vec3(c) + 0.5 - dt * F(uVel, c).xyz; o = S3(uSrc, q) / (1.0 + uDiss * dt); }"""
LIGHT = COMMON + """uniform sampler2D uDen; uniform vec3 uSun; uniform float uSigma; out vec4 o;
void main(){ vec3 q = vec3(cell()) + 0.5; vec3 sd = toCell(uSun); float sl = length(sd); sd /= sl; float tau = 0.0;
  for (int k = 1; k <= 16; k++){ vec3 p = q + sd * (1.5 * float(k)); if (any(lessThan(p, vec3(0.0))) || any(greaterThan(p, vec3(uN)))) break; tau += S3(uDen, p).x * 1.5 / sl; }
  o = vec4(exp(-uSigma * tau), 0, 0, 0); }"""
FORCE = COMMON + """uniform sampler2D uVel, uDen; uniform int uMode; uniform float uBurst, uBuoy, uArmPush, uArmEmit, uWavePush, uAmpRef, uAmbient, uGroundH, uTime;
uniform int uNJ; uniform vec4 uJp[12]; uniform vec4 uJv[12];                   /* joints: world pos + radius; velocity (cells/frame) + speed m/s */
uniform vec4 uBA[15]; uniform vec4 uBB[15];                                      /* body now: capsule ends, radius in uBA.w */
uniform int uNR; uniform vec4 uRing[8]; uniform vec4 uRingP[8]; uniform vec4 uLobe[16]; uniform vec4 uSegA[120]; uniform vec4 uSegB[120];
out vec4 o;
""" + "%NOISE%" + """
float capsule(vec3 p, vec3 a, vec3 b, out vec3 r){ vec3 pa = p - a, ba = b - a; float h = clamp(dot(pa, ba) / max(dot(ba, ba), 1e-8), 0.0, 1.0); r = pa - ba * h; return length(r); }
float front3(int k, vec3 X, out vec3 g, out float wn){                           /* render.py RING_FS front(), with its 3D normal */
  float dd[15]; vec3 gg[15]; float dm = 1e4;
  for (int s = 0; s < 15; s++){ vec3 r; float L = capsule(X, uSegA[k * 15 + s].xyz, uSegB[k * 15 + s].xyz, r); dd[s] = L + uSegA[k * 15 + s].w; gg[s] = r / max(L, 1e-5); dm = min(dm, dd[s]); }
  float sum = 0.0, ws = 0.0; g = vec3(0.0);
  for (int s = 0; s < 15; s++){ float e = exp(-(dd[s] - dm) / 0.15); sum += e; ws += e * uSegB[k * 15 + s].w; g += e * gg[s]; }
  wn = ws / sum; g = normalize(g + 1e-6); float d = dm - 0.15 * log(sum);
  vec3 u = normalize(X - uRing[k].xyz); float nz = fbm(u * 1.7 + vec3(uRingP[k].z, 0.0, uTime * 0.35)) - 0.47;
  return d * (1.0 + uRingP[k].w * nz * 2.0);
}
void main(){
  ivec3 c = cell(); vec3 X = world(vec3(c) + 0.5);
  float inside = 0.0;
  for (int s = 0; s < 15; s++){ vec3 r; inside = max(inside, 1.0 - smoothstep(0.6, 1.0, capsule(X, uBA[s].xyz, uBB[s].xyz, r) / uBA[s].w)); }
  if (uMode == 0){
    vec3 v = F(uVel, c).xyz;
    for (int j = 0; j < 12; j++){ if (j >= uNJ) break; vec3 r = X - uJp[j].xyz; float g = exp(-dot(r, r) / (2.0 * uJp[j].w * uJp[j].w));
      v = mix(v, uJv[j].xyz, clamp(g * uArmPush * dt, 0.0, 1.0)); }            /* the limb drags the air it moves through */
    for (int k = 0; k < 8; k++){ if (k >= uNR) break;
      vec3 gr; float wn; float d = front3(k, X, gr, wn); float x = (d - uRing[k].w) / uRingP[k].x;
      float push = exp(-0.45 * x * x) + 0.4 * smoothstep(-10.0, -4.0, x) * (1.0 - smoothstep(-0.5, 0.5, x));   /* the front, and the outflow behind it */
      vec3 u = normalize(X - uRing[k].xyz); vec4 L1 = uLobe[2 * k], L2 = uLobe[2 * k + 1];
      float lobe = min(0.12 + exp(L1.w * (dot(u, L1.xyz) - 1.0)) + exp(L2.w * (dot(u, L2.xyz) - 1.0)), 1.3);
      v += toCell(gr) * uWavePush * (uRingP[k].y / uAmpRef) * lobe * push * mix(0.5, 1.0, wn) * dt; }
    v.y += uBuoy * F(uDen, c).x * dt;                                             /* warm smoke rises (cell y = up) */
    v *= 1.0 - 0.7 * inside;
    o = vec4(v, 0.0);
  } else {
    float d = F(uDen, c).x;
    for (int j = 0; j < 12; j++){ if (j >= uNJ) break; vec3 r = X - uJp[j].xyz; d += uJv[j].w * exp(-dot(r, r) / (2.0 * uJp[j].w * uJp[j].w)) * uArmEmit * dt; }
    float h = X.z; float patchy = smoothstep(0.48, 0.8, fbm(vec3(X.xy * 0.8, uTime * 0.1)));
    d += uAmbient * patchy * (1.0 - smoothstep(0.0, uGroundH, h)) * dt;
    for (int k = 0; k < 8; k++){ if (k >= uNR) break;                             /* a hit bursts a puff around the hitting joint, the wave then blows it out */
      vec3 r = X - uRing[k].xyz; d += uBurst * (uRingP[k].y / uAmpRef) * smoothstep(0.8, 0.2, uRing[k].w) * exp(-dot(r, r) / (2.0 * 0.3 * 0.3)) * dt; }
    d *= 1.0 - 0.5 * inside * dt;
    o = vec4(min(d, 3.0), 0, 0, 0);
  }
}"""
RAY = COMMON + """uniform sampler2D uDen, uLight, uMask; uniform vec3 uRo; uniform mat3 uCam; uniform float uF, uSigma, uZbody, uStep, uAmbL, uSunL, uDetail, uTime;
uniform vec2 uRes, uScale; uniform vec3 uCol;
layout(location=0) out vec4 oS; layout(location=1) out vec4 oD;
%NOISE%
void main(){
  vec2 px = gl_FragCoord.xy * uScale;                                            /* image px, y down (target is image oriented) */
  vec3 dc = vec3((px - 0.5 * uRes) / uF, 1.0); float dl = length(dc); vec3 d = uCam * dc / dl;
  vec3 q0 = toCell(uRo - uO), qd = toCell(d);
  vec3 ta = (vec3(0.0) - q0) / qd, tb = (vec3(uN) - q0) / qd; vec3 tn = min(ta, tb), tf = max(ta, tb);
  float t0 = max(max(max(tn.x, tn.y), tn.z), 0.0), t1 = min(min(tf.x, tf.y), tf.z);
  if (d.z < -1e-5) t1 = min(t1, -uRo.z / d.z);                                   /* the floor */
  if (texture(uMask, px / uRes).r > 0.5) t1 = min(t1, uZbody * dl);              /* the dancer hides what is behind */
  float T = 1.0; vec3 C = vec3(0.0), D = vec3(0.0);
  float t = t0 + uStep * fract(52.9829189 * fract(dot(px, vec2(0.06711056, 0.00583715))));
  for (int i = 0; i < 260; i++){
    if (t > t1 || T < 0.01) break;
    float ds = uStep * (1.0 + 0.1 * t); vec3 q = q0 + qd * t; float rho = S3(uDen, q).x;
    if (rho > 1e-4){                                                             /* sub-cell detail: billows from 3D noise in the scene, slowly boiling */
      vec3 X = uRo + d * t; rho *= 0.04 + 2.4 * smoothstep(0.42, 0.78, fbm(X * uDetail + vec3(0.0, 0.0, -uTime * 0.5)));
      float a = 1.0 - exp(-uSigma * rho * ds); float L = S3(uLight, q).x;
      C += T * a * uCol * (uAmbL + uSunL * L);
      D += T * a * mix(vec3(1.0, 0.5, 0.12), vec3(0.2, 0.55, 1.0), smoothstep(0.8, 6.5, t));   /* depth code: near orange, far blue */
      T *= 1.0 - a;
    }
    t += ds;
  }
  oS = vec4(C, 1.0 - T); oD = vec4(D, 1.0 - T);
}"""

SEGS = [(11, 13), (13, 15), (15, 19), (12, 14), (14, 16), (16, 20), (23, 25), (25, 27), (24, 26), (26, 28), (11, 12), (23, 24), (11, 23), (12, 24), (0, 0)]
RADII = [0.07, 0.06, 0.05, 0.07, 0.06, 0.05, 0.09, 0.07, 0.09, 0.07, 0.08, 0.1, 0.13, 0.13, 0.12]
JOINTS = [13, 14, 15, 16, 19, 20, 25, 26, 27, 28]                               # elbows, wrists, index tips, knees, ankles


class Smoke:
    def __init__(self, ctx, W, H, cfg, S, P, f, fps, noise):
        self.ctx, self.W, self.H, self.cfg, self.f, self.fps = ctx, W, H, cfg, f, fps
        nx, ny, nz = cfg.get("grid", [112, 56, 128]); tx = int(cfg.get("tiles_x", 16)); assert nz % tx == 0
        self.N, self.tx = (nx, ny, nz), tx; self.ares = (tx * nx, (nz // tx) * ny)
        R0, r0 = np.asarray(S["rot"][0]), np.asarray(S["cam"][0])
        hz = lambda v: np.array([v[0], v[1], 0.0]) / np.hypot(v[0], v[1])
        self.eu, self.ev, self.ew = hz(R0[:, 0]), np.array([0.0, 0.0, 1.0]), hz(R0[:, 2])
        hips = 0.5 * (P[:, 23] + P[:, 24]); ud = float(np.mean((hips - r0) @ self.eu)); dd = float(np.mean((hips - r0) @ self.ew))
        Wd, Hd, near, back = cfg.get("width_m", 6.0), cfg.get("height_m", 3.0), cfg.get("near_m", 0.3), cfg.get("back_m", 3.0)
        self.O = np.array([r0[0], r0[1], 0.0]) + self.eu * (ud - Wd / 2) + self.ew * near
        self.cs = np.array([Wd / nx, Hd / ny, (dd + back - near) / nz])
        print("smoke 3D: %dx%dx%d cells of %.1f x %.1f x %.1f cm, %.1f m wide, %.1f m to %.1f m from the camera (dancer at %.1f m), atlas %dx%d"
              % (nx, ny, nz, *(self.cs * 100), Wd, near, dd + back, dd, *self.ares))
        def tex(res):
            t = ctx.texture(res, 4, dtype="f4"); t.filter = (moderngl.LINEAR, moderngl.LINEAR); t.repeat_x = t.repeat_y = False; return t
        A = self.ares
        self.vel, self.den, self.prs = [tex(A), tex(A)], [tex(A), tex(A)], [tex(A), tex(A)]; self.curl, self.div, self.light = tex(A), tex(A), tex(A)
        self.out = (W // 2, H // 2); self.tex, self.dtex = tex(self.out), tex(self.out)
        self.fb = {id(t): ctx.framebuffer([t]) for t in self.vel + self.den + self.prs + [self.curl, self.div, self.light]}
        self.fbo_out = ctx.framebuffer([self.tex, self.dtex])
        quad = ctx.buffer(np.array([-1, -1, 3, -1, -1, 3], "f4").tobytes())
        src = {"curl": CURL, "vort": VORT, "div": DIV, "scale": SCALE, "pres": PRES, "grad": GRAD, "adv": ADV, "light": LIGHT, "force": FORCE.replace("%NOISE%", noise), "ray": RAY.replace("%NOISE%", noise)}
        self.p = {k: ctx.program(vertex_shader=VS, fragment_shader=s) for k, s in src.items()}
        self.va = {k: ctx.vertex_array(p, [(quad, "2f", "p")]) for k, p in self.p.items()}
        dom = dict(uN=tuple(self.N), uTX=tx, uO=tuple(self.O), uEu=tuple(self.eu), uEv=tuple(self.ev), uEw=tuple(self.ew), uCs=tuple(self.cs))
        for p in self.p.values():
            for k, v in dom.items():
                if k in p: p[k].value = v
        for t in self.vel + self.den + self.prs: self.fb[id(t)].use(); self.fb[id(t)].clear(0, 0, 0, 0)
        self.sun = np.asarray(S["sun"], float)

    def _run(self, name, dst, tex=(), **u):
        p = self.p[name]
        for k, v in u.items():
            if k in p: p[k].value = v
        for unit, (k, t) in enumerate(tex):
            t.use(unit)
            if k in p: p[k].value = unit
        (self.fb[id(dst)] if dst is not None else self.fbo_out).use(); self.va[name].render(moderngl.TRIANGLES)

    def step(self, i, P, set_waves, amp_ref, t, arms=1.0):
        c, n = self.cfg, int(self.cfg.get("substeps", 2)); dt = 1.0 / n
        Q, Qp = P[i], P[max(i - 1, 0)]
        jp = [(*map(float, Q[j]), float(c.get("arm_radius_m", 0.12))) for j in JOINTS]
        jv = []
        for j in JOINTS:
            v = Q[j] - Qp[j]; jv.append((*(np.array([v @ self.eu, v @ self.ev, v @ self.ew]) / self.cs), float(np.linalg.norm(v) * self.fps) * arms))
        pf = self.p["force"]; pf["uNJ"].value = len(JOINTS) if arms > 0 else 0
        pf["uJp"].value = jp + [(0.0, 0.0, -9.0, 0.1)] * (12 - len(jp)); pf["uJv"].value = jv + [(0.0, 0.0, 0.0, 0.0)] * (12 - len(jv))
        pf["uBA"].value = [(*map(float, Q[a]), r) for (a, b), r in zip(SEGS, RADII)]; pf["uBB"].value = [(*map(float, Q[b]), 0.0) for a, b in SEGS]
        set_waves(pf)
        fu = dict(dt=dt, uBurst=float(c.get("burst", 0.6)), uBuoy=float(c.get("buoyancy", 0.02)), uArmPush=float(c.get("arm_push", 0.6)), uArmEmit=float(c.get("arm_emit", 0.004)),
                  uWavePush=float(c.get("wave_push_mps", 0.6)) / self.fps, uAmpRef=float(amp_ref), uAmbient=float(c.get("ambient", 0.01)),
                  uGroundH=float(c.get("ground_m", 0.35)), uTime=float(t))
        self.ctx.disable(moderngl.BLEND)
        for _ in range(n):
            self._run("force", self.vel[1], [("uVel", self.vel[0]), ("uDen", self.den[0])], uMode=0, **fu); self.vel.reverse()
            self._run("force", self.den[1], [("uVel", self.vel[0]), ("uDen", self.den[0])], uMode=1, **fu); self.den.reverse()
            self._run("curl", self.curl, [("uVel", self.vel[0])])
            self._run("vort", self.vel[1], [("uVel", self.vel[0]), ("uCurl", self.curl)], uK=float(c.get("vorticity", 0.6)), dt=dt); self.vel.reverse()
            self._run("div", self.div, [("uVel", self.vel[0])])
            self._run("scale", self.prs[1], [("uSrc", self.prs[0])], uK=0.8); self.prs.reverse()
            for _ in range(int(c.get("pressure_iter", 20))):
                self._run("pres", self.prs[1], [("uP", self.prs[0]), ("uDiv", self.div)]); self.prs.reverse()
            self._run("grad", self.vel[1], [("uP", self.prs[0]), ("uVel", self.vel[0])]); self.vel.reverse()
            self._run("adv", self.vel[1], [("uVel", self.vel[0]), ("uSrc", self.vel[0])], dt=dt, uDiss=float(c.get("vel_diss", 0.05))); self.vel.reverse()
            self._run("adv", self.den[1], [("uVel", self.vel[0]), ("uSrc", self.den[0])], dt=dt, uDiss=float(c.get("dye_diss", 0.04))); self.den.reverse()

    def render(self, R, ro, mask, zbody, t=0.0):
        """ray-march the volume from this frame's camera -> self.tex (lit smoke, premultiplied) and self.dtex (depth-coded)"""
        c = self.cfg; sg = float(c.get("sigma", 5.0))
        self._run("light", self.light, [("uDen", self.den[0])], uSun=tuple(self.sun), uSigma=sg)
        self._run("ray", None, [("uDen", self.den[0]), ("uLight", self.light), ("uMask", mask)],
                  uRo=tuple(map(float, ro)), uCam=tuple(np.asarray(R).T.flatten()), uF=self.f, uSigma=sg, uZbody=float(zbody), uStep=float(c.get("step_m", 0.04)),
                  uAmbL=float(c.get("ambient_light", 0.55)), uSunL=float(c.get("sun_light", 0.7)), uRes=(float(self.W), float(self.H)),
                  uScale=(self.W / self.out[0], self.H / self.out[1]), uDetail=float(c.get("detail_per_m", 2.5)), uTime=float(t), uCol=tuple(float(x) ** 2.2 for x in c.get("color", [0.9, 0.89, 0.87])))

    def read(self):
        """depth-coded smoke (half res, image orientation): rgb premultiplied, alpha"""
        return np.frombuffer(self.dtex.read(), "f4").reshape(self.out[1], self.out[0], 4)
