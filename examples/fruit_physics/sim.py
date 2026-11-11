"""Zero-gravity fruit vs. the tracked person, MuJoCo 4.

Everything lives in the solved camera frame (OpenCV: x right, y down, z forward, metres).
- Invisible person: capsules/spheres between MediaPipe joints. Each is a heavy free body driven
  kinematically: every substep its velocity is set to reach the (time-interpolated) tracked pose,
  so swinging arms hand real momentum to the fruit. Body parts never collide with each other.
- Fruit: free bodies, gravity 0, light air drag, bouncy soft contacts, spin from friction.
- Invisible room: table/counter top plane (from the picture: back edge y=0.376 m at z=1.25 m,
  tilt 15.6 deg), back wall z=2.3 m, and a soft spring that keeps fruit inside the camera frustum.
out: work/sim.json  (per frame: SUB shutter samples of fruit pos+quat, body capsules at frame time)
"""
import json, math
from pathlib import Path
import numpy as np, mujoco
from scipy.spatial.transform import Rotation as Rot

root = Path(__file__).parent
B = json.loads((root / "work" / "body_cam.json").read_text())
J = np.array(B["joints"]); N = len(J); F, W, H = B["f"], B["W"], B["H"]
FPS, SUBSTEPS = 25.0, 40
DT = 1.0 / FPS / SUBSTEPS
SHUTTER = [0.0, 0.125, 0.25, 0.375, 0.5]      # sample times (frames) for 180-degree motion blur
PITCH = math.radians(15.6); TABLE_P = np.array([0.0, 0.376, 1.25])
UP = np.array([0.0, -math.cos(PITCH), -math.sin(PITCH)])  # world up in camera coords

def mid(*k): return lambda j: j[list(k)].mean(0)
def jt(k): return lambda j: j[k]
# segments: name, end A, end B, radius
SEG = [
    ("head", lambda j: (j[7] + j[8]) / 2 + np.array([0, 0.02, 0.03]), None, 0.105),
    ("neck", lambda j: (j[7] + j[8]) / 2 + [0, 0.07, 0.03], mid(11, 12), 0.06),
    ("chest", mid(11, 12), lambda j: 0.55 * j[[11, 12]].mean(0) + 0.45 * j[[23, 24]].mean(0), 0.15),
    ("belly", lambda j: 0.55 * j[[11, 12]].mean(0) + 0.45 * j[[23, 24]].mean(0), mid(23, 24), 0.16),
    ("shoulders", jt(11), jt(12), 0.075),
    ("hips", jt(23), jt(24), 0.12),
    ("uarm_l", jt(11), jt(13), 0.055), ("farm_l", jt(13), jt(15), 0.045),
    ("uarm_r", jt(12), jt(14), 0.055), ("farm_r", jt(14), jt(16), 0.045),
    ("hand_l", mid(15, 17, 19, 21), None, 0.06), ("hand_r", mid(16, 18, 20, 22), None, 0.06),
    ("thigh_l", jt(23), jt(25), 0.085), ("thigh_r", jt(24), jt(26), 0.085),
]
# fixed capsule half-lengths (median over the shot) so the bodies keep their shape
HALF = {}
for name, a, b, r in SEG:
    HALF[name] = 0.0 if b is None else 0.5 * float(np.median([np.linalg.norm(a(J[i]) - b(J[i])) for i in range(N)]))

ARMS = [n for n in ("farm_l", "farm_r", "hand_l", "hand_r", "uarm_l", "uarm_r")]
def seg_dist(p, j, name):
    _, a, b, r = next(x for x in SEG if x[0] == name)
    pa = a(j); pb = pa if b is None else b(j); ab = pb - pa
    u = 0 if b is None else np.clip(np.dot(p - pa, ab) / (ab @ ab + 1e-12), 0, 1)
    return np.linalg.norm(p - (pa + u * ab)) - r

def make_fruit(seed):
    """Melons fixed (one each side, behind her). Apples/lemons: homes on the tracked wrist paths at
    chest-to-head height, mixed depths, no overlap with the body at frame 0 or with each other."""
    rng = np.random.default_rng(seed)
    spec = {"apple": (0.058, 0.22), "lemon": (0.048, 0.13), "melon": (0.135, 3.2)}
    out = [("melon", (-0.40, -0.36, 1.95), *spec["melon"]), ("melon", (0.38, -0.20, 1.85), *spec["melon"])]
    first = []
    for kind in ["apple"] * 5 + ["lemon"] * 5:
        s, m = spec[kind]
        for _ in range(4000):
            if rng.random() < 0.2:                           # a few ambient floaters
                z = rng.uniform(1.0, 1.9)
                p = np.array([rng.uniform(-0.65, 0.65) * (W / 2) / F * z, rng.uniform(-0.45, 0.05) * (H / 2) / F * z, z])
            else:
                i = rng.integers(15, N - 15); w = J[i, rng.choice([15, 16, 13, 14])]
                p = w + rng.normal(0, 0.04, 3) + [0, -0.02, rng.choice([-0.10, 0.0, 0.10])]
            if abs(p[0]) > 0.70 * (W / 2) / F * p[2] - s or p[1] < -0.55 * (H / 2) / F * p[2]: continue
            if UP @ (p - TABLE_P) < s + 0.10: continue
            if any(np.linalg.norm(p - np.array(q)) < s + t + 0.04 for _, q, t, _ in out): continue
            if min(seg_dist(p, J[0], x[0]) for x in SEG) < s + 0.06: continue
            hit = [i for i in range(5, N) if min(seg_dist(p, J[i], n) for n in ARMS) < s]
            out.append((kind, tuple(p.round(3)), s, m)); first.append(hit[0] if hit else None); break
    return out, first

def score(first):
    h = sorted(f for f in first if f is not None)
    if not h: return 0
    gaps = np.diff([0] + h + [N]); return len(h) - 0.02 * gaps.max()

if "--search" in __import__("sys").argv:
    res = sorted(((score(make_fruit(sd)[1]), sd) for sd in range(40)), reverse=True)[:5]
    for sc, sd in res: print("seed", sd, "score", round(sc, 2), "first hits", make_fruit(sd)[1])
    raise SystemExit
SEED = int(__import__("os").environ.get("FRUIT_SEED", "6"))
FRUIT, FIRST_HIT = make_fruit(SEED)
print("seed", SEED, "first hit frames", FIRST_HIT)

def body_pose(j, name, a, b):
    pa = a(j)
    if b is None: return pa, np.array([1.0, 0, 0, 0])
    pb = b(j); c = (pa + pb) / 2; d = pb - pa; d /= np.linalg.norm(d) + 1e-9
    q = Rot.align_vectors([d], [[0, 0, 1.0]])[0].as_quat()          # x y z w
    return c, np.array([q[3], q[0], q[1], q[2]])

def joints_at(t):  # t in frames, linear interpolation
    t = min(max(t, 0.0), N - 1.0); i = int(math.floor(t)); u = t - i
    return J[i] if i >= N - 1 else J[i] * (1 - u) + J[i + 1] * u

xml = [f'<mujoco><option timestep="{DT}" gravity="0 0 0" integrator="implicitfast"/>',
       '<default><geom solref="0.004 0.35" solimp="0.9 0.97 0.001" friction="0.7 0.01 0.001"/></default><worldbody>']
# room: table plane (contype 0 / conaffinity 1 touches fruit only)
zaxis = UP
xml.append(f'<geom name="table" type="plane" size="5 5 0.1" pos="{TABLE_P[0]} {TABLE_P[1]} {TABLE_P[2]}" '
           f'zaxis="{zaxis[0]} {zaxis[1]} {zaxis[2]}" contype="0" conaffinity="1"/>')
xml.append('<geom name="wall" type="plane" size="5 5 0.1" pos="0 0 2.3" zaxis="0 0 -1" contype="0" conaffinity="1"/>')
for name, a, b, r in SEG:
    c, q = body_pose(J[0], name, a, b)
    g = (f'<geom type="sphere" size="{r}"' if b is None else f'<geom type="capsule" size="{r} {HALF[name]}"')
    xml.append(f'<body name="b_{name}" pos="{c[0]} {c[1]} {c[2]}" quat="{q[0]} {q[1]} {q[2]} {q[3]}"><freejoint/>'
               f'{g} mass="40" contype="2" conaffinity="0"/></body>')
for k, (kind, p, s, m) in enumerate(FRUIT):
    if kind == "lemon": g = f'type="ellipsoid" size="{s*0.78} {s*0.78} {s*1.12}"'
    elif kind == "melon": g = f'type="ellipsoid" size="{s} {s} {s*1.08}"'
    else: g = f'type="sphere" size="{s}"'
    xml.append(f'<body name="f{k}" pos="{p[0]} {p[1]} {p[2]}"><freejoint/><geom {g} mass="{m}" contype="1" conaffinity="3"/></body>')
xml.append("</worldbody></mujoco>")
model = mujoco.MjModel.from_xml_string("\n".join(xml)); data = mujoco.MjData(model)

rng = np.random.default_rng(7)
bq = [model.jnt_qposadr[model.body(f"b_{s[0]}").jntadr[0]] for s in SEG]
bv = [model.jnt_dofadr[model.body(f"b_{s[0]}").jntadr[0]] for s in SEG]
fq = [model.jnt_qposadr[model.body(f"f{k}").jntadr[0]] for k in range(len(FRUIT))]
fv = [model.jnt_dofadr[model.body(f"f{k}").jntadr[0]] for k in range(len(FRUIT))]
fb = [model.body(f"f{k}").id for k in range(len(FRUIT))]
for k in range(len(FRUIT)):
    q0 = Rot.random(random_state=k + 3).as_quat(); data.qpos[fq[k] + 3:fq[k] + 7] = [q0[3], q0[0], q0[1], q0[2]]
    data.qvel[fv[k]:fv[k] + 3] = rng.normal(0, 0.025, 3)
    data.qvel[fv[k] + 3:fv[k] + 6] = rng.normal(0, 0.6, 3)
mujoco.mj_forward(model, data)

# clearance check at frame 0
for k, (kind, p, s, m) in enumerate(FRUIT):
    dmin = 9
    for name, a, b, r in SEG:
        pa = a(J[0]); pb = pa if b is None else b(J[0]); ab = pb - pa
        u = 0 if b is None else np.clip(np.dot(np.array(p) - pa, ab) / (ab @ ab), 0, 1)
        dmin = min(dmin, np.linalg.norm(np.array(p) - (pa + u * ab)) - r - s)
    print(f"{kind} {k}: clearance to body at frame 0 = {dmin:.3f} m")

def drive_body(t_next):
    j = joints_at(t_next)
    for (name, a, b, r), qa, va in zip(SEG, bq, bv):
        c, q = body_pose(j, name, a, b)
        cur_p = data.qpos[qa:qa + 3]; cur_q = data.qpos[qa + 3:qa + 7]
        data.qvel[va:va + 3] = (c - cur_p) / DT
        rc = Rot.from_quat([cur_q[1], cur_q[2], cur_q[3], cur_q[0]]); rt = Rot.from_quat([q[1], q[2], q[3], q[0]])
        data.qvel[va + 3:va + 6] = rc.inv().apply((rt * rc.inv()).as_rotvec()) / DT   # body-frame angular velocity

def room_forces():
    data.xfrc_applied[:] = 0
    for k, (kind, p, s, m) in enumerate(FRUIT):
        x = data.xpos[fb[k]]
        z = max(x[2], 0.5)
        lim_x = 0.80 * (W / 2) / F * z - s; lim_top = -0.88 * (H / 2) / F * z + s
        f = np.zeros(3); kw = 900.0 * m                      # stiff soft walls -> bounce off the frame edge
        if abs(x[0]) > lim_x: f[0] = -kw * (abs(x[0]) - lim_x) * np.sign(x[0])
        if x[1] < lim_top: f[1] = kw * (lim_top - x[1])
        if z < 0.85: f[2] = kw * (0.85 - z)
        home = np.array(p); om = 2 * math.pi / 4.5          # very weak pull home (period 4.5 s)
        f += -m * om * om * (x - home)
        v = data.qvel[fv[k]:fv[k] + 3]
        f += -2 * 0.25 * om * m * v                                   # air drag
        f += m * 0.004 * np.array([math.sin(data.time * 0.9 + k), math.cos(data.time * 0.7 + 2 * k), 0])  # float drift
        data.xfrc_applied[fb[k], :3] = f
    data.qvel[[i + 3 for v in fv for i in range(v, v + 3)]] *= (1 - 0.15 * DT)

out = []
for i in range(N):
    samples = []
    for sstep in range(SUBSTEPS):
        frac = sstep / SUBSTEPS
        if any(abs(frac - s) < 1e-9 for s in SHUTTER):
            samples.append([[*data.xpos[b].round(5).tolist(), *data.xquat[b].round(5).tolist()] for b in fb])
        drive_body(i + (sstep + 1) / SUBSTEPS)
        room_forces()
        mujoco.mj_step(model, data)
    j = J[i]
    caps = []
    for name, a, b, r in SEG:
        pa = a(j); pb = pa if b is None else b(j)
        caps.append([*np.round(pa, 4).tolist(), *np.round(pb, 4).tolist(), r])
    speed = [float(np.linalg.norm(data.qvel[v:v + 3])) for v in fv]
    out.append({"fruit": samples, "caps": caps, "speed": speed, "ncon": int(data.ncon)})
(root / "work" / "sim.json").write_text(json.dumps({
    "fruit_kinds": [[k, s] for k, p, s, m in FRUIT], "f": F, "W": W, "H": H, "shutter": SHUTTER,
    "table_point": TABLE_P.tolist(), "up": UP.tolist(), "frames": out}))
sp = np.array([o["speed"] for o in out])
print("max fruit speed per fruit (m/s):", sp.max(0).round(2))
print("frames with any contact:", sum(o["ncon"] > 0 for o in out))
