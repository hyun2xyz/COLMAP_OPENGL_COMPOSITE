"""Procedural Detailed 3D Modeled Prop & Fractured Debris Triangle Meshes for OpenGL PBR Rendering.

Each builder returns a dict:
  {
    "pos": (V, 3) float32,   # local coordinates in metres, centered near CoM
    "nrm": (V, 3) float32,   # unit normals
    "uv":  (V, 2) float32,   # surface UVs for procedural PBR grain/treads
    "mat": (V,)   float32,   # material ID (0..8)
    "idx": (T*3,) uint32,    # triangle indices
  }

Material IDs:
  0 = retro matte charcoal CRT plastic       5 = rough porous concrete cinder masonry
  1 = glossy curved dark CRT glass screen    6 = weathered industrial red painted steel
  2 = warm pine wood crate planks            7 = matte dark molded rubber tire tread
  3 = brushed/forged dark steel metal        8 = green fiberglass CRT circuit board / copper
  4 = glazed terracotta clay ceramic
"""
import math
import numpy as np


class MeshBuilder:
    def __init__(self):
        self.pos = []
        self.nrm = []
        self.uv = []
        self.mat = []
        self.ao = []
        self.idx = []
        self._vcount = 0

    def add_mesh(self, pos, nrm, uv, mat_id, idx, ao=1.0):
        pos = np.asarray(pos, dtype=np.float32)
        nrm = np.asarray(nrm, dtype=np.float32)
        uv = np.asarray(uv, dtype=np.float32)
        idx = np.asarray(idx, dtype=np.uint32) + self._vcount
        self.pos.append(pos)
        self.nrm.append(nrm)
        self.uv.append(uv)
        self.mat.append(np.full(len(pos), float(mat_id), dtype=np.float32))
        if np.ndim(ao) == 0:
            self.ao.append(np.full(len(pos), float(ao), dtype=np.float32))
        else:
            self.ao.append(np.asarray(ao, dtype=np.float32))
        self.idx.append(idx)
        self._vcount += len(pos)

    def add_box(self, center, size, mat_id, rot=None, ao=1.0):
        """Add an oriented 3D box with clean per-face normals & metric UVs."""
        cx, cy, cz = center
        hx, hy, hz = size[0] * 0.5, size[1] * 0.5, size[2] * 0.5
        # 6 faces: (normal, u_axis, v_axis, corner_offsets)
        faces = [
            ((1, 0, 0),  [(hx, -hy, -hz), (hx, hy, -hz), (hx, hy, hz), (hx, -hy, hz)], (size[1], size[2])),
            ((-1, 0, 0), [(-hx, hy, -hz), (-hx, -hy, -hz), (-hx, -hy, hz), (-hx, hy, hz)], (size[1], size[2])),
            ((0, 1, 0),  [(hx, hy, -hz), (-hx, hy, -hz), (-hx, hy, hz), (hx, hy, hz)], (size[0], size[2])),
            ((0, -1, 0), [(-hx, -hy, -hz), (hx, -hy, -hz), (hx, -hy, hz), (-hx, -hy, hz)], (size[0], size[2])),
            ((0, 0, 1),  [(-hx, -hy, hz), (hx, -hy, hz), (hx, hy, hz), (-hx, hy, hz)], (size[0], size[1])),
            ((0, 0, -1), [(-hx, hy, -hz), (hx, hy, -hz), (hx, -hy, -hz), (-hx, -hy, -hz)], (size[0], size[1])),
        ]
        P, N, U, I = [], [], [], []
        vc = 0
        for n, pts, (su, sv) in faces:
            for p, (u, v) in zip(pts, [(0, 0), (su, 0), (su, sv), (0, sv)]):
                P.append(p); N.append(n); U.append((u, v))
            I.extend([vc, vc + 1, vc + 2, vc, vc + 2, vc + 3])
            vc += 4
        P = np.array(P, dtype=np.float32)
        N = np.array(N, dtype=np.float32)
        if rot is not None:
            R = np.asarray(rot, dtype=np.float32)
            P = P @ R.T
            N = N @ R.T
        P += np.array([cx, cy, cz], dtype=np.float32)
        self.add_mesh(P, N, np.array(U, dtype=np.float32), mat_id, np.array(I, dtype=np.uint32), ao=ao)

    def add_lathe(self, profile_rz, n_seg, mat_id, center=(0, 0, 0), axis="z", uv_scale=2.0, ao=1.0):
        """Lathe a list of (radius, axial_z) profile points into a smooth 3D surface of revolution."""
        prof = np.asarray(profile_rz, dtype=np.float64)
        m = len(prof)
        ang = np.linspace(0, 2 * math.pi, n_seg + 1)
        ca, sa = np.cos(ang), np.sin(ang)

        # Compute smooth profile tangents -> outward normals
        dr = np.gradient(prof[:, 0])
        dz = np.gradient(prof[:, 1])
        # Outward radial/axial normal assuming profile goes bottom -> top (dz > 0 -> +r normal)
        nr = dz
        nz = -dr
        nl = np.hypot(nr, nz) + 1e-9
        nr /= nl; nz /= nl

        P, N, U, I = [], [], [], []
        arc = np.concatenate([[0.0], np.cumsum(np.hypot(np.diff(prof[:, 0]), np.diff(prof[:, 1])))])
        for i in range(m):
            r, z = prof[i]
            for j in range(n_seg + 1):
                x, y = r * ca[j], r * sa[j]
                nx, ny, nzz = nr[i] * ca[j], nr[i] * sa[j], nz[i]
                if axis == "z":
                    P.append((x, y, z)); N.append((nx, ny, nzz))
                elif axis == "y":
                    P.append((x, z, -y)); N.append((nx, nzz, -ny))
                else:
                    P.append((z, y, -x)); N.append((nzz, ny, -nx))
                U.append((j / n_seg * uv_scale, arc[i] * uv_scale))
        stride = n_seg + 1
        for i in range(m - 1):
            for j in range(n_seg):
                a = i * stride + j
                b = a + 1
                c = a + stride
                d = c + 1
                I.extend([a, b, d, a, d, c])
        P = np.array(P, dtype=np.float32) + np.array(center, dtype=np.float32)
        N = np.array(N, dtype=np.float32)
        self.add_mesh(P, N, np.array(U, dtype=np.float32), mat_id, np.array(I, dtype=np.uint32), ao=ao)

    def add_cylinder(self, center, radius, height, n_seg, mat_id, axis="z", ao=1.0):
        h2 = height * 0.5
        prof = [(0.0, -h2), (radius, -h2), (radius, -h2 + 0.002), (radius, h2 - 0.002), (radius, h2), (0.0, h2)]
        self.add_lathe(prof, n_seg, mat_id, center=center, axis=axis, ao=ao)

    def add_jagged_poly_slab(self, pts2d, thickness, mat_id, twist=0.0):
        """Extrude an arbitrary 2D convex/star polygon into a sharp faceted 3D shard plate."""
        pts = np.asarray(pts2d, dtype=np.float32)
        n = len(pts)
        hz = thickness * 0.5
        P, N, U, I = [], [], [], []
        # Top face (+z)
        c_top = len(P)
        P.append((0, 0, hz)); N.append((0, 0, 1)); U.append((0.5, 0.5))
        for x, y in pts:
            P.append((x, y, hz + twist * x * y)); N.append((0, 0, 1)); U.append((x * 2 + 0.5, y * 2 + 0.5))
        for k in range(n):
            I.extend([c_top, c_top + 1 + k, c_top + 1 + ((k + 1) % n)])
        # Bottom face (-z)
        c_bot = len(P)
        P.append((0, 0, -hz)); N.append((0, 0, -1)); U.append((0.5, 0.5))
        for x, y in pts:
            P.append((x, y, -hz + twist * x * y)); N.append((0, 0, -1)); U.append((x * 2 + 0.5, y * 2 + 0.5))
        for k in range(n):
            I.extend([c_bot, c_bot + 1 + ((k + 1) % n), c_bot + 1 + k])
        # Crisp flat-shaded fracture rim faces
        for k in range(n):
            p0 = pts[k]; p1 = pts[(k + 1) % n]
            edge = p1 - p0
            nx, ny = float(edge[1]), float(-edge[0])
            nl = math.hypot(nx, ny) + 1e-9
            nx /= nl; ny /= nl
            z0 = twist * p0[0] * p0[1]
            z1 = twist * p1[0] * p1[1]
            base = len(P)
            P.extend([
                (p0[0], p0[1], -hz + z0), (p1[0], p1[1], -hz + z1),
                (p1[0], p1[1],  hz + z1), (p0[0], p0[1],  hz + z0),
            ])
            N.extend([(nx, ny, 0.0)] * 4)
            U.extend([(0, 0), (nl * 2, 0), (nl * 2, thickness * 2), (0, thickness * 2)])
            I.extend([base, base + 1, base + 2, base, base + 2, base + 3])
        self.add_mesh(P, N, U, mat_id, I)

    def build(self):
        return {
            "pos": np.concatenate(self.pos, axis=0),
            "nrm": np.concatenate(self.nrm, axis=0),
            "uv":  np.concatenate(self.uv,  axis=0),
            "mat": np.concatenate(self.mat, axis=0),
            "ao":  np.concatenate(self.ao,  axis=0),
            "idx": np.concatenate(self.idx, axis=0),
        }


def rot_xyz(rx, ry, rz):
    cx, sx = math.cos(rx), math.sin(rx)
    cy, sy = math.cos(ry), math.sin(ry)
    cz, sz = math.cos(rz), math.sin(rz)
    Rx = np.array([[1, 0, 0], [0, cx, -sx], [0, sx, cx]])
    Ry = np.array([[cy, 0, sy], [0, 1, 0], [-sy, 0, cy]])
    Rz = np.array([[cz, -sz, 0], [sz, cz, 0], [0, 0, 1]])
    return Rz @ Ry @ Rx


def build_crt_tv():
    """Detailed Retro 1990s CRT TV (0.50m W x 0.44m D x 0.44m H): cabinet, bezel, curved glass tube, dials, antennas."""
    mb = MeshBuilder()
    # Main front cabinet housing (charcoal matte plastic, mat 0)
    mb.add_box((0.0, 0.02, 0.0), (0.50, 0.24, 0.44), 0)
    # Tapered rear cathode-tube bulge (mat 0)
    mb.add_box((0.0, 0.16, 0.0), (0.36, 0.16, 0.34), 0)
    mb.add_box((0.0, 0.23, 0.0), (0.20, 0.08, 0.20), 0)
    # Recessed front bezel border bars framing the CRT screen (front face is -Y)
    mb.add_box((0.0, -0.105, 0.185), (0.50, 0.03, 0.07), 0)   # top bezel
    mb.add_box((0.0, -0.105, -0.175), (0.50, 0.03, 0.09), 0)  # bottom control panel
    mb.add_box((-0.215, -0.105, 0.005), (0.07, 0.03, 0.32), 0)  # left bezel
    mb.add_box((0.185, -0.105, 0.005), (0.13, 0.03, 0.32), 0)   # right control strip
    # Glossy curved dark CRT glass screen (mat 1) — sculpted slight spherical pillow bulge!
    nu, nv = 12, 10
    us = np.linspace(-0.175, 0.115, nu + 1)
    vs = np.linspace(-0.125, 0.145, nv + 1)
    P, N, U, I = [], [], [], []
    for iv, z in enumerate(vs):
        for iu, x in enumerate(us):
            nx_u = (x + 0.03) / 0.15
            nz_v = (z - 0.01) / 0.14
            bulge = 0.026 * max(0.0, 1.0 - 0.45 * (nx_u * nx_u + nz_v * nz_v))
            y = -0.092 - bulge
            P.append((x, y, z))
            nn = np.array([nx_u * 0.24, -1.0, nz_v * 0.22], dtype=np.float32)
            nn /= np.linalg.norm(nn)
            N.append(nn)
            U.append((iu / nu, iv / nv))
    stride = nu + 1
    for iv in range(nv):
        for iu in range(nu):
            a = iv * stride + iu
            I.extend([a, a + 1, a + stride + 1, a, a + stride + 1, a + stride])
    mb.add_mesh(P, N, U, 1, I)
    # Retro side rotary tuning dials & speaker vent slits (metallic/dark steel mat 3)
    mb.add_cylinder((0.185, -0.125, 0.085), 0.024, 0.025, 16, 3, axis="y")
    mb.add_cylinder((0.185, -0.125, 0.020), 0.024, 0.025, 16, 3, axis="y")
    for k in range(5):
        mb.add_box((0.185, -0.122, -0.045 - k * 0.018), (0.065, 0.008, 0.007), 3)
    # Rubber feet
    for sx in (-0.19, 0.19):
        for sy in (-0.07, 0.12):
            mb.add_box((sx, sy, -0.23), (0.05, 0.05, 0.025), 7)
    # Twin V rabbit-ear metallic antennas on top!
    mb.add_cylinder((0.0, -0.01, 0.23), 0.038, 0.025, 14, 0, axis="z")
    mb.add_box((-0.075, -0.01, 0.34), (0.010, 0.010, 0.24), 3, rot=rot_xyz(0.0, -0.42, 0.0))
    mb.add_box((0.075, -0.01, 0.34), (0.010, 0.010, 0.24), 3, rot=rot_xyz(0.0, 0.42, 0.0))
    return mb.build()


def build_wood_crate():
    """Detailed Framed Wooden Shipping Crate (0.52m cube): corner rails, recessed pine slats, diagonal braces, iron brackets."""
    mb = MeshBuilder()
    S = 0.52
    h = S * 0.5
    t = 0.038
    # 12 outer structural timber frame beams (mat 2)
    for sx in (-1, 1):
        for sy in (-1, 1):
            mb.add_box((sx * (h - t * 0.5), sy * (h - t * 0.5), 0.0), (t, t, S), 2)
    for sz in (-1, 1):
        for sy in (-1, 1):
            mb.add_box((0.0, sy * (h - t * 0.5), sz * (h - t * 0.5)), (S - 2 * t, t, t), 2)
        for sx in (-1, 1):
            mb.add_box((sx * (h - t * 0.5), 0.0, sz * (h - t * 0.5)), (t, S - 2 * t, t), 2)
    # Recessed horizontal pine slats on all 6 faces (baked crevice AO=0.58 inside outer frame!)
    inner = S - 2 * t
    for side in (-1, 1):
        for k in range(4):
            off = -inner * 0.38 + k * (inner * 0.255)
            # +/- Y faces
            mb.add_box((0.0, side * (h - t * 0.72), off), (inner + 0.01, 0.018, inner * 0.21), 2, ao=0.58)
            # +/- X faces
            mb.add_box((side * (h - t * 0.72), 0.0, off), (0.018, inner + 0.01, inner * 0.21), 2, ao=0.58)
            # +/- Z faces
            mb.add_box((off, 0.0, side * (h - t * 0.72)), (inner * 0.21, inner + 0.01, 0.018), 2, ao=0.58)
        # Diagonal reinforcement board on +/- Y and +/- X faces!
        mb.add_box((0.0, side * (h - t * 0.35), 0.0), (inner * 1.24, 0.022, 0.052), 2, rot=rot_xyz(0, 0.785, 0), ao=0.84)
        mb.add_box((side * (h - t * 0.35), 0.0, 0.0), (0.022, inner * 1.24, 0.052), 2, rot=rot_xyz(0.785, 0, 0), ao=0.84)
    # 8 forged iron corner reinforcement brackets (mat 3)
    for sx in (-1, 1):
        for sy in (-1, 1):
            for sz in (-1, 1):
                mb.add_box((sx * (h - 0.015), sy * (h - 0.015), sz * (h - 0.015)), (0.045, 0.045, 0.045), 3)
    return mb.build()


def build_ceramic_vase():
    """Sculpted Glazed Terracotta Amphora / Ceramic Vase (0.36m dia x 0.56m tall) with rolled lip & twin curved handles."""
    mb = MeshBuilder()
    # Outer + inner hollow classical amphora lathe profile (bottom z=-0.28 -> top lip z=+0.28 -> hollow throat)
    prof = [
        (0.000, -0.280),
        (0.095, -0.280),
        (0.105, -0.260),
        (0.118, -0.210),
        (0.162, -0.120),
        (0.184, -0.020),
        (0.172,  0.080),
        (0.122,  0.155),
        (0.076,  0.205),
        (0.074,  0.245),
        (0.106,  0.272),
        (0.112,  0.282),
        (0.094,  0.284),
        (0.062,  0.248),
        (0.058,  0.170),
        (0.000,  0.150),
    ]
    mb.add_lathe(prof, 28, 4, uv_scale=3.0)
    # Twin sculpted amphora handles (segmented curved clay tubes on +X and -X)
    for sx in (-1, 1):
        pts = [
            (sx * 0.145, 0.0, 0.095),
            (sx * 0.195, 0.0, 0.145),
            (sx * 0.185, 0.0, 0.215),
            (sx * 0.125, 0.0, 0.252),
            (sx * 0.082, 0.0, 0.242),
        ]
        for k in range(len(pts) - 1):
            p0 = np.array(pts[k]); p1 = np.array(pts[k + 1])
            mid = 0.5 * (p0 + p1)
            d = p1 - p0; ln = float(np.linalg.norm(d))
            ang_y = math.atan2(d[0], d[2])
            mb.add_box(tuple(mid), (0.026, 0.028, ln + 0.014), 4, rot=rot_xyz(0, ang_y, 0))
    return mb.build()


def build_cinder_block():
    """Authentic Hollow Twin-Core Concrete Cinder Block (0.44m x 0.22m x 0.22m) with two real rectangular voids."""
    mb = MeshBuilder()
    # Front/back long walls (along X)
    mb.add_box((0.0, -0.090, 0.0), (0.44, 0.040, 0.22), 5)
    mb.add_box((0.0,  0.090, 0.0), (0.44, 0.040, 0.22), 5)
    # Left, center, and right cross-webs creating 2 crisp hollow rectangular cores (dark cavity AO=0.52!)
    mb.add_box((-0.198, 0.0, 0.0), (0.044, 0.140, 0.22), 5, ao=0.82)
    mb.add_box(( 0.000, 0.0, 0.0), (0.044, 0.140, 0.22), 5, ao=0.52)
    mb.add_box(( 0.198, 0.0, 0.0), (0.044, 0.140, 0.22), 5, ao=0.82)
    return mb.build()


def build_oil_drum():
    """Weathered Red Painted Steel Industrial Oil Drum (0.46m dia x 0.72m tall) with twin rolling hoops & top chimes."""
    mb = MeshBuilder()
    R = 0.23
    H2 = 0.36
    # Ribbed industrial drum profile with top/bottom chimes and twin center rolling beads!
    prof = [
        (0.000, -H2 + 0.015),
        (R - 0.012, -H2 + 0.015),
        (R - 0.008, -H2),
        (R + 0.008, -H2),
        (R + 0.008, -H2 + 0.020),
        (R, -H2 + 0.026),
        (R, -0.140),
        (R + 0.014, -0.125),
        (R + 0.014, -0.105),
        (R, -0.090),
        (R,  0.090),
        (R + 0.014,  0.105),
        (R + 0.014,  0.125),
        (R,  0.140),
        (R,  H2 - 0.026),
        (R + 0.008, H2 - 0.020),
        (R + 0.008, H2),
        (R - 0.008, H2),
        (R - 0.012, H2 - 0.016),
        (0.000, H2 - 0.016),
    ]
    mb.add_lathe(prof, 32, 6, uv_scale=2.5)
    # Metallic bung caps on top lid (mat 3)
    mb.add_cylinder((0.12, 0.0, H2 - 0.005), 0.028, 0.022, 14, 3, axis="z")
    mb.add_cylinder((-0.11, 0.05, H2 - 0.005), 0.018, 0.020, 12, 3, axis="z")
    return mb.build()


def build_car_tire():
    """Sculpted Car Tire + Recessed Steel Wheel Rim (0.62m OD x 0.24m width, wheel axle along Y)."""
    mb = MeshBuilder()
    # Thick rubber tire cross-section lathed around Y axle
    prof_tire = [
        (0.185, -0.095),
        (0.245, -0.118),
        (0.298, -0.112),
        (0.312, -0.088),
        (0.314, -0.035),
        (0.305, -0.025),
        (0.314, -0.015),
        (0.314,  0.015),
        (0.305,  0.025),
        (0.314,  0.035),
        (0.312,  0.088),
        (0.298,  0.112),
        (0.245,  0.118),
        (0.185,  0.095),
        (0.180,  0.060),
        (0.180, -0.060),
        (0.185, -0.095),
    ]
    mb.add_lathe(prof_tire, 34, 7, axis="y", uv_scale=4.0)
    # Sculpted tread lugs around outer circumference for authentic tire silhouette!
    for k in range(20):
        a = k * (2 * math.pi / 20)
        x = math.cos(a) * 0.312
        z = math.sin(a) * 0.312
        mb.add_box((x, -0.062, z), (0.020, 0.058, 0.042), 7, rot=rot_xyz(0, -a, 0))
        mb.add_box((x,  0.062, z), (0.020, 0.058, 0.042), 7, rot=rot_xyz(0, -a + 0.15, 0))
    # Recessed metallic wheel rim & hub (mat 3)
    prof_rim = [
        (0.000, -0.045),
        (0.065, -0.045),
        (0.085, -0.025),
        (0.165, -0.038),
        (0.186, -0.082),
        (0.186,  0.082),
        (0.165,  0.038),
        (0.085,  0.025),
        (0.065,  0.045),
        (0.000,  0.045),
    ]
    mb.add_lathe(prof_rim, 24, 3, axis="y", uv_scale=2.0, ao=0.56)
    # 5 wheel lug nuts on both sides
    for k in range(5):
        a = k * (2 * math.pi / 5)
        mb.add_cylinder((math.cos(a) * 0.055, -0.052, math.sin(a) * 0.055), 0.010, 0.016, 8, 3, axis="y")
        mb.add_cylinder((math.cos(a) * 0.055,  0.052, math.sin(a) * 0.055), 0.010, 0.016, 8, 3, axis="y")
    return mb.build()


def build_fracture_piece(kind: str, variant: int):
    """Build realistic jagged fractured debris piece meshes for each shatterable prop type.
    Returns (mesh_dict, half_extents_xyz) so MuJoCo collision box matches rendered geometry tightly!"""
    rng = np.random.default_rng(1000 + hash((kind, variant)) % 99991)
    mb = MeshBuilder()

    if kind == "wood_plank":
        # Snapped pine wood crate slats & corner beams!
        if variant % 4 == 0:
            # Chunky corner timber beam
            L = float(rng.uniform(0.24, 0.44))
            W = float(rng.uniform(0.036, 0.048))
            T = float(rng.uniform(0.034, 0.046))
            mb.add_box((0, 0, 0), (L, W, T), 2)
            return mb.build(), (L * 0.5, W * 0.5, T * 0.5)
        else:
            # Jagged splintered slat board
            L = float(rng.uniform(0.22, 0.46))
            W = float(rng.uniform(0.065, 0.115))
            T = float(rng.uniform(0.016, 0.024))
            pts = [
                (-L * 0.48, -W * 0.46),
                ( L * 0.44, -W * 0.50),
                ( L * 0.52, -W * 0.08),
                ( L * 0.40,  W * 0.48),
                (-L * 0.50,  W * 0.44),
                (-L * 0.42,  W * 0.05),
            ]
            mb.add_jagged_poly_slab(pts, T, 2)
            return mb.build(), (L * 0.48, W * 0.46, T * 0.5)

    elif kind == "vase_shard":
        # Sharp curved terracotta ceramic shell shard!
        R = float(rng.uniform(0.065, 0.135))
        n_v = int(rng.integers(5, 8))
        angs = np.sort(rng.uniform(0, 2 * math.pi, n_v))
        pts = [(math.cos(a) * R * float(rng.uniform(0.55, 1.05)),
                math.sin(a) * R * float(rng.uniform(0.45, 0.95))) for a in angs]
        T = float(rng.uniform(0.012, 0.020))
        mb.add_jagged_poly_slab(pts, T, 4, twist=float(rng.uniform(-1.6, 1.6)))
        return mb.build(), (R * 0.72, R * 0.65, max(T * 0.9, 0.016))

    elif kind == "concrete_chunk":
        # Heavy irregular faceted concrete rubble block!
        sx = float(rng.uniform(0.065, 0.145))
        sy = float(rng.uniform(0.055, 0.120))
        sz = float(rng.uniform(0.050, 0.105))
        n_v = int(rng.integers(5, 8))
        angs = np.sort(rng.uniform(0, 2 * math.pi, n_v))
        pts = [(math.cos(a) * sx * float(rng.uniform(0.65, 1.0)),
                math.sin(a) * sy * float(rng.uniform(0.65, 1.0))) for a in angs]
        mb.add_jagged_poly_slab(pts, sz * 1.4, 5)
        return mb.build(), (sx * 0.75, sy * 0.75, sz * 0.68)

    else:  # "crt_shard"
        # Mixed CRT debris: broken charcoal casing panels (mat 0), glossy dark glass shards (mat 1), green PCB (mat 8)!
        sub = variant % 7
        if sub in (0, 1, 2):
            # Broken charcoal TV plastic cabinet wall panel
            R = float(rng.uniform(0.095, 0.175))
            angs = np.sort(rng.uniform(0, 2 * math.pi, 6))
            pts = [(math.cos(a) * R * float(rng.uniform(0.6, 1.05)),
                    math.sin(a) * R * float(rng.uniform(0.5, 0.95))) for a in angs]
            mb.add_jagged_poly_slab(pts, 0.022, 0)
            return mb.build(), (R * 0.75, R * 0.68, 0.016)
        elif sub in (3, 4, 5):
            # Sharp glossy dark-glass CRT tube shard
            R = float(rng.uniform(0.075, 0.145))
            angs = np.sort(rng.uniform(0, 2 * math.pi, 5))
            pts = [(math.cos(a) * R * float(rng.uniform(0.5, 1.05)),
                    math.sin(a) * R * float(rng.uniform(0.45, 0.95))) for a in angs]
            mb.add_jagged_poly_slab(pts, 0.014, 1, twist=float(rng.uniform(-1.2, 1.2)))
            return mb.build(), (R * 0.72, R * 0.65, 0.014)
        else:
            # Green internal PCB circuit board + dark metal transformer chunk!
            mb.add_box((0, 0, 0), (0.19, 0.14, 0.012), 8)
            mb.add_box((0.03, -0.02, 0.032), (0.055, 0.055, 0.052), 3)
            mb.add_cylinder((-0.05, 0.03, 0.025), 0.018, 0.038, 12, 0, axis="z")
            return mb.build(), (0.095, 0.070, 0.032)


def scale_mesh(md, he, s):
    md2 = dict(md)
    md2["pos"] = md["pos"] * np.float32(s)
    return md2, (he[0] * s, he[1] * s, he[2] * s)


def build_all_meshes():
    """Return dictionary mapping mesh_name -> (mesh_dict, half_extents), scaled for bold cinema readability."""
    raw = {
        "crt_tv":       (build_crt_tv(),       (0.25, 0.22, 0.23)),
        "wood_crate":   (build_wood_crate(),   (0.26, 0.26, 0.26)),
        "ceramic_vase": (build_ceramic_vase(), (0.18, 0.18, 0.28)),
        "cinder_block": (build_cinder_block(), (0.22, 0.11, 0.11)),
        "oil_drum":     (build_oil_drum(),     (0.235, 0.235, 0.36)),
        "car_tire":     (build_car_tire(),     (0.315, 0.122, 0.315)),
    }
    for kind in ("wood_plank", "vase_shard", "concrete_chunk", "crt_shard"):
        for v in range(6):
            raw[f"{kind}_{v}"] = build_fracture_piece(kind, v)
    return {k: scale_mesh(md, he, 1.28) for k, (md, he) in raw.items()}
