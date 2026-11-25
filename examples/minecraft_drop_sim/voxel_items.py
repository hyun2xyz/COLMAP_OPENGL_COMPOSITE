"""Procedural 3D Modeled Crates, Pixel-Textured Blocks, Voxel Debris Cubes, and 3D Extruded 16x16 Pixel-Art Floating Items.

Unified PBR vertex layout (14 floats per vertex):
  pos(3f), nrm(3f), uv(2f), col(3f sRGB), pbr(3f: roughness, metallic, ao)
"""
import math
import numpy as np


class VoxelMeshBuilder:
    def __init__(self):
        self.pos, self.nrm, self.uv, self.col, self.pbr, self.idx = [], [], [], [], [], []
        self._vc = 0

    def add_quad(self, p0, p1, p2, p3, nrm, col, rough=0.65, metal=0.0, ao=1.0, uvs=((0, 0), (1, 0), (1, 1), (0, 1))):
        base = self._vc
        self.pos.extend([p0, p1, p2, p3])
        self.nrm.extend([nrm] * 4)
        self.uv.extend(uvs)
        self.col.extend([col] * 4)
        self.pbr.extend([(rough, metal, ao)] * 4)
        self.idx.extend([base, base + 1, base + 2, base, base + 2, base + 3])
        self._vc += 4

    def add_box(self, center, size, col, rough=0.65, metal=0.0, ao=1.0, rot=None):
        cx, cy, cz = center
        hx, hy, hz = size[0] * 0.5, size[1] * 0.5, size[2] * 0.5
        faces = [
            ((1, 0, 0),  [(hx, -hy, -hz), (hx, hy, -hz), (hx, hy, hz), (hx, -hy, hz)]),
            ((-1, 0, 0), [(-hx, hy, -hz), (-hx, -hy, -hz), (-hx, -hy, hz), (-hx, hy, hz)]),
            ((0, 1, 0),  [(hx, hy, -hz), (-hx, hy, -hz), (-hx, hy, hz), (hx, hy, hz)]),
            ((0, -1, 0), [(-hx, -hy, -hz), (hx, -hy, -hz), (hx, -hy, hz), (-hx, -hy, hz)]),
            ((0, 0, 1),  [(-hx, -hy, hz), (hx, -hy, hz), (hx, hy, hz), (-hx, hy, hz)]),
            ((0, 0, -1), [(-hx, hy, -hz), (hx, hy, -hz), (hx, -hy, -hz), (-hx, -hy, -hz)]),
        ]
        R = np.asarray(rot, dtype=np.float32) if rot is not None else None
        c_arr = np.array([cx, cy, cz], dtype=np.float32)
        for n, pts in faces:
            P = np.array(pts, dtype=np.float32)
            N = np.array(n, dtype=np.float32)
            if R is not None:
                P = P @ R.T
                N = N @ R.T
            P = P + c_arr
            self.add_quad(tuple(P[0]), tuple(P[1]), tuple(P[2]), tuple(P[3]), tuple(N), col, rough, metal, ao)

    def build(self):
        return {
            "pos": np.array(self.pos, dtype=np.float32),
            "nrm": np.array(self.nrm, dtype=np.float32),
            "uv":  np.array(self.uv,  dtype=np.float32),
            "col": np.array(self.col, dtype=np.float32),
            "pbr": np.array(self.pbr, dtype=np.float32),
            "idx": np.array(self.idx, dtype=np.uint32),
        }


def rot_xyz(rx, ry, rz):
    cx, sx = math.cos(rx), math.sin(rx)
    cy, sy = math.cos(ry), math.sin(ry)
    cz, sz = math.cos(rz), math.sin(rz)
    Rx = np.array([[1, 0, 0], [0, cx, -sx], [0, sx, cx]])
    Ry = np.array([[cy, 0, sy], [0, 1, 0], [-sy, 0, cy]])
    Rz = np.array([[cz, -sz, 0], [sz, cz, 0], [0, 0, 1]])
    return Rz @ Ry @ Rx


def extrude_pixel_sprite(rows, palette, px_m=0.025, depth_m=0.044):
    """Compile a 2D ASCII pixel-art sprite into a true thick 3D extruded voxel mesh (ONLY exterior faces!).
    Sprite stands upright in XZ plane (X=right, Z=up, Y=thickness) so Z-axis rotation spins cleanly like Minecraft!"""
    H = len(rows)
    W = len(rows[0])
    occ = [[rows[r][c] in palette for c in range(W)] for r in range(H)]
    mb = VoxelMeshBuilder()
    hy = depth_m * 0.5
    ox = -W * 0.5 * px_m
    oz = -H * 0.5 * px_m

    for r in range(H):
        for c in range(W):
            ch = rows[r][c]
            if ch not in palette:
                continue
            col, rough, metal = palette[ch]
            x0 = ox + c * px_m
            x1 = x0 + px_m
            # Row 0 is top -> highest Z!
            z1 = oz + (H - r) * px_m
            z0 = z1 - px_m

            # Front (-Y) and Back (+Y) faces (slightly brighter edge bevel contrast via subtle AO on neighbors)
            n_nb = sum(
                1 for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1))
                if 0 <= r + dr < H and 0 <= c + dc < W and occ[r + dr][c + dc]
            )
            face_ao = 0.90 + 0.025 * n_nb
            side_col = tuple(max(0.02, v * 0.86) for v in col)

            # Front (-Y, facing camera)
            mb.add_quad((x0, -hy, z0), (x1, -hy, z0), (x1, -hy, z1), (x0, -hy, z1), (0, -1, 0), col, rough, metal, face_ao)
            # Back (+Y)
            mb.add_quad((x1, hy, z0), (x0, hy, z0), (x0, hy, z1), (x1, hy, z1), (0, 1, 0), col, rough, metal, face_ao)

            # Only emit exposed thickness side faces (clean watertight 3D pixel edge!)
            if c == 0 or not occ[r][c - 1]:  # Left (-X)
                mb.add_quad((x0, hy, z0), (x0, -hy, z0), (x0, -hy, z1), (x0, hy, z1), (-1, 0, 0), side_col, rough, metal, 0.88)
            if c == W - 1 or not occ[r][c + 1]:  # Right (+X)
                mb.add_quad((x1, -hy, z0), (x1, hy, z0), (x1, hy, z1), (x1, -hy, z1), (1, 0, 0), side_col, rough, metal, 0.88)
            if r == 0 or not occ[r - 1][c]:  # Top (+Z)
                mb.add_quad((x0, -hy, z1), (x1, -hy, z1), (x1, hy, z1), (x0, hy, z1), (0, 0, 1), col, rough, metal, 1.0)
            if r == H - 1 or not occ[r + 1][c]:  # Bottom (-Z)
                mb.add_quad((x0, hy, z0), (x1, hy, z0), (x1, -hy, z0), (x0, -hy, z0), (0, 0, -1), side_col, rough, metal, 0.72)

    return mb.build()


# ==============================================================================
#  9 ICONIC 16x16 3D EXTRUDED PIXEL-ART FLOATING LOOT ITEMS!
# ==============================================================================

ITEM_SPRITES = {
    "item_diamond": (
        [
            "................",
            ".....OOOOOO.....",
            "...OOBBHHBBOO...",
            "..OBHHWWWHHBBO..",
            ".OBHWWWHHHCCBBO.",
            ".OBHWWCCCCCCCBO.",
            ".OBHCCCCCCCDBBO.",
            ".OBBCCCCCDDDDBO.",
            "..OBBCCDDDDDBO..",
            "...OBBDDDDDBO...",
            "....OBBDDDDO....",
            ".....OBBDDO.....",
            "......OBBO......",
            ".......OO.......",
            "................",
            "................",
        ],
        {
            "O": ((0.06, 0.24, 0.30), 0.25, 0.15),
            "D": ((0.12, 0.56, 0.66), 0.14, 0.18),
            "B": ((0.20, 0.78, 0.86), 0.10, 0.20),
            "C": ((0.36, 0.90, 0.96), 0.08, 0.18),
            "H": ((0.64, 0.96, 1.00), 0.06, 0.15),
            "W": ((0.92, 0.99, 1.00), 0.05, 0.10),
        },
        0.028, 0.052,
    ),
    "item_sword": (
        [
            ".............OO.",
            "............OHBO",
            "...........OHWBO",
            "..........OHWBBO",
            ".........OHWBBO.",
            "........OHWBBO..",
            ".......OHWBBO...",
            "......OHWBBO....",
            "..OO.OHWBBO.....",
            "..OGOHWBBO......",
            "...OGGGGO.......",
            "...ORGGGO.......",
            "..ORRGGO........",
            ".ORRO.OO........",
            ".OKO............",
            "..O.............",
        ],
        {
            "O": ((0.08, 0.14, 0.18), 0.40, 0.25),
            "B": ((0.18, 0.72, 0.82), 0.12, 0.35),
            "H": ((0.46, 0.92, 0.98), 0.09, 0.30),
            "W": ((0.86, 0.98, 1.00), 0.07, 0.25),
            "G": ((0.10, 0.42, 0.48), 0.28, 0.65),
            "R": ((0.46, 0.27, 0.12), 0.76, 0.02),
            "K": ((0.24, 0.58, 0.66), 0.22, 0.70),
        },
        0.029, 0.044,
    ),
    "item_gapple": (
        [
            "........OO......",
            ".......OTTO.....",
            "........OO......",
            "....OOOO..OOOO..",
            "..OOYYYYOOYYYYO.",
            ".OYWWYYYYYYYYGO.",
            ".OYWYYYYYYYYGGO.",
            ".OYYYYYYYYYGGDO.",
            ".OYYYYYYYYGGDDO.",
            ".OYYYYYYYGGDDDO.",
            "..OYYYYYGGDDDO..",
            "..OYYYYGGDDDDO..",
            "...OYYGGDDDDO...",
            "....OOGGDDOO....",
            "......OOOO......",
            "................",
        ],
        {
            "O": ((0.30, 0.18, 0.03), 0.32, 0.55),
            "T": ((0.42, 0.24, 0.08), 0.75, 0.05),
            "D": ((0.64, 0.40, 0.05), 0.22, 0.88),
            "G": ((0.84, 0.60, 0.08), 0.18, 0.92),
            "Y": ((0.98, 0.82, 0.22), 0.14, 0.95),
            "W": ((1.00, 0.96, 0.70), 0.08, 0.95),
        },
        0.027, 0.056,
    ),
    "item_pickaxe": (
        [
            "....OOOOOOO.....",
            "...OOHHHHBBBO...",
            "..OHWWWHHHBBBO..",
            "..OOBBHHWWWBBO..",
            "....OOOBBHWBBO..",
            "........ORBBBO..",
            ".......ORROBBO..",
            "......ORRO.OBO..",
            ".....ORRO...OO..",
            "....ORRO........",
            "...ORRO.........",
            "..ORRO..........",
            ".ORRO...........",
            ".ORO............",
            "..O.............",
            "................",
        ],
        {
            "O": ((0.08, 0.16, 0.20), 0.38, 0.25),
            "B": ((0.16, 0.68, 0.78), 0.14, 0.35),
            "H": ((0.44, 0.90, 0.96), 0.10, 0.30),
            "W": ((0.84, 0.98, 1.00), 0.08, 0.20),
            "R": ((0.50, 0.30, 0.13), 0.78, 0.02),
        },
        0.029, 0.044,
    ),
    "item_emerald": (
        [
            "................",
            "......OOOO......",
            "....OOGGGGOO....",
            "...OGHWWGGDGO...",
            "..OGHWGGGGDDGO..",
            "..OGWGGGGGDDGO..",
            "..OGGGGGGGDDGO..",
            "..OGGGGGGGDDGO..",
            "..OGGGGGGDDDGO..",
            "..OGGGGGDDDDGO..",
            "..OGGGGDDDDDGO..",
            "...OGGGDDDDGO...",
            "....OODDDDOO....",
            "......OOOO......",
            "................",
            "................",
        ],
        {
            "O": ((0.03, 0.22, 0.08), 0.25, 0.15),
            "D": ((0.06, 0.48, 0.18), 0.14, 0.18),
            "G": ((0.14, 0.78, 0.32), 0.10, 0.20),
            "H": ((0.46, 0.95, 0.56), 0.07, 0.16),
            "W": ((0.88, 1.00, 0.90), 0.05, 0.12),
        },
        0.028, 0.054,
    ),
    "item_heart": (
        [
            "................",
            "..OOOO....OOOO..",
            ".ORRRRO..ORRRRO.",
            "ORWWRRROORRRRRRO",
            "ORWRRRRRRRRRDDRO",
            "ORRRRRRRRRRDDDDO",
            "ORRRRRRRRRDDDDDO",
            ".ORRRRRRRRDDDDO.",
            "..ORRRRRRDDDDO..",
            "...ORRRRRDDDO...",
            "....ORRRDDDO....",
            ".....ORRDDO.....",
            "......ORDO......",
            ".......OO.......",
            "................",
            "................",
        ],
        {
            "O": ((0.22, 0.02, 0.04), 0.30, 0.15),
            "D": ((0.54, 0.04, 0.08), 0.18, 0.15),
            "R": ((0.88, 0.10, 0.16), 0.12, 0.14),
            "W": ((1.00, 0.74, 0.78), 0.06, 0.10),
        },
        0.027, 0.052,
    ),
    "item_potion": (
        [
            "......OOOO......",
            "......OCCO......",
            ".....OOGGOO.....",
            "......OGGO......",
            ".....OGGGGO.....",
            "....OGWGGGGO....",
            "...OGWWPPPPGO...",
            "..OGWPPPPPDDGO..",
            "..OGPPPPPPDDGO..",
            "..OGPPPPPPDDGO..",
            "..OGPPPPPDDDGO..",
            "..OGPPPPDDDDGO..",
            "...OGPPDDDDGO...",
            "....OOGGGGOO....",
            "......OOOO......",
            "................",
        ],
        {
            "O": ((0.14, 0.12, 0.22), 0.28, 0.18),
            "C": ((0.52, 0.34, 0.18), 0.80, 0.02),
            "G": ((0.76, 0.84, 0.92), 0.08, 0.12),
            "W": ((0.96, 0.98, 1.00), 0.05, 0.10),
            "P": ((0.78, 0.18, 0.92), 0.10, 0.20),
            "D": ((0.46, 0.08, 0.62), 0.14, 0.20),
        },
        0.027, 0.052,
    ),
    "item_ingot": (
        [
            "................",
            "................",
            "......OOOOOO....",
            "....OOYYYYYYOO..",
            "...OYYWWYYYYGGO.",
            "..OYYWWYYYYGGDO.",
            ".OYYWWYYYYGGDDO.",
            ".OYYYYYYYGGDDDO.",
            ".OYYYYGGDDDDDDO.",
            ".OGGGGGDDDDDDO..",
            ".OGGGDDDDDDDO...",
            "..OODDDDDDOO....",
            "....OOOOOO......",
            "................",
            "................",
            "................",
        ],
        {
            "O": ((0.30, 0.18, 0.03), 0.30, 0.65),
            "D": ((0.62, 0.40, 0.05), 0.20, 0.90),
            "G": ((0.84, 0.62, 0.08), 0.16, 0.94),
            "Y": ((0.98, 0.84, 0.24), 0.12, 0.96),
            "W": ((1.00, 0.96, 0.72), 0.08, 0.96),
        },
        0.028, 0.056,
    ),
    "item_star": (
        [
            ".......OO.......",
            "......OYYO......",
            "......OWYO......",
            ".....OWYYGO.....",
            ".OOOOOWYYGOOOOO.",
            "OYYYYWWYYYGGGGGO",
            ".OYYYYYYYYGGGGO.",
            "..OYYYYYYGGGGO..",
            "...OYYYYGGGGO...",
            "...OYYYYGGGGO...",
            "..OYYYYGOGGGGO..",
            "..OYYGGO.OGGGO..",
            ".OYYGGO...OGGGO.",
            ".OOGGO.....OGGO.",
            "..OOO.......OOO.",
            "................",
        ],
        {
            "O": ((0.32, 0.20, 0.03), 0.30, 0.55),
            "G": ((0.84, 0.58, 0.08), 0.16, 0.92),
            "Y": ((0.99, 0.84, 0.24), 0.12, 0.94),
            "W": ((1.00, 0.98, 0.78), 0.06, 0.95),
        },
        0.028, 0.048,
    ),
}


# ==============================================================================
#  FALLING 3D MODELED CRATES & PIXEL-TEXTURED VOXEL BLOCKS + VOXEL DEBRIS CUBES
# ==============================================================================

def build_pixel_cube_block(kind: str, size: float = 0.56, seed: int = 1):
    """Build a crisp 8x8-per-face 3D pixelated block (Diamond Ore, Gold Ore, Emerald Ore, Oak Crate, Brick, Obsidian)
    with slight 3D micro-bevel/recess relief on ore gems & seams so raking sun creates rich voxel contrast!"""
    rng = np.random.default_rng(seed)
    mb = VoxelMeshBuilder()
    hs = size * 0.5
    N = 8
    cell = size / N

    def texel_props(f_idx, ix, iy):
        n = float(rng.random())
        is_border = (ix == 0 or ix == N - 1 or iy == 0 or iy == N - 1)
        if kind == "diamond_ore":
            # Crisp pixelated stone with embedded bright cyan diamond ore clusters!
            gem = ((ix, iy) in {(2, 2), (3, 2), (2, 3), (5, 4), (6, 4), (5, 5), (3, 6)})
            if gem:
                return (0.24, 0.86, 0.94) if n > 0.35 else (0.58, 0.96, 1.00), 0.10, 0.25, 0.008
            v = 0.34 + 0.14 * n
            return (v, v, v * 0.98), 0.88, 0.0, 0.0
        elif kind == "gold_ore":
            gem = ((ix, iy) in {(1, 2), (2, 2), (2, 3), (5, 2), (5, 5), (6, 5), (4, 6)})
            if gem:
                return (0.94, 0.74, 0.16) if n > 0.35 else (0.76, 0.52, 0.08), 0.16, 0.92, 0.008
            v = 0.34 + 0.14 * n
            return (v, v * 0.99, v * 0.96), 0.88, 0.0, 0.0
        elif kind == "emerald_ore":
            gem = ((ix, iy) in {(2, 1), (3, 2), (5, 3), (5, 4), (2, 5), (3, 5), (6, 6)})
            if gem:
                return (0.14, 0.82, 0.32) if n > 0.35 else (0.48, 0.96, 0.56), 0.10, 0.22, 0.008
            v = 0.33 + 0.14 * n
            return (v, v, v * 0.98), 0.88, 0.0, 0.0
        elif kind == "brick_block":
            # Terracotta pixelated bricks with pale mortar seams!
            mortar = (iy in (0, 4)) or (iy < 4 and ix in (0, 4)) or (iy >= 4 and ix in (2, 6))
            if mortar:
                return (0.64, 0.61, 0.56), 0.92, 0.0, -0.005
            return (0.54 + 0.12 * n, 0.21 + 0.07 * n, 0.14 + 0.05 * n), 0.82, 0.0, 0.004
        elif kind == "obsidian_box":
            # Dark glossy carved obsidian-treasure box with gleaming gold trim corners!
            if (ix in (0, N - 1) and iy in (0, N - 1)) or (ix in (3, 4) and iy in (3, 4)):
                return (0.90, 0.70, 0.16), 0.18, 0.90, 0.008
            if is_border:
                return (0.16, 0.12, 0.22), 0.35, 0.20, 0.005
            return (0.08 + 0.07 * n, 0.06 + 0.04 * n, 0.13 + 0.08 * n), 0.24, 0.15, 0.0
        else:  # "wood_crate" (beveled pixel oak loot box with recessed slats & iron corner rivets!)
            if ix in (0, N - 1) and iy in (0, N - 1):
                return (0.24, 0.25, 0.27), 0.30, 0.85, 0.010  # iron corner bracket!
            if is_border:
                return (0.48 + 0.08 * n, 0.32 + 0.06 * n, 0.16 + 0.04 * n), 0.74, 0.0, 0.008
            if ix == iy or ix == (N - 1 - iy):
                return (0.44 + 0.08 * n, 0.29 + 0.05 * n, 0.14 + 0.04 * n), 0.76, 0.0, 0.003
            # Recessed dark wood plank slat inside frame (authentic crevice shadow!)
            seam = 0.74 if (iy % 2 == 0) else 1.0
            return (0.56 * seam + 0.06 * n, 0.39 * seam + 0.05 * n, 0.21 * seam + 0.03 * n), 0.80, 0.0, -0.010

    # Emit 6 faces of 8x8 voxel-textured cube with crisp per-cell extrusions!
    face_axes = [
        ((1, 0, 0),  (0, 1, 0),  (0, 0, 1)),
        ((-1, 0, 0), (0, -1, 0), (0, 0, 1)),
        ((0, 1, 0),  (-1, 0, 0), (0, 0, 1)),
        ((0, -1, 0), (1, 0, 0),  (0, 0, 1)),
        ((0, 0, 1),  (1, 0, 0),  (0, 1, 0)),
        ((0, 0, -1), (1, 0, 0),  (0, -1, 0)),
    ]
    for fi, (n_vec, u_vec, v_vec) in enumerate(face_axes):
        nv = np.array(n_vec, dtype=np.float32)
        uv = np.array(u_vec, dtype=np.float32)
        vv = np.array(v_vec, dtype=np.float32)
        for iy in range(N):
            for ix in range(N):
                col, rough, metal, relief = texel_props(fi, ix, iy)
                ao = 0.62 if relief < -0.004 else (1.0 if relief > 0.003 else 0.90)
                u0, u1 = -hs + ix * cell, -hs + (ix + 1) * cell
                v0, v1 = -hs + iy * cell, -hs + (iy + 1) * cell
                base_n = nv * (hs + relief)
                p0 = base_n + uv * u0 + vv * v0
                p1 = base_n + uv * u1 + vv * v0
                p2 = base_n + uv * u1 + vv * v1
                p3 = base_n + uv * u0 + vv * v1
                # Map face UVs 0..1 across entire block face so Minecraft crack overlay renders cleanly across the whole face!
                uvs = ((ix / N, iy / N), ((ix + 1) / N, iy / N), ((ix + 1) / N, (iy + 1) / N), (ix / N, (iy + 1) / N))
                mb.add_quad(tuple(p0), tuple(p1), tuple(p2), tuple(p3), n_vec, col, rough, metal, ao, uvs=uvs)

    return mb.build(), (hs, hs, hs)


def build_voxel_debris_cube(kind: str, variant: int):
    """Chunky Minecraft-style 3D pixel debris mini-cube (5cm..11cm) matching parent block colors."""
    rng = np.random.default_rng(7000 + hash((kind, variant)) % 99991)
    s = float(rng.uniform(0.055, 0.115))
    sx = s * float(rng.uniform(0.85, 1.25))
    sy = s * float(rng.uniform(0.85, 1.25))
    sz = s * float(rng.uniform(0.65, 1.15))
    palettes = {
        "diamond_ore": [((0.38, 0.38, 0.37), 0.88, 0.0), ((0.26, 0.88, 0.96), 0.10, 0.25), ((0.46, 0.46, 0.45), 0.88, 0.0)],
        "gold_ore":    [((0.38, 0.38, 0.36), 0.88, 0.0), ((0.94, 0.74, 0.16), 0.16, 0.92), ((0.48, 0.47, 0.45), 0.88, 0.0)],
        "emerald_ore": [((0.38, 0.38, 0.37), 0.88, 0.0), ((0.16, 0.84, 0.34), 0.10, 0.22), ((0.46, 0.46, 0.45), 0.88, 0.0)],
        "wood_crate":  [((0.56, 0.38, 0.20), 0.78, 0.0), ((0.44, 0.28, 0.14), 0.78, 0.0), ((0.26, 0.27, 0.29), 0.32, 0.82)],
        "brick_block": [((0.58, 0.23, 0.15), 0.82, 0.0), ((0.48, 0.19, 0.12), 0.84, 0.0), ((0.64, 0.61, 0.56), 0.90, 0.0)],
        "obsidian_box":[((0.10, 0.07, 0.16), 0.24, 0.18), ((0.16, 0.11, 0.24), 0.28, 0.18), ((0.90, 0.70, 0.16), 0.18, 0.90)],
    }
    plist = palettes.get(kind, palettes["wood_crate"])
    col, rough, metal = plist[variant % len(plist)]
    mb = VoxelMeshBuilder()
    mb.add_box((0, 0, 0), (sx, sy, sz), col, rough=rough, metal=metal, ao=0.95)
    return mb.build(), (sx * 0.5, sy * 0.5, sz * 0.5)


def build_all_minecraft_meshes():
    """Build all falling 3D blocks, voxel debris cubes, and 9 extruded 3D floating pixel-art loot items."""
    meshes = {}
    for idx, bkind in enumerate(("wood_crate", "diamond_ore", "gold_ore", "emerald_ore", "brick_block", "obsidian_box")):
        meshes[bkind] = build_pixel_cube_block(bkind, size=0.58, seed=101 + idx * 17)
        for v in range(6):
            meshes[f"deb_{bkind}_{v}"] = build_voxel_debris_cube(bkind, v)

    for iname, (rows, pal, px_m, depth_m) in ITEM_SPRITES.items():
        md = extrude_pixel_sprite(rows, pal, px_m=px_m, depth_m=depth_m)
        meshes[iname] = (md, (0.20, 0.04, 0.20))

    return meshes
