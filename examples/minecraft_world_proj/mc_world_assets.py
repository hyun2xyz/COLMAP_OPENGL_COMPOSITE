"""Authentic Beveled 3D Pixel-Art Minecraft World Objects & Structures Library.

Builds crisp beveled voxel-pixel meshes (14 floats/vert: pos3, nrm3, uv2, col3, pbr3) for
real-world 3D camera projection mapping onto architecture, walls, pillars, and nature parks:
  - grass_block, stone_brick, cobblestone, oak_planks, obsidian
  - diamond_ore, gold_ore, emerald_ore
  - crafting_table, chest, tnt_block, furnace (glowing), bookshelf, enchanting_table
  - torch (standing / wall-angled), lantern, beacon
  - oak_tree (multi-block trunk + voxel leaf canopy), nether_portal (4x5 obsidian arch + animated vortex)
  - village_cottage (cobblestone + oak planks + pitched voxel roof + glowing windows)

Material encoding in pbr3 = (roughness, metallic, ao_or_emissive):
  ao_or_emissive >= 0.0 -> vertex AO (0.15..1.0)
  ao_or_emissive <  0.0 -> self-emissive glow strength = -ao_or_emissive (for torches, furnaces, portals, beacons!)
"""
import math
import numpy as np


def _new_buf():
    return {"pos": [], "nrm": [], "uv": [], "col": [], "pbr": [], "idx": []}


def _finalize(out):
    return {
        "pos": np.array(out["pos"], dtype=np.float32),
        "nrm": np.array(out["nrm"], dtype=np.float32),
        "uv":  np.array(out["uv"],  dtype=np.float32),
        "col": np.array(out["col"], dtype=np.float32),
        "pbr": np.array(out["pbr"], dtype=np.float32),
        "idx": np.array(out["idx"], dtype=np.uint32),
    }


def _jitter(rgb, u, v, amt=0.035):
    h = math.sin(u * 127.1 + v * 311.7) * 43758.5453
    j = (h - math.floor(h) - 0.5) * 2.0 * amt
    return (
        float(np.clip(rgb[0] + j, 0.01, 0.99)),
        float(np.clip(rgb[1] + j, 0.01, 0.99)),
        float(np.clip(rgb[2] + j, 0.01, 0.99)),
    )


def _emit_quad(out, p0, p1, p2, p3, nrm, col, pbr):
    base = len(out["pos"])
    n = np.asarray(nrm, dtype=np.float32)
    n /= (np.linalg.norm(n) + 1e-8)
    for p, uv in zip((p0, p1, p2, p3), ((0.0, 0.0), (1.0, 0.0), (1.0, 1.0), (0.0, 1.0))):
        out["pos"].append(p)
        out["nrm"].append(tuple(n))
        out["uv"].append(uv)
        out["col"].append(col)
        out["pbr"].append(pbr)
    out["idx"].extend([base, base + 1, base + 2, base, base + 2, base + 3])


def _add_pixel_box(out, x0, x1, y0, y1, z0, z1, nx, ny, nz, col_fn, pbr_fn):
    """Add an axis-aligned Minecraft block subdivided into crisp per-pixel tiles on all 6 faces."""
    dx = (x1 - x0) / nx
    dy = (y1 - y0) / ny
    dz = (z1 - z0) / nz
    # +Y front
    for ix in range(nx):
        xa, xb = x0 + ix * dx, x0 + (ix + 1) * dx
        for iz in range(nz):
            zb, zt = z1 - (iz + 1) * dz, z1 - iz * dz
            _emit_quad(out, (xa, y1, zb), (xb, y1, zb), (xb, y1, zt), (xa, y1, zt), (0, 1, 0),
                       col_fn("+Y", ix, iz), pbr_fn("+Y", ix, iz))
    # -Y back
    for ix in range(nx):
        xb, xa = x1 - ix * dx, x1 - (ix + 1) * dx
        for iz in range(nz):
            zb, zt = z1 - (iz + 1) * dz, z1 - iz * dz
            _emit_quad(out, (xb, y0, zb), (xa, y0, zb), (xa, y0, zt), (xb, y0, zt), (0, -1, 0),
                       col_fn("-Y", ix, iz), pbr_fn("-Y", ix, iz))
    # +X right
    for iy in range(ny):
        yb, ya = y1 - iy * dy, y1 - (iy + 1) * dy
        for iz in range(nz):
            zb, zt = z1 - (iz + 1) * dz, z1 - iz * dz
            _emit_quad(out, (x1, yb, zb), (x1, ya, zb), (x1, ya, zt), (x1, yb, zt), (1, 0, 0),
                       col_fn("+X", iy, iz), pbr_fn("+X", iy, iz))
    # -X left
    for iy in range(ny):
        ya, yb = y0 + iy * dy, y0 + (iy + 1) * dy
        for iz in range(nz):
            zb, zt = z1 - (iz + 1) * dz, z1 - iz * dz
            _emit_quad(out, (x0, ya, zb), (x0, yb, zb), (x0, yb, zt), (x0, ya, zt), (-1, 0, 0),
                       col_fn("-X", iy, iz), pbr_fn("-X", iy, iz))
    # +Z top
    for ix in range(nx):
        xa, xb = x0 + ix * dx, x0 + (ix + 1) * dx
        for iy in range(ny):
            ya, yb = y0 + iy * dy, y0 + (iy + 1) * dy
            _emit_quad(out, (xa, yb, z1), (xb, yb, z1), (xb, ya, z1), (xa, ya, z1), (0, 0, 1),
                       col_fn("+Z", ix, iy), pbr_fn("+Z", ix, iy))
    # -Z bottom
    for ix in range(nx):
        xa, xb = x0 + ix * dx, x0 + (ix + 1) * dx
        for iy in range(ny):
            ya, yb = y0 + iy * dy, y0 + (iy + 1) * dy
            _emit_quad(out, (xa, ya, z0), (xb, ya, z0), (xb, yb, z0), (xa, yb, z0), (0, 0, -1),
                       col_fn("-Z", ix, iy), pbr_fn("-Z", ix, iy))


def _add_solid_box(out, center, size, col, pbr=(0.78, 0.0, 0.96), top_col=None):
    cx, cy, cz = center
    sx, sy, sz = size
    hx, hy, hz = sx * 0.5, sy * 0.5, sz * 0.5
    tc = top_col or col
    _emit_quad(out, (cx-hx, cy+hy, cz-hz), (cx+hx, cy+hy, cz-hz), (cx+hx, cy+hy, cz+hz), (cx-hx, cy+hy, cz+hz), (0, 1, 0), col, pbr)
    _emit_quad(out, (cx+hx, cy-hy, cz-hz), (cx-hx, cy-hy, cz-hz), (cx-hx, cy-hy, cz+hz), (cx+hx, cy-hy, cz+hz), (0, -1, 0), col, pbr)
    _emit_quad(out, (cx+hx, cy+hy, cz-hz), (cx+hx, cy-hy, cz-hz), (cx+hx, cy-hy, cz+hz), (cx+hx, cy+hy, cz+hz), (1, 0, 0), col, pbr)
    _emit_quad(out, (cx-hx, cy-hy, cz-hz), (cx-hx, cy+hy, cz-hz), (cx-hx, cy+hy, cz+hz), (cx-hx, cy-hy, cz+hz), (-1, 0, 0), col, pbr)
    _emit_quad(out, (cx-hx, cy+hy, cz+hz), (cx+hx, cy+hy, cz+hz), (cx+hx, cy-hy, cz+hz), (cx-hx, cy-hy, cz+hz), (0, 0, 1), tc, pbr)
    _emit_quad(out, (cx-hx, cy-hy, cz-hz), (cx+hx, cy-hy, cz-hz), (cx+hx, cy+hy, cz-hz), (cx-hx, cy+hy, cz-hz), (0, 0, -1), col, pbr)


# ==============================================================================
#  UNIT BLOCK BUILDERS (each sits cleanly on z in [0, 1], x,y in [-0.5, +0.5])
# ==============================================================================

def _build_grass_block(out, ox=0.0, oy=0.0, oz=0.0, s=1.0):
    GRASS_A = (0.32, 0.62, 0.21)
    GRASS_B = (0.26, 0.54, 0.17)
    DIRT_A  = (0.48, 0.34, 0.22)
    DIRT_B  = (0.38, 0.26, 0.17)
    PEBBLE  = (0.50, 0.50, 0.50)
    EDGE = [2, 3, 2, 1, 2, 3, 2, 2]
    def col(f, u, v):
        if f == "+Z":
            return _jitter(GRASS_A if (u + v) % 2 == 0 else GRASS_B, u, v, 0.045)
        if f == "-Z":
            return _jitter(DIRT_A, u, v, 0.04)
        if v < EDGE[u % 8]:
            return _jitter(GRASS_A if v == 0 else GRASS_B, u, v, 0.04)
        if (u * 7 + v * 13) % 19 == 0:
            return PEBBLE
        return _jitter(DIRT_A if (u + v) % 2 == 0 else DIRT_B, u, v, 0.045)
    def pbr(f, u, v):
        return (0.86, 0.0, 0.96)
    _add_pixel_box(out, ox - 0.5*s, ox + 0.5*s, oy - 0.5*s, oy + 0.5*s, oz, oz + s, 8, 8, 8, col, pbr)


def _build_ore_block(out, gem_col, ox=0.0, oy=0.0, oz=0.0, s=1.0, metal=0.15):
    STONE_A = (0.46, 0.47, 0.48)
    STONE_B = (0.37, 0.38, 0.39)
    GEMS = {(1, 2), (2, 2), (2, 3), (5, 1), (6, 1), (5, 4), (6, 5), (2, 5), (3, 6), (4, 3)}
    def col(f, u, v):
        if (u, v) in GEMS:
            return _jitter(gem_col, u, v, 0.05)
        return _jitter(STONE_A if (u * 3 + v * 5) % 4 != 0 else STONE_B, u, v, 0.045)
    def pbr(f, u, v):
        if (u, v) in GEMS:
            return (0.16, metal, 1.0)
        return (0.88, 0.0, 0.94)
    _add_pixel_box(out, ox - 0.5*s, ox + 0.5*s, oy - 0.5*s, oy + 0.5*s, oz, oz + s, 8, 8, 8, col, pbr)


def _build_stone_brick(out, ox=0.0, oy=0.0, oz=0.0, s=1.0, mossy=False):
    ST_A = (0.48, 0.49, 0.50)
    ST_B = (0.40, 0.41, 0.42)
    MORT = (0.24, 0.25, 0.26)
    MOSS = (0.26, 0.44, 0.19)
    def col(f, u, v):
        if v in (0, 4) or (v < 4 and u == 0) or (v >= 4 and u == 4):
            return MOSS if (mossy and (u + v) % 2 == 0) else MORT
        if mossy and (u * 5 + v * 7) % 11 == 0:
            return _jitter(MOSS, u, v, 0.04)
        return _jitter(ST_A if (u + v) % 3 != 0 else ST_B, u, v, 0.035)
    _add_pixel_box(out, ox - 0.5*s, ox + 0.5*s, oy - 0.5*s, oy + 0.5*s, oz, oz + s, 8, 8, 8, col, lambda f,u,v: (0.88, 0.0, 0.94))


def _build_oak_planks(out, ox=0.0, oy=0.0, oz=0.0, s=1.0):
    WD_A = (0.66, 0.51, 0.30)
    WD_B = (0.56, 0.42, 0.23)
    SEAM = (0.36, 0.25, 0.13)
    def col(f, u, v):
        if v in (0, 2, 4, 6):
            return SEAM
        if (v == 1 and u == 3) or (v == 5 and u == 6):
            return SEAM
        return _jitter(WD_A if (u + v) % 2 == 0 else WD_B, u, v, 0.035)
    _add_pixel_box(out, ox - 0.5*s, ox + 0.5*s, oy - 0.5*s, oy + 0.5*s, oz, oz + s, 8, 8, 8, col, lambda f,u,v: (0.78, 0.0, 0.96))


def _build_crafting_table(out, ox=0.0, oy=0.0, oz=0.0, s=1.0):
    TOP_W = (0.64, 0.44, 0.24)
    GRID  = (0.28, 0.16, 0.08)
    WD_A  = (0.62, 0.48, 0.28)
    WD_D  = (0.42, 0.28, 0.15)
    IRON  = (0.55, 0.57, 0.60)
    def col(f, u, v):
        if f == "+Z":
            if u in (0, 7) or v in (0, 7): return WD_D
            if u in (2, 5) or v in (2, 5): return GRID
            return _jitter(TOP_W, u, v, 0.03)
        if f == "-Z":
            return _jitter(WD_A, u, v, 0.03)
        if v == 0: return WD_D
        if u in (0, 7) or v == 7: return WD_D
        # Tool silhouettes on sides (saw & hammer!)
        if (u, v) in {(2, 2), (2, 3), (2, 4), (3, 3), (5, 2), (5, 3), (6, 2)}:
            return IRON
        if (u, v) in {(2, 5), (5, 4), (5, 5)}:
            return GRID
        return _jitter(WD_A, u, v, 0.035)
    def pbr(f, u, v):
        if f not in ("+Z", "-Z") and (u, v) in {(2, 2), (2, 3), (2, 4), (3, 3), (5, 2), (5, 3), (6, 2)}:
            return (0.30, 0.85, 1.0)
        return (0.78, 0.0, 0.96)
    _add_pixel_box(out, ox - 0.5*s, ox + 0.5*s, oy - 0.5*s, oy + 0.5*s, oz, oz + s, 8, 8, 8, col, pbr)


def _build_chest(out, ox=0.0, oy=0.0, oz=0.0, s=1.0):
    CH_W = (0.68, 0.47, 0.18)
    CH_D = (0.44, 0.28, 0.10)
    RIM  = (0.16, 0.13, 0.10)
    w = 0.88 * s
    h = 0.88 * s
    def col(f, u, v):
        if u in (0, 7) or v in (0, 7) or (f not in ("+Z", "-Z") and v == 3):
            return RIM
        return _jitter(CH_W if (u + v) % 2 == 0 else CH_D, u, v, 0.03)
    _add_pixel_box(out, ox - 0.5*w, ox + 0.5*w, oy - 0.5*w, oy + 0.5*w, oz, oz + h, 8, 8, 8, col, lambda f,u,v: (0.72, 0.0, 0.96))
    # 3D raised metal chest latch buckle on +Y front!
    _add_solid_box(out, (ox, oy + 0.5*w + 0.035*s, oz + 0.56*h), (0.14*s, 0.07*s, 0.22*s),
                   (0.76, 0.77, 0.79), pbr=(0.24, 0.85, 1.0))


def _build_tnt_block(out, ox=0.0, oy=0.0, oz=0.0, s=1.0):
    RED_A = (0.78, 0.16, 0.12)
    RED_B = (0.64, 0.12, 0.09)
    WHT   = (0.90, 0.89, 0.86)
    BLK   = (0.10, 0.10, 0.11)
    # 8x8 band letters pattern on rows v=3,4
    TNT_PX = {(1,3),(1,4),(3,3),(4,4),(6,3),(6,4),(0,3),(2,3),(5,3),(7,3)}
    def col(f, u, v):
        if f in ("+Z", "-Z"):
            if u in (0, 7) or v in (0, 7): return RED_B
            if 2 <= u <= 5 and 2 <= v <= 5: return BLK if (u + v) % 2 == 0 else (0.36, 0.34, 0.32)
            return _jitter(RED_A, u, v, 0.03)
        if v in (3, 4):
            return BLK if (u, v) in TNT_PX else _jitter(WHT, u, v, 0.015)
        return _jitter(RED_A if u % 2 == 0 else RED_B, u, v, 0.03)
    _add_pixel_box(out, ox - 0.5*s, ox + 0.5*s, oy - 0.5*s, oy + 0.5*s, oz, oz + s, 8, 8, 8, col, lambda f,u,v: (0.76, 0.0, 0.96))


def _build_furnace(out, ox=0.0, oy=0.0, oz=0.0, s=1.0):
    ST_A = (0.44, 0.45, 0.46)
    ST_B = (0.32, 0.33, 0.34)
    DARK = (0.10, 0.10, 0.11)
    FIRE1 = (1.00, 0.52, 0.08)
    FIRE2 = (1.00, 0.86, 0.22)
    def col(f, u, v):
        if f == "+Y":
            # Upper oven slot
            if v in (1, 2) and 1 <= u <= 6: return DARK
            # Glowing lower firebox!
            if v in (5, 6) and 2 <= u <= 5:
                return FIRE2 if (v == 6 and u in (3, 4)) else FIRE1
            if v == 4 and 1 <= u <= 6: return ST_B
        return _jitter(ST_A if (u + v) % 3 != 0 else ST_B, u, v, 0.04)
    def pbr(f, u, v):
        if f == "+Y" and v in (5, 6) and 2 <= u <= 5:
            return (0.20, 0.0, -2.6)  # Self-emissive glowing furnace fire!
        return (0.88, 0.0, 0.94)
    _add_pixel_box(out, ox - 0.5*s, ox + 0.5*s, oy - 0.5*s, oy + 0.5*s, oz, oz + s, 8, 8, 8, col, pbr)


def _build_bookshelf(out, ox=0.0, oy=0.0, oz=0.0, s=1.0):
    WD = (0.64, 0.49, 0.28)
    WD_D = (0.34, 0.23, 0.12)
    BOOKS = [(0.74, 0.18, 0.16), (0.18, 0.44, 0.72), (0.24, 0.58, 0.28), (0.82, 0.72, 0.38), (0.58, 0.28, 0.56), (0.78, 0.74, 0.66)]
    def col(f, u, v):
        if f in ("+Z", "-Z"): return _jitter(WD, u, v, 0.03)
        if v in (0, 3, 4, 7): return WD if v in (0, 7) else WD_D
        return _jitter(BOOKS[(u * 3 + v * 5) % len(BOOKS)], u, v, 0.03)
    _add_pixel_box(out, ox - 0.5*s, ox + 0.5*s, oy - 0.5*s, oy + 0.5*s, oz, oz + s, 8, 8, 8, col, lambda f,u,v: (0.78, 0.0, 0.96))


def _build_enchanting_table(out, ox=0.0, oy=0.0, oz=0.0, s=1.0):
    OBS_A = (0.11, 0.08, 0.17)
    OBS_B = (0.18, 0.12, 0.27)
    CRIM  = (0.68, 0.10, 0.14)
    DIA   = (0.16, 0.88, 0.86)
    h = 0.75 * s
    def col(f, u, v):
        if f == "+Z":
            if (u in (0, 7) and v in (0, 7)): return DIA
            return _jitter(CRIM, u, v, 0.025)
        if v == 0:
            return DIA if u in (0, 7) else CRIM
        if v == 1 and u in (1, 2, 5, 6): return CRIM
        return _jitter(OBS_A if (u + v) % 2 == 0 else OBS_B, u, v, 0.03)
    def pbr(f, u, v):
        if (f == "+Z" and u in (0, 7) and v in (0, 7)) or (v == 0 and u in (0, 7)):
            return (0.12, 0.1, 1.0)
        return (0.45, 0.05, 0.94)
    _add_pixel_box(out, ox - 0.5*s, ox + 0.5*s, oy - 0.5*s, oy + 0.5*s, oz, oz + h, 8, 8, 6, col, pbr)
    # Floating golden-trimmed crimson Enchanted Book hovering above the table!
    _add_solid_box(out, (ox, oy, oz + h + 0.24*s), (0.34*s, 0.26*s, 0.08*s), (0.62, 0.14, 0.18), pbr=(0.45, 0.2, 1.0), top_col=(0.92, 0.88, 0.74))
    _add_solid_box(out, (ox, oy, oz + h + 0.24*s), (0.36*s, 0.06*s, 0.09*s), (0.94, 0.76, 0.18), pbr=(0.22, 0.9, 1.0))


def _build_obsidian(out, ox=0.0, oy=0.0, oz=0.0, s=1.0):
    OBS_A = (0.10, 0.07, 0.16)
    OBS_B = (0.19, 0.12, 0.30)
    OBS_C = (0.28, 0.17, 0.44)
    def col(f, u, v):
        h = (u * 7 + v * 11) % 13
        if h == 0: return OBS_C
        return _jitter(OBS_A if h % 2 == 0 else OBS_B, u, v, 0.025)
    _add_pixel_box(out, ox - 0.5*s, ox + 0.5*s, oy - 0.5*s, oy + 0.5*s, oz, oz + s, 6, 6, 6, col, lambda f,u,v: (0.35, 0.10, 0.94))


def _build_torch(out, ox=0.0, oy=0.0, oz=0.0, s=1.0, tilt_y=0.0):
    """Minecraft Torch (tilt_y > 0 tilts forward away from wall for wall-mounted sconces!)."""
    w = 0.14 * s
    h_stick = 0.58 * s
    h_flame = 0.18 * s
    # Build upright around (0,0,0) then tilt & translate!
    tmp = _new_buf()
    _add_solid_box(tmp, (0.0, 0.0, h_stick * 0.5), (w, w, h_stick), (0.48, 0.34, 0.18), pbr=(0.82, 0.0, 0.96))
    _add_solid_box(tmp, (0.0, 0.0, h_stick + h_flame * 0.35), (w * 1.15, w * 1.15, h_flame * 0.7), (1.0, 0.52, 0.06), pbr=(0.2, 0.0, -3.2))
    _add_solid_box(tmp, (0.0, 0.0, h_stick + h_flame * 0.85), (w * 0.85, w * 0.85, h_flame * 0.6), (1.0, 0.94, 0.38), pbr=(0.2, 0.0, -4.0))
    c, sn = math.cos(tilt_y), math.sin(tilt_y)
    # Rotate around X axis (tilts +Z toward +Y or -Y!)
    base = len(out["pos"])
    for p, n, uv, col, pbr in zip(tmp["pos"], tmp["nrm"], tmp["uv"], tmp["col"], tmp["pbr"]):
        px, py, pz = p
        nx, ny, nz = n
        ry = py * c - pz * sn
        rz = py * sn + pz * c
        nry = ny * c - nz * sn
        nrz = ny * sn + nz * c
        out["pos"].append((px + ox, ry + oy, rz + oz))
        out["nrm"].append((nx, nry, nrz))
        out["uv"].append(uv)
        out["col"].append(col)
        out["pbr"].append(pbr)
    out["idx"].extend([base + ii for ii in tmp["idx"]])


def _build_lantern(out, ox=0.0, oy=0.0, oz=0.0, s=1.0, chain_h=0.0):
    IRON = (0.24, 0.26, 0.29)
    GLOW = (1.00, 0.76, 0.28)
    CORE = (1.00, 0.94, 0.62)
    w = 0.42 * s
    h = 0.52 * s
    _add_solid_box(out, (ox, oy, oz + 0.05*s), (w, w, 0.10*s), IRON, pbr=(0.35, 0.85, 0.95))
    _add_solid_box(out, (ox, oy, oz + 0.30*s), (w * 0.82, w * 0.82, 0.40*s), GLOW, pbr=(0.2, 0.0, -2.8), top_col=CORE)
    # 4 corner iron cage bars
    for sx in (-1, 1):
        for sy in (-1, 1):
            _add_solid_box(out, (ox + sx*w*0.44, oy + sy*w*0.44, oz + 0.30*s), (0.06*s, 0.06*s, 0.42*s), IRON, pbr=(0.35, 0.85, 0.95))
    _add_solid_box(out, (ox, oy, oz + h), (w * 1.05, w * 1.05, 0.09*s), IRON, pbr=(0.35, 0.85, 0.95))
    _add_solid_box(out, (ox, oy, oz + h + 0.08*s), (w * 0.52, w * 0.52, 0.10*s), IRON, pbr=(0.35, 0.85, 0.95))
    if chain_h > 0.05:
        n_links = max(1, int(chain_h / (0.22 * s)))
        for k in range(n_links):
            cz = oz + h + 0.16*s + k * (chain_h / n_links)
            sz_x = 0.11*s if k % 2 == 0 else 0.04*s
            sz_y = 0.04*s if k % 2 == 0 else 0.11*s
            _add_solid_box(out, (ox, oy, cz), (sz_x, sz_y, chain_h / n_links * 0.92), IRON, pbr=(0.32, 0.88, 0.95))


def _build_oak_tree(out, ox=0.0, oy=0.0, oz=0.0, s=1.0):
    """Authentic Blocky Minecraft Oak Tree (4-block oak trunk + classic multi-tier voxel leaf canopy!)."""
    BARK_A = (0.39, 0.29, 0.17)
    BARK_B = (0.30, 0.22, 0.12)
    LEAF_A = (0.20, 0.43, 0.13)
    LEAF_B = (0.15, 0.34, 0.09)
    LEAF_C = (0.26, 0.50, 0.16)
    # 4-block trunk (z = 0..4*s)
    for k in range(4):
        def bcol(f, u, v, kk=k):
            return _jitter(BARK_A if u % 2 == 0 else BARK_B, u, v + kk * 3, 0.03)
        _add_pixel_box(out, ox - 0.5*s, ox + 0.5*s, oy - 0.5*s, oy + 0.5*s, oz + k*s, oz + (k+1)*s, 4, 4, 4, bcol, lambda f,u,v: (0.90, 0.0, 0.94))
    # Classic Minecraft leaf canopy: layers z=3,4 (5x5 missing corners), z=5 (3x3), z=6 (cross + center)!
    leaf_coords = []
    for lz in (3, 4):
        for lx in range(-2, 3):
            for ly in range(-2, 3):
                if abs(lx) == 2 and abs(ly) == 2 and (lz == 4 or (lx + ly) % 4 == 0):
                    continue
                if lx == 0 and ly == 0 and lz == 3:
                    continue
                leaf_coords.append((lx, ly, lz))
    for lx in range(-1, 2):
        for ly in range(-1, 2):
            leaf_coords.append((lx, ly, 5))
    for lx, ly in ((0, 0), (-1, 0), (1, 0), (0, -1), (0, 1)):
        leaf_coords.append((lx, ly, 6))
    for lx, ly, lz in leaf_coords:
        c_base = LEAF_C if lz >= 5 else (LEAF_A if (lx + ly + lz) % 2 == 0 else LEAF_B)
        c_jit = _jitter(c_base, lx * 7 + lz, ly * 11, 0.05)
        _add_solid_box(out, (ox + lx*s, oy + ly*s, oz + (lz + 0.5)*s), (s * 0.995, s * 0.995, s * 0.995),
                       c_jit, pbr=(0.88, 0.0, 0.95 if lz >= 5 else 0.86), top_col=_jitter(LEAF_C, lx, ly, 0.04))


def _build_nether_portal(out, ox=0.0, oy=0.0, oz=0.0, s=1.0):
    """Iconic 4x5 Minecraft Obsidian Nether Portal Frame + Swirling Glowing Purple Portal Plane!"""
    # Bottom row (x=-1.5..+1.5, z=0), top row (z=4), left/right pillars (z=1..3)
    for ix in (-1.5, -0.5, 0.5, 1.5):
        _build_obsidian(out, ox + ix*s, oy, oz + 0.0*s, s)
        _build_obsidian(out, ox + ix*s, oy, oz + 4.0*s, s)
    for iz in (1.0, 2.0, 3.0):
        _build_obsidian(out, ox - 1.5*s, oy, oz + iz*s, s)
        _build_obsidian(out, ox + 1.5*s, oy, oz + iz*s, s)
    # Glowing swirling purple Nether Portal interior (subdivided 8x12 pixel grid!)
    P1 = (0.64, 0.14, 0.98)
    P2 = (0.44, 0.06, 0.82)
    P3 = (0.86, 0.45, 1.00)
    def pcol(f, u, v):
        r = math.hypot((u - 3.5) / 3.5, (v - 5.5) / 5.5)
        sw = math.sin(r * 7.5 + math.atan2(v - 5.5, u - 3.5) * 2.0)
        if sw > 0.45: return P3
        return P1 if sw > -0.2 else P2
    _add_pixel_box(out, ox - 1.0*s, ox + 1.0*s, oy - 0.14*s, oy + 0.14*s, oz + 1.0*s, oz + 4.0*s,
                   8, 1, 12, pcol, lambda f,u,v: (0.15, 0.0, -2.2))


def _build_beacon_pyramid(out, ox=0.0, oy=0.0, oz=0.0, s=1.0, beam_h=12.0):
    """3x3 Diamond-Gold Pyramid + Glowing Minecraft Beacon Block + Tall Cyan-White Sky Beam!"""
    GOLD = (0.94, 0.76, 0.18)
    DIA  = (0.20, 0.88, 0.86)
    # 3x3 base tier (gold & diamond blocks)
    for ix in (-1, 0, 1):
        for iy in (-1, 0, 1):
            c = DIA if (abs(ix) + abs(iy)) % 2 == 0 else GOLD
            _add_solid_box(out, (ox + ix*s, oy + iy*s, oz + 0.5*s), (s*0.99, s*0.99, s*0.99), c, pbr=(0.22, 0.85, 0.98))
    # Beacon housing on top (obsidian base + glowing cyan nether-star core)
    _add_solid_box(out, (ox, oy, oz + 1.12*s), (0.85*s, 0.85*s, 0.24*s), (0.12, 0.08, 0.19), pbr=(0.35, 0.1, 0.95))
    _add_solid_box(out, (ox, oy, oz + 1.58*s), (0.68*s, 0.68*s, 0.68*s), (0.52, 0.96, 0.98), pbr=(0.1, 0.0, -3.0))
    # Tall glowing cyan-white Minecraft Beacon sky beam!
    _add_solid_box(out, (ox, oy, oz + 1.9*s + beam_h*0.5), (0.34*s, 0.34*s, beam_h), (0.72, 0.98, 1.0), pbr=(0.1, 0.0, -3.5))


def _build_village_cottage(out, ox=0.0, oy=0.0, oz=0.0, s=1.0):
    """Classic Minecraft Village Oak & Cobblestone House (5x5 footprint, pitched oak roof, warm lit windows)."""
    # Cobblestone foundation + corner pillars, oak plank walls, glowing windows
    for ix in range(-2, 3):
        for iy in range(-2, 3):
            if abs(ix) == 2 or abs(iy) == 2:
                _build_stone_brick(out, ox + ix*s, oy + iy*s, oz + 0.0*s, s)
                for iz in (1, 2):
                    is_corner = abs(ix) == 2 and abs(iy) == 2
                    is_door = (iy == -2 and ix == 0 and iz in (1, 2))
                    is_win  = ((abs(ix) == 2 and iy == 0) or (iy == 2 and ix == 0)) and iz == 2
                    if is_door:
                        _add_solid_box(out, (ox + ix*s, oy + iy*s, oz + (iz+0.5)*s), (s*0.92, s*0.30, s*0.99), (0.46, 0.31, 0.16), pbr=(0.78, 0.0, 0.94))
                    elif is_win:
                        _add_solid_box(out, (ox + ix*s, oy + iy*s, oz + (iz+0.5)*s), (s*0.92, s*0.92, s*0.92), (1.0, 0.84, 0.42), pbr=(0.15, 0.0, -2.4))
                    elif is_corner:
                        _add_solid_box(out, (ox + ix*s, oy + iy*s, oz + (iz+0.5)*s), (s*0.99, s*0.99, s*0.99), (0.38, 0.28, 0.16), pbr=(0.88, 0.0, 0.94))
                    else:
                        _build_oak_planks(out, ox + ix*s, oy + iy*s, oz + iz*s, s)
    # Stepped oak stair pitched roof (tiers z=3, 4, 5)
    ROOF_A = (0.58, 0.41, 0.22)
    ROOF_B = (0.48, 0.33, 0.17)
    for tier, rx in ((3, 3), (4, 2), (5, 1), (6, 0)):
        for ix in range(-rx, rx + 1):
            for iy in range(-2, 3):
                c = ROOF_A if (ix + iy + tier) % 2 == 0 else ROOF_B
                _add_solid_box(out, (ox + ix*s, oy + iy*s, oz + (tier + 0.4)*s), (s*0.99, s*0.99, s*0.85), c, pbr=(0.80, 0.0, 0.95))


def build_all_world_assets():
    """Build reusable dictionary of all Minecraft world-projection props & architectural structures."""
    specs = {
        "grass_block":      lambda b: _build_grass_block(b),
        "stone_brick":      lambda b: _build_stone_brick(b, mossy=False),
        "mossy_brick":      lambda b: _build_stone_brick(b, mossy=True),
        "oak_planks":       lambda b: _build_oak_planks(b),
        "diamond_ore":      lambda b: _build_ore_block(b, (0.14, 0.88, 0.88), metal=0.10),
        "gold_ore":         lambda b: _build_ore_block(b, (0.96, 0.78, 0.18), metal=0.88),
        "emerald_ore":      lambda b: _build_ore_block(b, (0.12, 0.84, 0.36), metal=0.10),
        "crafting_table":   lambda b: _build_crafting_table(b),
        "chest":            lambda b: _build_chest(b),
        "tnt_block":        lambda b: _build_tnt_block(b),
        "furnace":          lambda b: _build_furnace(b),
        "bookshelf":        lambda b: _build_bookshelf(b),
        "enchanting_table": lambda b: _build_enchanting_table(b),
        "obsidian":         lambda b: _build_obsidian(b),
        "torch":            lambda b: _build_torch(b, tilt_y=0.0),
        "wall_torch":       lambda b: _build_torch(b, tilt_y=0.45),
        "lantern":          lambda b: _build_lantern(b, chain_h=0.0),
        "hanging_lantern":  lambda b: _build_lantern(b, chain_h=0.95),
        "oak_tree":         lambda b: _build_oak_tree(b),
        "nether_portal":    lambda b: _build_nether_portal(b),
        "beacon_pyramid":   lambda b: _build_beacon_pyramid(b, beam_h=14.0),
        "village_cottage":  lambda b: _build_village_cottage(b),
    }
    out = {}
    for k, fn in specs.items():
        buf = _new_buf()
        fn(buf)
        out[k] = _finalize(buf)
    return out
