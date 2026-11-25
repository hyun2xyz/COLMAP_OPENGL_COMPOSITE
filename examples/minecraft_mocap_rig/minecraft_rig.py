"""Authentic 3D Beveled Pixel-Art Minecraft Character (Steve) + Hierarchical 3D Mocap Retargeting.

Builds 11 articulated rigid-body meshes with 2-layer 3D pixel relief (head + 3D hair/eyes, chest + collar,
pelvis + belt, upper/lower arms with chamfered elbow bridges, upper/lower legs with knee bridges & 3D sneaker trim)
and solves smooth, ground-locked, twist-free hierarchical FK from MediaPipe 3D world landmarks P[N, 33, 3].

Local Character / Bone Convention (right-handed, det = +1):
  +X = character anatomical RIGHT
  +Y = character FRONT (face / chest / toes)
  +Z = character UP (limb meshes extend from joint pivot 0 downward along -Z!)
"""
import math
import numpy as np
from scipy.signal import savgol_filter
from scipy.spatial.transform import Rotation as Rot

# Classic Minecraft pixel unit: 32 pixels = 1.68m full character height
PX = 1.68 / 32.0  # 0.0525m per pixel


# ==============================================================================
#  BEVELED 3D PIXEL-ART MESH BUILDER (14 floats/vert: pos3, nrm3, uv2, col3, pbr3)
# ==============================================================================

def _add_beveled_box(out, center, size, face_cols, pbr=(0.75, 0.0, 1.0), bevel_frac=0.08):
    """Add a solid box with subtle beveled edges so sunlight catches crisp vinyl/voxel highlights."""
    cx, cy, cz = center
    sx, sy, sz = size
    hx, hy, hz = sx * 0.5, sy * 0.5, sz * 0.5
    b = min(hx, hy, hz) * bevel_frac

    # 6 main flat faces with inset corners for micro-bevel + smooth-shaded bevel strips
    # Face order: +Y (front), -Y (back), +X (right), -X (left), +Z (top), -Z (bottom)
    def push_quad(p0, p1, p2, p3, nrm, col, ao_scale=1.0):
        base = len(out["pos"])
        n = np.asarray(nrm, dtype=np.float32)
        n /= (np.linalg.norm(n) + 1e-8)
        r, m, ao = pbr
        ao_v = float(np.clip(ao * ao_scale, 0.15, 1.0))
        uvs = [(0.0, 0.0), (1.0, 0.0), (1.0, 1.0), (0.0, 1.0)]
        for p, uv in zip((p0, p1, p2, p3), uvs):
            out["pos"].append(p)
            out["nrm"].append(tuple(n))
            out["uv"].append(uv)
            out["col"].append(col)
            out["pbr"].append((r, m, ao_v))
        out["idx"].extend([base, base + 1, base + 2, base, base + 2, base + 3])

    ix, iy, iz = hx - b, hy - b, hz - b
    # +Y front
    c_f = face_cols.get("+Y", face_cols.get("all"))
    push_quad((cx-ix, cy+hy, cz-iz), (cx+ix, cy+hy, cz-iz), (cx+ix, cy+hy, cz+iz), (cx-ix, cy+hy, cz+iz), (0, 1, 0), c_f)
    # -Y back
    c_b = face_cols.get("-Y", face_cols.get("all"))
    push_quad((cx+ix, cy-hy, cz-iz), (cx-ix, cy-hy, cz-iz), (cx-ix, cy-hy, cz+iz), (cx+ix, cy-hy, cz+iz), (0, -1, 0), c_b)
    # +X right
    c_r = face_cols.get("+X", face_cols.get("all"))
    push_quad((cx+hx, cy+iy, cz-iz), (cx+hx, cy-iy, cz-iz), (cx+hx, cy-iy, cz+iz), (cx+hx, cy+iy, cz+iz), (1, 0, 0), c_r)
    # -X left
    c_l = face_cols.get("-X", face_cols.get("all"))
    push_quad((cx-hx, cy-iy, cz-iz), (cx-hx, cy+iy, cz-iz), (cx-hx, cy+iy, cz+iz), (cx-hx, cy-iy, cz+iz), (-1, 0, 0), c_l)
    # +Z top
    c_t = face_cols.get("+Z", face_cols.get("all"))
    push_quad((cx-ix, cy+iy, cz+hz), (cx+ix, cy+iy, cz+hz), (cx+ix, cy-iy, cz+hz), (cx-ix, cy-iy, cz+hz), (0, 0, 1), c_t)
    # -Z bottom
    c_d = face_cols.get("-Z", face_cols.get("all"))
    push_quad((cx-ix, cy-iy, cz-hz), (cx+ix, cy-iy, cz-hz), (cx+ix, cy+iy, cz-hz), (cx-ix, cy+iy, cz-hz), (0, 0, -1), c_d)

    if b > 1e-5:
        # 12 crisp chamfer edge strips (give real physical edge catchlights in daylight!)
        def edge_col(a, b_c):
            return (0.5*(a[0]+b_c[0]), 0.5*(a[1]+b_c[1]), 0.5*(a[2]+b_c[2]))
        # 4 vertical edges
        push_quad((cx+ix, cy+hy, cz-iz), (cx+hx, cy+iy, cz-iz), (cx+hx, cy+iy, cz+iz), (cx+ix, cy+hy, cz+iz), (1, 1, 0), edge_col(c_f, c_r), 0.96)
        push_quad((cx-hx, cy+iy, cz-iz), (cx-ix, cy+hy, cz-iz), (cx-ix, cy+hy, cz+iz), (cx-hx, cy+iy, cz+iz), (-1, 1, 0), edge_col(c_f, c_l), 0.96)
        push_quad((cx+hx, cy-iy, cz-iz), (cx+ix, cy-hy, cz-iz), (cx+ix, cy-hy, cz+iz), (cx+hx, cy-iy, cz+iz), (1, -1, 0), edge_col(c_b, c_r), 0.96)
        push_quad((cx-ix, cy-hy, cz-iz), (cx-hx, cy-iy, cz-iz), (cx-hx, cy-iy, cz+iz), (cx-ix, cy-hy, cz+iz), (-1, -1, 0), edge_col(c_b, c_l), 0.96)
        # 4 top edges
        push_quad((cx-ix, cy+hy, cz+iz), (cx+ix, cy+hy, cz+iz), (cx+ix, cy+iy, cz+hz), (cx-ix, cy+iy, cz+hz), (0, 1, 1), edge_col(c_f, c_t), 0.98)
        push_quad((cx+ix, cy-hy, cz+iz), (cx-ix, cy-hy, cz+iz), (cx-ix, cy-iy, cz+hz), (cx+ix, cy-iy, cz+hz), (0, -1, 1), edge_col(c_b, c_t), 0.98)
        push_quad((cx+hx, cy+iy, cz+iz), (cx+hx, cy-iy, cz+iz), (cx+ix, cy-iy, cz+hz), (cx+ix, cy+iy, cz+hz), (1, 0, 1), edge_col(c_r, c_t), 0.98)
        push_quad((cx-hx, cy-iy, cz+iz), (cx-hx, cy+iy, cz+iz), (cx-ix, cy+iy, cz+hz), (cx-ix, cy-iy, cz+hz), (-1, 0, 1), edge_col(c_l, c_t), 0.98)
        # 4 bottom edges
        push_quad((cx+ix, cy+hy, cz-iz), (cx-ix, cy+hy, cz-iz), (cx-ix, cy+iy, cz-hz), (cx+ix, cy+iy, cz-hz), (0, 1, -1), edge_col(c_f, c_d), 0.92)
        push_quad((cx-ix, cy-hy, cz-iz), (cx+ix, cy-hy, cz-iz), (cx+ix, cy-iy, cz-hz), (cx-ix, cy-iy, cz-hz), (0, -1, -1), edge_col(c_b, c_d), 0.92)
        push_quad((cx+hx, cy-iy, cz-iz), (cx+hx, cy+iy, cz-iz), (cx+ix, cy+iy, cz-hz), (cx+ix, cy-iy, cz-hz), (1, 0, -1), edge_col(c_r, c_d), 0.92)
        push_quad((cx-hx, cy+iy, cz-iz), (cx-hx, cy-iy, cz-iz), (cx-ix, cy-iy, cz-hz), (cx-ix, cy+iy, cz-hz), (-1, 0, -1), edge_col(c_l, c_d), 0.92)


def _add_pixel_grid_box(out, x0, x1, y0, y1, z0, z1, nx, ny, nz, color_fn, pbr_fn):
    """Build a solid Minecraft body block subdivided into crisp per-pixel surface tiles on all 6 faces!"""
    dx = (x1 - x0) / nx
    dy = (y1 - y0) / ny
    dz = (z1 - z0) / nz

    def emit_face(p0, p1, p2, p3, nrm, col, pbr):
        base = len(out["pos"])
        for p, uv in zip((p0, p1, p2, p3), ((0, 0), (1, 0), (1, 1), (0, 1))):
            out["pos"].append(p)
            out["nrm"].append(nrm)
            out["uv"].append(uv)
            out["col"].append(col)
            out["pbr"].append(pbr)
        out["idx"].extend([base, base + 1, base + 2, base, base + 2, base + 3])

    # +Y front (u=0..nx-1 from -X to +X, v=0..nz-1 from top +Z down to bottom -Z!)
    for ix in range(nx):
        xa, xb = x0 + ix * dx, x0 + (ix + 1) * dx
        for iz in range(nz):
            zb, zt = z1 - (iz + 1) * dz, z1 - iz * dz
            col = color_fn("+Y", ix, iz)
            pbr = pbr_fn("+Y", ix, iz)
            emit_face((xa, y1, zb), (xb, y1, zb), (xb, y1, zt), (xa, y1, zt), (0.0, 1.0, 0.0), col, pbr)
    # -Y back (u=0..nx-1 from +X to -X, v=0..nz-1 top to bottom)
    for ix in range(nx):
        xb, xa = x1 - ix * dx, x1 - (ix + 1) * dx
        for iz in range(nz):
            zb, zt = z1 - (iz + 1) * dz, z1 - iz * dz
            col = color_fn("-Y", ix, iz)
            pbr = pbr_fn("-Y", ix, iz)
            emit_face((xb, y0, zb), (xa, y0, zb), (xa, y0, zt), (xb, y0, zt), (0.0, -1.0, 0.0), col, pbr)
    # +X right (u=0..ny-1 from +Y to -Y, v=0..nz-1 top to bottom)
    for iy in range(ny):
        yb, ya = y1 - iy * dy, y1 - (iy + 1) * dy
        for iz in range(nz):
            zb, zt = z1 - (iz + 1) * dz, z1 - iz * dz
            col = color_fn("+X", iy, iz)
            pbr = pbr_fn("+X", iy, iz)
            emit_face((x1, yb, zb), (x1, ya, zb), (x1, ya, zt), (x1, yb, zt), (1.0, 0.0, 0.0), col, pbr)
    # -X left (u=0..ny-1 from -Y to +Y, v=0..nz-1 top to bottom)
    for iy in range(ny):
        ya, yb = y0 + iy * dy, y0 + (iy + 1) * dy
        for iz in range(nz):
            zb, zt = z1 - (iz + 1) * dz, z1 - iz * dz
            col = color_fn("-X", iy, iz)
            pbr = pbr_fn("-X", iy, iz)
            emit_face((x0, ya, zb), (x0, yb, zb), (x0, yb, zt), (x0, ya, zt), (-1.0, 0.0, 0.0), col, pbr)
    # +Z top (u=0..nx-1 from -X to +X, v=0..ny-1 from -Y to +Y)
    for ix in range(nx):
        xa, xb = x0 + ix * dx, x0 + (ix + 1) * dx
        for iy in range(ny):
            ya, yb = y0 + iy * dy, y0 + (iy + 1) * dy
            col = color_fn("+Z", ix, iy)
            pbr = pbr_fn("+Z", ix, iy)
            emit_face((xa, yb, z1), (xb, yb, z1), (xb, ya, z1), (xa, ya, z1), (0.0, 0.0, 1.0), col, pbr)
    # -Z bottom
    for ix in range(nx):
        xa, xb = x0 + ix * dx, x0 + (ix + 1) * dx
        for iy in range(ny):
            ya, yb = y0 + iy * dy, y0 + (iy + 1) * dy
            col = color_fn("-Z", ix, iy)
            pbr = pbr_fn("-Z", ix, iy)
            emit_face((xa, ya, z0), (xb, ya, z0), (xb, yb, z0), (xa, yb, z0), (0.0, 0.0, -1.0), col, pbr)


def _finalize(out):
    pos = np.array(out["pos"], dtype=np.float32)
    nrm = np.array(out["nrm"], dtype=np.float32)
    uv = np.array(out["uv"], dtype=np.float32)
    col = np.array(out["col"], dtype=np.float32)
    pbr = np.array(out["pbr"], dtype=np.float32)
    idx = np.array(out["idx"], dtype=np.uint32)
    return {"pos": pos, "nrm": nrm, "uv": uv, "col": col, "pbr": pbr, "idx": idx}


def _jitter(rgb, seed_i, seed_j, amt=0.032):
    h = math.sin(seed_i * 127.1 + seed_j * 311.7) * 43758.5453
    v = (h - math.floor(h) - 0.5) * 2.0 * amt
    return (float(np.clip(rgb[0] + v, 0.01, 0.99)),
            float(np.clip(rgb[1] + v, 0.01, 0.99)),
            float(np.clip(rgb[2] + v, 0.01, 0.99)))


# Classic Iconic Minecraft Steve Color Palette (sRGB 0..1)
SKIN_BASE  = (0.79, 0.58, 0.44)
SKIN_SHADE = (0.71, 0.50, 0.37)
SKIN_NOSE  = (0.66, 0.44, 0.32)
HAIR_DARK  = (0.20, 0.13, 0.08)
HAIR_MID   = (0.27, 0.17, 0.10)
BEARD_COL  = (0.31, 0.20, 0.12)
MOUTH_COL  = (0.50, 0.31, 0.23)
EYE_WHITE  = (0.95, 0.96, 0.97)
EYE_IRIS   = (0.25, 0.24, 0.62)
SHIRT_MAIN = (0.04, 0.65, 0.68)
SHIRT_DARK = (0.02, 0.53, 0.57)
SHIRT_LITE = (0.10, 0.74, 0.76)
JEANS_MAIN = (0.24, 0.24, 0.58)
JEANS_DARK = (0.18, 0.18, 0.47)
JEANS_CUFF = (0.20, 0.20, 0.50)
SHOE_TOP   = (0.34, 0.34, 0.36)
SHOE_SOLE  = (0.52, 0.52, 0.54)
BELT_COL   = (0.16, 0.12, 0.09)
BUCKLE_COL = (0.72, 0.68, 0.58)


def build_steve_head():
    """8x8x8 Steve Head (pivot at neck bottom z=0) + 3D raised hair bangs/crown voxel relief!"""
    out = {"pos": [], "nrm": [], "uv": [], "col": [], "pbr": [], "idx": []}
    w = 8 * PX
    # Iconic 8x8 front face grid (row 0 = top, row 7 = chin)
    # Codes: H=hair, h=hair_mid, S=skin, s=skin_shade, W=eye_white, E=eye_iris, N=nose, B=beard, M=mouth
    FRONT = [
        "HHHHHHHH",
        "HhhHHhhH",
        "HSSSSSSH",
        "SSSSSSSS",
        "SWESSEWS",
        "SSSNnSSS",
        "SSBMMBSS",
        "SSBBBBSS",
    ]
    def col_fn(face, u, v):
        if face == "+Y":
            ch = FRONT[v][u]
            if ch == "H": return _jitter(HAIR_DARK, u, v, 0.025)
            if ch == "h": return _jitter(HAIR_MID, u, v, 0.025)
            if ch == "W": return EYE_WHITE
            if ch == "E": return EYE_IRIS
            if ch in ("N", "n"): return _jitter(SKIN_NOSE, u, v, 0.015)
            if ch == "B": return _jitter(BEARD_COL, u, v, 0.020)
            if ch == "M": return MOUTH_COL
            return _jitter(SKIN_BASE if (u + v) % 2 == 0 else SKIN_SHADE, u, v, 0.022)
        elif face == "+Z":
            return _jitter(HAIR_DARK if (u + v) % 3 != 0 else HAIR_MID, u, v, 0.03)
        elif face == "-Y":
            if v <= 5:
                return _jitter(HAIR_DARK if (u + v) % 2 == 0 else HAIR_MID, u, v, 0.03)
            return _jitter(SKIN_SHADE, u, v, 0.02)
        elif face in ("+X", "-X"):
            # Hair wraps down back half & sideburns near front!
            is_hair = (v <= 1) or (v == 2 and (u >= 1 if face == "+X" else u <= 6)) or (v == 3 and (u >= 3 if face == "+X" else u <= 4)) or (v == 4 and (u >= 5 if face == "+X" else u <= 2))
            if is_hair:
                return _jitter(HAIR_DARK if (u + v) % 2 == 0 else HAIR_MID, u, v, 0.028)
            return _jitter(SKIN_BASE, u, v, 0.022)
        else:
            return SKIN_SHADE

    def pbr_fn(face, u, v):
        if face == "+Y" and FRONT[v][u] in ("W", "E"):
            return (0.14, 0.0, 1.0)  # subtle live eye catchlight!
        if face == "+Z" or (face == "+Y" and FRONT[v][u] in ("H", "h")):
            return (0.76, 0.0, 0.96)
        return (0.64, 0.0, 1.0)

    _add_pixel_grid_box(out, -w * 0.5, w * 0.5, -w * 0.5, w * 0.5, 0.0, w, 8, 8, 8, col_fn, pbr_fn)

    # Raised 3D second-layer voxel hair relief (+0.013m extrusion) for authentic CGI Minecraft depth!
    ex = 0.013
    # Top hair crown cap
    _add_beveled_box(out, (0.0, 0.0, w + ex * 0.5), (w + ex * 1.4, w + ex * 1.4, ex),
                     {"all": HAIR_DARK, "+Z": HAIR_MID}, pbr=(0.76, 0.0, 0.98), bevel_frac=0.22)
    # Front bangs relief strip across top 2 rows + sideburn tabs
    _add_beveled_box(out, (0.0, w * 0.5 + ex * 0.45, w - PX), (w + ex * 1.2, ex, 2 * PX),
                     {"all": HAIR_DARK, "+Y": HAIR_MID}, pbr=(0.74, 0.0, 0.98), bevel_frac=0.18)
    for sgn in (-1.0, 1.0):
        _add_beveled_box(out, (sgn * (3.5 * PX + ex * 0.2), w * 0.5 + ex * 0.45, w - 2.5 * PX), (PX + ex * 0.5, ex, PX),
                         {"all": HAIR_DARK}, pbr=(0.74, 0.0, 0.96), bevel_frac=0.18)
    # Back hair thick layer
    _add_beveled_box(out, (0.0, -w * 0.5 - ex * 0.45, w - 2.5 * PX), (w + ex * 1.2, ex, 5 * PX),
                     {"all": HAIR_DARK}, pbr=(0.76, 0.0, 0.96), bevel_frac=0.18)
    # Subtle 3D raised nose bridge (+0.008m) so side profiles read cleanly!
    _add_beveled_box(out, (0.0, w * 0.5 + 0.004, 2.5 * PX), (2 * PX, 0.008, PX),
                     {"all": SKIN_NOSE}, pbr=(0.62, 0.0, 0.98), bevel_frac=0.20)
    return _finalize(out)


def build_steve_chest():
    """8u wide x 8u tall x 4u deep Upper Torso (pivot at waist z=0 -> extends z in [0, 8u])."""
    out = {"pos": [], "nrm": [], "uv": [], "col": [], "pbr": [], "idx": []}
    wx, wy, wz = 8 * PX, 4 * PX, 8 * PX

    def col_fn(face, u, v):
        if face == "+Y":
            # Iconic Steve V-neck skin collar at top center!
            if (v == 0 and 2 <= u <= 5) or (v == 1 and 3 <= u <= 4):
                return _jitter(SKIN_SHADE if v == 1 else SKIN_BASE, u, v, 0.02)
            # Subtle collar edge shadow
            if (v == 1 and u in (2, 5)) or (v == 2 and u in (3, 4)):
                return _jitter(SHIRT_DARK, u, v, 0.02)
            return _jitter(SHIRT_MAIN if (u + v) % 3 != 0 else SHIRT_LITE, u, v, 0.03)
        elif face == "+Z":
            if 2 <= u <= 5 and 1 <= v <= 2:
                return SKIN_SHADE
            return _jitter(SHIRT_LITE, u, v, 0.025)
        else:
            return _jitter(SHIRT_DARK if (u + v) % 4 == 0 else SHIRT_MAIN, u, v, 0.03)

    def pbr_fn(face, u, v):
        return (0.82, 0.0, 0.98)

    _add_pixel_grid_box(out, -wx * 0.5, wx * 0.5, -wy * 0.5, wy * 0.5, -0.4 * PX, wz, 8, 4, 8, col_fn, pbr_fn)
    # Subtle 3D shirt shoulder seams / collar trim relief
    _add_beveled_box(out, (0.0, 0.0, wz - 0.5 * PX), (wx + 0.010, wy + 0.010, PX),
                     {"all": SHIRT_MAIN, "+Z": SHIRT_LITE}, pbr=(0.80, 0.0, 0.98), bevel_frac=0.16)
    return _finalize(out)


def build_steve_pelvis():
    """8u wide x 4u tall x 4u deep Lower Torso / Waist (pivot at pelvis center z=0 -> z in [-1.5u, +2.8u])."""
    out = {"pos": [], "nrm": [], "uv": [], "col": [], "pbr": [], "idx": []}
    wx, wy = 8 * PX, 4 * PX
    z0, z1 = -1.5 * PX, 2.8 * PX

    def col_fn(face, u, v):
        # Top 2 rows = untucked cyan shirt hem (asymmetric classic Steve notch!), bottom 2 rows = indigo jeans!
        if face == "+Y":
            if v == 0 or (v == 1 and u >= 2) or (v == 2 and u >= 5):
                return _jitter(SHIRT_MAIN if v == 0 else SHIRT_DARK, u, v, 0.025)
            return _jitter(JEANS_MAIN if (u + v) % 2 == 0 else JEANS_DARK, u, v, 0.025)
        elif face in ("-Y", "+X", "-X"):
            if v <= 1:
                return _jitter(SHIRT_DARK if v == 1 else SHIRT_MAIN, u, v, 0.025)
            return _jitter(JEANS_DARK, u, v, 0.025)
        elif face == "+Z":
            return SHIRT_DARK
        return JEANS_DARK

    def pbr_fn(face, u, v):
        return (0.85, 0.0, 0.96)

    _add_pixel_grid_box(out, -wx * 0.5, wx * 0.5, -wy * 0.5, wy * 0.5, z0, z1, 8, 4, 4, col_fn, pbr_fn)
    # Subtle 3D untucked shirt hem rim
    _add_beveled_box(out, (0.0, 0.0, 1.3 * PX), (wx + 0.012, wy + 0.012, 0.9 * PX),
                     {"all": SHIRT_MAIN, "-Z": SHIRT_DARK}, pbr=(0.82, 0.0, 0.96), bevel_frac=0.18)
    return _finalize(out)


def build_steve_arm_upper():
    """4u x 6u x 4u Upper Arm (pivot at shoulder z=0 -> extends downward to z=-6u, +1.2u cap overlap)."""
    out = {"pos": [], "nrm": [], "uv": [], "col": [], "pbr": [], "idx": []}
    w = 4 * PX
    z0, z1 = -6.0 * PX, 1.2 * PX

    def col_fn(face, u, v):
        if face == "+Z":
            return _jitter(SHIRT_LITE, u, v, 0.02)
        if face == "-Z":
            return SKIN_SHADE
        # Top 4 rows = cyan sleeve, bottom 3 rows = skin bicep
        if v <= 3:
            return _jitter(SHIRT_MAIN if v < 3 else SHIRT_DARK, u, v, 0.025)
        return _jitter(SKIN_BASE if (u + v) % 2 == 0 else SKIN_SHADE, u, v, 0.022)

    def pbr_fn(face, u, v):
        return (0.82, 0.0, 0.98) if v <= 3 else (0.64, 0.0, 0.98)

    _add_pixel_grid_box(out, -w * 0.5, w * 0.5, -w * 0.5, w * 0.5, z0, z1, 4, 4, 7, col_fn, pbr_fn)
    # 3D raised sleeve cuff band
    _add_beveled_box(out, (0.0, 0.0, -2.5 * PX), (w + 0.012, w + 0.012, 1.1 * PX),
                     {"all": SHIRT_MAIN, "-Z": SHIRT_DARK}, pbr=(0.80, 0.0, 0.98), bevel_frac=0.18)
    # Chamfered elbow joint bridge inside pivot so bending never opens a crack!
    _add_beveled_box(out, (0.0, 0.0, -6.0 * PX), (w * 0.92, w * 0.92, 1.4 * PX),
                     {"all": SKIN_SHADE}, pbr=(0.66, 0.0, 0.92), bevel_frac=0.32)
    return _finalize(out)


def build_steve_arm_lower():
    """4u x 6u x 4u Forearm + Hand (pivot at elbow z=0 -> extends downward to z=-6u, +0.8u elbow overlap)."""
    out = {"pos": [], "nrm": [], "uv": [], "col": [], "pbr": [], "idx": []}
    w = 4 * PX
    z0, z1 = -6.0 * PX, 0.8 * PX

    def col_fn(face, u, v):
        if face == "-Z":
            return _jitter(SKIN_SHADE, u, v, 0.02)
        if v >= 5:
            # Subtle knuckle / finger line shading near hand bottom
            return _jitter(SKIN_SHADE if (u in (1, 2)) else SKIN_BASE, u, v, 0.02)
        return _jitter(SKIN_BASE if (u + v) % 2 == 0 else SKIN_SHADE, u, v, 0.022)

    def pbr_fn(face, u, v):
        return (0.64, 0.0, 0.98)

    _add_pixel_grid_box(out, -w * 0.5, w * 0.5, -w * 0.5, w * 0.5, z0, z1, 4, 4, 7, col_fn, pbr_fn)
    return _finalize(out)


def build_steve_leg_upper():
    """4u x 6u x 4u Upper Thigh (pivot at hip z=0 -> extends downward to z=-6u, +1.0u hip overlap)."""
    out = {"pos": [], "nrm": [], "uv": [], "col": [], "pbr": [], "idx": []}
    w = 4 * PX
    z0, z1 = -6.0 * PX, 1.0 * PX

    def col_fn(face, u, v):
        return _jitter(JEANS_MAIN if (u + v) % 2 == 0 else JEANS_DARK, u, v, 0.026)

    def pbr_fn(face, u, v):
        return (0.88, 0.0, 0.96)

    _add_pixel_grid_box(out, -w * 0.5, w * 0.5, -w * 0.5, w * 0.5, z0, z1, 4, 4, 7, col_fn, pbr_fn)
    # Chamfered knee joint bridge for seamless deep knee bends!
    _add_beveled_box(out, (0.0, 0.0, -6.0 * PX), (w * 0.92, w * 0.92, 1.4 * PX),
                     {"all": JEANS_DARK}, pbr=(0.88, 0.0, 0.90), bevel_frac=0.32)
    return _finalize(out)


def build_steve_leg_lower():
    """4u x 6u x 4u Shin + Sneaker/Boot (pivot at knee z=0 -> extends downward to z=-6u, +0.8u knee overlap)."""
    out = {"pos": [], "nrm": [], "uv": [], "col": [], "pbr": [], "idx": []}
    w = 4 * PX
    z0, z1 = -6.0 * PX, 0.8 * PX

    def col_fn(face, u, v):
        if face == "-Z":
            return SHOE_SOLE
        if v <= 4:
            return _jitter(JEANS_MAIN if v < 4 else JEANS_CUFF, u, v, 0.025)
        elif v == 5:
            return _jitter(SHOE_TOP, u, v, 0.02)
        return _jitter(SHOE_SOLE, u, v, 0.015)

    def pbr_fn(face, u, v):
        return (0.88, 0.0, 0.96) if v <= 4 else (0.48, 0.0, 0.98)

    _add_pixel_grid_box(out, -w * 0.5, w * 0.5, -w * 0.5, w * 0.5, z0, z1, 4, 4, 7, col_fn, pbr_fn)
    # Raised 3D sneaker sole + front toe cap (+0.016m forward on +Y so foot heading reads crisply!)
    _add_beveled_box(out, (0.0, 0.008, -5.45 * PX), (w + 0.014, w + 0.026, 1.15 * PX),
                     {"all": SHOE_TOP, "-Z": SHOE_SOLE, "+Y": SHOE_SOLE}, pbr=(0.45, 0.0, 0.98), bevel_frac=0.16)
    return _finalize(out)


def build_minecraft_character_meshes():
    """Return dict of all articulated Steve meshes."""
    arm_u = build_steve_arm_upper()
    arm_l = build_steve_arm_lower()
    leg_u = build_steve_leg_upper()
    leg_l = build_steve_leg_lower()
    return {
        "pelvis":      build_steve_pelvis(),
        "chest":       build_steve_chest(),
        "head":        build_steve_head(),
        "arm_upper_R": arm_u,
        "arm_lower_R": arm_l,
        "arm_upper_L": arm_u,
        "arm_lower_L": arm_l,
        "leg_upper_R": leg_u,
        "leg_lower_R": leg_l,
        "leg_upper_L": leg_u,
        "leg_lower_L": leg_l,
    }


# ==============================================================================
#  HIERARCHICAL 3D MOCAP RETARGETING (MediaPipe P[N,33,3] -> 11 Rigid Rig Parts)
# ==============================================================================

def _norm(v, fallback=(0.0, 0.0, 1.0)):
    n = float(np.linalg.norm(v))
    if n < 1e-6:
        return np.asarray(fallback, dtype=np.float64)
    return v / n


def _frame_from_rfu(right_hint, fwd_hint, up_vec):
    """Build exact right-handed 3x3 rotation matrix [R, F, U] with exact U and clean orthogonal R, F."""
    U = _norm(up_vec, (0.0, 0.0, 1.0))
    # Prefer right_hint projected orthogonal to U, fallback to cross(fwd_hint, U)
    r_p = right_hint - U * float(np.dot(right_hint, U))
    if np.linalg.norm(r_p) < 1e-4:
        r_p = np.cross(fwd_hint, U)
    R = _norm(r_p, (-1.0, 0.0, 0.0))
    F = _norm(np.cross(U, R), (0.0, -1.0, 0.0))
    # Verify handedness: cross(R, F) = U!
    R = _norm(np.cross(F, U), (-1.0, 0.0, 0.0))
    return np.column_stack([R, F, U])


def _limb_frame(dir_down, fwd_ref):
    """Build right-handed [R, F, U] for a limb whose mesh extends along -Z (so U = -dir_down),
    using parallel-transported fwd_ref to prevent axial candy-wrapper twist!"""
    U = -_norm(dir_down, (0.0, 0.0, -1.0))
    f_p = fwd_ref - U * float(np.dot(fwd_ref, U))
    if np.linalg.norm(f_p) < 1e-3:
        # Gimbal fallback orthogonal to U
        alt = np.array([1.0, 0.0, 0.0]) if abs(U[0]) < 0.8 else np.array([0.0, 1.0, 0.0])
        f_p = alt - U * float(np.dot(alt, U))
    F = _norm(f_p)
    R = _norm(np.cross(F, U))
    F = _norm(np.cross(U, R))
    return np.column_stack([R, F, U])


def _smooth_rotations(R_seq, sigma=1.1):
    """Sign-consistent quaternion Gaussian smoothing across a sequence of 3x3 rotation matrices."""
    N = len(R_seq)
    qs = np.array([Rot.from_matrix(R).as_quat() for R in R_seq], dtype=np.float64)
    # Enforce hemisphere continuity (q_i . q_{i-1} >= 0) so 360-deg spins NEVER pop!
    for i in range(1, N):
        if np.dot(qs[i], qs[i - 1]) < 0.0:
            qs[i] = -qs[i]
    # Apply light Gaussian kernel over quaternion components & re-normalize
    rad = max(1, int(math.ceil(sigma * 2.5)))
    ks = np.arange(-rad, rad + 1, dtype=np.float64)
    w = np.exp(-0.5 * (ks / max(sigma, 0.1)) ** 2)
    w /= w.sum()
    q_sm = np.zeros_like(qs)
    for i in range(N):
        acc = np.zeros(4, dtype=np.float64)
        for dk, wk in zip(ks, w):
            j = int(np.clip(i + dk, 0, N - 1))
            acc += wk * qs[j]
        q_sm[i] = acc / (np.linalg.norm(acc) + 1e-9)
    return np.array([Rot.from_quat(q).as_matrix() for q in q_sm], dtype=np.float64)


def _clamp_rel_rot(R_parent, R_child, max_xyz_deg):
    """Clamp child's local Euler angles relative to parent cleanly inside anatomical limits."""
    R_rel = R_parent.T @ R_child
    eul = Rot.from_matrix(R_rel).as_euler("xyz", degrees=True)
    lim = np.asarray(max_xyz_deg, dtype=np.float64)
    eul_c = np.clip(eul, -lim, lim)
    return R_parent @ Rot.from_euler("xyz", eul_c, degrees=True).as_matrix()


def solve_minecraft_rig(P_raw, scale_char=1.0, img_lm=None, cam_C=None, cam_R=None, f_px=None, W=1920, H=1080):
    """Convert MediaPipe 3D world landmarks P_raw[N, 33, 3] into smooth, ground-locked 11-part Minecraft FK poses.

    When img_lm + camera intrinsics/extrinsics are provided, reconstructs root depth directly from the dancer's
    real 2D visual stature (low-pass filtered) + unprojects live 2D hip center -> eliminates 100% of raw MediaPipe
    single-foot depth spikes (no more sudden shrinking/growing during knee lifts or 360-deg spins!).

    Returns:
      rig_pos:  dict[part_name -> (N, 3) float32] world pivot positions
      rig_rot:  dict[part_name -> (N, 3, 3) float32] world rotation matrices
      rig_jnts: (N, 12, 3) float32 solved Minecraft FK joint positions (for breakdown skeleton/rig pass!)
    """
    N = len(P_raw)
    # 1. Savitzky-Golay temporal smoothing (preserves snappy dance pops, removes MediaPipe Z jitter!)
    win = min(7, N if N % 2 == 1 else N - 1)
    P = np.zeros_like(P_raw, dtype=np.float64)
    for j in range(33):
        for a in range(3):
            P[:, j, a] = savgol_filter(P_raw[:, j, a], window_length=win, polyorder=2)

    u = PX * float(scale_char)
    # Authentic Minecraft FK bone segment lengths (scaled by scale_char)
    L_PELV_TO_CHEST = 2.5 * u
    L_CHEST_TO_HEAD = 8.0 * u
    L_ARM_UPPER     = 6.0 * u
    L_ARM_LOWER     = 6.0 * u
    L_LEG_UPPER     = 6.0 * u
    L_LEG_LOWER     = 6.0 * u

    # Reference floor elevation from real dancer's grounded feet (p10 across clip)
    real_feet_z = np.minimum(np.minimum(P[:, 29, 2], P[:, 30, 2]), np.minimum(P[:, 31, 2], P[:, 32, 2]))
    floor_ref = float(np.percentile(real_feet_z, 10.0))

    R_pelv_raw = np.zeros((N, 3, 3))
    R_chst_raw = np.zeros((N, 3, 3))
    R_head_raw = np.zeros((N, 3, 3))

    prev_fwd = np.array([0.0, -1.0, 0.0], dtype=np.float64)
    for i in range(N):
        hip_c = 0.5 * (P[i, 23] + P[i, 24])
        sh_c  = 0.5 * (P[i, 11] + P[i, 12])
        up_sp = _norm(sh_c - hip_c, (0.0, 0.0, 1.0))

        # MediaPipe 12=Right Shoulder, 11=Left Shoulder -> R_vec = P[12] - P[11] points character +X (right)!
        sh_R = P[i, 12] - P[i, 11]
        hp_R = P[i, 24] - P[i, 23]
        # Blend shoulder & hip right vectors gently for stable pelvis/chest facing even on side profiles
        torso_R_c = sh_R * 0.78 + hp_R * 0.22
        torso_R_p = hp_R * 0.72 + sh_R * 0.28

        # Protect against degenerate foreshortened side-profile shoulder collapse using nose & previous forward!
        if np.linalg.norm(torso_R_c - up_sp * np.dot(torso_R_c, up_sp)) < 0.08:
            nose_f = P[i, 0] - 0.5 * (P[i, 7] + P[i, 8])
            fwd_fb = _norm(nose_f * 0.5 + prev_fwd * 0.5, prev_fwd)
            torso_R_c = np.cross(fwd_fb, up_sp)
            torso_R_p = torso_R_c

        Rc = _frame_from_rfu(torso_R_c, prev_fwd, up_sp)
        Rp = _frame_from_rfu(torso_R_p, Rc[:, 1], _norm(up_sp * 0.7 + np.array([0.0, 0.0, 0.3])))
        prev_fwd = Rc[:, 1].copy()

        # Real 3D head orientation from ears (8=R, 7=L) + nose (0) + eye-to-mouth vertical!
        ear_c = 0.5 * (P[i, 7] + P[i, 8])
        eye_c = 0.5 * (P[i, 2] + P[i, 5])
        mth_c = 0.5 * (P[i, 9] + P[i, 10])
        h_R = P[i, 8] - P[i, 7]
        h_F = P[i, 0] - ear_c
        h_U = _norm((eye_c - mth_c) * 0.65 + Rc[:, 2] * 0.35, Rc[:, 2])
        if np.linalg.norm(h_R) < 0.05:
            h_R = np.cross(h_F, h_U)
        Rh = _frame_from_rfu(h_R, h_F, h_U)
        # Ensure head front roughly agrees with nose direction (not accidentally inverted on profile)
        if np.dot(Rh[:, 1], _norm(h_F, Rc[:, 1])) < -0.25:
            Rh = _frame_from_rfu(-h_R, h_F, h_U)

        R_pelv_raw[i] = Rp
        R_chst_raw[i] = Rc
        R_head_raw[i] = Rh

    # Smooth core torso/head rotations with hemisphere-continuous quaternion filtering!
    R_pelv = _smooth_rotations(R_pelv_raw, sigma=1.4)
    R_chst = _smooth_rotations(R_chst_raw, sigma=1.2)
    R_head = _smooth_rotations(R_head_raw, sigma=1.3)

    # Enforce clean anatomical twist limits (chest vs pelvis, head vs chest)
    for i in range(N):
        R_chst[i] = _clamp_rel_rot(R_pelv[i], R_chst[i], (22.0, 22.0, 38.0))
        R_head[i] = _clamp_rel_rot(R_chst[i], R_head[i], (32.0, 24.0, 62.0))

    # Solve articulated limbs via smooth direction vectors + parallel-transported twist frames
    names = ["pelvis", "chest", "head",
             "arm_upper_R", "arm_lower_R", "arm_upper_L", "arm_lower_L",
             "leg_upper_R", "leg_lower_R", "leg_upper_L", "leg_lower_L"]
    R_limb_raw = {k: np.zeros((N, 3, 3)) for k in names[3:]}

    for i in range(N):
        Rc = R_chst[i]
        Rp = R_pelv[i]
        # Arms: R = (12->14->16), L = (11->13->15)
        for side, i_sh, i_el, i_wr in (("R", 12, 14, 16), ("L", 11, 13, 15)):
            d_u = _norm(P[i, i_el] - P[i, i_sh], -Rc[:, 2])
            d_l = _norm(P[i, i_wr] - P[i, i_el], d_u)
            # Align upper arm +Y front gently with chest front + elbow bend direction
            Ru = _limb_frame(d_u, Rc[:, 1])
            # Parallel-transport Ru's +Y onto forearm so elbow hinge is 100% twist-free!
            Rl = _limb_frame(d_l, Ru[:, 1] * 0.75 + Rc[:, 1] * 0.25)
            R_limb_raw[f"arm_upper_{side}"][i] = Ru
            R_limb_raw[f"arm_lower_{side}"][i] = Rl

        # Legs: R = (24->26->28), L = (23->25->27)
        for side, i_hp, i_kn, i_an, i_ft in (("R", 24, 26, 28, 32), ("L", 23, 25, 27, 31)):
            d_u = _norm(P[i, i_kn] - P[i, i_hp], -Rp[:, 2])
            d_l = _norm(P[i, i_an] - P[i, i_kn], d_u)
            # Blend pelvis front with foot-toe forward hint for ultra-natural knee & shoe pointing!
            foot_f = P[i, i_ft] - P[i, i_an]
            foot_f_xy = np.array([foot_f[0], foot_f[1], 0.0])
            f_hint = _norm(Rp[:, 1] * 0.78 + _norm(foot_f_xy, Rp[:, 1]) * 0.22, Rp[:, 1])
            Ru = _limb_frame(d_u, f_hint)
            Rl = _limb_frame(d_l, Ru[:, 1] * 0.80 + f_hint * 0.20)
            R_limb_raw[f"leg_upper_{side}"][i] = Ru
            R_limb_raw[f"leg_lower_{side}"][i] = Rl

    R_all = {"pelvis": R_pelv, "chest": R_chst, "head": R_head}
    for k, arr in R_limb_raw.items():
        R_all[k] = _smooth_rotations(arr, sigma=0.95)

    # Forward Kinematics assembly + exact Ground Contact Locking!
    pos_all = {k: np.zeros((N, 3), dtype=np.float64) for k in names}
    jnts = np.zeros((N, 12, 3), dtype=np.float64)

    # Local shoe bottom 4 corners on leg_lower mesh (z = -6u)
    w_s = 2.0 * u
    shoe_corners = np.array([
        [-w_s, -w_s, -L_LEG_LOWER], [w_s, -w_s, -L_LEG_LOWER],
        [w_s,   w_s, -L_LEG_LOWER], [-w_s, w_s, -L_LEG_LOWER],
    ], dtype=np.float64)

    # Smooth horizontal hip anchor & vertical jump lift
    # When 2D landmarks & camera are supplied, lock root 3D depth cleanly to smooth real 2D stature!
    hip_track = 0.5 * (P[:, 23] + P[:, 24])
    if img_lm is not None and cam_C is not None and cam_R is not None and f_px is not None:
        # Measure real standing 2D span from head/shoulders down to LOWEST grounded foot
        v_foot = np.max(img_lm[:, [27, 28, 29, 30, 31, 32], 1], axis=1)
        v_head = img_lm[:, 0, 1]
        v_sh   = 0.5 * (img_lm[:, 11, 1] + img_lm[:, 12, 1])
        h_full = np.maximum(v_foot - v_head, 150.0)
        h_shf  = np.maximum(v_foot - v_sh, 120.0)
        # Calibrate physical stature (meters) against median depth of raw 3D pose
        raw_dep = np.array([float(((hip_track[i] - cam_C[i]) @ cam_R[i])[2]) for i in range(N)], dtype=np.float64)
        dep_med = float(np.median(raw_dep))
        H_full_m = dep_med * float(np.median(h_full)) / float(f_px)
        H_shf_m  = dep_med * float(np.median(h_shf))  / float(f_px)
        dep_est = 0.55 * (float(f_px) * H_full_m / h_full) + 0.45 * (float(f_px) * H_shf_m / h_shf)
        # Heavy Savitzky-Golay smoothing (window=41 -> ~1.35s) kills 100% of jump/spin depth pumping while
        # preserving smooth true walking distance! Plus clamp gently around median so size never breathes wildly.
        w_dep = min(41, N if N % 2 == 1 else N - 1)
        dep_sm = savgol_filter(dep_est, window_length=w_dep, polyorder=2)
        # Keep depth smoothly anchored within natural sidewalk band (gentle 75% damping toward median avoids telephoto zoom exaggeration)
        dep_sm = dep_med + 0.72 * (dep_sm - dep_med)

        # Unproject smooth 2D hip center (u_hip, v_hip) to exact dep_sm[i] along live camera ray!
        u_hip = savgol_filter(0.5 * (img_lm[:, 23, 0] + img_lm[:, 24, 0]), window_length=win, polyorder=2)
        v_hip = savgol_filter(0.5 * (img_lm[:, 23, 1] + img_lm[:, 24, 1]), window_length=win, polyorder=2)
        for i in range(N):
            ray_cam = np.array([(u_hip[i] - W * 0.5) / float(f_px), (v_hip[i] - H * 0.5) / float(f_px), 1.0], dtype=np.float64)
            p_world = cam_C[i] + cam_R[i] @ (ray_cam * dep_sm[i])
            hip_track[i, 0] = p_world[0]
            hip_track[i, 1] = p_world[1]

    lift_track = np.clip(real_feet_z - floor_ref - 0.035, 0.0, 0.35)
    # Extra light smoothing on lift_track so grounded stances stay rock-steady at z=0.000m!
    lift_track = savgol_filter(lift_track, window_length=win, polyorder=2)
    lift_track = np.clip(lift_track, 0.0, 0.35)

    for i in range(N):
        Rp = R_all["pelvis"][i]
        Rc = R_all["chest"][i]
        Rh = R_all["head"][i]

        # Provisional pelvis center at live hip_track
        p_pelv = hip_track[i].copy()
        # Local hip sockets: (+-2u, 0, -0.6u)
        p_hip_R = p_pelv + Rp @ np.array([+2.0 * u, 0.0, -0.6 * u])
        p_hip_L = p_pelv + Rp @ np.array([-2.0 * u, 0.0, -0.6 * u])
        p_kn_R  = p_hip_R + R_all["leg_upper_R"][i] @ np.array([0.0, 0.0, -L_LEG_UPPER])
        p_kn_L  = p_hip_L + R_all["leg_upper_L"][i] @ np.array([0.0, 0.0, -L_LEG_UPPER])

        # Exact lowest shoe corner Z across BOTH feet -> lock smoothly to exact ground + real jump lift!
        c_R = p_kn_R[None, :] + (shoe_corners @ R_all["leg_lower_R"][i].T)
        c_L = p_kn_L[None, :] + (shoe_corners @ R_all["leg_lower_L"][i].T)
        min_z = float(min(c_R[:, 2].min(), c_L[:, 2].min()))
        dz_lock = float(lift_track[i]) - min_z

        # Apply exact vertical ground lock to root and recompute clean FK chain!
        p_pelv[2] += dz_lock
        p_chst  = p_pelv + Rp @ np.array([0.0, 0.0, L_PELV_TO_CHEST])
        p_head  = p_chst + Rc @ np.array([0.0, 0.0, L_CHEST_TO_HEAD])
        p_sh_R  = p_chst + Rc @ np.array([+6.0 * u, 0.0, 6.8 * u])
        p_sh_L  = p_chst + Rc @ np.array([-6.0 * u, 0.0, 6.8 * u])
        p_el_R  = p_sh_R + R_all["arm_upper_R"][i] @ np.array([0.0, 0.0, -L_ARM_UPPER])
        p_el_L  = p_sh_L + R_all["arm_upper_L"][i] @ np.array([0.0, 0.0, -L_ARM_UPPER])
        p_wr_R  = p_el_R + R_all["arm_lower_R"][i] @ np.array([0.0, 0.0, -L_ARM_LOWER])
        p_wr_L  = p_el_L + R_all["arm_lower_L"][i] @ np.array([0.0, 0.0, -L_ARM_LOWER])

        p_hip_R[2] += dz_lock
        p_hip_L[2] += dz_lock
        p_kn_R[2]  += dz_lock
        p_kn_L[2]  += dz_lock
        p_an_R  = p_kn_R + R_all["leg_lower_R"][i] @ np.array([0.0, 0.0, -L_LEG_LOWER])
        p_an_L  = p_kn_L + R_all["leg_lower_L"][i] @ np.array([0.0, 0.0, -L_LEG_LOWER])

        pos_all["pelvis"][i]      = p_pelv
        pos_all["chest"][i]       = p_chst
        pos_all["head"][i]        = p_head
        pos_all["arm_upper_R"][i] = p_sh_R
        pos_all["arm_lower_R"][i] = p_el_R
        pos_all["arm_upper_L"][i] = p_sh_L
        pos_all["arm_lower_L"][i] = p_el_L
        pos_all["leg_upper_R"][i] = p_hip_R
        pos_all["leg_lower_R"][i] = p_kn_R
        pos_all["leg_upper_L"][i] = p_hip_L
        pos_all["leg_lower_L"][i] = p_kn_L

        # Store 12 FK rig joints for diagnostic rig visualization pass
        jnts[i] = np.array([
            p_pelv, p_chst, p_head, p_head + Rh @ np.array([0.0, 0.0, 4.0 * u]),
            p_sh_R, p_el_R, p_wr_R,
            p_sh_L, p_el_L, p_wr_L,
            p_kn_R, p_an_R,
        ])

    return (
        {k: v.astype(np.float32) for k, v in pos_all.items()},
        {k: v.astype(np.float32) for k, v in R_all.items()},
        jnts.astype(np.float32),
    )
