"""Measure where the light comes from, using the plate itself and the solved camera.

Sampled pixels of every 3rd keyframe become world rays (the ground-aligned camera).
The brightest pixels above the horizon (lamps, windows, open sides) are placed in 3D
at the depth of the nearest reconstructed COLMAP points, so their direction and solid
angle are taken from the OBJECT's position, not from the camera's:
  * key + second  the two strongest groups of bright sources (2-means on directions
              seen from the object), each with direction, colour, size and irradiance
  * ambient   mean radiance of the other directions above the horizon
  * bounce    radiance of the real floor around the object (seen from the camera)
  * grain     per-channel noise of a flat floor patch (robust high-pass)
The separate CC0 HDRI is then turned (yaw only) so its own brightest region faces the
same way, and scaled so its ambient level matches the plate. Only what the camera saw
is measured: light behind the camera is unknown, and clipped highlights make the key
irradiance a lower bound. Check qa/light-estimate.jpg and qa/light-probe.jpg.
"""
from __future__ import annotations

import cv2
import numpy as np

from .common import StepError, checkpoint, frame_path, load_json, save_json, shot_dir

STEP = 4            # sample every 4th pixel
FRAME_STEP = 9      # every 3rd keyframe (keyframes are every 3rd frame)


def to_linear(srgb):
    return np.where(srgb <= .04045, srgb/12.92, ((srgb+.055)/1.055)**2.4)


def luminance(rgb):
    return rgb@np.array([.2126, .7152, .0722])


def camera(s, i):
    return np.array(s["frames"][i]["matrixWorld"]).reshape(4, 4).T


def pixel_rays(s, c, step=STEP):
    """World ray directions and per-pixel solid angles (steradians) for a sampled grid."""
    W, H = s["width"], s["height"]
    v, u = np.mgrid[step/2:H:step, step/2:W:step]
    x, y = (u-s["cx"])/s["fx"], -(v-s["cy"])/s["fy"]
    local = np.stack([x, y, -np.ones_like(x)], -1)
    dirs = local@c[:3, :3].T
    dirs /= np.linalg.norm(dirs, axis=-1, keepdims=True)
    omega = step*step/(s["fx"]*s["fy"])/(1+x*x+y*y)**1.5
    return u.astype(int), v.astype(int), dirs, omega


def project(s, c, point):
    p = np.linalg.inv(c)@np.append(point, 1.)
    if p[2] >= -1e-6:
        return None
    return np.array([s["cx"]+s["fx"]*p[0]/-p[2], s["cy"]-s["fy"]*p[1]/-p[2]])


def direction_angles(d):
    return float(np.degrees(np.arctan2(d[2], d[0]))), float(np.degrees(np.arcsin(np.clip(d[1], -1, 1))))


def hdri_directions(h, w):
    rows, cols = np.mgrid[0:h, 0:w]
    lon = ((cols+.5)/w-.5)*2*np.pi
    lat = (.5-(rows+.5)/h)*np.pi
    dirs = np.stack([np.cos(lat)*np.cos(lon), np.sin(lat), np.cos(lat)*np.sin(lon)], -1)
    omega = np.cos(lat)*(2*np.pi/w)*(np.pi/h)
    return dirs, omega


def bright_centre(dirs, lum, omega, upper, quantile):
    sel = upper & (lum >= np.quantile(lum[upper], quantile))
    weight = (lum*omega)[sel]
    centre = (dirs[sel]*weight[:, None]).sum(0)
    return sel, centre/np.linalg.norm(centre)


def two_groups(dirs, weight, iterations=20):
    """Weighted 2-means on unit vectors; returns groups sorted by total weight."""
    centres = [dirs[np.argmax(weight)]]
    far = np.argmin(dirs@centres[0])
    centres.append(dirs[far])
    centres = np.array(centres)
    for _ in range(iterations):
        label = np.argmax(dirs@centres.T, 1)
        for k in range(2):
            if (label == k).any():
                m = (dirs[label == k]*weight[label == k, None]).sum(0)
                centres[k] = m/np.linalg.norm(m)
    groups = []
    for k in range(2):
        sel = label == k
        if sel.any():
            groups.append((float(weight[sel].sum()), centres[k], sel))
    return sorted(groups, key=lambda g: -g[0])


def landmarks(d, layout):
    """Reconstructed points in ground-aligned world coordinates, per keyframe index."""
    import pycolmap
    r = pycolmap.Reconstruction(str(d/"colmap/fit"))
    t = layout["transform"]
    R, origin, scale = np.array(t["rotation"]), np.array(t["origin"]), t["scale"]
    per_frame = {}
    for im in r.images.values():
        xy, xyz = [], []
        for pt in im.points2D:
            if pt.has_point3D():
                xy.append(pt.xy)
                xyz.append(r.points3D[pt.point3D_id].xyz)
        if xy:
            per_frame[int(im.name[6:12])] = (np.array(xy), scale*(np.array(xyz)-origin)@R.T)
    return per_frame


def estimate(cfg, d, hdri_path):
    from scipy.spatial import cKDTree
    clip_gain = float(cfg.get("relight", {}).get("clipped_gain", 8.))
    s = load_json(d/"Camera_Three_ground.json")
    layout = load_json(d/"layout.json")
    N = s["count"]
    radius = layout["base_radius"]
    target = np.array([0., .25*radius, 0.])          # the middle of the CG, just above the floor
    points = landmarks(d, layout)
    keys = sorted(points)[::3]
    samples = []
    for i in keys:
        c = camera(s, i)
        u, v, dirs, omega = pixel_rays(s, c)
        img = to_linear(cv2.imread(str(frame_path(d, i)))[:, :, ::-1].astype("f4")/255.)
        samples.append((i, c, u.reshape(-1), v.reshape(-1), dirs.reshape(-1, 3), img[v, u].reshape(-1, 3),
                        omega.reshape(-1)))
    frames = len(samples)
    dirs = np.concatenate([x[4] for x in samples])
    rgb = np.concatenate([x[5] for x in samples])
    omega = np.concatenate([x[6] for x in samples])
    lum = luminance(rgb)
    upper = dirs[:, 1] > .03
    if upper.mean() < .05:
        raise StepError("Less than 5% of the picture is above the object's horizon (a top-down shot?). The light "
                        "cannot be measured from this plate, so the sphere relight cannot run; keep the render step's "
                        "result for this shot, or choose footage whose camera sees lamps, windows or sky.")
    threshold = np.quantile(lum[upper], .97)
    clipped_all, src_dirs, src_weight, src_rgb, placed = [], [], [], [], 0
    for i, c, u, v, di, col, om in samples:
        sel = (di[:, 1] > .03) & (luminance(col) >= threshold)
        if not sel.any():
            continue
        xy, xyz = points[i]
        tree = cKDTree(xy)
        cam = c[:3, 3]
        along = (xyz-cam)@di[sel].T                     # distance of every landmark along every bright ray
        dist_all = np.linalg.norm(xyz-cam, axis=1)
        far_cap = float(np.quantile(dist_all, .95))
        near = tree.query_ball_point(np.stack([u[sel], v[sel]], 1), r=48)
        depth = np.empty(sel.sum())
        for k, idx in enumerate(near):
            if idx:
                depth[k] = np.median(along[idx, k])
                placed += 1
            else:
                depth[k] = far_cap
        depth = np.clip(depth, .5, far_cap)
        pos = cam+di[sel]*depth[:, None]
        to = pos-target
        dist_obj = np.linalg.norm(to, axis=1)
        dobj = to/dist_obj[:, None]
        clipped = col[sel].max(1) > .98
        # A clipped pixel only says "at least white". Its real radiance is assumed clip_gain x higher.
        w = luminance(col[sel])*np.where(clipped, clip_gain, 1.)*om[sel]*(depth/dist_obj)**2
        keep = dobj[:, 1] > .02
        src_dirs.append(dobj[keep])
        src_weight.append(w[keep])
        src_rgb.append(col[sel][keep])
        clipped_all.append(clipped[keep])
    src_dirs, src_weight = np.concatenate(src_dirs), np.concatenate(src_weight)
    src_rgb, clipped_all = np.concatenate(src_rgb), np.concatenate(clipped_all)
    if len(src_dirs) < 20:
        raise StepError("Too few bright sources above the object to measure the light; keep the render step's result.")
    lights = []
    for weight, centre, sel in two_groups(src_dirs, src_weight):
        colour = (src_rgb[sel]*src_weight[sel, None]).sum(0)
        colour = colour/max(luminance(colour), 1e-8)
        spread = float(np.degrees(np.arccos(np.clip(src_dirs[sel]@centre, -1, 1))).mean())
        az, el = direction_angles(centre)
        lights.append(dict(direction=centre.tolist(), azimuth_deg=az, elevation_deg=el, rgb=colour.tolist(),
                           irradiance=float(weight/frames), mean_spread_deg=spread,
                           clipped_fraction=float(clipped_all[sel].mean())))
    key = np.array(lights[0]["direction"])
    clipped = float(clipped_all.mean())
    rest = upper & (lum < threshold)
    ambient_L = float((lum*omega)[rest].sum()/omega[rest].sum())
    ambient_rgb = (rgb[rest]*omega[rest][:, None]).sum(0)/omega[rest].sum()
    bright = None
    # Real floor around the object, seen in the middle frame.
    mid = N//2
    c = camera(s, mid)
    u, v, rays, _ = pixel_rays(s, c, 2)
    t = -c[1, 3]/np.where(np.abs(rays[..., 1]) > 1e-6, rays[..., 1], np.nan)
    hit = c[:3, 3]+t[..., None]*rays
    near = (t > 0) & (np.hypot(hit[..., 0], hit[..., 2]) < 2.2*radius) & (np.hypot(hit[..., 0], hit[..., 2]) > 1.1*radius)
    if near.sum() < 200:
        near = (t > 0) & (np.hypot(hit[..., 0], hit[..., 2]) < 3.5*radius)
    srgb = cv2.imread(str(frame_path(d, mid)))[:, :, ::-1].astype("f4")/255.
    floor_rgb = np.median(to_linear(srgb[v[near], u[near]]), axis=0)
    high = srgb-cv2.GaussianBlur(srgb, (0, 0), 1.2)
    patch = high[v[near], u[near]]
    grain = 1.4826*np.median(np.abs(patch-np.median(patch, 0)), 0)*255

    floor_light = [l["irradiance"]*max(l["direction"][1], 0.) for l in lights]
    floor_ambient = np.pi*ambient_L
    for l, e in zip(lights, floor_light):
        l["floor_fraction"] = float(e/(sum(floor_light)+floor_ambient))

    hdr = cv2.imread(str(hdri_path), cv2.IMREAD_UNCHANGED)
    if hdr is None:
        raise StepError(f"HDRI missing: {hdri_path}. Run: python -m pipeline fetch {cfg['name']}")
    hdr = cv2.resize(hdr, (512, 256), interpolation=cv2.INTER_AREA)[:, :, ::-1].astype("f4")
    hdirs, homega = hdri_directions(*hdr.shape[:2])
    hdirs, homega, hrgb = hdirs.reshape(-1, 3), homega.reshape(-1), hdr.reshape(-1, 3)
    hlum = luminance(hrgb)
    hupper = hdirs[:, 1] > .03
    hbright, hkey = bright_centre(hdirs, hlum, homega, hupper, .97)
    hrest = hupper & ~hbright
    hdri_ambient = float((hlum*homega)[hrest].sum()/homega[hrest].sum())
    key_az, key_el = direction_angles(key)
    hdri_az, hdri_el = direction_angles(hkey)
    yaw = np.radians(hdri_az-key_az)
    result = dict(
        method="plate pixels -> world rays through the solved, ground-aligned camera (every 3rd keyframe, every "
               f"{STEP}th pixel); bright sources placed at the depth of nearby COLMAP landmarks and seen from the "
               "object; ambient uses the distant-light assumption",
        frames_sampled=frames, upper_hemisphere_seen_fraction=float(omega[upper].sum()/frames/(2*np.pi)),
        measured_from=target.tolist(), sources_placed_on_landmarks=float(placed/max(len(src_dirs), 1)),
        clipped_gain=clip_gain,
        lights=lights, key_direction=key.tolist(), key_azimuth_deg=key_az, key_elevation_deg=key_el,
        key_clipped_fraction=clipped, ambient_radiance=ambient_L, ambient_rgb=ambient_rgb.tolist(),
        floor_radiance_rgb=floor_rgb.tolist(), floor_key_fraction=lights[0]["floor_fraction"],
        plate_grain_std_8bit=grain.tolist(),
        hdri=dict(file=str(hdri_path), key_azimuth_deg=hdri_az, key_elevation_deg=hdri_el,
                  yaw_rad=float(yaw), ambient_radiance=hdri_ambient, scale=float(ambient_L/max(hdri_ambient, 1e-8))),
        limitations=["Only directions the camera saw are measured; light behind the camera is unknown",
                     f"{clipped:.0%} of the bright pixels are clipped; their radiance is ASSUMED {clip_gain:g}x the "
                     "clipped value (relight.clipped_gain), which sets how dark the shadows are",
                     "Bright sources without nearby landmarks are placed at a far distance",
                     "Lamps and openings are merged into two groups",
                     "The HDRI is only turned and scaled to match; it is still a different place"])
    return result, s, None, hdr


def tonemap(rgb):
    return np.uint8(np.clip((rgb/(1+rgb))**(1/2.2)*255, 0, 255))


def latlong(dirs, rgb, w=512, h=256):
    lon = np.arctan2(dirs[:, 2], dirs[:, 0])
    lat = np.arcsin(np.clip(dirs[:, 1], -1, 1))
    cols = np.clip(((lon/(2*np.pi)+.5)*w).astype(int), 0, w-1)
    rows = np.clip(((.5-lat/np.pi)*h).astype(int), 0, h-1)
    acc, cnt = np.zeros((h, w, 3), "f4"), np.zeros((h, w, 1), "f4")
    np.add.at(acc, (rows, cols), rgb)
    np.add.at(cnt, (rows, cols), 1)
    return acc/np.maximum(cnt, 1), cnt[:, :, 0] > 0


def marker(img, direction, colour, w=512, h=256):
    az, el = direction_angles(np.array(direction))
    x, y = int((az/360+.5)*w), int((.5-el/180)*h)
    cv2.drawMarker(img, (x, y), colour, cv2.MARKER_CROSS, 22, 2)


def visualise(d, result, s, bright_dirs, hdr):
    radius = load_json(d/"layout.json")["base_radius"]
    mid = s["count"]//2
    c = camera(s, mid)
    vis = cv2.imread(str(frame_path(d, mid)))
    u, v, dirs, _ = pixel_rays(s, c)
    img = cv2.imread(str(frame_path(d, mid))).astype("f4")/255.
    lum = luminance(to_linear(img[v, u][:, :, ::-1]))
    upper = dirs[..., 1] > .03
    sel = upper & (lum >= np.quantile(lum[upper], .97))
    for x, y in zip(u[sel], v[sel]):
        cv2.circle(vis, (int(x), int(y)), 2, (0, 140, 255), -1)
    base = np.array(result["measured_from"])
    a = project(s, c, base)
    if a is not None:
        cv2.drawMarker(vis, tuple(int(t) for t in a), (0, 0, 255), cv2.MARKER_CROSS, 26, 2)
        for light, colour in zip(result["lights"], ((0, 255, 255), (255, 200, 0))):
            b = project(s, c, base+np.array(light["direction"])*radius*1.4)
            if b is not None:
                cv2.arrowedLine(vis, tuple(int(t) for t in a), tuple(int(t) for t in b), colour, 3, tipLength=.12)
    lines = [f"{name}: azimuth {l['azimuth_deg']:.0f}, elevation {l['elevation_deg']:.0f} deg, floor share {l['floor_fraction']:.0%}"
             for name, l in zip(("key (yellow)", "second (blue)"), result["lights"])]
    lines.append(f"upper hemisphere seen {result['upper_hemisphere_seen_fraction']:.0%}; sources on landmarks "
                 f"{result['sources_placed_on_landmarks']:.0%}; HDRI turned {np.degrees(result['hdri']['yaw_rad']):.0f} deg")
    for k, line in enumerate(lines):
        cv2.putText(vis, line, (12, 28+24*k), cv2.FONT_HERSHEY_SIMPLEX, .62, (0, 0, 0), 4)
        cv2.putText(vis, line, (12, 28+24*k), cv2.FONT_HERSHEY_SIMPLEX, .62, (255, 255, 255), 1)
    (d/"qa").mkdir(exist_ok=True)
    cv2.imwrite(str(d/"qa/light-estimate.jpg"), vis)

    # Plate probe (what the camera saw) next to the turned and scaled HDRI, both as lat-long maps.
    dirs_all, rgb_all = [], []
    for i in range(0, s["count"], 8):
        ci = camera(s, i)
        u, v, di, _ = pixel_rays(s, ci)
        im = to_linear(cv2.imread(str(frame_path(d, i)))[:, :, ::-1].astype("f4")/255.)
        dirs_all.append(di.reshape(-1, 3))
        rgb_all.append(im[v, u].reshape(-1, 3))
    probe, seen = latlong(np.concatenate(dirs_all), np.concatenate(rgb_all))
    probe_img = tonemap(probe*3)[:, :, ::-1].copy()
    probe_img[~seen] = (40, 40, 40)
    shift = int(round(-result["hdri"]["yaw_rad"]/(2*np.pi)*hdr.shape[1]))
    turned = np.roll(hdr, shift, axis=1)*result["hdri"]["scale"]
    hdri_img = tonemap(turned*3)[:, :, ::-1].copy()
    for img_, label in ((probe_img, "plate as the camera saw it (crosses: lights seen from the object)"), (hdri_img, "HDRI turned + scaled")):
        for light, colour in zip(result["lights"], ((0, 255, 255), (255, 200, 0))):
            marker(img_, light["direction"], colour)
        cv2.line(img_, (0, 128), (511, 128), (255, 255, 255), 1)
        cv2.putText(img_, label, (8, 18), cv2.FONT_HERSHEY_SIMPLEX, .5, (255, 255, 255), 1)
    cv2.imwrite(str(d/"qa/light-probe.jpg"), np.vstack([probe_img, hdri_img]))


def main(cfg, work, args):
    from .fetch import lighting_path
    d = shot_dir(cfg, work)
    if not (d/"layout.json").is_file():
        raise StepError("Run ground first: python -m pipeline ground "+cfg["name"])
    result, s, bright_dirs, hdr = estimate(cfg, d, lighting_path(cfg, work))
    save_json(d/"light.json", result)
    visualise(d, result, s, bright_dirs, hdr)
    checkpoint(d, "light_estimated", next="python -m pipeline relight "+cfg["name"])
    for light in result["lights"]:
        print({k: round(v, 3) if isinstance(v, float) else v for k, v in light.items() if k != "direction"})
    print({k: result[k] for k in ("upper_hemisphere_seen_fraction", "sources_placed_on_landmarks", "ambient_radiance")})
    print("Look at qa/light-estimate.jpg (orange: brightest directions, yellow arrow: toward the key light) and "
          "qa/light-probe.jpg (the HDRI's bright area should sit under the same cross as the plate's).")
