"""Simulation on the CPU, saved per frame for the OpenGL renderer.

water          Water/bake-water.mjs SPH (370 particles, Node) -> Three.js MarchingCubes surfaces
rigid_spheres  7 spheres, low gravity, floor/wall/pair impulses
cloth          33x25 position-based cloth pinned to four posts, gravity + wind gust
All values are art-directed in arbitrary scene units, not measured physics.
"""
from __future__ import annotations

import numpy as np

from .common import ROOT, StepError, checkpoint, load_json, run, save_json, shot_dir, tool


def water(cfg, d, cam, layout):
    cache = ROOT/"Water/water-particles.bin"
    if not cache.is_file():
        run([tool("node"), ROOT/"Water/bake-water.mjs"])
    if not (ROOT/"Source/node_modules/three").is_dir():
        raise StepError("three.js is missing. Run: npm ci --prefix Source")
    out = d/"sim/surfaces"
    if out.is_dir() and len(list(out.glob("*.bin"))) == cam["count"]:
        print("surfaces exist", out)
    else:
        run([tool("node"), ROOT/"Source/shot_surfaces.mjs", out, cam["count"], cam["fps"][0]/cam["fps"][1]])
    return dict(type="Water/bake-water.mjs SPH, 370 particles; Three.js MarchingCubes surface per frame",
                frames=len(list(out.glob("*.bin"))), cache=str(cache), units="arbitrary", pass_=True)


def spheres(cfg, d, cam, layout):
    L = layout["water_scale"]
    N = cam["count"]
    fps = cam["fps"][0]/cam["fps"][1]
    radii = L*np.array([.135, .155, .12, .145, .13, .15, .125])
    mass = (radii/L)**3
    angle = np.arange(7)*2*np.pi/7
    pos = np.c_[.53*L*np.cos(angle), L*(.35+.20*(np.arange(7) % 3)), .53*L*np.sin(angle)]
    vel = np.c_[-.22*L*np.sin(angle), L*(.1+.04*np.arange(7)), .22*L*np.cos(angle)]
    g = 1.1*L
    dt = 1/(fps*12)
    history, energies, overlap, contacts = [], [], [], 0
    for _ in range(N):
        history.append(pos.copy())
        energies.append(float(np.sum(mass*(.5*np.sum(vel*vel, 1)+g*pos[:, 1]))))
        overlap.append(max([max(0, float(radii[i]+radii[j]-np.linalg.norm(pos[i]-pos[j])))
                            for i in range(7) for j in range(i)] or [0]))
        for _ in range(12):
            vel[:, 1] -= g*dt
            pos += vel*dt
            for i in range(7):
                if pos[i, 1] < radii[i]:
                    pos[i, 1] = radii[i]
                    vel[i, 1] = abs(vel[i, 1])*.79
                    vel[i, [0, 2]] *= .985
                    contacts += 1
                q = pos[i, [0, 2]]
                rad = np.linalg.norm(q)
                bound = .84*L-radii[i]
                if rad > bound:
                    normal = q/rad
                    pos[i, [0, 2]] = normal*bound
                    v = vel[i, [0, 2]]
                    vn = np.dot(v, normal)
                    if vn > 0:
                        vel[i, [0, 2]] = v-1.82*vn*normal
            for _ in range(2):
                for i in range(7):
                    for j in range(i):
                        delta = pos[i]-pos[j]
                        distance = np.linalg.norm(delta)
                        limit = radii[i]+radii[j]
                        if distance >= limit or distance < 1e-9:
                            continue
                        normal = delta/distance
                        wi, wj = 1/mass[i], 1/mass[j]
                        correction = (limit-distance)/(wi+wj)
                        pos[i] += normal*correction*wi
                        pos[j] -= normal*correction*wj
                        vn = np.dot(vel[i]-vel[j], normal)
                        if vn < 0:
                            impulse = -(1+.88)*vn/(wi+wj)
                            vel[i] += normal*impulse*wi
                            vel[j] -= normal*impulse*wj
                            contacts += 1
            pos[:, 1] = np.maximum(pos[:, 1], radii)
    history = np.array(history)
    (d/"sim").mkdir(exist_ok=True)
    np.savez_compressed(d/"sim/spheres.npz", positions=history, radii=radii)
    layout.update(base_radius=.88*L, center=[0, 0, 0])
    report = dict(type="7 rigid spheres, gravity, pair collisions, floor and circular boundary", frames=N, substeps=12,
                  contacts=contacts, finite=bool(np.isfinite(history).all()),
                  min_floor_clearance=float(np.min(history[:, :, 1]-radii)), maximum_overlap=float(max(overlap)),
                  energy_ratio_max=float(max(energies)/energies[0]), units="arbitrary; art-directed low gravity")
    report["pass_"] = bool(report["finite"] and report["min_floor_clearance"] >= -1e-6
                           and report["maximum_overlap"] < L*.01 and report["energy_ratio_max"] < 1.05)
    return report


def cloth(cfg, d, cam, layout):
    L = layout["water_scale"]
    nx, nz = 33, 25
    fps = cam["fps"][0]/cam["fps"][1]
    xx, zz = np.meshgrid(np.linspace(-.8, .8, nx), np.linspace(-.6, .6, nz))
    rest = np.stack([xx, np.full_like(xx, .72), zz], -1).reshape(-1, 3)*L
    pin = np.array([0, nx-1, (nz-1)*nx, nz*nx-1])
    mass = np.ones(len(rest))
    mass[pin] = 0
    grid = np.arange(len(rest)).reshape(nz, nx)
    groups = []
    for dz, dx in [(0, 1), (1, 0), (1, 1), (1, -1)]:
        for parity in range(2):
            edges = [[grid[z, x], grid[z+dz, x+dx]] for z in range(nz) for x in range(nx)
                     if (x if dx else z) % 2 == parity and 0 <= x+dx < nx and z+dz < nz]
            e = np.array(edges)
            ii, jj = e.T
            groups.append((ii, jj, np.linalg.norm(rest[jj]-rest[ii], axis=1)))
    tri = []
    for z in range(nz-1):
        for x in range(nx-1):
            i = grid[z, x]
            tri += [[i, i+nx, i+1], [i+1, i+nx, i+nx+1]]
    tri = np.array(tri)
    pos, prev = rest.copy(), rest.copy()
    frames, strains = [], []
    dt = 1/(fps*10)
    for f in range(cam["count"]):
        for sub in range(10):
            t = (f*10+sub)*dt
            acc = np.zeros_like(pos)
            acc[:, 0] = .12*L
            acc[:, 2] = .08*L*np.cos(t*2)
            # Lift varies across the membrane so it never moves like a rigid sheet.
            acc[:, 1] = L*(-.5+2.1*np.sin(t*3.8+rest[:, 0]/L*3.2+rest[:, 2]/L*2.7))
            nxt = pos+(pos-prev)*.996+acc*dt*dt
            prev = pos.copy()
            pos = nxt
            for _ in range(10):
                for ii, jj, length in groups:
                    delta = pos[jj]-pos[ii]
                    dist = np.maximum(np.linalg.norm(delta, axis=1), 1e-10)
                    w = mass[ii]+mass[jj]
                    corr = delta*((dist-length)/dist/np.maximum(w, 1))[:, None]*.8
                    pos[ii] += corr*mass[ii, None]
                    pos[jj] -= corr*mass[jj, None]
                pos[pin] = rest[pin]
            pos[:, 1] = np.maximum(pos[:, 1], .03*L)
        frames.append(pos.copy())
        strains.append(max(float(np.max(abs(np.linalg.norm(pos[jj]-pos[ii], axis=1)/length-1))) for ii, jj, length in groups))
    frames = np.array(frames)
    (d/"sim").mkdir(exist_ok=True)
    np.savez_compressed(d/"sim/cloth.npz", positions=frames.astype("f4"), triangles=tri, pins=rest[pin].astype("f4"))
    layout.update(center=[0, 0, 0])
    report = dict(type="Four-corner pinned cloth, structural/shear PBD constraints, spatial wind", vertices=len(rest),
                  triangles=len(tri), frames=len(frames), substeps=10, constraint_iterations=10,
                  finite=bool(np.isfinite(frames).all()), maximum_strain=max(strains),
                  maximum_pin_error=float(abs(frames[:, pin]-rest[pin]).max()), minimum_height=float(frames[:, :, 1].min()),
                  units="arbitrary; art-directed wind, no measured material")
    report["pass_"] = bool(report["finite"] and report["maximum_strain"] < .15 and report["maximum_pin_error"] < 1e-8
                           and report["minimum_height"] > 0)
    return report


def main(cfg, work, args):
    d = shot_dir(cfg, work)
    cam = load_json(d/"Camera_Three_ground.json")
    layout = load_json(d/"layout.json")
    kind = cfg["kind"]
    simulate = dict(water=water, rigid_spheres=spheres, cloth=cloth).get(kind)
    if simulate is None:
        raise StepError(f"Unknown kind {kind}")
    report = simulate(cfg, d, cam, layout)
    report["pass"] = report.pop("pass_")
    save_json(d/"layout.json", layout)
    save_json(d/"sim/simulation-qa.json", report)
    print(report)
    if not report["pass"]:
        raise StepError("Simulation QA failed; see sim/simulation-qa.json")
    checkpoint(d, "simulated", next="python -m pipeline render "+cfg["name"])
