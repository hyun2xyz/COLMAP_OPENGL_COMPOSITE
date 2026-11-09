"""Native OpenGL composite (ModernGL, no browser WebGL).

Per frame: the solved camera gives view x projection, the plate frame is uploaded
as a texture, CG is drawn with depth testing at 1.5x supersampling, then
downsampled with premultiplied alpha. Outputs:
  Composite.mp4  CG over the plate (H.264, BT.709)
  Alpha.mov      straight/unmatted RGBA (ProRes 4444) for After Effects
Water refraction samples THIS plate, so its alpha pass only fits this shot.

--shadow analytic  soft blob/ring shadows (original accepted shots)
--shadow ring      25 projected silhouettes of the actual ring mesh, side light (water final)
--shadow none      no floor shadow
"""
from __future__ import annotations

import time

import cv2
import numpy as np

from .common import (ROOT, StepError, Encoder, checkpoint, frame_path, gl_context, gl_info, gl_matrix,
                     load_json, new_output, projection, save_json, shot_dir)


def torus(major, minor):
    vertices = []

    def v(a, b):
        n = np.array([np.cos(a)*np.cos(b), np.sin(b), np.sin(a)*np.cos(b)])
        p = np.array([major*np.cos(a), minor+.008, major*np.sin(a)])+minor*n
        return [*p, *n]
    da, db = 2*np.pi/96, 2*np.pi/12
    for i in range(96):
        for j in range(12):
            a, b = i*da, j*db
            vertices += [v(a, b), v(a, b+db), v(a+da, b), v(a+da, b), v(a, b+db), v(a+da, b+db)]
    return np.float32(vertices)


def sphere():
    data = []

    def v(a, b):
        n = [np.sin(b)*np.cos(a), np.cos(b), np.sin(b)*np.sin(a)]
        return n+n
    da, db = 2*np.pi/48, np.pi/24
    for i in range(48):
        for j in range(24):
            a, b = i*da, j*db
            data += [v(a, b), v(a+da, b), v(a, b+db), v(a, b+db), v(a+da, b), v(a+da, b+db)]
    return np.float32(data)


class RingShadow:
    """Static ring shadow: 25 native OpenGL silhouettes of the real torus under a sampled area light."""

    def __init__(self, ctx, mesh, radius, light):
        import moderngl as gl
        self.ctx = ctx
        self.extent = radius*1.85
        size = 1024
        light = np.array(light, dtype="f4")
        cast = ctx.program(vertex_shader="""#version 330
in vec3 vertex; uniform vec3 light; uniform float extent;
void main(){vec2 floorPoint=vertex.xz-light.xz*(vertex.y/light.y);gl_Position=vec4(floorPoint/extent,0.,1.);}""",
                           fragment_shader="""#version 330
out float value; void main(){value=1.;}""")
        cast["extent"] = self.extent
        b = ctx.buffer(mesh.tobytes())
        v = ctx.vertex_array(cast, [(b, "3f 12x", "vertex")])
        tex = ctx.texture((size, size), 1)
        fb = ctx.framebuffer([tex])
        ctx.disable(gl.DEPTH_TEST | gl.CULL_FACE | gl.BLEND)
        masks, weights = [], []
        for x in [-2, -1, 0, 1, 2]:
            for z in [-2, -1, 0, 1, 2]:
                cast["light"] = tuple(light+[x*.17, 0, z*.13])
                fb.use()
                fb.clear(0, 0, 0, 0)
                v.render()
                masks.append(np.frombuffer(fb.read(components=1, alignment=1), np.uint8).reshape(size, size)/255.)
                weights.append(np.exp(-(x*x+z*z)/3.2))
        alpha = np.average(np.stack(masks), axis=0, weights=np.array(weights)).astype("f4")
        alpha = cv2.GaussianBlur(alpha, (5, 5), .85)
        self.map = ctx.texture((size, size), 1, alpha.tobytes(), dtype="f4")
        self.map.repeat_x = self.map.repeat_y = False
        self.map.filter = (gl.LINEAR, gl.LINEAR)
        self.program = ctx.program(vertex_shader="""#version 330
in vec3 vertex; uniform mat4 vp; uniform float extent; out vec2 uv;
void main(){uv=vertex.xz/(2.*extent)+.5;gl_Position=vp*vec4(vertex,1.);}""", fragment_shader="""#version 330
in vec2 uv;uniform sampler2D shadowMap;out vec4 col;
void main(){float visibility=texture(shadowMap,uv).r;float alpha=1.-pow(1.-.72*visibility,1./2.2);col=vec4(0.,0.,0.,alpha);}""")
        self.program["extent"] = self.extent
        self.program["shadowMap"] = 4
        e = self.extent
        plane = np.float32([[-e, .001, -e], [-e, .001, e], [e, .001, -e], [e, .001, -e], [-e, .001, e], [e, .001, e]])
        self.buffer = ctx.buffer(plane.tobytes())
        self.vao = ctx.vertex_array(self.program, [(self.buffer, "3f", "vertex")])
        self.metadata = dict(method="25 projected silhouettes of the actual torus, area-light average",
                             direction_toward_light=light.tolist(), samples=25, lighting_measured=False)
        v.release(), b.release(), cast.release(), fb.release(), tex.release()

    def render(self, vp):
        self.map.use(4)
        gl_matrix(self.program, "vp", vp)
        self.vao.render()


def render(cfg, d, out, frames=None, shadow=None, ssaa=1.5):
    import moderngl as gl
    s = load_json(d/"Camera_Three_ground.json")
    layout = load_json(d/"layout.json")
    kind = layout["kind"]
    opts = cfg.get("render", {})
    shadow = shadow or opts.get("shadow", "analytic")
    key = tuple(opts.get("key_light", [-.4, 1., .3]))
    sim = np.load(d/"sim/spheres.npz") if kind == "rigid_spheres" else None
    clothdata = np.load(d/"sim/cloth.npz") if kind == "cloth" else None
    W, H = s["width"], s["height"]
    rw, rh = int(W*ssaa), int(H*ssaa)
    fps = s["fps"][0]/s["fps"][1]
    ctx = gl_context()
    print(gl_info(ctx), flush=True)
    shaders = ROOT/"Water/shaders"
    vs = (shaders/"surface.vert").read_text()
    back = ctx.program(vertex_shader=vs, fragment_shader=(shaders/"exit.frag").read_text())
    front = ctx.program(vertex_shader=vs, fragment_shader=(shaders/"water.frag").read_text())
    for i, n in enumerate(["positions", "normals", "plate", "hdr"]):
        front[n] = i
    front["dim"] = (float(rw), float(rh))
    gold = ctx.program(vertex_shader=vs, fragment_shader="""#version 330
in vec3 p,n; uniform vec3 eye,paint,key;uniform float reflectionLod;uniform sampler2D hdr;out vec4 col;
vec3 env(vec3 d){d=normalize(d);return textureLod(hdr,vec2(atan(d.z,d.x)/6.283185+.5,asin(clamp(d.y,-1.,1.))/3.141593+.5),reflectionLod).rgb;}
void main(){vec3 N=normalize(n),V=normalize(eye-p);if(dot(N,V)<0.)N=-N;vec3 R=reflect(-V,N);
vec3 c=env(R)*paint*.55+paint*vec3(.2413793,.2205882,.2083333)*(max(dot(N,normalize(key)),0.)+.3);
c=c/(c+.65);col=vec4(pow(c,vec3(1./2.2)),1.);}""")
    gold["hdr"] = 3
    gold["key"] = key
    gold["paint"] = (.58, .34, .12)
    gold["reflectionLod"] = 0.
    blob = ctx.program(vertex_shader=vs, fragment_shader="""#version 330
in vec3 p,n;uniform float radius,opacity,ringRadius;uniform vec2 centerXZ;out vec4 col;
void main(){float dist=length(p.xz-centerXZ);float q=(ringRadius>0.?abs(dist-ringRadius):dist)/radius;
float a=opacity*(.24*exp(-q*q*2.5)+.16*exp(-q*q*18.));col=vec4(0.,0.,0.,a);}""")

    def tex(comps, dtype="f1"):
        t = ctx.texture((rw, rh), comps, dtype=dtype)
        t.repeat_x = t.repeat_y = False
        t.filter = (gl.LINEAR, gl.LINEAR)
        return t
    exittex = [tex(4, "f2"), tex(4, "f2")]
    exitfb = ctx.framebuffer(exittex, ctx.depth_renderbuffer((rw, rh)))
    color = tex(4)
    beauty = ctx.framebuffer([color], ctx.depth_renderbuffer((rw, rh)))
    hdr = cv2.imread(layout["hdri"], cv2.IMREAD_UNCHANGED)
    if hdr is None:
        raise StepError("HDRI missing. Run: python -m pipeline fetch "+cfg["name"])
    ht = ctx.texture((hdr.shape[1], hdr.shape[0]), 3, np.ascontiguousarray(hdr[::-1, :, ::-1], dtype="f2").tobytes(), dtype="f2")
    ht.repeat_y = False
    ht.filter = (gl.LINEAR, gl.LINEAR)
    ht.build_mipmaps()
    if kind == "water":
        ht.filter = (gl.LINEAR, gl.LINEAR)
    base = torus(layout["base_radius"], layout["base_radius"]*.09)
    gv = ctx.vertex_array(gold, [(ctx.buffer(base.tobytes()), "3f 3f", "vertex", "normal")])
    sv = ctx.vertex_array(gold, [(ctx.buffer(sphere().tobytes()), "3f 3f", "vertex", "normal")]) if kind == "rigid_spheres" else None
    r = layout["water_scale"]*(1.2 if kind == "rigid_spheres" else .9)
    plane = np.float32([[-r, .001, -r, 0, 1, 0], [-r, .001, r, 0, 1, 0], [r, .001, -r, 0, 1, 0],
                        [r, .001, -r, 0, 1, 0], [-r, .001, r, 0, 1, 0], [r, .001, r, 0, 1, 0]])
    pv = ctx.vertex_array(blob, [(ctx.buffer(plane.tobytes()), "3f 12x", "vertex")])
    blob["radius"] = r*.6
    blob["centerXZ"] = (0., 0.)
    blob["opacity"] = 1.
    blob["ringRadius"] = 0.
    ring = RingShadow(ctx, base, layout["base_radius"], key) if kind == "water" and shadow == "ring" else None
    programs = [back, front, gold, blob]
    if kind == "cloth":
        clothpr = ctx.program(vertex_shader=vs, fragment_shader="""#version 330
in vec3 p,n;uniform vec3 eye;uniform sampler2D hdr;uniform float scale;out vec4 col;
void main(){vec3 N=normalize(n);if(dot(N,eye-p)<0.)N=-N;
float light=max(dot(N,normalize(vec3(.6,1.,-.4))),0.);
float weave=.96+.04*sin(p.x/scale*360.)*sin(p.z/scale*360.);
float stripe=smoothstep(.035,.05,abs(fract((p.x/scale+.8)*2.5)-.5));
vec3 dye=mix(vec3(.72,.59,.32),vec3(.64,.17,.035),stripe);
vec3 env=textureLod(hdr,vec2(.5,.85),6.).rgb;
vec3 c=dye*(.3+.7*light)*weave+min(env,vec3(1.))*.05;
col=vec4(pow(c,vec3(1./2.2)),1.);}""")
        clothpr["hdr"] = 3
        clothpr["scale"] = layout["water_scale"]
        clothshadow = ctx.program(vertex_shader=vs, fragment_shader="""#version 330
out vec4 col;void main(){col=vec4(0.,0.,0.,.32);}""")
        programs += [clothpr, clothshadow]
        posts, L = [], layout["water_scale"]
        for anchor in clothdata["pins"]:
            for j in range(24):
                t0, t1 = j*2*np.pi/24, (j+1)*2*np.pi/24
                n0, n1 = np.array([np.cos(t0), 0, np.sin(t0)]), np.array([np.cos(t1), 0, np.sin(t1)])
                p0 = np.array([anchor[0], 0, anchor[2]])+n0*.012*L
                p1 = np.array([anchor[0], 0, anchor[2]])+n1*.012*L
                up = np.array([0, anchor[1]+.015*L, 0])
                posts += [[*p0, *n0], [*(p0+up), *n0], [*p1, *n1], [*p1, *n1], [*(p0+up), *n0], [*(p1+up), *n1]]
        postv = ctx.vertex_array(gold, [(ctx.buffer(np.float32(posts).tobytes()), "3f 3f", "vertex", "normal")])
    P = projection(s)
    M = np.eye(4)
    M[:3, :3] *= layout["water_scale"]
    M[:3, 3] = layout["center"]
    indices = frames if frames else range(s["count"])
    enc = alphaenc = None
    if not frames:
        enc = Encoder(out/"Composite.mp4", W, H, fps)
        alphaenc = Encoder(out/"Alpha.mov", W, H, fps, alpha=True)
    stats, start = [], time.time()
    for i in indices:
        cam = np.array(s["frames"][i]["matrixWorld"]).reshape(4, 4).T
        vp = P@np.linalg.inv(cam)
        plate = cv2.imread(str(frame_path(d, i)))[:, :, ::-1]
        pt = ctx.texture((W, H), 3, np.ascontiguousarray(plate[::-1]).tobytes())
        pt.repeat_x = pt.repeat_y = False
        pt.filter = (gl.LINEAR, gl.LINEAR)
        for pr in programs:
            gl_matrix(pr, "vp", vp)
            gl_matrix(pr, "crop", np.eye(4))
            gl_matrix(pr, "model", M if pr in (back, front) else np.eye(4))
            if "eye" in pr:
                pr["eye"] = tuple(cam[:3, 3])
        if kind == "water":
            vb = ctx.buffer((d/f"sim/surfaces/{i:03}.bin").read_bytes())
            vas = [ctx.vertex_array(pr, [(vb, "3f 3f", "vertex", "normal")]) for pr in (back, front)]
            ctx.enable(gl.DEPTH_TEST | gl.CULL_FACE)
            ctx.disable(gl.BLEND)
            ctx.cull_face = "front"
            exitfb.use()
            exitfb.clear(0, 0, 0, 0, depth=1)
            vas[0].render()
        ctx.enable(gl.DEPTH_TEST)
        ctx.disable(gl.BLEND)
        beauty.use()
        beauty.clear(0, 0, 0, 0, depth=1)
        ctx.disable(gl.CULL_FACE)
        if kind == "water":
            if shadow == "analytic":
                pv.render()
            elif shadow == "ring":
                ring.render(vp)
        elif kind == "rigid_spheres":
            ctx.enable(gl.BLEND)
            ctx.blend_func = (gl.SRC_ALPHA, gl.ONE_MINUS_SRC_ALPHA, gl.ONE, gl.ONE_MINUS_SRC_ALPHA)
            ctx.depth_func = "<="
            if shadow != "none":
                blob["centerXZ"] = (0., 0.)
                blob["radius"] = float(layout["base_radius"]*.13)
                blob["ringRadius"] = float(layout["base_radius"])
                blob["opacity"] = 1.
                pv.render()
                blob["ringRadius"] = 0.
                for center, rad in zip(sim["positions"][i], sim["radii"]):
                    blob["centerXZ"] = tuple(center[[0, 2]]+np.array([.22, .15])*center[1])
                    blob["radius"] = float(rad*(1+.3*center[1]/rad))
                    blob["opacity"] = float(1.3/(1+.35*center[1]/rad))
                    pv.render()
            ctx.disable(gl.BLEND)
            ctx.depth_func = "<"
        else:
            vertices, tri = clothdata["positions"][i], clothdata["triangles"]
            normals = np.zeros_like(vertices)
            fn = np.cross(vertices[tri[:, 1]]-vertices[tri[:, 0]], vertices[tri[:, 2]]-vertices[tri[:, 0]])
            for corner in range(3):
                np.add.at(normals, tri[:, corner], fn)
            normals /= np.maximum(np.linalg.norm(normals, axis=1)[:, None], 1e-10)
            mesh = np.concatenate([vertices[tri.ravel()], normals[tri.ravel()]], axis=1).astype("f4")
            cb = ctx.buffer(mesh.tobytes())
            cv = ctx.vertex_array(clothpr, [(cb, "3f 3f", "vertex", "normal")])
            if shadow != "none":
                projected = np.concatenate([mesh, np.float32(posts)])
                projected[:, 0] -= .6*projected[:, 1]
                projected[:, 2] += .4*projected[:, 1]
                projected[:, 1] = .002
                csb = ctx.buffer(projected.tobytes())
                csv = ctx.vertex_array(clothshadow, [(csb, "3f 12x", "vertex")])
                csv.render()
                csv.release(), csb.release()
        ht.use(3)
        gold["paint"] = (.58, .34, .12)
        gold["reflectionLod"] = 0. if kind == "water" else 2.
        if kind != "cloth":
            gv.render()
        else:
            gold["paint"] = (.25, .23, .20)
            gold["reflectionLod"] = 5.
            postv.render()
            cv.render()
            cv.release(), cb.release()
        if kind == "water":
            for j, t in enumerate([*exittex, pt, ht]):
                t.use(j)
            ctx.enable(gl.CULL_FACE)
            ctx.cull_face = "back"
            vas[1].render()
        elif kind == "rigid_spheres":
            colors = [(.82, .30, .08), (.08, .65, .62), (.86, .78, .5), (.25, .45, .8), (.78, .12, .15), (.9, .9, .85), (.48, .16, .65)]
            gold["reflectionLod"] = 4.
            for center, rad, paint in zip(sim["positions"][i], sim["radii"], colors):
                obj = np.eye(4)
                obj[:3, :3] *= rad
                obj[:3, 3] = center
                gl_matrix(gold, "model", obj)
                gold["paint"] = paint
                sv.render()
        rgba = np.frombuffer(beauty.read(components=4, alignment=1), np.uint8).reshape(rh, rw, 4)[::-1]
        # Premultiply before downsampling, then return a genuine straight-alpha pass.
        f = rgba.astype("f")/255.
        f[:, :, :3] *= f[:, :, 3:]
        f = cv2.resize(f, (W, H), interpolation=cv2.INTER_AREA)
        al = f[:, :, 3:]
        rgb = np.divide(f[:, :, :3], np.maximum(al, 1e-8))
        straight = np.uint8(np.clip(np.concatenate([rgb, al], 2)*255, 0, 255))
        composite = np.uint8(np.clip(rgb*255*al+plate*(1-al), 0, 255))
        occupied = al[:, :, 0] > .05
        ys, xs = np.where(occupied)
        stats.append(dict(frame=i, alpha_pixels=int(occupied.sum()),
                          bounds=[int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max())] if len(xs) else None))
        if enc:
            enc.write(composite)
            alphaenc.write(straight)
        if frames or i in (0, s["count"]//2, s["count"]-1):
            cv2.imwrite(str(out/f"frame_{i:03}.png"), composite[:, :, ::-1])
            cv2.imwrite(str(out/f"alpha_{i:03}.png"), straight[:, :, [2, 1, 0, 3]])
        if i % 24 == 0:
            print(f"frame {i}/{s['count']}", flush=True)
        if kind == "water":
            for v in vas:
                v.release()
            vb.release()
        pt.release()
    if enc:
        enc.finish()
        alphaenc.finish()
    report = dict(renderer="Native OpenGL via ModernGL (not browser WebGL)", gpu=gl_info(ctx), frames=len(stats),
                  size=[W, H], fps=s["fps"], ssaa=ssaa, shadow=shadow, key_light=list(key),
                  elapsed_seconds=round(time.time()-start, 1), hdri=layout["hdri"], kind=kind,
                  ring_shadow=ring.metadata if ring else None, stats=stats,
                  alpha="straight/unmatted" + ("; water refraction embeds this plate" if kind == "water" else ""),
                  limitations=["Separate CC0 HDRI, not lighting captured with the plate",
                               "Art-directed key light and shadows", "Scale normalized by an assumed camera height"])
    save_json(out/"render-report.json", report)
    ctx.release()
    return report


def main(cfg, work, args):
    d = shot_dir(cfg, work)
    if not args.output and not args.frames and (d/"render/final/render-report.json").is_file():
        print("already rendered:", d/"render/final", "(pass --output NEW_FOLDER to render again)")
        return
    out = new_output(args.output or d/"render"/"final")
    render(cfg, d, out, frames=args.frames, shadow=args.shadow, ssaa=args.ssaa)
    if not args.frames:
        checkpoint(d, "rendered", output=str(out), next=f"python -m pipeline qa-video {cfg['name']} --render {out}")
    print("Render complete:", out)
