"""Cloth integration pass: one estimated sun, sky fill, real shadow map, blur and grain.

Keeps the solved camera, the PBD vertices and the plate. Adds:
  * warm directional sun + cool sky + faint ground bounce, in linear light
  * UV-locked woven fabric detail (cannot slide over the moving cloth)
  * 2048 shadow map of the actual cloth and posts -> self shadow + soft floor shadow
  * premultiplied 0.45 px blur of CG color AND coverage (no dark/bright outline)
  * seeded film grain only where the CG is, matched to the plate
The sun direction is estimated by eye from shadows in the plate; record it as such.
"""
from __future__ import annotations

import time

import cv2
import numpy as np

from .common import (Encoder, StepError, checkpoint, frame_path, gl_context, gl_info, gl_matrix, load_json, lookat,
                     new_output, projection, save_json, sha256, shot_dir)

VERT = """#version 330
in vec3 position,normal,tangent,bitangent;in vec2 texcoord;
uniform mat4 vp;out vec3 p,n,tu,tv;out vec2 uv;
void main(){p=position;n=normal;tu=tangent;tv=bitangent;uv=texcoord;gl_Position=vp*vec4(p,1.);}
"""
SHADOW_FUNCTION = """
uniform mat4 lightVP;uniform sampler2D shadowDepth;uniform vec3 sun;
float visibility(vec3 pos,vec3 N,float radius){
 vec4 sc=lightVP*vec4(pos,1.);vec3 q=sc.xyz/sc.w*.5+.5;
 if(any(lessThan(q,vec3(0.)))||any(greaterThan(q,vec3(1.))))return 1.;
 vec2 dx=dFdx(q.xy),dy=dFdy(q.xy);float zx=dFdx(q.z),zy=dFdy(q.z);
 float det=dx.x*dy.y-dx.y*dy.x;
 vec2 depthSlope=abs(det)>1e-14?vec2(zx*dy.y-zy*dx.y,dx.x*zy-dy.x*zx)/det:vec2(0.);
 depthSlope=clamp(depthSlope,vec2(-4.),vec2(4.));
 float bias=radius>2.?(.00013+.00010*(1.-abs(dot(N,sun)))):(.0008+.0004*(1.-abs(dot(N,sun))));
 float sum=0.;float weight=0.;
 for(int y=-2;y<=2;y++)for(int x=-2;x<=2;x++){
  float w=exp(-float(x*x+y*y)*.35);vec2 offset=vec2(x,y)*radius/2048.;
  float depth=texture(shadowDepth,q.xy+offset).r;
  sum+=w*step(q.z+dot(depthSlope,offset)-bias,depth);weight+=w;
 }return sum/weight;
}
"""
FRAG = """#version 330
in vec3 p,n,tu,tv;in vec2 uv;uniform vec3 eye;uniform int fabric;
layout(location=0) out vec4 beauty;layout(location=1) out vec4 diagnostics;
"""+SHADOW_FUNCTION+"""
float hash(vec2 a){return fract(sin(dot(a,vec2(127.1,311.7)))*43758.5453);}
float noise(vec2 a){vec2 i=floor(a),f=fract(a);f=f*f*(3.-2.*f);
 return mix(mix(hash(i),hash(i+vec2(1,0)),f.x),mix(hash(i+vec2(0,1)),hash(i+1.),f.x),f.y);}
void main(){
 vec3 N=normalize(n),V=normalize(eye-p);if(dot(N,V)<0.)N=-N;
 vec3 base=pow(vec3(.28,.265,.23),vec3(2.2));float cavity=1.;
 if(fabric==1){
  vec3 T=normalize(tu),B=normalize(tv);
  float phase=6.283185*(uv.x*5.5+uv.y*1.3+.35*noise(uv*4.));
  float phase2=6.283185*(uv.y*8.-uv.x*1.2+.3*noise(uv*5.));
  float slopeU=.12*cos(phase)-.025*cos(phase2);
  float slopeV=.035*cos(phase)+.085*cos(phase2);
  vec2 threads=uv*vec2(150.,115.);
  float filtered=1.-smoothstep(.3,.75,max(fwidth(threads.x),fwidth(threads.y)));
  slopeU+=.07*cos(threads.x*6.283185)*filtered;
  slopeV+=.07*cos(threads.y*6.283185)*filtered;
  N=normalize(N-T*slopeU-B*slopeV);
  float stripeDistance=abs(fract(uv.x*4.)-.5);
  float aa=max(fwidth(stripeDistance),.0001);
  float stripe=1.-smoothstep(.032-aa,.032+aa,stripeDistance);
  vec3 dye=mix(vec3(.66,.36,.20),vec3(.73,.66,.49),stripe);
  float grain=.94+.09*noise(uv*vec2(58.,43.));
  float weave=1.-.035*filtered*(.5+.5*sin(threads.x*6.283185)*sin(threads.y*6.283185));
  base=pow(dye,vec3(2.2))*grain*weave;
  cavity=.94+.06*(.5+.5*sin(phase)*sin(phase2));
 }
 float nl=max(dot(N,sun),0.);float vis=visibility(p,normalize(n),1.0);
 vec3 direct=vec3(1.06,1.0,.91)*1.65*nl*vis;
 vec3 sky=vec3(.155,.19,.255)*(.35+.65*max(N.y,0.));
 vec3 bounce=vec3(.055,.065,.035)*(.45+.55*max(-N.y,0.));
 float sheen=pow(1.-max(dot(N,V),0.),5.)*.045*nl*vis;
 beauty=vec4(base*(direct+(sky+bounce)*cavity)+vec3(1.,.9,.75)*sheen,1.);
 diagnostics=vec4(nl,vis,fabric==1?1.:0.,1.);
}
"""
FLOOR_VERT = """#version 330
in vec3 position;uniform mat4 vp;out vec3 p;
void main(){p=position;gl_Position=vp*vec4(position,1.);}
"""
FLOOR_FRAG = """#version 330
in vec3 p;out vec4 col;
"""+SHADOW_FUNCTION+"""
void main(){float v=visibility(p,vec3(0,1,0),3.2);col=vec4(vec3(1.-v),1.);}
"""


def to_srgb(value):
    return np.where(value <= .0031308, 12.92*value, 1.055*np.maximum(value, 0)**(1/2.4)-.055)


def posts(pins, L):
    mesh = []
    for anchor in pins:
        for j in range(24):
            ns = [np.array([np.cos(t), 0, np.sin(t)]) for t in (j*np.pi/12, (j+1)*np.pi/12)]
            a = np.array([anchor[0], 0, anchor[2]])+ns[0]*.012*L
            b = np.array([anchor[0], 0, anchor[2]])+ns[1]*.012*L
            up = np.array([0, anchor[1]+.015*L, 0])
            for pos, n in [(a, ns[0]), (a+up, ns[0]), (b, ns[1]), (b, ns[1]), (a+up, ns[0]), (b+up, ns[1])]:
                mesh.append([*pos, *n, 1, 0, 0, 0, 0, 1, 0, 0])
    return np.float32(mesh)


def main(cfg, work, args):
    import moderngl as gl
    if cfg["kind"] == "rigid_spheres":
        from .relight_spheres import main as spheres
        return spheres(cfg, work, args)
    if cfg["kind"] != "cloth" or "relight" not in cfg:
        raise StepError("relight supports cloth shots (with 'relight' settings) and rigid-sphere shots.")
    d = shot_dir(cfg, work)
    rl = cfg["relight"]
    sun = np.array(rl["sun"], float)
    sun /= np.linalg.norm(sun)
    out = new_output(args.output or d/"relight/final")
    camdata = load_json(d/"Camera_Three_ground.json")
    layout = load_json(d/"layout.json")
    data = np.load(d/"sim/cloth.npz")
    L = layout["water_scale"]
    W, H = camdata["width"], camdata["height"]
    rw, rh = W*2, H*2
    fps = camdata["fps"][0]/camdata["fps"][1]
    P = projection(camdata)
    ctx = gl_context()
    pr = ctx.program(vertex_shader=VERT, fragment_shader=FRAG)
    fp = ctx.program(vertex_shader=FLOOR_VERT, fragment_shader=FLOOR_FRAG)
    for prog in (pr, fp):
        prog["sun"] = tuple(sun)
        prog["shadowDepth"] = 0
    sp = ctx.program(vertex_shader="""#version 330
in vec3 position;uniform mat4 vp;void main(){gl_Position=vp*vec4(position,1.);}""",
                     fragment_shader="#version 330\nvoid main(){}")
    depth = ctx.depth_texture((2048, 2048))
    depth.compare_func = ""
    depth.repeat_x = depth.repeat_y = False
    depth.filter = (gl.NEAREST, gl.NEAREST)
    sf = ctx.framebuffer(depth_attachment=depth)
    near, far, extent = .1, 8., 1.4
    ortho = np.diag([1/extent, 1/extent, -2/(far-near), 1.])
    ortho[2, 3] = -(far+near)/(far-near)
    lightVP = ortho@lookat(sun*3.+np.array([0, .2, 0]), np.array([0., .2, 0.]))
    for prog in (pr, fp):
        gl_matrix(prog, "lightVP", lightVP)
    gl_matrix(sp, "vp", lightVP)
    beauty = ctx.framebuffer([ctx.texture((rw, rh), 4, dtype="f2"), ctx.texture((rw, rh), 4, dtype="f2")],
                             ctx.depth_renderbuffer((rw, rh)))
    floorfb = ctx.framebuffer([ctx.texture((rw, rh), 4, dtype="f2")], ctx.depth_renderbuffer((rw, rh)))
    pb = ctx.buffer(posts(data["pins"], L).tobytes())
    attr, names = "3f 3f 3f 3f 2f", ["position", "normal", "tangent", "bitangent", "texcoord"]
    pv = ctx.vertex_array(pr, [(pb, attr, *names)])
    ps = ctx.vertex_array(sp, [(pb, "3f 44x", "position")])
    r = 1.5
    floor = np.float32([[-r, .0005, -r], [-r, .0005, r], [r, .0005, -r], [r, .0005, -r], [-r, .0005, r], [r, .0005, r]])
    fv = ctx.vertex_array(fp, [(ctx.buffer(floor.tobytes()), "3f", "position")])
    u, v = np.meshgrid(np.linspace(0, 1, 33), np.linspace(0, 1, 25))
    uv = np.stack([u, v], -1).reshape(-1, 2)
    tri = data["triangles"]
    stills_only = args.frames is not None and len(args.frames) > 0
    indices = args.frames if stills_only else range(camdata["count"])
    enc = None if stills_only else Encoder(out/"Composite.mp4", W, H, fps)
    records, started = [], time.time()
    for i in indices:
        vertices = data["positions"][i]
        normal = np.zeros_like(vertices)
        fn = np.cross(vertices[tri[:, 1]]-vertices[tri[:, 0]], vertices[tri[:, 2]]-vertices[tri[:, 0]])
        for k in range(3):
            np.add.at(normal, tri[:, k], fn)
        normal /= np.maximum(np.linalg.norm(normal, axis=1, keepdims=True), 1e-10)
        grid = vertices.reshape(25, 33, 3)
        tu = np.gradient(grid, axis=1).reshape(-1, 3)
        tv = np.gradient(grid, axis=0).reshape(-1, 3)
        tu /= np.linalg.norm(tu, axis=1, keepdims=True)
        tv /= np.linalg.norm(tv, axis=1, keepdims=True)
        mesh = np.float32(np.concatenate([vertices, normal, tu, tv, uv], 1)[tri.ravel()])
        cb = ctx.buffer(mesh.tobytes())
        cv = ctx.vertex_array(pr, [(cb, attr, *names)])
        cs = ctx.vertex_array(sp, [(cb, "3f 44x", "position")])
        cam = np.array(camdata["frames"][i]["matrixWorld"]).reshape(4, 4).T
        vp = P@np.linalg.inv(cam)
        gl_matrix(pr, "vp", vp)
        gl_matrix(fp, "vp", vp)
        pr["eye"] = tuple(cam[:3, 3])
        ctx.enable(gl.DEPTH_TEST)
        ctx.disable(gl.CULL_FACE | gl.BLEND)
        ctx.depth_func = "<"
        sf.use()
        sf.clear(depth=1.)
        cs.render()
        ps.render()
        depth.use(0)
        beauty.use()
        beauty.clear(0, 0, 0, 0, depth=1.)
        pr["fabric"] = 0
        pv.render()
        pr["fabric"] = 1
        cv.render()
        floorfb.use()
        floorfb.clear(0, 0, 0, 0, depth=1.)
        fv.render()
        rgba = np.frombuffer(beauty.read(components=4, dtype="f4", alignment=1, attachment=0), np.float32).reshape(rh, rw, 4)[::-1].copy()
        diag = np.frombuffer(beauty.read(components=4, dtype="f4", alignment=1, attachment=1), np.float32).reshape(rh, rw, 4)[::-1]
        shadow = np.frombuffer(floorfb.read(components=4, dtype="f4", alignment=1), np.float32).reshape(rh, rw, 4)[::-1, :, 0]
        rgba[:, :, :3] *= rgba[:, :, 3:]
        rgba = cv2.resize(rgba, (W, H), interpolation=cv2.INTER_AREA)
        diag = cv2.resize(diag, (W, H), interpolation=cv2.INTER_AREA)
        shadow = cv2.resize(shadow, (W, H), interpolation=cv2.INTER_AREA)
        plate = cv2.imread(str(frame_path(d, i)))[:, :, ::-1]
        pl = plate.astype("f4")/255.
        linear = np.where(pl <= .04045, pl/12.92, ((pl+.055)/1.055)**2.4)
        floor_color = linear*(1.-shadow[:, :, None]*np.array([.76, .72, .63]))
        softened = cv2.GaussianBlur(rgba, (0, 0), rl.get("blur_sigma", .45), borderType=cv2.BORDER_REPLICATE)
        comp = softened[:, :, :3]+floor_color*(1.-softened[:, :, 3:])
        srgb = to_srgb(comp)*255
        rng = np.random.default_rng(rl.get("seed", 20261106)+i)
        grain = cv2.GaussianBlur(rng.standard_normal((H, W)).astype("f4"), (0, 0), .35)
        grain *= rl.get("grain_luma", .8)/max(float(grain.std()), 1e-6)
        chroma = rng.standard_normal((H, W, 3)).astype("f4")*rl.get("grain_chroma", .18)
        result = np.uint8(np.clip(srgb+(grain[:, :, None]+chroma)*softened[:, :, 3:]+.5, 0, 255))
        clothmask = diag[:, :, 2] > .95
        mask = rgba[:, :, 3] > .05
        lum = result.astype("f4")@np.array([.2126, .7152, .0722])
        records.append(dict(frame=i, cloth_pixels=int(clothmask.sum()),
                            cloth_luma_std=float(lum[clothmask].std()) if clothmask.any() else 0.,
                            self_shadow_pixels=int((diag[:, :, 1][clothmask] < .8).sum()),
                            visible_ground_shadow_pixels=int(((shadow > .1) & ~mask).sum()),
                            finite=bool(np.isfinite(comp).all())))
        if enc:
            enc.write(result)
        if stills_only or i in (0, camdata["count"]//2, camdata["count"]-1):
            cv2.imwrite(str(out/f"frame_{i:03}.png"), result[:, :, ::-1])
            cv2.imwrite(str(out/f"shadow_{i:03}.png"), np.uint8(np.clip(shadow*255, 0, 255)))
        if i % 24 == 0:
            print(f"relight {i}/{camdata['count']}", flush=True)
        cv.release(), cs.release(), cb.release()
    if enc:
        enc.finish()
    report = dict(renderer="Native OpenGL via ModernGL", gpu=gl_info(ctx), gl_error=ctx.error, frames=len(records),
                  size=[W, H], fps=fps, ssaa=2, sun_direction_world=sun.tolist(), sun_inference=rl.get("sun_note"),
                  integration=dict(blur_sigma=rl.get("blur_sigma", .45), grain_luma=rl.get("grain_luma", .8),
                                   grain_chroma=rl.get("grain_chroma", .18), seed=rl.get("seed", 20261106)),
                  lighting="Warm directional sun + cool sky fill + faint ground bounce; no new HDRI",
                  shadows="2048 depth map of the actual cloth/posts; PCF self shadow and soft floor shadow",
                  unchanged_inputs={n: sha256(d/n) for n in ("sim/cloth.npz", "Camera_Three_ground.json")},
                  stats=records, elapsed_seconds=round(time.time()-started, 1),
                  limitations=["Sun direction/elevation/intensity estimated by eye", "Floor is the fitted local plane, no terrain relief",
                               "Weave and wrinkles are shading detail, not simulated geometry"])
    save_json(out/"render-report.json", report)
    ctx.release()
    if not all(r["finite"] for r in records):
        raise StepError("Non-finite pixels in the relight pass")
    if not stills_only:
        checkpoint(d, "relit", output=str(out), next=f"play {out/'Composite.mp4'} start to end")
    print("Relight complete:", out)
