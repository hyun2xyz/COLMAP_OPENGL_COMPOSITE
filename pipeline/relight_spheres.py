"""Light integration for rigid-sphere shots: measured light, HDRI reflections, soft and contact shadows.

Reads light.json from `python -m pipeline light` (light groups measured from the plate, ambient
level, real floor colour around the object, plate grain, HDRI turn and scale). Per frame, in
native OpenGL at 2x supersampling:
  * one shadow map per measured light; soft edges grow with the blocker distance and the light's
    measured size (PCSS)
  * spheres + gold ring: diffuse from the measured lights, HDRI irradiance above the horizon and
    the real floor colour below it; GGX specular and HDRI reflections (turned to the plate's light);
    a subtle object-space texture that rotates with each sphere, so it never slides; self shadows
  * floor catcher: light occlusion weighted by each light's share of the floor's light, plus
    ambient occlusion of the spheres and the ring (contact shadows)
The composite is done the way After Effects does it: straight RGB + alpha over the plate in
display space. Shadows are black with alpha (a shadow catcher). CG gets the plate's measured grain
and a slight blur. Everything measured or assumed is written to render-report.json.
"""
from __future__ import annotations

import time

import cv2
import numpy as np

from .common import (Encoder, StepError, checkpoint, frame_path, gl_context, gl_info, gl_matrix, load_json, lookat,
                     new_output, projection, save_json, shot_dir)
from .render import sphere, torus

VERT = """#version 330
in vec3 vertex,normal;uniform mat4 vp,model;out vec3 wp,wn,op;
void main(){op=vertex;vec4 w=model*vec4(vertex,1.);wp=w.xyz;wn=mat3(model)*normal;gl_Position=vp*w;}"""
DEPTH_VERT = """#version 330
in vec3 vertex;uniform mat4 vp,model;void main(){gl_Position=vp*model*vec4(vertex,1.);}"""

COMMON = """
uniform sampler2D shadow0,shadow1;uniform mat4 lightVP0,lightVP1;uniform vec3 lightDir0,lightDir1,lightE0,lightE1;
uniform float lightTan0,lightTan1,lightExtent,lightDepth,searchDist;uniform int lightCount;
float hash12(vec2 p){vec3 q=fract(vec3(p.xyx)*.1031);q+=dot(q,q.yzx+33.33);return fract((q.x+q.y)*q.z);}
float soft(sampler2D map,mat4 lvp,vec3 pos,float tanSize,float bias){
 vec4 sc=lvp*vec4(pos,1.);vec3 q=sc.xyz/sc.w*.5+.5;
 if(any(lessThan(q.xy,vec2(0.)))||any(greaterThan(q.xy,vec2(1.)))||q.z>1.)return 1.;
 float rot=hash12(gl_FragCoord.xy)*6.283185;
 float search=max(tanSize*searchDist/(2.*lightExtent),2./2048.);
 float blockers=0.,sum=0.;
 for(int i=0;i<16;i++){float r=sqrt((float(i)+.5)/16.),a=float(i)*2.39996+rot;
  float z=texture(map,q.xy+vec2(cos(a),sin(a))*r*search).r;if(z<q.z-bias){sum+=z;blockers+=1.;}}
 if(blockers<.5)return 1.;
 float dz=(q.z-sum/blockers)*lightDepth;
 float radius=max(dz*tanSize/(2.*lightExtent),1.5/2048.);
 float lit=0.;
 for(int i=0;i<32;i++){float r=sqrt((float(i)+.5)/32.),a=float(i)*2.39996+rot;
  lit+=step(q.z-bias,texture(map,q.xy+vec2(cos(a),sin(a))*r*radius).r);}
 return lit/32.;
}
float visibility(int k,vec3 pos,vec3 N){
 if(k==0)return soft(shadow0,lightVP0,pos,lightTan0,.0015+.003*(1.-abs(dot(N,lightDir0))));
 return soft(shadow1,lightVP1,pos,lightTan1,.0015+.003*(1.-abs(dot(N,lightDir1))));
}
"""

OBJECT_FRAG = """#version 330
in vec3 wp,wn,op;
uniform vec3 eye,albedo,f0,floorRadiance;uniform mat3 objRot;
uniform float roughness,metal,texAmount,bumpAmount,texScale,seed,envYaw,envScale,envMaxLod;
uniform sampler2D hdr;
layout(location=0) out vec4 col;
"""+COMMON+"""
vec3 env(vec3 d,float lod){d=normalize(d);float u=(atan(d.z,d.x)+envYaw)/6.283185+.5;
 float v=asin(clamp(d.y,-1.,1.))/3.141593+.5;return textureLod(hdr,vec2(u,v),lod).rgb*envScale;}
float hash13(vec3 p){p=fract(p*.1031);p+=dot(p,p.zyx+31.32);return fract((p.x+p.y)*p.z);}
float vnoise(vec3 p){vec3 i=floor(p),f=fract(p);f=f*f*(3.-2.*f);
 return mix(mix(mix(hash13(i),hash13(i+vec3(1,0,0)),f.x),mix(hash13(i+vec3(0,1,0)),hash13(i+vec3(1,1,0)),f.x),f.y),
            mix(mix(hash13(i+vec3(0,0,1)),hash13(i+vec3(1,0,1)),f.x),mix(hash13(i+vec3(0,1,1)),hash13(i+vec3(1,1,1)),f.x),f.y),f.z);}
float fbm(vec3 p){float s=0.,a=.5;for(int i=0;i<4;i++){s+=a*vnoise(p);p*=2.03;a*=.5;}return s;}
vec3 irradiance(vec3 n){        // cosine-weighted: half the hemisphere is floor when n is horizontal
 return mix(env(n,envMaxLod),floorRadiance,clamp(.5-.5*n.y,0.,1.));}
vec3 reflected(vec3 d,float r){  // horizon blurs with roughness
 float w=.03+.5*r;return mix(env(d,r*envMaxLod*.85),floorRadiance,smoothstep(w,-w,d.y));}
void main(){
 vec3 V=normalize(eye-wp),N=normalize(wn);if(dot(N,V)<0.)N=-N;
 vec3 tp=op*texScale+seed;float n1=fbm(tp),n2=fbm(tp*2.7+11.);
 float e=.035;vec3 g=vec3(fbm(tp+vec3(e,0,0))-n1,fbm(tp+vec3(0,e,0))-n1,fbm(tp+vec3(0,0,e))-n1)/e;
 g=objRot*g;N=normalize(N-bumpAmount*(g-N*dot(g,N)));
 vec3 alb=albedo*(1.+texAmount*(n1-.5)*2.);
 float r=clamp(roughness+.1*(n2-.5),.06,1.);
 vec3 F0=mix(vec3(.04),f0,metal);
 float NdV=max(dot(N,V),1e-3);
 vec3 diffuse=vec3(0.),spec=vec3(0.);
 for(int k=0;k<2;k++){
  if(k>=lightCount)break;
  vec3 L=k==0?lightDir0:lightDir1;vec3 E=k==0?lightE0:lightE1;
  float NdL=max(dot(N,L),0.);if(NdL<=0.)continue;
  float vis=visibility(k,wp,N);
  vec3 H=normalize(L+V);float NdH=max(dot(N,H),0.),a=max(r*r,.04),a2=a*a,dd=NdH*NdH*(a2-1.)+1.;
  float D=a2/(3.141593*dd*dd);float kk=(r+1.)*(r+1.)/8.;
  float G=NdL/(NdL*(1.-kk)+kk)*NdV/(NdV*(1.-kk)+kk);
  vec3 F=F0+(1.-F0)*pow(1.-max(dot(H,V),0.),5.);
  diffuse+=alb/3.141593*E*NdL*vis;
  spec+=D*G*F/(4.*NdV*max(NdL,1e-3))*E*NdL*vis;
 }
 diffuse+=alb*irradiance(N);
 vec3 R=reflect(-V,N);
 vec3 Fe=F0+(max(vec3(1.-r),F0)-F0)*pow(1.-NdV,5.);
 spec+=reflected(R,r)*Fe;
 col=vec4(diffuse*(1.-metal)+spec,1.);
}"""

FLOOR_FRAG = """#version 330
in vec3 wp,wn,op;uniform vec4 occ[80];uniform int occCount;uniform float share0,share1,ambientShare,aoStrength;
layout(location=0) out vec4 col;
"""+COMMON+"""
void main(){
 vec3 up=vec3(0.,1.,0.);float blocked=0.;
 if(lightCount>0)blocked+=share0*(1.-visibility(0,wp,up));
 if(lightCount>1)blocked+=share1*(1.-visibility(1,wp,up));
 float ao=0.;
 for(int i=0;i<80;i++){if(i>=occCount)break;vec3 d=occ[i].xyz-wp;float l=length(d);
  ao+=occ[i].w*occ[i].w/(l*l)*max(d.y/l,0.);}
 ao=clamp(ao,0.,1.);
 col=vec4(max(1.-blocked-ambientShare*aoStrength*ao,.04),blocked,ao,1.);
}"""

PALETTE = [(.82, .30, .08), (.08, .65, .62), (.86, .78, .5), (.25, .45, .8), (.78, .12, .15), (.9, .9, .85), (.48, .16, .65)]


def rotations(positions, radii, seed):
    """Orientation of every sphere in every frame: rolling-like spin from its own motion."""
    rng = np.random.default_rng(seed)
    n, k = positions.shape[:2]
    out = np.zeros((n, k, 3, 3))
    for j in range(k):
        q, _ = np.linalg.qr(rng.standard_normal((3, 3)))
        out[0, j] = q
        for i in range(1, n):
            v = positions[i, j]-positions[i-1, j]
            axis = np.cross([0., 1., 0.], v)
            angle = np.linalg.norm(v)/radii[j]*.85
            if np.linalg.norm(axis) < 1e-9 or angle < 1e-9:
                out[i, j] = out[i-1, j]
                continue
            axis /= np.linalg.norm(axis)
            K = np.array([[0, -axis[2], axis[1]], [axis[2], 0, -axis[0]], [-axis[1], axis[0], 0]])
            rot = np.eye(3)+np.sin(angle)*K+(1-np.cos(angle))*K@K
            out[i, j] = rot@out[i-1, j]
    return out


def to_srgb(x):
    x = np.maximum(x, 0)
    knee = .82
    x = np.where(x < knee, x, knee+(1-knee)*(1-np.exp(-(x-knee)/(1-knee))))
    return np.where(x <= .0031308, 12.92*x, 1.055*x**(1/2.4)-.055)


def main(cfg, work, args):
    import moderngl as gl
    from .fetch import lighting_path
    from . import light as light_step
    d = shot_dir(cfg, work)
    rl = cfg.get("relight", {})
    if not (d/"light.json").is_file():
        print("light.json missing; measuring the light first", flush=True)
        light_step.main(cfg, work, args)
    lt = load_json(d/"light.json")
    stills = args.frames is not None and len(args.frames) > 0
    out = new_output(args.output or d/"relight/final")
    s = load_json(d/"Camera_Three_ground.json")
    layout = load_json(d/"layout.json")
    sim = np.load(d/"sim/spheres.npz")
    W, H, N = s["width"], s["height"], s["count"]
    fps = s["fps"][0]/s["fps"][1]
    rw, rh = W*2, H*2
    P = projection(s)
    base = layout["base_radius"]
    seed = int(rl.get("seed", 20261110))
    exposure = float(rl.get("exposure", 1.))
    lights = lt["lights"][:2]
    ctx = gl_context()
    obj = ctx.program(vertex_shader=VERT, fragment_shader=OBJECT_FRAG)
    flo = ctx.program(vertex_shader=VERT, fragment_shader=FLOOR_FRAG)
    dep = ctx.program(vertex_shader=DEPTH_VERT, fragment_shader="#version 330\nvoid main(){}")

    # Light space: orthographic boxes along each measured light direction.
    extent, near, far = base*2.2, .1, 30.
    ortho = np.diag([1/extent, 1/extent, -2/(far-near), 1.])
    ortho[2, 3] = -(far+near)/(far-near)
    shadow_maps, light_vps = [], []
    for k, light in enumerate(lights):
        direction = np.array(light["direction"], float)
        target = np.array([0., .3*base, 0.])
        light_vps.append(ortho@lookat(target+direction*15., target))
        tex = ctx.depth_texture((2048, 2048))
        tex.compare_func = ""
        tex.repeat_x = tex.repeat_y = False
        tex.filter = (gl.NEAREST, gl.NEAREST)
        shadow_maps.append((tex, ctx.framebuffer(depth_attachment=tex)))
    max_radius = float(rl.get("max_light_radius_deg", 12.))
    light_radius = [min(light["mean_spread_deg"]*1.5, max_radius) for light in lights]

    def put(prog, name, value):
        if name in prog:            # the GLSL compiler drops uniforms a shader does not use
            if name.startswith("lightVP"):
                gl_matrix(prog, name, value)
            else:
                prog[name] = value
    for prog in (obj, flo):
        put(prog, "lightCount", len(lights))
        put(prog, "lightExtent", extent)
        put(prog, "lightDepth", far-near)
        put(prog, "searchDist", base*1.5)
        for k, light in enumerate(lights):
            put(prog, f"shadow{k}", 5+k)
            put(prog, f"lightVP{k}", light_vps[k])
            put(prog, f"lightDir{k}", tuple(light["direction"]))
            put(prog, f"lightE{k}", tuple(np.array(light["rgb"])*light["irradiance"]*exposure))
            put(prog, f"lightTan{k}", float(np.tan(np.radians(light_radius[k]))))
    hdr = cv2.imread(str(lighting_path(cfg, work)), cv2.IMREAD_UNCHANGED)
    if hdr is None:
        raise StepError("HDRI missing. Run: python -m pipeline fetch "+cfg["name"])
    ht = ctx.texture((hdr.shape[1], hdr.shape[0]), 3, np.ascontiguousarray(hdr[::-1, :, ::-1], dtype="f4").tobytes(),
                     dtype="f4")
    ht.repeat_y = False
    ht.build_mipmaps()
    ht.filter = (gl.LINEAR_MIPMAP_LINEAR, gl.LINEAR)
    max_lod = float(np.floor(np.log2(max(hdr.shape[:2]))))
    obj["hdr"] = 3
    obj["envYaw"] = float(lt["hdri"]["yaw_rad"])
    obj["envScale"] = float(lt["hdri"]["scale"]*exposure)
    obj["envMaxLod"] = max_lod-1
    obj["floorRadiance"] = tuple(np.array(lt["floor_radiance_rgb"])*exposure)
    obj["texScale"] = 3.
    shares = [l.get("floor_fraction", 0.) for l in lights]
    flo["share0"] = float(shares[0]) if shares else 0.
    flo["share1"] = float(shares[1]) if len(shares) > 1 else 0.
    flo["ambientShare"] = float(max(1.-sum(shares), 0.))
    flo["aoStrength"] = float(rl.get("ao_strength", 1.))

    ring_mesh = torus(base, base*.09)
    ring_buf = ctx.buffer(ring_mesh.tobytes())
    sph_buf = ctx.buffer(sphere().tobytes())
    ring_obj = ctx.vertex_array(obj, [(ring_buf, "3f 3f", "vertex", "normal")])
    sph_obj = ctx.vertex_array(obj, [(sph_buf, "3f 3f", "vertex", "normal")])
    ring_dep = ctx.vertex_array(dep, [(ring_buf, "3f 12x", "vertex")])
    sph_dep = ctx.vertex_array(dep, [(sph_buf, "3f 12x", "vertex")])
    e = base*3.2
    plane = np.float32([[-e, .0005, -e, 0, 1, 0], [-e, .0005, e, 0, 1, 0], [e, .0005, -e, 0, 1, 0],
                        [e, .0005, -e, 0, 1, 0], [-e, .0005, e, 0, 1, 0], [e, .0005, e, 0, 1, 0]])
    floor_vao = ctx.vertex_array(flo, [(ctx.buffer(plane.tobytes()), "3f 3f", "vertex", "normal")])
    beauty = ctx.framebuffer([ctx.texture((rw, rh), 4, dtype="f4")], ctx.depth_renderbuffer((rw, rh)))
    floorfb = ctx.framebuffer([ctx.texture((rw, rh), 4, dtype="f4")])
    minor = base*.09
    ring_occ = [(base*np.cos(a), minor+.008, base*np.sin(a), minor) for a in np.linspace(0, 2*np.pi, 64, endpoint=False)]
    rots = rotations(sim["positions"], sim["radii"], seed)
    grain_std = np.array(lt["plate_grain_std_8bit"])/255.*float(rl.get("grain_scale", 1.))
    blur = float(rl.get("blur_sigma", .5))
    indices = args.frames if stills else range(N)
    enc = alphaenc = None
    if not stills:
        enc = Encoder(out/"Composite.mp4", W, H, fps)
        alphaenc = Encoder(out/"Alpha.mov", W, H, fps, alpha=True)
    stats, started = [], time.time()
    for i in indices:
        cam = np.array(s["frames"][i]["matrixWorld"]).reshape(4, 4).T
        vp = P@np.linalg.inv(cam)
        models = []
        for center, rad, rot in zip(sim["positions"][i], sim["radii"], rots[i]):
            m = np.eye(4)
            m[:3, :3] = rot*rad
            m[:3, 3] = center
            models.append((m, rot))
        # shadow maps
        ctx.enable(gl.DEPTH_TEST)
        ctx.disable(gl.CULL_FACE | gl.BLEND)
        for k in range(len(lights)):
            tex, fb = shadow_maps[k]
            fb.use()
            fb.clear(depth=1.)
            gl_matrix(dep, "vp", light_vps[k])
            gl_matrix(dep, "model", np.eye(4))
            ring_dep.render()
            for m, _ in models:
                gl_matrix(dep, "model", m)
                sph_dep.render()
        for k in range(len(lights)):
            shadow_maps[k][0].use(5+k)
        ht.use(3)
        # objects
        beauty.use()
        beauty.clear(0, 0, 0, 0, depth=1.)
        gl_matrix(obj, "vp", vp)
        obj["eye"] = tuple(cam[:3, 3])
        gl_matrix(obj, "model", np.eye(4))
        obj["objRot"] = tuple(np.eye(3).T.ravel())
        obj["albedo"] = (.5, .38, .2)
        obj["f0"] = (1., .71, .29)
        obj["metal"] = 1.
        obj["roughness"] = .3
        obj["texAmount"] = .06
        obj["bumpAmount"] = .012
        obj["seed"] = 3.
        ring_obj.render()
        for j, ((m, rot), paint) in enumerate(zip(models, PALETTE)):
            gl_matrix(obj, "model", m)
            obj["objRot"] = tuple(rot.T.ravel())
            obj["albedo"] = tuple(np.array(paint)**2.2)
            obj["f0"] = (.04, .04, .04)
            obj["metal"] = 0.
            obj["roughness"] = .2+.025*j
            obj["texAmount"] = .045
            obj["bumpAmount"] = .008
            obj["seed"] = 17.+j*5.3
            sph_obj.render()
        # floor catcher
        floorfb.use()
        floorfb.clear(1, 0, 0, 0)
        ctx.disable(gl.DEPTH_TEST)
        gl_matrix(flo, "vp", vp)
        gl_matrix(flo, "model", np.eye(4))
        occ = [(*c, r) for c, r in zip(sim["positions"][i], sim["radii"])]+ring_occ
        flo["occ"].write(np.float32(occ+[(0, 0, 0, 0)]*(80-len(occ))).tobytes())
        flo["occCount"] = len(occ)
        floor_vao.render()
        rgba = np.frombuffer(beauty.read(components=4, dtype="f4", alignment=1), np.float32).reshape(rh, rw, 4)[::-1]
        fl = np.frombuffer(floorfb.read(components=4, dtype="f4", alignment=1), np.float32).reshape(rh, rw, 4)[::-1]
        a = rgba[:, :, 3:].copy()
        cg = to_srgb(rgba[:, :, :3])*a                          # premultiplied, display space
        shadow_a = 1.-np.power(np.clip(fl[:, :, :1], 0, 1), 1/2.2)
        cg = cv2.resize(cg, (W, H), interpolation=cv2.INTER_AREA)
        a = cv2.resize(a, (W, H), interpolation=cv2.INTER_AREA)[:, :, None]
        shadow_a = cv2.resize(shadow_a, (W, H), interpolation=cv2.INTER_AREA)[:, :, None]
        if blur > 0:
            cg = cv2.GaussianBlur(cg, (0, 0), blur)
            a = cv2.GaussianBlur(a, (0, 0), blur)[:, :, None]
        total = a+shadow_a*(1-a)
        rng = np.random.default_rng(seed+i)
        luma = cv2.GaussianBlur(rng.standard_normal((H, W)).astype("f4"), (0, 0), .45)
        luma /= max(float(luma.std()), 1e-6)
        noise = luma[:, :, None]*grain_std+rng.standard_normal((H, W, 3)).astype("f4")*grain_std*.35
        cg = cg+noise*a
        plate = cv2.imread(str(frame_path(d, i)))[:, :, ::-1].astype("f4")/255.
        comp = np.uint8(np.clip((cg+plate*(1-total))*255+.5, 0, 255))
        straight = np.uint8(np.clip(np.concatenate([cg/np.maximum(total, 1e-6), total], 2)*255+.5, 0, 255))
        stats.append(dict(frame=i, cg_pixels=int((a > .05).sum()), shadow_pixels=int((shadow_a > .02).sum()),
                          max_shadow_alpha=float(shadow_a.max()), finite=bool(np.isfinite(rgba).all())))
        if enc:
            enc.write(comp)
            alphaenc.write(straight)
        if stills or i in (0, N//2, N-1):
            cv2.imwrite(str(out/f"frame_{i:03}.png"), comp[:, :, ::-1])
            cv2.imwrite(str(out/f"alpha_{i:03}.png"), straight[:, :, [2, 1, 0, 3]])
        if i % 24 == 0:
            print(f"relight {i}/{N}", flush=True)
    if enc:
        enc.finish()
        alphaenc.finish()
    report = dict(renderer="Native OpenGL via ModernGL (not browser WebGL)", gpu=gl_info(ctx), gl_error=ctx.error,
                  frames=len(stats), size=[W, H], fps=s["fps"], ssaa=2, kind="rigid_spheres",
                  light=dict(source="light.json, measured from the plate", lights=lights,
                             ambient_radiance=lt["ambient_radiance"], floor_radiance_rgb=lt["floor_radiance_rgb"],
                             clipped_gain=lt.get("clipped_gain"), exposure=exposure),
                  reflections=dict(hdri=str(lighting_path(cfg, work)), yaw_rad=lt["hdri"]["yaw_rad"],
                                   scale=lt["hdri"]["scale"], below_horizon="real floor colour from the plate"),
                  shadows=dict(method="one 2048 shadow map per light, PCSS soft edges from the light's measured spread; "
                                      "ambient occlusion of spheres and ring on the floor",
                               light_radius_deg=light_radius, max_light_radius_deg=max_radius,
                               floor_shares=shares, ao_strength=flo["aoStrength"].value),
                  texture="object-space fbm albedo/roughness/bump, rotates with each sphere",
                  integration=dict(blur_sigma=blur, grain_std_8bit=(grain_std*255).tolist(), seed=seed),
                  alpha="straight/unmatted: CG plus black shadow-catcher alpha",
                  stats=stats, elapsed_seconds=round(time.time()-started, 1),
                  limitations=lt["limitations"]+["Floor is the fitted plane: shadows also fall where real objects stand",
                                                 "No reflection of the CG in the glossy real floor"])
    save_json(out/"render-report.json", report)
    ctx.release()
    if not all(r["finite"] for r in stats):
        raise StepError("Non-finite pixels in the relight pass")
    if not stills:
        checkpoint(d, "relit", output=str(out), next=f"python -m pipeline qa-video {cfg['name']} --render {out}")
    print("Relight complete:", out)
