"""Space distortion: the air the arms sweep through bends the picture like hot glass (ModernGL, native OpenGL 3.3).

    python render.py [piece_dir] --still 40,90,140      quick look -> work/still_sheet.jpg
    python render.py [piece_dir]                        all frames -> work/comp, work/fx, work/field, then out/*.mp4

Per frame, three passes:
  1. trail mesh: for every limb chain in piece.json trail.limbs (MediaPipe ids, e.g. elbow-wrist-index) the chain's
     positions over the last trail.age_frames frames, in PLANE coordinates (trails.py), become a swept ribbon. It is
     projected with THIS frame's solved camera, so where the arm was stays put in the scene while the camera slides.
     Each fragment adds (motion vector * w, w, w^2) into a half-resolution float target; w = speed * age fade * edge fade.
  2. separable blur of that field (soft, lens-like gradients).
  2b. outline pulses: at limb hits (sharp stops) a band hugging the silhouette near that joint moves outward; it is added
     to the same field (outward motion), using a distance field from the current MediaPipe mask.
  2c. the trail air is kept close to the body: it fades with distance from the current silhouette (distort.near_m).
  2d. sound waves (RING_FS, half res): the strongest hits (ring.min_hit, one per ring.gap_frames) send ring.echoes waves
     out of the BODY: the front is the smooth-min distance to the body at the hit (15 joint capsules, parts far from the
     hitting joint start late), wobbled per direction with noise and weighted by two random direction lobes. Each pixel is
     measured at a 3D point on a proxy (the floor and an upright wall through the dancer, softly joined), so the waves wrap
     the body curve by curve, sweep over the floor to the bottom of the frame and open past the frame edges.
  2e. smoke (smoke.py, 3D Stable Fluids in a box in the scene): dragged by the limb joints, blown towards the lens and away by
     the sound waves, pushed out of the body; ray-marched from each camera (hidden behind the dancer), read through the warp.
  3. composite: plate looked up at px + warp, warp = smear along the motion + lens bend from the field gradient + fbm
     haze + rings, 7 taps along the warp (smear), each read once per wavelength (prism, see spectral()), dark rim where
     the gradient is steep, clear-glass reflection and highlight, and a per-wavelength glint on the tilted air (rainbow).
     The current dancer mask (MediaPipe) lowers the warp on the body by distort.body_keep so the dancer stays readable.
Outputs fx/%04d.png = the effect over transparent (alpha = how much the picture moved), for breakdown.py, and
pass/{grid,heat,normal,prism,smoke}/%04d.jpg (warp grid, distortion field in a thermal colour map, normal map, the colour
the prism added, smoke flow and density) for breakdown_passes.py."""
import json, shutil, subprocess, sys, time
from pathlib import Path
import numpy as np, cv2, moderngl

here = Path(__file__).resolve().parent
sys.path.insert(0, str(here.parent / "shader_live" / "pipeline"))
from slv import load, work, frame, scene   # noqa: E402

TRAIL_VS = """#version 330
uniform vec3 uRo; uniform mat3 uCam; uniform float uF; uniform vec2 uRes;
in vec3 pos; in vec3 prv; in vec3 attr;          // attr: age 0..1, u along the chain 0..1, gain
out vec2 vMv; out vec3 vA;
vec2 proj(vec3 P){ vec3 q = transpose(uCam) * (P - uRo); return uF * q.xy / q.z + 0.5 * uRes; }
void main(){
  vec2 p = proj(pos); vMv = p - proj(prv); vA = attr;
  gl_Position = vec4(p.x / uRes.x * 2.0 - 1.0, 1.0 - p.y / uRes.y * 2.0, 0.0, 1.0);
}"""
TRAIL_FS = """#version 330
uniform float uFadeIn;
in vec2 vMv; in vec3 vA; out vec4 o;
void main(){
  float age = vA.x, u = vA.y;
  float sp = length(vMv);                                     // px per frame of that bit of arm, seen now
  float w = vA.z * smoothstep(1.5, 14.0, sp) * smoothstep(0.0, uFadeIn, age) * pow(1.0 - age, 1.6) * smoothstep(0.0, 0.3, u) * smoothstep(1.0, 0.8, u);
  vec2 dir = vMv / max(sp, 1e-3) * min(sp, 60.0) / 60.0;
  o = vec4(dir * w, w, w * w);
}"""
FULL_VS = "#version 330\nin vec2 p; void main(){ gl_Position = vec4(p, 0.0, 1.0); }"
BLUR_FS = """#version 330
uniform sampler2D uSrc; uniform vec2 uDir; out vec4 o;
void main(){
  vec2 r = vec2(textureSize(uSrc, 0)); vec2 uv = gl_FragCoord.xy / r; vec4 s = vec4(0.0); float tw = 0.0;
  for (int k = -24; k <= 24; k++){ float g = exp(-float(k * k) / 200.0); s += g * texture(uSrc, uv + uDir * float(k) / r); tw += g; }
  o = s / tw;
}"""
NOISE = """float hash21(vec2 p){ vec3 q = fract(vec3(p.xyx) * 0.1031); q += dot(q, q.yzx + 33.33); return fract((q.x + q.y) * q.z); }
float vnoise(vec3 p){ vec3 i = floor(p), f = fract(p); f = f * f * (3.0 - 2.0 * f);
  float a = mix(hash21(i.xy + i.z * 17.0), hash21(i.xy + vec2(1, 0) + i.z * 17.0), f.x), b = mix(hash21(i.xy + vec2(0, 1) + i.z * 17.0), hash21(i.xy + vec2(1, 1) + i.z * 17.0), f.x);
  float c = mix(hash21(i.xy + (i.z + 1.0) * 17.0), hash21(i.xy + vec2(1, 0) + (i.z + 1.0) * 17.0), f.x), d = mix(hash21(i.xy + vec2(0, 1) + (i.z + 1.0) * 17.0), hash21(i.xy + vec2(1, 1) + (i.z + 1.0) * 17.0), f.x);
  return mix(mix(a, b, f.y), mix(c, d, f.y), f.z); }
float fbm(vec3 p){ float a = 0.5, s = 0.0; for (int i = 0; i < 4; i++){ s += a * vnoise(p); p = p * 2.03 + vec3(17.1, 5.3, 9.7); a *= 0.5; } return s; }
"""
RING_FS = """#version 330
uniform vec2 uRes, uScale; uniform float iTime;
uniform vec3 uRo; uniform mat3 uCam; uniform float uF, uZref; uniform vec3 uWallN; uniform float uWallD;
uniform int uNR; uniform vec4 uRing[8]; uniform vec4 uRingP[8]; uniform vec4 uLobe[16];
uniform vec4 uSegA[120]; uniform vec4 uSegB[120];
out vec4 o;
""" + NOISE + """
/* sound waves out of the body. Each pixel needs a 3D point, so the scene is a proxy: the floor (z = 0) and an upright wall
   through the dancer facing the camera, joined by a soft crease (smooth min of the two ray distances, no seam).
   A wave's front is not a sphere around one point: it is the distance to the dancer's BODY at the moment of the hit (15
   capsules between MediaPipe joints), smooth-min'd so it has no creases, where body parts far from the hitting joint lag
   behind (uSegA.w). So near the body the front follows the arms, legs and torso, curve by curve. Then
   - the front speed wobbles with direction (3D noise on the direction from the hit, uRingP.w), so far out it stays organic;
   - the amplitude has two random direction lobes (uLobe, von Mises), so each wave goes out in its own directions;
   - every hit sends a couple of these (echoes), each with its own lobes, speed and noise seed. */
vec3 proxy(vec2 p){
  vec3 d = uCam * vec3((p - 0.5 * uRes) / uF, 1.0);
  float tf = d.z < -1e-5 ? -uRo.z / d.z : 1e4; float dn = dot(d, uWallN); float tw = dn > 1e-5 ? (uWallD - dot(uRo, uWallN)) / dn : 1e4;
  float mn = min(tf, tw); float t = mn - 0.4 * log(exp(-(tf - mn) / 0.4) + exp(-(tw - mn) / 0.4));
  return uRo + max(t, 0.0) * d;
}
float capsule(vec3 p, vec3 a, vec3 b){ vec3 pa = p - a, ba = b - a; float h = clamp(dot(pa, ba) / max(dot(ba, ba), 1e-8), 0.0, 1.0); return length(pa - ba * h); }
float front(int k, vec3 X, out float wn){
  float dd[15]; float dm = 1e4;
  for (int s = 0; s < 15; s++){ vec4 A = uSegA[k * 15 + s]; dd[s] = capsule(X, A.xyz, uSegB[k * 15 + s].xyz) + A.w; dm = min(dm, dd[s]); }
  float sum = 0.0, ws = 0.0;
  for (int s = 0; s < 15; s++){ float e = exp(-(dd[s] - dm) / 0.15); sum += e; ws += e * uSegB[k * 15 + s].w; }
  wn = ws / sum; float d = dm - 0.15 * log(sum);
  vec3 u = normalize(X - uRing[k].xyz);
  float nz = fbm(u * 1.7 + vec3(uRingP[k].z, 0.0, iTime * 0.35)) - 0.47;
  return d * (1.0 + uRingP[k].w * nz * 2.0);
}
void main(){
  vec2 p = gl_FragCoord.xy * uScale;                                             /* image px (this target is image oriented) */
  vec3 s = vec3(0.0);
  if (uNR > 0){
    vec3 X0 = proxy(p), Xx = proxy(p + vec2(3.0, 0.0)), Xy = proxy(p + vec2(0.0, 3.0));
    float zc = (transpose(uCam) * (X0 - uRo)).z; float A = clamp(uZref / max(zc, 0.05), 0.4, 3.0);   /* nearer -> bigger */
    for (int k = 0; k < 8; k++){
      if (k >= uNR) break;
      float wn, w1, w2; float d0 = front(k, X0, wn), dx = front(k, Xx, w1), dy = front(k, Xy, w2);
      vec2 gd = vec2(dx - d0, dy - d0); vec2 dir = gd / max(length(gd), 1e-9);
      float x = (d0 - uRing[k].w) / uRingP[k].x; float env = exp(-0.45 * x * x);
      if (env < 1e-3) continue;
      vec3 u = normalize(X0 - uRing[k].xyz); vec4 L1 = uLobe[2 * k], L2 = uLobe[2 * k + 1];
      float lobe = min(0.12 + exp(L1.w * (dot(u, L1.xyz) - 1.0)) + exp(L2.w * (dot(u, L2.xyz) - 1.0)), 1.3);
      float amp = uRingP[k].y * A * lobe * mix(0.45, 1.0, wn);
      s.xy += dir * amp * sin(2.6 * x) * env;                                    /* push along the front's normal, a few crests */
      s.z += amp * env / 20.0;
    }
  }
  o = vec4(s, 1.0);
}"""
COMP_FS = """#version 330
uniform sampler2D uPlate; uniform sampler2D uField; uniform sampler2D uMask; uniform vec2 uRes; uniform float iTime;
uniform float uStrength, uRipple, uEdge, uKeep, uHaze, uGain;
uniform float uShock, uRefl, uSpec, uLift, uNear; uniform int uNW; uniform vec4 uWave[24]; uniform float uWaveW[24]; uniform float uWaveL[24]; uniform sampler2D uDist;
uniform sampler2D uRingT, uSmoke; uniform float uSmokeOn;
uniform int uNS; uniform float uSpecK[12]; uniform vec3 uSpecW[12]; uniform float uGlint, uTilt;               /* prism: per wavelength, warp scale (n-1)/(n_d-1) and RGB weight */
layout(location=0) out vec4 oComp; layout(location=1) out vec4 oFx; layout(location=2) out vec4 oField;
layout(location=3) out vec4 oNormal; layout(location=4) out vec4 oAux; layout(location=5) out vec4 oPrism;
vec3 toLin(vec3 c){ return pow(max(c, 0.0), vec3(2.2)); }
vec3 plate(vec2 px){ return toLin(texture(uPlate, clamp(px, vec2(0.5), uRes - 0.5) / uRes).rgb); }
vec4 field(vec2 px){ return texture(uField, vec2(px.x / uRes.x, 1.0 - px.y / uRes.y)); }
""" + NOISE + """/* outline pulses ("pang"): at a hit, the band of air hugging the dancer's silhouette near that joint is pushed outward.
   uDist = distance (px) from the current silhouette, outside only. Each pulse: centre px, front distance px, amplitude;
   width px; reach px (how far along the outline from the joint it spreads). It feeds the SAME field as the arm trails. */
float sdist(vec2 p){ return texture(uDist, p / uRes).r; }
float pulse(vec2 p){
  float d = sdist(p); float s = 0.0;
  for (int k = 0; k < 24; k++){
    if (k >= uNW) break;
    vec4 Q = uWave[k]; float x = (d - Q.z) / uWaveW[k]; vec2 r = p - Q.xy;
    s += exp(-x * x) * exp(-dot(r, r) / (2.0 * uWaveL[k] * uWaveL[k])) * Q.w;
  }
  return s * uShock * smoothstep(0.0, 5.0, d);
}
/* the trail air only lives close to the body: it fades with distance from the current silhouette (uNear px ~ distort.near_m),
   so a stroke that the arm has left behind thins out instead of hanging in space as a separate sheet */
float near(vec2 p){ float d = sdist(p) / uNear; return exp(-d * d); }
float air(vec2 p){ return field(p).z * near(p) + pulse(p); }
void main(){
  vec2 px = vec2(gl_FragCoord.x, uRes.y - gl_FragCoord.y);
  float e = 3.0;
  vec2 gd = vec2(sdist(px + vec2(e, 0)) - sdist(px - vec2(e, 0)), sdist(px + vec2(0, e)) - sdist(px - vec2(0, e)));
  vec2 nout = gd / max(length(gd), 1e-4);                                        /* away from the body */
  float pw = pulse(px), nr = near(px);
  vec4 F = field(px); float w = F.z * nr + pw;
  float cov = 1.0 - exp(-uGain * w);                                              /* 0..1: how much air is bent here */
  float m = texture(uMask, px / uRes).r; cov *= 1.0 - uKeep * smoothstep(0.3, 0.8, m);
  vec2 mv = (F.xy * nr + nout * pw) / max(w, 1e-3);                              /* motion direction: the stroke, or outward for a pulse */
  vec2 g = vec2(air(px + vec2(e, 0)) - air(px - vec2(e, 0)), air(px + vec2(0, e)) - air(px - vec2(0, e))) / (2.0 * e);
  float gm = length(g) * 40.0;
  vec3 q = vec3(px * 0.011 - mv * iTime * 0.6, iTime * 1.7);                   /* fine shimmer, drifting with the stroke: hot air, not a water sheet */
  vec2 hz = vec2(fbm(q), fbm(q + vec3(5.2, 1.3, 2.7))) - 0.5;
  vec2 warp = -mv * uStrength * cov                                              /* drag the picture along the stroke */
            - g * uRipple * 900.0 * cov                                          /* lens bend at the rim of the bent air */
            + hz * uHaze * 12.0 * cov;                                           /* heat-haze shimmer inside it */
  float keep = 1.0 - uKeep * smoothstep(0.3, 0.8, m);
  vec3 rg = texture(uRingT, px / uRes).xyz; vec2 rw = rg.xy * keep; float rcov = (1.0 - exp(-rg.z)) * keep;
  /* prism: every smear tap is read once per wavelength. Glass bends short wavelengths more (Cauchy n = A + B / lambda^2,
     A and B from n_d = 1.5 and the Abbe number), so each wavelength takes the warp scaled by (n - 1) / (n_d - 1), and the
     reads are summed with that wavelength's RGB (CIE 1931 fit, Wyman et al. 2013), weights normalised to white. */
  vec3 acc = vec3(0.0), mono = vec3(0.0); float tw = 0.0;
  for (int k = 0; k < 7; k++){
    float t = float(k) / 6.0; float wt = 1.0 - 0.6 * t; vec2 o = warp * (0.35 + 0.65 * t) + rw;
    vec3 s = vec3(0.0);
    for (int j = 0; j < 12; j++){ if (j >= uNS) break; s += uSpecW[j] * plate(px + o * uSpecK[j]); }
    acc += wt * s; mono += wt * plate(px + o); tw += wt;
  }
  vec3 bent = acc / tw;
  float rim = clamp(gm, 0.0, 1.0) * cov;
  /* glass: the bent air as a height field gives a normal; light it like clear glass */
  vec3 n = normalize(vec3(-g * 55.0 * cov - rw * uTilt, 1.0));
  cov = max(cov, rcov);
  float fres = pow(1.0 - n.z, 0.8);                                            /* 0 face-on, -> 1 on the steep rims */
  vec3 Ld = normalize(vec3(-0.45, -0.75, 0.55)); vec3 Hh = normalize(Ld + vec3(0.0, 0.0, 1.0));   /* sun from the upper left (shadows fall right) */
  float spec = pow(max(dot(n, Hh), 0.0), 14.0) * smoothstep(0.02, 0.2, fres) * cov;   /* broad and soft: no thin white contour lines */
  vec2 rp = px + n.xy * 70.0;                                                /* what the tilted glass reflects: the plate further along the normal */
  vec3 refl = (plate(rp) + plate(rp + vec2(9.0, 5.0)) + plate(rp - vec2(7.0, 8.0))) / 3.0;
  vec3 col = bent * (1.0 - uEdge * rim) * (1.0 + uLift * cov);                 /* clear glass lets light through, a touch brighter */
  col += (refl * 1.25 + 0.012) * clamp(fres * uRefl, 0.0, 0.45) * 0.55;           /* reflection only ADDS light: clear, never a dark film */
  col += vec3(1.0, 0.97, 0.92) * uSpec * spec;
  /* smoke (smoke.py): a 3D volume ray-marched from this camera (lit, premultiplied, already hidden behind the dancer);
     it hangs in the same air, so it is read through the warp */
  float sa = 0.0;
  if (uSmokeOn > 0.5){ vec4 smk = texture(uSmoke, (px + (warp + rw) * 0.6) / uRes); col = col * (1.0 - smk.a) + smk.rgb; sa = smk.a; }
  /* prism glint: the ripple crests as tilted glass catch the sun; each wavelength's normal tilts by its own (n - 1), so the
     highlight lands a little apart per wavelength and the glint comes out as a small rainbow band */
  vec3 glint = vec3(0.0);
  for (int j = 0; j < 12; j++){ if (j >= uNS) break; vec3 nj = normalize(vec3(n.xy / n.z * uSpecK[j], 1.0)); glint += uSpecW[j] * pow(max(dot(nj, Hh), 0.0), 400.0); }
  col += glint * uGlint;
  oComp = vec4(col, 1.0);
  vec2 wt2 = warp + rw;
  float aw = clamp(length(wt2) / 6.0 + rim * uEdge + fres * uRefl + spec * uSpec + sa, 0.0, 1.0);
  oFx = vec4(col * aw, aw);
  oField = vec4(0.5 + 0.5 * clamp(mv * cov, -1.0, 1.0), cov, 1.0);
  /* breakdown passes: the air's normal, the total warp (px), the colour the prism added */
  oNormal = vec4(n * 0.5 + 0.5, 1.0);
  oAux = vec4(wt2, length(wt2), cov);
  oPrism = vec4(bent - mono / tw, 1.0);
}"""

def main():
    c = load(sys.argv[1] if len(sys.argv) > 1 and not sys.argv[1].startswith("--") else here / "piece"); S = scene(c)
    W, H, N, fps = c["W"], c["H"], c["N"], c["fps"]; f = float(S["f"]); tr, ds = c["trail"], c["distort"]
    arg = lambda k, d=None: (sys.argv[sys.argv.index(k) + 1] if k in sys.argv else d)
    stills = [int(x) for x in arg("--still").split(",")] if "--still" in sys.argv else None
    L = np.load(work(c, "limbs.npz")); P = L["P"].astype(np.float64)
    ctx = moderngl.create_standalone_context(require=330)
    trail = ctx.program(vertex_shader=TRAIL_VS, fragment_shader=TRAIL_FS)
    blur = ctx.program(vertex_shader=FULL_VS, fragment_shader=BLUR_FS); comp = ctx.program(vertex_shader=FULL_VS, fragment_shader=COMP_FS)
    ringp = ctx.program(vertex_shader=FULL_VS, fragment_shader=RING_FS)
    quad = ctx.buffer(np.array([-1, -1, 3, -1, -1, 3], "f4").tobytes())
    vb, vc, vr = (ctx.vertex_array(q, [(quad, "2f", "p")]) for q in (blur, comp, ringp))
    hw, hh = W // 2, H // 2
    fA, fB, rT = (ctx.texture((hw, hh), 4, dtype="f4") for _ in range(3))
    for t in (fA, fB, rT): t.filter = (moderngl.LINEAR, moderngl.LINEAR); t.repeat_x = t.repeat_y = False
    fbA, fbB, fbR = ctx.framebuffer([fA]), ctx.framebuffer([fB]), ctx.framebuffer([rT])
    plate_t = ctx.texture((W, H), 3); plate_t.filter = (moderngl.LINEAR, moderngl.LINEAR)
    mask_t = ctx.texture((W, H), 1); mask_t.filter = (moderngl.LINEAR, moderngl.LINEAR)
    dist_t = ctx.texture((W, H), 1, dtype="f4"); dist_t.filter = (moderngl.LINEAR, moderngl.LINEAR)
    oc, ofx, ofd, onm, oax, opr = (ctx.texture((W, H), 4, dtype="f4") for _ in range(6)); fbo = ctx.framebuffer([oc, ofx, ofd, onm, oax, opr])
    K_s, W_s = spectral(c.get("prism", {})); comp["uNS"].value = len(K_s)
    comp["uSpecK"].value = list(K_s) + [1.0] * (12 - len(K_s)); comp["uSpecW"].value = [tuple(w) for w in W_s] + [(0.0, 0.0, 0.0)] * (12 - len(W_s))
    comp["uGlint"].value = float(c.get("prism", {}).get("glint", 0.0)); comp["uTilt"].value = float(c.get("prism", {}).get("tilt", 0.03))
    for k, v in (("uStrength", ds["strength"]), ("uRipple", ds["ripple"]), ("uEdge", ds["edge_dark"]),
                 ("uKeep", ds["body_keep"]), ("uHaze", ds["haze"]), ("uGain", ds.get("gain", 1.8)),
                 ("uShock", ds.get("shock", 14.0)), ("uRefl", ds.get("reflect", 0.9)), ("uSpec", ds.get("spec", 0.7)), ("uLift", ds.get("lift", 0.06))): comp[k].value = float(v)
    sc = c.get("smoke", {}); sm = None
    if sc.get("enable", False):
        from smoke import Smoke
        sm = Smoke(ctx, W, H, sc, S, P, f, fps, NOISE); comp["uSmokeOn"].value = 1.0
    else:
        comp["uSmokeOn"].value = 0.0
    near_m = float(ds.get("near_m", 9.0)); z_body = float(np.median(L["depth"]))           # air fades past near_m from the outline
    waves = shock_events(c, L, fps); rg = c.get("ring", {})
    kept = []                                                                     # strongest hits first, at most one per gap_frames
    for h in sorted(waves["hits"], key=lambda h: -h[3]) if rg.get("amp_px", 0) > 0 else []:
        if h[3] >= rg.get("min_hit", 0.5) and all(abs(h[0] - e[0]) >= rg.get("gap_frames", 16) for e in kept): kept.append(h)
    kept.sort(key=lambda h: h[0]); R0 = np.asarray(S["rot"][0]); fw0 = np.array([R0[0][2], R0[1][2]]) / np.hypot(R0[0][2], R0[1][2])
    sound = wave_events(kept, P, rg, fw0)
    print("sound waves: %d hits at frames %s -> %d waves" % (len(kept), [h[0] for h in kept], len(sound)))
    comp["uRes"].value = (W, H); trail["uRes"].value = (W, H); trail["uF"].value = f; trail["uFadeIn"].value = float(tr.get("fade_in", 0.08))
    ringp["uRes"].value = (W, H); ringp["uScale"].value = (W / hw, H / hh); ringp["uF"].value = f
    limbs, gains = tr["limbs"], tr.get("limb_gain", [1.0] * len(tr["limbs"])); K, sub = int(tr["age_frames"]), int(tr.get("sub", 3)); NU = 9

    def at(t):                                                                   # all 33 landmarks at fractional frame t (clamped)
        t = float(np.clip(t, 0, N - 1)); i = min(int(t), N - 2); u = t - i; return P[i] * (1 - u) + P[i + 1] * u
    def chain(Q, ids):                                                           # NU points along the limb chain, tip pushed 25% past the last joint
        pts = [Q[j] for j in ids]; pts.append(pts[-1] + float(tr.get("tip", 0.25)) * (pts[-1] - pts[-2]))
        seg = np.array([np.linalg.norm(pts[k + 1] - pts[k]) for k in range(len(pts) - 1)]); cum = np.concatenate([[0], np.cumsum(seg)]) / max(seg.sum(), 1e-6)
        return np.array([np.array([np.interp(u, cum, [p[a] for p in pts]) for a in range(3)]) for u in np.linspace(0, 1, NU)])
    def mesh(i):
        ages = np.arange(K * sub + 1) / sub; verts = []
        for li, ids in enumerate(limbs):
            G = np.array([chain(at(i - a), ids) for a in ages]); Gp = np.array([chain(at(i - a - 1.0), ids) for a in ages])   # [A, NU, 3]
            for a in range(len(ages) - 1):
                for u in range(NU - 1):
                    for (aa, uu) in ((a, u), (a + 1, u), (a + 1, u + 1), (a, u), (a + 1, u + 1), (a, u + 1)):
                        verts.append([*G[aa, uu], *Gp[aa, uu], ages[aa] / ages[-1], uu / (NU - 1), gains[li]])
        return np.array(verts, "f4")
    def simulate(i):
        """inputs for frame i: plate, mask, distance field, trail field, sound-wave field; then the smoke steps once"""
        pl = frame(c, i); plate_t.write(pl[:, :, ::-1].tobytes())
        m = cv2.imread(str(work(c, "mask", f"{i:04d}.png")), cv2.IMREAD_GRAYSCALE); mask_t.write(m.tobytes())
        mb = (cv2.GaussianBlur(m, (0, 0), 2.0) > 127).astype(np.uint8)           # distance from the silhouette, outside the body
        dt = cv2.GaussianBlur(cv2.distanceTransform(1 - mb, cv2.DIST_L2, 5), (0, 0), 2.5); dist_t.write(dt.astype("f4").tobytes())
        R, ro = S["rot"][i], S["cam"][i]; Rt = tuple(np.asarray(R).T.flatten())
        trail["uRo"].value = tuple(ro); trail["uCam"].value = Rt
        vbo = ctx.buffer(mesh(i).tobytes()); va = ctx.vertex_array(trail, [(vbo, "3f 3f 3f", "pos", "prv", "attr")])
        fbA.use(); fbA.clear(0, 0, 0, 0); ctx.enable(moderngl.BLEND); ctx.blend_func = (moderngl.ONE, moderngl.ONE); va.render(moderngl.TRIANGLES)
        ctx.disable(moderngl.BLEND); va.release(); vbo.release()
        fbB.use(); fA.use(0); blur["uSrc"].value = 0; blur["uDir"].value = (1.0, 0.0); vb.render(moderngl.TRIANGLES)
        fbA.use(); fB.use(0); blur["uDir"].value = (0.0, 1.0); vb.render(moderngl.TRIANGLES)
        zi = float(np.clip(L["depth"][i], 0.5 * z_body, 2.0 * z_body)); uNear = f * near_m / zi
        fw = np.array([R[0][2], R[1][2], 0.0]); fw /= max(np.linalg.norm(fw), 1e-6)           # horizontal view direction
        hip = 0.5 * (P[i, 23] + P[i, 24])
        for k, v in (("uRo", tuple(ro)), ("uCam", Rt), ("uZref", zi), ("uWallN", tuple(fw)), ("uWallD", float(fw @ hip) + rg.get("wall_back_m", 0.3)), ("iTime", i / fps)):
            ringp[k].value = v
        act = ring_uniforms(sound, i, fps, rg); set_waves(ringp, act, rg)
        fbR.use(); vr.render(moderngl.TRIANGLES)
        if sm is not None:                                                      # 3D smoke: step in the scene, then ray-march from this camera
            sw = lambda prog: set_waves(prog, act, rg); amp = float(rg.get("amp_px", 14.0))
            for _ in range(int(sc.get("preroll", 40)) if i == 0 else 0): sm.step(i, P, lambda prog: set_waves(prog, [], rg), amp, 0.0, arms=0.0)
            sm.step(i, P, sw, amp, i / fps)
            zb = float((np.asarray(R).T @ (hip - np.asarray(ro)))[2]); sm.render(R, ro, mask_t, zb, i / fps)
        return pl, uNear
    def compose(i, pl, uNear):
        fbo.use(); plate_t.use(0); fA.use(1); mask_t.use(2); dist_t.use(3); rT.use(4); comp["uPlate"].value = 0; comp["uField"].value = 1; comp["uMask"].value = 2; comp["uDist"].value = 3; comp["uRingT"].value = 4
        if sm is not None: sm.tex.use(5); comp["uSmoke"].value = 5
        R, ro = S["rot"][i], S["cam"][i]; wl = wave_uniforms(waves, i, R, ro, f, W, H, fps)
        comp["uNW"].value = len(wl); pad = wl + [(0.0, 0.0, 0.0, 0.0, 1.0, 1.0)] * (24 - len(wl))
        comp["uWave"].value = [w[:4] for w in pad]; comp["uWaveW"].value = [w[4] for w in pad]; comp["uWaveL"].value = [w[5] for w in pad]
        comp["uNear"].value = uNear; comp["iTime"].value = i / fps; vc.render(moderngl.TRIANGLES); ctx.finish()
        rd = lambda t: np.frombuffer(t.read(), "f4").reshape(H, W, 4)[::-1]
        co, fx, fd = rd(oc), rd(ofx), rd(ofd)
        o8 = (np.clip(co[..., :3], 0, 1) ** (1 / 2.2) * 255 + 0.5).astype(np.uint8)[:, :, ::-1]
        a = np.clip(fx[..., 3:4], 0, 1); col = np.where(a > 1e-4, fx[..., :3] / np.maximum(a, 1e-4), 0) ** (1 / 2.2)
        fx8 = np.dstack([(np.clip(col, 0, 1)[:, :, ::-1] * 255 + 0.5).astype(np.uint8), (a[..., 0] * 255 + 0.5).astype(np.uint8)])
        fd8 = (np.clip(fd[..., :3], 0, 1) * 255 + 0.5).astype(np.uint8)[:, :, ::-1]
        return o8, fx8, fd8, passes(pl, rd(onm), rd(oax), rd(opr), sm.read() if sm is not None else None)
    if stills:
        rows = []; todo = set(stills)
        for i in (range(max(stills) + 1) if sm is not None else sorted(todo)):
            pl, un = simulate(i)
            if i not in todo: continue
            o, fx, fd, ps = compose(i, pl, un); cv2.imwrite(str(work(c, f"still_{i:04d}.png")), o)
            t = lambda im, s: label(cv2.resize(im, (640, 360)), s)
            rows.append(np.hstack([t(pl, "plate"), t(ps["grid"], "warp grid"), t(ps["heat"], "distortion field"), t(ps["smoke"], "smoke sim")]))
            rows.append(np.hstack([t(ps["normal"], "normal"), t(ps["prism"], "prism"), t(o, f"comp {i}"), t(cv2.resize(o[270:810, 480:1440], (W, H)), "comp zoom")]))
        cv2.imwrite(str(work(c, "still_sheet.jpg")), np.vstack(rows), [cv2.IMWRITE_JPEG_QUALITY, 90]); print("stills", stills, "->", work(c, "still_sheet.jpg")); return
    for d in ("comp", "fx", "field", *("pass/" + k for k in PASSES)): shutil.rmtree(work(c, d), ignore_errors=True); work(c, d).mkdir(parents=True)
    ts = time.time()
    for i in range(N):
        pl, un = simulate(i); o, fx, fd, ps = compose(i, pl, un)
        cv2.imwrite(str(work(c, "comp", f"{i:04d}.png")), o, [cv2.IMWRITE_PNG_COMPRESSION, 1])
        cv2.imwrite(str(work(c, "fx", f"{i:04d}.png")), fx, [cv2.IMWRITE_PNG_COMPRESSION, 1]); cv2.imwrite(str(work(c, "field", f"{i:04d}.jpg")), fd)
        for k in PASSES: cv2.imwrite(str(work(c, "pass", k, f"{i:04d}.jpg")), ps[k], [cv2.IMWRITE_JPEG_QUALITY, 93])
        if i % 40 == 0: print("frame", i, "%.0fs" % (time.time() - ts), flush=True)
    encode(c)

SEGS = [(11, 13), (13, 15), (15, 19), (12, 14), (14, 16), (16, 20), (23, 25), (25, 27), (24, 26), (26, 28),
        (11, 12), (23, 24), (11, 23), (12, 24), (0, 0)]                          # arms, hands, legs, shoulders, hips, torso sides, head

def wave_events(hits, P, rg, fw=None):
    """Sound waves. Each kept hit sends ring.echoes waves (ring.echo_frames apart). A wave carries the body at the hit as 15
    capsules; parts far from the hitting joint lag behind (ring.lag_m, gaussian ring.focus_m), so the front starts at the
    hitting limb and then wraps the rest of the body. Random per wave (seeded, ring.seed): two direction lobes, speed +-20%,
    a noise seed for the direction-dependent wobble (ring.wobble)."""
    rng = np.random.default_rng(int(rg.get("seed", 7))); out = []
    for t0, j, C, a in hits:
        Q = P[t0]; A = np.array([Q[s[0]] for s in SEGS]); B = np.array([Q[s[1]] for s in SEGS])
        wgt = np.exp(-np.sum((0.5 * (A + B) - C) ** 2, 1) / (2 * rg.get("focus_m", 0.4) ** 2))
        lag = (1.0 - wgt) * rg.get("lag_m", 0.5); lag[-1] -= 0.1                  # the head is a 10 cm ball
        for e in range(int(rg.get("echoes", 2))):
            lobes = []
            for l in range(2):
                if e == 0 and fw is not None:                                     # first wave: one lobe at the lens, one away from it (+-35 deg)
                    base = np.arctan2(-fw[1], -fw[0]) if l == 0 else np.arctan2(fw[1], fw[0]); az, el = base + rng.uniform(-0.6, 0.6), rng.uniform(-0.2, 0.35)
                else:
                    az, el = rng.uniform(0, 2 * np.pi), rng.uniform(-0.35, 0.6)
                lobes.append((np.cos(el) * np.cos(az), np.cos(el) * np.sin(az), np.sin(el), rng.uniform(1.5, 4.5)))
            out.append(dict(t0=t0 + e * rg.get("echo_frames", 4), C=C, a=a * 0.75 ** e, speed=rg.get("speed_mps", 5.5) * rng.uniform(0.8, 1.2),
                            seed=rng.uniform(0, 50), lobes=lobes, A=A, B=B, lag=lag, wgt=wgt))
    return out

def ring_uniforms(ev, i, fps, rg):
    out = []; life = rg.get("life_s", 1.3)
    for w in ev:
        age = (i - w["t0"]) / fps
        if age < 0 or age > life: continue
        amp = rg.get("amp_px", 14.0) * w["a"] * (1.0 - age / life) ** rg.get("decay", 1.0) * min(1.0, age / 0.06)
        out.append((amp, 0.05 + w["speed"] * age, w))
    out.sort(key=lambda x: -x[0]); return out[:8]

def set_waves(prog, act, rg):
    ring, rp, lobe, sa, sb = [], [], [], [], []
    for amp, r, w in act:
        ring.append((*map(float, w["C"]), r)); rp.append((rg.get("width_m", 0.16) + rg.get("grow", 0.05) * r, amp, w["seed"], rg.get("wobble", 0.25)))
        lobe += [tuple(map(float, l)) for l in w["lobes"]]
        sa += [(*map(float, w["A"][s]), float(w["lag"][s])) for s in range(len(SEGS))]; sb += [(*map(float, w["B"][s]), float(w["wgt"][s])) for s in range(len(SEGS))]
    n = len(act); z4 = (0.0, 0.0, 0.0, 0.0)
    prog["uNR"].value = n; prog["uRing"].value = ring + [z4] * (8 - n); prog["uRingP"].value = rp + [(1.0, 0.0, 0.0, 0.0)] * (8 - n)
    prog["uLobe"].value = lobe + [(1.0, 0.0, 0.0, 1.0)] * (16 - len(lobe))
    prog["uSegA"].value = sa + [z4] * (120 - len(sa)); prog["uSegB"].value = sb + [z4] * (120 - len(sb))

PASSES = ("grid", "heat", "normal", "prism", "smoke")

def spectral(pr):
    """Prism weights. n(lambda) = A + B / lambda^2 (Cauchy) with n_d = 1.5 at 587.6 nm and the Abbe number V = (n_d - 1) / (n_F - n_C);
    real crown glass is V ~ 60, prism.abbe ~ 5 exaggerates the split so it reads on screen. Each of prism.samples wavelengths
    (400-700 nm) gets warp scale k = (n - 1) / (n_d - 1) and an RGB weight: CIE 1931 x,y,z from the piecewise-Gaussian fit of
    Wyman, Sloan & Shirley, "Simple Analytic Approximations to the CIE XYZ Color Matching Functions", JCGT 2(2), 2013,
    then XYZ -> linear sRGB (IEC 61966-2-1), negatives clipped, each channel normalised so the weights sum to white.
    prism.abbe = 0 turns the prism off (one sample, k = 1)."""
    V, ns = float(pr.get("abbe", 0.0)), int(pr.get("samples", 8))
    if V <= 0: return np.ones(1), np.ones((1, 3))
    lam = 400.0 + 300.0 * (np.arange(ns) + 0.5) / ns
    g = lambda x, m, s1, s2: np.exp(-0.5 * ((x - m) / np.where(x < m, s1, s2)) ** 2)
    X = 1.056 * g(lam, 599.8, 37.9, 31.0) + 0.362 * g(lam, 442.0, 16.0, 26.7) - 0.065 * g(lam, 501.1, 20.4, 26.2)
    Y = 0.821 * g(lam, 568.8, 46.9, 40.5) + 0.286 * g(lam, 530.9, 16.3, 31.1)
    Z = 1.217 * g(lam, 437.0, 11.8, 36.0) + 0.681 * g(lam, 459.0, 26.0, 13.8)
    M = np.array([[3.2406, -1.5372, -0.4986], [-0.9689, 1.8758, 0.0415], [0.0557, -0.2040, 1.0570]])
    rgb = np.clip(np.stack([X, Y, Z], 1) @ M.T, 0, None); rgb /= rgb.sum(0, keepdims=True)
    B = (0.5 / V) / (1 / 0.4861 ** 2 - 1 / 0.6563 ** 2); A = 1.5 - B / 0.5876 ** 2
    k = (A + B / (lam / 1000.0) ** 2 - 1.0) / 0.5
    print("prism: Abbe %.1f, %d wavelengths, warp scale %.3f (400nm) .. %.3f (700nm)" % (V, ns, k[0], k[-1]))
    return k, rgb

_CMAP = None
def thermal(x):
    """0..1 -> BGR uint8. matplotlib's 'inferno' (CC0, Smith & van der Walt) if matplotlib is installed, else Google's Turbo
    polynomial (Copyright 2019 Google LLC, Apache-2.0, Mikhailov / Ruofei Du: gist.github.com/mikhailov-work/0d177465a8151eb6ede1768d51d476c7)."""
    global _CMAP
    if _CMAP is None:
        try:
            from matplotlib import colormaps; _CMAP = (colormaps["inferno"](np.linspace(0, 1, 256))[:, :3] * 255).astype(np.uint8)
        except ImportError:
            t = np.linspace(0, 1, 256); v4 = np.stack([np.ones_like(t), t, t * t, t ** 3], 1); v2 = np.stack([t ** 4, t ** 5], 1)
            k4 = np.array([[0.13572138, 4.61539260, -42.66032258, 132.13108234], [0.09140261, 2.19418839, 4.84296658, -14.18503333], [0.10667330, 12.64194608, -60.58204836, 110.36276771]])
            k2 = np.array([[-152.94239396, 59.28637943], [4.27729857, 2.82956604], [-89.90310912, 27.34824973]])
            _CMAP = (np.clip(v4 @ k4.T + v2 @ k2.T, 0, 1) * 255).astype(np.uint8)
    return _CMAP[(np.clip(x, 0, 1) * 255).astype(np.int32)][..., ::-1]

def passes(pl, nm, ax, pr, smk=None, cell=60.0):
    """Breakdown images (BGR uint8) like the reference post: warp grid, distortion field (thermal), normal map, prism split."""
    H, W = pl.shape[:2]; gray = cv2.cvtColor(pl, cv2.COLOR_BGR2GRAY).astype(np.float32) / 255.0
    yy, xx = np.mgrid[0:H, 0:W].astype(np.float32); u, v = xx + ax[..., 0], yy + ax[..., 1]   # the plate pixel each output pixel shows
    gl = np.maximum(np.clip(2.2 - np.abs(((u / cell) + 0.5) % 1.0 - 0.5) * cell, 0, 1), np.clip(2.2 - np.abs(((v / cell) + 0.5) % 1.0 - 0.5) * cell, 0, 1))
    grid = np.clip(gray * 0.42 + gl * 0.85, 0, 1)
    heat = thermal(np.sqrt(np.clip(ax[..., 2] / 40.0, 0, 1))).astype(np.float32) / 255.0
    heat = np.maximum(heat, (gray * 0.22)[..., None])
    nrm = np.clip(nm[..., :3], 0, 1)[..., ::-1] * 0.88 + (gray * 0.12)[..., None]
    pv = np.clip(np.maximum(pr[..., :3], 0) ** (1 / 2.2) * 3.2, 0, 1)[..., ::-1] + (gray * 0.16)[..., None]
    q = lambda im: (np.clip(im, 0, 1) * 255 + 0.5).astype(np.uint8)
    sv = (gray * 0.2)[..., None] * np.ones(3)                                    # 3D smoke, depth-coded: near the lens orange, far blue
    if smk is not None:
        sk = cv2.resize(smk, (W, H)); a = np.clip(sk[..., 3:4], 0, 1)
        sv = sv * (1.0 - a) + np.clip(sk[..., 2::-1], 0, 1) ** (1 / 2.2) * 1.15
    return {"grid": q(np.dstack([grid] * 3)), "heat": q(heat), "normal": q(nrm), "prism": q(pv), "smoke": q(sv)}

def label(im, s):
    im = im.copy(); cv2.putText(im, s, (12, im.shape[0] - 14), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 4, cv2.LINE_AA)
    cv2.putText(im, s, (12, im.shape[0] - 14), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 1, cv2.LINE_AA); return im

def shock_events(c, L, fps):
    """Hits: a joint that was moving fast and stops within 3 frames (speed drop in px/frame above shock.hit_px).
    Each hit becomes pulses (shock.pulses, shock.gap_frames apart) at the joint's 3D position at that frame."""
    from scipy.signal import find_peaks
    sh = c.get("shock", {}); img, P = L["img"], L["P"].astype(np.float64); ev = []
    for j, thr in sh.get("joints", {"15": 10, "16": 10, "13": 10, "14": 10, "25": 15, "26": 15}).items():
        j = int(j); v = np.linalg.norm(np.diff(img[:, j], axis=0), axis=1); dec = np.zeros_like(v); dec[2:-1] = v[:-3] - v[3:]
        for p in find_peaks(dec, height=thr, distance=7)[0]: ev.append((int(p), j, float(dec[p])))
    ev.sort(); keep = []
    for e in ev:                                                                  # one burst per joint group within 3 frames, keep the strongest
        if keep and e[0] - keep[-1][0] <= 2 and np.linalg.norm(P[e[0], e[1]] - P[keep[-1][0], keep[-1][1]]) < 0.25:
            if e[2] > keep[-1][2]: keep[-1] = e
            continue
        keep.append(e)
    out = []
    for t0, j, s in keep:
        a = float(np.clip(s / 25.0, 0.45, 1.0))
        for k in range(int(sh.get("pulses", 2))): out.append((t0 + k * float(sh.get("gap_frames", 2.5)), P[t0, j], a * (1.0 - 0.3 * k)))
    print("shock: %d hits -> %d pulses at frames %s" % (len(keep), len(out), [e[0] for e in keep]))
    hits = [(t0, j, P[t0, j], float(np.clip(s / 25.0, 0.45, 1.0))) for t0, j, s in keep]
    return {"pulses": out, "hits": hits, "life": float(sh.get("life_s", 0.55)), "rmax": float(sh.get("radius_m", 0.55)), "tau": float(sh.get("tau_s", 0.14)), "reach": float(sh.get("reach_m", 0.35))}

def wave_uniforms(waves, i, R, ro, f, W, H, fps):
    out = []
    for t0, C, a in waves["pulses"]:
        age = (i - t0) / fps
        if age < 0 or age > waves["life"]: continue
        q = np.asarray(R).T @ (C - ro)
        if q[2] <= 0.1: continue
        r_m = 0.01 + waves["rmax"] * (1.0 - np.exp(-age / waves["tau"])); r = f * r_m / q[2]      # front: distance from the outline
        amp = a * (1.0 - age / waves["life"]) ** 1.6 * min(1.0, age * fps / 1.0)
        reach = f * (waves["reach"] + 0.8 * r_m) / q[2]                                   # how far along the outline it spreads
        out.append((f * q[0] / q[2] + W / 2, f * q[1] / q[2] + H / 2, r, amp, 8.0 + 0.18 * r, reach))
    out.sort(key=lambda w: -w[3]); return out[:24]

def encode(c):
    out = c["dir"] / "out"; out.mkdir(exist_ok=True); nm = c["name"]; r = f"{c['fps_num']}/{c['fps_den']}"
    tags = ["-pix_fmt", "yuv420p", "-color_primaries", "bt709", "-color_trc", "bt709", "-colorspace", "bt709", "-movflags", "+faststart"]
    run = lambda a: subprocess.run(["ffmpeg", "-v", "error", "-y"] + a, check=True)
    x264 = ["-c:v", "libx264", "-crf", "14", "-preset", "slow"]
    run(["-framerate", r, "-i", str(work(c, "comp", "%04d.png"))] + x264 + tags + [str(out / f"{nm}_Composite.mp4")])
    run(["-framerate", r, "-i", str(work(c, "frames", "%04d.jpg"))] + x264 + tags + [str(out / f"{nm}_Plate.mp4")])
    # stacked before/after, effect on top: the same layout as the reference post
    lab = lambda t, p: f"drawtext=fontfile='C\\:/Windows/Fonts/arialbd.ttf':text='{t}':x=w-tw-28:y=h-th-24:fontsize=44:fontcolor=white:shadowcolor=black@0.6:shadowx=2:shadowy=2"
    run(["-framerate", r, "-i", str(work(c, "comp", "%04d.png")), "-framerate", r, "-i", str(work(c, "frames", "%04d.jpg")), "-filter_complex",
         f"[0]scale=1080:608,{lab('Space Distortion VFX', 0)}[a];[1]scale=1080:608,{lab('Before', 1)}[b];[a][b]vstack,pad=1080:1350:0:67:color=black"] + x264 + tags + [str(out / f"{nm}_Compare_1080x1350.mp4")])
    print("encoded ->", out)

if __name__ == "__main__":
    main()
