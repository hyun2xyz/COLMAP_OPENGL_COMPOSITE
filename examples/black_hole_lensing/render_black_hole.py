"""Effect #4: Interstellar Gargantua Relativistic Gravitational Lensing Black Hole (Native OpenGL 3.3 Raytracer).

    python render_black_hole.py --still 15,110,190,260    quick look -> work/still_bh_sheet.jpg
    python render_black_hole.py --render                full render -> out/Black_Hole_Lensing_Hall_*.mp4

Real 12-meter 6-DOF COLMAP architectural walkthrough (Pexels 7578546) with a hovering 3D Black Hole:
- Exact 120-step Schwarzschild null-geodesic curved-spacetime ray integrator (d^2u/dphi^2 + u = 1.5*r_s*u^2).
- Escaping deflected rays project back into the live COLMAP camera to warp the ACTUAL sunlit white pillars,
  windows, ceiling beams, and oak hardwood floor into a real-time moving Einstein Ring!
- Tilted relativistic volumetric swirling plasma Accretion Disk with Keplerian differential rotation +
  physical Doppler beaming (incandescent blue-white approaching left edge, warm crimson-amber receding right edge),
  naturally bent over top & underneath the pitch-black event horizon shadow by curved spacetime!
- Crisp luminous Photon Ring + hardwood floor warm orange volumetric glow spill + optical bloom + sensor grain.
- ZERO text/HUD overlays.
"""
import argparse, json, math, shutil, subprocess, sys, time
from pathlib import Path
import numpy as np, cv2, moderngl

here = Path(__file__).resolve().parent
ROOT = here.parents[1]
sys.path.insert(0, str(ROOT / "examples/shader_live/pipeline"))
from slv import load, work, frame, scene  # noqa: E402

SRC_PIECE = ROOT / "examples/liquid_glass/piece_glass"


def find_ffmpeg():
    tl = ROOT / "tools.local.json"
    if tl.exists():
        d = json.loads(tl.read_text(encoding="utf-8"))
        p = d.get("ffmpeg") or (d.get("tools") or {}).get("ffmpeg")
        if p and Path(p).exists():
            return str(p)
    return shutil.which("ffmpeg") or "ffmpeg"


VS_FSQ = "#version 330\nin vec2 p; void main(){ gl_Position = vec4(p, 0.0, 1.0); }"

BH_FS = """#version 330
uniform sampler2D uPlate;
uniform vec2 uRes;
uniform float uF, uTime, uRs, uFrame, uBlackFloor, uGrainAmt;
uniform vec3 uRo, uBhPos;
uniform mat3 uCam, uDiskBasis;  /* columns = disk X, disk Y, disk Normal */

out vec4 oCol;

float hash21(vec2 p){ vec3 q=fract(vec3(p.xyx)*0.1031); q+=dot(q,q.yzx+33.33); return fract((q.x+q.y)*q.z); }
float vnoise(vec3 p){
  vec3 i=floor(p), f=fract(p); f=f*f*(3.0-2.0*f);
  float a=mix(hash21(i.xy+i.z*17.0), hash21(i.xy+vec2(1,0)+i.z*17.0), f.x);
  float b=mix(hash21(i.xy+vec2(0,1)+i.z*17.0), hash21(i.xy+vec2(1,1)+i.z*17.0), f.x);
  float c=mix(hash21(i.xy+(i.z+1.0)*17.0), hash21(i.xy+vec2(1,0)+(i.z+1.0)*17.0), f.x);
  float d=mix(hash21(i.xy+vec2(0,1)+(i.z+1.0)*17.0), hash21(i.xy+vec2(1,1)+(i.z+1.0)*17.0), f.x);
  return mix(mix(a,b,f.y), mix(c,d,f.y), f.z);
}
float fbm(vec3 p){
  float s=0.0, a=0.5;
  for(int k=0;k<4;k++){ s+=a*vnoise(p); p=p*2.07+vec3(13.1,7.3,5.7); a*=0.5; }
  return s;
}
vec3 acesFilm(vec3 x){ return clamp((x*(2.51*x+0.03))/(x*(2.43*x+0.59)+0.14),0.0,1.0); }

/* Project a deflected world ray direction back into the live COLMAP camera plate (linear RGB) */
vec3 sampleLiveRayLin(vec3 dirW) {
  vec3 q = transpose(uCam) * dirW;
  vec2 px = uF * q.xy / max(q.z, 0.15) + 0.5 * uRes;
  vec2 uv = clamp(px / uRes, vec2(0.002), vec2(0.998));
  return pow(max(texture(uPlate, uv).rgb, vec3(0.001)), vec3(2.2));
}

/* Relativistic Gargantua Accretion Disk emission at local disk-space position pL */
vec4 sampleDisk(vec3 pL, vec3 dirL) {
  float r = length(pL.xy);
  float rIn  = uRs * 2.15;
  float rOut = uRs * 7.80;
  if (r < rIn || r > rOut) return vec4(0.0);
  float u = (r - rIn) / (rOut - rIn);

  /* Differentially rotating Keplerian plasma spirals (inner rings whip faster!) */
  float ang = atan(pL.y, pL.x);
  float omega = 1.95 / pow(max(r / uRs, 1.0), 1.30);
  float swirlAng = ang - uTime * omega + 3.6 * log(max(r / rIn, 1.0));
  vec3 sp = vec3(cos(swirlAng) * 2.5, sin(swirlAng) * 2.5, u * 5.6 - uTime * 0.55);
  float n1 = fbm(sp);
  float n2 = fbm(sp * 2.4 + vec3(4.1, -2.7, uTime * 0.9));
  /* Fine blazing concentric relativistic plasma filaments */
  float rings = 0.52 + 0.48 * sin(r / uRs * 13.5 + n1 * 4.8);
  float filament = smoothstep(0.18, 0.80, n1 * 0.62 + n2 * 0.38) * rings;

  /* Blazing ISCO inner ridge + long wispy fiery outer fade */
  float innerGlow = exp(-pow(u / 0.14, 2.0)) * 0.90;
  float radEnv = smoothstep(0.0, 0.06, u) * pow(max(1.0 - u, 0.0), 1.45);
  float dens = (filament * 0.85 + innerGlow) * radEnv;
  if (dens < 0.002) return vec4(0.0);

  /* Relativistic Doppler beaming: tangential velocity CCW (-sin, cos, 0) in disk plane */
  vec3 velL = normalize(vec3(-pL.y, pL.x, 0.0));
  float doppler = dot(velL, -dirL);
  float beam = pow(clamp(1.0 + 0.48 * doppler, 0.45, 1.85), 2.1);

  /* Interstellar Gargantua fiery blackbody palette: blinding white-gold ISCO -> blazing amber-orange -> deep crimson dust! */
  float temp = clamp((1.0 - u * 0.94) + 0.22 * doppler + 0.10 * n2, 0.0, 1.0);
  vec3 colCold = vec3(0.82, 0.09, 0.008);
  vec3 colMid  = vec3(1.00, 0.36, 0.025);
  vec3 colHot  = vec3(1.00, 0.76, 0.28);
  vec3 colCore = vec3(1.00, 0.96, 0.86);
  vec3 col = mix(colCold, colMid, smoothstep(0.0, 0.44, temp));
  col = mix(col, colHot, smoothstep(0.44, 0.80, temp));
  col = mix(col, colCore, smoothstep(0.80, 1.0, temp));

  return vec4(col * dens * beam * 3.90, clamp(dens * 1.55, 0.0, 0.98));
}

void main() {
  vec2 px0 = vec2(gl_FragCoord.x, uRes.y - gl_FragCoord.y);
  vec2 uv0 = px0 / uRes;
  vec3 rawSrgb = texture(uPlate, clamp(uv0, vec2(0.002), vec2(0.998))).rgb;
  vec3 rawLin = pow(max(rawSrgb, vec3(0.0005)), vec3(2.2));
  vec3 rd0 = normalize(uCam * vec3((px0 - 0.5 * uRes) / uF, 1.0));

  /* Real oak hardwood floor warm amber glow spill directly beneath the blazing accretion disk! */
  vec3 floorSpill = vec3(0.0);
  if (rd0.z < -0.005) {
    float tf = -uRo.z / rd0.z;
    if (tf > 0.2 && tf < 28.0) {
      vec3 Xf = uRo + rd0 * tf;
      float df = length(Xf - uBhPos);
      float att = exp(-df * df / 7.2) * 0.38;
      floorSpill = vec3(1.00, 0.44, 0.09) * att * (0.94 + 0.06 * sin(uTime * 5.0));
    }
  }

  /* Fast bounding-sphere cull for untouched background pixels (100% seamless raw plate!) */
  vec3 oc = uRo - uBhPos;
  float distCam = length(oc);
  float bClose = length(cross(oc, rd0));
  float maxInfluence = uRs * 14.5;
  if (bClose > maxInfluence || dot(-oc, rd0) < -uRs * 6.0) {
    vec3 outLin = rawLin + floorSpill / (1.0 + floorSpill * 0.45);
    oCol = vec4(pow(clamp(outLin, 0.0, 1.0), vec3(1.0 / 2.2)), 1.0);
    return;
  }

  /* Transform ray into black-hole local space (centered at uBhPos) */
  vec3 p = oc;
  vec3 v = rd0;
  /* Angular momentum h = |p x v| for exact GR Schwarzschild geodesic acceleration a = -1.5 * h^2 * r_s / r^5 * p */
  vec3 hVec = cross(p, v);
  float h2 = dot(hVec, hVec);

  /* Advance straight to outer sphere of influence (r = 15 * r_s) so all 130 geodesic steps focus near horizon! */
  float rStart = uRs * 15.0;
  if (length(p) > rStart) {
    float tProj = -dot(p, v);
    float d2 = dot(p, p) - tProj * tProj;
    float dtEnter = tProj - sqrt(max(rStart * rStart - d2, 0.0));
    if (dtEnter > 0.0) p += v * dtEnter;
  }

  vec3 diskAccum = vec3(0.0);
  float trans = 1.0;
  bool captured = false;
  float minR = 1e6;
  vec3 pPrevL = transpose(uDiskBasis) * p;
  float rCapture = uRs * 1.06;
  float rEsc = max(distCam + uRs * 8.0, uRs * 18.0);

  for (int step_i = 0; step_i < 130; step_i++) {
    float r = length(p);
    minR = min(minR, r);
    if (r <= rCapture) {
      captured = true;
      break;
    }
    if (r > rEsc && dot(p, v) > 0.0) {
      break;
    }

    /* Adaptive step size: razor-sharp tiny steps near photon sphere (1.5 r_s), larger steps far away */
    float dt = clamp(r * 0.076, uRs * 0.032, uRs * 0.62);
    vec3 acc = -1.5 * h2 * uRs / max(pow(r, 5.0), 1e-6) * p;
    vec3 vNext = normalize(v + acc * dt);
    vec3 pNext = p + 0.5 * (v + vNext) * dt;

    /* Exact equatorial plane crossing test for razor-crisp Gargantua accretion disk! */
    vec3 pNextL = transpose(uDiskBasis) * pNext;
    if ((pPrevL.z > 0.0 && pNextL.z <= 0.0) || (pPrevL.z < 0.0 && pNextL.z >= 0.0)) {
      float frac = abs(pPrevL.z) / max(abs(pPrevL.z) + abs(pNextL.z), 1e-5);
      vec3 hitL = mix(pPrevL, pNextL, frac);
      vec3 dirL = normalize(transpose(uDiskBasis) * v);
      vec4 dk = sampleDisk(hitL, dirL);
      diskAccum += trans * dk.rgb;
      trans *= (1.0 - dk.a * 0.92);
      if (trans < 0.015) break;
    }
    /* Thin silky volumetric glow hugging disk plane (keeps shadow cavity above/below 100% pure pitch black!) */
    if (abs(pNextL.z) < uRs * 0.14) {
      float rCyl = length(pNextL.xy);
      if (rCyl > uRs * 2.05 && rCyl < uRs * 7.6) {
        float hFade = exp(-pow(pNextL.z / (uRs * 0.065), 2.0));
        vec3 dirL = normalize(transpose(uDiskBasis) * v);
        vec4 dk = sampleDisk(vec3(pNextL.xy, 0.0), dirL);
        float wVol = hFade * (dt / (uRs * 2.2));
        diskAccum += trans * dk.rgb * wVol * 0.36;
      }
    }

    p = pNext;
    v = vNext;
    pPrevL = pNextL;
  }

  /* 1. Background: PURE PITCH-BLACK VOID if captured inside event horizon, else LIVE GRAVITATIONALLY LENSED HALL! */
  vec3 bgLin = vec3(0.0);
  if (!captured) {
    float lf = smoothstep(maxInfluence, uRs * 4.8, bClose);
    float lensFade = lf * lf;
    vec3 bentDir = normalize(mix(rd0, v, lensFade));
    bgLin = sampleLiveRayLin(bentDir) + floorSpill;
  }

  /* 2. Razor-sharp Interstellar Photon Ring hugging the exact critical shadow edge (b_crit ~ 2.60 r_s)! */
  float pRing = (!captured) ? exp(-pow((minR - uRs * 1.50) / (uRs * 0.090), 2.0)) : 0.0;
  float edgeRim = exp(-pow((bClose - uRs * 2.61) / (uRs * 0.105), 2.0));
  vec3 photonCol = vec3(1.00, 0.74, 0.32) * (pRing * 3.10 + edgeRim * 1.30);

  /* 3. Atmospheric golden-amber bloom ONLY outside the pitch-black shadow cavity (NEVER fogs the black void!) */
  float outsideShadow = smoothstep(uRs * 2.40, uRs * 3.25, bClose);
  float halo = exp(-pow(max(bClose - uRs * 2.60, 0.0) / (uRs * 3.8), 1.45)) * outsideShadow * 0.28;
  vec3 haloCol = vec3(1.00, 0.42, 0.08) * halo;

  /* 4. Soft HDR highlight compression ONLY on emissive plasma (preserves raw plate exposure 100% identically!) */
  vec3 emLin = diskAccum + photonCol + haloCol;
  vec3 emSoft = emLin / (1.0 + emLin * 0.22);
  vec3 totalLin = bgLin * trans + emSoft;

  /* Subtle dark black-floor match on the pure void + sensor grain ONLY where black hole / lensing acts */
  vec3 col = pow(clamp(totalLin, 0.0, 1.0), vec3(1.0 / 2.2));
  float effectW = clamp(smoothstep(maxInfluence, uRs * 5.0, bClose) + length(emSoft) + (captured ? 1.0 : 0.0), 0.0, 1.0);
  if (effectW > 0.01) {
    vec3 graded = max(col, vec3(uBlackFloor * 0.45));
    vec2 p1 = gl_FragCoord.xy + vec2(uFrame * 17.3, uFrame * 31.7);
    vec2 p2 = floor(gl_FragCoord.xy * 0.5) + vec2(uFrame * 11.1, uFrame * 23.9);
    float g = (hash21(p1)+hash21(p1+7.13)-1.0)*0.62 + (hash21(p2)+hash21(p2+5.71)-1.0)*0.38;
    float lum = dot(graded, vec3(0.299, 0.587, 0.114));
    graded += vec3(g) * (uGrainAmt * (1.08 - 0.50 * smoothstep(0.55, 0.98, lum)));
    col = mix(col, graded, effectW);
  }
  oCol = vec4(clamp(col, 0.0, 1.0), 1.0);
}
"""


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--still", default=None)
    ap.add_argument("--render", action="store_true")
    args = ap.parse_args()

    c = load(SRC_PIECE)
    S = scene(c)
    C = np.asarray(S["cam"], dtype=np.float64)
    R_all = np.asarray(S["rot"], dtype=np.float64)
    f_px = float(S["f"])
    N = len(C)
    fps = float(c["fps"])
    pl0 = frame(c, 0)
    H, W = pl0.shape[:2]

    gray0 = cv2.cvtColor(pl0, cv2.COLOR_BGR2GRAY).astype(np.float32) / 255.0
    black_floor = float(np.clip(np.percentile(gray0, 1.0) * 0.65, 0.004, 0.025))
    hp0 = gray0 - cv2.GaussianBlur(gray0, (0, 0), 1.2)
    grain_amt = float(np.clip(float(np.median(np.abs(hp0 - np.median(hp0)))) * 1.4826 * 1.10, 0.008, 0.022))

    # Hover Gargantua Black Hole dead-center in the modern sunlit archway (y=+4.65m, z=1.68m!)
    bh_pos = np.array([0.04, 4.65, 1.68], dtype=np.float64)
    r_s = 0.172  # Shadow radius b_crit = 2.60*r_s ~ 0.447m (0.89m obsidian sphere), full plasma disk ~2.7m wide!

    # Near edge-on Interstellar Gargantua tilt: 8.5 deg pitch + -9.5 deg roll!
    ax_p, ax_r = math.radians(8.5), math.radians(-9.5)
    Rx = np.array([[1, 0, 0], [0, math.cos(ax_p), -math.sin(ax_p)], [0, math.sin(ax_p), math.cos(ax_p)]])
    Ry = np.array([[math.cos(ax_r), 0, math.sin(ax_r)], [0, 1, 0], [-math.sin(ax_r), 0, math.cos(ax_r)]])
    disk_basis = (Ry @ Rx).astype(np.float32)

    ctx = moderngl.create_standalone_context(require=330)
    prog = ctx.program(vertex_shader=VS_FSQ, fragment_shader=BH_FS)
    fsq = ctx.buffer(np.array([-1, -1, 1, -1, -1, 1, -1, 1, 1, -1, 1, 1], dtype=np.float32).tobytes())
    vao = ctx.vertex_array(prog, [(fsq, "2f", "p")])
    t_out = ctx.texture((W, H), 4, dtype="f4")
    fbo = ctx.framebuffer([t_out])
    plate_t = ctx.texture((W, H), 3); plate_t.filter = (moderngl.LINEAR, moderngl.LINEAR); plate_t.repeat_x = plate_t.repeat_y = False

    prog["uRes"].value = (float(W), float(H))
    prog["uF"].value = f_px
    prog["uRs"].value = r_s
    prog["uBhPos"].value = tuple(map(float, bh_pos))
    prog["uDiskBasis"].value = tuple(disk_basis.T.flatten())
    prog["uBlackFloor"].value = black_floor
    prog["uGrainAmt"].value = grain_amt

    def render_one(i):
        pl = frame(c, i)
        plate_t.write(pl[:, :, ::-1].tobytes())
        ro = C[i]; R = R_all[i]
        Rt = tuple(R.T.flatten().astype(np.float32))
        fbo.use(); ctx.viewport = (0, 0, W, H); ctx.disable(moderngl.DEPTH_TEST); ctx.disable(moderngl.BLEND)
        plate_t.use(0)
        prog["uPlate"].value = 0
        prog["uRo"].value = tuple(map(float, ro))
        prog["uCam"].value = Rt
        prog["uTime"].value = float(i / fps)
        prog["uFrame"].value = float(i)
        vao.render(moderngl.TRIANGLES)
        ctx.finish()
        arr = np.frombuffer(t_out.read(), "f4").reshape(H, W, 4)[::-1]
        img = (np.clip(arr[..., :3], 0, 1) * 255 + 0.5).astype(np.uint8)[:, :, ::-1]
        return pl, img

    wk = here / "work"; wk.mkdir(parents=True, exist_ok=True)
    if args.still:
        idxs = [int(x) for x in args.still.split(",")]
        rows = []
        for idx in idxs:
            pl, comp = render_one(idx)
            a, b = cv2.resize(pl, (640, 360)), cv2.resize(comp, (640, 360))
            cv2.putText(b, f"black hole f={idx}", (10, 344), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 0, 0), 3)
            cv2.putText(b, f"black hole f={idx}", (10, 344), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (80, 220, 255), 1)
            rows.append(np.hstack([a, b]))
        out_p = wk / "still_bh_sheet.jpg"
        cv2.imwrite(str(out_p), np.vstack(rows), [cv2.IMWRITE_JPEG_QUALITY, 92])
        print("stills ->", out_p)
        return

    if args.render:
        comp_dir = wk / "comp"
        shutil.rmtree(comp_dir, ignore_errors=True); comp_dir.mkdir(parents=True)
        ts = time.time()
        for i in range(N):
            _, comp = render_one(i)
            cv2.imwrite(str(comp_dir / f"{i:04d}.png"), comp, [cv2.IMWRITE_PNG_COMPRESSION, 1])
            if i % 50 == 0 or i == N - 1:
                print(f"black_hole frame {i}/{N} ({time.time() - ts:.1f}s)", flush=True)
        out_dir = here / "out"; out_dir.mkdir(parents=True, exist_ok=True)
        r = f"{c['fps_num']}/{c['fps_den']}"; ff = find_ffmpeg()
        tags = ["-pix_fmt", "yuv420p", "-color_primaries", "bt709", "-color_trc", "bt709", "-colorspace", "bt709", "-movflags", "+faststart"]
        x264 = ["-c:v", "libx264", "-profile:v", "high", "-level", "4.1", "-crf", "14", "-preset", "slow"]
        run = lambda a: subprocess.run([ff, "-v", "error", "-y"] + a, check=True)
        run(["-framerate", r, "-i", str(comp_dir / "%04d.png")] + x264 + tags + [str(out_dir / "Black_Hole_Lensing_Hall_Composite.mp4")])
        run([
            "-framerate", r, "-i", str(comp_dir / "%04d.png"),
            "-framerate", r, "-i", str(work(c, "frames", "%04d.jpg")),
            "-filter_complex", "[0]scale=1080:608[a];[1]scale=1080:608[b];[a][b]vstack,pad=1080:1350:0:67:color=black",
        ] + x264 + tags + [str(out_dir / "Black_Hole_Lensing_Hall_Compare_1080x1350.mp4")])
        print("encoded ->", out_dir)


if __name__ == "__main__":
    main()
