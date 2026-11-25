"""Effect #2: Magnetic Ferrofluid Rosensweig Spikes Reacting to Dancer's Hands (Native OpenGL 3.3 SDF Raymarcher).

    python render_ferrofluid.py --still 15,65,125,180    quick look -> work/still_ferro_sheet.jpg
    python render_ferrofluid.py --render               full render -> out/Ferrofluid_Magnetic_Spikes_*.mp4

Real street-dance footage (Pexels 5359146) with live 3D raymarched black-chrome magnetic ferrofluid:
- Physics spring-damped levitating ferrofluid masses & tendril satellites orbit and stretch smoothly
  toward the dancer's MediaPipe 3D wrists (P[15], P[16]), PLUS a living ferrofluid pool on the brick
  sidewalk (z=0) that erupts upward into bristling spires as her hands sweep through the air!
- True 3D Rosensweig magnetic instability SDF: conical hexagonal + icosahedral liquid-metal spikes
  dynamically orient and sharpen along the live magnetic dipole field vector B(p) pointing to her hands!
- Luxury pitch-black liquid-chrome clearcoat PBR shading (F0=0.82, crisp rough=0.04) reflecting the live
  street plate + architectural louver skylight + subtle dark-oil iridescent rim + grounded soft shadows
  on the herringbone brick floor + MediaPipe depth occlusion (foreground orbs pass IN FRONT OF her,
  flank/background masses & rear floor pool tuck cleanly BEHIND her!) + optical sensor grain.
- ZERO text/HUD overlays.
"""
import argparse, json, math, shutil, subprocess, sys, time
from pathlib import Path
import numpy as np, cv2, moderngl

here = Path(__file__).resolve().parent
ROOT = here.parents[1]
sys.path.insert(0, str(ROOT / "examples/shader_live/pipeline"))
from slv import load, work, frame, scene  # noqa: E402

SRC_PIECE = ROOT / "examples/space_distortion/piece_street"


def find_ffmpeg():
    tl = ROOT / "tools.local.json"
    if tl.exists():
        d = json.loads(tl.read_text(encoding="utf-8"))
        p = d.get("ffmpeg") or (d.get("tools") or {}).get("ffmpeg")
        if p and Path(p).exists():
            return str(p)
    return shutil.which("ffmpeg") or "ffmpeg"


def smooth_1d(arr, sigma=1.4):
    """Zero-phase Gaussian temporal smoothing along axis 0 for silky physical motion."""
    rad = max(2, int(round(sigma * 3.0)))
    xs = np.arange(-rad, rad + 1, dtype=np.float64)
    k = np.exp(-0.5 * (xs / sigma) ** 2)
    k /= k.sum()
    out = np.empty_like(arr, dtype=np.float64)
    pad = np.pad(arr, ((rad, rad), (0, 0)), mode="edge")
    for c in range(arr.shape[1]):
        out[:, c] = np.convolve(pad[:, c], k, mode="valid")
    return out


def sim_ferrofluid_orbs(P_body, fps):
    """Simulate 8 spring-damped 3D magnetic ferrofluid masses coupled to the dancer's hands & feet.
    Returns:
      orbs:   (N, 8, 4) float32 -> (x, y, z, radius)
      handL:  (N, 3) float32 -> smoothed left wrist/hand magnet
      handR:  (N, 3) float32 -> smoothed right wrist/hand magnet
      feetL:  (N, 2) float32 -> smoothed left ankle XY
      feetR:  (N, 2) float32 -> smoothed right ankle XY
      energy: (N,) float32 -> smoothed hand speed magnetic excitation (0..1)
    """
    N = len(P_body)
    dt = 1.0 / fps
    # Hand centers: slight palm projection past wrists (15->19 left, 16->20 right)
    hL_raw = 0.65 * P_body[:, 15] + 0.35 * P_body[:, 19]
    hR_raw = 0.65 * P_body[:, 16] + 0.35 * P_body[:, 20]
    hL = smooth_1d(hL_raw, sigma=1.4)
    hR = smooth_1d(hR_raw, sigma=1.4)
    fL = smooth_1d(P_body[:, 27, :2], sigma=1.8)
    fR = smooth_1d(P_body[:, 28, :2], sigma=1.8)
    pelvis = smooth_1d(0.5 * (P_body[:, 23] + P_body[:, 24]), sigma=2.2)

    # Hand velocity -> drives magnetic field intensity & Rosensweig spike bristle height!
    vL = np.gradient(hL, dt, axis=0)
    vR = np.gradient(hR, dt, axis=0)
    spd = np.linalg.norm(vL, axis=1) + np.linalg.norm(vR, axis=1)
    energy = smooth_1d(np.clip((spd - 0.45) / 2.8, 0.0, 1.0)[:, None], sigma=2.5)[:, 0]

    # 8 spring-damped ferrofluid orbs:
    #   0: Hero Left-Hand foreground magnet orb (orbits smoothly off left palm, slightly in front of body!)
    #   1: Hero Right-Hand foreground magnet orb (orbits smoothly off right palm, slightly in front of body!)
    #   2: Whipping Left-Hand trailing liquid satellite
    #   3: Whipping Right-Hand trailing liquid satellite
    #   4: Rising Magnetic Floor-to-Left-Hand Tendril Blob (pulls upward from floor puddle!)
    #   5: Rising Magnetic Floor-to-Right-Hand Tendril Blob
    #   6: Large Sculptural Flank-Right Ferrofluid Monolith Orb (slightly BEHIND dancer plane -> depth occlusion!)
    #   7: Large Sculptural Flank-Left Ferrofluid Monolith Orb (slightly BEHIND dancer plane -> depth occlusion!)
    K = np.array([56.0, 56.0, 26.0, 26.0, 34.0, 34.0, 20.0, 20.0], dtype=np.float64)
    D = np.array([ 8.4,  8.4,  5.6,  5.6,  7.0,  7.0,  5.4,  5.4], dtype=np.float64)
    RAD = np.array([0.108, 0.108, 0.074, 0.074, 0.098, 0.098, 0.145, 0.135], dtype=np.float64)

    def targets_at(i):
        t = i * dt
        e = energy[i]
        w1 = np.array([0.04 * math.cos(t * 2.4), -0.03 * math.sin(t * 1.9), 0.04 * math.sin(t * 2.7)])
        w2 = np.array([0.04 * math.sin(t * 2.2), -0.03 * math.cos(t * 2.1), 0.04 * math.cos(t * 2.5)])
        px, py, pz = pelvis[i]

        dirL = hL[i] - pelvis[i]; dirL[1] = 0.0
        nL = dirL / (np.linalg.norm(dirL) + 1e-6)
        dirR = hR[i] - pelvis[i]; dirR[1] = 0.0
        nR = dirR / (np.linalg.norm(dirR) + 1e-6)

        t0 = hL[i] + nL * 0.18 + np.array([0.04, -0.17, 0.02]) + w1
        t1 = hR[i] + nR * 0.18 + np.array([-0.04, -0.17, 0.02]) + w2
        t2 = hL[i] + nL * 0.30 + np.array([0.09, -0.12, -0.13]) - w1 * 0.8
        t3 = hR[i] + nR * 0.30 + np.array([-0.09, -0.12, -0.13]) - w2 * 0.8

        # Keep hero hand orbs safely inside upper camera frame even when she reaches straight up!
        t0[2] = min(t0[2], 1.62); t1[2] = min(t1[2], 1.62)
        t2[2] = min(t2[2], 1.50); t3[2] = min(t3[2], 1.50)

        # Rising magnetic spires stretching directly out of the black floor pool toward her hands!
        t4 = np.array([0.52 * hL[i, 0] + 0.48 * (px + 0.46), py - 0.22, 0.15 + 0.19 * e + 0.04 * math.sin(t * 3.1)])
        t5 = np.array([0.52 * hR[i, 0] + 0.48 * (px - 0.50), py - 0.20, 0.14 + 0.18 * e + 0.04 * math.cos(t * 2.9)])

        # Hero hovering flank spiky ferrofluid spheres framed cleanly against bright grey corrugated shutter (behind dancer!)
        t6 = np.array([px + 0.64 + 0.22 * (hL[i, 0] - px), py + 0.20, 1.06 + 0.22 * (hL[i, 2] - 1.1) + 0.06 * math.sin(t * 1.8)])
        t7 = np.array([px - 0.66 + 0.20 * (hR[i, 0] - px), py + 0.20, 0.94 + 0.20 * (hR[i, 2] - 1.0) + 0.06 * math.cos(t * 1.6)])
        return np.stack([t0, t1, t2, t3, t4, t5, t6, t7], axis=0)

    pos = targets_at(0).copy()
    vel = np.zeros_like(pos)
    orbs = np.zeros((N, 8, 4), dtype=np.float32)

    sub = 4
    sdt = dt / sub
    for i in range(N):
        tgt = targets_at(i)
        for _ in range(sub):
            acc = K[:, None] * (tgt - pos) - D[:, None] * vel
            vel += acc * sdt
            pos += vel * sdt
            pos[:, 2] = np.maximum(pos[:, 2], 0.09)
        # Subtle magnetic volume breathing with hand energy
        pulse = 1.0 + 0.10 * energy[i] * np.array([1.0, 1.0, 0.8, 0.8, 1.2, 1.2, 0.9, 0.9])
        orbs[i, :, :3] = pos.astype(np.float32)
        orbs[i, :, 3] = (RAD * pulse).astype(np.float32)

    return (
        orbs,
        hL.astype(np.float32),
        hR.astype(np.float32),
        fL.astype(np.float32),
        fR.astype(np.float32),
        energy.astype(np.float32),
    )


VS_FSQ = "#version 330\nin vec2 p; void main(){ gl_Position = vec4(p, 0.0, 1.0); }"

FERRO_FS = """#version 330
uniform sampler2D uPlate, uPersonMask;
uniform vec2 uRes;
uniform float uF, uTime, uFrame, uPersonDepth, uEnergy, uBlackFloor, uGrainAmt;
uniform vec3 uRo, uHandL, uHandR, uPoolCenter;
uniform vec2 uFootL, uFootR;
uniform mat3 uCam;
uniform vec4 uOrbs[8];  /* xyz = world pos, w = radius */

out vec4 oCol;

float hash21(vec2 p){ vec3 q=fract(vec3(p.xyx)*0.1031); q+=dot(q,q.yzx+33.33); return fract((q.x+q.y)*q.z); }

/* Smooth minimum for organic liquid-metal magnetic tendril merging */
float smin(float a, float b, float k) {
  float h = clamp(0.5 + 0.5 * (b - a) / k, 0.0, 1.0);
  return mix(b, a, h) - k * h * (1.0 - h);
}

/* Magnetic dipole field direction & intensity driven live by dancer's two hands! */
vec4 magField(vec3 p) {
  vec3 dL = uHandL - p;
  vec3 dR = uHandR - p;
  float r2L = dot(dL, dL) + 0.10;
  float r2R = dot(dR, dR) + 0.10;
  vec3 B = dL / (r2L * sqrt(r2L)) + dR / (r2R * sqrt(r2R));
  float mag = clamp(length(B) * 0.36 + uEnergy * 0.38, 0.18, 1.65);
  return vec4(normalize(B + vec3(0.0, 0.0, 0.25)), mag);
}

/* Crisp 3D Rosensweig magnetic spike field: dramatic needle-pointed conical ferrofluid crowns! */
float rosensweigSpikes(vec3 p, vec3 Bdir, float freq, float tPhase, float isFloor) {
  /* 1. Magnetic-field-aligned hexagonal spire lattice (authentic standing Rosensweig cones erupting off floor pool!) */
  vec3 up = abs(Bdir.z) < 0.90 ? vec3(0.0, 0.0, 1.0) : vec3(1.0, 0.0, 0.0);
  vec3 t1 = normalize(cross(up, Bdir));
  vec3 t2 = cross(Bdir, t1);
  vec2 uvB = vec2(dot(p, t1), dot(p, t2)) * freq + vec2(tPhase * 0.48, -tPhase * 0.34);
  float h1 = cos(uvB.x);
  float h2 = cos(0.5 * uvB.x + 0.8660254 * uvB.y);
  float h3 = cos(0.5 * uvB.x - 0.8660254 * uvB.y);
  float hexRaw = clamp((h1 + h2 + h3 + 1.5) / 4.5, 0.0, 1.0);
  /* Sharp pointed conical cusp (like real ferrofluid over a strong neodymium magnet!) */
  float hexCone = pow(hexRaw, 1.85);

  /* 2. 12-axis icosahedral hedgehog lattice so spherical levitating orbs bristle with unmistakable 3D pointed cones! */
  const vec3 g1 = vec3( 0.525731,  0.850651,  0.000000);
  const vec3 g2 = vec3(-0.525731,  0.850651,  0.000000);
  const vec3 g3 = vec3( 0.000000,  0.525731,  0.850651);
  const vec3 g4 = vec3( 0.000000, -0.525731,  0.850651);
  const vec3 g5 = vec3( 0.850651,  0.000000,  0.525731);
  const vec3 g6 = vec3( 0.850651,  0.000000, -0.525731);
  vec3 q = p * (freq * 1.05) + Bdir * (tPhase * 0.55);
  /* Sharpened triangular-cusp waves -> distinct pointed sea-urchin ferrofluid crowns! */
  float ico = (pow(0.5 + 0.5 * cos(dot(q, g1)), 2.0) + pow(0.5 + 0.5 * cos(dot(q, g2)), 2.0) +
               pow(0.5 + 0.5 * cos(dot(q, g3)), 2.0) + pow(0.5 + 0.5 * cos(dot(q, g4)), 2.0) +
               pow(0.5 + 0.5 * cos(dot(q, g5)), 2.0) + pow(0.5 + 0.5 * cos(dot(q, g6)), 2.0)) / 6.0;
  float icoCone = pow(clamp((ico - 0.24) / 0.62, 0.0, 1.0), 1.65);

  return mix(icoCone, hexCone, mix(0.25, 0.88, isFloor));
}

/* Base smooth ferrofluid shape (cheap bounding evaluation before adding Rosensweig spikes) */
float mapBase(vec3 p, out float isFloor) {
  /* 1. Paper-thin organic pitch-black liquid puddle hugging the brick floor (zero thick vertical slab edge!) */
  vec2 dp = p.xy - uPoolCenter.xy;
  float rPool = length(dp * vec2(0.86, 1.45));
  float ang = atan(dp.y, dp.x);
  float edgeWav = 0.09 * sin(ang * 3.0 + uTime * 1.1) + 0.06 * cos(ang * 5.0 - uTime * 1.5) + 0.03 * sin(ang * 7.0 + uTime * 0.8);
  float poolRad = 0.86 + edgeWav;
  /* Keep sneakers crisp & unburied by parting puddle smoothly around both ankles */
  float dShoeL = length(p.xy - uFootL);
  float dShoeR = length(p.xy - uFootR);
  float shoeClear = smoothstep(0.11, 0.25, min(dShoeL, dShoeR));
  /* Smooth meniscus taper: drops gently all the way to z=0 at pool edge (zero artificial step!) */
  float radTaper = pow(smoothstep(poolRad, 0.0, rPool), 1.45);
  float handMound = (exp(-dot(p.xy - uHandL.xy, p.xy - uHandL.xy) * 2.6) +
                     exp(-dot(p.xy - uHandR.xy, p.xy - uHandR.xy) * 2.6)) * 0.055;
  float domeH = (0.042 * radTaper + handMound * smoothstep(poolRad, 0.12, rPool)) * shoeClear;
  /* Smooth ellipsoid-like meniscus SDF so puddle melts naturally into the herringbone bricks */
  float dPool = max(p.z - domeH, -p.z - 0.012);
  dPool = max(dPool, (rPool - poolRad) * 0.65);

  /* 2. 8 Spring-damped levitating ferrofluid orbs smoothly merged with magnetic tendrils! */
  float dOrbs = 1e5;
  for (int k = 0; k < 8; k++) {
    vec3 c = uOrbs[k].xyz;
    float rad = uOrbs[k].w;
    float d = length(p - c) - rad;
    float blendK = (k < 4) ? 0.16 : ((k < 6) ? 0.24 : 0.18);
    dOrbs = (k == 0) ? d : smin(dOrbs, d, blendK);
  }

  /* Smooth magnetic tendril bridge where rising spires pull upward out of the floor pool! */
  float dTotal = smin(dPool, dOrbs, 0.19);
  isFloor = smoothstep(0.16, 0.015, p.z) * smoothstep(0.08, -0.02, dPool - dOrbs);
  return dTotal;
}

float map(vec3 p) {
  float isFloor;
  float d = mapBase(p, isFloor);
  /* Smoothly ramp Rosensweig spike displacement inside tight shell (guarantees 100% connected spikes, ZERO detached dots!) */
  if (d < 0.15) {
    vec4 mf = magField(p);
    float spk = rosensweigSpikes(p, mf.xyz, mix(32.0, 26.0, isFloor), uTime * 2.0, isFloor);
    float amp = mix(0.088, 0.115, isFloor) * clamp(0.75 + 0.45 * mf.w, 0.72, 1.28);
    if (isFloor > 0.15) {
      float rPool = length((p.xy - uPoolCenter.xy) * vec2(0.86, 1.45));
      float rimFade = smoothstep(0.86, 0.28, rPool);
      float shoeClear = smoothstep(0.13, 0.28, min(length(p.xy - uFootL), length(p.xy - uFootR)));
      amp *= mix(1.0, rimFade * shoeClear, isFloor);
    }
    float attach = smoothstep(0.14, 0.015, max(d, 0.0));
    d -= spk * amp * attach;
  }
  return d;
}

vec3 calcNormal(vec3 p) {
  const float e = 0.0018;
  vec2 k = vec2(1.0, -1.0);
  return normalize(
    k.xyy * map(p + k.xyy * e) +
    k.yyx * map(p + k.yyx * e) +
    k.yxy * map(p + k.yxy * e) +
    k.xxx * map(p + k.xxx * e)
  );
}

/* Soft analytical SDF shadow ray for grounded cast shadows on brick floor & self-shadowing between spikes! */
float calcSoftShadow(vec3 ro, vec3 rd, float mint, float maxt, float kSharp) {
  float res = 1.0;
  float t = mint;
  for (int i = 0; i < 30; i++) {
    float h = map(ro + rd * t);
    res = min(res, kSharp * h / max(t, 0.02));
    if (res < 0.02) return 0.0;
    t += clamp(h * 0.45, 0.02, 0.18);
    if (t > maxt) break;
  }
  return clamp(res, 0.0, 1.0);
}

/* Sample live street plate + high-contrast obsidian-black clearcoat IBL (JET-BLACK body + razor white highlights!) */
vec3 sampleChromeEnv(vec3 R) {
  vec3 q = transpose(uCam) * R;
  vec3 plateRef = vec3(0.18, 0.18, 0.18);
  if (q.z > 0.15) {
    vec2 px = uF * q.xy / q.z + 0.5 * uRes;
    vec2 uv = px / uRes;
    if (uv.x > 0.01 && uv.x < 0.99 && uv.y > 0.01 && uv.y < 0.99) {
      plateRef = pow(max(texture(uPlate, uv).rgb, vec3(0.01)), vec3(2.2));
    }
  }
  /* High-contrast studio/street obsidian reflection map: deep dark lows so fluid reads JET-BLACK, crisp bright skylight bands! */
  float up = clamp(R.z * 0.5 + 0.5, 0.0, 1.0);
  vec3 groundCol = vec3(0.025, 0.020, 0.018);
  vec3 horizCol  = vec3(0.16, 0.16, 0.17);
  vec3 skyCol    = vec3(0.42, 0.46, 0.52);
  vec3 procEnv = mix(groundCol, horizCol, smoothstep(0.38, 0.60, up));
  procEnv = mix(procEnv, skyCol, smoothstep(0.60, 0.95, up));

  /* Crisp narrow architectural skylight / louver specular ribbons -> iconic liquid-ink wet highlights! */
  float strip1 = pow(clamp(dot(R, normalize(vec3(-0.38, -0.60, 0.70))), 0.0, 1.0), 18.0);
  float strip2 = pow(clamp(dot(R, normalize(vec3( 0.58, -0.52, 0.62))), 0.0, 1.0), 22.0);
  float louverBand = 0.72 + 0.28 * pow(0.5 + 0.5 * sin(R.z * 28.0), 3.0);
  procEnv = procEnv * louverBand + vec3(3.2, 3.05, 2.85) * strip1 + vec3(1.8, 2.0, 2.25) * strip2;

  float usePlate = smoothstep(0.15, 0.55, q.z);
  return mix(procEnv, plateRef * 0.45 + procEnv * 0.55, usePlate * 0.50);
}

vec3 acesFilm(vec3 x){ return clamp((x*(2.51*x+0.03))/(x*(2.43*x+0.59)+0.14),0.0,1.0); }

void main() {
  vec2 px0 = vec2(gl_FragCoord.x, uRes.y - gl_FragCoord.y);
  vec2 uv0 = px0 / uRes;
  vec3 rawSrgb = texture(uPlate, clamp(uv0, vec2(0.002), vec2(0.998))).rgb;
  float pMask = texture(uPersonMask, clamp(uv0, vec2(0.001), vec2(0.999))).r;
  vec3 rd = normalize(uCam * vec3((px0 - 0.5 * uRes) / uF, 1.0));

  vec3 Lkey = normalize(vec3(-0.38, -0.55, 0.74));
  vec3 Lrim = normalize(vec3( 0.65, -0.35, 0.67));

  vec3 bMin = uPoolCenter + vec3(-1.65, -0.95, -0.06);
  vec3 bMax = uPoolCenter + vec3( 1.65,  0.85,  2.15);
  vec3 invD = 1.0 / (rd + vec3(1e-6));
  vec3 t0s = (bMin - uRo) * invD;
  vec3 t1s = (bMax - uRo) * invD;
  vec3 tSm = min(t0s, t1s);
  vec3 tBg = max(t0s, t1s);
  float tEnter = max(max(tSm.x, tSm.y), max(tSm.z, 0.15));
  float tExit  = min(min(tBg.x, tBg.y), tBg.z);

  /* 1. Grounded cast shadow on herringbone brick floor (z = 0) beneath hovering orbs & spiky spires! */
  vec3 bgCol = rawSrgb;
  float cgA = 0.0;
  if (rd.z < -0.005) {
    float tf = -uRo.z / rd.z;
    if (tf > 0.3 && tf < 14.0) {
      vec3 Xf = uRo + rd * tf;
      if (length(Xf.xy - uPoolCenter.xy) < 2.2) {
        float notP = 1.0 - smoothstep(0.15, 0.62, pMask);
        float sh = calcSoftShadow(Xf + vec3(0.0, 0.0, 0.02), Lkey, 0.05, 2.4, 5.5);
        float sAmt = (1.0 - sh) * 0.58 * notP;
        if (sAmt > 0.005) {
          bgCol = mix(bgCol, bgCol * vec3(0.76, 0.78, 0.86) * 0.36, sAmt);
          cgA = max(cgA, sAmt * 0.6);
        }
      }
    }
  }

  /* 2. High-precision Raymarch 3D Rosensweig Magnetic Ferrofluid SDF (0.17 Lipschitz step -> zero tip speckles!) */
  bool hit = false;
  float tHit = tEnter;
  if (tEnter < tExit) {
    float t = tEnter;
    for (int i = 0; i < 200; i++) {
      vec3 p = uRo + rd * t;
      float d = map(p);
      if (d < 0.0012) {
        hit = true;
        tHit = t;
        break;
      }
      t += max(d * 0.17, 0.0022);
      if (t > tExit) break;
    }
  }

  vec3 col = bgCol;
  if (hit) {
    vec3 p = uRo + rd * tHit;
    float camDepth = dot(p - uRo, uCam[2]);
    float behindMask = (camDepth > uPersonDepth + 0.04) ? (1.0 - smoothstep(0.12, 0.58, pMask)) : 1.0;

    if (behindMask > 0.01) {
      vec3 N = calcNormal(p);
      vec3 V = -rd;
      float NdotV = max(dot(N, V), 0.0);
      vec3 R = reflect(-V, N);

      /* Deep valley AO between spikes makes every spike tip gleam against pitch-black obsidian valleys! */
      float ao = clamp((map(p + N * 0.025) / 0.025) * 0.55 + (map(p + N * 0.075) / 0.075) * 0.45, 0.05, 1.0);
      float selfSh = calcSoftShadow(p + N * 0.012, Lkey, 0.025, 1.6, 8.0);

      /* AUTHENTIC JET-BLACK OIL / FERROFLUID DIELECTRIC CLEARCOAT (F0 = 0.05 -> jet-black center, glossy mirror edges!) */
      float fresnel = mix(0.055, 0.92, pow(1.0 - NdotV, 4.2));
      vec3 envRef = sampleChromeEnv(R);

      /* Razor-sharp wet clearcoat specular pins along spike ridges & tips! */
      vec3 Hkey = normalize(Lkey + V);
      vec3 Hrim = normalize(Lrim + V);
      float specKey = pow(max(dot(N, Hkey), 0.0), 180.0) * 4.5 * mix(0.20, 1.0, selfSh);
      float specSheen = pow(max(dot(N, Hkey), 0.0), 34.0) * 0.34 * mix(0.20, 1.0, selfSh);
      float specRim = pow(max(dot(N, Hrim), 0.0), 110.0) * 2.4;

      /* Subtle obsidian oil iridescence on razor grazing angles */
      vec3 oilTint = mix(vec3(0.84, 0.92, 1.06), vec3(1.06, 0.92, 0.80), 0.5 + 0.5 * sin(NdotV * 11.0 + p.z * 6.0));

      vec3 obsidianBase = vec3(0.004, 0.005, 0.007) * (0.30 + 0.70 * max(dot(N, Lkey), 0.0));
      vec3 linCol = obsidianBase * ao
                  + envRef * fresnel * oilTint * (0.22 + 0.78 * ao) * 0.85
                  + vec3(1.00, 0.97, 0.92) * (specKey + specSheen)
                  + vec3(0.88, 0.95, 1.05) * specRim * ao;

      /* Clean razor-thin rim gleam when foreground pitch-black ferrofluid crosses her black shirt */
      float edgePop = pow(1.0 - NdotV, 5.2) * (0.32 + 0.45 * ao);
      linCol += vec3(0.72, 0.78, 0.86) * edgePop;

      vec3 ferroSrgb = pow(clamp(acesFilm(linCol), 0.0, 1.0), vec3(1.0 / 2.2));
      /* Smooth liquid contact alpha at very bottom puddle edge (z < 0.004) so ink puddle blends seamlessly onto bricks! */
      float edgeAlpha = behindMask * smoothstep(0.001, 0.010, p.z + 0.008);
      col = mix(bgCol, ferroSrgb, edgeAlpha);
      cgA = max(cgA, edgeAlpha);
    }
  }

  /* Optical camera match: subtle black-floor lift + plate sensor grain ONLY on rendered ferrofluid/shadow pixels */
  if (cgA > 0.01) {
    vec3 graded = max(col, vec3(uBlackFloor));
    vec2 p1 = gl_FragCoord.xy + vec2(uFrame * 17.3, uFrame * 31.7);
    vec2 p2 = floor(gl_FragCoord.xy * 0.5) + vec2(uFrame * 11.1, uFrame * 23.9);
    float g = (hash21(p1)+hash21(p1+7.13)-1.0)*0.62 + (hash21(p2)+hash21(p2+5.71)-1.0)*0.38;
    float lum = dot(graded, vec3(0.299, 0.587, 0.114));
    graded += vec3(g) * (uGrainAmt * (1.10 - 0.55 * smoothstep(0.55, 0.98, lum)));
    col = mix(col, graded, cgA);
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

    L = np.load(work(c, "limbs.npz"))
    P_body = np.asarray(L["P"], dtype=np.float64)
    p_depths = np.asarray(L["depth"], dtype=np.float64)

    orbs, handL, handR, feetL, feetR, energy = sim_ferrofluid_orbs(P_body, fps)
    pool_xy = np.median(0.5 * (feetL + feetR), axis=0)
    pool_center = (float(pool_xy[0]), float(pool_xy[1] - 0.14), 0.0)
    print(f"Ferrofluid Magnetic Spikes: {N}f @ {fps:.2f}fps | pool={pool_center} | median depth={np.median(p_depths):.2f}m")

    gray0 = cv2.cvtColor(pl0, cv2.COLOR_BGR2GRAY).astype(np.float32) / 255.0
    black_floor = float(np.clip(np.percentile(gray0, 1.0) * 0.85, 0.008, 0.045))
    hp0 = gray0 - cv2.GaussianBlur(gray0, (0, 0), 1.2)
    grain_amt = float(np.clip(float(np.median(np.abs(hp0 - np.median(hp0)))) * 1.4826 * 1.10, 0.008, 0.022))

    ctx = moderngl.create_standalone_context(require=330)
    prog = ctx.program(vertex_shader=VS_FSQ, fragment_shader=FERRO_FS)
    fsq = ctx.buffer(np.array([-1, -1, 1, -1, -1, 1, -1, 1, 1, -1, 1, 1], dtype=np.float32).tobytes())
    vao = ctx.vertex_array(prog, [(fsq, "2f", "p")])
    t_out = ctx.texture((W, H), 4, dtype="f4")
    fbo = ctx.framebuffer([t_out])
    plate_t = ctx.texture((W, H), 3); plate_t.filter = (moderngl.LINEAR, moderngl.LINEAR); plate_t.repeat_x = plate_t.repeat_y = False
    mask_t = ctx.texture((W, H), 1, dtype="f4"); mask_t.filter = (moderngl.LINEAR, moderngl.LINEAR); mask_t.repeat_x = mask_t.repeat_y = False

    prog["uRes"].value = (float(W), float(H))
    prog["uF"].value = f_px
    prog["uPoolCenter"].value = pool_center
    prog["uBlackFloor"].value = black_floor
    prog["uGrainAmt"].value = grain_amt

    def render_one(i):
        pl = frame(c, i)
        m_raw = cv2.imread(str(work(c, "mask", f"{i:04d}.png")), cv2.IMREAD_GRAYSCALE)
        m_f = cv2.GaussianBlur(m_raw.astype(np.float32) / 255.0, (0, 0), 0.9) if m_raw is not None else np.zeros((H, W), np.float32)
        plate_t.write(pl[:, :, ::-1].tobytes())
        mask_t.write(m_f.astype(np.float32).tobytes())

        ro = C[i]; R = R_all[i]
        Rt = tuple(R.T.flatten().astype(np.float32))
        fbo.use(); ctx.viewport = (0, 0, W, H); ctx.disable(moderngl.DEPTH_TEST); ctx.disable(moderngl.BLEND)
        plate_t.use(0); mask_t.use(1)
        prog["uPlate"].value = 0
        prog["uPersonMask"].value = 1
        prog["uRo"].value = tuple(map(float, ro))
        prog["uCam"].value = Rt
        prog["uTime"].value = float(i / fps)
        prog["uFrame"].value = float(i)
        prog["uPersonDepth"].value = float(p_depths[i])
        prog["uEnergy"].value = float(energy[i])
        prog["uHandL"].value = tuple(map(float, handL[i]))
        prog["uHandR"].value = tuple(map(float, handR[i]))
        prog["uFootL"].value = tuple(map(float, feetL[i]))
        prog["uFootR"].value = tuple(map(float, feetR[i]))
        prog["uOrbs"].write(orbs[i].tobytes())
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
            cv2.putText(b, f"ferrofluid f={idx}", (10, 344), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 0, 0), 3)
            cv2.putText(b, f"ferrofluid f={idx}", (10, 344), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (200, 230, 255), 1)
            rows.append(np.hstack([a, b]))
        out_p = wk / "still_ferro_sheet.jpg"
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
                print(f"ferrofluid frame {i}/{N} ({time.time() - ts:.1f}s)", flush=True)
        out_dir = here / "out"; out_dir.mkdir(parents=True, exist_ok=True)
        r = f"{c['fps_num']}/{c['fps_den']}"; ff = find_ffmpeg()
        tags = ["-pix_fmt", "yuv420p", "-color_primaries", "bt709", "-color_trc", "bt709", "-colorspace", "bt709", "-movflags", "+faststart"]
        x264 = ["-c:v", "libx264", "-profile:v", "high", "-level", "4.1", "-crf", "14", "-preset", "slow"]
        run = lambda a: subprocess.run([ff, "-v", "error", "-y"] + a, check=True)
        run(["-framerate", r, "-i", str(comp_dir / "%04d.png")] + x264 + tags + [str(out_dir / "Ferrofluid_Magnetic_Spikes_Composite.mp4")])
        run([
            "-framerate", r, "-i", str(comp_dir / "%04d.png"),
            "-framerate", r, "-i", str(work(c, "frames", "%04d.jpg")),
            "-filter_complex", "[0]scale=1080:608[a];[1]scale=1080:608[b];[a][b]vstack,pad=1080:1350:0:67:color=black",
        ] + x264 + tags + [str(out_dir / "Ferrofluid_Magnetic_Spikes_Compare_1080x1350.mp4")])
        print("encoded ->", out_dir)


if __name__ == "__main__":
    main()
