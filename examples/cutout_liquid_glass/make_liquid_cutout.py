"""Cutout-to-Liquid-Glass OpenGL Sculptor (inspired by @byrottenbird transparent glass silhouettes).

Takes ANY background photo + ANY cutout subject image (transparent RGBA PNG, explicit mask,
or automatic MediaPipe cutout / same-image auto-cutout + inpaint) and transforms the subject
into a 100% transparent, thick sculpted blown-glass / viscous liquid-water volume rendered in
native OpenGL 3.30 (ModernGL) with:
  - Multi-scale 3D inflated thickness + bilateral internal fold/hair relief + organic melt flow
  - Dual-interface thick-glass optical refraction (IOR 1.50) with subtle grazing chromatic dispersion
  - Plate-derived high-contrast HDR room reflection (blazing window highlights + deep dark glass pockets)
  - Subtle floor caustic/shadow grounding + optical halation + measured photographic sensor grain

Usage:
  python make_liquid_cutout.py --bg bg.jpg --subject subj.jpg --mask mask.png --out result.png
  python make_liquid_cutout.py --bg bg.jpg --subject cutout_rgba.png --out result.png
  python make_liquid_cutout.py --bg photo.jpg --same-image --out result.png
  python make_liquid_cutout.py --demo   # renders 3 showcase examples into examples/cutout_liquid_glass/out/
"""
import argparse
import math
import pathlib
import sys
import numpy as np
import cv2
import moderngl

REPO = pathlib.Path(__file__).resolve().parents[2]
HERE = pathlib.Path(__file__).resolve().parent


FS_QUAD_VS = """#version 330
in vec2 inPos;
out vec2 vUv;
void main() {
  vUv = inPos * 0.5 + 0.5;
  gl_Position = vec4(inPos, 0.0, 1.0);
}
"""


LIQUID_CUTOUT_FS = """#version 330
in vec2 vUv;
out vec4 oCol;

uniform sampler2D uBg;       /* top-down RGB background plate */
uniform sampler2D uEnv;      /* blurred/high-contrast spherical reflection map built from uBg */
uniform sampler2D uHeight;   /* R=3D thickness+relief height, G=clean melted alpha, B=edge thickness, A=internal detail */
uniform vec2 uRes;
uniform float uRefract;
uniform float uRelief;
uniform float uContrast;
uniform float uSeed;
uniform vec2 uFootUv;        /* approximate bottom-center UV for subtle floor caustic/shadow */
uniform float uFootW;        /* approximate width in UV */
uniform float uHasGround;

float hash12(vec2 p) {
  vec3 p3 = fract(vec3(p.xyx) * 0.1031);
  p3 += dot(p3, p3.yzx + 33.33);
  return fract((p3.x + p3.y) * p3.z);
}

vec3 samplePlate(vec2 uv) {
  /* Single-fold clamp avoids edge streaks while keeping strong refractions organic */
  vec2 c = clamp(uv, vec2(-0.45), vec2(1.45));
  c = 1.0 - abs(1.0 - abs(c));
  return texture(uBg, clamp(c, vec2(0.002), vec2(0.998))).rgb;
}

/* Plate-derived HDR spherical environment reflection:
   Produces the authentic thick blown-glass contrast of @byrottenbird:
   deep obsidian/graphite liquid shadow bands directly beside razor-crisp pure white window specular ribbons,
   tinted subtly by the actual scene environment. */
vec3 sampleGlassEnv(vec3 R, vec2 uv) {
  vec2 rUv = uv + vec2(R.x, -R.y) * (0.48 / max(0.32 + abs(R.z), 0.32));
  vec2 c = 1.0 - abs(1.0 - abs(clamp(rUv, vec2(-0.85), vec2(1.85))));
  vec3 rawSharp = texture(uBg, clamp(c, vec2(0.003), vec2(0.997))).rgb;
  vec3 rawSoft  = texture(uEnv, clamp(c, vec2(0.003), vec2(0.997))).rgb;
  vec3 env = mix(rawSharp, rawSoft, 0.22);

  float lum = dot(env, vec3(0.299, 0.587, 0.114));
  /* Deep architectural room contrast: glass reflecting dark ceilings/beams/shadowed walls drops
     to deep glossy charcoal/obsidian, while bright windows/sky explode into crisp white ribbons! */
  float az = atan(R.x, max(abs(R.z), 0.05));
  /* Soft architectural reflection bands (ceiling beam shadow, window bright panel, dark lower room) */
  float roomBands = smoothstep(-0.35, 0.45, sin(az * 2.2 + R.y * 3.1) * 0.55 + cos(R.y * 4.2 - 0.6) * 0.45);
  float darkPocket = mix(0.05, 0.85, pow(smoothstep(0.18, 0.86, lum), 1.65)) * mix(0.12, 1.0, roomBands);

  /* Razor-sharp pure white HDR window & overhead specular ribbons */
  float hiWindow = pow(smoothstep(0.62, 0.96, lum), 2.1) * 2.65;
  float spec1 = pow(max(0.0, dot(R, normalize(vec3(0.34, 0.68, 0.65)))), 38.0) * 2.15;
  float spec2 = pow(max(0.0, dot(R, normalize(vec3(-0.62, 0.44, 0.65)))), 52.0) * 1.45;
  float specBand = pow(max(0.0, 1.0 - abs(dot(R, normalize(vec3(0.22, -0.78, 0.58))))), 24.0) * 0.65;

  /* Keep reflections colorless/true-to-scene (desaturate slightly in highlights like real flint glass) */
  vec3 tintEnv = mix(vec3(lum), env, 0.55);
  vec3 col = tintEnv * darkPocket;
  col += vec3(0.97, 0.99, 1.00) * (hiWindow + spec1 + spec2 + specBand);
  return col;
}

void main() {
  vec2 uv = vec2(vUv.x, 1.0 - vUv.y);
  vec2 texel = 1.0 / uRes;
  vec3 bgCol = texture(uBg, clamp(uv, vec2(0.001), vec2(0.999))).rgb;

  /* Subtle grounded glass floor caustic + delicate shadow under lowest point of figure */
  if (uHasGround > 0.5) {
    vec2 fd = vec2((uv.x - uFootUv.x) / max(uFootW * 1.15, 0.04), (uv.y - uFootUv.y) / 0.042);
    float r2 = dot(fd, fd);
    if (r2 < 5.0) {
      float groundFade = smoothstep(uFootUv.y - 0.025, uFootUv.y + 0.015, uv.y);
      float shad = exp(-r2 * 1.1) * 0.18 * groundFade;
      float caust = exp(-dot(fd - vec2(0.12, 0.25), fd - vec2(0.12, 0.25)) * 1.8) * 0.22 * groundFade;
      bgCol = bgCol * (1.0 - shad) + vec3(1.0, 0.98, 0.95) * caust * (0.45 + 0.55 * bgCol);
    }
  }

  vec4 hm = texture(uHeight, uv);
  float alpha = hm.g;
  if (alpha < 0.004) {
    oCol = vec4(bgCol, 1.0);
    return;
  }

  /* Multi-radius Sobel gradient on 3D sculpted height field for ultra-smooth, razor-clean 3D glass normals! */
  vec2 d1 = texel * 1.6;
  vec2 d2 = texel * 4.2;
  float hL1 = texture(uHeight, uv - vec2(d1.x, 0.0)).r;
  float hR1 = texture(uHeight, uv + vec2(d1.x, 0.0)).r;
  float hU1 = texture(uHeight, uv - vec2(0.0, d1.y)).r;
  float hD1 = texture(uHeight, uv + vec2(0.0, d1.y)).r;

  float hL2 = texture(uHeight, uv - vec2(d2.x, 0.0)).r;
  float hR2 = texture(uHeight, uv + vec2(d2.x, 0.0)).r;
  float hU2 = texture(uHeight, uv - vec2(0.0, d2.y)).r;
  float hD2 = texture(uHeight, uv + vec2(0.0, d2.y)).r;

  float gx = (hR1 - hL1) * 0.65 + (hR2 - hL2) * 0.35;
  float gy = (hD1 - hU1) * 0.65 + (hD2 - hU2) * 0.35;
  /* Scale gradient by aspect ratio so circular curves have isotropic normals */
  vec2 grad = vec2(gx * (uRes.x / 1000.0), gy * (uRes.y / 1000.0)) * uRelief;

  /* 3D surface normal (pointing toward viewer +Z; +X right, -Y up in image coords) */
  vec3 N = normalize(vec3(-grad.x, grad.y, 1.0));
  vec3 V = vec3(0.0, 0.0, 1.0);
  float NdotV = clamp(N.z, 0.0, 1.0);
  float slope = length(N.xy);

  /* Curvature proxy (Laplacian) highlights crisp sculptural ridges & internal glass folds */
  float lap = (hL1 + hR1 + hU1 + hD1 - 4.0 * hm.r) * 18.0;

  /* Dual-interface thick-glass optical refraction (IOR 1.50):
     Strong magnification & bending of windows/foliage/walls behind the transparent body */
  float thick = hm.b;
  vec2 refrVec = vec2(N.x, -N.y) * uRefract * (0.42 + 0.95 * thick);
  /* Second internal bounce adds thick blown-glass depth along ridges */
  vec2 refrBack = refrVec * 1.48 + vec2(grad.x, grad.y) * 0.008;
  /* Whisper-subtle optical dispersion ONLY on steepest curves (pure colorless blown glass, zero rainbow wash!) */
  vec2 disp = refrVec * (0.012 + 0.022 * slope);

  vec3 refrFront = vec3(
    samplePlate(uv + refrVec + disp).r,
    samplePlate(uv + refrVec).g,
    samplePlate(uv + refrVec - disp).b
  );
  vec3 refrDeep = samplePlate(uv + refrBack);
  vec3 refrCol = mix(refrFront, refrDeep, smoothstep(0.28, 0.85, slope) * 0.32);

  /* Subtle internal glass clarity & deep grazing TIR pockets (pure colorless thick blown glass!) */
  float tirPocket = smoothstep(0.76, 0.22, NdotV) * smoothstep(0.05, 0.55, thick);
  /* Deep obsidian total-internal-reflection curves right beside clear magnified core! */
  vec3 transCol = refrCol * mix(1.06, 0.14, tirPocket * 0.80 * uContrast);

  /* High-contrast plate-derived HDR environment reflection (dark obsidian pockets + blazing white speculars!) */
  vec3 R = reflect(-V, N);
  vec3 envRefl = sampleGlassEnv(R, uv);
  /* Secondary sharper reflection off back wall of thick glass volume */
  vec3 R2 = reflect(-V, normalize(vec3(-N.x * 0.78, -N.y * 0.78, max(N.z, 0.22))));
  vec3 envInner = sampleGlassEnv(R2, uv) * 0.45;

  /* Physical Schlick Fresnel + sculptural slope reflection so broad glossy curves read dramatically */
  float F0 = 0.065;
  float fresnel = F0 + (1.0 - F0) * pow(1.0 - NdotV, 2.25);
  float foldSpec = smoothstep(0.18, 0.82, slope) * 0.46 * uContrast;
  float reflMix = clamp(fresnel + foldSpec * 0.48, 0.0, 0.94);

  vec3 glassCol = mix(transCol, envRefl, reflMix);
  /* Add delicate inner-wall liquid-chrome glint along smooth curved sculptural ridges */
  float ridgeGlint = pow(clamp(abs(lap), 0.0, 1.0), 1.35) * smoothstep(0.15, 0.72, slope);
  glassCol += envInner * (0.24 + 0.42 * ridgeGlint) * smoothstep(0.20, 0.85, slope);

  /* Crisp white liquid-meniscus rim catchlight ONLY on light-facing silhouette contours (no uniform outline!) */
  float rimBand = smoothstep(0.66, 0.12, NdotV) * smoothstep(0.05, 0.40, alpha);
  float rimDir = pow(clamp(dot(normalize(N.xy + vec2(1e-5)), normalize(vec2(0.35, 0.88))) * 0.5 + 0.5, 0.0, 1.0), 2.0);
  glassCol += vec3(0.97, 0.99, 1.0) * (rimBand * rimDir * 0.52);

  /* Gentle optical highlight roll-off (filmic shoulder) */
  glassCol = glassCol / (1.0 + max(glassCol - 0.84, vec3(0.0)) * 0.52);

  /* Subtle dark refractive outer meniscus rim (1-2px total-internal-reflection glass edge) */
  float outerEdge = smoothstep(0.02, 0.35, alpha) * (1.0 - smoothstep(0.35, 0.88, alpha));
  glassCol = mix(glassCol, glassCol * 0.45, outerEdge * 0.32);

  /* Blend cleanly onto plate using soft anti-aliased liquid alpha */
  vec3 finalCol = mix(bgCol, clamp(glassCol, 0.0, 1.0), alpha);
  oCol = vec4(finalCol, 1.0);
}
"""


POST_FS = """#version 330
in vec2 vUv;
out vec4 oCol;

uniform sampler2D uSrc;     /* bottom-up FBO texture */
uniform sampler2D uHeight;  /* top-down mask (G=alpha) */
uniform vec2 uRes;
uniform float uBlackFloor;
uniform float uGrainAmt;
uniform float uSeed;

float hash12(vec2 p) {
  vec3 p3 = fract(vec3(p.xyx) * 0.1031);
  p3 += dot(p3, p3.yzx + 33.33);
  return fract((p3.x + p3.y) * p3.z);
}

void main() {
  vec2 uvFbo = vUv;
  vec2 uvTop = vec2(vUv.x, 1.0 - vUv.y);
  vec2 tex = 1.0 / uRes;
  vec3 c0 = texture(uSrc, uvFbo).rgb;
  float a = texture(uHeight, uvTop).g;

  /* Soft optical halation bloom ONLY around blazing white glass highlights (>0.84) */
  vec3 bloom = vec3(0.0);
  float wsum = 0.0;
  for (int i = -2; i <= 2; i++) {
    for (int j = -2; j <= 2; j++) {
      vec2 off = vec2(float(i), float(j)) * tex * 2.8;
      vec3 s = texture(uSrc, uvFbo + off).rgb;
      float w = exp(-0.28 * float(i*i + j*j));
      float hi = smoothstep(0.82, 1.00, max(s.r, max(s.g, s.b)));
      bloom += s * hi * w;
      wsum += w;
    }
  }
  bloom /= max(wsum, 1e-4);

  /* Subtle OLPF optical lens softness on CG glass so edges match the real camera lens */
  vec3 soft = (
    c0 * 4.0 +
    texture(uSrc, uvFbo + vec2(tex.x, 0.0)).rgb +
    texture(uSrc, uvFbo - vec2(tex.x, 0.0)).rgb +
    texture(uSrc, uvFbo + vec2(0.0, tex.y)).rgb +
    texture(uSrc, uvFbo - vec2(0.0, tex.y)).rgb
  ) / 8.0;
  vec3 col = mix(c0, soft, a * 0.24) + bloom * (0.22 * smoothstep(0.0, 0.4, a + 0.15));

  /* Match camera black floor & photographic sensor grain ONLY inside/near the glass cutout */
  float cgMix = smoothstep(0.02, 0.35, a);
  if (cgMix > 0.005) {
    vec3 graded = max(col, vec3(uBlackFloor));
    vec2 p1 = gl_FragCoord.xy + vec2(uSeed * 13.7, uSeed * 29.3);
    vec2 p2 = floor(gl_FragCoord.xy * 0.5) + vec2(uSeed * 7.1, uSeed * 19.9);
    float g = (hash12(p1) + hash12(p1 + 5.13) - 1.0) * 0.62 + (hash12(p2) + hash12(p2 + 3.71) - 1.0) * 0.38;
    float lum = dot(graded, vec3(0.299, 0.587, 0.114));
    graded += vec3(g) * (uGrainAmt * (1.10 - 0.60 * smoothstep(0.55, 0.98, lum)));
    col = mix(col, graded, cgMix);
  }
  oCol = vec4(clamp(col, 0.0, 1.0), 1.0);
}
"""


def extract_subject_mask_auto(img_bgr: np.ndarray) -> np.ndarray:
    """Automatic human/subject cutout ('누끼') using MediaPipe Pose Landmarker segmentation + edge polish."""
    import mediapipe as mp
    from mediapipe.tasks import python as mp_python
    from mediapipe.tasks.python import vision

    model_candidates = [
        REPO / "examples" / "fruit_physics" / "models" / "pose_landmarker_heavy.task",
        REPO / "models" / "pose_landmarker_heavy.task",
    ]
    model_path = next((p for p in model_candidates if p.exists()), None)
    if model_path is None:
        raise FileNotFoundError("pose_landmarker_heavy.task not found; pass --mask or an RGBA PNG subject.")

    opts = vision.PoseLandmarkerOptions(
        base_options=mp_python.BaseOptions(model_asset_path=str(model_path)),
        running_mode=vision.RunningMode.IMAGE,
        num_poses=1,
        output_segmentation_masks=True,
    )
    rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    with vision.PoseLandmarker.create_from_options(opts) as det:
        res = det.detect(mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb))
    if not res.segmentation_masks:
        raise RuntimeError("No subject detected by MediaPipe; provide --mask or transparent RGBA PNG.")
    m = np.asarray(res.segmentation_masks[0].numpy_view(), dtype=np.float32)
    if m.ndim == 3:
        m = m[..., 0]
    m = cv2.resize(m, (img_bgr.shape[1], img_bgr.shape[0]), interpolation=cv2.INTER_LINEAR)
    return np.clip(m, 0.0, 1.0)


def measure_plate_optics(bg_bgr: np.ndarray):
    gray = cv2.cvtColor(bg_bgr, cv2.COLOR_BGR2GRAY).astype(np.float32) / 255.0
    black_floor = float(np.clip(np.percentile(gray, 1.0) * 0.85, 0.005, 0.055))
    hp = gray - cv2.GaussianBlur(gray, (0, 0), 1.2)
    mad = float(np.median(np.abs(hp - np.median(hp))))
    grain = float(np.clip(mad * 1.4826 * 1.15, 0.006, 0.032))
    return black_floor, grain


def place_cutout_on_canvas(bg_shape, subj_bgr, subj_mask, scale=0.80, offset_x=0.0, offset_y=0.04):
    """Crop subject cleanly to its mask bbox and place it into the background frame coordinates."""
    H, W = bg_shape[:2]
    ys, xs = np.where(subj_mask > 0.25)
    if len(ys) == 0:
        raise ValueError("Cutout mask is empty.")
    y0, y1 = max(0, int(ys.min()) - 12), min(subj_bgr.shape[0], int(ys.max()) + 12)
    x0, x1 = max(0, int(xs.min()) - 12), min(subj_bgr.shape[1], int(xs.max()) + 12)
    crop_img = subj_bgr[y0:y1, x0:x1]
    crop_m = subj_mask[y0:y1, x0:x1]

    ch, cw = crop_img.shape[:2]
    target_h = max(64, int(round(H * scale)))
    s = target_h / float(ch)
    target_w = max(32, int(round(cw * s)))
    if target_w > int(W * 0.92):
        s = (W * 0.92) / float(cw)
        target_w = int(round(cw * s))
        target_h = int(round(ch * s))

    rs_img = cv2.resize(crop_img, (target_w, target_h), interpolation=cv2.INTER_AREA if s < 1 else cv2.INTER_CUBIC)
    rs_m = cv2.resize(crop_m, (target_w, target_h), interpolation=cv2.INTER_AREA if s < 1 else cv2.INTER_LINEAR)

    canvas_img = np.zeros((H, W, 3), dtype=np.uint8)
    canvas_m = np.zeros((H, W), dtype=np.float32)
    cx = int(round(W * (0.5 + offset_x)))
    cy = int(round(H * (0.5 + offset_y)))
    px0 = cx - target_w // 2
    py0 = cy - target_h // 2

    # Safe intersection clip
    sx0, sy0 = max(0, -px0), max(0, -py0)
    dx0, dy0 = max(0, px0), max(0, py0)
    cw_eff = min(target_w - sx0, W - dx0)
    ch_eff = min(target_h - sy0, H - dy0)
    if cw_eff > 0 and ch_eff > 0:
        canvas_img[dy0:dy0 + ch_eff, dx0:dx0 + cw_eff] = rs_img[sy0:sy0 + ch_eff, sx0:sx0 + cw_eff]
        canvas_m[dy0:dy0 + ch_eff, dx0:dx0 + cw_eff] = rs_m[sy0:sy0 + ch_eff, sx0:sx0 + cw_eff]
    return canvas_img, canvas_m


def build_sculpture_maps(subj_bgr: np.ndarray, mask: np.ndarray, melt_amt: float = 1.0, detail_amt: float = 0.24, seed: int = 7):
    """Build 4-channel float32 3D liquid-glass sculpture maps from a 2D cutout:
      R = Total sculpted 3D height H(x,y) [inflated 3D pillow + internal hair/fold relief + liquid wave]
      G = Organic anti-aliased melted glass alpha mask
      B = Macro volumetric thickness (0 at rim -> 1 in deep body core)
      A = Internal sculptural detail channel
    """
    H, W = mask.shape
    rng = np.random.default_rng(seed)

    # 1. Subtle organic fluid edge warp ('viscous melted glass' silhouette feel like @byrottenbird)
    if melt_amt > 0.01:
        # Low-frequency smooth displacement field + medium drip undulation
        grid_lo = rng.normal(0.0, 1.0, (max(8, H // 90), max(8, W // 90), 2)).astype(np.float32)
        grid_md = rng.normal(0.0, 1.0, (max(16, H // 34), max(16, W // 34), 2)).astype(np.float32)
        flow_lo = cv2.resize(grid_lo, (W, H), interpolation=cv2.INTER_CUBIC)
        flow_md = cv2.resize(grid_md, (W, H), interpolation=cv2.INTER_CUBIC)
        flow = flow_lo * (7.5 * melt_amt) + flow_md * (2.6 * melt_amt)
        # Apply gentle liquid flow only near silhouette edges & hair so pose stays 100% recognizable!
        yy, xx = np.mgrid[0:H, 0:W].astype(np.float32)
        mx = np.clip(xx + flow[..., 0], 0, W - 1)
        my = np.clip(yy + flow[..., 1], 0, H - 1)
        m_warped = cv2.remap(mask.astype(np.float32), mx, my, cv2.INTER_LINEAR)
        img_warped = cv2.remap(subj_bgr, mx, my, cv2.INTER_LINEAR)
    else:
        m_warped = mask.astype(np.float32)
        img_warped = subj_bgr

    # Clean, silky liquid-glass alpha contour
    m_smooth = cv2.GaussianBlur(m_warped, (0, 0), 1.8)
    alpha = np.clip((m_smooth - 0.32) / (0.66 - 0.32), 0.0, 1.0)
    alpha = alpha * alpha * (3.0 - 2.0 * alpha)

    # 2. Multi-scale 3D volumetric thickness via Euclidean Distance Transform + Gaussian inflation
    bin_m = (alpha > 0.45).astype(np.uint8)
    dist = cv2.distanceTransform(bin_m, cv2.DIST_L2, 5).astype(np.float32)
    max_d = max(float(dist.max()), 12.0)
    # Local body scale gives lush 3D cylindrical arms/legs/head AND deep smooth torso
    r_ref = max(max_d * 0.62, 18.0)
    d_norm = np.clip(dist / r_ref, 0.0, 1.35)
    h_dist = np.power(np.clip(d_norm, 0.0, 1.0), 0.50) + 0.16 * np.clip(d_norm - 1.0, 0.0, 0.35)
    h_dist = cv2.GaussianBlur(h_dist, (0, 0), max(4.5, r_ref * 0.11))

    # Multi-scale Gaussian silhouette thickness preserves delicate thin features (flying hair, fingers!)
    g_thin = cv2.GaussianBlur(alpha, (0, 0), 6.0)
    g_med  = cv2.GaussianBlur(alpha, (0, 0), 18.0)
    g_wide = cv2.GaussianBlur(alpha, (0, 0), 40.0)
    h_macro = (0.55 * h_dist + 0.14 * g_thin + 0.18 * g_med + 0.13 * g_wide) * np.clip(alpha * 1.5, 0.0, 1.0)
    h_macro = cv2.GaussianBlur(h_macro, (0, 0), 3.4)

    # 3. Broad sculptural anatomical/clothing fold relief (NO high-freq fabric/skin noise — butter-smooth glass!)
    gray = cv2.cvtColor(img_warped, cv2.COLOR_BGR2GRAY).astype(np.float32) / 255.0
    gray_clean = cv2.bilateralFilter(gray, 11, 0.18, 12.0)
    low_f = cv2.GaussianBlur(gray_clean, (0, 0), 42.0)
    mid_f = cv2.GaussianBlur(gray_clean, (0, 0), 11.0)
    fold_relief = (mid_f - low_f)
    inside = alpha > 0.6
    if inside.sum() > 50:
        std_r = float(np.std(fold_relief[inside])) + 1e-4
        fold_relief = np.tanh(fold_relief / (std_r * 2.4))
    else:
        fold_relief = np.zeros_like(gray)
    inner_mask = np.clip(dist / max(r_ref * 0.22, 6.0), 0.0, 1.0) * alpha
    h_detail = fold_relief * inner_mask * detail_amt

    # 4. Large-scale viscous liquid-glass surface swells (gives broad liquid-chrome arcs!)
    w1 = cv2.resize(rng.normal(0.0, 1.0, (max(8, H // 75), max(8, W // 75))).astype(np.float32), (W, H), interpolation=cv2.INTER_CUBIC)
    w2 = cv2.resize(rng.normal(0.0, 1.0, (max(12, H // 38), max(12, W // 38))).astype(np.float32), (W, H), interpolation=cv2.INTER_CUBIC)
    h_wave = (w1 * 0.068 + w2 * 0.024) * inner_mask * melt_amt

    h_total = np.clip(h_macro + h_detail + h_wave, 0.0, 1.50).astype(np.float32)
    h_total = cv2.GaussianBlur(h_total, (0, 0), 3.0)

    # Compute bottom-center foot UV for subtle grounded floor caustic
    ys, xs = np.where(alpha > 0.4)
    if len(ys) > 0:
        y_bot = float(np.percentile(ys, 99.0))
        near_bot = xs[ys >= y_bot - max(18.0, H * 0.03)]
        foot_x = float(np.median(near_bot)) / W if len(near_bot) > 0 else float(np.mean(xs)) / W
        foot_y = min(0.99, y_bot / H)
        foot_w = max(0.05, float(np.percentile(xs, 90) - np.percentile(xs, 10)) / W * 0.42)
    else:
        foot_x, foot_y, foot_w = 0.5, 0.9, 0.1

    maps = np.dstack([h_total, alpha.astype(np.float32), np.clip(h_macro, 0.0, 1.0).astype(np.float32), np.clip(h_detail * 0.5 + 0.5, 0.0, 1.0).astype(np.float32)])
    return maps, (foot_x, foot_y), foot_w


def render_liquid_cutout(
    bg_bgr: np.ndarray,
    subj_bgr: np.ndarray,
    subj_mask: np.ndarray,
    scale: float = 0.80,
    offset_x: float = 0.0,
    offset_y: float = 0.04,
    same_image: bool = False,
    refract: float = 0.088,
    relief: float = 5.4,
    contrast: float = 1.08,
    melt_amt: float = 0.95,
    detail_amt: float = 0.25,
    has_ground: bool = True,
    seed: int = 7,
) -> tuple[np.ndarray, np.ndarray]:
    """Render transparent liquid-glass cutout composite via ModernGL 3.30. Returns (result_bgr, placed_mask_u8)."""
    H, W = bg_bgr.shape[:2]
    if same_image:
        placed_img = subj_bgr.copy()
        placed_m = subj_mask.astype(np.float32)
    else:
        placed_img, placed_m = place_cutout_on_canvas((H, W), subj_bgr, subj_mask, scale=scale, offset_x=offset_x, offset_y=offset_y)

    maps, foot_uv, foot_w = build_sculpture_maps(placed_img, placed_m, melt_amt=melt_amt, detail_amt=detail_amt, seed=seed)
    black_floor, grain_amt = measure_plate_optics(bg_bgr)

    # Softly blurred environment texture for smooth glassy reflection rolloff
    env_bgr = cv2.GaussianBlur(bg_bgr, (0, 0), max(5.0, W * 0.008))

    ctx = moderngl.create_standalone_context(require=330)
    try:
        quad = ctx.buffer(np.array([-1, -1, 1, -1, -1, 1, -1, 1, 1, -1, 1, 1], dtype=np.float32).tobytes())
        prog_glass = ctx.program(vertex_shader=FS_QUAD_VS, fragment_shader=LIQUID_CUTOUT_FS)
        prog_post = ctx.program(vertex_shader=FS_QUAD_VS, fragment_shader=POST_FS)
        va_glass = ctx.vertex_array(prog_glass, [(quad, "2f", "inPos")])
        va_post = ctx.vertex_array(prog_post, [(quad, "2f", "inPos")])

        t_bg = ctx.texture((W, H), 3, bg_bgr[:, :, ::-1].tobytes())
        t_bg.filter = (moderngl.LINEAR, moderngl.LINEAR)
        t_bg.repeat_x = t_bg.repeat_y = False

        t_env = ctx.texture((W, H), 3, env_bgr[:, :, ::-1].tobytes())
        t_env.filter = (moderngl.LINEAR, moderngl.LINEAR)
        t_env.repeat_x = t_env.repeat_y = False

        t_h = ctx.texture((W, H), 4, maps.astype(np.float32).tobytes(), dtype="f4")
        t_h.filter = (moderngl.LINEAR, moderngl.LINEAR)
        t_h.repeat_x = t_h.repeat_y = False

        t_mid = ctx.texture((W, H), 4, dtype="f4")
        t_mid.filter = (moderngl.LINEAR, moderngl.LINEAR)
        fbo_mid = ctx.framebuffer([t_mid])

        t_out = ctx.texture((W, H), 4, dtype="f4")
        fbo_out = ctx.framebuffer([t_out])

        # Pass 1: Thick-glass sculpture refraction + plate-derived HDR reflection + dark TIR
        fbo_mid.use()
        t_bg.use(0); t_env.use(1); t_h.use(2)
        prog_glass["uBg"].value = 0
        prog_glass["uEnv"].value = 1
        prog_glass["uHeight"].value = 2
        prog_glass["uRes"].value = (float(W), float(H))
        prog_glass["uRefract"].value = float(refract)
        prog_glass["uRelief"].value = float(relief)
        prog_glass["uContrast"].value = float(contrast)
        if "uSeed" in prog_glass:
            prog_glass["uSeed"].value = float(seed)
        prog_glass["uFootUv"].value = (float(foot_uv[0]), float(foot_uv[1]))
        prog_glass["uFootW"].value = float(foot_w)
        prog_glass["uHasGround"].value = 1.0 if has_ground else 0.0
        va_glass.render(moderngl.TRIANGLES)

        # Pass 2: Optical halation bloom + OLPF lens softness + measured photographic grain
        fbo_out.use()
        t_mid.use(0); t_h.use(1)
        prog_post["uSrc"].value = 0
        prog_post["uHeight"].value = 1
        prog_post["uRes"].value = (float(W), float(H))
        prog_post["uBlackFloor"].value = float(black_floor)
        prog_post["uGrainAmt"].value = float(grain_amt)
        prog_post["uSeed"].value = float(seed)
        va_post.render(moderngl.TRIANGLES)

        ctx.finish()
        arr = np.frombuffer(t_out.read(), "f4").reshape(H, W, 4)[::-1]
        out_bgr = (np.clip(arr[..., :3], 0.0, 1.0) * 255.0 + 0.5).astype(np.uint8)[:, :, ::-1]
        return out_bgr, (np.clip(maps[..., 1], 0.0, 1.0) * 255.0 + 0.5).astype(np.uint8)
    finally:
        ctx.release()


def load_subject_and_mask(subj_path: pathlib.Path, mask_path: pathlib.Path | None):
    raw = cv2.imread(str(subj_path), cv2.IMREAD_UNCHANGED)
    if raw is None:
        raise FileNotFoundError(f"Cannot read subject image: {subj_path}")
    if mask_path is not None:
        m = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
        if m is None:
            raise FileNotFoundError(f"Cannot read mask image: {mask_path}")
        bgr = raw[:, :, :3] if raw.ndim == 3 else cv2.cvtColor(raw, cv2.COLOR_GRAY2BGR)
        m = cv2.resize(m, (bgr.shape[1], bgr.shape[0]), interpolation=cv2.INTER_LINEAR).astype(np.float32) / 255.0
        return bgr, m
    if raw.ndim == 3 and raw.shape[2] == 4:
        return raw[:, :, :3], raw[:, :, 3].astype(np.float32) / 255.0
    bgr = raw[:, :, :3] if raw.ndim == 3 else cv2.cvtColor(raw, cv2.COLOR_GRAY2BGR)
    return bgr, extract_subject_mask_auto(bgr)


def make_clean_compare(bg_bgr, subj_bgr, subj_mask, result_bgr, out_path: pathlib.Path):
    """Create a zero-text clean 3-panel showcase: [Background | Cutout Subject | Liquid Glass Result]"""
    H, W = result_bgr.shape[:2]
    tw = 640
    th = int(round(tw * H / W))
    p_bg = cv2.resize(bg_bgr, (tw, th), interpolation=cv2.INTER_AREA)

    # Render subject cutout over subtle dark checkerboard for clear visual reading
    sh, sw = subj_bgr.shape[:2]
    cb = np.zeros((sh, sw, 3), dtype=np.uint8)
    yy, xx = np.mgrid[0:sh, 0:sw]
    chk = (((yy // 24) + (xx // 24)) % 2 == 0)
    cb[chk] = (46, 46, 50)
    cb[~chk] = (32, 32, 36)
    m3 = np.clip(subj_mask[..., None], 0.0, 1.0)
    cut_vis = (subj_bgr.astype(np.float32) * m3 + cb.astype(np.float32) * (1.0 - m3)).astype(np.uint8)
    p_cut = cv2.resize(cut_vis, (tw, th), interpolation=cv2.INTER_AREA)
    p_res = cv2.resize(result_bgr, (tw, th), interpolation=cv2.INTER_AREA)

    gap = np.full((th, 6, 3), 16, dtype=np.uint8)
    row = np.hstack([p_bg, gap, p_cut, gap, p_res])
    cv2.imwrite(str(out_path), row, [cv2.IMWRITE_JPEG_QUALITY, 94])


def build_temporal_clean_tunnel():
    """Reconstruct clean graffiti tunnel background behind the dancer by masking out the dancer across frames."""
    wdir = REPO / "examples" / "space_distortion" / "piece_tunnel" / "work"
    acc = None
    wacc = None
    for idx in range(0, 216, 12):
        fr = cv2.imread(str(wdir / "frames" / f"{idx:04d}.jpg"))
        m = cv2.imread(str(wdir / "mask" / f"{idx:04d}.png"), cv2.IMREAD_GRAYSCALE)
        if fr is None or m is None:
            continue
        dil = cv2.dilate((m > 25).astype(np.uint8), cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (31, 31)))
        keep = (1.0 - dil.astype(np.float32))[..., None]
        if acc is None:
            acc = fr.astype(np.float64) * keep
            wacc = keep.astype(np.float64)
        else:
            acc += fr.astype(np.float64) * keep
            wacc += keep.astype(np.float64)
    base = cv2.imread(str(wdir / "frames" / "0090.jpg"))
    m90 = cv2.imread(str(wdir / "mask" / "0090.png"), cv2.IMREAD_GRAYSCALE)
    hole = (wacc[..., 0] < 0.5)
    clean = np.where(wacc > 0.5, acc / np.maximum(wacc, 1.0), base.astype(np.float64)).astype(np.uint8)
    if hole.any():
        clean = cv2.inpaint(clean, hole.astype(np.uint8) * 255, 18, cv2.INPAINT_TELEA)
    return clean, base, m90.astype(np.float32) / 255.0


def run_demos():
    out_dir = HERE / "out"
    out_dir.mkdir(parents=True, exist_ok=True)

    # Demo 1: Cross-image composite — Sunlit Architectural Hall with Greenery + Mid-Jump Flying-Hair Figure!
    bg1 = cv2.imread(str(REPO / "examples" / "liquid_glass" / "piece_glass" / "work" / "frames" / "0120.jpg"))
    s1, m1 = load_subject_and_mask(
        REPO / "examples" / "slit_scan" / "piece" / "work" / "frames" / "0140.jpg",
        REPO / "examples" / "slit_scan" / "piece" / "work" / "mask" / "0140.png",
    )
    res1, _ = render_liquid_cutout(
        bg1, s1, m1, scale=0.84, offset_x=0.02, offset_y=0.04,
        refract=0.115, relief=6.4, contrast=1.24, melt_amt=1.05, detail_amt=0.16, seed=11
    )
    cv2.imwrite(str(out_dir / "Demo1_SunlitHall_LiquidCutout.png"), res1)
    make_clean_compare(bg1, s1, m1, res1, out_dir / "Demo1_SunlitHall_Compare.jpg")
    print("wrote Demo 1 ->", out_dir / "Demo1_SunlitHall_LiquidCutout.png")

    # Demo 2: Cross-image composite — Bright Window Interior Room + Outstretched Sculptural Dancer!
    bg2 = cv2.imread(str(REPO / "examples" / "lidar_scan" / "piece_lidar" / "work" / "frames" / "0085.jpg"))
    s2, m2 = load_subject_and_mask(
        REPO / "examples" / "space_distortion" / "piece" / "work" / "frames" / "0020.jpg",
        REPO / "examples" / "space_distortion" / "piece" / "work" / "mask" / "0020.png",
    )
    res2, _ = render_liquid_cutout(
        bg2, s2, m2, scale=0.86, offset_x=0.00, offset_y=0.05,
        refract=0.118, relief=6.6, contrast=1.26, melt_amt=0.95, detail_amt=0.15, seed=23
    )
    cv2.imwrite(str(out_dir / "Demo2_WindowRoom_LiquidCutout.png"), res2)
    make_clean_compare(bg2, s2, m2, res2, out_dir / "Demo2_WindowRoom_Compare.jpg")
    print("wrote Demo 2 ->", out_dir / "Demo2_WindowRoom_LiquidCutout.png")

    # Demo 3: Same-image in-place transformation — Decayed Graffiti Concrete Tunnel (exact @byrottenbird raw mood!)
    bg3_clean, orig3, m3 = build_temporal_clean_tunnel()
    res3, _ = render_liquid_cutout(
        bg3_clean, orig3, m3, same_image=True,
        refract=0.112, relief=6.4, contrast=1.25, melt_amt=1.00, detail_amt=0.16, seed=37
    )
    cv2.imwrite(str(out_dir / "Demo3_GraffitiTunnel_SameImage_LiquidCutout.png"), res3)
    make_clean_compare(orig3, orig3, m3, res3, out_dir / "Demo3_GraffitiTunnel_Compare.jpg")
    print("wrote Demo 3 ->", out_dir / "Demo3_GraffitiTunnel_SameImage_LiquidCutout.png")


def main():
    ap = argparse.ArgumentParser(description="Transform any cutout image into thick transparent liquid glass.")
    ap.add_argument("--demo", action="store_true", help="Run 3 showcase examples into examples/cutout_liquid_glass/out/")
    ap.add_argument("--bg", type=pathlib.Path, help="Background photo path")
    ap.add_argument("--subject", type=pathlib.Path, help="Subject photo or RGBA cutout PNG path")
    ap.add_argument("--mask", type=pathlib.Path, default=None, help="Optional grayscale cutout mask PNG")
    ap.add_argument("--same-image", action="store_true", help="Cut out subject from --bg, inpaint bg, replace in-place")
    ap.add_argument("--scale", type=float, default=0.82)
    ap.add_argument("--offset-x", type=float, default=0.0)
    ap.add_argument("--offset-y", type=float, default=0.04)
    ap.add_argument("--refract", type=float, default=0.092)
    ap.add_argument("--relief", type=float, default=5.6)
    ap.add_argument("--contrast", type=float, default=1.12)
    ap.add_argument("--melt", type=float, default=1.0)
    ap.add_argument("--out", type=pathlib.Path, default=HERE / "out" / "liquid_cutout.png")
    ap.add_argument("--compare", type=pathlib.Path, default=None)
    args = ap.parse_args()

    if args.demo:
        run_demos()
        return

    if args.bg is None:
        ap.error("Provide --bg (or run with --demo).")
    bg = cv2.imread(str(args.bg))
    if bg is None:
        raise FileNotFoundError(f"Cannot read background: {args.bg}")

    if args.same_image:
        subj = bg.copy()
        mask = load_subject_and_mask(args.bg, args.mask)[1] if args.mask else extract_subject_mask_auto(bg)
        dil = cv2.dilate((mask > 0.2).astype(np.uint8), cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (27, 27)))
        bg_inpaint = cv2.inpaint(bg, dil * 255, 18, cv2.INPAINT_TELEA)
        res, _ = render_liquid_cutout(
            bg_inpaint, subj, mask, same_image=True,
            refract=args.refract, relief=args.relief, contrast=args.contrast, melt_amt=args.melt
        )
    else:
        if args.subject is None:
            ap.error("Provide --subject (or use --same-image).")
        subj, mask = load_subject_and_mask(args.subject, args.mask)
        res, _ = render_liquid_cutout(
            bg, subj, mask, scale=args.scale, offset_x=args.offset_x, offset_y=args.offset_y,
            refract=args.refract, relief=args.relief, contrast=args.contrast, melt_amt=args.melt
        )

    args.out.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(args.out), res)
    print("wrote ->", args.out)
    if args.compare:
        args.compare.parent.mkdir(parents=True, exist_ok=True)
        make_clean_compare(bg, subj, mask, res, args.compare)
        print("compare ->", args.compare)


if __name__ == "__main__":
    main()
