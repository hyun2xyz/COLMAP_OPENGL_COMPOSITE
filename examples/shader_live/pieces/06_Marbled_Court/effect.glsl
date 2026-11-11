// Marbled Court: wet marbling ink spreads over the tan paint of a basketball court.
// Plane coords: origin = centre circle, x along the court, z up. The ink replaces only tan paint (chroma key on the plate),
// keeps the plate's own shading (hoop shadows, wear) and gets a sun glint from its own height field.
// Technique reference: domain-warped fbm ("warping" as used in many Shadertoy marble/ink shaders). Code written for this piece.

vec2 warp1(vec2 p, float t){ return vec2(fbm(p + vec2(0.0, t)), fbm(p + vec2(5.2, 1.3 - t))); }
float ink(vec2 q, float t, out vec2 w2){
  vec2 p = q * 0.16;
  vec2 a = warp1(p, t);
  w2 = vec2(fbm(p + 3.2 * a + vec2(1.7, 9.2) + 0.6 * t), fbm(p + 3.2 * a + vec2(8.3, 2.8) - 0.5 * t));
  return fbm(p + 3.6 * w2);
}
vec3 pigment(float v){
  // five pigments in bands, with thin cream veins between them
  vec3 c0 = vec3(0.62, 0.05, 0.16), c1 = vec3(0.95, 0.42, 0.06), c2 = vec3(0.93, 0.86, 0.70), c3 = vec3(0.02, 0.30, 0.33), c4 = vec3(0.10, 0.07, 0.20);
  float s = fract(v * 2.6) * 5.0; float i = floor(s), f = fract(s);
  vec3 a = i < 0.5 ? c0 : i < 1.5 ? c1 : i < 2.5 ? c2 : i < 3.5 ? c3 : c4;
  vec3 b = i < 0.5 ? c1 : i < 1.5 ? c2 : i < 2.5 ? c3 : i < 3.5 ? c4 : c0;
  vec3 col = mix(a, b, smoothstep(0.80, 0.98, f));
  float vein = smoothstep(0.06, 0.0, abs(f - 0.89)); return mix(col, vec3(0.96, 0.92, 0.82), vein * 0.75);
}
void effect(in vec3 ro, in vec3 rd, in vec2 px, inout FX fx){
  float t = hitPlane(ro, rd, 0.0); if (t <= 0.0) return;
  vec2 q = (ro + rd * t).xy;
  vec3 pc = plate(px); float L = luma(pc);
  // tan paint key: chroma close to the court's tan, bright enough to be paint and not asphalt
  vec3 ch = pc / max(pc.r + pc.g + pc.b, 1e-4);
  float key = (1.0 - smoothstep(0.035, 0.075, distance(ch, vec3(0.442, 0.350, 0.208)))) * smoothstep(0.05, 0.10, L);
  // the ink front: starts at the centre circle and floods the court, ragged edge
  float tt = clamp(iTime / (uDur * 0.82), 0.0, 1.0); float R = mix(1.2, 34.0, tt * tt * (3.0 - 2.0 * tt));
  float d = length(q) + 5.0 * (fbm(q * 0.22 + 3.0) - 0.5) + 1.6 * (fbm(q * 0.9) - 0.5);
  float front = smoothstep(R, R - 1.2, d);
  if (key * front < 0.002) return;
  vec2 w2; float tm = iTime * 0.045;
  float v = ink(q, tm, w2);
  vec3 col = pigment(v + 0.15 * w2.x);
  // wet paint: normal from the ink field, sun glint + a little sky sheen
  float e = 0.12; vec2 wd; float vx = ink(q + vec2(e, 0.0), tm, wd), vy = ink(q + vec2(0.0, e), tm, wd);
  vec3 n = normalize(vec3(-(vx - v) / e * 0.55, -(vy - v) / e * 0.55, 1.0));
  vec3 h = normalize(uSun - rd); float spec = pow(max(dot(n, h), 0.0), 220.0) * 2.2 + pow(max(dot(n, h), 0.0), 24.0) * 0.10;
  float fres = 0.04 + 0.96 * pow(1.0 - max(dot(n, -rd), 0.0), 5.0);
  float shade = clamp(L / 0.36, 0.25, 1.25);                       // keep the plate's shadows and wear
  float diff = 0.75 + 0.25 * max(dot(n, uSun), 0.0) / max(uSun.z, 0.2);
  col = col * 0.80 * shade * diff + (spec * vec3(1.0, 0.95, 0.85) + fres * vec3(0.35, 0.45, 0.6) * 0.5) * shade;
  float edge = smoothstep(R - 2.4, R - 1.2, d);                     // darker, thicker rim at the advancing front
  col *= 1.0 - 0.35 * edge;
  fx.over = col; fx.a = key * front * 0.96;
}
