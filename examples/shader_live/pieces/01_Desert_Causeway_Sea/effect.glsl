// Desert Causeway Sea: the sand flat on the left of a desert road becomes open sea, lit by the plate's own low sun.
// Plane coords: origin on the centre line, x along the road, y to the left, z up, metres. Road half width 3.3 m.
// The water mirrors the REAL sky: each reflected ray is looked up in the plate, so the orange horizon glow and the sun glitter
// line up with the sun that is actually in frame (uSun is that sun's direction, read from its pixel).
// Technique reference: procedural ocean with summed sharp-crested wave octaves and Fresnel sky reflection
// (the approach popularised by Shadertoy sea shaders such as "Seascape"). Code written for this piece.

const float SEA = -0.35;                       // water level below the road, metres
// the coast swings away from the road with distance (0.125 m per m), so the roadside banner and bushes, whose bearing from the
// camera stays under 0.11, are never in front of water and need no holdout matte
float shoreY(float x){ return 6.9 + 1.4 * sin(x * 0.07 - 2.6) + 1.6 * (fbm(vec2(x * 0.13, 3.0)) - 0.5) + 0.125 * max(x - 35.0, 0.0); }
float wave(vec2 p, float t){
  float h = 0.0, a = 0.050, f = 0.85; vec2 d = normalize(vec2(-0.35, -1.0));        // swell runs towards the road
  for (int i = 0; i < 5; i++){
    float ph = dot(p, d) * f + t * sqrt(9.8 * f) + 2.2 * vnoise(p * f * 0.33 + float(i) * 7.1);
    h += a * (1.0 - abs(sin(ph))); d = mat2(0.80, -0.60, 0.60, 0.80) * d; f *= 1.87; a *= 0.50; }
  return h;
}
void effect(in vec3 ro, in vec3 rd, in vec2 px, inout FX fx){
  if (rd.z > -0.0006) return;
  float t = hitPlane(ro, rd, SEA); vec3 P = ro + rd * t;
  float lap = 0.28 * sin(iTime * 1.1 + P.x * 0.35) + 0.18 * sin(iTime * 0.7 - P.x * 0.21 + 1.3);   // the waterline breathes in and out
  float ds = P.y - shoreY(P.x) - lap;                                             // metres out from the waterline
  // wet sand just above the waterline (on the real plate)
  float tg = hitPlane(ro, rd, 0.0); vec3 G = ro + rd * tg; float dg = G.y - shoreY(G.x);
  float wet = smoothstep(-1.5, -0.3, dg) * step(dg, 0.6); fx.mul = vec3(1.0 - 0.38 * wet, 1.0 - 0.34 * wet, 1.0 - 0.30 * wet);
  if (ds < 0.0) return;
  // waves: filtered with distance so the far field does not turn to noise
  float e = 0.03 + 0.0012 * t; float tt = iTime;
  float h0 = wave(P.xy, tt), hx = wave(P.xy + vec2(e, 0.0), tt), hy = wave(P.xy + vec2(0.0, e), tt);
  float calm = 1.0 / (1.0 + t * 0.0035);
  vec3 n = normalize(vec3(-(hx - h0) / e * calm, -(hy - h0) / e * calm, 1.0));
  n.xy += 0.035 * (vec2(vnoise(vec3(P.xy * 9.0, tt * 2.0)), vnoise(vec3(P.xy * 9.0 + 5.0, tt * 2.0))) - 0.5) / (1.0 + t * 0.02); n = normalize(n);
  vec3 r = reflect(rd, n); r.z = abs(r.z) + 0.004; r = normalize(r);
  // the sky the reflected ray sees = the plate, looked up at that direction; outside the frame, the nearest edge
  vec3 q = transpose(uCam) * r; vec2 sp = uF * q.xy / max(q.z, 0.05) + 0.5 * uRes;
  vec3 skyc;
  float horizonRow = project(ro + vec3(1e5, 0.0, 30.0)).y;                         // keep the lookup above the horizon line
  skyc = plate(vec2(clamp(sp.x, 4.0, uRes.x - 4.0), clamp(sp.y, 4.0, horizonRow - 3.0)));
  float sd = max(dot(r, uSun), 0.0);
  vec3 sun = vec3(1.0, 0.82, 0.55) * (26.0 * pow(sd, 2200.0) + 1.6 * pow(sd, 120.0));   // the disc is clipped in the plate: put its energy back
  float fres = 0.02 + 0.98 * pow(1.0 - max(dot(n, -rd), 0.0), 5.0);
  vec3 body = vec3(0.012, 0.050, 0.060) * (0.6 + 2.5 * h0);                         // deep water, a little lighter on the crests
  vec3 col = mix(body, skyc, fres) + sun * fres;
  // shallows: the plate's own sand shows through near the waterline; a thin foam line where the water laps
  float shallow = smoothstep(2.6, 0.0, ds); vec3 sand = plate(px) * vec3(0.55, 0.62, 0.58);
  col = mix(col, mix(sand, col, 0.35 + 0.65 * fres), shallow * 0.8);
  float foam = smoothstep(0.35, 0.0, ds) * (0.55 + 0.45 * vnoise(vec3(P.xy * 6.0, tt * 1.5))) + 0.5 * smoothstep(0.9, 1.0, vnoise(vec3(P.xy * vec2(0.5, 2.5), tt * 0.6)) * smoothstep(3.5, 0.5, ds));
  col = mix(col, luma(skyc) * vec3(1.0, 0.95, 0.88) * 0.9 + 0.05, clamp(foam, 0.0, 1.0) * 0.75);
  float haze = smoothstep(260.0, 1100.0, t);                                      // the last pixels under the horizon stay the plate's haze
  fx.over = col; fx.a = smoothstep(0.0, 0.12, ds) * (1.0 - haze); fx.mul = mix(fx.mul, vec3(1.0), fx.a);
}
