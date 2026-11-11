// Lava Fissures: the drive lane of a car park cracks open from one point; lava sits 0.7 m down in the cracks.
// Plane coords: origin on the lane, x along the lane, z up, metres. The cracks are a Voronoi border field marched below
// the surface along the real view ray, so the crack walls show true parallax as the drone slides and climbs.
// Technique reference: Voronoi border distance (iq's "voronoi edges" construction, used by many Shadertoy crack / lava shaders)
// and fbm lava flow. Code written for this piece.

const float CELL = 2.9;      // slab size, metres
const float DEPTH = 0.75;    // lava level below the asphalt
vec3 voro(vec2 x){           // x: border distance (cell units), yz: cell id
  vec2 n = floor(x), f = fract(x), mg = vec2(0.0), mr = vec2(0.0); float md = 8.0;
  for (int j = -1; j <= 1; j++) for (int i = -1; i <= 1; i++){
    vec2 g = vec2(float(i), float(j)); vec2 r = g + hash22(n + g) - f; float d = dot(r, r);
    if (d < md){ md = d; mr = r; mg = g; } }
  md = 8.0;
  for (int j = -2; j <= 2; j++) for (int i = -2; i <= 2; i++){
    vec2 g = mg + vec2(float(i), float(j)); vec2 r = g + hash22(n + g) - f;
    if (dot(mr - r, mr - r) > 1e-5) md = min(md, dot(0.5 * (mr + r), normalize(r - mr))); }
  return vec3(md, hash22(n + mg));
}
float lane(vec2 q){          // 1 on the drive lane, 0 in the parking bays (cars stay untouched)
  float e = 0.9 * (fbm(q * 0.35) - 0.5);
  return smoothstep(3.3, 2.3, q.y + e) * smoothstep(-2.6, -1.7, q.y + e) * smoothstep(13.5, 9.0, q.x + e);
}
float grow(vec2 q){          // 0 -> 1 as the fracture front passes
  float d = length(q * vec2(1.0, 1.6)) + 2.5 * (fbm(q * 0.4 + 7.0) - 0.5);
  return smoothstep(0.0, 3.0, 1.0 + iTime * 4.6 - d);
}
float crackWidth(vec2 q){    // half width at the surface, metres
  return (0.17 + 0.30 * fbm(q * 0.6 + 2.0)) * grow(q) * lane(q);
}
float border(vec2 q, out vec2 id){
  vec2 w = q + 0.35 * vec2(fbm(q * 1.1), fbm(q * 1.1 + 4.7)) - 0.17;
  vec3 v = voro(w / CELL); id = v.yz; float d = v.x * CELL;
  float fine = voro(w / (CELL * 0.31) + 3.0).x * CELL * 0.31;            // hairline cracks near the big ones
  return min(d, fine + 0.10 + 0.9 * smoothstep(0.0, 1.2, d));
}
vec3 lava(vec2 p){
  float t = iTime * 0.35;
  float a = fbm(vec3(p * 2.2 + vec2(t, 0.0), t * 0.7)); float b = fbm(vec3(p * 5.0 - vec2(0.0, t * 1.3), t));
  float T = clamp(1.7 * a + 0.8 * b - 0.75, 0.0, 1.3);
  vec3 c = mix(vec3(0.03, 0.004, 0.0), vec3(0.9, 0.06, 0.0), smoothstep(0.05, 0.45, T));
  c = mix(c, vec3(1.9, 0.55, 0.04), smoothstep(0.40, 0.80, T)); c = mix(c, vec3(3.4, 2.2, 0.7), smoothstep(0.80, 1.15, T));
  return c;
}
void effect(in vec3 ro, in vec3 rd, in vec2 px, inout FX fx){
  float t = hitPlane(ro, rd, 0.0); if (t <= 0.0 || rd.z > -0.05) return;
  vec2 q = (ro + rd * t).xy; float g = grow(q) * lane(q); if (g < 0.001) return;
  vec2 id; float bd = border(q, id); float w = crackWidth(q);
  vec3 glowCol = vec3(1.0, 0.33, 0.05);
  float flick = 0.85 + 0.3 * vnoise(vec3(q * 0.7, iTime * 3.0));
  if (bd < w){
    // inside a crack mouth: march down along the view ray until a wall or the lava
    vec2 dxy = rd.xy / (-rd.z); float hitZ = DEPTH; bool wall = false; vec2 p = q;
    for (int i = 1; i <= 28; i++){
      float z = DEPTH * float(i) / 28.0; p = q + dxy * z; vec2 id2;
      float wz = crackWidth(p) * (1.0 - 0.62 * z / DEPTH) + 0.02 * (vnoise(p * 9.0 + z * 5.0) - 0.5);
      if (border(p, id2) > wz){ hitZ = z; wall = true; break; }
    }
    vec3 col;
    if (wall){
      float k = hitZ / DEPTH;                                              // 0 at the rim, 1 at the lava
      vec3 rock = vec3(0.035, 0.03, 0.028) * (0.6 + 0.8 * vnoise(p * 14.0 + hitZ * 9.0));
      col = rock * (1.0 - k) * 2.0 * luma(plate(px)) / 0.12 + glowCol * (0.05 + 1.5 * k * k) * flick;   // daylight at the top, lava light below
    } else col = lava(p) * flick;
    fx.over = col; fx.a = smoothstep(0.0, 0.03, w - bd);
  }
  // scorched rim, slab tilt, glow spill and heat shimmer on the asphalt around the cracks
  float e = max(bd - w, 0.0);
  float tilt = (id.x - 0.5) * 0.22 * g;                                    // each slab catches the light a little differently
  fx.mul = vec3(1.0 + tilt) * (1.0 - 0.55 * g * exp(-e / 0.10));
  fx.add = glowCol * (0.42 * exp(-e / 0.16) + 0.07 * exp(-e / 0.7)) * g * flick * (1.0 - fx.a) * step(0.001, w);
  fx.warp = 2.6 * g * exp(-e / 0.8) * (vec2(vnoise(vec3(q * 3.0, iTime * 2.5)), vnoise(vec3(q * 3.0 + 9.1, iTime * 2.5))) - 0.5);
}
