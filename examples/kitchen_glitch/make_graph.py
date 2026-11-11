"""Writes the ffmpeg filtergraph for a macroblock datamosh glitch on a moving person.
All pixel work happens inside ffmpeg; this script only writes the graph text.

What a broken P-frame decoder does, rebuilt with filters:
  1. Subject mask = motion: |frame n - frame n-3| averaged per 40 px block, held with lagfun (so the
     place the person just left keeps breaking), grown by one block, thresholded.
  2. Every macroblock refreshes on its own clock (sample-and-hold):
       L(block) = 2..LMAX frames, phase random, lag(block,n) = (n + phase) mod L -> show frame n - lag.
     Blocks refreshing at different moments leave stale copies of moving limbs (staircase trails).
  3. While a block is stale its 'motion vector' keeps being applied (displace): it slides lag * v pixels,
     v per block, mostly downward with a sideways component -> blocks drip and smear.
  4. Inside a block the picture is mosaicked (pixelize avg), like an under-coded macroblock.
  5. Bursts change the share of broken blocks over time.

usage: python make_graph.py [LMAX] [FRAMES] > glitch.ffgraph
"""
import sys
W, H, BLOCK, MOSAIC = 1080, 1920, 40, 20
LMAX = int(sys.argv[1]) if len(sys.argv) > 1 else 12
FRAMES = int(sys.argv[2]) if len(sys.argv) > 2 else 391
SPEED, DMAX, TH = 6.0, 110, 12          # drag px/frame, clamp, motion threshold (block mean |diff|)
GW, GH = W // BLOCK, H // BLOCK          # 27 x 48 control grid
STEP = 16                                # lag k is coded as grey 16*k in the control map

def h(a, b, c=""):  # hash -> [0,1) per block (optionally per time bucket)
    return f"mod(abs(sin(X*{a}+Y*{b}{c}))*43758.5453\\,1)"

L = f"(2+floor({h(39.3468, 11.135)}*{LMAX - 1}))"
lag = f"mod(N+floor({h(12.9898, 78.233)}*{L})\\,{L})"
dx = f"128+clip(round({lag}*{SPEED}*({h(5.71, 21.9)}-0.5)*1.8)\\,-{DMAX}\\,{DMAX})"
dy = f"128+clip(round({lag}*{SPEED}*(0.25+0.75*{h(17.3, 3.77)}))\\,-{DMAX}\\,{DMAX})"
inten = "(0.6+0.38*gt(sin(N*0.19)+sin(N*0.067+1.3)\\,0.8))"       # base share of broken blocks + bursts
alive = f"lt({h(7.13, 93.17, '+floor(N/4)*3.71')}\\,{inten})"
on = f"255*gt(r(X\\,Y)\\,{TH})*{alive}"

g = []
g.append(f"[0:v]scale={W}:{H}:flags=lanczos,format=gbrp,split=3[orig][g0][mk]")
g.append(f"[g0]pixelize=w={MOSAIC}:h={MOSAIC}:mode=avg,split={LMAX}" + "".join(f"[m{k}]" for k in range(LMAX)))
for k in range(1, LMAX):
    g.append(f"[m{k}]tpad=start={k}:start_mode=clone,trim=end_frame={FRAMES}[d{k}]")
g.append("[mk]format=gray,split=2[ma][mb]")
g.append(f"[mb]tpad=start=3:start_mode=clone,trim=end_frame={FRAMES}[mb3]")
g.append(f"[ma][mb3]blend=all_mode=difference,scale={GW}:{GH}:flags=area,lagfun=decay=0.88,format=gbrp,"
         f"split=4[ag][clk][cx][cy]")
g.append(f"[clk]geq=r='{STEP}*{lag}':g='{STEP}*{lag}':b='{STEP}*{lag}',split={LMAX - 1}" + "".join(f"[q{k}]" for k in range(1, LMAX)))
for k in range(1, LMAX):
    e = f"255*eq(val\\,{STEP * k})"
    g.append(f"[q{k}]lutrgb=r='{e}':g='{e}':b='{e}',scale={W}:{H}:flags=neighbor[s{k}]")
prev = "m0"
for k in range(1, LMAX):
    out = "held" if k == LMAX - 1 else f"o{k}"
    g.append(f"[{prev}][d{k}][s{k}]maskedmerge[{out}]")
    prev = out
g.append(f"[cx]geq=r='{dx}':g='{dx}':b='{dx}',scale={W}:{H}:flags=neighbor,negate[xm]")
g.append(f"[cy]geq=r='{dy}':g='{dy}':b='{dy}',scale={W}:{H}:flags=neighbor,negate[ym]")
g.append("[held][xm][ym]displace=edge=smear[gl]")
g.append(f"[ag]geq=r='{on}':g='{on}':b='{on}',dilation,split=2[dg][dbg]")
g.append(f"[dg]scale={W}:{H}:flags=neighbor[dm]")
g.append(f"[dbg]scale={W // 4}:{H // 4}:flags=neighbor,format=gray[maskdbg]")
g.append("[orig][gl][dm]maskedmerge,format=yuv420p[v]")
print(";\n".join(g))
