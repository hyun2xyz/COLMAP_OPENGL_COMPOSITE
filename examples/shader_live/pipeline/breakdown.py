"""Breakdown for one piece, same look as the earlier series (1080x1350 @30fps, white ground, Arial 21px captions on
white boxes, right-to-left 6-frame wipes with a 2px white edge, the two fixed footer lines).

Timeline (output frames):
  0-89      RESULT | ORIGINAL, same source frames
  then one continuous source clock through the passes
            01 camera track + plane (SfM points, plane grid)      60 f
            02 the GLSL effect alone, on the solved grid in grey  54 f
            03 the effect's RGBA pass on a checkerboard           45 f
            04 composite                                          36 f
  wipe back to RESULT source 0, hold 30 f, play the whole shot
usage: python breakdown.py <piece_dir> [still frames a,b,c]      needs work/track (slv.py track), work/fx, work/comp"""
import json, math, subprocess, sys
from pathlib import Path
import numpy as np, cv2
from PIL import Image, ImageDraw, ImageFont
from slv import load, work, frame, overlay

C = load(sys.argv[1]); only = [int(x) for x in sys.argv[2].split(",")] if len(sys.argv) > 2 else None
NSRC = C["N"]; FPS = float(C["fps"]); W, H = 1080, 1350
PIC = (0, 371, 1080, 608); CMP = [(0, 55, 1080, 608), (0, 688, 1080, 608)]; BG = (252, 254, 253); GRAY = (125, 128, 126)
import os
FONT = next((f for f in [os.environ.get("SLV_FONT", ""), "C:/Windows/Fonts/arial.ttf", "/System/Library/Fonts/Supplemental/Arial.ttf", "/Library/Fonts/Arial.ttf",
                         "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", "/usr/share/fonts/truetype/liberation/LiberationSans-Regular.ttf"] if f and os.path.exists(f)), None)
if FONT is None: sys.exit("No caption font found. Set SLV_FONT to a .ttf file.")
LABELS = C.get("labels") or ["01 CAMERA TRACK (SIFT HOMOGRAPHY) + WORLD PLANE" if C.get("solver") == "homography" else "01 CAMERA TRACK (COLMAP) + WORLD PLANE", "02 WORLD-LOCKED GLSL SHADER (OpenGL)", "03 SHADER ALPHA (RGBA PASS)", "04 PLATE INTERACTION / COMPOSITE"]
FOOTER = ["Only an LLM + Using My Custom Skills (Colmap + OpenGL, etc.)", "See Caption for Details!"]
src_of = lambda o: min(int(math.floor(o * FPS / 30 + 1e-6)), NSRC - 1)
WIPE = 6; PASSES = [("track", 60), ("grey", 54), ("alpha", 45), ("result", 36)]
P0 = 90; STARTS = [int(x) for x in np.cumsum([P0] + [d for _, d in PASSES])]; P1 = STARTS[-1]
RES_START = P1 + WIPE + 30; TOTAL = RES_START + int(math.ceil(NSRC * 30 / FPS))

class RectLine:
    def __init__(self, text, size=21):
        self.ss = 4; self.font = ImageFont.truetype(FONT, size * self.ss); self.text = text
        self.width = math.ceil(ImageDraw.Draw(Image.new("L", (1, 1))).textlength(text, font=self.font) / self.ss) + 6; self.cache = {}
    def render(self, color):
        if color not in self.cache:
            fg = Image.new("RGBA", (self.width * self.ss, 24 * self.ss))
            ImageDraw.Draw(fg).text((3 * self.ss, 18 * self.ss), self.text, font=self.font, anchor="ls", fill=(*color, 255))
            fg = fg.resize((self.width, 24), Image.Resampling.LANCZOS); bg = Image.new("RGBA", (self.width, 24), (255, 255, 255, 255)); bg.alpha_composite(fg)
            self.cache[color] = np.asarray(bg)
        return self.cache[color]
lines = {t: RectLine(t) for t in ["RESULT", "ORIGINAL", "Process", *LABELS, *FOOTER]}
def layer(stage):
    lay = np.zeros((H, W, 4), np.uint8)
    def place(t, x, y, color=(0, 0, 0), right=False):
        L = lines[t]; x = x - L.width if right else x; lay[y:y + 24, x:x + L.width] = L.render(color)
    if stage == "intro":
        for t, (x, y, w, h) in zip(["RESULT", "ORIGINAL"], CMP): place(t, x + 24, y + 24)
        x, y, w, h = CMP[1]
    else:
        x, y, w, h = PIC; place("RESULT" if stage == "result" else "Process", x + 24, y + 24)
        if stage != "result":
            for j in range(int(stage) + 1): place(LABELS[j], x + 40, y + 51 + 27 * j, (0, 0, 0) if j == int(stage) else GRAY)
    for k, t in enumerate(FOOTER): place(t, x + w - 24, y + h - 24 - 24 - 27 + 27 * k, right=True)
    return lay
LAYERS = {s: layer(s) for s in ["intro", *map(str, range(len(LABELS))), "result"]}

def fit(im, box=PIC): return cv2.cvtColor(cv2.resize(im[..., :3], (box[2], box[3]), interpolation=cv2.INTER_AREA), cv2.COLOR_BGR2RGB)
def rd(p):
    im = cv2.imread(str(p), cv2.IMREAD_UNCHANGED)
    if im is None: raise FileNotFoundError(p)
    return im
def over(bg, fx): a = fx[..., 3:].astype(np.float32) / 255; return np.clip(bg * (1 - a) + fx[..., :3] * a, 0, 255).astype(np.uint8)
def grey(j):
    bg = overlay(C, j, None, pts=False, bg=np.full((C["H"], C["W"], 3), 96, np.uint8)); return over(bg.astype(np.float32), rd(work(C, "fx", f"{j:04d}.png")))
def checker(j, n=40):
    yy, xx = np.mgrid[0:C["H"], 0:C["W"]]; ch = np.where(((yy // n + xx // n) % 2)[..., None] == 0, 205, 245).astype(np.float32).repeat(3, 2)
    return over(ch, rd(work(C, "fx", f"{j:04d}.png")))
src = {"original": lambda j: fit(frame(C, j)), "result": lambda j: fit(rd(work(C, "comp", f"{j:04d}.png"))),
       "track": lambda j: fit(rd(work(C, "track", f"{j:04d}.jpg"))), "grey": lambda j: fit(grey(j)), "alpha": lambda j: fit(checker(j))}
def wipe(old, new, k):
    u = min(1.0, max(0.0, k / (WIPE - 1))); e = round(PIC[2] * (1 - u))
    if u <= 0: return old.copy()
    if u >= 1: return new.copy()
    pic = old.copy(); pic[:, e:] = new[:, e:]; pic[:, max(0, e - 1):e + 1] = 255; return pic

def make(i):
    canvas = np.full((H, W, 3), BG, np.uint8)
    if i < P0:
        j = src_of(i); stage = "intro"
        for (x, y, w, h), im in zip(CMP, [rd(work(C, "comp", f"{j:04d}.png")), frame(C, j)]): canvas[y:y + h, x:x + w] = fit(im, (x, y, w, h))
    else:
        if i < P1:
            k = int(np.searchsorted(STARTS, i, side="right") - 1); jj = i - STARTS[k]; s_ = src_of(i - P0); name = PASSES[k][0]; stage = str(k); a = src[name](s_)
            if jj < WIPE and k > 0: a = wipe(src[PASSES[k - 1][0]](s_), a, jj)
        elif i < P1 + WIPE: stage = "result"; a = wipe(src["result"](src_of(P1 - 1 - P0)), src["result"](0), i - P1)
        else: stage = "result"; a = src["result"](0 if i < RES_START else src_of(i - RES_START))
        canvas[PIC[1]:PIC[1] + PIC[3], PIC[0]:PIC[0] + PIC[2]] = a
    L = LAYERS[stage]; m = L[:, :, 3] > 0; canvas[m] = L[:, :, :3][m]; return canvas

out = C["dir"] / "out"; out.mkdir(exist_ok=True); mp4 = out / f"{C['name']}_Breakdown_1080x1350.mp4"
if only:
    sheet = np.hstack([make(i) for i in only]); cv2.imwrite(str(work(C, "bd_sheet.jpg")), cv2.cvtColor(sheet, cv2.COLOR_RGB2BGR)[::2, ::2], [cv2.IMWRITE_JPEG_QUALITY, 88]); sys.exit()
p = subprocess.Popen(["ffmpeg", "-v", "error", "-y", "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{W}x{H}", "-r", "30", "-i", "-", "-c:v", "libx264", "-crf", "17",
                      "-preset", "slow", "-pix_fmt", "yuv420p", "-color_primaries", "bt709", "-color_trc", "bt709", "-colorspace", "bt709", "-movflags", "+faststart", str(mp4)], stdin=subprocess.PIPE)
for i in range(TOTAL): p.stdin.write(make(i).tobytes())
p.stdin.close(); p.wait(); print("breakdown", mp4.name, TOTAL, "frames", "rc", p.returncode)
