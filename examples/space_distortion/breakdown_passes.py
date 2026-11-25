"""Pass breakdown in the layout of the reference post: how the space distortion is built, one pass at a time.

    python breakdown_passes.py [piece_dir ...]          default: piece piece_tunnel

Needs render.py to have run on every piece (work/frames, work/pass/{grid,heat,normal,prism}, work/comp).
out (in the FIRST piece's out/):
  <name>_Passes_Breakdown_1080x1350.mp4   pieces stacked (2 max), a vertical wipe walks through the passes:
                                          Original Plate -> Warp Grid -> Distortion Field -> Normal Map -> Prism Split -> 3D Smoke -> VFX
out (in every piece's out/):
  <name>_Passes_Grid_1920x540.mp4         all seven passes side by side for the whole clip"""
import subprocess, sys
from pathlib import Path
import numpy as np, cv2
from PIL import Image, ImageDraw, ImageFont

here = Path(__file__).resolve().parent
sys.path.insert(0, str(here.parent / "shader_live" / "pipeline"))
from slv import load, work   # noqa: E402

STAGES = [("plate", "Original Plate", "원본. 카메라는 COLMAP으로, 팔·관절은 MediaPipe로 장면 3D에 올립니다"),
          ("grid", "Warp Grid", "출력의 각 점이 원본 어디를 읽는지. 격자가 휜 만큼 그림이 끌립니다"),
          ("heat", "Distortion Field", "열화상: 옮겨지는 거리(px). 손 궤적 + 윤곽 파동 + 장면 속 물결 고리"),
          ("normal", "Normal Map", "휘어진 공기를 유리 면으로 본 법선. 반사와 하이라이트가 여기서 나옵니다"),
          ("prism", "Prism Split", "파장 8개를 따로 굴절(코시 식, 아베수 3): 무지개가 갈라지는 곳"),
          ("smoke", "3D Smoke", "장면 속 3D 연기(깊이 색: 주황 = 카메라 가까이, 파랑 = 멀리). 음파가 앞뒤로 불어냅니다"),
          ("comp", "Space Distortion VFX", "최종 합성")]
FOOT = ["COLMAP camera  ·  MediaPipe pose  ·  ModernGL (OpenGL 3.3)  ·  3D Stable Fluids smoke"]
FONTS = {"bold": ["C:/Windows/Fonts/malgunbd.ttf", "/System/Library/Fonts/AppleSDGothicNeo.ttc", "/usr/share/fonts/opentype/noto/NotoSansCJK-Bold.ttc"],
         "reg": ["C:/Windows/Fonts/malgun.ttf", "/System/Library/Fonts/AppleSDGothicNeo.ttc", "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc"]}

def font(kind, size):
    for p in FONTS[kind]:
        if Path(p).exists(): return ImageFont.truetype(p, size)
    return ImageFont.load_default()

def read(c, key, i):
    p = {"plate": work(c, "frames", f"{i:04d}.jpg"), "comp": work(c, "comp", f"{i:04d}.png")}.get(key) or work(c, "pass", key, f"{i:04d}.jpg")
    im = cv2.imread(str(p))
    if im is None: sys.exit(f"missing {p}: run render.py on {c['dir'].name} first")
    return im

def overlay(W, H, items):
    """RGBA text layer: items = (text, xy, size, kind, anchor). Drawn once, blended on every frame."""
    im = Image.new("RGBA", (W, H), (0, 0, 0, 0)); d = ImageDraw.Draw(im)
    for t, xy, size, kind, anchor in items:
        f = font(kind, size)
        d.text((xy[0] + 2, xy[1] + 2), t, font=f, fill=(0, 0, 0, 170), anchor=anchor); d.text(xy, t, font=f, fill=(255, 255, 255, 255), anchor=anchor)
    a = np.asarray(im).astype(np.float32) / 255.0; return a[..., 2::-1].copy(), a[..., 3:4].copy()

def blend(img, ov):
    rgb, a = ov; return (img.astype(np.float32) * (1 - a) + rgb * 255.0 * a).astype(np.uint8)

def encoder(path, W, H, rate):
    return subprocess.Popen(["ffmpeg", "-v", "error", "-y", "-f", "rawvideo", "-pix_fmt", "bgr24", "-s", f"{W}x{H}", "-r", rate, "-i", "-",
                             "-c:v", "libx264", "-crf", "16", "-preset", "slow", "-pix_fmt", "yuv420p", "-color_primaries", "bt709",
                             "-color_trc", "bt709", "-colorspace", "bt709", "-movflags", "+faststart", str(path)], stdin=subprocess.PIPE)

def stacked(cs, rate=25, per=60, wipe=16, hold=80):
    """Reference layout: 1080x1350, each piece 1080x608, a vertical line wipes from one pass to the next."""
    W, H, pw, ph = 1080, 1350, 1080, 608; n = len(cs); top = (H - n * ph) // 2
    total = per * (len(STAGES) - 1) + hold
    ovs = []
    for k, (_, en, ko) in enumerate(STAGES):
        items = [(f"{k + 1:02d} / {len(STAGES):02d}   {ko}", (W // 2, top // 2), 25, "reg", "mm")]
        items += [(en, (pw - 26, top + (j + 1) * ph - 22), 42, "bold", "rs") for j in range(n)]
        items += [(FOOT[0], (W // 2, H - top // 2), 20, "reg", "mm")]
        ovs.append(overlay(W, H, items))
    out = cs[0]["dir"] / "out"; out.mkdir(exist_ok=True); path = out / f"{cs[0]['name']}_Passes_Breakdown_1080x1350.mp4"
    enc = encoder(path, W, H, str(rate))
    for t in range(total):
        k = min(t // per, len(STAGES) - 1); u = (t - k * per) / wipe
        canvas = np.zeros((H, W, 3), np.uint8)
        for j, c in enumerate(cs):
            i = int(t / rate * c["fps"]) % c["N"]
            new = cv2.resize(read(c, STAGES[k][0], i), (pw, ph), interpolation=cv2.INTER_AREA)
            if k > 0 and u < 1.0:                                                 # wipe: new pass left of the line, old pass right of it
                old = cv2.resize(read(c, STAGES[k - 1][0], i), (pw, ph), interpolation=cv2.INTER_AREA)
                x = int((0.5 - 0.5 * np.cos(np.pi * u)) * pw); new[:, x:] = old[:, x:]; cv2.line(new, (x, 0), (x, ph), (255, 255, 255), 2, cv2.LINE_AA)
            canvas[top + j * ph: top + (j + 1) * ph] = new
        lab = k if (k == 0 or u >= 0.5) else k - 1
        enc.stdin.write(blend(canvas, ovs[lab]).tobytes())
    enc.stdin.close(); enc.wait(); print("->", path, total, "frames")

def grid(c):
    """Every pass of one piece at once, 4x2 tiles of 480x270, for the whole clip."""
    tw, th = 480, 270; W, H = 4 * tw, 2 * th
    items = []
    for k, (_, en, ko) in enumerate(STAGES):
        x, y = (k % 4) * tw, (k // 4) * th
        items += [(f"{k + 1:02d} {en}", (x + 12, y + th - 34), 20, "bold", "ls"), (ko if len(ko) < 34 else ko[:32] + "…", (x + 12, y + th - 12), 13, "reg", "ls")]
    ov = overlay(W, H, items)
    path = c["dir"] / "out" / f"{c['name']}_Passes_Grid_1920x540.mp4"; (c["dir"] / "out").mkdir(exist_ok=True)
    enc = encoder(path, W, H, f"{c['fps_num']}/{c['fps_den']}")
    for i in range(c["N"]):
        canvas = np.zeros((H, W, 3), np.uint8)
        for k, (key, _, _) in enumerate(STAGES):
            x, y = (k % 4) * tw, (k // 4) * th; canvas[y:y + th, x:x + tw] = cv2.resize(read(c, key, i), (tw, th), interpolation=cv2.INTER_AREA)
        enc.stdin.write(blend(canvas, ov).tobytes())
    enc.stdin.close(); enc.wait(); print("->", path, c["N"], "frames")

if __name__ == "__main__":
    dirs = sys.argv[1:] or [here / "piece", here / "piece_tunnel"]
    cs = [load(d) for d in dirs]
    if not any(c.get("smoke", {}).get("enable") for c in cs):          # smoke off: no smoke stage, no smoke credit
        STAGES[:] = [st for st in STAGES if st[0] != "smoke"]; FOOT[0] = FOOT[0].replace("  ·  3D Stable Fluids smoke", "")
    stacked(cs[:2])
    for c in cs: grid(c)
