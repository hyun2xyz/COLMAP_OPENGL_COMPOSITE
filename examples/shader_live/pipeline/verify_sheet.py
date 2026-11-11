"""python verify_sheet.py <out.jpg> piece [piece ...]: 4 frames (5%, 35%, 65%, 95%) read back from each encoded Composite.mp4"""
import sys, json, cv2, numpy as np
from pathlib import Path
ROOT = Path(__file__).resolve().parent.parent; rows = []
for n in sys.argv[2:]:
    d = ROOT / "pieces" / n; nm = json.loads((d / "piece.json").read_text(encoding="utf-8"))["name"]
    cap = cv2.VideoCapture(str(d / "out" / f"{nm}_Composite.mp4")); N = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)); fr = []
    for u in (0.05, 0.35, 0.65, 0.95):
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(u * (N - 1))); ok, im = cap.read(); fr.append(cv2.resize(im, (480, 270)) if ok else np.zeros((270, 480, 3), np.uint8))
    cv2.putText(fr[0], f"{n} ({N} f)", (6, 18), 0, 0.5, (0, 255, 255), 1, cv2.LINE_AA); rows.append(np.hstack(fr))
Path(sys.argv[1]).resolve().parent.mkdir(parents=True, exist_ok=True); cv2.imwrite(sys.argv[1], np.vstack(rows), [cv2.IMWRITE_JPEG_QUALITY, 86]); print("sheet", sys.argv[1], len(rows))
