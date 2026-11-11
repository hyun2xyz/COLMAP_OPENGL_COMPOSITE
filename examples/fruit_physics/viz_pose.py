import json, cv2, numpy as np
from pathlib import Path
root=Path(__file__).parent; P=json.loads((root/"work/pose.json").read_text())
E=[(11,12),(11,13),(13,15),(12,14),(14,16),(11,23),(12,24),(23,24),(23,25),(24,26),(25,27),(26,28),(15,19),(16,20),(0,11),(0,12)]
tiles=[]
for i in [0,60,120,180,240,300,360,390]:
    im=cv2.imread(str(root/f"work/frames/{i:04d}.png")); m=cv2.imread(str(root/f"work/mask/{i:04d}.png"),0)
    H,W=im.shape[:2]; tint=im.copy(); tint[m>127]=(0.5*tint[m>127]+[0,90,0]).astype(np.uint8)
    pts=[(int(p[0]*W),int(p[1]*H)) for p in P[i]["img"]]
    for a,b in E: cv2.line(tint,pts[a],pts[b],(255,0,255),3)
    for k,p in enumerate(pts): cv2.circle(tint,p,5,(0,255,255) if P[i]["img"][k][3]>.5 else (0,0,255),-1)
    tiles.append(cv2.resize(tint,(240,427)))
cv2.imwrite(str(root/"work/pose_check.jpg"),np.hstack(tiles))
