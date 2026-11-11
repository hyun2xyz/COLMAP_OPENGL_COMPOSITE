import json, cv2, numpy as np
from pathlib import Path
root=Path(__file__).parent; B=json.loads((root/"work/body_cam.json").read_text()); f=B["f"]
J=np.array(B["joints"]); E=[(11,12),(11,13),(13,15),(12,14),(14,16),(11,23),(12,24),(23,24),(0,11),(0,12)]
tiles=[]
for i in [0,60,120,180,240,300,360,390]:
    im=cv2.imread(str(root/f"work/frames/{i:04d}.png"))
    uv=J[i][:,:2]/J[i][:,2:]*f+[360,640]
    for a,b in E: cv2.line(im,tuple(np.int32(uv[a])),tuple(np.int32(uv[b])),(0,0,255),3)
    cv2.putText(im,f"hip z {J[i][[23,24],2].mean():.2f}m",(20,60),0,1.4,(0,0,255),3)
    tiles.append(cv2.resize(im,(240,427)))
cv2.imwrite(str(root/"work/body_check.jpg"),np.hstack(tiles))
