import json, cv2, numpy as np, subprocess
from pathlib import Path
root=Path(__file__).parent; S=json.loads((root/"work/sim.json").read_text()); f=S["f"]
col={"apple":(40,40,220),"lemon":(40,220,240),"melon":(40,140,40)}
def proj(p): return np.array([p[0]/p[2]*f+360, p[1]/p[2]*f+640])
p=subprocess.Popen(["ffmpeg","-v","error","-y","-f","rawvideo","-pix_fmt","bgr24","-s","360x640","-r","25","-i","-","-c:v","libx264","-crf","23","-pix_fmt","yuv420p",str(root/"work/sim_preview.mp4")],stdin=subprocess.PIPE)
for i,fr in enumerate(S["frames"]):
    im=cv2.imread(str(root/f"work/frames/{i:04d}.png"))
    for c in fr["caps"]:
        a=proj(c[:3]); b=proj(c[3:6]); r=int(c[6]/c[2]*f)
        cv2.line(im,tuple(np.int32(a)),tuple(np.int32(b)),(255,0,255),max(1,2*r)) if c[:3]!=c[3:6] else cv2.circle(im,tuple(np.int32(a)),r,(255,0,255),-1)
    im=cv2.addWeighted(im,0.75,cv2.imread(str(root/f"work/frames/{i:04d}.png")),0.25,0)
    for (kind,s),st in zip(S["fruit_kinds"],fr["fruit"][0]):
        uv=proj(st[:3]); cv2.circle(im,tuple(np.int32(uv)),int(s/st[2]*f),col[kind],-1); cv2.putText(im,f"{st[2]:.2f}",tuple(np.int32(uv)),0,0.6,(0,0,0),2)
    cv2.putText(im,str(i),(10,40),0,1.2,(0,0,0),3)
    p.stdin.write(cv2.resize(im,(360,640)).tobytes())
p.stdin.close(); p.wait()
