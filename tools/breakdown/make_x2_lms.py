"""Build final Instagram-ready Breakdown x2 + hyun2LMS Logo Motion outro (1080x1350 @ 30fps).

Matches the exact house recipe from examples/space_distortion/reel/Breakdown_x2_LMS/:
- Input breakdown played 2x back-to-back (x2)
- 1.0s smooth fade dissolve at offset = 2*D - 1.0 into #fcfefd canvas + hyun2LMS ProRes 4444 logo motion
- 1.0s start-frame hold during dissolve so none of the 7.67s motion is hidden
- Full 7.67s motion ("learn it make it share it" -> "hyun2LMS" -> "lms.hyun2.cloud")
- 2.5s end-frame static hold on "lms.hyun2.cloud"
- H.264 High@4.1, CRF 12, maxrate 9M, bufsize 18M, GOP 60, yuv420p, bt709, +faststart, 48kHz stereo AAC silent audio

Usage:
    python make_x2_lms.py <input_breakdown_mp4> <output_x2_lms_mp4>
"""
import argparse, hashlib, json, shutil, subprocess, sys
from pathlib import Path
import cv2, numpy as np

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_LOGO = Path("C:/Users/Q/Documents/카카오톡 받은 파일/hyun2LMS_logo_motion_1920x1080_30fps_ProRes4444.mov")


def find_bins():
    tl = ROOT / "tools.local.json"
    ff = None
    if tl.exists():
        d = json.loads(tl.read_text(encoding="utf-8"))
        p = d.get("ffmpeg") or (d.get("tools") or {}).get("ffmpeg")
        if p and Path(p).exists():
            ff = Path(p)
    if ff is None:
        w = shutil.which("ffmpeg")
        ff = Path(w) if w else Path("ffmpeg")
    fp = ff.with_name("ffprobe.exe" if ff.suffix.lower() == ".exe" else "ffprobe")
    return str(ff), str(fp if fp.exists() else (shutil.which("ffprobe") or "ffprobe"))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("src", type=Path)
    ap.add_argument("dst", type=Path)
    ap.add_argument("--logo", type=Path, default=DEFAULT_LOGO)
    ap.add_argument("--qa-sheet", type=Path, default=None)
    args = ap.parse_args()

    if not args.src.exists():
        sys.exit(f"Missing input breakdown: {args.src}")
    if not args.logo.exists():
        sys.exit(f"Missing LMS logo motion: {args.logo}")
    if args.dst.exists():
        sys.exit(f"Refusing to overwrite existing output: {args.dst}")

    ff, fp = find_bins()
    probe = json.loads(subprocess.check_output([
        fp, "-v", "error", "-show_streams", "-show_format", "-of", "json", str(args.src)
    ]))
    dur = float(probe["format"]["duration"])
    vs0 = [s for s in probe["streams"] if s["codec_type"] == "video"][0]
    n_src = int(vs0.get("nb_frames") or round(dur * 30))
    off = 2.0 * dur - 1.0

    args.dst.parent.mkdir(parents=True, exist_ok=True)
    fc = (
        "[0:v]fps=30,settb=1/30,setsar=1,format=yuv420p[a];"
        "[1:v]fps=30,settb=1/30,setsar=1,format=yuv420p[b];"
        "[a][b]concat=n=2:v=1:a=0,settb=1/30[bd];"
        "[2:v]fps=30,scale=1080:-2:flags=lanczos,format=yuva444p,"
        "tpad=start_mode=clone:start_duration=1:stop_mode=clone:stop_duration=2.5[lg];"
        "[3:v][lg]overlay=(W-w)/2:(H-h)/2:format=auto,fps=30,settb=1/30,setsar=1,format=yuv420p[end];"
        f"[bd][end]xfade=transition=fade:duration=1:offset={off:.4f},format=yuv420p[v]"
    )
    cmd = [
        ff, "-v", "error", "-y",
        "-i", str(args.src),
        "-i", str(args.src),
        "-i", str(args.logo),
        "-f", "lavfi", "-i", "color=c=0xfcfefd:s=1080x1350:r=30:d=11.1667",
        "-f", "lavfi", "-i", "anullsrc=channel_layout=stereo:sample_rate=48000",
        "-filter_complex", fc,
        "-map", "[v]", "-map", "4:a",
        "-c:v", "libx264", "-preset", "slow", "-profile:v", "high", "-level", "4.1",
        "-crf", "12", "-maxrate", "9M", "-bufsize", "18M", "-g", "60",
        "-pix_fmt", "yuv420p", "-color_primaries", "bt709", "-color_trc", "bt709", "-colorspace", "bt709",
        "-c:a", "aac", "-b:a", "128k", "-ar", "48000",
        "-shortest", "-movflags", "+faststart",
        str(args.dst),
    ]
    subprocess.run(cmd, check=True)

    # Probe output + verify zero black frames
    pr_out = json.loads(subprocess.check_output([
        fp, "-v", "error", "-count_frames", "-show_streams", "-show_format", "-of", "json", str(args.dst)
    ]))
    vs = [s for s in pr_out["streams"] if s["codec_type"] == "video"][0]
    au = [s for s in pr_out["streams"] if s["codec_type"] == "audio"][0]
    n_out = int(vs.get("nb_read_frames") or vs.get("nb_frames"))
    dur_out = float(pr_out["format"]["duration"])

    blk = subprocess.run([
        ff, "-v", "info", "-i", str(args.dst),
        "-vf", "blackdetect=d=0.03:pix_th=0.05", "-an", "-f", "null", "-"
    ], capture_output=True, text=True)
    black_hits = [ln.strip() for ln in blk.stderr.splitlines() if "black_start" in ln]

    # Visual QA sheet (8 key frames: pass1 start, pass1 end, pass2 start, dissolve mid, LMS cursor, learn it, hyun2LMS, final lms.hyun2.cloud hold)
    if args.qa_sheet:
        cap = cv2.VideoCapture(str(args.dst))
        f_diss = int(round(off * 30))
        idxs = [
            12,
            n_src - 2,
            n_src,
            f_diss + 15,
            f_diss + 32,
            f_diss + 95,
            f_diss + 195,
            n_out - 2,
        ]
        tiles = []
        for fi in idxs:
            fi = max(0, min(fi, n_out - 1))
            cap.set(cv2.CAP_PROP_POS_FRAMES, fi)
            ok, im = cap.read()
            im = cv2.resize(im, (270, 338), interpolation=cv2.INTER_AREA)
            cv2.putText(im, f"f={fi} ({fi/30.0:.1f}s)", (8, 26), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 0, 0), 3, cv2.LINE_AA)
            cv2.putText(im, f"f={fi} ({fi/30.0:.1f}s)", (8, 26), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 220, 40), 2, cv2.LINE_AA)
            tiles.append(im)
        cap.release()
        grid = np.vstack([np.hstack(tiles[:4]), np.hstack(tiles[4:])])
        args.qa_sheet.parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(args.qa_sheet), grid, [cv2.IMWRITE_JPEG_QUALITY, 92])

    info = {
        "file": args.dst.name,
        "size_mb": round(args.dst.stat().st_size / 1e6, 2),
        "video": f"{vs['width']}x{vs['height']} @ {vs['r_frame_rate']} {vs['pix_fmt']} ({vs.get('color_space')})",
        "audio": f"{au['codec_name']} {au['sample_rate']}Hz {au['channels']}ch",
        "src_frames": n_src,
        "out_frames": n_out,
        "duration_s": round(dur_out, 3),
        "dissolve_offset_s": round(off, 4),
        "black_frame_hits": len(black_hits),
        "sha256": hashlib.sha256(args.dst.read_bytes()).hexdigest(),
    }
    print(json.dumps(info, indent=2))
    if black_hits:
        sys.exit(f"Black frames detected: {black_hits}")


if __name__ == "__main__":
    main()
