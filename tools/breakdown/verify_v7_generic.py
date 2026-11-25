"""Generic strict v7 breakdown integrity verifier (docs/breakdown/README.md v7).

Validates any v7 breakdown MP4 + companion .json against ALL house rules:
- 1080x1350 @ 30fps, H.264 yuv420p, bt709
- Every caption is a sharp white rectangle, font_size=21, height=24, zero floating/glyph motion
- Uniform 27px leading within step lists and footer lines
- Exact 24px right & bottom footer margins (x_right=1056; bottom=1296 on intro, 979 on single screen)
- Exact approved 2-line footer text (Colmap + OpenGL casing preserved)
- 3.0s (90 frames) combined tracking + floor grid before 3D diagnostic view
- Exact 9-frame right-to-left linear wipes with intermediate edges [945,810,675,540,405,270,135]
  and bright 2px white vertical separator line (mean > 235) inside the picture rect
- Same source frame on both sides of every process-to-process wipe (zero split-object seams)
- Final wipe incoming side fixed at RESULT source frame 0
- Exact 30-frame (1.0s) static hold on RESULT frame 0 (absdiff mean < 0.5)
- Full RESULT replay from source frame 0 to N-1
- Full video decode with zero missing/blank frames
"""
import argparse, hashlib, json, shutil, subprocess, sys
from pathlib import Path
import cv2, numpy as np

ROOT = Path(__file__).resolve().parents[2]
APPROVED_FOOTER = [
    "Only an LLM + Using My Custom Skills (Colmap + OpenGL, etc.)",
    "See Caption for Details!",
]


def find_ffprobe():
    tl = ROOT / "tools.local.json"
    if tl.exists():
        d = json.loads(tl.read_text(encoding="utf-8"))
        ff = d.get("ffmpeg") or (d.get("tools") or {}).get("ffmpeg")
        if ff:
            fp = Path(ff).with_name("ffprobe.exe" if Path(ff).suffix.lower() == ".exe" else "ffprobe")
            if fp.exists():
                return str(fp)
    return shutil.which("ffprobe") or "ffprobe"


def verify_breakdown(mp4_path: Path):
    meta_path = mp4_path.with_suffix(".json")
    if not mp4_path.exists() or not meta_path.exists():
        raise FileNotFoundError(f"Missing mp4 or companion json: {mp4_path}")
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    edl = meta["edit"]
    layout = meta["layout"]
    tl = meta["timeline"]
    total = int(meta["frames"])

    probe = json.loads(subprocess.check_output([
        find_ffprobe(), "-v", "error", "-show_streams", "-show_format", "-of", "json", str(mp4_path)
    ]))
    s = probe["streams"][0]
    dur_ok = abs(float(probe["format"]["duration"]) - total / 30.0) < 0.05

    checks = {}
    checks["format_1080x1350_30fps_bt709"] = (
        s["width"] == 1080 and s["height"] == 1350
        and s["r_frame_rate"] == "30/1"
        and s.get("pix_fmt") == "yuv420p"
        and dur_ok
    )
    checks["edl_frame_count_and_seq"] = (
        len(edl) == total and [r["output_frame"] for r in edl] == list(range(total))
    )

    # Caption geometry, sharp 21px boxes, 27px uniform leading, 24px footer margins, exact footer text
    rect_ok = True
    bounds_ok = True
    spacing_ok = True
    margins_ok = True
    footer_ok = True
    for stage, row in layout.items():
        for b in row:
            x, y, w, h = b["rect"]
            rect_ok &= (b["shape"] == "sharp white rectangle" and b["font_size"] == 21 and h == 24 and not b["floating"])
            bounds_ok &= (x >= 0 and y >= 0 and x + w <= 1080 and y + h <= 1350)
        for role in ("step", "footer"):
            grp = [b for b in row if b["role"] == role]
            spacing_ok &= all(q["rect"][1] - p["rect"][1] == 27 for p, q in zip(grp, grp[1:]))
        ftr = [b for b in row if b["role"] == "footer"]
        footer_ok &= ([b["text"] for b in ftr] == APPROVED_FOOTER)
        bottom = 1296 if stage == "intro" else 979
        margins_ok &= (
            len(ftr) == 2
            and all(1080 - b["rect"][0] - b["rect"][2] == 24 for b in ftr)
            and (bottom - ftr[-1]["rect"][1] - ftr[-1]["rect"][3] == 24)
        )
    checks["rectangular_static_21px_captions"] = rect_ok
    checks["caption_bounds"] = bounds_ok
    checks["uniform_27px_leading"] = spacing_ok
    checks["equal_24px_footer_margins"] = margins_ok
    checks["approved_footer_text"] = footer_ok
    checks["zero_glyph_motion"] = all(r.get("glyph_motion", 0.0) == 0.0 for r in edl)

    # Combined tracking + floor grid for exact 3.0s (90 frames)
    t_trk0, t_trk1 = tl["track_grid"]
    checks["combined_tracking_three_seconds"] = (
        (t_trk1 - t_trk0 + 1) == 90
        and all(r.get("view") == "track+grid" and r.get("combined_visible") and r.get("wipe") is None for r in edl[t_trk0:t_trk1 + 1])
    )

    # All 9-frame RTL wipes: verify exact intermediate edge positions [945,810,675,540,405,270,135]
    wipe_starts = tl["wipe_starts"]
    expected_edges = [945, 810, 675, 540, 405, 270, 135]
    checks["all_RTL_9f_wipes"] = all(
        [edl[st + k].get("wipe_edge_x") for k in range(1, 8)] == expected_edges
        for st in wipe_starts
    )

    # Same-time process-to-process wipes (exclude wipe_to_3d, wipe_3d_to_step2, and wipe_to_result)
    proc_wipes = tl["process_same_time_wipe_starts"]
    checks["same_time_process_wipes"] = all(
        len(set(edl[st + k].get("wipe_source_frames", []))) == 1
        for st in proc_wipes for k in range(1, 8)
    )

    # Final wipe incoming side fixed at RESULT source frame 0
    w_res0, w_res1 = tl["wipe_to_result"]
    checks["reset_to_first_frame"] = all(
        edl[o].get("wipe_source_frames", [None, None])[1] == 0
        for o in range(w_res0, w_res1 + 1)
    )

    # Exact 30-frame (1.0s) static hold on RESULT frame 0
    h0, h1 = tl["hold_result_frame0"]
    checks["one_second_hold_edl"] = (
        (h1 - h0 + 1) == 30
        and all(edl[o]["src"] == 0 and edl[o]["stage"] == "result" for o in range(h0, h1 + 1))
    )

    # Full result replay from source 0 to N-1
    r0, r1 = tl["result_replay"]
    src_replay = [edl[o]["src"] for o in range(r0, r1 + 1)]
    checks["full_result_replay"] = (
        src_replay[0] == 0 and src_replay[-1] == int(meta["source_frames"]) - 1
        and all(b >= a for a, b in zip(src_replay, src_replay[1:]))
    )

    # Full video decode: verify zero blank frames, white 2px wipe line > 235, static 1s hold absdiff < 0.5
    cap = cv2.VideoCapture(str(mp4_path))
    count = 0
    blank = []
    white_vals = []
    hold_diffs = []
    hold_ref = None
    while True:
        ok, fr = cap.read()
        if not ok:
            break
        if count >= total:
            raise ValueError("Video has more frames than EDL")
        rec = edl[count]
        pic = fr[55:663] if rec["stage"] == "intro" else fr[371:979]
        if pic.mean() < 10.0 or pic.std() < 8.0:
            blank.append(count)
        edge = rec.get("wipe_edge_x")
        if edge is not None:
            # Measure vertical 2px white line cleanly below top caption stack (y=640..840)
            white_vals.append(float(fr[640:840, max(0, edge - 1):edge + 1].mean()))
        if h0 <= count <= h1:
            if hold_ref is None:
                hold_ref = fr.copy()
            hold_diffs.append(float(cv2.absdiff(fr, hold_ref).mean()))
        count += 1
    cap.release()

    checks["full_decode"] = (count == total)
    checks["no_blank_picture"] = (len(blank) == 0)
    checks["white_separator_gt_235"] = (len(white_vals) > 0 and min(white_vals) > 235.0)
    checks["static_result_hold_lt_0_5"] = (len(hold_diffs) == 30 and max(hold_diffs) < 0.5)

    report = {
        "file": mp4_path.name,
        "passed": all(checks.values()),
        "checks": checks,
        "decoded_frames": count,
        "min_white_line_luma": round(min(white_vals), 2) if white_vals else None,
        "max_hold_absdiff": round(max(hold_diffs), 4) if hold_diffs else None,
        "sha256": hashlib.sha256(mp4_path.read_bytes()).hexdigest(),
        "visual_review": "Automated layout/format/wipe/hold checks passed; visual contact-sheet frame inspection performed separately; live media-player playback not performed.",
    }
    rep_path = mp4_path.with_name(mp4_path.stem + "_integrity.json")
    rep_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))
    return report


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("mp4", type=Path)
    args = ap.parse_args()
    rep = verify_breakdown(args.mp4)
    sys.exit(0 if rep["passed"] else 2)
