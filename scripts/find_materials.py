"""Find licensed footage and HDRI for a new shot, look at it, and scaffold shots/<name>.json.

Run from the repository root with the project .venv Python. Rules: docs/materials.md.

  footage "drone abandoned hall"    search Pexels / Pixabay (API key in env), else print search pages
  hdri "forest overcast"            search Poly Haven HDRIs (CC0) through its public API
  sheet <file-or-url> --start 2     12-frame contact sheet of the part you would use
  new-shot my_shot --like hall_water --page <url> --download <url> --creator <name> --license pexels
                                    download, hash, probe, contact sheet, write shots/my_shot.json

Search results are candidates, not approvals. Open the distributor page, confirm the
licence there, and look at the contact sheet before a long solve. Footage stays in
work/ (ignored by git). API keys are read from the environment and never printed.
"""
from __future__ import annotations

import argparse
from datetime import date
import json
import os
from pathlib import Path
import subprocess
import sys
import urllib.parse
import urllib.request

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from pipeline.common import SHOTS, StepError, load_json, probe, save_json, sha256, tool, work_root  # noqa: E402
from pipeline.fetch import download  # noqa: E402

AGENT = "Mozilla/5.0 (compatible; COLMAP_OPENGL_COMPOSITE classroom material search)"
POLYHAVEN = "https://api.polyhaven.com"

LICENSES = {
    "pexels": dict(
        license="Pexels License", license_url="https://www.pexels.com/license/",
        permissions="Free download, use and modification; attribution optional. Not CC0. No resale of unaltered "
                    "footage, no stock-platform redistribution, no implied endorsement."),
    "pixabay": dict(
        license="Pixabay Content License", license_url="https://pixabay.com/service/license-summary/",
        permissions="Free use and modification; attribution optional. Not CC0. No sale or redistribution of "
                    "unaltered content (including on other stock platforms), no implied endorsement, no misleading "
                    "or offensive use of people, brands or places shown."),
    "cc0": dict(
        license="CC0 1.0", license_url="https://creativecommons.org/publicdomain/zero/1.0/",
        permissions="Public domain dedication. Still confirm on the page that the uploader owns the footage."),
}


# ---------------------------------------------------------------- network

def get_json(url, headers=None):
    request = urllib.request.Request(url, headers={"User-Agent": AGENT, **(headers or {})})
    with urllib.request.urlopen(request, timeout=30) as response:
        return json.loads(response.read().decode("utf-8"))


# ---------------------------------------------------------------- footage

def pick_file(files):
    """Smallest landscape MP4 that is at least 1920 wide, else the widest one."""
    files = [f for f in files if f.get("link") or f.get("url")]
    files = [f for f in files if (f.get("width") or 0) >= (f.get("height") or 0)]
    if not files:
        return None
    wide = sorted((f for f in files if (f.get("width") or 0) >= 1920), key=lambda f: f["width"])
    return wide[0] if wide else max(files, key=lambda f: f.get("width") or 0)


def pexels_candidates(data, min_seconds):
    out = []
    for video in data.get("videos", []):
        if video.get("width", 0) < video.get("height", 0) or video.get("duration", 0) < min_seconds:
            continue
        chosen = pick_file([f for f in video.get("video_files", []) if f.get("file_type") == "video/mp4"])
        if not chosen:
            continue
        out.append(dict(
            id=f"pexels:{video['id']}", page=video["url"], creator=(video.get("user") or {}).get("name", ""),
            seconds=video["duration"], size=[chosen["width"], chosen["height"]], fps=chosen.get("fps"),
            download=chosen["link"], preview=video.get("image"), license="pexels"))
    return out


def pixabay_candidates(data, min_seconds):
    out = []
    for hit in data.get("hits", []):
        if hit.get("duration", 0) < min_seconds:
            continue
        chosen = pick_file(list((hit.get("videos") or {}).values()))
        if not chosen:
            continue
        out.append(dict(
            id=f"pixabay:{hit['id']}", page=hit["pageURL"], creator=hit.get("user", ""),
            seconds=hit["duration"], size=[chosen["width"], chosen["height"]], fps=None,
            download=chosen["url"], preview=chosen.get("thumbnail"), license="pixabay"))
    return out


def footage(args):
    query = " ".join(args.query)
    found, used = [], []
    if os.environ.get("PEXELS_API_KEY"):
        used.append("Pexels")
        url = "https://api.pexels.com/videos/search?" + urllib.parse.urlencode(
            dict(query=query, orientation="landscape", per_page=min(80, args.count*3)))
        found += pexels_candidates(get_json(url, {"Authorization": os.environ["PEXELS_API_KEY"]}), args.min_seconds)
    if os.environ.get("PIXABAY_API_KEY"):
        used.append("Pixabay")
        url = "https://pixabay.com/api/videos/?" + urllib.parse.urlencode(
            dict(key=os.environ["PIXABAY_API_KEY"], q=query, video_type="film", per_page=min(200, args.count*3)))
        found += pixabay_candidates(get_json(url), args.min_seconds)
    found = found[:args.count]
    if args.json:
        print(json.dumps(dict(query=query, sources=used, candidates=found), ensure_ascii=False, indent=2))
        return
    if not used:
        q = urllib.parse.quote(query.replace(" ", "-"))
        print("No PEXELS_API_KEY or PIXABAY_API_KEY in the environment. Open these search pages instead:")
        print(f"  https://www.pexels.com/search/videos/{q}/?orientation=landscape")
        print(f"  https://pixabay.com/videos/search/{urllib.parse.quote(query)}/")
        print("Free API keys: https://www.pexels.com/api/ and https://pixabay.com/api/docs/ (keep them in .env, never in git).")
    for c in found:
        print(f"{c['id']:16} {c['seconds']:>3}s {c['size'][0]}x{c['size'][1]} {c['creator']}")
        print(f"  page     {c['page']}")
        print(f"  download {c['download']}")
    if used:
        print(f"\n{len(found)} candidates from {' + '.join(used)} (videos provided by {' and '.join(used)}).")
    print("Next: open each page, confirm the licence there, then run `sheet <download>` to look at the frames.")


# ---------------------------------------------------------------- HDRI

def hdri_matches(assets, words, category=None):
    rows = []
    for asset_id, info in assets.items():
        hay = " ".join([asset_id, info.get("name", ""), *info.get("tags", []), *info.get("categories", [])]).lower()
        if category and category not in [c.lower() for c in info.get("categories", [])]:
            continue
        score = sum(word in hay for word in words)
        if words and score == 0:
            continue
        rows.append((score, info.get("date_published", 0), asset_id, info))
    rows.sort(key=lambda r: (-r[0], -r[1]))
    return rows


def hdri_file(asset_id, resolution="1k"):
    files = get_json(f"{POLYHAVEN}/files/{urllib.parse.quote(asset_id)}")
    return files["hdri"][resolution]["hdr"]["url"]


def hdri(args):
    words = [w.lower() for w in args.query]
    assets = get_json(f"{POLYHAVEN}/assets?t=hdris")
    rows = hdri_matches(assets, words, args.category.lower() if args.category else None)[:args.count]
    if args.json:
        print(json.dumps([dict(id=a, name=i.get("name"), categories=i.get("categories"), authors=list(i.get("authors", {})),
                               page=f"https://polyhaven.com/a/{a}") for _, _, a, i in rows], ensure_ascii=False, indent=2))
        return
    for _, _, asset_id, info in rows:
        print(f"{asset_id:34} {', '.join(info.get('categories', []))[:60]}")
        print(f"  {info.get('name')} — {', '.join(info.get('authors', {}))} — https://polyhaven.com/a/{asset_id}")
    print(f"\n{len(rows)} HDRIs (CC0, Powered by Poly Haven). Pick one whose light matches the plate: "
          "indoor/outdoor, sun or overcast, time of day. Use it with `new-shot --hdri <id>`.")


# ---------------------------------------------------------------- look before choosing

def candidates_dir(work):
    d = Path(work)/"_candidates"
    d.mkdir(parents=True, exist_ok=True)
    return d


def local_copy(source, work):
    if Path(source).is_file():
        return Path(source)
    name = Path(urllib.parse.urlparse(source).path).name or "candidate.mp4"
    return download(source, candidates_dir(work)/name)


def contact_sheet(clip, out, start, seconds, frames=12):
    out = Path(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run([tool("ffmpeg"), "-v", "error", "-y", "-ss", str(start), "-t", str(seconds), "-i", str(clip),
                    "-vf", f"fps={frames}/{seconds},scale=480:-2,tile=4x{(frames+3)//4}", "-frames:v", "1", str(out)],
                   check=True)
    return out


def describe(clip):
    info = probe(clip)
    video = next(s for s in info["streams"] if s.get("codec_type") == "video")
    num, den = (int(v) for v in video.get("avg_frame_rate", "0/1").split("/"))
    return dict(seconds=round(float(info["format"]["duration"]), 3), size=[video["width"], video["height"]],
                fps=round(num/den, 3) if den else None)


def sheet(args):
    work = work_root(args.work)
    clip = local_copy(args.source, work)
    meta = describe(clip)
    seconds = args.duration or min(5, meta["seconds"]-args.start)
    out = Path(args.out) if args.out else candidates_dir(work)/f"{clip.stem}_sheet_{args.start:g}s.jpg"
    contact_sheet(clip, out, args.start, seconds)
    print(json.dumps(dict(clip=str(clip), **meta, sheet=str(out), checked=f"{args.start:g}-{args.start+seconds:g}s"),
                     ensure_ascii=False, indent=2))
    print("Look for: camera moving sideways/forward (not only turning), no cut, fixed zoom, sharp static texture, "
          "a flat place for the object.")


# ---------------------------------------------------------------- scaffold

def new_shot(args):
    target = SHOTS/f"{args.name}.json"
    if target.exists():
        raise StepError(f"{target} already exists. Choose another name; existing shots are never overwritten.")
    like = load_json(SHOTS/f"{args.like}.json")
    work = work_root(args.work)
    clip = download(args.download, Path(work)/args.name/"source/original.mp4")
    meta = describe(clip)
    if meta["size"][0] < meta["size"][1]:
        raise StepError("The clip is portrait. The pipeline expects a landscape plate.")
    if args.start+args.duration > meta["seconds"]+1e-3:
        raise StepError(f"The clip is {meta['seconds']}s; --start + --duration must fit inside it.")
    cfg = dict(like)
    cfg["title"] = args.title or f"{args.name} (from {args.like})"
    if args.kind:
        cfg["kind"] = args.kind
    cfg["source"] = dict(title=args.source_title or args.name, creator=args.creator, page=args.page,
                         **LICENSES[args.license], download=args.download, sha256=sha256(clip),
                         checked=date.today().isoformat())
    if args.hdri:
        assets = get_json(f"{POLYHAVEN}/assets?t=hdris")
        if args.hdri not in assets:
            raise StepError(f"Poly Haven has no HDRI called {args.hdri}. Search with: find_materials.py hdri <words>")
        info = assets[args.hdri]
        url = hdri_file(args.hdri)
        file = Path(urllib.parse.urlparse(url).path).name
        hdr = download(url, Path(work)/"lighting"/file)
        cfg["hdri"] = dict(file=file, title=info.get("name", args.hdri), author=", ".join(info.get("authors", {})),
                           page=f"https://polyhaven.com/a/{args.hdri}", license="CC0", credit="Powered by Poly Haven",
                           download=url, sha256=sha256(hdr))
    cfg["plate"] = {**like["plate"], "start": args.start, "duration": args.duration}
    cfg["camera"] = {**like["camera"], "note": f"Starting value copied from {args.like}. Assumed, not measured; "
                     "if qa-camera fails compare 750 / 850 / 1000 in separate --work folders."}
    cfg["ground"] = {**like["ground"], "occlusion": "TODO: check qa/ground-fit.jpg and describe what passes in front.",
                     "shadow": like["ground"].get("shadow", "")}
    cfg["notes"] = f"Scaffolded from {args.like} on {date.today().isoformat()}. Ground, focal and render settings are " \
                   "the example's values; confirm them with qa-camera and qa/ground-fit.jpg."
    save_json(target, cfg)
    out = contact_sheet(clip, Path(work)/args.name/"source/plate_sheet.jpg", args.start, args.duration)
    print(json.dumps(dict(shot=str(target), clip=str(clip), **meta, sheet=str(out)), ensure_ascii=False, indent=2))
    if cfg["ground"].get("topdown"):
        print(f"Note: {args.like} looks straight down (ground.topdown). If this camera looks forward, delete "
              "ground.topdown from the new shot file.")
    print(f"Next: open the sheet, then `python -m pipeline composite {args.name}` "
          "(run solve through scripts/jobs.py on slow machines).")


# ---------------------------------------------------------------- CLI

def parser():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="command", required=True)
    f = sub.add_parser("footage", help="search licensed footage")
    f.add_argument("query", nargs="+")
    f.add_argument("--count", type=int, default=10)
    f.add_argument("--min-seconds", type=float, default=6)
    f.add_argument("--json", action="store_true")
    h = sub.add_parser("hdri", help="search Poly Haven HDRIs")
    h.add_argument("query", nargs="*")
    h.add_argument("--category", help="e.g. indoor, outdoor, urban, nature, skies, sunrise-sunset")
    h.add_argument("--count", type=int, default=10)
    h.add_argument("--json", action="store_true")
    s = sub.add_parser("sheet", help="contact sheet of a clip section")
    s.add_argument("source", help="local file or download URL")
    s.add_argument("--start", type=float, default=0)
    s.add_argument("--duration", type=float)
    s.add_argument("--out")
    s.add_argument("--work")
    n = sub.add_parser("new-shot", help="download, hash and write shots/<name>.json")
    n.add_argument("name")
    n.add_argument("--like", required=True, help="closest example shot: hall_water, courtyard_spheres, quarry_cloth")
    n.add_argument("--page", required=True)
    n.add_argument("--download", required=True)
    n.add_argument("--creator", required=True)
    n.add_argument("--license", required=True, choices=sorted(LICENSES))
    n.add_argument("--source-title")
    n.add_argument("--title")
    n.add_argument("--kind", choices=["water", "rigid_spheres", "cloth"])
    n.add_argument("--start", type=float, default=0)
    n.add_argument("--duration", type=float, default=4)
    n.add_argument("--hdri", help="Poly Haven asset id; default keeps the example's HDRI")
    n.add_argument("--work")
    return p


def main(argv=None):
    args = parser().parse_args(argv)
    try:
        {"footage": footage, "hdri": hdri, "sheet": sheet, "new-shot": new_shot}[args.command](args)
    except StepError as error:
        print(f"STOP: {error}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
