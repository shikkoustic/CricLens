"""CricketVision (WACV 2025): annotations -> stroke table, and source videos -> 480p stroke clips.

Annotations are VIA projects, one per source video (P{n}_V{m}.json). Each stroke is
three temporal segments (Buildup, Execution, FollowThrough) plus one keyframe point per
phase holding the batter bbox and 1-10 scores for head/shoulder/hands/hips/feet.
Handedness lives on the Buildup point, foot type + stroke type on Execution, dismissal
on FollowThrough. Option ids are decoded through each file's own `attribute` table.

    python datasets/cricketvision.py parse     # -> data/processed/cricketvision_strokes.parquet
    python datasets/cricketvision.py stream    # stream Videos.zip, cut clips, delete sources
"""
from __future__ import annotations

import glob
import json
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
import video  # noqa: E402
from taxonomy import unify  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data/raw/cricketvision"
CLIPS = ROOT / "data/interim/clips/cricketvision"
TABLE = ROOT / "data/processed/cricketvision_strokes.parquet"
VIDEOS_URL = ("https://www.dropbox.com/scl/fo/qejvkt7obg53xsaep18c2/AOeSPTVXrzFZKN8toQ8X1Iw"
              "?rlkey=6uknqoegtv63xlapz6efhvrtu&dl=1")
PHASES = ("Buildup", "Execution", "FollowThrough")
PARTS = ("Head", "Shoulder", "Hands", "Hips", "Feet")
PAD = 0.3   # seconds of context kept before Buildup and after FollowThrough
TOL = 0.05  # tolerance when matching keyframe points to phase segments
# 183 of 203 files ship a StrokeType option table that drops Sweep and renumbers Block to 5,
# yet their values still use 6 for Block and never use 5. Decoding every file with the full
# 7-option table reproduces the per-stroke totals published in the dataset README.
STROKE_IDS = {"0": "OffDrive", "1": "OnDrive", "2": "Cut", "3": "Glance", "4": "Hook", "5": "Sweep", "6": "Block"}


def _decode(attrs: dict, av: dict) -> dict:
    out = {}
    for aid, val in av.items():
        a = attrs.get(aid)
        if a is None:
            continue
        opts = STROKE_IDS if a["aname"] == "StrokeType" else (a.get("options") or {})
        out[a["aname"]] = opts.get(str(val), val) if opts else val
    return out


def parse_file(path: str) -> list[dict]:
    d = json.load(open(path))
    attrs = d["attribute"]
    fname = next(iter(d["file"].values()))["fname"]
    segs, points = [], []
    for v in d["metadata"].values():
        a = _decode(attrs, v["av"])
        phase = a.get("TEMPORAL-SEGMENTS")
        if phase not in PHASES:
            continue
        if len(v["z"]) == 2:
            segs.append((v["z"][0], v["z"][1], phase))
        elif len(v["z"]) == 1:
            points.append((v["z"][0], phase, v["xy"], a))
    segs.sort()
    strokes, cur = [], None
    for s, e, ph in segs:
        if ph == "Buildup":
            cur = {"Buildup": (s, e)}
            strokes.append(cur)
        elif cur is not None and ph not in cur and abs(s - list(cur.values())[-1][1]) < 0.5:
            cur[ph] = (s, e)
    rows = []
    for k, st in enumerate(strokes):
        row = {"video": Path(fname).stem, "stroke_idx": k, "complete": all(p in st for p in PHASES)}
        for ph in PHASES:
            row[f"{ph.lower()}_start"], row[f"{ph.lower()}_end"] = st.get(ph, (None, None))
        for z, ph, xy, a in points:
            if ph in st and st[ph][0] - TOL <= z <= st[ph][1] + TOL:
                p = ph.lower()
                for part in PARTS:
                    sc = a.get(f"{part}_Score")
                    row[f"{p}_{part.lower()}"] = int(sc) if sc not in (None, "") else None
                if len(xy) == 5:
                    row[f"{p}_bbox"] = [round(x, 1) for x in xy[1:]]  # x, y, w, h in source px
                if ph == "Buildup":
                    row["handedness"] = a.get("Handedness")
                elif ph == "Execution":
                    row["foot"], row["stroke"] = a.get("FootType"), a.get("StrokeType")
                else:
                    row["dismissed"] = a.get("Dismissed")
        rows.append(row)
    return rows


def parse() -> pd.DataFrame:
    rows = [r for f in sorted(glob.glob(str(RAW / "json/*.json"))) for r in parse_file(f)]
    df = pd.DataFrame(rows)
    for part in (p.lower() for p in PARTS):  # per-part score = mean over the three phases
        cols = [f"{ph.lower()}_{part}" for ph in PHASES if f"{ph.lower()}_{part}" in df]
        df[f"score_{part}"] = df[cols].mean(axis=1)
    df["score_overall"] = df[[f"score_{p.lower()}" for p in PARTS]].mean(axis=1)
    df["clip_start"] = (df["buildup_start"] - PAD).clip(lower=0)
    df["clip_end"] = df["followthrough_end"].fillna(df["execution_end"]) + PAD
    df["clip_id"] = "cv_" + df["video"] + "_" + df["stroke_idx"].map("{:04d}".format)
    mapped = df["stroke"].map(lambda s: unify("cricketvision", s) if isinstance(s, str) else ("other", "unknown"))
    df["shot"], df["side"] = zip(*mapped)
    TABLE.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(TABLE, index=False)
    return df


def _cut_video(src: Path, strokes: pd.DataFrame) -> int:
    p = video.probe(src)
    size = video.target_size(p["width"], p["height"])
    jobs = []
    with ThreadPoolExecutor(4) as ex:
        for r in strokes.itertuples():
            dst = CLIPS / f"{r.clip_id}.mp4"
            if not dst.exists() and pd.notna(r.clip_end):
                jobs.append(ex.submit(video.transcode, src, dst, r.clip_start, r.clip_end, size))
        for j in jobs:
            j.result()
    return len(jobs)


def stream() -> None:
    import httpx
    from zipstream import iter_members

    df = pd.read_parquet(TABLE)
    by_video = {v: g for v, g in df.groupby("video")}
    tmp = ROOT / "data/interim/_cv_tmp"
    tmp.mkdir(parents=True, exist_ok=True)
    CLIPS.mkdir(parents=True, exist_ok=True)
    done_marker = tmp / "done.txt"
    done = set(done_marker.read_text().split()) if done_marker.exists() else set()

    opened: set[str] = set()

    def open_out(name: str):
        stem = Path(name).stem
        if not name.endswith(".mp4") or stem in done or stem not in by_video:
            return None
        opened.add(name)
        return open(tmp / Path(name).name, "wb")

    t0, nbytes = time.time(), 0
    with httpx.stream("GET", VIDEOS_URL, follow_redirects=True, timeout=120) as resp:
        resp.raise_for_status()

        def chunks():
            nonlocal nbytes
            for c in resp.iter_bytes(1 << 20):
                nbytes += len(c)
                yield c

        for name, size in iter_members(chunks(), open_out):
            if name not in opened:  # directory entry, unannotated or already-done video
                continue
            stem, src = Path(name).stem, tmp / Path(name).name
            if src.is_file():
                try:
                    n = _cut_video(src, by_video[stem])
                    with open(done_marker, "a") as f:
                        f.write(stem + "\n")
                    done.add(stem)
                    status = f"cut {n}"
                except Exception as e:  # keep going; the video is retried on the next run
                    status = f"ERROR {e!r}"[:200]
                src.unlink()
                el = time.time() - t0
                print(f"[{len(done)}/{len(by_video)}] {stem} {size/1e6:.0f}MB {status} | "
                      f"{nbytes/1e9:.1f}GB @ {nbytes/1e6/el:.1f}MB/s", flush=True)
    print("stream finished", flush=True)


if __name__ == "__main__":
    {"parse": lambda: print(parse().describe(include="all").T.head(0)), "stream": stream}[sys.argv[1]]()
