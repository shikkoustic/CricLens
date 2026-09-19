"""Merge per-source clip tables into one manifest: dedupe across sources, then split by match.

Steps
  1. load data/processed/*_clips.parquet and cricketvision_strokes.parquet, drop failed clips
  2. perceptual-hash frames sampled at 5 fps from every 480p clip
  3. candidate duplicates come from a cheap 64-bit pHash LSH search; a pair is accepted only
     if its 256-bit pHash frame sequences, aligned by the best time offset, differ by
     <= VERIFY_BITS bits per frame. Broadcast deliveries from one match share camera angle and
     pitch, so the 64-bit test alone merges different balls (and chains them into huge
     clusters); the aligned 256-bit test separates them (same stroke ~0 bits, different
     balls ~58, unrelated ~84). One clip per verified cluster is kept as canonical.
     CricketVision ships some source videos twice under different annotator folders; their
     annotations are near-identical copies (not independent ratings), so we keep one clip
     and report how closely the two copies match
  4. splits are assigned per group (source match, else duplicate cluster, else clip) with
     StratifiedGroupKFold, so one match / one delivery never sits in two splits
Output: data/processed/manifest.parquet (+ .csv) and data/processed/manifest_report.txt
"""
from __future__ import annotations

import sys
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import cv2
import imagehash
import numpy as np
import pandas as pd
from PIL import Image
from sklearn.model_selection import StratifiedGroupKFold

ROOT = Path(__file__).resolve().parents[1]
PROCESSED = ROOT / "data/processed"
HASHES = PROCESSED / "frame_hashes.parquet"
HASHES256 = PROCESSED / "frame_hashes256.parquet"
VERIFY_BITS = 20     # max mean differing bits (of 256) per aligned frame for a verified duplicate
MIN_OVERLAP = 0.6    # aligned overlap must cover this share of the shorter clip
PARTS = ["head", "shoulder", "hands", "hips", "feet"]
SAMPLE_FPS = 5
FRAME_DIST = 6       # max Hamming distance (of 64 bits) for two frames to count as the same
DUP_FRACTION = 0.6   # share of the shorter clip's frames that must match to call it a duplicate
SEED = 13

LICENSES = {
    "cricketvision": "No license stated (WACV 2025); academic research use",
    "cricshot10k": "Research / non-commercial only (IEEE Access 2026)",
    "cricshot10": "CC0 per repo; data by request; HF mirror rokmr/cricket-shot (Apache-2.0 per uploader)",
    "ipl2023": "Apache-2.0 per Kaggle uploader",
    "kucricshot": "Not stated (ICCIT 2023); academic research use",
}


def _load() -> pd.DataFrame:
    frames = []
    for p in sorted(PROCESSED.glob("*_clips.parquet")):
        frames.append(pd.read_parquet(p))
    cv_table = PROCESSED / "cricketvision_strokes.parquet"
    if cv_table.exists():
        cv = pd.read_parquet(cv_table)
        cv["source"], cv["label_orig"], cv["group"] = "cricketvision", cv["stroke"], "cricketvision:" + cv["video"]
        cv["path"] = "data/interim/clips/cricketvision/" + cv["clip_id"] + ".mp4"
        cv["ok"] = cv["path"].map(lambda p: (ROOT / p).exists())
        frames.append(cv)
    df = pd.concat(frames, ignore_index=True)
    # match ids are always re-derived from source paths with the current rule, so tables
    # written by an older ingest run pick up fixes without re-transcoding
    sys.path.insert(0, str(Path(__file__).parent))
    from ingest_clips import group_for
    has_rel = df.get("src_relpath").notna() if "src_relpath" in df else pd.Series(False, index=df.index)
    df.loc[has_rel, "group"] = [group_for(s, Path(r)) for s, r in zip(df.loc[has_rel, "source"], df.loc[has_rel, "src_relpath"])]
    df = df[df["ok"].fillna(False)].copy()
    df["license"] = df["source"].map(LICENSES)
    return df.reset_index(drop=True)


def _hash_clip(path: str) -> list[int]:
    cap = cv2.VideoCapture(str(ROOT / path))
    fps = cap.get(cv2.CAP_PROP_FPS) or 25
    step = max(1, round(fps / SAMPLE_FPS))
    out, i = [], 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        if i % step == 0:
            img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY))
            out.append(int(str(imagehash.phash(img)), 16))
        i += 1
    cap.release()
    return out


def frame_hashes(df: pd.DataFrame, workers: int = 6) -> dict[str, list[int]]:
    cached = pd.read_parquet(HASHES) if HASHES.exists() else pd.DataFrame(columns=["clip_id", "hashes"])
    have = {c: [_unsigned(int(x)) for x in hs] for c, hs in zip(cached["clip_id"], cached["hashes"])}
    todo = df[~df["clip_id"].isin(have)]
    if len(todo):
        with ProcessPoolExecutor(workers) as ex:
            for cid, hs in zip(todo["clip_id"], ex.map(_hash_clip, todo["path"], chunksize=16)):
                have[cid] = hs
        pd.DataFrame({"clip_id": list(have), "hashes": [[_signed(x) for x in v] for v in have.values()]}).to_parquet(HASHES)
    return have


def _signed(h: int) -> int:  # 64-bit pHash -> int64 for parquet
    return h - (1 << 64) if h >= (1 << 63) else h


def _unsigned(h: int) -> int:
    return h + (1 << 64) if h < 0 else h


def candidate_edges(ids: list[str], hashes: dict[str, list[int]]) -> list[tuple[int, int]]:
    """Cheap recall stage: 64-bit pHash LSH + 'most frames within FRAME_DIST' test."""
    band_index: dict[tuple[int, int], set[int]] = defaultdict(set)  # 4 bands x 16 bits (LSH)
    for ci, cid in enumerate(ids):
        for h in hashes.get(cid, []):
            for b in range(4):
                band_index[(b, (h >> (16 * b)) & 0xFFFF)].add(ci)
    arrs = [np.array(hashes.get(c, []), dtype=np.uint64) for c in ids]
    edges = []
    for ci, cid in enumerate(ids):
        cands: dict[int, int] = defaultdict(int)
        for h in hashes.get(cid, []):
            for b in range(4):
                for cj in band_index[(b, (h >> (16 * b)) & 0xFFFF)]:
                    if cj > ci:
                        cands[cj] += 1
        a = arrs[ci]
        for cj, hits in cands.items():
            b = arrs[cj]
            if not len(a) or not len(b):
                continue
            short, long_ = (a, b) if len(a) <= len(b) else (b, a)
            if hits < max(2, int(0.3 * len(short))):
                continue
            x = short[:, None] ^ long_[None, :]
            dist = np.unpackbits(x.view(np.uint8).reshape(*x.shape, 8), axis=-1).sum(-1)
            if (dist.min(axis=1) <= FRAME_DIST).mean() >= DUP_FRACTION:
                edges.append((ci, cj))
    return edges


def _hash_clip256(path: str) -> list[str]:
    cap = cv2.VideoCapture(str(ROOT / path))
    fps = cap.get(cv2.CAP_PROP_FPS) or 25
    step = max(1, round(fps / SAMPLE_FPS))
    out, i = [], 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        if i % step == 0:
            out.append(str(imagehash.phash(Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)), hash_size=16)))
        i += 1
    cap.release()
    return out


def hashes256(paths: dict[str, str], workers: int = 6) -> dict[str, np.ndarray]:
    cached = pd.read_parquet(HASHES256) if HASHES256.exists() else pd.DataFrame(columns=["clip_id", "hashes"])
    have = {c: list(h) for c, h in zip(cached["clip_id"], cached["hashes"])}
    todo = [c for c in paths if c not in have]
    if todo:
        with ProcessPoolExecutor(workers) as ex:
            for cid, hs in zip(todo, ex.map(_hash_clip256, [paths[c] for c in todo], chunksize=16)):
                have[cid] = hs
        pd.DataFrame({"clip_id": list(have), "hashes": list(have.values())}).to_parquet(HASHES256)
    return {c: np.array([np.unpackbits(np.frombuffer(bytes.fromhex(h), dtype=np.uint8)) for h in hs], dtype=bool)
            for c, hs in have.items() if c in paths and hs}


def aligned_bits(a: np.ndarray, b: np.ndarray) -> float:
    """Mean differing bits per frame at the best time offset (overlap >= MIN_OVERLAP of shorter)."""
    if len(a) > len(b):
        a, b = b, a
    need = max(2, int(np.ceil(MIN_OVERLAP * len(a)))) if len(a) >= 2 else 1
    best = float("inf")
    for off in range(-(len(a) - need), len(b) - need + 1):
        lo, hi = max(0, -off), min(len(a), len(b) - off)
        if hi - lo >= need:
            best = min(best, float((a[lo:hi] != b[lo + off:hi + off]).sum(1).mean()))
    return best


def duplicate_clusters(df: pd.DataFrame, hashes: dict[str, list[int]]) -> dict[str, str]:
    ids = df["clip_id"].tolist()
    edges = candidate_edges(ids, hashes)
    involved = {ids[i] for e in edges for i in e}
    h256 = hashes256({c: p for c, p in zip(df["clip_id"], df["path"]) if c in involved})
    parent = list(range(len(ids)))

    def find(x: int) -> int:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    verified = 0
    for ci, cj in edges:
        a, b = h256.get(ids[ci]), h256.get(ids[cj])
        if a is not None and b is not None and aligned_bits(a, b) <= VERIFY_BITS:
            parent[find(ci)] = find(cj)
            verified += 1
    print(f"dedupe: {len(edges)} candidate pairs from 64-bit search, {verified} verified at 256 bits", flush=True)
    return {cid: ids[find(i)] for i, cid in enumerate(ids)}


def annotator_agreement(df: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    """CricketVision strokes that appear under two annotator folders.

    The two annotation sets agree almost perfectly (Spearman ~0.98, ~0.08 mean abs diff), so they
    are copies of one annotation rather than independent ratings and must NOT be read as
    inter-rater agreement. Scores are averaged (a no-op in practice) and the match is reported."""
    cv = df[df["source"] == "cricketvision"]
    multi = cv.groupby("dup_cluster").filter(lambda g: len(g) > 1 and g["video"].nunique() > 1)
    cols = [f"score_{p}" for p in PARTS] + ["score_overall"]
    if len(multi):
        means = multi.groupby("dup_cluster")[cols].mean()
        n = multi.groupby("dup_cluster").size()
        for c in cols:
            df.loc[df["source"].eq("cricketvision"), f"{c}_avg"] = df["dup_cluster"].map(means[c]).fillna(df[c])
        df["n_raters"] = df["dup_cluster"].map(n).where(df["source"].eq("cricketvision")).fillna(
            df["source"].eq("cricketvision").astype(float))
    lines = [f"CricketVision strokes present twice (same source video under two annotator folders): {multi['dup_cluster'].nunique()}",
             "  their two annotation copies match as follows (near-identical => copies, NOT inter-rater agreement):"]
    pairs = [g.head(2) for _, g in multi.groupby("dup_cluster")]
    if pairs:
        a = pd.DataFrame([p.iloc[0][cols + ["stroke", "foot", "handedness"]] for p in pairs]).reset_index(drop=True)
        b = pd.DataFrame([p.iloc[1][cols + ["stroke", "foot", "handedness"]] for p in pairs]).reset_index(drop=True)
        for c in cols:
            ok = a[c].notna() & b[c].notna()
            r = a.loc[ok, c].astype(float).corr(b.loc[ok, c].astype(float), method="spearman")
            mae = (a.loc[ok, c].astype(float) - b.loc[ok, c].astype(float)).abs().mean()
            lines.append(f"  copy vs copy {c:15} Spearman {r:.2f}  mean abs diff {mae:.2f}  (n={int(ok.sum())})")
        for c in ("stroke", "foot", "handedness"):
            ok = a[c].notna() & b[c].notna()
            lines.append(f"  copy vs copy {c:10} {(a.loc[ok, c] == b.loc[ok, c]).mean():.1%} (n={int(ok.sum())})")
    return df, lines


def assign_splits(df: pd.DataFrame) -> pd.Series:
    groups = df["group"].fillna(df["dup_cluster"])
    y = df["shot"].where(df["shot"] != "other", "zz_other")
    sgkf = StratifiedGroupKFold(n_splits=7, shuffle=True, random_state=SEED)
    split = pd.Series("train", index=df.index)
    folds = list(sgkf.split(df, y, groups))
    split.iloc[folds[0][1]] = "test"
    split.iloc[folds[1][1]] = "val"
    return split


def build() -> pd.DataFrame:
    df = _load()
    hashes = frame_hashes(df)
    df["dup_cluster"] = df["clip_id"].map(duplicate_clusters(df, hashes))
    df, agreement = annotator_agreement(df)
    # canonical copy per cluster: prefer CricketVision (richest labels), then longest clip
    rank = {"cricketvision": 0, "cricshot10k": 1, "cricshot10": 2, "ipl2023": 3, "kucricshot": 4}
    df["_r"] = df["source"].map(rank)
    order = df.sort_values(["_r", "duration"], ascending=[True, False])
    df["is_canonical"] = ~order.duplicated("dup_cluster").reindex(df.index)
    # a duplicate cluster spanning several matches must stay in one split: merge its groups
    g = df["group"].fillna(df["dup_cluster"])
    df["group"] = g.groupby(df["dup_cluster"]).transform("first")
    df["split"] = assign_splits(df)
    df = df.drop(columns=["_r"])
    df.to_parquet(PROCESSED / "manifest.parquet", index=False)
    df.drop(columns=[c for c in df.columns if df[c].map(type).eq(list).any() or c.endswith("_bbox")], errors="ignore") \
      .to_csv(PROCESSED / "manifest.csv", index=False)
    report(df, agreement)
    return df


def report(df: pd.DataFrame, extra: list[str] | None = None) -> None:
    c = df[df["is_canonical"]]
    lines = [
        f"clips (decoded ok): {len(df)} | canonical after dedupe: {len(c)} | duplicates removed: {len(df) - len(c)}",
        "", "canonical clips per source x split:", pd.crosstab(c["source"], c["split"], margins=True).to_string(),
        "", "canonical clips per unified shot x split:", pd.crosstab(c["shot"], c["split"], margins=True).to_string(),
        "", "cross-source duplicate clusters:",
        df.groupby("dup_cluster")["source"].agg(lambda s: "+".join(sorted(set(s)))).loc[lambda s: s.str.contains(r"\+")].value_counts().to_string(),
    ]
    sizes = df.groupby("dup_cluster").size()
    lines += ["", "removed as duplicates per source:",
              df.assign(removed=~df["is_canonical"]).groupby("source")["removed"].agg(["sum", "size"]).to_string(),
              f"largest duplicate cluster: {int(sizes.max())} clips"]
    lines += [""] + (extra or [])
    leak = df.groupby("group")["split"].nunique().gt(1).sum()
    lines += ["", f"groups present in more than one split (must be 0): {leak}"]
    (PROCESSED / "manifest_report.txt").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    build()
