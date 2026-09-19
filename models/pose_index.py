"""Collect every pose run into one index for training, and build the CricShot10k keeper audit sheets.

Inputs : kaggle/chunks/pose-cv-c*/out   (CricketVision, batter from hand-drawn boxes)
         kaggle/chunks/pose-rest-c*/out (other sources, batter from the batter finder)
Outputs: data/processed/pose_index.parquet  one row per clip: joint file path, window, quality
             flags (usable, finder confidence), joined with manifest labels and splits
         data/processed/pose_report.txt     totals per source and per split
         kaggle/pose-rest-audit/cricshot10k_*.jpg  start/contact/end frames of CricShot10k clips
             for checking how often the wicketkeeper is picked instead of the batter
"""
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
OUT_INDEX = ROOT / "data/processed/pose_index.parquet"
OUT_REPORT = ROOT / "data/processed/pose_report.txt"
AUDIT = ROOT / "kaggle/pose-rest-audit"
FINDER_MIN_CONF = 0.9


def collect() -> pd.DataFrame:
    rows = []
    for job, how in (("pose-cv", "annotated_box"), ("pose-rest", "batter_finder")):
        for chunk in sorted(ROOT.glob(f"kaggle/chunks/{job}-c*/out")):
            pc = chunk / "per_clip.csv"
            if not pc.exists():
                continue
            d = pd.read_csv(pc)
            d["kps_path"] = [str((chunk / "kps" / f"{c}.npz").relative_to(ROOT)) for c in d.clip_id]
            d["batter_chosen_by"] = how
            d["chunk"] = chunk.parent.name
            rows.append(d)
    idx = pd.concat(rows, ignore_index=True)
    idx = idx[[c for c in idx.columns if not c.startswith("_")]]
    m = pd.read_parquet(ROOT / "data/processed/manifest.parquet")
    keep = ["clip_id", "shot", "side", "label_orig", "split", "group", "handedness", "foot",
            "score_head", "score_shoulder", "score_hands", "score_hips", "score_feet", "score_overall"]
    idx = idx.merge(m[[c for c in keep if c in m]], on="clip_id", how="left", suffixes=("", "_manifest"))
    idx["kps_exists"] = [(ROOT / p).exists() for p in idx.kps_path]
    # Train-ready = right person + good joints. Thresholds come from the 400-clip visual audit
    # (kaggle/pose-rest-audit): finder confidence >= 0.9 keeps 98.6% right strikers and removes
    # 26 of 30 wrong picks; CricketVision clips must match the hand-drawn batter box.
    finder_ok = (idx.batter_chosen_by == "batter_finder") & (idx.finder_p >= FINDER_MIN_CONF)
    box_ok = (idx.batter_chosen_by == "annotated_box") & (idx.cv_iou_final > 0.5)
    idx["train_ready"] = idx.usable & idx.kps_exists & (finder_ok | box_ok)
    return idx


def report(idx: pd.DataFrame) -> str:
    lines = [f"clips with pose: {len(idx)} | joint files present: {int(idx.kps_exists.sum())} | "
             f"usable: {idx.usable.mean():.1%} | train-ready (usable + confident batter): {int(idx.train_ready.sum())}", "",
             "per source:",
             idx.groupby("source").agg(clips=("clip_id", "size"), usable=("usable", "mean"), train_ready=("train_ready", "sum"),
                                       finder_low_conf=("finder_p", lambda s: int((s < 0.5).sum()))).round(3).to_string(), "",
             "train-ready clips per split x shot:",
             pd.crosstab(idx[idx.train_ready].shot, idx[idx.train_ready].split, margins=True).to_string()]
    return "\n".join(lines)


def audit_sheets() -> None:
    AUDIT.mkdir(parents=True, exist_ok=True)
    frames = sorted(p for p in ROOT.glob("kaggle/chunks/pose-rest-c*/out/frames/cs10k_*.jpg"))
    for k in range(0, len(frames), 10):
        tiles = [cv2.imread(str(f)) for f in frames[k:k + 10]]
        tiles = [cv2.resize(t, (1200, int(t.shape[0] * 1200 / t.shape[1]))) for t in tiles if t is not None]
        if tiles:
            cv2.imwrite(str(AUDIT / f"cricshot10k_{k // 10:02d}.jpg"), np.vstack(tiles), [cv2.IMWRITE_JPEG_QUALITY, 80])
    print(f"audit sheets: {len(frames)} CricShot10k clips in {AUDIT}")


if __name__ == "__main__":
    idx = collect()
    idx.to_parquet(OUT_INDEX, index=False)
    txt = report(idx)
    OUT_REPORT.write_text(txt + "\n")
    print(txt)
    audit_sheets()
