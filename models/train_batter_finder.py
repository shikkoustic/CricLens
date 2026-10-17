"""Train the batter finder: given every tracked person in a clip, pick the striker.

Training data:
  - kaggle/chunks/pose-cv-c*/out/track_features.csv: CricketVision, one row per tracked person,
    label = matches the hand-annotated batter box.
  - kaggle/chunks/pose-rest-c*/out/track_features.csv: the other 5 sources (amittalmale, cricshot10,
    cricshot10k, ipl2023, kucricshot), which have no annotated box. Labels come from weak supervision
    instead: kaggle/pose-rest-audit/verdicts.csv (400 clips, hand pass/fail) plus a larger visual
    review (2,559 clips, see models/finder_review.py and PROGRESS.md) that confirmed or corrected the
    "chosen" track the current model already picks (stored per clip in the pose kps.npz). A clip
    contributes a positive label (chosen track = batter) only when a human confirmed it; a clip
    flagged wrong contributes only negatives (every tracked person, including the wrongly-chosen one,
    labelled not-batter) since the actual batter track wasn't re-identified. Unreviewed clips are
    dropped entirely rather than trusting the model's own unconfirmed pick.

Stumps features are merged in when kaggle/stumps/out/stump_features.csv exists (CricketVision only;
missing for the other sources, which HistGradientBoostingClassifier handles natively as NaN).
Evaluation is per clip: the top-scored person must be the batter. Splits are grouped by source video
(regex per source; some sources have no recoverable video id and fall back to one clip per group).

    python models/train_batter_finder.py
"""
import re
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.model_selection import GroupKFold

ROOT = Path(__file__).resolve().parents[1]
CV_TRACKS = sorted(ROOT.glob("kaggle/chunks/pose-cv-c*/out/track_features.csv"))
REST_TRACKS = sorted(ROOT.glob("kaggle/chunks/pose-rest-c*/out/track_features.csv"))
REST_KPS_DIRS = sorted(ROOT.glob("kaggle/chunks/pose-rest-c*/out/kps"))
VERDICTS = ROOT / "kaggle/pose-rest-audit/verdicts.csv"
REVIEW_SAMPLE = ROOT / "kaggle/finder-review-sample.csv"  # the 2,559-clip sample list, see PROGRESS.md
STUMPS = ROOT / "kaggle/stumps/out/stump_features.csv"
OUT = ROOT / "models/batter_finder.joblib"
BASE = ["n_tracks", "cover", "first", "cx", "cy", "w", "h", "aspect", "growth", "move", "striker", "bat",
        "size_rank", "keeper_behind"]
STUMP = ["st_found", "st_dx", "st_dy", "st_far", "st_in_front", "st_behind", "st_beside"]

VIDEO_PATTERNS = {
    "cv": re.compile(r"^cv_(P\d+_V\d+)_"),
    "amt": re.compile(r"^(amt_[a-z-]+_.+)-\d+-\d+$"),  # strip the trailing frame-range suffix
    "cs10k": re.compile(r"^cs10k_[a-z-]+_[a-z-]+-(vid\d+)-\d+$"),
    "ipl": re.compile(r"^ipl_[a-z-]+_[a-z-]+-(\d+-[a-z]+-[a-z]+)-"),
}


def video_group(clip_id: str) -> str:
    for prefix, pat in VIDEO_PATTERNS.items():
        if clip_id.startswith(prefix + "_") or prefix == "cv":
            m = pat.match(clip_id)
            if m:
                return f"{prefix}_{m.group(1)}"
    return clip_id  # no recoverable video id (cricshot10, kucricshot): one clip per group


def chosen_tids(clip_ids: set[str]) -> dict[str, int]:
    """Read chosen_tid out of each clip's kps.npz (the current model's own pick, used as the
    unlabelled starting point that verdicts/review either confirm or reject)."""
    out: dict[str, int] = {}
    remaining = set(clip_ids)
    for d in REST_KPS_DIRS:
        if not remaining:
            break
        for clip_id in list(remaining):
            f = d / f"{clip_id}.npz"
            if f.exists():
                z = np.load(f)
                out[clip_id] = int(z["chosen_tid"])
                remaining.discard(clip_id)
    return out


def rest_labels() -> dict[str, int]:
    """clip_id -> correct tid, for every clip with a human-confirmed answer. Verdicts.csv (pass/fail,
    400 clips) takes precedence over the larger review sample where both cover the same clip."""
    confirmed: dict[str, bool] = {}
    if VERDICTS.exists():
        v = pd.read_csv(VERDICTS)
        for row in v.itertuples():
            confirmed[row.clip_id] = pd.isna(row.error)  # True = chosen track confirmed correct
    if REVIEW_SAMPLE.exists():
        for clip_id in pd.read_csv(REVIEW_SAMPLE).clip_id:
            confirmed.setdefault(clip_id, True)  # full visual review found no errors (PROGRESS.md)
    tids = chosen_tids(set(confirmed))
    return {cid: tids[cid] for cid, ok in confirmed.items() if ok and cid in tids}, \
           {cid for cid, ok in confirmed.items() if not ok and cid in tids}


def load() -> tuple[pd.DataFrame, list[str]]:
    if not CV_TRACKS:
        raise SystemExit("no finished pose-cv chunks yet (kaggle/chunks/pose-cv-c*/out/track_features.csv)")
    cv = pd.concat([pd.read_csv(f) for f in CV_TRACKS], ignore_index=True)
    cv = cv[cv.groupby("clip_id")["label"].transform("max") == 1]  # clips where the annotated batter was tracked
    cv["video"] = cv["clip_id"].str.extract(r"^cv_(P\d+_V\d+)_")[0]

    frames = [cv]
    if REST_TRACKS:
        rest = pd.concat([pd.read_csv(f) for f in REST_TRACKS], ignore_index=True)
        correct_tid, wrong_only = rest_labels()
        keep = set(correct_tid) | wrong_only
        rest = rest[rest.clip_id.isin(keep)].copy()
        rest["label"] = [
            1 if cid in correct_tid and tid == correct_tid[cid] else 0
            for cid, tid in zip(rest.clip_id, rest.tid)
        ]
        rest["video"] = rest["clip_id"].map(video_group)
        print(f"pose-rest: {len(correct_tid)} confirmed-correct clips, {len(wrong_only)} confirmed-wrong "
              f"clips (negatives only), {rest.clip_id.nunique()} clips / {len(rest)} tracked people kept")
        frames.append(rest)
    d = pd.concat(frames, ignore_index=True)

    feats = BASE.copy()
    if STUMPS.exists():
        d = d.merge(pd.read_csv(STUMPS), on=["clip_id", "tid"], how="left")
        feats += [c for c in STUMP if c in d]
    # per-clip relative features: where each person sits compared with the others in the same clip
    for c in ("cy", "h", "striker", "bat"):
        d[f"{c}_rel"] = d[c] - d.groupby("clip_id")[c].transform("mean")
        feats.append(f"{c}_rel")
    return d, feats


def top1(d: pd.DataFrame, score: np.ndarray) -> float:
    """Per-clip accuracy: the top-scored person must be the batter. Clips with no positive label
    (confirmed-wrong, negatives-only) can't ever score here and are excluded rather than counted
    as automatic misses."""
    d = d.assign(score=score)
    d = d[d.groupby("clip_id")["label"].transform("max") == 1]
    pick = d.loc[d.groupby("clip_id")["score"].idxmax()]
    return float(pick["label"].mean())


def main() -> None:
    d, feats = load()
    X, y, g = d[feats].to_numpy(float), d["label"].to_numpy(int), d["video"].to_numpy()
    print(f"{d.clip_id.nunique()} clips, {len(d)} tracked people, {len(feats)} features")
    rule = d["striker"] * 2 + d["bat"] * 1.5 - (d["cx"] - 0.5).abs() * 0.6 - (d["cy"] - 0.45).abs() * 0.4
    print(f"hand-written rule (v3) top-1: {top1(d, rule.to_numpy()):.3f}")
    oof = np.zeros(len(d))
    for tr, te in GroupKFold(n_splits=5).split(X, y, g):
        m = HistGradientBoostingClassifier(max_iter=400, learning_rate=0.05, max_leaf_nodes=31,
                                           class_weight="balanced", random_state=13).fit(X[tr], y[tr])
        oof[te] = m.predict_proba(X[te])[:, 1]
    print(f"batter finder (5-fold, grouped by video) top-1: {top1(d, oof):.3f}")
    final = HistGradientBoostingClassifier(max_iter=400, learning_rate=0.05, max_leaf_nodes=31,
                                           class_weight="balanced", random_state=13).fit(X, y)
    joblib.dump({"model": final, "features": feats}, OUT)
    print("saved", OUT)


if __name__ == "__main__":
    main()
