"""Train the batter finder: given every tracked person in a clip, pick the striker.

Training data: kaggle/chunks/pose-cv-c*/out/track_features.csv (one row per tracked person in each
CricketVision clip, label = matches the annotated batter box). Stumps features are merged in
when kaggle/stumps/out/stump_features.csv exists (feet in front of / behind / beside the
stumps). Evaluation is per clip: the top-scored person must be the annotated batter.
Splits are grouped by source video, so no match is in both train and test.

    python models/train_batter_finder.py
"""
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.model_selection import GroupKFold

ROOT = Path(__file__).resolve().parents[1]
TRACKS = sorted(ROOT.glob("kaggle/chunks/pose-cv-c*/out/track_features.csv"))  # one file per finished chunk
STUMPS = ROOT / "kaggle/stumps/out/stump_features.csv"
OUT = ROOT / "models/batter_finder.joblib"
BASE = ["n_tracks", "cover", "first", "cx", "cy", "w", "h", "aspect", "growth", "move", "striker", "bat",
        "size_rank", "keeper_behind"]
STUMP = ["st_found", "st_dx", "st_dy", "st_far", "st_in_front", "st_behind", "st_beside"]


def load() -> tuple[pd.DataFrame, list[str]]:
    if not TRACKS:
        raise SystemExit("no finished pose-cv chunks yet (kaggle/chunks/pose-cv-c*/out/track_features.csv)")
    d = pd.concat([pd.read_csv(f) for f in TRACKS], ignore_index=True)
    d = d[d.groupby("clip_id")["label"].transform("max") == 1]  # clips where the annotated batter was tracked
    feats = BASE.copy()
    if STUMPS.exists():
        d = d.merge(pd.read_csv(STUMPS), on=["clip_id", "tid"], how="left")
        feats += [c for c in STUMP if c in d]
    # per-clip relative features: where each person sits compared with the others in the same clip
    for c in ("cy", "h", "striker", "bat"):
        d[f"{c}_rel"] = d[c] - d.groupby("clip_id")[c].transform("mean")
        feats.append(f"{c}_rel")
    d["video"] = d["clip_id"].str.extract(r"^cv_(P\d+_V\d+)_")[0]
    return d, feats


def top1(d: pd.DataFrame, score: np.ndarray) -> float:
    d = d.assign(score=score)
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
