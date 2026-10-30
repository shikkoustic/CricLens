"""Precompute the small reference file the app needs: scorer target normalisation (must match
models/train_technique_scorer.py exactly), the contrastive scorer's diff normalisation (must match
models/train_technique_scorer_contrastive.py's load_data()/run() exactly), and per-shot quantiles
that put a new clip's numbers in context. Aggregate statistics only -- no clip-level data leaves
data/ (the per-exemplar scores in kaggle/exemplars.csv are already committed labels, not raw clips,
and app/pipeline.py reads those directly rather than duplicating them here).

    python app/build_reference.py   ->  app/reference.json
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from models.train_technique_scorer_contrastive import EXEMPLARS, to_residual  # noqa: E402

TARGETS = ["head", "shoulder", "hands", "hips", "feet", "overall"]
QS = np.linspace(0, 100, 21)


def contrastive_diff_stats() -> dict:
    """diff_mu/diff_sd for the contrastive scorer's residual-target run (models/
    train_technique_scorer_contrastive.py --residual-targets): the mean/std of (query - exemplar)
    score diffs over the exact train-split pairs that run trained on. Needed to denormalise the
    model's output at inference; score-only, so no sequence data is touched here."""
    idx = pd.read_parquet(ROOT / "data/processed/pose_index.parquet")
    score_cols = [f"score_{t}" for t in TARGETS]
    d = idx[idx.train_ready & idx.score_overall.notna() & (idx.shot != "other")][["clip_id", "split", "shot"] + score_cols]
    exemplars = pd.read_csv(EXEMPLARS)
    d = d[~d.clip_id.isin(exemplars.clip_id)]
    tr = d[d.split == "train"]
    ex_by_shot = {s: g for s, g in exemplars.groupby("shot")}

    diffs = []
    for r in tr.itertuples():
        qy = to_residual(np.array([getattr(r, c) for c in score_cols], np.float32))
        for e in ex_by_shot[r.shot].itertuples():
            ey = to_residual(np.array([getattr(e, c) for c in score_cols], np.float32))
            diffs.append(qy - ey)
    diffs = np.stack(diffs)
    mu, sd = diffs.mean(0).astype(np.float32), diffs.std(0).astype(np.float32)
    sd[sd < 1e-3] = 1.0
    return {"contrastive_diff_mu": mu.round(6).tolist(), "contrastive_diff_sd": sd.round(6).tolist()}


def quantiles(s: pd.Series) -> list[float] | None:
    s = s.dropna()
    return [round(float(v), 4) for v in np.percentile(s, QS)] if len(s) >= 30 else None


def main():
    idx = pd.read_parquet(ROOT / "data/processed/pose_index.parquet")
    seq = pd.read_parquet(ROOT / "data/processed/sequences.parquet")[["clip_id"]]
    cols = [f"score_{t}" for t in TARGETS]
    scored = idx[idx.train_ready & idx.score_overall.notna()].merge(seq, on="clip_id")
    ref = {"scorer_mu": scored[cols].mean().round(6).tolist(), "scorer_sd": scored[cols].std().round(6).tolist(),
           "targets": TARGETS, "quantile_levels": QS.tolist(), "score": {}, "stride_cm": {}, "swing_mps": {},
           **contrastive_diff_stats()}

    for shot, g in [("all", scored)] + list(scored.groupby("shot")):
        q = {t: quantiles(g[f"score_{t}"]) for t in TARGETS}
        if q["overall"]:
            ref["score"][shot] = q

    pc = pd.read_parquet(ROOT / "data/processed/iva/pitch_calibration.parquet")
    pc = pc.drop(columns=["source"]).merge(idx[["clip_id", "shot"]], on="clip_id")
    for shot, g in [("all", pc)] + list(pc.groupby("shot")):
        for key, col in (("stride_cm", "stride_cm"), ("swing_mps", "swing_speed_mps")):
            q = quantiles(g[col])
            if q:
                ref[key][shot] = q

    out = ROOT / "app/reference.json"
    json.dump(ref, open(out, "w"), indent=1)
    print(f"wrote {out}: score shots={list(ref['score'])}, stride shots={list(ref['stride_cm'])}")


if __name__ == "__main__":
    main()
