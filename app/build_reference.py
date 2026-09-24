"""Precompute the small reference file the app needs: scorer target normalisation (must match
models/train_technique_scorer.py exactly) and per-shot quantiles that put a new clip's numbers in
context. Aggregate statistics only -- no clip-level data leaves data/.

    python app/build_reference.py   ->  app/reference.json
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
TARGETS = ["head", "shoulder", "hands", "hips", "feet", "overall"]
QS = np.linspace(0, 100, 21)


def quantiles(s: pd.Series) -> list[float] | None:
    s = s.dropna()
    return [round(float(v), 4) for v in np.percentile(s, QS)] if len(s) >= 30 else None


def main():
    idx = pd.read_parquet(ROOT / "data/processed/pose_index.parquet")
    seq = pd.read_parquet(ROOT / "data/processed/sequences.parquet")[["clip_id"]]
    cols = [f"score_{t}" for t in TARGETS]
    scored = idx[idx.train_ready & idx.score_overall.notna()].merge(seq, on="clip_id")
    ref = {"scorer_mu": scored[cols].mean().round(6).tolist(), "scorer_sd": scored[cols].std().round(6).tolist(),
           "targets": TARGETS, "quantile_levels": QS.tolist(), "score": {}, "stride_cm": {}, "swing_mps": {}}

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
