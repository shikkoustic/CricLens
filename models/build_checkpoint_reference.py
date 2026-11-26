"""Measure the contact checkpoints across the scored clips: validate them, then write the bands
the app compares an uploaded clip against.

Two outputs:
  app/checkpoints_reference.json  -- per stroke, the quartiles among well-rated clips of that
                                     stroke (the "what good looks like" band) plus the whole
                                     population's quantiles, so the app can place a value.
  stdout                          -- the validation table that keeps the feature honest: how
                                     each checkpoint actually correlates with coach scores.

    python models/build_checkpoint_reference.py

The correlations are weak by design of the data, not a bug to tune away: the five coach part
scores are one latent factor (docs/paper/finding_label_collinearity.md), and these checkpoints
are deliberately not fitted to them. What the table is for is checking the *signs* come out
where coaching says they should, and recording the effect sizes so the app never overclaims.
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from app.shots import SCORED  # noqa: E402
from models.contact_checkpoints import CHECKPOINTS, measure  # noqa: E402

LEVELS = list(range(0, 101, 5))
WELL_RATED_PCT = 75     # "what good looks like" = the top quarter of that stroke's coach scores
MIN_BAND_N = 25         # below this a stroke's band is too thin to show; the app falls back to all
# The band spans the 10th-90th percentile of well-rated clips, not the interquartile range. The
# UI calls it "the range well-rated drives sit in", and the IQR is not that range: by definition
# only half of well-rated clips fall inside it, so even the best clips in the test set landed
# outside 3 of the 5 bands and read as faulty. At 10-90, four in five well-rated clips are inside
# and a marker outside the band means something.
BAND_LO, BAND_HI = 10, 90


def main():
    seq = pd.read_parquet(ROOT / "data/processed/sequences.parquet").set_index("clip_id")
    idx = pd.read_parquet(ROOT / "data/processed/pose_index.parquet")
    d = idx[idx.train_ready & idx.score_overall.notna() & idx.clip_id.isin(seq.index)
            & idx.shot.isin(SCORED)]
    print(f"measuring {len(d)} scored, train-ready clips of the five offered strokes\n")

    rows = []
    for r in d.itertuples():
        z = np.load(ROOT / seq.loc[r.clip_id, "seq_path"])
        ci = float(z["contact_idx"]) if z["contact_idx"].shape == () else 17.0
        m = measure(z["seq"], 0, int(round(ci)))
        if m is None:
            continue
        rows.append({**{k: m[k] for k in CHECKPOINTS}, "shot": r.shot,
                     "score": float(r.score_overall), "split": r.split})
    df = pd.DataFrame(rows).replace([np.inf, -np.inf], np.nan)
    print(f"measured {len(df)} of {len(d)} ({len(df) / len(d):.1%}; the rest had no usable pose at contact)\n")

    # ---- validation: does each checkpoint point the way coaching says it should? --------------
    print("Correlation with the coaches' overall score. 'expected' is what coaching orthodoxy")
    print("predicts the sign should be; agreement is the point, the effect sizes are small.\n")
    print(f"{'checkpoint':>18} {'rho':>8} {'p':>10} {'expected':>10} {'agrees':>8}   best stroke")
    for c, meta in CHECKPOINTS.items():
        m = df[c].notna()
        rho, p = spearmanr(df.loc[m, c], df.loc[m, "score"])
        want = {"low": "-", "high": "+", "mid": "?"}[meta["better"]]
        agrees = "--" if want == "?" else ("yes" if (rho < 0) == (want == "-") else "no")
        per = [(s, *spearmanr(g[c], g["score"], nan_policy="omit")) for s, g in df.groupby("shot")
               if g[c].notna().sum() >= 60]
        bs = max(per, key=lambda t: abs(t[1])) if per else None
        tail = f"{bs[0]} {bs[1]:+.3f} (p={bs[2]:.1g})" if bs else "--"
        print(f"{c:>18} {rho:>8.3f} {p:>10.2g} {want:>10} {agrees:>8}   {tail}")

    # ---- the bands the app shows --------------------------------------------------------------
    ref = {"levels": LEVELS, "well_rated_pct": WELL_RATED_PCT, "band_pcts": [BAND_LO, BAND_HI],
           "checkpoints": {}, "band": {}, "n": {}}
    for c in CHECKPOINTS:
        v = df[c].dropna()
        ref["checkpoints"][c] = {**CHECKPOINTS[c], "all": np.percentile(v, LEVELS).round(4).tolist()}
    for shot in list(SCORED) + ["all"]:
        g = df if shot == "all" else df[df.shot == shot]
        cut = g.score.quantile(WELL_RATED_PCT / 100)
        good = g[g.score >= cut]
        ref["n"][shot] = {"clips": int(len(g)), "well_rated": int(len(good)),
                          "score_cut": round(float(cut), 2)}
        ref["band"][shot] = {}
        for c in CHECKPOINTS:
            v = good[c].dropna()
            if len(v) < MIN_BAND_N:
                continue
            ref["band"][shot][c] = {"lo": round(float(v.quantile(BAND_LO / 100)), 3),
                                    "p25": round(float(v.quantile(.25)), 3),
                                    "p50": round(float(v.quantile(.50)), 3),
                                    "p75": round(float(v.quantile(.75)), 3),
                                    "hi": round(float(v.quantile(BAND_HI / 100)), 3),
                                    "n": int(len(v))}
    (ROOT / "app/checkpoints_reference.json").write_text(json.dumps(ref, indent=1) + "\n")

    print(f"\nWhat well-rated players do (median, with the {BAND_LO}-{BAND_HI} band in brackets):")
    print(f"{'stroke':>13} " + " ".join(f"{c[:13]:>15}" for c in CHECKPOINTS))
    for shot in list(SCORED) + ["all"]:
        cells = []
        for c in CHECKPOINTS:
            b = ref["band"][shot].get(c)
            cells.append(f"{b['p50']:>6.2f} [{b['lo']:.2f}-{b['hi']:.2f}]".rjust(15) if b else "--".rjust(15))
        n = ref["n"][shot]
        print(f"{shot:>13} " + " ".join(cells) + f"   (n={n['well_rated']} of {n['clips']})")
    print("\nwrote app/checkpoints_reference.json")


if __name__ == "__main__":
    main()
