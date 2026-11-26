"""Pick the demo clips app/server.py offers first, and write app/demo_clips.json.

Why this exists: the technique scores are regressions onto CricketVision coaches' own 0-10
marks, whose median is 6.27, so a clip scoring 6.4 is an average shot -- correct, but a poor
thing to open a demo on. This picks held-out test clips that the deployed scorer itself rates
well for their stroke, so there is something worth looking at in the first rows of the picker.

The selection is made with the *model's* prediction, not the human label: a clip a coach marked
9/10 that the scorer rates 5 would show the app in a worse light than a random one. Candidates
are drawn from clips whose label is also high (>= MIN_LABEL), so a high prediction isn't just
the model being wrong in a flattering direction.

    python scripts/pick_demo_clips.py            # rewrites app/demo_clips.json

Needs data/processed/{pose_index,sequences}.parquet and the trained scorer (scripts/fetch_data.sh).
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from app.shots import SCORED, SCORED_IDX, SHOTS  # noqa: E402
from models.train_shot_classifier import make_model  # noqa: E402
from models.train_technique_scorer_contrastive import (ContrastiveScorer, from_residual,  # noqa: E402
                                                        normalise, to_residual)

MIN_LABEL = 7.5   # only consider clips coaches themselves rated well
MIN_PCT = 55      # and that the scorer places above typical for *that stroke's* reference
PER_SHOT = 6      # 6 x 5 strokes = 30 clips, enough to fill the picker's first rows

# The classifier has to agree with the stroke the picker advertises, too. Without this the set
# skews hard to drive: a well-played cut and a well-played drive share the full extension the
# joint sequence keys on, and an end-to-end run of the score-only selection called 6 of 10 clips
# a drive (50% accuracy against the dataset's ~75%). A clip captioned "Cut" that the app then
# reports as a drive is the exact failure this selection exists to avoid.
MIN_CLS_P = 0.5

# The bar is a per-stroke percentile rather than a flat score because coaches mark the strokes on
# different scales: in the test split the median defence is 5.33 and the median drive 7.40. A flat
# 7.0 bar admitted 29 drives and 2 defences, which would leave the panel's defence row empty.
# Judging each clip against its own stroke's reference -- the same distribution the app reports a
# percentile against -- keeps "rated well" meaning the same thing for all five.
#
# It is set at 55 rather than 75 because of a ceiling in the scorer itself. Among the 150 candidates
# the classifier gets right, the highest percentile any cut reaches is 66 and any pull/hook 69,
# against 90 for drive and 87 for defence -- the regression's output is compressed for those two
# strokes, so clips coaches marked 7.6-9.5 still land mid-distribution. A 75 bar therefore empties
# the cut and pull/hook rows for a reason that has nothing to do with the clips. Taking each
# stroke's best PER_SHOT above "typical" keeps the panel balanced, and app/static/app.js words each
# clip's badge from its own percentile so none of them is advertised as better than it scored.


def seq_tensor(seq_idx, clip_id):
    """The scorer's input for one clip: the saved joint sequence, normalised the same way training did."""
    return torch.from_numpy(normalise(np.load(ROOT / seq_idx.loc[clip_id, "seq_path"])["seq"]))


def main():
    ref = json.load(open(ROOT / "app/reference.json"))
    targets = ref["targets"]
    scorer = ContrastiveScorer(51, len(targets), hidden=128)
    scorer.load_state_dict(torch.load(
        ROOT / "models/technique_scorer_contrastive_residual_seed2/contrastive_scorer.pt", map_location="cpu"))
    scorer.eval()

    clf = make_model("lstm", 51, len(SHOTS), 32)
    clf.load_state_dict(torch.load(ROOT / "models/shot_classifier/lstm.pt", map_location="cpu"))
    clf.eval()

    seq = pd.read_parquet(ROOT / "data/processed/sequences.parquet").set_index("clip_id")
    idx = pd.read_parquet(ROOT / "data/processed/pose_index.parquet")
    score_cols = [f"score_{t}" for t in targets]

    # Exemplar embeddings, grouped by shot -- same set the app scores against.
    ex = pd.read_csv(ROOT / "kaggle/exemplars.csv")
    diff_mu, diff_sd = np.array(ref["contrastive_diff_mu"]), np.array(ref["contrastive_diff_sd"])
    banks = {}
    with torch.no_grad():
        for shot, g in ex.groupby("shot"):
            g = g[g.clip_id.isin(seq.index)]
            if g.empty:
                continue
            x = torch.stack([seq_tensor(seq, c) for c in g.clip_id])
            banks[shot] = (scorer.embed(x),
                           torch.from_numpy(to_residual(g[score_cols].to_numpy(np.float32))))

    levels = np.array(ref["quantile_levels"])

    def pct_of(shot, value):
        """Where value sits in that stroke's overall-score reference, as the app computes it."""
        q = np.array(ref["score"][shot if shot in ref["score"] else "all"]["overall"])
        return float(np.interp(value, q, levels))

    cand = idx[(idx.split == "test") & idx.shot.isin(SCORED)
               & (idx.score_overall >= MIN_LABEL) & idx.clip_id.isin(seq.index)]
    rows = []
    with torch.no_grad():
        for r in cand.itertuples():
            ze, ey = banks[r.shot]
            x = seq_tensor(seq, r.clip_id)[None]
            zq = scorer.embed(x).expand(len(ze), -1)
            diff = scorer.head(torch.cat([zq, ze, zq - ze], dim=-1)).numpy()
            pred = from_residual((ey.numpy() + diff * diff_sd + diff_mu).mean(axis=0))

            # Same masked softmax over the five offered strokes that app/pipeline.py applies.
            logits = clf(x)[0].numpy()
            masked = np.full_like(logits, -np.inf)
            masked[SCORED_IDX] = logits[SCORED_IDX]
            e = np.exp(masked - masked[SCORED_IDX].max())
            probs = e / e.sum()
            top = SHOTS[int(np.nanargmax(np.where(np.isfinite(probs), probs, -np.inf)))]

            rows.append((r.clip_id, r.source, r.shot, float(r.score_overall),
                         float(np.clip(pred[targets.index("overall")], 0, 10)),
                         top, float(probs[SHOTS.index(r.shot)])))

    d = pd.DataFrame(rows, columns=["clip_id", "source", "shot", "label_score", "model_score",
                                    "pred_shot", "pred_p"])
    d["model_pct"] = [pct_of(r.shot, r.model_score) for r in d.itertuples()]
    good = d[(d.model_pct >= MIN_PCT) & (d.pred_shot == d.shot) & (d.pred_p >= MIN_CLS_P)]
    good = good.sort_values("model_pct", ascending=False)
    pick = good.groupby("shot").head(PER_SHOT).sort_values(["shot", "model_pct"], ascending=[True, False])

    out = {"note": "Held-out test clips the scorer rates well. Generated by scripts/pick_demo_clips.py.",
           "clips": [{"clip_id": r.clip_id, "source": r.source, "shot": r.shot,
                      "label_score": round(r.label_score, 2), "model_score": round(r.model_score, 2),
                      "model_pct": round(r.model_pct)}
                     for r in pick.itertuples()]}
    (ROOT / "app/demo_clips.json").write_text(json.dumps(out, indent=1) + "\n")
    print(f"{len(cand)} candidates -> {len(good)} above the {MIN_PCT}th percentile of their own "
          f"stroke and classified right -> picked {len(pick)}: "
          + ", ".join(f"{s} {n}" for s, n in pick.groupby('shot').size().items()))
    print(f"model score {pick.model_score.min():.2f}-{pick.model_score.max():.2f} "
          f"({pick.model_pct.min():.0f}th-{pick.model_pct.max():.0f}th percentile), "
          f"coach label {pick.label_score.min():.2f}-{pick.label_score.max():.2f}")


if __name__ == "__main__":
    main()
