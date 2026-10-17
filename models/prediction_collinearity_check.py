"""Do either technique scorer's PREDICTIONS actually differentiate the five body parts, or do they just
reproduce the label collinearity (docs/paper/finding_label_collinearity.md: labels' PC1 = 97.5% of
variance)? partial_spearman_vs_overall (models/evaluate.py) answers this per-part; this does the same
PCA/correlation check label_correlation_analysis.py ran on the labels, but on each model's test-set
predictions, so the comparison is apples to apples.

Loads both trained checkpoints (VAE baseline, contrastive) and predicts on the test split.

    python models/prediction_collinearity_check.py
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from models.train_shot_classifier import normalise  # noqa: E402
from models.train_technique_scorer import PARTS, TARGETS, VAERegressor, load_scored_sequences  # noqa: E402
from models.train_technique_scorer_contrastive import (ContrastiveScorer, cache_sequences,  # noqa: E402
                                                         load_data, predict_scores)


def pc1_share(df: pd.DataFrame, cols: list[str]) -> float:
    X = df[cols].dropna().to_numpy(float)
    X = (X - X.mean(0)) / X.std(0)
    cov = np.cov(X, rowvar=False)
    eigvals = np.linalg.eigvalsh(cov)[::-1]
    return float(eigvals[0] / eigvals.sum())


def main():
    device = "cpu"

    # VAE baseline
    d, mu_t, sd_t = load_scored_sequences()
    te = d[d.split == "test"].reset_index(drop=True)
    z0 = np.load(ROOT / d.seq_path.iloc[0])
    t_len, in_dim = z0["seq"].shape[0], z0["seq"].shape[1] * z0["seq"].shape[2]
    vae = VAERegressor(in_dim, t_len, len(TARGETS))
    vae.load_state_dict(torch.load(ROOT / "models/technique_scorer/vae_regressor.pt", map_location=device))
    vae.eval()
    preds = []
    with torch.no_grad():
        for r in te.itertuples():
            seq = torch.from_numpy(normalise(np.load(ROOT / r.seq_path)["seq"])).unsqueeze(0)
            _, mu, _, sc = vae(seq)
            preds.append((sc.numpy()[0] * sd_t) + mu_t)
    vae_pred = np.array(preds)
    vae_df = te.assign(**{f"pred_{p}": vae_pred[:, i] for i, p in enumerate(TARGETS)})

    # Contrastive
    cd, exemplars = load_data()
    cte = cd[cd.split == "test"].reset_index(drop=True)
    all_clips = set(cd.clip_id) | set(exemplars.clip_id)
    seq = pd.read_parquet(ROOT / "data/processed/sequences.parquet")
    seq_path_by_clip = {**dict(zip(cd.clip_id, cd.seq_path)),
                         **dict(zip(exemplars.clip_id, seq.set_index("clip_id").loc[exemplars.clip_id, "seq_path"]))}
    cache = cache_sequences(all_clips, seq_path_by_clip)
    score_cols = [f"score_{p}" for p in TARGETS]
    ex_by_shot_df = {s: g for s, g in exemplars.groupby("shot")}
    tr = cd[cd.split == "train"]
    diffs = []
    for r in tr.itertuples():
        qy = np.array([getattr(r, c) for c in score_cols], np.float32)
        for e in ex_by_shot_df[r.shot].itertuples():
            ey = np.array([getattr(e, c) for c in score_cols], np.float32)
            diffs.append(qy - ey)
    diffs = np.stack(diffs)
    diff_mu, diff_sd = diffs.mean(0).astype(np.float32), diffs.std(0).astype(np.float32)
    diff_sd[diff_sd < 1e-3] = 1.0
    con = ContrastiveScorer(next(iter(cache.values())).shape[1], len(TARGETS))
    con.load_state_dict(torch.load(ROOT / "models/technique_scorer_contrastive/contrastive_scorer.pt",
                                    map_location=device))
    con_pred = predict_scores(con, cte, exemplars, cache, diff_mu, diff_sd, device)
    con_df = cte.assign(**{f"pred_{p}": con_pred[:, i] for i, p in enumerate(TARGETS)})

    label_cols = [f"score_{p}" for p in PARTS]
    pred_cols = [f"pred_{p}" for p in PARTS]
    print("PC1 share of variance across the 5 body-part scores (97.5% on labels is the known baseline):")
    print(f"  labels (test split, VAE's rows)         : {pc1_share(vae_df, label_cols):.3f}")
    print(f"  VAE baseline predictions                 : {pc1_share(vae_df, pred_cols):.3f}")
    print(f"  labels (test split, contrastive's rows)  : {pc1_share(con_df, label_cols):.3f}")
    print(f"  contrastive predictions                  : {pc1_share(con_df, pred_cols):.3f}")

    print("\nMean pairwise correlation between the 5 predicted part scores (labels ~0.95-0.99):")
    print(f"  VAE baseline        : {vae_df[pred_cols].corr().to_numpy()[np.triu_indices(5, 1)].mean():.3f}")
    print(f"  contrastive         : {con_df[pred_cols].corr().to_numpy()[np.triu_indices(5, 1)].mean():.3f}")


if __name__ == "__main__":
    main()
