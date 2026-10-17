"""Ablation: does the residual-target trick alone (score_part - score_overall, no contrastive
mechanism) also fix the VAE baseline's prediction collinearity, or is that specific to the contrastive
setup? PROGRESS.md 2026-10-17: residual targets took the contrastive model from predictions MORE
collinear than the labels (PC1 ~0.998) to 0.991, and partial-rho vs overall from 0.034 to ~0.12. This
isolates whether that gain needs the contrastive query-vs-exemplar framing, or whether it's just the
target transform doing the work -- an important control before claiming contrastive regression itself
is the novelty, rather than "train on relative-to-overall targets" being the real fix (which would
apply to any architecture, baseline included).

Reuses VAERegressor/ScoredDataset/load_scored_sequences from models/train_technique_scorer.py
unmodified (that script stays the untouched reference baseline); only the target construction and
prediction reconstruction differ, via train_technique_scorer_contrastive's to_residual/from_residual.

    python models/ablation_vae_residual.py --epochs 60
"""
import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from models.evaluate import evaluate_rating, format_report  # noqa: E402
from models.train_shot_classifier import normalise  # noqa: E402
from models.train_technique_scorer import PARTS, TARGETS, VAERegressor, load_scored_sequences, vae_loss  # noqa: E402
from models.train_technique_scorer_contrastive import from_residual, to_residual  # noqa: E402

OUT_DIR = ROOT / "models/technique_scorer_vae_residual"


class ResidualScoredDataset(Dataset):
    """Same as train_technique_scorer.ScoredDataset, but targets are to_residual'd before standardising."""

    def __init__(self, df: pd.DataFrame, mu: np.ndarray, sd: np.ndarray):
        self.df = df.reset_index(drop=True)
        self.mu, self.sd = mu, sd

    def __len__(self):
        return len(self.df)

    def __getitem__(self, i):
        r = self.df.iloc[i]
        seq = normalise(np.load(ROOT / r.seq_path)["seq"])
        y = to_residual(r[[f"score_{p}" for p in TARGETS]].to_numpy(np.float32))
        y = (y - self.mu) / self.sd
        return torch.from_numpy(seq), torch.from_numpy(y)


def run(epochs: int, limit: int | None, device: str, batch_size: int = 64, patience: int = 8,
        kl_weight: float = 0.001, reg_weight: float = 10.0, seed: int = 0) -> dict:
    torch.manual_seed(seed)
    np.random.seed(seed)
    d, _, _ = load_scored_sequences(limit)  # mu_t/sd_t from the base loader are for ABSOLUTE targets; unused here
    tr, va, te = (d[d.split == s] for s in ("train", "val", "test"))
    print(f"train={len(tr)} val={len(va)} test={len(te)}")

    score_cols = [f"score_{p}" for p in TARGETS]
    y_res = np.stack([to_residual(r) for r in tr[score_cols].to_numpy(np.float32)])
    mu_t, sd_t = y_res.mean(0).astype(np.float32), y_res.std(0).astype(np.float32)
    sd_t[sd_t < 1e-3] = 1.0
    print(f"residual-target distribution (train): mu={mu_t.round(2)} sd={sd_t.round(2)}")

    z0 = np.load(ROOT / d.seq_path.iloc[0])
    t_len, in_dim = z0["seq"].shape[0], z0["seq"].shape[1] * z0["seq"].shape[2]
    model = VAERegressor(in_dim, t_len, len(TARGETS)).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)

    dl_tr = DataLoader(ResidualScoredDataset(tr, mu_t, sd_t), batch_size=batch_size, shuffle=True)
    dl_va = DataLoader(ResidualScoredDataset(va, mu_t, sd_t), batch_size=256)

    def predict(dl):
        model.eval(); preds = []
        with torch.no_grad():
            for xb, _ in dl:
                _, mu, _, sc = model(xb.to(device))
                preds.append(from_residual((sc.cpu().numpy() * sd_t) + mu_t))
        return np.concatenate(preds)

    best_rho, best_state, bad = -1.0, None, 0
    for ep in range(epochs):
        model.train(); t0 = time.time(); tot = {"loss": 0.0, "recon": 0.0, "kl": 0.0, "reg": 0.0}
        for xb, yb in dl_tr:
            xb, yb = xb.to(device), yb.to(device)
            opt.zero_grad()
            recon, mvu, logvar, scores = model(xb)
            loss, rl, kl, rg = vae_loss(recon, xb, mvu, logvar, scores, yb, kl_weight, reg_weight)
            loss.backward(); opt.step()
            for k, v in zip(tot, (loss.item(), rl, kl, rg)):
                tot[k] += v * len(yb)
        va_pred = predict(dl_va)
        va_df = va.assign(**{f"pred_{p}": va_pred[:, i] for i, p in enumerate(TARGETS)})
        r = evaluate_rating(va_df, {p: f"score_{p}" for p in PARTS}, {p: f"pred_{p}" for p in PARTS},
                             overall_cols=("score_overall", "pred_overall"))
        print(f"  epoch {ep+1}/{epochs}  loss={tot['loss']/len(tr):.3f}  val_rho={r['mean_spearman']:.3f}  "
              f"val_partial_rho={r['mean_partial_spearman_vs_overall']:.3f}  ({time.time()-t0:.1f}s)")
        if r["mean_spearman"] > best_rho:
            best_rho, best_state, bad = r["mean_spearman"], {k: v.cpu().clone() for k, v in model.state_dict().items()}, 0
        else:
            bad += 1
            if bad >= patience:
                print(f"  early stop (no val mean-Spearman improvement in {patience} epochs)"); break

    model.load_state_dict(best_state)
    dl_te = DataLoader(ResidualScoredDataset(te, mu_t, sd_t), batch_size=256)
    te_pred = predict(dl_te)
    te_df = te.assign(**{f"pred_{p}": te_pred[:, i] for i, p in enumerate(TARGETS)})
    result = evaluate_rating(te_df, {p: f"score_{p}" for p in PARTS}, {p: f"pred_{p}" for p in PARTS},
                              by="source", overall_cols=("score_overall", "pred_overall"))
    overall_spearman = float(pd.Series(te_df.score_overall).rank().corr(pd.Series(te_df.pred_overall).rank(), method="pearson"))

    pred_cols = [f"pred_{p}" for p in PARTS]
    X = te_df[pred_cols].to_numpy(float); X = (X - X.mean(0)) / X.std(0)
    pc1_share = float(np.linalg.eigvalsh(np.cov(X, rowvar=False))[::-1][0] / np.linalg.eigvalsh(np.cov(X, rowvar=False)).sum())
    mean_pairwise_corr = float(te_df[pred_cols].corr().to_numpy()[np.triu_indices(5, 1)].mean())

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    torch.save(best_state, OUT_DIR / "vae_residual.pt")
    print(format_report(result, "\nTEST"))
    print(f"\noverall-score Spearman (test): {overall_spearman:.3f}")
    print(f"predictions PC1 share: {pc1_share:.3f}  mean pairwise correlation: {mean_pairwise_corr:.3f}")
    summary = {"seed": seed, "best_val_mean_spearman": best_rho, "test_mean_spearman": result["mean_spearman"],
               "test_mean_r_l2": result["mean_r_l2"], "test_mean_partial_spearman_vs_overall": result["mean_partial_spearman_vs_overall"],
               "test_overall_spearman": overall_spearman, "pc1_share": pc1_share, "mean_pairwise_corr": mean_pairwise_corr,
               "per_part": {p: result["per_part"][p] for p in PARTS}, "n_train": len(tr), "n_val": len(va), "n_test": len(te)}
    json.dump(summary, open(OUT_DIR / "summary.json", "w"), indent=2)
    return summary


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=60)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    run(args.epochs, args.limit, args.device, seed=args.seed)


if __name__ == "__main__":
    main()
