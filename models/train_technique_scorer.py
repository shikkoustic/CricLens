"""Technique scorer: VAE + regression on CricketVision scores (PROGRESS.md step 3).

Input: the same resampled, spatially-normalised pose sequences as the shot classifier
(models/train_shot_classifier.py's normalise()), restricted to the 4,872 train-ready clips carrying
CricketVision's head/shoulder/hands/hips/feet/overall scores.

Why a VAE. A plain regressor can memorise idiosyncrasies of 3,718 training sequences; forcing the
encoder through a bottleneck that must also reconstruct the input is a mild regulariser and gives a
compact per-clip representation (the latent mean) that could be reused later (e.g. for the coaching
LLM, or nearest-exemplar retrieval, per docs/paper/aligned_papers.md's contrastive-regression note).
Architecture: GRU encoder -> (mu, logvar) -> reparameterised z -> GRU decoder (reconstruction) and,
separately, an MLP regression head reading the same z -> 6 scores (5 parts + overall, predicted
jointly rather than deriving one from the other).

Evaluation is the point of this script, more than the number. docs/paper/finding_label_collinearity.md
found CricketVision's five part scores correlate at 0.95-0.99 in the labels themselves -- so a model
matching or beating I3D-AE-LSTM's raw per-part Spearman (~0.84, spread 0.003) would not by itself show
independent per-part discrimination; it could just be fitting the shared factor those labels share.
models/evaluate.py's partial_spearman_vs_overall is reported alongside raw Spearman for exactly this
reason: raw Spearman says "does this track quality", partial says "does this track THIS BODY PART
beyond quality". Both numbers are needed to make any claim about part-specific technique feedback.

    python models/train_technique_scorer.py --epochs 60
    python models/train_technique_scorer.py --epochs 2 --limit 200   # fast smoke test
"""
import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from models.evaluate import evaluate_rating, format_report  # noqa: E402
from models.train_shot_classifier import normalise  # noqa: E402  -- same spatial normalisation, one definition

PARTS = ["head", "shoulder", "hands", "hips", "feet"]
TARGETS = PARTS + ["overall"]  # predict all 6 jointly; "overall" is not derived post-hoc from the 5 parts
OUT_DIR = ROOT / "models/technique_scorer"


def load_scored_sequences(limit: int | None = None) -> tuple[pd.DataFrame, np.ndarray, np.ndarray]:
    idx = pd.read_parquet(ROOT / "data/processed/pose_index.parquet")
    seq = pd.read_parquet(ROOT / "data/processed/sequences.parquet")
    score_cols = [f"score_{p}" for p in TARGETS]
    d = idx[idx.train_ready & idx.score_overall.notna()][["clip_id", "split", "source", "handedness"] + score_cols]
    d = d.merge(seq[["clip_id", "seq_path"]], on="clip_id")
    d = d.head(limit) if limit else d
    mu = d[score_cols].mean().to_numpy(np.float32)  # standardise targets for stable joint training across
    sd = d[score_cols].std().to_numpy(np.float32)   # parts with different natural scales (0-4 vs 0-15 raw)
    return d.reset_index(drop=True), mu, sd


class ScoredDataset(Dataset):
    def __init__(self, df: pd.DataFrame, mu: np.ndarray, sd: np.ndarray):
        self.df = df.reset_index(drop=True)
        self.mu, self.sd = mu, sd

    def __len__(self):
        return len(self.df)

    def __getitem__(self, i):
        r = self.df.iloc[i]
        seq = normalise(np.load(ROOT / r.seq_path)["seq"])
        y = (r[[f"score_{p}" for p in TARGETS]].to_numpy(np.float32) - self.mu) / self.sd
        return torch.from_numpy(seq), torch.from_numpy(y)


class VAERegressor(nn.Module):
    """GRU encoder -> VAE bottleneck -> GRU decoder (reconstruction) + MLP head (score regression)."""

    def __init__(self, in_dim: int, t_len: int, n_targets: int, hidden: int = 128, latent: int = 32):
        super().__init__()
        self.t_len, self.latent = t_len, latent
        self.encoder = nn.GRU(in_dim, hidden, num_layers=2, batch_first=True, dropout=0.2)
        self.to_mu = nn.Linear(hidden, latent)
        self.to_logvar = nn.Linear(hidden, latent)
        self.dec_init = nn.Linear(latent, hidden)
        self.decoder = nn.GRU(in_dim, hidden, num_layers=2, batch_first=True, dropout=0.2)
        self.dec_out = nn.Linear(hidden, in_dim)
        self.reg_head = nn.Sequential(nn.LayerNorm(latent), nn.Linear(latent, hidden), nn.ReLU(),
                                       nn.Dropout(0.2), nn.Linear(hidden, n_targets))

    def encode(self, x):
        _, h = self.encoder(x)  # h: (layers, B, hidden)
        h_last = h[-1]
        return self.to_mu(h_last), self.to_logvar(h_last)

    def reparameterise(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        return mu + torch.randn_like(std) * std if self.training else mu

    def decode(self, z, x_teacher):
        """Teacher-forced reconstruction: decoder GRU seeded with z, fed the true sequence shifted by one
        step (standard seq2seq training trick; at this input scale attention/scheduled sampling is not
        needed). x_teacher: (B, T, in_dim)."""
        h0 = self.dec_init(z).unsqueeze(0).repeat(2, 1, 1)  # (layers=2, B, hidden)
        dec_in = F.pad(x_teacher[:, :-1], (0, 0, 1, 0))  # shift right, zero at t=0
        out, _ = self.decoder(dec_in, h0)
        return self.dec_out(out)

    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterise(mu, logvar)
        recon = self.decode(z, x)
        scores = self.reg_head(mu if not self.training else z)  # regress from mu at eval time: deterministic
        return recon, mu, logvar, scores


def vae_loss(recon, x, mu, logvar, scores, y, kl_weight: float, reg_weight: float):
    recon_loss = F.mse_loss(recon, x)
    kl = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())
    reg_loss = F.mse_loss(scores, y)
    return recon_loss + kl_weight * kl + reg_weight * reg_loss, recon_loss.item(), kl.item(), reg_loss.item()


def run(epochs: int, limit: int | None, device: str, batch_size: int = 64, patience: int = 8,
        kl_weight: float = 0.01, reg_weight: float = 5.0) -> dict:
    d, mu_t, sd_t = load_scored_sequences(limit)
    tr, va, te = (d[d.split == s] for s in ("train", "val", "test"))
    print(f"train={len(tr)} val={len(va)} test={len(te)}")

    z0 = np.load(ROOT / d.seq_path.iloc[0])
    t_len, in_dim = z0["seq"].shape[0], z0["seq"].shape[1] * z0["seq"].shape[2]
    model = VAERegressor(in_dim, t_len, len(TARGETS)).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)

    dl_tr = DataLoader(ScoredDataset(tr, mu_t, sd_t), batch_size=batch_size, shuffle=True)
    dl_va = DataLoader(ScoredDataset(va, mu_t, sd_t), batch_size=256)

    def predict(dl):
        model.eval(); preds = []
        with torch.no_grad():
            for xb, _ in dl:
                _, mu, _, sc = model(xb.to(device))
                preds.append((sc.cpu().numpy() * sd_t) + mu_t)
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
        print(f"  epoch {ep+1}/{epochs}  loss={tot['loss']/len(tr):.3f} (recon={tot['recon']/len(tr):.3f} "
              f"kl={tot['kl']/len(tr):.3f} reg={tot['reg']/len(tr):.3f})  val_rho={r['mean_spearman']:.3f}  "
              f"val_partial_rho={r['mean_partial_spearman_vs_overall']:.3f}  ({time.time()-t0:.1f}s)")
        if r["mean_spearman"] > best_rho:
            best_rho, best_state, bad = r["mean_spearman"], {k: v.cpu().clone() for k, v in model.state_dict().items()}, 0
        else:
            bad += 1
            if bad >= patience:
                print(f"  early stop (no val mean-Spearman improvement in {patience} epochs)"); break

    model.load_state_dict(best_state)
    dl_te = DataLoader(ScoredDataset(te, mu_t, sd_t), batch_size=256)
    te_pred = predict(dl_te)
    te_df = te.assign(**{f"pred_{p}": te_pred[:, i] for i, p in enumerate(TARGETS)})
    result = evaluate_rating(te_df, {p: f"score_{p}" for p in PARTS}, {p: f"pred_{p}" for p in PARTS},
                              by="source", overall_cols=("score_overall", "pred_overall"))
    result_h = evaluate_rating(te_df.dropna(subset=["handedness"]), {p: f"score_{p}" for p in PARTS},
                                {p: f"pred_{p}" for p in PARTS}, by="handedness",
                                overall_cols=("score_overall", "pred_overall")) if te_df.handedness.notna().any() else None
    overall_spearman = float(pd.Series(te_df.score_overall).rank().corr(pd.Series(te_df.pred_overall).rank(), method="pearson"))

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    torch.save(best_state, OUT_DIR / "vae_regressor.pt")
    print(format_report(result, "\nTEST"))
    if result_h:
        print(format_report(result_h, "TEST by handedness"))
    print(f"\noverall-score Spearman (test): {overall_spearman:.3f}  (I3D-AE-LSTM baseline: 0.84)")
    summary = {"best_val_mean_spearman": best_rho, "test_mean_spearman": result["mean_spearman"],
               "test_mean_r_l2": result["mean_r_l2"], "test_mean_partial_spearman_vs_overall": result["mean_partial_spearman_vs_overall"],
               "test_overall_spearman": overall_spearman, "per_part": {p: result["per_part"][p] for p in PARTS},
               "n_train": len(tr), "n_val": len(va), "n_test": len(te)}
    json.dump(summary, open(OUT_DIR / "summary.json", "w"), indent=2)
    return summary


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=60)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()
    run(args.epochs, args.limit, args.device)


if __name__ == "__main__":
    main()
