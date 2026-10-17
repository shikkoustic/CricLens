"""Technique scorer v2: contrastive regression against fixed exemplars (PROGRESS.md "novelty" step).

Rather than regressing a query clip straight to an absolute score (models/train_technique_scorer.py,
the VAE baseline), this follows the CoRe/TSA/MCoRe family (docs/paper/aligned_papers.md, docs/paper/
dossier.md B10 Qi et al.): predict the score *difference* between a query clip and each of several
same-shot "exemplar" clips with known scores (kaggle/exemplars.csv, 50 clips -- 10 per shot type,
spread across score tiers, picked and visually verified in models/exemplar_review.py), then average
exemplar_score + predicted_diff across all of that shot's exemplars at inference. The averaging is
the robustness step CoRe gets from a single exemplar's noise; see PROGRESS.md for why this is the
candidate novelty for the paper rather than a from-scratch architecture change.

Exemplars are drawn only from the train split (kaggle/exemplars.csv) and are never themselves scored
as queries, so val/test clips never leak into the reference set. A Siamese GRU encoder (shared weights
for query and exemplar) embeds both sequences; the diff head reads [z_q, z_e, z_q - z_e] and predicts
all 6 targets (5 parts + overall) jointly, same rationale as the VAE baseline for not deriving overall
post-hoc from the parts.

    python models/train_technique_scorer_contrastive.py --epochs 60
    python models/train_technique_scorer_contrastive.py --epochs 2 --limit 200   # fast smoke test
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
from torch.utils.data import DataLoader, Dataset

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from models.evaluate import evaluate_rating, format_report  # noqa: E402
from models.train_shot_classifier import normalise  # noqa: E402
from models.train_technique_scorer import PARTS, TARGETS  # noqa: E402

EXEMPLARS = ROOT / "kaggle/exemplars.csv"
OUT_DIR = ROOT / "models/technique_scorer_contrastive"


def load_data(limit: int | None = None) -> tuple[pd.DataFrame, pd.DataFrame]:
    idx = pd.read_parquet(ROOT / "data/processed/pose_index.parquet")
    seq = pd.read_parquet(ROOT / "data/processed/sequences.parquet")
    score_cols = [f"score_{p}" for p in TARGETS]
    d = idx[idx.train_ready & idx.score_overall.notna() & (idx.shot != "other")][
        ["clip_id", "split", "source", "handedness", "shot"] + score_cols]
    d = d.merge(seq[["clip_id", "seq_path"]], on="clip_id")

    exemplars = pd.read_csv(EXEMPLARS)
    d = d[~d.clip_id.isin(exemplars.clip_id)]  # exemplars are references, never queries
    d = d.head(limit) if limit else d
    return d.reset_index(drop=True), exemplars.reset_index(drop=True)


def cache_sequences(clip_ids: set[str], seq_path_by_clip: dict[str, str]) -> dict[str, np.ndarray]:
    """Load + normalise every referenced sequence once, shared across every (query, exemplar) pair --
    each exemplar is reused in thousands of pairs, so this avoids re-reading its npz every time."""
    return {cid: normalise(np.load(ROOT / seq_path_by_clip[cid])["seq"]) for cid in clip_ids}


class PairDataset(Dataset):
    """One item per (query, same-shot exemplar) pair. diff_mu/diff_sd standardise the target diffs
    for stable training (same reasoning as the VAE baseline standardising absolute scores)."""

    def __init__(self, df: pd.DataFrame, exemplars: pd.DataFrame, cache: dict[str, np.ndarray],
                 diff_mu: np.ndarray, diff_sd: np.ndarray):
        score_cols = [f"score_{p}" for p in TARGETS]
        ex_by_shot = {s: g for s, g in exemplars.groupby("shot")}
        self.pairs = []  # (query clip_id, exemplar clip_id, query_scores, exemplar_scores)
        for r in df.itertuples():
            qy = np.array([getattr(r, c) for c in score_cols], np.float32)
            for e in ex_by_shot[r.shot].itertuples():
                ey = np.array([getattr(e, c) for c in score_cols], np.float32)
                self.pairs.append((r.clip_id, e.clip_id, qy, ey))
        self.cache, self.diff_mu, self.diff_sd = cache, diff_mu, diff_sd

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, i):
        qid, eid, qy, ey = self.pairs[i]
        diff = (qy - ey - self.diff_mu) / self.diff_sd
        return (torch.from_numpy(self.cache[qid]), torch.from_numpy(self.cache[eid]),
                torch.from_numpy(diff), torch.from_numpy(ey))


class ContrastiveScorer(nn.Module):
    """Siamese GRU encoder (shared weights) + diff head on [z_q, z_e, z_q - z_e]."""

    def __init__(self, in_dim: int, n_targets: int, hidden: int = 128):
        super().__init__()
        self.encoder = nn.GRU(in_dim, hidden, num_layers=2, batch_first=True, dropout=0.2)
        self.head = nn.Sequential(nn.LayerNorm(hidden * 3), nn.Linear(hidden * 3, hidden), nn.ReLU(),
                                   nn.Dropout(0.2), nn.Linear(hidden, n_targets))

    def embed(self, x):
        _, h = self.encoder(x)
        return h[-1]

    def forward(self, xq, xe):
        zq, ze = self.embed(xq), self.embed(xe)
        return self.head(torch.cat([zq, ze, zq - ze], dim=-1))


def predict_scores(model: ContrastiveScorer, df: pd.DataFrame, exemplars: pd.DataFrame,
                    cache: dict[str, np.ndarray], diff_mu: np.ndarray, diff_sd: np.ndarray,
                    device: str, batch_size: int = 256) -> np.ndarray:
    """Average exemplar_score + predicted_diff across every exemplar of the query's shot type --
    the CoRe-style robustness step. Returns (n_queries, n_targets)."""
    model.eval()
    score_cols = [f"score_{p}" for p in TARGETS]
    ex_by_shot = {s: g[score_cols].to_numpy(np.float32) for s, g in exemplars.groupby("shot")}
    ex_ids_by_shot = {s: g.clip_id.tolist() for s, g in exemplars.groupby("shot")}
    out = np.zeros((len(df), len(TARGETS)), np.float32)
    with torch.no_grad():
        for shot, grp in df.groupby("shot"):
            ex_scores = ex_by_shot[shot]  # (n_ex, n_targets)
            ex_ids = ex_ids_by_shot[shot]
            xe_all = torch.stack([torch.from_numpy(cache[eid]) for eid in ex_ids]).to(device)  # (n_ex, T, D)
            idx = grp.index.to_numpy()
            for s in range(0, len(idx), batch_size):
                sub = idx[s:s + batch_size]
                xq = torch.stack([torch.from_numpy(cache[df.clip_id[i]]) for i in sub]).to(device)  # (B, T, D)
                b, n_ex = xq.shape[0], xe_all.shape[0]
                xq_rep = xq.unsqueeze(1).expand(-1, n_ex, -1, -1).reshape(b * n_ex, *xq.shape[1:])
                xe_rep = xe_all.unsqueeze(0).expand(b, -1, -1, -1).reshape(b * n_ex, *xe_all.shape[1:])
                diff = model(xq_rep, xe_rep).cpu().numpy().reshape(b, n_ex, -1) * diff_sd + diff_mu
                pred = ex_scores[None, :, :] + diff  # (B, n_ex, n_targets)
                out[sub] = pred.mean(axis=1)
    return out


def run(epochs: int, limit: int | None, device: str, batch_size: int = 64, patience: int = 8,
        lr: float = 1e-3, hidden: int = 128, weight_decay: float = 1e-4, seed: int = 0,
        tag: str = "") -> dict:
    torch.manual_seed(seed)
    np.random.seed(seed)
    d, exemplars = load_data(limit)
    tr, va, te = (d[d.split == s] for s in ("train", "val", "test"))
    print(f"train={len(tr)} val={len(va)} test={len(te)}  exemplars={len(exemplars)}")

    all_clips = set(d.clip_id) | set(exemplars.clip_id)
    seq_path_by_clip = {**dict(zip(d.clip_id, d.seq_path)),
                         **dict(zip(exemplars.clip_id,
                                    pd.read_parquet(ROOT / "data/processed/sequences.parquet")
                                    .set_index("clip_id").loc[exemplars.clip_id, "seq_path"]))}
    cache = cache_sequences(all_clips, seq_path_by_clip)

    score_cols = [f"score_{p}" for p in TARGETS]
    ex_by_shot_df = {s: g for s, g in exemplars.groupby("shot")}
    diffs = []
    for r in tr.itertuples():
        qy = np.array([getattr(r, c) for c in score_cols], np.float32)
        for e in ex_by_shot_df[r.shot].itertuples():
            ey = np.array([getattr(e, c) for c in score_cols], np.float32)
            diffs.append(qy - ey)
    diffs = np.stack(diffs)
    diff_mu, diff_sd = diffs.mean(0).astype(np.float32), diffs.std(0).astype(np.float32)
    diff_sd[diff_sd < 1e-3] = 1.0
    print(f"diff distribution (train pairs): mu={diff_mu.round(2)} sd={diff_sd.round(2)}")

    in_dim = next(iter(cache.values())).shape[1]
    model = ContrastiveScorer(in_dim, len(TARGETS), hidden=hidden).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

    dl_tr = DataLoader(PairDataset(tr, exemplars, cache, diff_mu, diff_sd), batch_size=batch_size, shuffle=True)

    best_rho, best_state, bad = -1.0, None, 0
    for ep in range(epochs):
        model.train(); t0 = time.time(); tot_loss = 0.0
        for xq, xe, diff, _ in dl_tr:
            xq, xe, diff = xq.to(device), xe.to(device), diff.to(device)
            opt.zero_grad()
            pred = model(xq, xe)
            loss = nn.functional.mse_loss(pred, diff)
            loss.backward(); opt.step()
            tot_loss += loss.item() * len(diff)
        va_pred = predict_scores(model, va.reset_index(drop=True), exemplars, cache, diff_mu, diff_sd, device)
        va_df = va.reset_index(drop=True).assign(**{f"pred_{p}": va_pred[:, i] for i, p in enumerate(TARGETS)})
        r = evaluate_rating(va_df, {p: f"score_{p}" for p in PARTS}, {p: f"pred_{p}" for p in PARTS},
                             overall_cols=("score_overall", "pred_overall"))
        print(f"  epoch {ep+1}/{epochs}  loss={tot_loss/len(dl_tr.dataset):.3f}  val_rho={r['mean_spearman']:.3f}  "
              f"val_partial_rho={r['mean_partial_spearman_vs_overall']:.3f}  ({time.time()-t0:.1f}s)")
        if r["mean_spearman"] > best_rho:
            best_rho, best_state, bad = r["mean_spearman"], {k: v.cpu().clone() for k, v in model.state_dict().items()}, 0
        else:
            bad += 1
            if bad >= patience:
                print(f"  early stop (no val mean-Spearman improvement in {patience} epochs)"); break

    model.load_state_dict(best_state)
    te_r = te.reset_index(drop=True)
    te_pred = predict_scores(model, te_r, exemplars, cache, diff_mu, diff_sd, device)
    te_df = te_r.assign(**{f"pred_{p}": te_pred[:, i] for i, p in enumerate(TARGETS)})
    result = evaluate_rating(te_df, {p: f"score_{p}" for p in PARTS}, {p: f"pred_{p}" for p in PARTS},
                              by="source", overall_cols=("score_overall", "pred_overall"))
    result_h = evaluate_rating(te_df.dropna(subset=["handedness"]), {p: f"score_{p}" for p in PARTS},
                                {p: f"pred_{p}" for p in PARTS}, by="handedness",
                                overall_cols=("score_overall", "pred_overall")) if te_df.handedness.notna().any() else None
    overall_spearman = float(pd.Series(te_df.score_overall).rank().corr(pd.Series(te_df.pred_overall).rank(), method="pearson"))

    out_dir = OUT_DIR if not tag else OUT_DIR.parent / f"{OUT_DIR.name}_{tag}"
    out_dir.mkdir(parents=True, exist_ok=True)
    torch.save(best_state, out_dir / "contrastive_scorer.pt")
    print(format_report(result, "\nTEST"))
    if result_h:
        print(format_report(result_h, "TEST by handedness"))
    print(f"\noverall-score Spearman (test): {overall_spearman:.3f}  (I3D-AE-LSTM baseline: 0.84)")
    summary = {"tag": tag, "lr": lr, "hidden": hidden, "weight_decay": weight_decay, "seed": seed,
               "patience": patience, "best_epoch_val_mean_spearman": best_rho,
               "test_mean_spearman": result["mean_spearman"],
               "test_mean_r_l2": result["mean_r_l2"], "test_mean_partial_spearman_vs_overall": result["mean_partial_spearman_vs_overall"],
               "test_overall_spearman": overall_spearman, "per_part": {p: result["per_part"][p] for p in PARTS},
               "n_train": len(tr), "n_val": len(va), "n_test": len(te), "n_exemplars": len(exemplars)}
    json.dump(summary, open(out_dir / "summary.json", "w"), indent=2)
    return summary


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=60)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--hidden", type=int, default=128)
    ap.add_argument("--weight-decay", type=float, default=1e-4)
    ap.add_argument("--patience", type=int, default=8)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--tag", default="")
    args = ap.parse_args()
    run(args.epochs, args.limit, args.device, args.batch_size, patience=args.patience, lr=args.lr,
        hidden=args.hidden, weight_decay=args.weight_decay, seed=args.seed, tag=args.tag)


if __name__ == "__main__":
    main()
