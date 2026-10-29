"""Technique scorer v3: same contrastive setup as train_technique_scorer_contrastive.py, but the
Siamese encoder is a small ST-GCN (Yan et al., AAAI 2018, arXiv:1801.07455) instead of a GRU.

Why: the GRU treats each frame's 17 joints as one flat 51-d vector, so it never sees which joints are
anatomically connected -- "left wrist" and "left elbow" are just two unrelated input dims to it. An
ST-GCN block instead mixes each joint's features with its skeletal neighbours (shoulder<->elbow<->wrist,
hip<->knee<->ankle, ...) via a fixed graph-adjacency matmul (spatial conv), then runs a depthwise-ish
temporal conv across frames per joint -- the standard ST-GCN block order. This is the user-requested
"genuinely different architecture, not just a cheaper one" alternative to the GRU for the paper's
technique-scorer ablation (see PROGRESS.md): same data, same exemplar-pairing/diff-head/training loop
as v2, only the encoder changes, so any gap is attributable to the encoder alone.

    python models/train_technique_scorer_stgcn.py --epochs 60
    python models/train_technique_scorer_stgcn.py --epochs 2 --limit 200   # fast smoke test
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
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from models.evaluate import evaluate_rating, format_report  # noqa: E402
from models.train_shot_classifier import normalise  # noqa: E402
from models.train_technique_scorer import PARTS, TARGETS  # noqa: E402
from models.train_technique_scorer_contrastive import (  # noqa: E402
    EXEMPLARS, PairDataset, cache_sequences, from_residual, load_data, to_residual,
)

OUT_DIR = ROOT / "models/technique_scorer_stgcn"

# Standard 17-point COCO skeleton (pycocotools "skeleton" field, 0-indexed): the same joint ordering
# this project already uses (HIP_L, HIP_R, SHO_L, SHO_R = 11, 12, 5, 6 in train_shot_classifier.py).
COCO17_EDGES = [
    (15, 13), (13, 11), (16, 14), (14, 12), (11, 12),  # legs: ankle-knee-hip, hip-hip
    (5, 11), (6, 12), (5, 6),                           # torso: shoulder-hip, shoulder-shoulder
    (5, 7), (7, 9), (6, 8), (8, 10),                    # arms: shoulder-elbow-wrist
    (1, 2), (0, 1), (0, 2), (1, 3), (2, 4), (3, 5), (4, 6),  # face/head: eyes-nose-ears-shoulders
]
N_JOINTS = 17


def build_adjacency() -> torch.Tensor:
    """Symmetric-normalised adjacency with self-loops (Kipf & Welling GCN convention):
    A_hat = D^-1/2 (A + I) D^-1/2. Fixed for the whole model -- no learned/attention graph, since the
    skeleton topology is known and constant across every clip."""
    a = np.eye(N_JOINTS, dtype=np.float32)
    for i, j in COCO17_EDGES:
        a[i, j] = 1.0
        a[j, i] = 1.0
    d = a.sum(1)
    d_inv_sqrt = np.diag(1.0 / np.sqrt(d))
    a_hat = d_inv_sqrt @ a @ d_inv_sqrt
    return torch.tensor(a_hat, dtype=torch.float32)


class STGCNBlock(nn.Module):
    """One spatial-graph-conv + temporal-conv block, ST-GCN style. x: (B, T, J, C_in) -> (B, T, J, C_out)."""

    def __init__(self, in_c: int, out_c: int, temporal_kernel: int = 9, dropout: float = 0.2):
        super().__init__()
        self.spatial = nn.Linear(in_c, out_c)
        self.temporal = nn.Conv1d(out_c, out_c, kernel_size=temporal_kernel, padding=temporal_kernel // 2)
        self.bn = nn.BatchNorm1d(out_c)
        self.residual = nn.Linear(in_c, out_c) if in_c != out_c else nn.Identity()
        self.act = nn.ReLU()
        self.drop = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, a_hat: torch.Tensor) -> torch.Tensor:
        b, t, j, c = x.shape
        res = self.residual(x)                                   # (B, T, J, C_out)
        xs = torch.einsum("ij,btjc->btic", a_hat, x)              # spatial: mix each joint with neighbours
        xs = self.spatial(xs)                                     # (B, T, J, C_out)
        xt = xs.permute(0, 2, 3, 1).reshape(b * j, xs.shape[-1], t)  # (B*J, C_out, T) -- per-joint time series
        xt = self.bn(self.temporal(xt))
        xt = xt.reshape(b, j, xs.shape[-1], t).permute(0, 3, 1, 2)    # back to (B, T, J, C_out)
        return self.drop(self.act(xt + res))


class STGCNEncoder(nn.Module):
    """Stack of STGCNBlocks + global average pool over (T, J) -> (B, hidden)."""

    def __init__(self, in_c: int = 3, hidden: int = 128, dropout: float = 0.2):
        super().__init__()
        self.register_buffer("a_hat", build_adjacency())
        mid = hidden // 2
        self.block1 = STGCNBlock(in_c, mid, dropout=dropout)
        self.block2 = STGCNBlock(mid, hidden, dropout=dropout)
        self.block3 = STGCNBlock(hidden, hidden, dropout=dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B, T, 51) flattened (joint, channel) -- same cache format as the GRU encoder."""
        b, t, _ = x.shape
        x = x.view(b, t, N_JOINTS, 3)
        x = self.block1(x, self.a_hat)
        x = self.block2(x, self.a_hat)
        x = self.block3(x, self.a_hat)
        return x.mean(dim=(1, 2))  # (B, hidden)


class STGCNScorer(nn.Module):
    """Siamese ST-GCN encoder (shared weights) + diff head on [z_q, z_e, z_q - z_e] -- identical head
    to ContrastiveScorer (train_technique_scorer_contrastive.py) so only the encoder differs."""

    def __init__(self, n_targets: int, hidden: int = 128, dropout: float = 0.2):
        super().__init__()
        self.encoder = STGCNEncoder(in_c=3, hidden=hidden, dropout=dropout)
        self.head = nn.Sequential(nn.LayerNorm(hidden * 3), nn.Linear(hidden * 3, hidden), nn.ReLU(),
                                   nn.Dropout(dropout), nn.Linear(hidden, n_targets))

    def embed(self, x):
        return self.encoder(x)

    def forward(self, xq, xe):
        zq, ze = self.embed(xq), self.embed(xe)
        return self.head(torch.cat([zq, ze, zq - ze], dim=-1))


def predict_scores(model: STGCNScorer, df: pd.DataFrame, exemplars: pd.DataFrame,
                    cache: dict[str, np.ndarray], diff_mu: np.ndarray, diff_sd: np.ndarray,
                    device: str, batch_size: int = 256, residual_targets: bool = False) -> np.ndarray:
    """Same exemplar-averaging as train_technique_scorer_contrastive.predict_scores, just calling the
    ST-GCN model instead of the GRU one (duplicated rather than imported since the model type differs)."""
    model.eval()
    score_cols = [f"score_{p}" for p in TARGETS]
    xf = to_residual if residual_targets else (lambda y: y)
    ex_by_shot = {s: xf(g[score_cols].to_numpy(np.float32)) for s, g in exemplars.groupby("shot")}
    ex_ids_by_shot = {s: g.clip_id.tolist() for s, g in exemplars.groupby("shot")}
    out = np.zeros((len(df), len(TARGETS)), np.float32)
    with torch.no_grad():
        for shot, grp in df.groupby("shot"):
            ex_scores = ex_by_shot[shot]
            ex_ids = ex_ids_by_shot[shot]
            xe_all = torch.stack([torch.from_numpy(cache[eid]) for eid in ex_ids]).to(device)
            idx = grp.index.to_numpy()
            for s in range(0, len(idx), batch_size):
                sub = idx[s:s + batch_size]
                xq = torch.stack([torch.from_numpy(cache[df.clip_id[i]]) for i in sub]).to(device)
                b, n_ex = xq.shape[0], xe_all.shape[0]
                xq_rep = xq.unsqueeze(1).expand(-1, n_ex, -1, -1).reshape(b * n_ex, *xq.shape[1:])
                xe_rep = xe_all.unsqueeze(0).expand(b, -1, -1, -1).reshape(b * n_ex, *xe_all.shape[1:])
                diff = model(xq_rep, xe_rep).cpu().numpy().reshape(b, n_ex, -1) * diff_sd + diff_mu
                pred = ex_scores[None, :, :] + diff
                out[sub] = pred.mean(axis=1)
    return from_residual(out) if residual_targets else out


def run(epochs: int, limit: int | None, device: str, batch_size: int = 64, patience: int = 8,
        lr: float = 1e-3, hidden: int = 128, weight_decay: float = 1e-4, seed: int = 0,
        tag: str = "", residual_targets: bool = False, select_metric: str = "spearman",
        exemplars_path: Path = EXEMPLARS) -> dict:
    torch.manual_seed(seed)
    np.random.seed(seed)
    d, exemplars = load_data(limit, exemplars_path)
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
    xf = to_residual if residual_targets else (lambda y: y)
    diffs = []
    for r in tr.itertuples():
        qy = xf(np.array([getattr(r, c) for c in score_cols], np.float32))
        for e in ex_by_shot_df[r.shot].itertuples():
            ey = xf(np.array([getattr(e, c) for c in score_cols], np.float32))
            diffs.append(qy - ey)
    diffs = np.stack(diffs)
    diff_mu, diff_sd = diffs.mean(0).astype(np.float32), diffs.std(0).astype(np.float32)
    diff_sd[diff_sd < 1e-3] = 1.0
    print(f"diff distribution (train pairs, residual_targets={residual_targets}): "
          f"mu={diff_mu.round(2)} sd={diff_sd.round(2)}")

    model = STGCNScorer(len(TARGETS), hidden=hidden).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

    dl_tr = DataLoader(PairDataset(tr, exemplars, cache, diff_mu, diff_sd, residual_targets),
                        batch_size=batch_size, shuffle=True)

    def select_score(r: dict) -> float:
        if select_metric == "combined":
            return r["mean_spearman"] + r["mean_partial_spearman_vs_overall"]
        return r["mean_spearman"]

    best_rho, bad = -1.0, 0
    best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}  # untrained fallback
    for ep in range(epochs):
        model.train(); t0 = time.time(); tot_loss = 0.0
        for xq, xe, diff, _ in dl_tr:
            xq, xe, diff = xq.to(device), xe.to(device), diff.to(device)
            opt.zero_grad()
            pred = model(xq, xe)
            loss = nn.functional.mse_loss(pred, diff)
            loss.backward(); opt.step()
            tot_loss += loss.item() * len(diff)
        va_pred = predict_scores(model, va.reset_index(drop=True), exemplars, cache, diff_mu, diff_sd, device, residual_targets=residual_targets)
        va_df = va.reset_index(drop=True).assign(**{f"pred_{p}": va_pred[:, i] for i, p in enumerate(TARGETS)})
        r = evaluate_rating(va_df, {p: f"score_{p}" for p in PARTS}, {p: f"pred_{p}" for p in PARTS},
                             overall_cols=("score_overall", "pred_overall"))
        score = select_score(r)
        print(f"  epoch {ep+1}/{epochs}  loss={tot_loss/len(dl_tr.dataset):.3f}  val_rho={r['mean_spearman']:.3f}  "
              f"val_partial_rho={r['mean_partial_spearman_vs_overall']:.3f}  select={score:.3f}  ({time.time()-t0:.1f}s)")
        if np.isfinite(score) and score > best_rho:
            best_rho, best_state, bad = score, {k: v.cpu().clone() for k, v in model.state_dict().items()}, 0
        else:
            bad += 1
            if bad >= patience:
                print(f"  early stop (no val {select_metric} improvement in {patience} epochs)"); break

    model.load_state_dict(best_state)
    te_r = te.reset_index(drop=True)
    te_pred = predict_scores(model, te_r, exemplars, cache, diff_mu, diff_sd, device, residual_targets=residual_targets)
    te_df = te_r.assign(**{f"pred_{p}": te_pred[:, i] for i, p in enumerate(TARGETS)})
    result = evaluate_rating(te_df, {p: f"score_{p}" for p in PARTS}, {p: f"pred_{p}" for p in PARTS},
                              by="source", overall_cols=("score_overall", "pred_overall"))
    result_h = evaluate_rating(te_df.dropna(subset=["handedness"]), {p: f"score_{p}" for p in PARTS},
                                {p: f"pred_{p}" for p in PARTS}, by="handedness",
                                overall_cols=("score_overall", "pred_overall")) if te_df.handedness.notna().any() else None
    overall_spearman = float(pd.Series(te_df.score_overall).rank().corr(pd.Series(te_df.pred_overall).rank(), method="pearson"))

    out_dir = OUT_DIR if not tag else OUT_DIR.parent / f"{OUT_DIR.name}_{tag}"
    out_dir.mkdir(parents=True, exist_ok=True)
    torch.save(best_state, out_dir / "stgcn_scorer.pt")
    print(format_report(result, "\nTEST"))
    if result_h:
        print(format_report(result_h, "TEST by handedness"))
    print(f"\noverall-score Spearman (test): {overall_spearman:.3f}  (I3D-AE-LSTM baseline: 0.84)")
    summary = {"tag": tag, "lr": lr, "hidden": hidden, "weight_decay": weight_decay, "seed": seed,
               "patience": patience, "residual_targets": residual_targets, "select_metric": select_metric,
               "best_epoch_val_select_score": best_rho,
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
    ap.add_argument("--residual-targets", action="store_true",
                     help="train parts on (part - overall) instead of absolute part scores")
    ap.add_argument("--select-metric", choices=["spearman", "combined"], default="spearman",
                     help="combined = mean_spearman + partial-rho, for the checkpoint selection/early stop")
    ap.add_argument("--exemplars-path", default=str(EXEMPLARS), help="override kaggle/exemplars.csv")
    args = ap.parse_args()
    run(args.epochs, args.limit, args.device, args.batch_size, patience=args.patience, lr=args.lr,
        hidden=args.hidden, weight_decay=args.weight_decay, seed=args.seed, tag=args.tag,
        residual_targets=args.residual_targets, select_metric=args.select_metric,
        exemplars_path=Path(args.exemplars_path))


if __name__ == "__main__":
    main()
