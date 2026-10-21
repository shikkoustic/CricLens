"""Technique scorer v3b: part-localised pose features with OVERLAPPING joint groups (PROGRESS.md
2026-10-21 follow-up). v3 (train_technique_scorer_partlocal.py) gave each part head strictly disjoint
joints (e.g. "hands" = elbows+wrists only) and came back as a clean negative result, 3-seed confirmed on
Kaggle: worse than v2 on every metric. Two untested explanations were noted, not yet tried: (a) 2-4 joints
alone may be too little signal, (b) `diff_g` (global context) was `.detach()`-ed from part heads, possibly
cutting off more useful context than it isolated noise.

This isolates explanation (a) only, leaving (b) unchanged (still detached) so only one variable moves at a
time. Each part's joint group now includes its biomechanical neighbours along the kinetic chain, instead
of being strictly disjoint:
  head:     nose, eyes, ears, shoulders            (was: nose, eyes, ears)
  shoulder: shoulders, elbows, hips                 (was: shoulders only)
  hands:    shoulders, elbows, wrists               (was: elbows, wrists)
  hips:     shoulders, hips, knees                  (was: hips only)
  feet:     hips, knees, ankles                     (was: knees, ankles)
Still smaller than the full 17-joint body for every part (6 joints vs 17), so it's still "local", just with
a buffer of anatomically adjacent joints for arm-chain / leg-chain context instead of a bare joint cluster.

v2 (train_technique_scorer_contrastive.py --residual-targets) fixed the training *target*: each part
predicts (part - overall) instead of the raw score, which stopped the model from satisfying its loss by
just copying "overall" into all 5 outputs. That moved mean partial-rho-vs-overall from ~0.019 to ~0.137 --
real, but still a thin signal. The *input* side was never changed: one shared GRU reads all 17 joints for
every part, so even with a fixed target, the model has to somehow carve one encoding into 5 part-specific
views after the fact. If there's no joint-level evidence specific to "hands" separate from "feet" reaching
the part heads, no amount of target engineering can manufacture that signal from nothing.

Before building this, checked whether the ceiling itself might be low (i.e. whether CricketVision's labels
even contain independent per-part signal to find): the dataset's apparent "double-annotated" clips
(docs/DATASHEET.md) are copies of one rating (Spearman 0.98 between the two folders), not independent
re-scores, so they can't serve as an inter-rater ceiling -- that check is not available with this data.
Proceeding on the input-side lever instead, since it's testable without new annotation.

Change: the shared global GRU (unchanged) still drives the overall-score head. Each of the 5 part heads
additionally gets its own small Siamese GRU reading *only* that part's own COCO joint columns (e.g. the
hands head only ever sees wrist+elbow columns), giving it real, separate visual evidence instead of a
slice of one shared full-body encoding. Always trains on residual targets (the v2 fix stays) -- this
isolates whether part-local input helps on top of it, not instead of it.

    python models/train_technique_scorer_partlocal_overlap.py --epochs 60
    python models/train_technique_scorer_partlocal_overlap.py --epochs 2 --limit 200   # fast smoke test
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
from models.train_technique_scorer import PARTS, TARGETS  # noqa: E402
from models.train_technique_scorer_contrastive import (  # noqa: E402
    EXEMPLARS, PairDataset, cache_sequences, load_data, predict_scores, to_residual)

# COCO-17 joint order (same indexing as models/train_shot_classifier.py's HIP_L/HIP_R/SHO_L/SHO_R):
# 0 nose, 1 l_eye, 2 r_eye, 3 l_ear, 4 r_ear, 5 l_shoulder, 6 r_shoulder, 7 l_elbow, 8 r_elbow,
# 9 l_wrist, 10 r_wrist, 11 l_hip, 12 r_hip, 13 l_knee, 14 r_knee, 15 l_ankle, 16 r_ankle
JOINT_GROUPS = {
    "head": [0, 1, 2, 3, 4, 5, 6],
    "shoulder": [5, 6, 7, 8, 11, 12],
    "hands": [5, 6, 7, 8, 9, 10],
    "hips": [5, 6, 11, 12, 13, 14],
    "feet": [11, 12, 13, 14, 15, 16],
}
# flattened (T, 51) sequence columns for each part's joints (3 cols per joint: x, y, confidence)
PART_COLS = {p: [c for j in js for c in (3 * j, 3 * j + 1, 3 * j + 2)] for p, js in JOINT_GROUPS.items()}
OUT_DIR = ROOT / "models/technique_scorer_partlocal_overlap"


class PartLocalScorer(nn.Module):
    """Shared global GRU (all 51 dims) drives the overall-score head, same as v2's ContrastiveScorer.
    Each part additionally gets its own small Siamese GRU reading only that part's joint columns, so the
    part head has real part-specific input rather than a slice of one shared full-body encoding. diff_g
    (the global query-vs-exemplar difference) is passed to part heads as context only (detached), so
    part-head gradients refine the local encoders, not the shared global one."""

    def __init__(self, in_dim: int, hidden: int = 128, local_hidden: int = 48):
        super().__init__()
        self.global_encoder = nn.GRU(in_dim, hidden, num_layers=2, batch_first=True, dropout=0.2)
        self.overall_head = nn.Sequential(nn.LayerNorm(hidden * 3), nn.Linear(hidden * 3, hidden), nn.ReLU(),
                                           nn.Dropout(0.2), nn.Linear(hidden, 1))
        self.local_encoders = nn.ModuleDict({
            p: nn.GRU(len(cols), local_hidden, num_layers=1, batch_first=True)
            for p, cols in PART_COLS.items()})
        local_in = local_hidden * 3 + hidden
        self.part_heads = nn.ModuleDict({
            p: nn.Sequential(nn.LayerNorm(local_in), nn.Linear(local_in, local_hidden), nn.ReLU(),
                              nn.Dropout(0.2), nn.Linear(local_hidden, 1))
            for p in PART_COLS})

    def embed_global(self, x):
        _, h = self.global_encoder(x)
        return h[-1]

    def embed_part(self, x, part):
        _, h = self.local_encoders[part](x[:, :, PART_COLS[part]])
        return h[-1]

    def forward(self, xq, xe):
        zq_g, ze_g = self.embed_global(xq), self.embed_global(xe)
        diff_g = zq_g - ze_g
        overall_out = self.overall_head(torch.cat([zq_g, ze_g, diff_g], dim=-1))
        part_outs = []
        for p in PARTS:
            zq_p, ze_p = self.embed_part(xq, p), self.embed_part(xe, p)
            feat = torch.cat([zq_p, ze_p, zq_p - ze_p, diff_g.detach()], dim=-1)
            part_outs.append(self.part_heads[p](feat))
        return torch.cat(part_outs + [overall_out], dim=-1)  # order matches TARGETS = PARTS + ["overall"]


def run(epochs: int, limit: int | None, device: str, batch_size: int = 64, patience: int = 8,
        lr: float = 1e-3, hidden: int = 128, local_hidden: int = 48, weight_decay: float = 1e-4,
        seed: int = 0, tag: str = "", select_metric: str = "combined",
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
    diffs = []
    for r in tr.itertuples():
        qy = to_residual(np.array([getattr(r, c) for c in score_cols], np.float32))
        for e in ex_by_shot_df[r.shot].itertuples():
            ey = to_residual(np.array([getattr(e, c) for c in score_cols], np.float32))
            diffs.append(qy - ey)
    diffs = np.stack(diffs)
    diff_mu, diff_sd = diffs.mean(0).astype(np.float32), diffs.std(0).astype(np.float32)
    diff_sd[diff_sd < 1e-3] = 1.0
    print(f"diff distribution (train pairs, residual targets): mu={diff_mu.round(2)} sd={diff_sd.round(2)}")

    in_dim = next(iter(cache.values())).shape[1]
    model = PartLocalScorer(in_dim, hidden=hidden, local_hidden=local_hidden).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

    dl_tr = DataLoader(PairDataset(tr, exemplars, cache, diff_mu, diff_sd, residual_targets=True),
                        batch_size=batch_size, shuffle=True)

    def select_score(r: dict) -> float:
        if select_metric == "combined":
            return r["mean_spearman"] + r["mean_partial_spearman_vs_overall"]
        return r["mean_spearman"]

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
        va_pred = predict_scores(model, va.reset_index(drop=True), exemplars, cache, diff_mu, diff_sd,
                                  device, residual_targets=True)
        va_df = va.reset_index(drop=True).assign(**{f"pred_{p}": va_pred[:, i] for i, p in enumerate(TARGETS)})
        r = evaluate_rating(va_df, {p: f"score_{p}" for p in PARTS}, {p: f"pred_{p}" for p in PARTS},
                             overall_cols=("score_overall", "pred_overall"))
        score = select_score(r)
        print(f"  epoch {ep+1}/{epochs}  loss={tot_loss/len(dl_tr.dataset):.3f}  val_rho={r['mean_spearman']:.3f}  "
              f"val_partial_rho={r['mean_partial_spearman_vs_overall']:.3f}  select={score:.3f}  ({time.time()-t0:.1f}s)")
        if score > best_rho:
            best_rho, best_state, bad = score, {k: v.cpu().clone() for k, v in model.state_dict().items()}, 0
        else:
            bad += 1
            if bad >= patience:
                print(f"  early stop (no val {select_metric} improvement in {patience} epochs)"); break

    model.load_state_dict(best_state)
    te_r = te.reset_index(drop=True)
    te_pred = predict_scores(model, te_r, exemplars, cache, diff_mu, diff_sd, device, residual_targets=True)
    te_df = te_r.assign(**{f"pred_{p}": te_pred[:, i] for i, p in enumerate(TARGETS)})
    result = evaluate_rating(te_df, {p: f"score_{p}" for p in PARTS}, {p: f"pred_{p}" for p in PARTS},
                              by="source", overall_cols=("score_overall", "pred_overall"))
    result_h = evaluate_rating(te_df.dropna(subset=["handedness"]), {p: f"score_{p}" for p in PARTS},
                                {p: f"pred_{p}" for p in PARTS}, by="handedness",
                                overall_cols=("score_overall", "pred_overall")) if te_df.handedness.notna().any() else None
    overall_spearman = float(pd.Series(te_df.score_overall).rank().corr(pd.Series(te_df.pred_overall).rank(), method="pearson"))

    out_dir = OUT_DIR if not tag else OUT_DIR.parent / f"{OUT_DIR.name}_{tag}"
    out_dir.mkdir(parents=True, exist_ok=True)
    torch.save(best_state, out_dir / "partlocal_scorer.pt")
    print(format_report(result, "\nTEST"))
    if result_h:
        print(format_report(result_h, "TEST by handedness"))
    print(f"\noverall-score Spearman (test): {overall_spearman:.3f}  (I3D-AE-LSTM baseline: 0.84)")
    summary = {"tag": tag, "lr": lr, "hidden": hidden, "local_hidden": local_hidden,
               "weight_decay": weight_decay, "seed": seed, "patience": patience,
               "select_metric": select_metric, "best_epoch_val_select_score": best_rho,
               "test_mean_spearman": result["mean_spearman"], "test_mean_r_l2": result["mean_r_l2"],
               "test_mean_partial_spearman_vs_overall": result["mean_partial_spearman_vs_overall"],
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
    ap.add_argument("--local-hidden", type=int, default=48)
    ap.add_argument("--weight-decay", type=float, default=1e-4)
    ap.add_argument("--patience", type=int, default=8)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--tag", default="")
    ap.add_argument("--select-metric", choices=["spearman", "combined"], default="combined")
    ap.add_argument("--exemplars-path", default=str(EXEMPLARS))
    args = ap.parse_args()
    run(args.epochs, args.limit, args.device, args.batch_size, patience=args.patience, lr=args.lr,
        hidden=args.hidden, local_hidden=args.local_hidden, weight_decay=args.weight_decay,
        seed=args.seed, tag=args.tag, select_metric=args.select_metric,
        exemplars_path=Path(args.exemplars_path))


if __name__ == "__main__":
    main()
