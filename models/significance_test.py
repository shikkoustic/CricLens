"""Is the technique-scorer headline result (contrastive regression + residual targets vs. the
plain VAE baseline) real, or within the noise of which 496 test clips happened to land in the
test split?

Every number reported so far (PROGRESS.md, results_phase3.md, contributions.md) answers "how much
does this vary across training seeds" -- a real question, but a different one from "on THIS fixed
test set, is model A actually better than model B, or could the gap be explained by which clips
are in the test set." The standard tool for that second question is a paired bootstrap over the
test set (Efron & Tibshirani 1993; standard practice for comparing two systems' scores on one
fixed test set, e.g. Berg-Kirkpatrick et al. 2012): resample the test clips with replacement many
times, recompute both models' metrics on each resample, and read off how often the gap holds up.

Both models are evaluated on the identical test population: train_ready, scored, shot != "other",
exemplars excluded (the stricter of the two training scripts' own filters) -- the contrastive
model structurally cannot be evaluated on a looser set, so the VAE baseline is restricted to match
for a fair paired comparison, even though its own reported number elsewhere used no such shot/
exemplar exclusion. Numbers may differ slightly from previously reported ones for that reason;
see the printed n and the VAE restricted-set number this script reports.

    python models/significance_test.py                  # 2000 resamples
    python models/significance_test.py --n-boot 5000
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from app.build_reference import contrastive_diff_stats  # noqa: E402
from models.evaluate import evaluate_rating  # noqa: E402
from models.train_shot_classifier import normalise  # noqa: E402
from models.train_technique_scorer import PARTS, TARGETS, VAERegressor, load_scored_sequences  # noqa: E402
from models.train_technique_scorer_contrastive import (  # noqa: E402
    EXEMPLARS, ContrastiveScorer, predict_scores,
)

VAE_CKPT = ROOT / "models/technique_scorer/vae_regressor.pt"
CONTRASTIVE_CKPT = ROOT / "models/technique_scorer_contrastive_residual_seed2/contrastive_scorer.pt"


def build_test_set() -> tuple[pd.DataFrame, pd.DataFrame]:
    """The contrastive model's own population (train_ready, scored, shot != other, exemplars
    excluded), restricted to the test split -- see module docstring for why both models use this
    same set rather than each their own training script's looser default."""
    d, _, _ = load_scored_sequences()
    shot = pd.read_parquet(ROOT / "data/processed/pose_index.parquet", columns=["clip_id", "shot"])
    d = d.merge(shot, on="clip_id")
    exemplars = pd.read_csv(EXEMPLARS)
    d = d[(d.shot != "other") & ~d.clip_id.isin(exemplars.clip_id)]
    te = d[d.split == "test"].reset_index(drop=True)
    return te, exemplars


def vae_predictions(te: pd.DataFrame) -> np.ndarray:
    _, mu_t, sd_t = load_scored_sequences()  # same normalisation the checkpoint was trained with
    model = VAERegressor(51, 32, len(TARGETS))
    model.load_state_dict(torch.load(VAE_CKPT, map_location="cpu"))
    model.eval()
    x = torch.from_numpy(np.stack([normalise(np.load(ROOT / p)["seq"]) for p in te.seq_path]))
    with torch.no_grad():
        _, mu, _, sc = model(x)
        pred = sc.numpy() * sd_t + mu_t
    return pred


def contrastive_predictions(te: pd.DataFrame, exemplars: pd.DataFrame) -> np.ndarray:
    model = ContrastiveScorer(51, len(TARGETS), hidden=128)
    model.load_state_dict(torch.load(CONTRASTIVE_CKPT, map_location="cpu"))
    model.eval()
    stats = contrastive_diff_stats()
    diff_mu, diff_sd = np.array(stats["contrastive_diff_mu"], np.float32), np.array(stats["contrastive_diff_sd"], np.float32)

    all_clips = set(te.clip_id) | set(exemplars.clip_id)
    seq_path = dict(zip(te.clip_id, te.seq_path))
    ex_seq = pd.read_parquet(ROOT / "data/processed/sequences.parquet").set_index("clip_id")
    for cid in exemplars.clip_id:
        if cid not in seq_path:
            seq_path[cid] = ex_seq.loc[cid, "seq_path"]
    cache = {cid: normalise(np.load(ROOT / seq_path[cid])["seq"]) for cid in all_clips}

    return predict_scores(model, te, exemplars, cache, diff_mu, diff_sd, "cpu", residual_targets=True)


def metrics(te: pd.DataFrame, pred: np.ndarray, idx: np.ndarray) -> dict:
    sub = te.iloc[idx].reset_index(drop=True)
    sub_pred = pred[idx]
    d = sub.assign(**{f"pred_{p}": sub_pred[:, i] for i, p in enumerate(TARGETS)})
    r = evaluate_rating(d, {p: f"score_{p}" for p in PARTS}, {p: f"pred_{p}" for p in PARTS},
                         overall_cols=("score_overall", "pred_overall"))
    return {"mean_spearman": r["mean_spearman"], "mean_partial_rho": r["mean_partial_spearman_vs_overall"]}


def bootstrap(te: pd.DataFrame, pred_vae: np.ndarray, pred_con: np.ndarray, n_boot: int, seed: int) -> dict:
    rng = np.random.default_rng(seed)
    n = len(te)
    diffs = {"mean_spearman": [], "mean_partial_rho": []}
    for _ in range(n_boot):
        idx = rng.integers(0, n, n)  # resample WITH replacement, same indices for both models (paired)
        mv, mc = metrics(te, pred_vae, idx), metrics(te, pred_con, idx)
        for k in diffs:
            if np.isfinite(mv[k]) and np.isfinite(mc[k]):
                diffs[k].append(mc[k] - mv[k])
    out = {}
    for k, vals in diffs.items():
        v = np.array(vals)
        out[k] = {"n_valid": len(v), "mean_diff": float(v.mean()),
                   "ci95": [float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))],
                   "p_contrastive_better": float((v > 0).mean()),
                   "p_vae_better": float((v < 0).mean())}
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-boot", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    te, exemplars = build_test_set()
    print(f"test set: n={len(te)} clips (train_ready, scored, shot!=other, exemplars excluded)")

    pred_vae = vae_predictions(te)
    pred_con = contrastive_predictions(te, exemplars)

    m_vae = metrics(te, pred_vae, np.arange(len(te)))
    m_con = metrics(te, pred_con, np.arange(len(te)))
    print(f"\npoint estimates on the full (restricted) test set:")
    print(f"  VAE baseline        mean_spearman={m_vae['mean_spearman']:.3f}  mean_partial_rho={m_vae['mean_partial_rho']:.3f}")
    print(f"  Contrastive+residual mean_spearman={m_con['mean_spearman']:.3f}  mean_partial_rho={m_con['mean_partial_rho']:.3f}")

    print(f"\nrunning {args.n_boot} paired bootstrap resamples...")
    result = bootstrap(te, pred_vae, pred_con, args.n_boot, args.seed)
    for k, r in result.items():
        print(f"\n{k}: contrastive - vae, mean diff = {r['mean_diff']:+.4f}, "
              f"95% CI [{r['ci95'][0]:+.4f}, {r['ci95'][1]:+.4f}]  (n_valid={r['n_valid']}/{args.n_boot})")
        print(f"  P(contrastive better) = {r['p_contrastive_better']:.4f}   P(vae better) = {r['p_vae_better']:.4f}")


if __name__ == "__main__":
    main()
