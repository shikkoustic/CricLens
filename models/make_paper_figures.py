"""Generate the manuscript's result figures from local data and local checkpoints.

Figures produced (600 dpi PNG, white background, no grid -- JEI artwork spec):
  fig3_label_collinearity.png  5x5 Spearman matrix of the five body-part LABEL scores
  fig4_part_spread.png         per-clip spread of the five PREDICTED part scores,
                               absolute-regression baseline vs residual-target model

    python models/make_paper_figures.py
"""
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from models.significance_test import (  # noqa: E402
    build_test_set, contrastive_predictions, vae_predictions,
)
from models.train_technique_scorer import PARTS, TARGETS  # noqa: E402

OUT = ROOT / "docs/paper/figures"
OUT.mkdir(parents=True, exist_ok=True)

plt.rcParams.update({
    "font.family": "serif", "font.serif": ["Times New Roman", "DejaVu Serif"],
    "font.size": 9, "axes.grid": False, "figure.facecolor": "white",
    "axes.facecolor": "white", "savefig.facecolor": "white",
})
LABEL = {"head": "Head", "shoulder": "Shoulders", "hands": "Hands", "hips": "Hips", "feet": "Feet"}


def fig3_label_collinearity():
    idx = pd.read_parquet(ROOT / "data/processed/pose_index.parquet")
    # match the paper's population exactly: shot != "other" is excluded there too
    scored = idx[idx.train_ready & idx.score_overall.notna() & (idx.shot != "other")]
    cols = [f"score_{p}" for p in PARTS]
    corr = scored[cols].corr(method="spearman").to_numpy()

    fig, ax = plt.subplots(figsize=(3.3, 2.9))
    im = ax.imshow(corr, cmap="viridis", vmin=0.90, vmax=1.0)
    names = [LABEL[p] for p in PARTS]
    ax.set_xticks(range(5), names, rotation=45, ha="right")
    ax.set_yticks(range(5), names)
    for i in range(5):
        for j in range(5):
            ax.text(j, i, f"{corr[i, j]:.2f}", ha="center", va="center",
                    color="white" if corr[i, j] < 0.975 else "black", fontsize=8)
    cb = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cb.set_label("Spearman correlation", fontsize=8)
    cb.ax.tick_params(labelsize=8)
    ax.set_title(f"n = {len(scored):,} scored clips", fontsize=9)
    fig.tight_layout()
    p = OUT / "fig3_label_collinearity.png"
    fig.savefig(p, dpi=600, bbox_inches="tight")
    plt.close(fig)
    off = corr[~np.eye(5, dtype=bool)]
    print(f"fig3 -> {p}  (off-diagonal range {off.min():.3f}-{off.max():.3f})")


def _predictions():
    te, exemplars = build_test_set()
    print(f"  test clips: {len(te)}")
    return te, vae_predictions(te), contrastive_predictions(te, exemplars)


def fig4_partial_rho(te, pv, pc):
    """Per-part partial Spearman vs overall -- the metric the paper's claim rests on."""
    from models.evaluate import partial_spearman_vs_overall
    rows = {}
    for name, pred in (("Absolute-regression baseline", pv), ("Residual targets", pc)):
        d = te.assign(**{f"pred_{t}": pred[:, i] for i, t in enumerate(TARGETS)})
        rows[name] = [partial_spearman_vs_overall(
            d, f"score_{p}", f"pred_{p}", "score_overall", "pred_overall") for p in PARTS]

    fig, ax = plt.subplots(figsize=(3.4, 2.6))
    x = np.arange(len(PARTS)); w = 0.38
    ax.bar(x - w / 2, rows["Absolute-regression baseline"], w, color="0.45",
           label="Absolute-regression baseline")
    ax.bar(x + w / 2, rows["Residual targets"], w, color="#1f77b4",
           label="Residual targets (this work)")
    ax.axhline(0, color="black", lw=0.8)
    ax.set_xticks(x, [LABEL[p] for p in PARTS], fontsize=8)
    ax.set_ylabel("Partial Spearman vs. overall", fontsize=8)
    ax.tick_params(labelsize=8)
    ax.set_ylim(top=max(rows["Residual targets"]) * 1.55)
    ax.legend(fontsize=7, loc="upper left", frameon=True, facecolor="white",
              edgecolor="none", framealpha=1.0)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    fig.tight_layout()
    p = OUT / "fig4_partial_rho_per_part.png"
    fig.savefig(p, dpi=600, bbox_inches="tight")
    plt.close(fig)
    print(f"fig4 -> {p}")
    for k, v in rows.items():
        print(f"  {k}: " + ", ".join(f"{p}={x:.3f}" for p, x in zip(PARTS, v)))


def fig5_bootstrap(te, pv, pc, n_boot=2000, seed=0):
    """Paired-bootstrap distribution of the partial-rho gain (the significance result)."""
    from models.significance_test import metrics
    rng = np.random.default_rng(seed)
    n = len(te)
    diffs = []
    for _ in range(n_boot):
        idx = rng.integers(0, n, n)
        a, b = metrics(te, pv, idx), metrics(te, pc, idx)
        if np.isfinite(a["mean_partial_rho"]) and np.isfinite(b["mean_partial_rho"]):
            diffs.append(b["mean_partial_rho"] - a["mean_partial_rho"])
    diffs = np.array(diffs)
    lo, hi = np.percentile(diffs, [2.5, 97.5])

    fig, ax = plt.subplots(figsize=(3.4, 2.5))
    counts, _, _ = ax.hist(diffs, bins=45, color="#1f77b4", alpha=0.85)
    # cap the marker lines at the data height so the legend sits on clean white
    top = counts.max() * 1.04
    ax.vlines(0, 0, top, color="#d62728", ls="--", lw=1.2, label="No difference")
    ax.vlines([lo, hi], 0, top, color="black", ls=":", lw=1.0,
              label=f"95% CI [{lo:+.3f}, {hi:+.3f}]")
    ax.set_xlabel("Gain in partial Spearman (residual $-$ baseline)", fontsize=8)
    ax.set_ylabel(f"Bootstrap resamples (n = {len(diffs):,})", fontsize=8)
    ax.tick_params(labelsize=8)
    ax.set_ylim(0, top * 1.38)
    ax.legend(fontsize=7, loc="upper left", frameon=True, facecolor="white",
              edgecolor="none", framealpha=1.0)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    fig.tight_layout()
    p = OUT / "fig5_bootstrap_gain.png"
    fig.savefig(p, dpi=600, bbox_inches="tight")
    plt.close(fig)
    print(f"fig5 -> {p}")
    print(f"  mean gain {diffs.mean():+.4f}, 95% CI [{lo:+.4f}, {hi:+.4f}], "
          f"P(residual better) = {(diffs > 0).mean():.4f}")


if __name__ == "__main__":
    fig3_label_collinearity()
    te, pv, pc = _predictions()
    fig4_partial_rho(te, pv, pc)
    fig5_bootstrap(te, pv, pc)
