"""Are CricketVision's five per-body-part scores actually five independent signals?

docs/paper/dossier.md (Part IV, Gap 6) flagged this from Moodley & van der Haar's WACV 2025 results: their
five per-part Spearman correlations against ground truth differ by only 0.003 (0.83844-0.84146). That is
consistent with either (a) a model that genuinely, independently nails all five parts almost equally well,
or (b) a model predicting one overall-quality signal that happens to correlate similarly with all five
labels because the labels themselves are collinear. This script tests which, directly on the labels --
no model needed. If the LABELS are collinear, no amount of modelling can produce independently
discriminative per-part predictions evaluated against them; the field's implicit claim of demonstrated
per-body-part cricket technique assessment does not hold.

Five checks, each answering a different form of the question:
  1. Raw Spearman correlation between the five part scores.
  2. PCA: how much of the joint variance is one latent factor.
  3. Partial correlation between each pair, controlling for the other three parts.
  4. Residual correlation between parts after regressing each on score_overall alone.
  5. R^2: how much of each part's variance score_overall alone explains.

Run:  python models/label_correlation_analysis.py
Output: docs/paper/label_correlation/*.csv (the five tables) + a printed summary.
"""
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "docs/paper/label_correlation"
PARTS = ["head", "shoulder", "hands", "hips", "feet"]


def load_scores() -> tuple[pd.DataFrame, pd.Series]:
    idx = pd.read_parquet(ROOT / "data/processed/pose_index.parquet")
    t = idx[idx.train_ready & idx.score_overall.notna()].copy()
    S = t[[f"score_{p}" for p in PARTS]].rename(columns=lambda c: c.replace("score_", ""))
    return S, t["score_overall"]


def partial_correlations(S: pd.DataFrame) -> pd.DataFrame:
    """Partial Pearson correlation via the precision matrix: corr(i,j | rest) = -P_ij / sqrt(P_ii * P_jj)."""
    R = S.corr(method="pearson").to_numpy()
    P = np.linalg.inv(R)
    D = np.sqrt(np.diag(P))
    partial = -P / np.outer(D, D)
    np.fill_diagonal(partial, 1.0)
    return pd.DataFrame(partial, index=S.columns, columns=S.columns)


def residual_correlations(S: pd.DataFrame, overall: pd.Series) -> pd.DataFrame:
    """Spearman correlation between parts after removing each part's linear fit on score_overall --
    whatever correlation remains is structure the overall score does NOT already explain."""
    x = overall.to_numpy()
    resid = {}
    for p in S.columns:
        y = S[p].to_numpy()
        b = np.polyfit(x, y, 1)
        resid[p] = y - np.polyval(b, x)
    return pd.DataFrame(resid).corr(method="spearman")


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    S, overall = load_scores()
    n = len(S)

    raw = S.corr(method="spearman")
    Z = (S - S.mean()) / S.std()
    eigvals = np.sort(np.linalg.eigvalsh(np.cov(Z.T)))[::-1]
    var_explained = pd.Series(eigvals / eigvals.sum(), index=[f"PC{i}" for i in range(1, 6)])
    partial = partial_correlations(S)
    resid = residual_correlations(S, overall)
    r2_overall = pd.Series({p: np.corrcoef(overall, S[p])[0, 1] ** 2 for p in PARTS}, name="R2_with_overall")

    raw.to_csv(OUT / "1_raw_spearman.csv")
    var_explained.to_csv(OUT / "2_pca_variance_explained.csv", header=["variance_explained"])
    partial.to_csv(OUT / "3_partial_correlation.csv")
    resid.to_csv(OUT / "4_residual_correlation_after_overall.csv")
    r2_overall.to_csv(OUT / "5_r2_with_overall.csv")

    off = lambda m: m.where(~np.eye(len(m), dtype=bool))  # noqa: E731
    print(f"n = {n} scored, train-ready clips\n")
    print("1. Raw Spearman between the five part scores:")
    print(raw.round(3).to_string())
    print(f"   min off-diagonal: {off(raw).min().min():.3f}  max: {off(raw).max().max():.3f}\n")
    print("2. PCA variance explained:")
    print(var_explained.round(4).to_string())
    print(f"   PC1 alone: {var_explained.iloc[0]:.1%}\n")
    print("3. Partial correlation (each pair, controlling for the other three):")
    print(partial.round(3).to_string())
    print(f"   mean |off-diagonal|: {off(partial).abs().mean().mean():.3f}\n")
    print("4. Residual correlation after regressing out score_overall:")
    print(resid.round(3).to_string())
    print(f"   range: [{off(resid).min().min():.3f}, {off(resid).max().max():.3f}]\n")
    print("5. R^2 of each part with score_overall alone:")
    print(r2_overall.round(3).to_string())
    print(f"\nwritten to {OUT}")


if __name__ == "__main__":
    main()
