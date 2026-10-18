"""Shared evaluation: the one place every model's numbers come from.

Built before any model exists, on purpose -- so the shot classifier, the technique scorer, and every
later variant of each are scored the same way, on the same splits, with the same metric definitions,
and their numbers are directly comparable to each other and to the literature (docs/paper/dossier.md).

Two tasks:
  CLASSIFY  -- shot label. accuracy, macro/weighted F1, per-class report, confusion matrix.
  RATE      -- per-body-part technique score. Spearman rank correlation (the AQA-standard metric,
               Fisher-z averaged when combining several parts/groups, matching USDL/CoRe/TSA/FineParser)
               and R-l2 (relative L2 distance, introduced by CoRe, arXiv:2108.07797 eq. for R-l2), so our
               numbers slot directly into the comparison tables in docs/paper/dossier.md.

Every report function accepts an optional `by` column (source, handedness, shot, ...) and returns a
per-group breakdown alongside the overall number -- this is how the cross-source and handedness-bias
questions (docs/paper/aligned_papers.md, scope_report.md) get answered once a model exists.

    from models.evaluate import evaluate_classification, evaluate_rating, check_split_integrity
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import classification_report, confusion_matrix

# CricketVision's annotated score scale (docs/DATASHEET.md; observed range in pose_index.parquet is
# 1.0-10.0, not the 0-9 stated in Moodley & van der Haar WACV 2025 -- see docs/paper/dossier.md A1).
SCORE_MIN, SCORE_MAX = 1.0, 10.0
SCORE_PARTS = ["head", "shoulder", "hands", "hips", "feet"]


def check_split_integrity(df: pd.DataFrame, split_col: str = "split", group_col: str = "group") -> dict:
    """Verify no match/dedupe-cluster group spans two splits. Run this before trusting any number."""
    g = df.groupby(group_col)[split_col].nunique()
    bad = g[g > 1]
    return {"ok": bad.empty, "groups_checked": int(g.shape[0]), "leaking_groups": bad.index.tolist()}


def spearman(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Spearman rank correlation. NaN-safe: pairs where either side is NaN are dropped."""
    a, b = np.asarray(y_true, float), np.asarray(y_pred, float)
    ok = np.isfinite(a) & np.isfinite(b)
    if ok.sum() < 2:
        return float("nan")
    ra, rb = pd.Series(a[ok]).rank(), pd.Series(b[ok]).rank()
    return float(np.corrcoef(ra, rb)[0, 1])


def fisher_mean(rhos: list[float]) -> float:
    """Average several Spearman correlations via the Fisher z-transform (USDL/CoRe convention), so
    combining e.g. per-source or per-part rhos doesn't just average the raw correlations."""
    r = np.array([x for x in rhos if np.isfinite(x)], float)
    if len(r) == 0:
        return float("nan")
    r = np.clip(r, -0.9999, 0.9999)
    z = np.arctanh(r)
    return float(np.tanh(z.mean()))


def relative_l2(y_true: np.ndarray, y_pred: np.ndarray, smin: float = SCORE_MIN, smax: float = SCORE_MAX) -> float:
    """R-l2 (CoRe, arXiv:2108.07797): mean squared error normalised by the score range, x100 to match
    how the literature reports it (docs/paper/dossier.md tables are all in this scale)."""
    a, b = np.asarray(y_true, float), np.asarray(y_pred, float)
    ok = np.isfinite(a) & np.isfinite(b)
    if not ok.any():
        return float("nan")
    return float(np.mean(((a[ok] - b[ok]) / (smax - smin)) ** 2) * 100)


def evaluate_classification(df: pd.DataFrame, y_true: str, y_pred: str, by: str | None = None) -> dict:
    """df must have columns y_true, y_pred (and `by` if given). Returns overall metrics, the sklearn
    per-class report, the confusion matrix (as a labelled DataFrame), and a per-`by` accuracy/F1 table."""
    d = df.dropna(subset=[y_true, y_pred])
    labels = sorted(set(d[y_true]) | set(d[y_pred]))
    report = classification_report(d[y_true], d[y_pred], labels=labels, output_dict=True, zero_division=0)
    cm = pd.DataFrame(confusion_matrix(d[y_true], d[y_pred], labels=labels), index=labels, columns=labels)
    out = {"n": len(d), "accuracy": float(report["accuracy"]), "macro_f1": float(report["macro avg"]["f1-score"]),
           "weighted_f1": float(report["weighted avg"]["f1-score"]), "per_class": report, "confusion_matrix": cm}
    if by:
        rows = []
        for g, gd in d.groupby(by):
            r = classification_report(gd[y_true], gd[y_pred], labels=labels, output_dict=True, zero_division=0)
            rows.append({by: g, "n": len(gd), "accuracy": r["accuracy"], "macro_f1": r["macro avg"]["f1-score"]})
        out["by_" + by] = pd.DataFrame(rows).sort_values("n", ascending=False)
    return out


def partial_spearman_vs_overall(df: pd.DataFrame, true_col: str, pred_col: str,
                                 true_overall_col: str, pred_overall_col: str) -> float:
    """Whether a part prediction carries anything beyond the shared overall-quality factor
    (docs/paper/finding_label_collinearity.md: the 5 label scores correlate at 0.95-0.99, so raw
    per-part Spearman cannot show independent per-part discrimination on its own).

    Residualise the true part score against the true overall score, and the predicted part score
    against the predicted overall score (same construction as check 4 in
    models/label_correlation_analysis.py, applied across model predictions instead of within labels),
    then correlate the two residuals. Nonzero means the model captures part-specific signal that
    isn't explained by "how good was the shot overall"; near-zero means it's just fitting that
    shared factor, however good the raw per-part Spearman looks."""
    d = df[[true_col, pred_col, true_overall_col, pred_overall_col]].dropna()
    if len(d) < 3:
        return float("nan")

    def resid(y, x):
        b = np.polyfit(x, y, 1)
        return y - np.polyval(b, x)

    rt = resid(d[true_col].to_numpy(float), d[true_overall_col].to_numpy(float))
    rp = resid(d[pred_col].to_numpy(float), d[pred_overall_col].to_numpy(float))
    return spearman(rt, rp)


def evaluate_rating(df: pd.DataFrame, true_cols: dict[str, str], pred_cols: dict[str, str],
                     by: str | None = None, overall_cols: tuple[str, str] | None = None) -> dict:
    """true_cols/pred_cols: {part_name: column_name}, e.g. {"head": "score_head", ...}. Returns per-part
    Spearman + R-l2, the Fisher-z mean across parts, and (if `by`) the same breakdown per group -- this
    is the table that checks whether per-part scores are actually discriminative (docs/paper/dossier.md,
    Part IV, Gap 6): if every part's Spearman and R-l2 are near-identical, the parts likely aren't.

    overall_cols: (true_overall_col, pred_overall_col), if given, adds partial_spearman_vs_overall per
    part -- required reading alongside raw Spearman, per finding_label_collinearity.md."""
    parts = list(true_cols)
    per_part = {p: {"n": int(df[[true_cols[p], pred_cols[p]]].dropna().shape[0]),
                     "spearman": spearman(df[true_cols[p]], df[pred_cols[p]]),
                     "r_l2": relative_l2(df[true_cols[p]], df[pred_cols[p]])} for p in parts}
    if overall_cols:
        to, po = overall_cols
        for p in parts:
            per_part[p]["partial_spearman_vs_overall"] = partial_spearman_vs_overall(df, true_cols[p], pred_cols[p], to, po)
    out = {"per_part": per_part, "mean_spearman": fisher_mean([per_part[p]["spearman"] for p in parts]),
           "mean_r_l2": float(np.mean([per_part[p]["r_l2"] for p in parts if np.isfinite(per_part[p]["r_l2"])]))}
    if overall_cols:
        out["mean_partial_spearman_vs_overall"] = fisher_mean([per_part[p]["partial_spearman_vs_overall"] for p in parts])
    if by:
        rows = []
        for g, gd in df.groupby(by):
            rhos = [spearman(gd[true_cols[p]], gd[pred_cols[p]]) for p in parts]
            rows.append({by: g, "n": len(gd), "mean_spearman": fisher_mean(rhos)})
        out["by_" + by] = pd.DataFrame(rows).sort_values("n", ascending=False)
    return out


def part_correlation_matrix(df: pd.DataFrame, cols: dict[str, str] = None) -> pd.DataFrame:
    """Correlation between the five CricketVision label scores themselves (not predictions) -- the
    check from docs/paper/dossier.md Part IV / Gap 6: are the five body parts discriminative labels,
    or near-collinear copies of one overall score?"""
    cols = cols or {p: f"score_{p}" for p in SCORE_PARTS}
    return df[list(cols.values())].rename(columns={v: k for k, v in cols.items()}).corr(method="spearman")


def format_report(d: dict, title: str = "") -> str:
    """Plain-text summary for a classification or rating result dict, for quick eyeballing / logging."""
    lines = [title] if title else []
    if "accuracy" in d:
        lines.append(f"n={d['n']}  accuracy={d['accuracy']:.3f}  macro-F1={d['macro_f1']:.3f}  weighted-F1={d['weighted_f1']:.3f}")
    if "mean_spearman" in d:
        extra = f"  mean partial-rho(vs overall)={d['mean_partial_spearman_vs_overall']:.3f}" if "mean_partial_spearman_vs_overall" in d else ""
        lines.append(f"mean Spearman={d['mean_spearman']:.3f}  mean R-l2={d['mean_r_l2']:.2f}{extra}")
        for p, m in d["per_part"].items():
            pv = f"  partial={m['partial_spearman_vs_overall']:.3f}" if "partial_spearman_vs_overall" in m else ""
            lines.append(f"  {p:10s} n={m['n']:5d}  rho={m['spearman']:.3f}  R-l2={m['r_l2']:.2f}{pv}")
    for k, v in d.items():
        if k.startswith("by_") and isinstance(v, pd.DataFrame):
            lines.append(f"-- {k} --\n{v.to_string(index=False)}")
    return "\n".join(lines)
