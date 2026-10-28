"""Technique scorer v3: swap the pose-GRU encoder for the pretrained Timesformer video encoder --
Kaggle GPU run (self-contained, no repo imports).

Motivation (PROGRESS.md 2026-10-28): the supervisor's architectural-change-to-the-GRU ask was aimed at
this model (models/train_technique_scorer_contrastive.py's `ContrastiveScorer`, a 2-layer GRU on pose
joints), not the shot classifier. A literature check on the shot-classifier side found that our
"contact_residual" adapter idea is mechanically AdaptFormer (Chen et al. 2022, arXiv:2205.13535) --
not a novel mechanism -- so this script does NOT introduce a new gate/adapter. It tests something more
defensible instead: real video + pretrained Timesformer beat every pose-based sequence model for shot
classification (0.797-0.811 vs 0.766-0.785 macro-F1); this is the direct, honest test of whether that
same finding transfers to technique SCORING, by replacing the GRU encoder with the same pretrained
Timesformer backbone, keeping the rest of the proven contrastive/diff-vs-exemplar design unchanged.

Architecture: `TimesformerContrastiveScorer` embeds a clip as the CLS token of
`TimesformerModel(pixel_values).last_hidden_state[:, 0]` (verified against the installed transformers
source: this is exactly what TimesformerForVideoClassification's own head reads), projects it down to
`hidden` (128, matching the old GRU's hidden size for a fair comparison), then reuses the exact same
diff-head design as the GRU version: LayerNorm(hidden*3) -> Linear -> ReLU -> Dropout -> Linear, reading
[z_query, z_exemplar, z_query - z_exemplar]. Same residual-targets option, same exemplar-averaging
prediction at eval time (kaggle/exemplars.csv, 50 clips/10 per shot, never used as queries), same
evaluation metrics (models/evaluate.py's spearman/relative_l2/partial_spearman_vs_overall, copied here
verbatim so every model's numbers stay comparable).

COMPUTE REDESIGN, stated honestly (not hidden): the original GRU script builds ALL (query, exemplar)
pairs for every query (~10 pairs/query, ~37k pairs/epoch) and backprops through BOTH the query's and
the exemplar's encoder forward pass every pair -- cheap for a GRU on 51-dim pose vectors, but a
Timesformer forward pass is ~100x more expensive, making that exact design infeasible here (back-of-
envelope: ~8+ hours/epoch). Two changes, both well-precedented in contrastive-learning literature
(SimSiam/MoCo-style stop-gradient on a reference branch), not novel architecture claims:
  1. Exemplar embeddings are computed ONCE per epoch with no_grad() (50 clips, cheap) and reused for
     every query pairing that epoch, instead of re-embedding the same 50 exemplars inside every pair's
     forward pass (eliminates ~74x redundant exemplar computation/epoch).
  2. Each query is paired with ONE randomly-sampled same-shot exemplar per epoch (resampled every
     epoch) for the TRAINING loss, instead of all ~10 -- over several epochs a query still sees a
     good spread of exemplars, just not all-at-once. Validation/test prediction is unchanged: still
     averages over ALL same-shot exemplars (the CoRe-style robustness step), since that only runs once
     per epoch, not once per training step.
With these changes, per-epoch cost is roughly comparable to the shot classifier's (~3,712 train queries
vs its ~11,133 clips), not 10x worse.

Inputs : shikkoustic/criclens-all (criclens-processed.tgz.bin -> data/processed/pose_index.parquet;
         kaggle/chunks/pose-cv-c*/out/kps/<clip>.npz for window/box data; clips/cricketvision/<clip_id>.mp4
         read directly from the dataset's unpacked clips/ folder; kaggle/exemplars.csv for the 50
         reference clips -- all scored clips are CricketVision only, confirmed before using it)
Outputs: timesformer_scorer.pt, summary.json
"""
import glob
import io
import json
import os
import subprocess
import sys
import time

if not os.environ.get("CRICLENS_LOCAL"):
    subprocess.run([sys.executable, "-m", "pip", "install", "-q", "-U", "transformers>=4.48"], check=False)

import cv2
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from transformers import TimesformerModel

OUT = os.environ.get("CRICLENS_OUT", "/kaggle/working")
LOCAL = os.environ.get("CRICLENS_LOCAL")
LOCAL_CLIPS = os.environ.get("CRICLENS_LOCAL_CLIPS")  # dir containing <source>/<clip_id>.mp4, for smoke tests
_limit_env = os.environ.get("CRICLENS_LIMIT")
LIMIT = int(_limit_env) if _limit_env else None
os.makedirs(OUT, exist_ok=True)
if LOCAL:
    PROC = LOCAL
else:
    PROC = "/tmp/proc"
    os.makedirs(PROC, exist_ok=True)
    _tgz = glob.glob("/kaggle/input/**/criclens-processed.tgz.bin", recursive=True)[0]
    subprocess.run(["tar", "-xzf", _tgz, "-C", PROC, "--wildcards",
                     "data/processed/*", "kaggle/chunks/pose-cv-*/out/kps/*"], check=True)

# kaggle/exemplars.csv (50 clips, checked into git) is NOT part of criclens-processed.tgz.bin -- it's a
# tiny repo file the GRU scorer reads directly off disk, which doesn't exist on Kaggle (no repo checkout
# there). Embedded verbatim here instead of depending on a dataset re-upload, consistent with this
# project's "self-contained Kaggle script" convention. CRICLENS_EXEMPLARS overrides with a local path,
# for smoke tests against a custom small exemplar set.
EXEMPLARS_CSV = """shot,tier,clip_id,source,score_overall,score_head,score_shoulder,score_hands,score_hips,score_feet,quality,handedness
cut,high,cv_P5_V32_0041,cricketvision,7.2666666666666675,7.666666666666667,6.666666666666667,7.666666666666667,6.666666666666667,7.666666666666667,3.543698722756358,Left
cut,high,cv_P6_V8_0021,cricketvision,7.533333333333333,8.0,7.0,8.0,7.0,7.666666666666667,6.455930936569641,Right
cut,low,cv_P1_V11_0011,cricketvision,4.2,4.666666666666667,4.0,4.0,4.0,4.333333333333333,2.969988376646481,Right
cut,low,cv_P2_V3_0034,cricketvision,5.4,6.0,5.0,6.0,5.0,5.0,4.404193007227944,Right
cut,mid,cv_P2_V3_0040,cricketvision,5.8,6.333333333333333,5.666666666666667,6.0,5.333333333333333,5.666666666666667,3.562923515264085,Right
cut,mid,cv_P1_V12_0032,cricketvision,6.8,7.666666666666667,6.666666666666667,7.0,6.333333333333333,6.333333333333333,5.293145263522387,Right
cut,very_high,cv_P1_V7_0138,cricketvision,8.866666666666665,9.333333333333334,8.666666666666666,8.666666666666666,9.0,8.666666666666666,2.423095957556753,Right
cut,very_high,cv_P7_V10_0016,cricketvision,8.866666666666667,9.333333333333334,9.0,9.333333333333334,8.0,8.666666666666666,2.513005544316261,Left
cut,very_low,cv_P3_V14_0012,cricketvision,3.1333333333333333,3.6666666666666665,3.0,2.6666666666666665,3.0,3.333333333333333,3.542769142175746,Left
cut,very_low,cv_P5_V22_0024,cricketvision,3.866666666666667,4.666666666666667,4.666666666666667,4.333333333333333,2.6666666666666665,3.0,5.381930928351242,Right
defence,high,cv_P7_V20_0020,cricketvision,7.333333333333333,8.0,7.333333333333333,7.333333333333333,6.666666666666667,7.333333333333333,3.190489694437717,Left
defence,high,cv_P5_V39_0039,cricketvision,7.466666666666667,8.0,7.0,7.666666666666667,7.333333333333333,7.333333333333333,2.649573237533127,Right
defence,low,cv_P5_V12_0010,cricketvision,4.2,4.666666666666667,4.333333333333333,4.0,4.0,4.0,3.8846300242753062,Right
defence,low,cv_P1_V24_0040,cricketvision,5.133333333333334,5.666666666666667,4.666666666666667,5.333333333333333,5.0,5.0,9.279800902226263,Right
defence,mid,cv_P2_V19_0000,cricketvision,6.266666666666668,6.666666666666667,5.666666666666667,6.666666666666667,5.666666666666667,6.666666666666667,2.9053297799248226,Right
defence,mid,cv_P1_V30_0041,cricketvision,6.7333333333333325,7.333333333333333,6.333333333333333,6.666666666666667,7.0,6.333333333333333,3.641012133481351,Right
defence,very_high,cv_P1_V22_0022,cricketvision,8.933333333333334,9.333333333333334,8.666666666666666,9.0,8.666666666666666,9.0,2.4796680672190075,Left
defence,very_high,cv_P1_V22_0046,cricketvision,9.066666666666666,9.333333333333334,8.666666666666666,9.0,9.0,9.333333333333334,2.636328128823385,Right
defence,very_low,cv_P9_V1_0031,cricketvision,3.733333333333333,4.333333333333333,4.0,3.6666666666666665,3.333333333333333,3.333333333333333,5.147522876301836,Left
defence,very_low,cv_P6_V25_0017,cricketvision,3.733333333333333,4.333333333333333,3.6666666666666665,3.333333333333333,3.6666666666666665,3.6666666666666665,4.823433380597369,Right
drive,high,cv_P5_V39_0014,cricketvision,7.2666666666666675,7.666666666666667,7.0,7.0,7.333333333333333,7.333333333333333,4.259273599589278,Left
drive,high,cv_P6_V11_0030,cricketvision,7.466666666666666,8.0,7.333333333333333,7.333333333333333,7.0,7.666666666666667,5.247725771808702,Left
drive,low,cv_P5_V32_0010,cricketvision,4.266666666666668,4.666666666666667,3.6666666666666665,4.333333333333333,4.0,4.666666666666667,4.471021448664016,Right
drive,low,cv_P9_V5_0017,cricketvision,4.533333333333333,5.0,4.666666666666667,4.333333333333333,4.0,4.666666666666667,3.850713249713946,Left
drive,mid,cv_P5_V39_0040,cricketvision,5.666666666666666,6.333333333333333,5.333333333333333,5.0,6.0,5.666666666666667,3.853585907577532,Right
drive,mid,cv_P2_V24_0017,cricketvision,6.533333333333333,7.0,6.0,7.0,6.0,6.666666666666667,4.103950065869171,Right
drive,very_high,cv_P7_V10_0030,cricketvision,8.733333333333334,9.333333333333334,8.666666666666666,9.0,8.333333333333334,8.333333333333334,4.182922298231223,Left
drive,very_high,cv_P6_V8_0036,cricketvision,8.8,9.333333333333334,8.333333333333334,9.333333333333334,8.0,9.0,3.314723942646437,Left
drive,very_low,cv_P5_V20_0027,cricketvision,3.333333333333333,4.0,3.0,3.333333333333333,3.0,3.333333333333333,3.166437102135566,Left
drive,very_low,cv_P4_V6_0058,cricketvision,3.4666666666666672,4.0,3.6666666666666665,3.333333333333333,3.333333333333333,3.0,3.1046502737181,Left
flick_glance,high,cv_P5_V29_0042,cricketvision,7.933333333333333,8.333333333333334,7.666666666666667,8.333333333333334,7.666666666666667,7.666666666666667,3.9918003422886392,Left
flick_glance,high,cv_P1_V22_0025,cricketvision,8.333333333333334,8.666666666666666,8.333333333333334,8.333333333333334,8.333333333333334,8.0,3.633025054245141,Left
flick_glance,low,cv_P1_V22_0105,cricketvision,4.066666666666667,4.666666666666667,3.6666666666666665,4.666666666666667,3.6666666666666665,3.6666666666666665,5.133437877042182,Left
flick_glance,low,cv_P5_V22_0022,cricketvision,4.266666666666668,4.666666666666667,3.6666666666666665,4.333333333333333,4.0,4.666666666666667,8.155389994226718,Right
flick_glance,mid,cv_P2_V24_0069,cricketvision,5.8,6.333333333333333,5.333333333333333,6.333333333333333,5.333333333333333,5.666666666666667,3.2627323105839547,Right
flick_glance,mid,cv_P5_V11_0028,cricketvision,6.466666666666666,7.0,6.333333333333333,6.666666666666667,6.333333333333333,6.0,4.108786460588619,Right
flick_glance,very_high,cv_P7_V10_0034,cricketvision,8.533333333333335,9.333333333333334,8.666666666666666,9.0,7.333333333333333,8.333333333333334,3.416406097842509,Left
flick_glance,very_high,cv_P1_V24_0052,cricketvision,8.799999999999999,8.666666666666666,9.0,8.666666666666666,8.666666666666666,9.0,3.1653094012161307,Left
flick_glance,very_low,cv_P5_V39_0032,cricketvision,3.866666666666666,4.333333333333333,3.333333333333333,4.0,3.333333333333333,4.333333333333333,3.475865450809299,Right
flick_glance,very_low,cv_P4_V6_0054,cricketvision,3.9333333333333336,4.666666666666667,4.0,3.6666666666666665,3.6666666666666665,3.6666666666666665,2.613750342906425,Right
pull_hook,high,cv_P6_V11_0009,cricketvision,7.2,7.666666666666667,7.0,7.666666666666667,6.666666666666667,7.0,3.736046656648553,Left
pull_hook,high,cv_P5_V1_0042,cricketvision,7.4,8.0,7.333333333333333,7.666666666666667,7.0,7.0,3.4324203991646494,Right
pull_hook,low,cv_P2_V27_0018,cricketvision,4.133333333333334,4.666666666666667,3.6666666666666665,4.666666666666667,3.6666666666666665,4.0,2.732115005736273,Right
pull_hook,low,cv_P4_V6_0033,cricketvision,5.466666666666667,6.0,5.333333333333333,6.0,5.0,5.0,2.9740412636704083,Left
pull_hook,mid,cv_P5_V20_0004,cricketvision,6.133333333333334,6.666666666666667,5.666666666666667,5.666666666666667,6.666666666666667,6.0,4.886237100890098,Right
pull_hook,mid,cv_P6_V1_0016,cricketvision,6.266666666666668,6.666666666666667,6.0,6.666666666666667,5.666666666666667,6.333333333333333,5.106756880960072,Right
pull_hook,very_high,cv_P5_V31_0029,cricketvision,8.799999999999999,9.333333333333334,8.666666666666666,8.666666666666666,8.666666666666666,8.666666666666666,4.024178141867457,Right
pull_hook,very_high,cv_P1_V22_0083,cricketvision,8.866666666666667,9.333333333333334,8.666666666666666,8.333333333333334,9.0,9.0,3.2932759506728524,Right
pull_hook,very_low,cv_P9_V1_0029,cricketvision,3.4,4.0,3.0,4.0,3.0,3.0,0.1575554176139807,Right
pull_hook,very_low,cv_P2_V27_0079,cricketvision,3.4666666666666672,4.0,3.333333333333333,3.0,3.333333333333333,3.6666666666666665,3.819291587385272,Left
"""


def clips_dir():
    if LOCAL_CLIPS:
        return LOCAL_CLIPS
    return [c for c in glob.glob("/kaggle/input/**/clips", recursive=True) if os.path.isdir(f"{c}/cricketvision")][0]


NUM_FRAMES = 8
IMG_SIZE = 224
PAD = 0.12
IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], np.float32)
PARTS = ["head", "shoulder", "hands", "hips", "feet"]
TARGETS = PARTS + ["overall"]
SCORE_MIN, SCORE_MAX = 1.0, 10.0


# ---------------------------------------------------------------------------
# Video pipeline -- copied verbatim from kaggle/shot-classifier-timesformer/timesformer_train.py (same
# verified crop/sampling logic; this script is self-contained per Kaggle convention, so it's duplicated
# rather than imported).
# ---------------------------------------------------------------------------

def padded_crop(frame, box, pad=PAD):
    h, w = frame.shape[:2]
    x0, y0, x1, y1 = box
    bw, bh = x1 - x0, y1 - y0
    x0, x1 = x0 - pad * bw, x1 + pad * bw
    y0, y1 = y0 - pad * bh, y1 + pad * bh
    ix0, iy0, ix1, iy1 = int(np.floor(x0)), int(np.floor(y0)), int(np.ceil(x1)), int(np.ceil(y1))
    out_w, out_h = max(ix1 - ix0, 1), max(iy1 - iy0, 1)
    out = np.zeros((out_h, out_w, 3), np.uint8)
    sx0, sy0, sx1, sy1 = max(ix0, 0), max(iy0, 0), min(ix1, w), min(iy1, h)
    if sx1 > sx0 and sy1 > sy0:
        out[sy0 - iy0:sy1 - iy0, sx0 - ix0:sx1 - ix0] = frame[sy0:sy1, sx0:sx1]
    return cv2.resize(out, (IMG_SIZE, IMG_SIZE))


def read_clip_pixels(video_path, boxes, window, num_frames=NUM_FRAMES):
    """Sample num_frames evenly across [lo, hi], crop each to its tracked batter box. Returns
    (num_frames, 3, H, W) float32 ImageNet-normalised RGB, or None if the clip can't be read."""
    lo, c, hi = (int(v) for v in window)
    if hi <= lo:
        return None
    valid = np.isfinite(boxes).all(axis=1)
    if not valid.any():
        return None
    valid_idx = np.flatnonzero(valid)
    idxs = np.linspace(lo, hi, num_frames).round().astype(int)
    idxs = np.clip(idxs, 0, len(boxes) - 1)
    cap = cv2.VideoCapture(str(video_path))
    frames = []
    for i in idxs:
        box_i = i if valid[i] else valid_idx[np.argmin(np.abs(valid_idx - i))]
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(i))
        ok, f = cap.read()
        if not ok:
            cap.release()
            return None
        crop = padded_crop(cv2.cvtColor(f, cv2.COLOR_BGR2RGB), boxes[box_i])
        frames.append(crop)
    cap.release()
    arr = np.stack(frames).astype(np.float32) / 255.0
    arr = (arr - IMAGENET_MEAN) / IMAGENET_STD
    return arr.transpose(0, 3, 1, 2)


# ---------------------------------------------------------------------------
# Evaluation -- copied verbatim from models/evaluate.py (same metric definitions, so numbers stay
# comparable to every other model in the project).
# ---------------------------------------------------------------------------

def spearman(y_true, y_pred):
    a, b = np.asarray(y_true, float), np.asarray(y_pred, float)
    ok = np.isfinite(a) & np.isfinite(b)
    if ok.sum() < 2:
        return float("nan")
    ra, rb = pd.Series(a[ok]).rank(), pd.Series(b[ok]).rank()
    return float(np.corrcoef(ra, rb)[0, 1])


def fisher_mean(rhos):
    r = np.array([x for x in rhos if np.isfinite(x)], float)
    if len(r) == 0:
        return float("nan")
    r = np.clip(r, -0.9999, 0.9999)
    return float(np.tanh(np.arctanh(r).mean()))


def relative_l2(y_true, y_pred, smin=SCORE_MIN, smax=SCORE_MAX):
    a, b = np.asarray(y_true, float), np.asarray(y_pred, float)
    ok = np.isfinite(a) & np.isfinite(b)
    if not ok.any():
        return float("nan")
    return float(np.mean(((a[ok] - b[ok]) / (smax - smin)) ** 2) * 100)


def partial_spearman_vs_overall(df, true_col, pred_col, true_overall_col, pred_overall_col):
    d = df[[true_col, pred_col, true_overall_col, pred_overall_col]].dropna()
    if len(d) < 3:
        return float("nan")

    def resid(y, x):
        b = np.polyfit(x, y, 1)
        return y - np.polyval(b, x)

    rt = resid(d[true_col].to_numpy(float), d[true_overall_col].to_numpy(float))
    rp = resid(d[pred_col].to_numpy(float), d[pred_overall_col].to_numpy(float))
    return spearman(rt, rp)


def evaluate_rating(df, true_cols, pred_cols, by=None, overall_cols=None):
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


def format_report(d, title=""):
    lines = [title] if title else []
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


def to_residual(y):
    out = y.copy()
    out[..., :5] -= y[..., 5:6]
    return out


def from_residual(y):
    out = y.copy()
    out[..., :5] += y[..., 5:6]
    return out


# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------

def load_data(limit=None):
    idx = pd.read_parquet(f"{PROC}/data/processed/pose_index.parquet")
    score_cols = [f"score_{p}" for p in TARGETS]
    d = idx[idx.train_ready & idx.score_overall.notna() & (idx.shot != "other")][
        ["clip_id", "split", "source", "handedness", "shot", "kps_path"] + score_cols]
    exemplars_override = os.environ.get("CRICLENS_EXEMPLARS")
    exemplars = pd.read_csv(exemplars_override) if exemplars_override else pd.read_csv(io.StringIO(EXEMPLARS_CSV))
    ex_merged = exemplars.merge(idx[["clip_id", "kps_path"]], on="clip_id", how="left")
    assert ex_merged.kps_path.notna().all(), "some exemplars missing kps_path -- check EXEMPLARS_PATH"
    d = d[~d.clip_id.isin(exemplars.clip_id)]
    d = d.head(limit) if limit else d
    return d.reset_index(drop=True), ex_merged.reset_index(drop=True)


def embed_clip(clip_id, source, kps_path, clips_root):
    z = np.load(kps_path)
    video_path = f"{clips_root}/{source}/{clip_id}.mp4"
    return read_clip_pixels(video_path, z["boxes"], z["window"])


class QueryDataset(Dataset):
    """One item per query clip. Pairs with a same-shot exemplar's (cached, no-grad) embedding are
    assembled in the training loop, not here -- see module docstring's COMPUTE REDESIGN section."""

    def __init__(self, df, clips_root):
        self.df = df.reset_index(drop=True)
        self.clips_root = clips_root

    def __len__(self):
        return len(self.df)

    def __getitem__(self, i):
        r = self.df.iloc[i]
        pix = embed_clip(r.clip_id, r.source, r.kps_path, self.clips_root)
        if pix is None:
            pix = np.zeros((NUM_FRAMES, 3, IMG_SIZE, IMG_SIZE), np.float32)
        y = np.array([getattr(r, f"score_{p}") for p in TARGETS], np.float32)
        return torch.from_numpy(pix), torch.from_numpy(y), i


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------

class TimesformerContrastiveScorer(nn.Module):
    """Same diff-head design as models/train_technique_scorer_contrastive.py's ContrastiveScorer, with
    the 2-layer GRU encoder replaced by a pretrained Timesformer backbone. embed() takes the CLS token
    of the final layernorm'd hidden state -- verified against the installed transformers source
    (TimesformerForVideoClassification.forward: `sequence_output = outputs[0][:, 0]`, same tensor the
    pretrained classification head itself reads) -- then projects down to `hidden` (128, matching the
    old GRU's hidden size) before the diff head, for a fair architectural comparison."""

    def __init__(self, hidden=128):
        super().__init__()
        self.backbone = TimesformerModel.from_pretrained("facebook/timesformer-base-finetuned-k400")
        self.proj = nn.Linear(self.backbone.config.hidden_size, hidden)
        self.head = nn.Sequential(nn.LayerNorm(hidden * 3), nn.Linear(hidden * 3, hidden), nn.ReLU(),
                                   nn.Dropout(0.2), nn.Linear(hidden, len(TARGETS)))

    def embed(self, pixel_values):
        cls = self.backbone(pixel_values=pixel_values).last_hidden_state[:, 0]
        return self.proj(cls)

    def forward(self, zq, ze):
        """Takes already-embedded query/exemplar vectors (not pixels) -- the caller decides whether
        each side needs gradients (see run()'s no_grad exemplar-embedding step)."""
        return self.head(torch.cat([zq, ze, zq - ze], dim=-1))


def find_checkpoint(gate_tag):
    own = f"{OUT}/checkpoint_{gate_tag}.pt"
    if os.path.exists(own):
        return own
    hits = glob.glob(f"/kaggle/input/**/checkpoint_{gate_tag}.pt", recursive=True)
    return hits[0] if hits else None


# ---------------------------------------------------------------------------
# Train / eval
# ---------------------------------------------------------------------------

def embed_exemplars(model, exemplars, clips_root, device, amp, batch_size=8):
    """Embed all 50 exemplars once (no_grad -- see module docstring), batched. Returns
    {clip_id: (hidden,) tensor on device} plus the raw/target-space scores per exemplar."""
    model.eval()
    embeds = {}
    rows = exemplars.to_dict("records")
    with torch.no_grad(), torch.amp.autocast("cuda", enabled=amp):
        for s in range(0, len(rows), batch_size):
            batch = rows[s:s + batch_size]
            pix = []
            for r in batch:
                p = embed_clip(r["clip_id"], r["source"], r["kps_path"], clips_root)
                pix.append(p if p is not None else np.zeros((NUM_FRAMES, 3, IMG_SIZE, IMG_SIZE), np.float32))
            x = torch.from_numpy(np.stack(pix)).to(device)
            z = model.embed(x)
            for r, zi in zip(batch, z):
                embeds[r["clip_id"]] = zi.float()
    return embeds


def predict_scores(model, df, exemplars, ex_embeds, clips_root, device, amp,
                    batch_size=8, residual_targets=False):
    """Average exemplar_score + predicted_diff across every same-shot exemplar (CoRe-style robustness
    step) -- same design as the GRU version's predict_scores, adapted to use the pre-embedded
    exemplars instead of re-embedding them per query."""
    model.eval()
    score_cols = [f"score_{p}" for p in TARGETS]
    xf = to_residual if residual_targets else (lambda y: y)
    ex_by_shot = {s: xf(g[score_cols].to_numpy(np.float32)) for s, g in exemplars.groupby("shot")}
    ex_ids_by_shot = {s: g.clip_id.tolist() for s, g in exemplars.groupby("shot")}
    out = np.zeros((len(df), len(TARGETS)), np.float32)
    with torch.no_grad(), torch.amp.autocast("cuda", enabled=amp):
        for shot, grp in df.groupby("shot"):
            ex_scores = ex_by_shot[shot]
            ze_all = torch.stack([ex_embeds[eid] for eid in ex_ids_by_shot[shot]]).to(device)  # (n_ex, hidden)
            idx = grp.index.to_numpy()
            for s in range(0, len(idx), batch_size):
                sub = idx[s:s + batch_size]
                pix = [embed_clip(df.clip_id[i], df.source[i], df.kps_path[i], clips_root) for i in sub]
                pix = [p if p is not None else np.zeros((NUM_FRAMES, 3, IMG_SIZE, IMG_SIZE), np.float32) for p in pix]
                xq = torch.from_numpy(np.stack(pix)).to(device)
                zq = model.embed(xq)  # (b, hidden)
                b, n_ex = zq.shape[0], ze_all.shape[0]
                zq_rep = zq.unsqueeze(1).expand(-1, n_ex, -1).reshape(b * n_ex, -1)
                ze_rep = ze_all.unsqueeze(0).expand(b, -1, -1).reshape(b * n_ex, -1)
                diff = model(zq_rep, ze_rep).float().cpu().numpy().reshape(b, n_ex, -1)
                pred = ex_scores[None, :, :] + diff
                out[sub] = pred.mean(axis=1)
    return from_residual(out) if residual_targets else out


def run(epochs, device, tr, va, te, exemplars, clips_root, batch_size=8, patience=8, lr=1e-4,
        hidden=128, seed=0, residual_targets=True, select_metric="combined", tag="timesformer"):
    torch.manual_seed(seed); np.random.seed(seed)
    print(f"[{tag}, seed={seed}] train={len(tr)} val={len(va)} test={len(te)} exemplars={len(exemplars)}", flush=True)

    model = TimesformerContrastiveScorer(hidden=hidden).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    amp = device == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=amp)

    score_cols = [f"score_{p}" for p in TARGETS]
    xf = to_residual if residual_targets else (lambda y: y)
    ex_by_shot_df = {s: g for s, g in exemplars.groupby("shot")}

    dl_tr = DataLoader(QueryDataset(tr, clips_root), batch_size=batch_size, shuffle=True, num_workers=4)
    tr_idx = tr.reset_index(drop=True)

    # best_state starts as the untrained model, not None -- val_select_score can come back nan (e.g. a
    # tiny or degenerate val split where Spearman is undefined), and `nan > best_score` is always False
    # in Python, so a None starting point would leave best_state unset and crash at the final
    # load_state_dict if every epoch's score happened to be nan. An untrained fallback is always valid.
    start_ep, best_score, best_state, bad = 0, -1e9, {k: v.cpu().clone() for k, v in model.state_dict().items()}, 0
    ckpt_path = find_checkpoint(tag)
    if ckpt_path:
        ckpt = torch.load(ckpt_path, map_location=device)
        model.load_state_dict(ckpt["model"]); opt.load_state_dict(ckpt["opt"]); scaler.load_state_dict(ckpt["scaler"])
        start_ep, best_score, best_state, bad = ckpt["epoch"], ckpt["best_score"], ckpt["best_state"], ckpt["bad"]
        print(f"[{tag}, seed={seed}] RESUMED: epoch={start_ep} best_score={best_score:.3f}", flush=True)

    n_batches = len(dl_tr)
    for ep in range(start_ep, epochs):
        # One random same-shot exemplar per query, resampled every epoch (module docstring: COMPUTE
        # REDESIGN #2) -- exemplar embeddings for this epoch computed once, no_grad, reused for every
        # query pairing below instead of re-embedded per pair.
        rng = np.random.RandomState(seed * 1000 + ep)
        ex_choice = {}
        for r in tr_idx.itertuples():
            cand = ex_by_shot_df[r.shot]
            ex_choice[r.Index] = cand.iloc[rng.randint(len(cand))]
        ex_embeds = embed_exemplars(model, exemplars, clips_root, device, amp)

        model.train(); t0 = time.time(); tot_loss = 0.0; seen = 0; t_last_print = t0
        for bi, (xq, y, idxs) in enumerate(dl_tr, 1):
            xq, y = xq.to(device), y.to(device)
            qy = xf(y.cpu().numpy())
            ex_rows = [ex_choice[int(i)] for i in idxs]
            ey = np.stack([xf(np.array([getattr(r, c) for c in score_cols], np.float32)) for r in ex_rows])
            diff_target = torch.from_numpy(qy - ey).to(device)
            ze = torch.stack([ex_embeds[r.clip_id] for r in ex_rows]).to(device)  # no_grad already (embed_exemplars)

            opt.zero_grad()
            with torch.amp.autocast("cuda", enabled=amp):
                zq = model.embed(xq)
                pred = model(zq, ze)
                loss = nn.functional.mse_loss(pred, diff_target)
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(opt); scaler.update()
            tot_loss += loss.item() * len(y); seen += len(y)
            now = time.time()
            if now - t_last_print >= 60 or bi == n_batches:
                rate = seen / (now - t0)
                eta_min = (n_batches * batch_size - seen) / rate / 60 if rate > 0 else float("nan")
                print(f"    [{tag}, seed={seed}] epoch {ep+1}/{epochs}  batch {bi}/{n_batches}  "
                      f"running_loss={tot_loss/seen:.3f}  ({rate:.1f} clips/s, ~{eta_min:.1f} min left this epoch)",
                      flush=True)
                t_last_print = now

        va_ex_embeds = embed_exemplars(model, exemplars, clips_root, device, amp)
        va_pred = predict_scores(model, va.reset_index(drop=True), exemplars, va_ex_embeds, clips_root,
                                  device, amp, residual_targets=residual_targets)
        va_df = va.reset_index(drop=True).assign(**{f"pred_{p}": va_pred[:, i] for i, p in enumerate(TARGETS)})
        r = evaluate_rating(va_df, {p: f"score_{p}" for p in PARTS}, {p: f"pred_{p}" for p in PARTS},
                             overall_cols=("score_overall", "pred_overall"))
        score = (r["mean_spearman"] + r["mean_partial_spearman_vs_overall"]) if select_metric == "combined" else r["mean_spearman"]
        print(f"  [{tag}, seed={seed}] epoch {ep+1}/{epochs}  train_loss={tot_loss/len(tr_idx):.3f}  "
              f"val_rho={r['mean_spearman']:.3f}  val_partial_rho={r['mean_partial_spearman_vs_overall']:.3f}  "
              f"select={score:.3f}  ({time.time()-t0:.1f}s)", flush=True)

        if score > best_score:
            best_score, best_state, bad = score, {k: v.cpu().clone() for k, v in model.state_dict().items()}, 0
        else:
            bad += 1
        torch.save({"model": model.state_dict(), "opt": opt.state_dict(), "scaler": scaler.state_dict(),
                    "epoch": ep + 1, "best_score": best_score, "best_state": best_state, "bad": bad},
                   f"{OUT}/checkpoint_{tag}.pt")
        if bad >= patience:
            print(f"  [{tag}, seed={seed}] early stop", flush=True)
            break

    model.load_state_dict(best_state); model.eval()
    te_ex_embeds = embed_exemplars(model, exemplars, clips_root, device, amp)
    te_r = te.reset_index(drop=True)
    te_pred = predict_scores(model, te_r, exemplars, te_ex_embeds, clips_root, device, amp,
                              residual_targets=residual_targets)
    te_df = te_r.assign(**{f"pred_{p}": te_pred[:, i] for i, p in enumerate(TARGETS)})
    result = evaluate_rating(te_df, {p: f"score_{p}" for p in PARTS}, {p: f"pred_{p}" for p in PARTS},
                              by="source", overall_cols=("score_overall", "pred_overall"))
    overall_spearman = spearman(te_df.score_overall, te_df.pred_overall)
    print(format_report(result, f"\n[{tag}, seed={seed}] TEST"), flush=True)
    print(f"\n[{tag}, seed={seed}] overall-score Spearman (test): {overall_spearman:.3f}  "
          f"(I3D-AE-LSTM baseline: 0.84; pose-GRU contrastive v2: 0.588 mean-part-rho/0.122 partial-rho)", flush=True)

    torch.save(best_state, f"{OUT}/scorer_{tag}_seed{seed}.pt")
    summary = {"tag": tag, "seed": seed, "hidden": hidden, "lr": lr, "residual_targets": residual_targets,
               "select_metric": select_metric, "best_val_select_score": best_score,
               "test_mean_spearman": result["mean_spearman"], "test_mean_r_l2": result["mean_r_l2"],
               "test_mean_partial_spearman_vs_overall": result["mean_partial_spearman_vs_overall"],
               "test_overall_spearman": overall_spearman, "per_part": {p: result["per_part"][p] for p in PARTS},
               "n_train": len(tr), "n_val": len(va), "n_test": len(te), "n_exemplars": len(exemplars)}
    json.dump(summary, open(f"{OUT}/summary.json", "w"), indent=2)
    del model, opt, scaler, best_state
    if device == "cuda":
        torch.cuda.empty_cache()
    return summary


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    epochs = int(os.environ.get("CRICLENS_EPOCHS", "15"))
    seed = int(os.environ.get("CRICLENS_SEED", "0"))
    residual_targets = os.environ.get("CRICLENS_RESIDUAL", "1") == "1"  # v2's winning config, on by default
    print(f"device={device}  epochs={epochs}  seed={seed}  residual_targets={residual_targets}", flush=True)

    d, exemplars = load_data(LIMIT)
    tr, va, te = (d[d.split == s] for s in ("train", "val", "test"))
    root = clips_dir()

    summary = run(epochs, device, tr, va, te, exemplars, root, seed=seed, residual_targets=residual_targets)
    print("\n=== summary ===", flush=True)
    print(json.dumps({k: v for k, v in summary.items() if k != "per_part"}, indent=2), flush=True)


if __name__ == "__main__":
    main()
