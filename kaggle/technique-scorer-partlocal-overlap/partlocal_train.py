"""Technique scorer v3b, part-localised pose input with OVERLAPPING joint groups -- Kaggle GPU run
(self-contained, no repo imports; mirrors models/train_technique_scorer_partlocal_overlap.py). v3 (the
plain, disjoint-joint-groups version, `kaggle/technique-scorer-partlocal/`) came back as a clean negative
result on Kaggle, 3-seed confirmed: worse than v2 (shared input) on every metric. See PROGRESS.md
2026-10-21 for the full rationale and v3's result.

This follow-up tests one specific explanation for that negative result: that 2-4 disjoint joints per part
may simply be too little signal. Each part's joint group now includes its biomechanical neighbours along
the kinetic chain (e.g. "hands" = shoulders+elbows+wrists, not just elbows+wrists) -- see JOINT_GROUPS
below. The other untested explanation from v3 (that detaching the shared global context from part heads
cuts off useful signal) is deliberately left unchanged here, so only one variable moves at a time.

Runs 3 seeds (0, 1, 2) back to back for an immediate variance read, same protocol as the v2 contrastive
scorer's seed check (PROGRESS.md 2026-10-17) and v3's. Same hyperparameters as v3 (lr=3e-4, hidden=128,
local_hidden=48, patience=15) so the only thing that differs from v3's run is JOINT_GROUPS.

Inputs : shikkoustic/criclens-all (criclens-processed.tgz.bin -> data/processed/pose_index.parquet +
          kaggle/chunks/pose-*/out/kps/<clip>.npz; sequences are resampled on the fly, see resample_clip --
          data/processed/sequences.parquet itself isn't part of the packed tarball)
Outputs: summary_seed<k>.json (one per seed), summary_all.json (aggregate mean/std across seeds),
         partlocal_scorer_seed<k>.pt
"""
import glob
import io
import json
import os
import subprocess
import sys
import time

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

OUT = os.environ.get("CRICLENS_OUT", "/kaggle/working")
LOCAL = os.environ.get("CRICLENS_LOCAL")  # repo root, for a fast local smoke test; unset on Kaggle
LIMIT = int(os.environ.get("CRICLENS_LIMIT", "0"))
os.makedirs(OUT, exist_ok=True)
if LOCAL:
    PROC = LOCAL
else:
    PROC = "/tmp/proc"
    os.makedirs(PROC, exist_ok=True)
    _tgz = glob.glob("/kaggle/input/**/criclens-processed.tgz.bin", recursive=True)[0]
    subprocess.run(["tar", "-xzf", _tgz, "-C", PROC, "--wildcards",
                     "data/processed/*", "kaggle/chunks/pose-*/out/kps/*"], check=True)

PARTS = ["head", "shoulder", "hands", "hips", "feet"]
TARGETS = PARTS + ["overall"]
SCORE_MIN, SCORE_MAX = 1.0, 10.0

# COCO-17 joint order: 0 nose, 1 l_eye, 2 r_eye, 3 l_ear, 4 r_ear, 5 l_shoulder, 6 r_shoulder, 7 l_elbow,
# 8 r_elbow, 9 l_wrist, 10 r_wrist, 11 l_hip, 12 r_hip, 13 l_knee, 14 r_knee, 15 l_ankle, 16 r_ankle
JOINT_GROUPS = {"head": [0, 1, 2, 3, 4, 5, 6], "shoulder": [5, 6, 7, 8, 11, 12], "hands": [5, 6, 7, 8, 9, 10],
                "hips": [5, 6, 11, 12, 13, 14], "feet": [11, 12, 13, 14, 15, 16]}
PART_COLS = {p: [c for j in js for c in (3 * j, 3 * j + 1, 3 * j + 2)] for p, js in JOINT_GROUPS.items()}
HIP_L, HIP_R, SHO_L, SHO_R = 11, 12, 5, 6

# kaggle/exemplars.csv verbatim (models/exemplar_review.py; 50 clips, train-split only) -- embedded since
# this kernel only sees this file + the dataset, not the repo. Any edit to kaggle/exemplars.csv needs
# re-pasting here.
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


# ---- evaluation (models/evaluate.py, copied: spearman / fisher_mean / relative_l2 / partial_spearman_vs_overall /
# evaluate_rating / format_report, unchanged) ----

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


# ---- data (models/train_shot_classifier.py normalise + train_technique_scorer_contrastive.py, copied) ----

T_OUT = 32  # models/resample_sequences.py: covers the window-length distribution (p50=27, p90=36)


def resample_clip(kps, window, fps, t_out=T_OUT):
    """models/resample_sequences.py's resample_clip, copied verbatim: data/processed/sequences.parquet
    (built by that local CPU script) isn't part of criclens-processed.tgz.bin -- only the raw per-clip
    kps.npz chunks are -- so this kernel resamples on the fly instead of relying on the prebuilt parquet."""
    lo, c, hi = (int(v) for v in window)
    kw = kps[lo:hi + 1]
    wf = len(kw)
    if wf < 2:
        return None
    t_in = np.arange(wf) / fps
    dur = t_in[-1]
    if dur <= 0:
        return None
    tn_in = t_in / dur
    tn_out = np.linspace(0.0, 1.0, t_out)
    seq = np.full((t_out, 17, 3), np.nan, np.float32)
    for j in range(17):
        for ch in range(3):
            v = kw[:, j, ch]
            ok = np.isfinite(v)
            if ok.sum() < 2:
                continue
            seq[:, j, ch] = np.interp(tn_out, tn_in[ok], v[ok])
    return seq


def normalise(seq):
    """seq: (T, 17, 3). Per-frame hip-centred, per-clip scale-normalised (x, y); confidence untouched."""
    xy = seq[:, :, :2]
    hip = np.nanmean(xy[:, [HIP_L, HIP_R]], axis=1, keepdims=True)
    centred = xy - hip
    sho = np.nanmean(xy[:, [SHO_L, SHO_R]], axis=1)
    scale = np.nanmedian(np.linalg.norm(sho - hip[:, 0], axis=-1))
    scale = scale if np.isfinite(scale) and scale > 1e-3 else 1.0
    out = seq.copy()
    out[:, :, :2] = np.nan_to_num(centred / scale, nan=0.0)
    out[:, :, 2] = np.nan_to_num(out[:, :, 2], nan=0.0)
    return out.reshape(seq.shape[0], -1).astype(np.float32)


def to_residual(y):
    out = y.copy()
    out[..., :5] -= y[..., 5:6]
    return out


def from_residual(y):
    out = y.copy()
    out[..., :5] += y[..., 5:6]
    return out


def load_data(limit=None):
    idx = pd.read_parquet(f"{PROC}/data/processed/pose_index.parquet")
    score_cols = [f"score_{p}" for p in TARGETS]
    d = idx[idx.train_ready & idx.score_overall.notna() & (idx.shot != "other")][
        ["clip_id", "split", "source", "handedness", "shot", "kps_path"] + score_cols]
    exemplars = pd.read_csv(io.StringIO(EXEMPLARS_CSV))
    d = d[~d.clip_id.isin(exemplars.clip_id)]
    limit = limit or (LIMIT or None)
    d = d.head(limit) if limit else d
    return d.reset_index(drop=True), exemplars.reset_index(drop=True), idx, None


def cache_sequences(clip_ids, kps_path_by_clip):
    cache, failed = {}, []
    for cid in clip_ids:
        z = np.load(f"{PROC}/{kps_path_by_clip[cid]}")
        seq = resample_clip(z["kps"], z["window"], float(z["fps"]))
        if seq is None:
            failed.append(cid)
            continue
        cache[cid] = normalise(seq)
    if failed:
        print(f"WARNING: {len(failed)} clips failed to resample (window too short/zero duration), "
              f"dropped: {failed[:10]}{'...' if len(failed) > 10 else ''}", flush=True)
    return cache, failed


class PairDataset(Dataset):
    def __init__(self, df, exemplars, cache, diff_mu, diff_sd):
        score_cols = [f"score_{p}" for p in TARGETS]
        ex_by_shot = {s: g for s, g in exemplars.groupby("shot")}
        self.pairs = []
        for r in df.itertuples():
            qy = to_residual(np.array([getattr(r, c) for c in score_cols], np.float32))
            for e in ex_by_shot[r.shot].itertuples():
                ey = to_residual(np.array([getattr(e, c) for c in score_cols], np.float32))
                self.pairs.append((r.clip_id, e.clip_id, qy, ey))
        self.cache, self.diff_mu, self.diff_sd = cache, diff_mu, diff_sd

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, i):
        qid, eid, qy, ey = self.pairs[i]
        diff = (qy - ey - self.diff_mu) / self.diff_sd
        return (torch.from_numpy(self.cache[qid]), torch.from_numpy(self.cache[eid]),
                torch.from_numpy(diff), torch.from_numpy(ey))


# ---- model (models/train_technique_scorer_partlocal.py, copied) ----

class PartLocalScorer(nn.Module):
    def __init__(self, in_dim, hidden=128, local_hidden=48):
        super().__init__()
        self.global_encoder = nn.GRU(in_dim, hidden, num_layers=2, batch_first=True, dropout=0.2)
        self.overall_head = nn.Sequential(nn.LayerNorm(hidden * 3), nn.Linear(hidden * 3, hidden), nn.ReLU(),
                                           nn.Dropout(0.2), nn.Linear(hidden, 1))
        self.local_encoders = nn.ModuleDict({
            p: nn.GRU(len(cols), local_hidden, num_layers=1, batch_first=True) for p, cols in PART_COLS.items()})
        local_in = local_hidden * 3 + hidden
        self.part_heads = nn.ModuleDict({
            p: nn.Sequential(nn.LayerNorm(local_in), nn.Linear(local_in, local_hidden), nn.ReLU(),
                              nn.Dropout(0.2), nn.Linear(local_hidden, 1)) for p in PART_COLS})

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
        return torch.cat(part_outs + [overall_out], dim=-1)


def predict_scores(model, df, exemplars, cache, diff_mu, diff_sd, device, batch_size=256):
    model.eval()
    score_cols = [f"score_{p}" for p in TARGETS]
    ex_by_shot = {s: to_residual(g[score_cols].to_numpy(np.float32)) for s, g in exemplars.groupby("shot")}
    ex_ids_by_shot = {s: g.clip_id.tolist() for s, g in exemplars.groupby("shot")}
    out = np.zeros((len(df), len(TARGETS)), np.float32)
    with torch.no_grad():
        for shot, grp in df.groupby("shot"):
            ex_scores, ex_ids = ex_by_shot[shot], ex_ids_by_shot[shot]
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
    return from_residual(out)


def run_one_seed(seed, d, exemplars, cache, device, epochs=60, batch_size=64, patience=15,
                  lr=3e-4, hidden=128, local_hidden=48, weight_decay=1e-4, select_metric="combined"):
    torch.manual_seed(seed); np.random.seed(seed)
    tr, va, te = (d[d.split == s] for s in ("train", "val", "test"))
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

    in_dim = next(iter(cache.values())).shape[1]
    model = PartLocalScorer(in_dim, hidden=hidden, local_hidden=local_hidden).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    dl_tr = DataLoader(PairDataset(tr, exemplars, cache, diff_mu, diff_sd), batch_size=batch_size, shuffle=True)

    def select_score(r):
        return r["mean_spearman"] + r["mean_partial_spearman_vs_overall"] if select_metric == "combined" else r["mean_spearman"]

    best_rho, best_state, bad = -1.0, None, 0
    for ep in range(epochs):
        model.train(); t0 = time.time(); tot_loss = 0.0
        for xq, xe, diff, _ in dl_tr:
            xq, xe, diff = xq.to(device), xe.to(device), diff.to(device)
            opt.zero_grad()
            loss = nn.functional.mse_loss(model(xq, xe), diff)
            loss.backward(); opt.step()
            tot_loss += loss.item() * len(diff)
        va_pred = predict_scores(model, va.reset_index(drop=True), exemplars, cache, diff_mu, diff_sd, device)
        va_df = va.reset_index(drop=True).assign(**{f"pred_{p}": va_pred[:, i] for i, p in enumerate(TARGETS)})
        r = evaluate_rating(va_df, {p: f"score_{p}" for p in PARTS}, {p: f"pred_{p}" for p in PARTS},
                             overall_cols=("score_overall", "pred_overall"))
        score = select_score(r)
        print(f"  [seed {seed}] epoch {ep+1}/{epochs}  loss={tot_loss/len(dl_tr.dataset):.3f}  "
              f"val_rho={r['mean_spearman']:.3f}  val_partial_rho={r['mean_partial_spearman_vs_overall']:.3f}  "
              f"select={score:.3f}  ({time.time()-t0:.1f}s)", flush=True)
        if score > best_rho:
            best_rho, best_state, bad = score, {k: v.cpu().clone() for k, v in model.state_dict().items()}, 0
        else:
            bad += 1
            if bad >= patience:
                print(f"  [seed {seed}] early stop (no val {select_metric} improvement in {patience} epochs)", flush=True)
                break

    model.load_state_dict(best_state)
    te_r = te.reset_index(drop=True)
    te_pred = predict_scores(model, te_r, exemplars, cache, diff_mu, diff_sd, device)
    te_df = te_r.assign(**{f"pred_{p}": te_pred[:, i] for i, p in enumerate(TARGETS)})
    result = evaluate_rating(te_df, {p: f"score_{p}" for p in PARTS}, {p: f"pred_{p}" for p in PARTS},
                              by="source", overall_cols=("score_overall", "pred_overall"))
    overall_spearman = spearman(te_df.score_overall, te_df.pred_overall)
    torch.save(best_state, f"{OUT}/partlocal_scorer_seed{seed}.pt")
    print(format_report(result, f"\nTEST (seed {seed})"), flush=True)
    print(f"overall-score Spearman (test): {overall_spearman:.3f}  (I3D-AE-LSTM baseline: 0.84)", flush=True)
    summary = {"seed": seed, "lr": lr, "hidden": hidden, "local_hidden": local_hidden,
               "weight_decay": weight_decay, "select_metric": select_metric,
               "best_epoch_val_select_score": best_rho, "test_mean_spearman": result["mean_spearman"],
               "test_mean_r_l2": result["mean_r_l2"],
               "test_mean_partial_spearman_vs_overall": result["mean_partial_spearman_vs_overall"],
               "test_overall_spearman": overall_spearman,
               "per_part": {p: result["per_part"][p] for p in PARTS},
               "n_train": len(tr), "n_val": len(va), "n_test": len(te), "n_exemplars": len(exemplars)}
    json.dump(summary, open(f"{OUT}/summary_seed{seed}.json", "w"), indent=2)
    return summary


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    epochs = int(os.environ.get("CRICLENS_EPOCHS", "60"))
    seeds = [int(s) for s in os.environ.get("CRICLENS_SEEDS", "0,1,2").split(",")]
    print(f"device={device}  epochs={epochs}  seeds={seeds}", flush=True)
    d, exemplars, idx, _ = load_data()
    print(f"train={int((d.split=='train').sum())} val={int((d.split=='val').sum())} "
          f"test={int((d.split=='test').sum())}  exemplars={len(exemplars)}", flush=True)
    kps_path_by_clip = {**dict(zip(d.clip_id, d.kps_path)),
                         **dict(zip(exemplars.clip_id,
                                    idx.set_index("clip_id").loc[exemplars.clip_id, "kps_path"]))}
    cache, failed = cache_sequences(set(kps_path_by_clip), kps_path_by_clip)
    print(f"cached {len(cache)} sequences ({len(failed)} failed)", flush=True)
    if set(failed) & set(exemplars.clip_id):
        print("WARNING: an exemplar failed to resample -- check kaggle/exemplars.csv", flush=True)
    d = d[~d.clip_id.isin(failed)].reset_index(drop=True)
    exemplars = exemplars[~exemplars.clip_id.isin(failed)].reset_index(drop=True)

    summaries = [run_one_seed(seed, d, exemplars, cache, device, epochs=epochs) for seed in seeds]
    agg = {
        "test_mean_spearman": {"mean": float(np.mean([s["test_mean_spearman"] for s in summaries])),
                                "std": float(np.std([s["test_mean_spearman"] for s in summaries]))},
        "test_mean_partial_spearman_vs_overall": {
            "mean": float(np.mean([s["test_mean_partial_spearman_vs_overall"] for s in summaries])),
            "std": float(np.std([s["test_mean_partial_spearman_vs_overall"] for s in summaries]))},
        "test_mean_r_l2": {"mean": float(np.mean([s["test_mean_r_l2"] for s in summaries])),
                            "std": float(np.std([s["test_mean_r_l2"] for s in summaries]))},
        "per_seed": summaries,
    }
    json.dump(agg, open(f"{OUT}/summary_all.json", "w"), indent=2)
    print("\n=== AGGREGATE (3 seeds) ===", flush=True)
    print(f"mean_spearman: {agg['test_mean_spearman']['mean']:.3f} +/- {agg['test_mean_spearman']['std']:.3f}", flush=True)
    print(f"partial_rho_vs_overall: {agg['test_mean_partial_spearman_vs_overall']['mean']:.3f} "
          f"+/- {agg['test_mean_partial_spearman_vs_overall']['std']:.3f}", flush=True)
    print("reference (v2 contrastive + residual targets, 3 seeds, PROGRESS.md 2026-10-17): "
          "mean_spearman 0.595, partial_rho 0.137", flush=True)


if __name__ == "__main__":
    main()
