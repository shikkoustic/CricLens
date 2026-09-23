"""Frame-rate normalisation: put every clip's window on one common temporal sampling.

Source clips are 23-30 fps (mostly 25 or 30), and the analysis window is defined in seconds
(kaggle/pose-*/*.py: PRE=0.8s, FOLLOW=0.6s) then converted to frames via each clip's own fps. So the
same real-world swing duration currently produces different frame counts depending on source fps
(models/pose_index.py's report: mean 26.1 frames at 30fps vs 32.0 frames at 25fps for a similar window
in seconds) -- a sequence model sees the same shot "play out" over a different number of steps depending
on which broadcast it came from. That is the problem PROGRESS.md's "frame-rate normalisation" step names.

The window also varies in real duration per clip (0.1-1.44s: contact-anchored, clipped at camera cuts),
so fixing the frame *rate* alone would still leave clips at different frame *counts*. The standard fix in
the AQA literature (MTL-AQA, USDL, CoRe, TSA: 96-103 raw frames resampled to a fixed number of snippets)
is to resample directly onto a fixed number of output frames via normalised time, which solves both
problems in one step: two clips of the same real duration produce numerically comparable trajectories
regardless of source fps, and every clip becomes the same length for batching.

What this script does NOT do: spatial normalisation (centering on the hips, scaling by torso length,
COCO-index remapping) or NaN-mask feature engineering. Those are model input choices and belong to the
training code (tasks #4/#5), not this shared preprocessing step -- different models may want different
spatial features from the same resampled trajectory.

Output: one row per clip in data/processed/sequences.parquet + one .npz per clip in
data/processed/sequences/<clip_id>.npz holding:
  seq         (T, 17, 3) float32  -- x, y, confidence resampled to T frames over normalised window time
  valid       (T, 17)    bool     -- False where every native frame for that joint was NaN (nothing to
                                      interpolate from); seq is NaN there, callers must mask or impute
  contact_idx float32             -- where contact falls in the resampled sequence, in [0, T-1]
  native_fps, native_frames, win_sec  -- provenance, for anyone who wants to weight or filter by these

    python models/resample_sequences.py            # all clips with kps_exists
    python models/resample_sequences.py --train-ready-only
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = ROOT / "data/processed/sequences"
OUT_INDEX = ROOT / "data/processed/sequences.parquet"
T = 32  # output frames: covers the window-length distribution (p50=27, p90=36) without extreme resampling


def resample_clip(kps: np.ndarray, window: np.ndarray, fps: float, t_out: int = T) -> dict | None:
    lo, c, hi = (int(v) for v in window)
    kw = kps[lo:hi + 1]  # (Wf, 17, 3)
    wf = len(kw)
    if wf < 2:
        return None
    t_in = np.arange(wf) / fps  # seconds from window start, at native fps
    dur = t_in[-1]
    if dur <= 0:
        return None
    tn_in = t_in / dur  # normalised to [0, 1]
    tn_out = np.linspace(0.0, 1.0, t_out)
    contact_idx = float(np.interp((c - lo) / fps / dur, [0.0, 1.0], [0.0, t_out - 1.0]))

    seq = np.full((t_out, 17, 3), np.nan, np.float32)
    valid = np.zeros((t_out, 17), bool)
    for j in range(17):
        for ch in range(3):  # x, y, confidence -- all resampled the same way; ch=2 (confidence) is a soft signal
            v = kw[:, j, ch]
            ok = np.isfinite(v)
            if ok.sum() < 2:  # nothing (or one point) to interpolate from: leave NaN, mark invalid
                continue
            seq[:, j, ch] = np.interp(tn_out, tn_in[ok], v[ok])
            if ch == 0:
                valid[:, j] = True  # x defines validity; x/y/conf share the same source mask by construction
    return {"seq": seq, "valid": valid, "contact_idx": contact_idx,
            "native_fps": float(fps), "native_frames": wf, "win_sec": float(dur)}


def main(train_ready_only: bool) -> None:
    idx = pd.read_parquet(ROOT / "data/processed/pose_index.parquet")
    idx = idx[idx.kps_exists]
    if train_ready_only:
        idx = idx[idx.train_ready]
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    rows, failed = [], 0
    for r in tqdm(idx.itertuples(), total=len(idx)):
        try:
            z = np.load(ROOT / r.kps_path)
            out = resample_clip(z["kps"], z["window"], float(z["fps"]))
        except Exception:
            out = None
        if out is None:
            failed += 1
            continue
        np.savez_compressed(OUT_DIR / f"{r.clip_id}.npz", seq=out["seq"], valid=out["valid"],
                             contact_idx=out["contact_idx"])
        rows.append({"clip_id": r.clip_id, "seq_path": str((OUT_DIR / f"{r.clip_id}.npz").relative_to(ROOT)),
                     "native_fps": out["native_fps"], "native_frames": out["native_frames"], "win_sec": out["win_sec"],
                     "contact_idx": out["contact_idx"], "valid_joint_frac": float(out["valid"].mean())})
    d = pd.DataFrame(rows)
    d.to_parquet(OUT_INDEX, index=False)
    print(f"resampled {len(d)} clips to T={T} frames ({failed} skipped: window too short or zero duration)")
    print(f"native_fps range seen: {d.native_fps.min():.2f}-{d.native_fps.max():.2f}, "
          f"native_frames range: {d.native_frames.min()}-{d.native_frames.max()}")
    print(f"mean fraction of joints with any real data: {d.valid_joint_frac.mean():.3f}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--train-ready-only", action="store_true")
    main(**vars(ap.parse_args()))
