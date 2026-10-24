"""Shot classifier (GRU only) -- Kaggle GPU run, to produce a pretrained encoder for the technique scorer.

Mirrors models/train_shot_classifier.py's GRU config exactly (same architecture, hyperparameters, loss),
self-contained since Kaggle kernels only see this pushed file + the input dataset (same convention as
kaggle/pose-pad/pose_pad.py, kaggle/technique-scorer-partlocal/partlocal_train.py). Trains on ALL
train_ready clips from ALL 6 sources (16,460 clips), not just CricketVision's scored subset -- this is
the point: PROGRESS.md 2026-10-24's pretraining idea is to give the technique scorer's encoder a head
start on general shot mechanics from a much bigger pool before fine-tuning on the small (~3,662-clip)
scored set.

GRU specifically (not the LSTM that was the shot-classifier's own chosen default) because the technique
scorer's encoder (models/train_technique_scorer_contrastive.py's ContrastiveScorer) is an
nn.GRU(51, 128, num_layers=2, dropout=0.2) -- same shape, so this encoder's state_dict loads into it
directly with no architecture surgery.

Sequences are resampled on the fly from the raw kps.npz chunks (data/processed/sequences.parquet isn't
part of criclens-processed.tgz.bin -- caught and fixed the same way in the partlocal job).

Inputs : shikkoustic/criclens-all (criclens-processed.tgz.bin -> data/processed/pose_index.parquet +
          kaggle/chunks/pose-*/out/kps/<clip>.npz)
Outputs: shot_gru_encoder.pt (just the GRU encoder's state_dict, for reuse), shot_gru_full.pt (full model,
          for reference), summary.json
"""
import glob
import json
import os
import subprocess
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

OUT = os.environ.get("CRICLENS_OUT", "/kaggle/working")
LOCAL = os.environ.get("CRICLENS_LOCAL")
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

HIP_L, HIP_R, SHO_L, SHO_R = 11, 12, 5, 6
T_OUT = 32


def resample_clip(kps, window, fps, t_out=T_OUT):
    """models/resample_sequences.py's resample_clip, copied verbatim (see partlocal_train.py for the
    same pattern and rationale)."""
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
    """models/train_shot_classifier.py's normalise, copied verbatim."""
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


def load_labelled_clips(limit=None):
    idx = pd.read_parquet(f"{PROC}/data/processed/pose_index.parquet")
    d = idx[idx.train_ready & (idx.shot != "other")][["clip_id", "shot", "split", "source", "handedness", "kps_path"]]
    limit = limit or (LIMIT or None)
    return d.head(limit) if limit else d


def cache_sequences(df):
    cache, failed = {}, []
    for r in df.itertuples():
        z = np.load(f"{PROC}/{r.kps_path}")
        seq = resample_clip(z["kps"], z["window"], float(z["fps"]))
        if seq is None:
            failed.append(r.clip_id)
            continue
        cache[r.clip_id] = normalise(seq)
    return cache, failed


class SeqDataset(Dataset):
    def __init__(self, df, classes, cache):
        self.df = df.reset_index(drop=True)
        self.c2i = {c: i for i, c in enumerate(classes)}
        self.cache = cache

    def __len__(self):
        return len(self.df)

    def __getitem__(self, i):
        r = self.df.iloc[i]
        return torch.from_numpy(self.cache[r.clip_id]), self.c2i[r.shot]


class RecurrentClassifier(nn.Module):
    """Same GRU shape as models/train_shot_classifier.py's RecurrentClassifier(cell="gru") and as
    ContrastiveScorer's encoder, so self.rnn's state_dict transfers directly -- but the READOUT is
    deliberately changed from the original (mean-pool over all 32 timesteps) to final-hidden-state only
    (h[-1]), to match exactly how ContrastiveScorer.embed() reads its own encoder. PROGRESS.md 2026-10-24:
    the first pretraining attempt used the original mean-pool readout and came back negative; the
    pretrained weights were shaped for a mean-pooled representation the scorer never actually uses, which
    is a likely reason transfer hurt instead of helped. This fix makes the pretraining task read the
    encoder exactly the way the downstream task will, so the weights are optimised for the right thing."""

    def __init__(self, in_dim, n_classes, hidden=128, layers=2):
        super().__init__()
        self.rnn = nn.GRU(in_dim, hidden, num_layers=layers, batch_first=True, dropout=0.2 if layers > 1 else 0)
        self.head = nn.Sequential(nn.LayerNorm(hidden), nn.Linear(hidden, n_classes))

    def forward(self, x):
        _, h = self.rnn(x)
        return self.head(h[-1])


def spearman_free_eval(preds, trues, classes):
    """Plain accuracy + macro-F1 (sklearn not needed for a go/no-go read; kept minimal/self-contained)."""
    from collections import Counter
    n = len(trues)
    acc = sum(p == t for p, t in zip(preds, trues)) / n
    f1s = []
    for c in classes:
        tp = sum(p == c and t == c for p, t in zip(preds, trues))
        fp = sum(p == c and t != c for p, t in zip(preds, trues))
        fn = sum(p != c and t == c for p, t in zip(preds, trues))
        prec = tp / (tp + fp) if (tp + fp) else 0.0
        rec = tp / (tp + fn) if (tp + fn) else 0.0
        f1s.append(2 * prec * rec / (prec + rec) if (prec + rec) else 0.0)
    return acc, float(np.mean(f1s))


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    epochs = int(os.environ.get("CRICLENS_EPOCHS", "30"))
    seed = int(os.environ.get("CRICLENS_SEED", "0"))
    torch.manual_seed(seed); np.random.seed(seed)
    print(f"device={device}  epochs={epochs}  seed={seed}", flush=True)

    d = load_labelled_clips()
    classes = sorted(d.shot.unique())
    tr, va, te = (d[d.split == s] for s in ("train", "val", "test"))
    print(f"train={len(tr)} val={len(va)} test={len(te)} classes={classes}", flush=True)

    cache, failed = cache_sequences(d)
    print(f"cached {len(cache)} sequences ({len(failed)} failed)", flush=True)
    tr, va, te = (x[~x.clip_id.isin(failed)] for x in (tr, va, te))

    model = RecurrentClassifier(in_dim=51, n_classes=len(classes)).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)
    counts = tr.shot.value_counts().reindex(classes).to_numpy()
    weight = torch.tensor(counts.sum() / (len(classes) * counts), dtype=torch.float32, device=device)
    crit = nn.CrossEntropyLoss(weight=weight)

    dl_tr = DataLoader(SeqDataset(tr, classes, cache), batch_size=64, shuffle=True)
    dl_va = DataLoader(SeqDataset(va, classes, cache), batch_size=256)
    dl_te = DataLoader(SeqDataset(te, classes, cache), batch_size=256)

    best_f1, best_state, bad, patience = -1.0, None, 0, 6
    for ep in range(epochs):
        model.train(); t0 = time.time(); tot_loss = 0.0
        for xb, yb in dl_tr:
            xb, yb = xb.to(device), yb.to(device)
            opt.zero_grad(); loss = crit(model(xb), yb); loss.backward(); opt.step()
            tot_loss += loss.item() * len(yb)
        model.eval(); preds, trues = [], []
        with torch.no_grad():
            for xb, yb in dl_va:
                preds += model(xb.to(device)).argmax(1).cpu().tolist(); trues += yb.tolist()
        acc, f1 = spearman_free_eval(preds, trues, list(range(len(classes))))
        print(f"  epoch {ep+1}/{epochs}  train_loss={tot_loss/len(tr):.3f}  val_acc={acc:.3f}  "
              f"val_macroF1={f1:.3f}  ({time.time()-t0:.1f}s)", flush=True)
        if f1 > best_f1:
            best_f1, best_state, bad = f1, {k: v.cpu().clone() for k, v in model.state_dict().items()}, 0
        else:
            bad += 1
            if bad >= patience:
                print(f"  early stop (no val macro-F1 improvement in {patience} epochs)", flush=True)
                break

    model.load_state_dict(best_state); model.eval()
    preds, trues = [], []
    with torch.no_grad():
        for xb, yb in dl_te:
            preds += model(xb.to(device)).argmax(1).cpu().tolist(); trues += yb.tolist()
    acc, f1 = spearman_free_eval(preds, trues, list(range(len(classes))))
    print(f"\nTEST  acc={acc:.3f}  macro_f1={f1:.3f}", flush=True)

    torch.save(best_state, f"{OUT}/shot_gru_full.pt")
    torch.save(model.rnn.state_dict(), f"{OUT}/shot_gru_encoder.pt")
    summary = {"arch": "gru", "seed": seed, "best_val_macro_f1": best_f1, "test_accuracy": acc,
               "test_macro_f1": f1, "n_train": len(tr), "n_val": len(va), "n_test": len(te), "classes": classes}
    json.dump(summary, open(f"{OUT}/summary.json", "w"), indent=2)


if __name__ == "__main__":
    main()
