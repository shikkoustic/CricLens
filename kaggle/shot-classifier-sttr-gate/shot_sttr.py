"""Shot classifier v2: divided spatio-temporal attention (ST-TR/Timesformer-inspired) + NAFNet-style
gated feed-forward -- Kaggle GPU run (self-contained, no repo imports; mirrors
models/train_shot_classifier_sttr.py exactly). See that script's docstring for the full rationale and
literature grounding (ST-TR arXiv:2008.07404, NAFNet arXiv:2204.04676, GLU variants arXiv:2002.05202).

Runs both gate variants (simple = NAFNet's fixed multiplicative gate; adaptive = SwiGLU-style learned
gate) back to back in one job, on ALL 16,460 train_ready clips (all 6 sources) -- the shot-classification
task's full pool, not the small CricketVision-scored subset used for technique scoring.

Inputs : shikkoustic/criclens-all (criclens-processed.tgz.bin -> data/processed/pose_index.parquet +
          kaggle/chunks/pose-*/out/kps/<clip>.npz; sequences resampled on the fly, same pattern as
          kaggle/technique-scorer-partlocal/partlocal_train.py)
Outputs: sttr_<gate>.pt, summary.json
"""
import glob
import json
import os
import subprocess
import time

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


def normalise_tokens(seq):
    xy = seq[:, :, :2]
    hip = np.nanmean(xy[:, [HIP_L, HIP_R]], axis=1, keepdims=True)
    centred = xy - hip
    sho = np.nanmean(xy[:, [SHO_L, SHO_R]], axis=1)
    scale = np.nanmedian(np.linalg.norm(sho - hip[:, 0], axis=-1))
    scale = scale if np.isfinite(scale) and scale > 1e-3 else 1.0
    out = seq.copy()
    out[:, :, :2] = np.nan_to_num(centred / scale, nan=0.0)
    out[:, :, 2] = np.nan_to_num(out[:, :, 2], nan=0.0)
    return out.astype(np.float32)


def load_labelled_clips(limit=None):
    idx = pd.read_parquet(f"{PROC}/data/processed/pose_index.parquet")
    d = idx[idx.train_ready & (idx.shot != "other")][
        ["clip_id", "shot", "split", "source", "handedness", "kps_path"]]
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
        cache[r.clip_id] = normalise_tokens(seq)
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


class GatedFeedForward(nn.Module):
    def __init__(self, d_model, hidden, gate="simple", dropout=0.1):
        super().__init__()
        self.gate = gate
        if gate == "simple":
            self.proj_in = nn.Linear(d_model, hidden * 2)
        else:
            self.proj_a = nn.Linear(d_model, hidden)
            self.proj_b = nn.Linear(d_model, hidden)
        self.proj_out = nn.Linear(hidden, d_model)
        self.drop = nn.Dropout(dropout)

    def forward(self, x):
        if self.gate == "simple":
            a, b = self.proj_in(x).chunk(2, dim=-1)
            h = a * b
        else:
            h = self.proj_a(x) * nn.functional.silu(self.proj_b(x))
        return self.proj_out(self.drop(h))


class DividedAttentionBlock(nn.Module):
    def __init__(self, d_model, heads, ff_hidden, gate, dropout=0.1):
        super().__init__()
        self.ln_s = nn.LayerNorm(d_model)
        self.ssa = nn.MultiheadAttention(d_model, heads, dropout=dropout, batch_first=True)
        self.ln_t = nn.LayerNorm(d_model)
        self.tsa = nn.MultiheadAttention(d_model, heads, dropout=dropout, batch_first=True)
        self.ln_f = nn.LayerNorm(d_model)
        self.ff = GatedFeedForward(d_model, ff_hidden, gate=gate, dropout=dropout)

    def forward(self, x):
        b, t, j, d = x.shape
        xs = self.ln_s(x).reshape(b * t, j, d)
        attn_s, _ = self.ssa(xs, xs, xs)
        x = x + attn_s.reshape(b, t, j, d)

        xt = self.ln_t(x).permute(0, 2, 1, 3).reshape(b * j, t, d)
        attn_t, _ = self.tsa(xt, xt, xt)
        x = x + attn_t.reshape(b, j, t, d).permute(0, 2, 1, 3)

        x = x + self.ff(self.ln_f(x))
        return x


class PoseSTTRClassifier(nn.Module):
    def __init__(self, n_classes, d_model=64, heads=4, ff_hidden=128, n_blocks=3, n_joints=17,
                 t_len=T_OUT, dropout=0.1, gate="simple"):
        super().__init__()
        self.embed = nn.Linear(3, d_model)
        self.pos_t = nn.Parameter(torch.randn(1, t_len, 1, d_model) * 0.02)
        self.pos_j = nn.Parameter(torch.randn(1, 1, n_joints, d_model) * 0.02)
        self.blocks = nn.ModuleList([DividedAttentionBlock(d_model, heads, ff_hidden, gate, dropout)
                                      for _ in range(n_blocks)])
        self.ln_out = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, n_classes)

    def forward(self, x):
        h = self.embed(x) + self.pos_t + self.pos_j
        for blk in self.blocks:
            h = blk(h)
        h = self.ln_out(h).mean(dim=(1, 2))
        return self.head(h)


def classify_eval(preds, trues, classes):
    n = len(trues)
    acc = sum(p == t for p, t in zip(preds, trues)) / n
    f1s, supports = [], []
    for c in range(len(classes)):
        tp = sum(p == c and t == c for p, t in zip(preds, trues))
        fp = sum(p == c and t != c for p, t in zip(preds, trues))
        fn = sum(p != c and t == c for p, t in zip(preds, trues))
        sup = sum(t == c for t in trues)
        prec = tp / (tp + fp) if (tp + fp) else 0.0
        rec = tp / (tp + fn) if (tp + fn) else 0.0
        f1 = 2 * prec * rec / (prec + rec) if (prec + rec) else 0.0
        f1s.append(f1); supports.append(sup)
    macro_f1 = float(np.mean(f1s))
    weighted_f1 = float(np.average(f1s, weights=supports)) if sum(supports) else 0.0
    return acc, macro_f1, weighted_f1


def run(gate, epochs, device, d, cache, classes, batch_size=64, patience=6, lr=1e-3, seed=0):
    torch.manual_seed(seed); np.random.seed(seed)
    tr, va, te = (d[d.split == s] for s in ("train", "val", "test"))
    print(f"[gate={gate}, seed={seed}] train={len(tr)} val={len(va)} test={len(te)}", flush=True)

    model = PoseSTTRClassifier(len(classes), gate=gate).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    counts = tr.shot.value_counts().reindex(classes).to_numpy()
    weight = torch.tensor(counts.sum() / (len(classes) * counts), dtype=torch.float32, device=device)
    crit = nn.CrossEntropyLoss(weight=weight)

    dl_tr = DataLoader(SeqDataset(tr, classes, cache), batch_size=batch_size, shuffle=True)
    dl_va = DataLoader(SeqDataset(va, classes, cache), batch_size=256)
    dl_te = DataLoader(SeqDataset(te, classes, cache), batch_size=256)

    best_f1, best_state, bad = -1.0, None, 0
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
        acc, macro_f1, weighted_f1 = classify_eval(preds, trues, classes)
        print(f"  [gate={gate}, seed={seed}] epoch {ep+1}/{epochs}  train_loss={tot_loss/len(tr):.3f}  "
              f"val_acc={acc:.3f}  val_macroF1={macro_f1:.3f}  ({time.time()-t0:.1f}s)", flush=True)
        if macro_f1 > best_f1:
            best_f1, best_state, bad = macro_f1, {k: v.cpu().clone() for k, v in model.state_dict().items()}, 0
        else:
            bad += 1
            if bad >= patience:
                print(f"  [gate={gate}, seed={seed}] early stop (no val macro-F1 improvement in {patience} epochs)", flush=True)
                break

    model.load_state_dict(best_state); model.eval()
    preds, trues = [], []
    with torch.no_grad():
        for xb, yb in dl_te:
            preds += model(xb.to(device)).argmax(1).cpu().tolist(); trues += yb.tolist()
    acc, macro_f1, weighted_f1 = classify_eval(preds, trues, classes)
    print(f"\n[gate={gate}, seed={seed}] TEST  acc={acc:.3f}  macro_f1={macro_f1:.3f}  weighted_f1={weighted_f1:.3f}", flush=True)

    torch.save(best_state, f"{OUT}/sttr_{gate}_seed{seed}.pt")
    return {"gate": gate, "seed": seed, "best_val_macro_f1": best_f1, "test_accuracy": acc,
            "test_macro_f1": macro_f1, "test_weighted_f1": weighted_f1,
            "n_train": len(tr), "n_val": len(va), "n_test": len(te), "classes": classes}


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    epochs = int(os.environ.get("CRICLENS_EPOCHS", "30"))
    gates = os.environ.get("CRICLENS_GATES", "simple,adaptive").split(",")
    seeds = [int(s) for s in os.environ.get("CRICLENS_SEEDS", "0,1,2").split(",")]
    print(f"device={device}  epochs={epochs}  gates={gates}  seeds={seeds}", flush=True)

    d_all = load_labelled_clips()
    classes = sorted(d_all.shot.unique())
    cache, failed = cache_sequences(d_all)
    d_all = d_all[~d_all.clip_id.isin(failed)]
    print(f"cached {len(cache)} sequences ({len(failed)} failed), classes={classes}", flush=True)

    summary = [run(g, epochs, device, d_all, cache, classes, seed=s) for g in gates for s in seeds]
    json.dump(summary, open(f"{OUT}/summary.json", "w"), indent=2)

    agg = {}
    for g in gates:
        rows = [s for s in summary if s["gate"] == g]
        agg[g] = {"test_macro_f1": {"mean": float(np.mean([r["test_macro_f1"] for r in rows])),
                                     "std": float(np.std([r["test_macro_f1"] for r in rows]))},
                   "test_accuracy": {"mean": float(np.mean([r["test_accuracy"] for r in rows])),
                                      "std": float(np.std([r["test_accuracy"] for r in rows]))}}
    json.dump(agg, open(f"{OUT}/summary_agg.json", "w"), indent=2)
    print("\n=== AGGREGATE (3 seeds per gate) ===", flush=True)
    for g, a in agg.items():
        print(f"[{g}] test_macroF1={a['test_macro_f1']['mean']:.3f} +/- {a['test_macro_f1']['std']:.3f}  "
              f"test_acc={a['test_accuracy']['mean']:.3f} +/- {a['test_accuracy']['std']:.3f}", flush=True)
    print("\nreference (existing architecture comparison): LSTM macro-F1 ~0.785, Transformer ~0.766, "
          "GRU (final-hidden readout) 0.777", flush=True)


if __name__ == "__main__":
    main()
