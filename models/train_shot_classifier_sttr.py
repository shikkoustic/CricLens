"""Shot classifier v2: divided spatio-temporal attention (ST-TR/Timesformer-inspired) + NAFNet-style
gated feed-forward, in place of the RNN/LSTM/GRU/Transformer family compared in train_shot_classifier.py.

Rationale (PROGRESS.md 2026-10-26, following a supervisor conversation): the supervisor's suggestion was
to try (a) Timesformer-style "divided" attention -- separate temporal attention (which frames matter, e.g.
ball-bat contact) and spatial/"object" attention (what the pose looks like at that frame) -- and (b)
NAFNet's trick of replacing standard nonlinear activations with a multiplicative gate for efficiency, plus
an "adaptive" (learned) version of that gate.

Literature check before building (see chat / docs/paper -- not yet written up as a formal doc): neither
piece is novel on its own. Divided spatial/temporal self-attention on SKELETON data specifically is
ST-TR (Plizzari et al., 2020/2021, arXiv:2008.07404) -- Spatial Self-Attention (SSA) over joints within a
frame, Temporal Self-Attention (TSA) over frames for one joint, in a two-stream, partly GCN-hybrid design
tuned for NTU-RGB+D's ~56k clips. NAFNet's fixed multiplicative gate (Chen et al., ECCV 2022,
arXiv:2204.04676) and its "adaptive"/learned counterpart (GLU variants -- GEGLU/SwiGLU, Shazeer 2020,
arXiv:2002.05202) are both established, not new. What's being tested here is the combination, applied to
this specific pose-based cricket shot task -- an application/combination claim, not a mechanism claim,
consistent with how the project's other novelty claims have been framed.

Architecture (deliberately simplified from ST-TR's 9-layer, two-stream, GCN-hybrid design -- that was
tuned for ~56k training clips; this task has ~11,133, about 5x less, so a much smaller single-stream
model is used to avoid overfitting):
  1. Per-joint linear embedding: (T=32, J=17, 3) -> (T, J, d_model), plus learned temporal and joint
     positional embeddings (broadcast-added).
  2. N blocks, each: Spatial Self-Attention (attention across the 17 joints, shared across frames) ->
     residual+norm -> Temporal Self-Attention (attention across the 32 frames, shared across joints) ->
     residual+norm -> gated feed-forward (NAFNet-style "simple" gate: split channels, multiply; or
     "adaptive" gate: SwiGLU-style learned gate) -> residual+norm.
  3. Mean-pool over all (T x J) tokens -> linear classification head.

Same train/val/test split, same class-balanced loss, same evaluation as train_shot_classifier.py, for
direct comparability to the existing RNN/LSTM/GRU/Transformer numbers (LSTM/Transformer ~0.785/0.766
macro-F1 tied-best; see docs/paper/results_phase3.md).

    python models/train_shot_classifier_sttr.py --gate simple --epochs 30
    python models/train_shot_classifier_sttr.py --gate adaptive --epochs 2 --limit 200   # fast smoke test
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
from torch.utils.data import DataLoader, Dataset

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from models.evaluate import evaluate_classification, format_report  # noqa: E402

HIP_L, HIP_R, SHO_L, SHO_R = 11, 12, 5, 6
OUT_DIR = ROOT / "models/shot_classifier_sttr"


def load_labelled_sequences(limit: int | None = None) -> pd.DataFrame:
    idx = pd.read_parquet(ROOT / "data/processed/pose_index.parquet")
    d = idx[idx.train_ready & (idx.shot != "other")][
        ["clip_id", "shot", "split", "source", "handedness", "kps_path"]]
    return d.head(limit) if limit else d


T_OUT = 32


def resample_clip(kps: np.ndarray, window: np.ndarray, fps: float, t_out: int = T_OUT) -> np.ndarray | None:
    """Same resampling as models/resample_sequences.py, inlined (this script reads kps.npz directly via
    kps_path rather than the prebuilt data/processed/sequences.parquet, for consistency with how the
    Kaggle mirror of this script has to work -- see kaggle/technique-scorer-partlocal/partlocal_train.py's
    docstring for why)."""
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


def normalise_tokens(seq: np.ndarray) -> np.ndarray:
    """Like train_shot_classifier.py's normalise, but does NOT flatten -- keeps (T, 17, 3) so each joint
    stays a separate token for the spatial attention to operate over."""
    xy = seq[:, :, :2]
    hip = np.nanmean(xy[:, [HIP_L, HIP_R]], axis=1, keepdims=True)
    centred = xy - hip
    sho = np.nanmean(xy[:, [SHO_L, SHO_R]], axis=1)
    scale = np.nanmedian(np.linalg.norm(sho - hip[:, 0], axis=-1))
    scale = scale if np.isfinite(scale) and scale > 1e-3 else 1.0
    out = seq.copy()
    out[:, :, :2] = np.nan_to_num(centred / scale, nan=0.0)
    out[:, :, 2] = np.nan_to_num(out[:, :, 2], nan=0.0)
    return out.astype(np.float32)  # (T, 17, 3)


def cache_sequences(df: pd.DataFrame) -> tuple[dict[str, np.ndarray], list[str]]:
    cache, failed = {}, []
    for r in df.itertuples():
        z = np.load(ROOT / r.kps_path)
        seq = resample_clip(z["kps"], z["window"], float(z["fps"]))
        if seq is None:
            failed.append(r.clip_id)
            continue
        cache[r.clip_id] = normalise_tokens(seq)
    return cache, failed


class SeqDataset(Dataset):
    def __init__(self, df: pd.DataFrame, classes: list[str], cache: dict[str, np.ndarray]):
        self.df = df.reset_index(drop=True)
        self.c2i = {c: i for i, c in enumerate(classes)}
        self.cache = cache

    def __len__(self):
        return len(self.df)

    def __getitem__(self, i):
        r = self.df.iloc[i]
        return torch.from_numpy(self.cache[r.clip_id]), self.c2i[r.shot]


class GatedFeedForward(nn.Module):
    """NAFNet-style ('simple') or GLU/SwiGLU-style ('adaptive') gated feed-forward, in place of a
    standard Linear->activation->Linear block. 'simple': split one projection's channels in half and
    multiply them (NAFNet, arXiv:2204.04676) -- no extra learned parameters in the gate itself.
    'adaptive': a second, separately-learned projection gates the first via SiLU (SwiGLU, Shazeer 2020,
    arXiv:2002.05202) -- the gate itself is learned, not just a fixed self-multiply."""

    def __init__(self, d_model: int, hidden: int, gate: str = "simple", dropout: float = 0.1):
        super().__init__()
        self.gate = gate
        if gate == "simple":
            self.proj_in = nn.Linear(d_model, hidden * 2)
        elif gate == "adaptive":
            self.proj_a = nn.Linear(d_model, hidden)
            self.proj_b = nn.Linear(d_model, hidden)
        else:
            raise ValueError(gate)
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
    """One ST-TR/Timesformer-style block: Spatial Self-Attention (across joints, shared over frames),
    then Temporal Self-Attention (across frames, shared over joints), then a gated feed-forward -- each
    with its own residual connection and pre-norm, standard Transformer-block convention."""

    def __init__(self, d_model: int, heads: int, ff_hidden: int, gate: str, dropout: float = 0.1):
        super().__init__()
        self.ln_s = nn.LayerNorm(d_model)
        self.ssa = nn.MultiheadAttention(d_model, heads, dropout=dropout, batch_first=True)
        self.ln_t = nn.LayerNorm(d_model)
        self.tsa = nn.MultiheadAttention(d_model, heads, dropout=dropout, batch_first=True)
        self.ln_f = nn.LayerNorm(d_model)
        self.ff = GatedFeedForward(d_model, ff_hidden, gate=gate, dropout=dropout)

    def forward(self, x):
        # x: (B, T, J, D)
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
    def __init__(self, n_classes: int, d_model: int = 64, heads: int = 4, ff_hidden: int = 128,
                 n_blocks: int = 3, n_joints: int = 17, t_len: int = T_OUT, dropout: float = 0.1,
                 gate: str = "simple"):
        super().__init__()
        self.embed = nn.Linear(3, d_model)
        self.pos_t = nn.Parameter(torch.randn(1, t_len, 1, d_model) * 0.02)
        self.pos_j = nn.Parameter(torch.randn(1, 1, n_joints, d_model) * 0.02)
        self.blocks = nn.ModuleList([DividedAttentionBlock(d_model, heads, ff_hidden, gate, dropout)
                                      for _ in range(n_blocks)])
        self.ln_out = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, n_classes)

    def forward(self, x):
        # x: (B, T, J, 3)
        h = self.embed(x) + self.pos_t + self.pos_j
        for blk in self.blocks:
            h = blk(h)
        h = self.ln_out(h).mean(dim=(1, 2))  # mean-pool over all (T x J) tokens
        return self.head(h)


def run(gate: str, epochs: int, limit: int | None, device: str, batch_size: int = 64, patience: int = 6,
        lr: float = 1e-3, seed: int = 0) -> dict:
    torch.manual_seed(seed); np.random.seed(seed)
    d = load_labelled_sequences(limit)
    classes = sorted(d.shot.unique())
    cache, failed = cache_sequences(d)
    d = d[~d.clip_id.isin(failed)]
    tr, va, te = (d[d.split == s] for s in ("train", "val", "test"))
    print(f"[gate={gate}] train={len(tr)} val={len(va)} test={len(te)} classes={classes} "
          f"({len(failed)} failed to resample)")

    model = PoseSTTRClassifier(len(classes), gate=gate).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    counts = tr.shot.value_counts().reindex(classes).to_numpy()
    weight = torch.tensor(counts.sum() / (len(classes) * counts), dtype=torch.float32, device=device)
    crit = nn.CrossEntropyLoss(weight=weight)

    dl_tr = DataLoader(SeqDataset(tr, classes, cache), batch_size=batch_size, shuffle=True)
    dl_va = DataLoader(SeqDataset(va, classes, cache), batch_size=256)
    dl_te = DataLoader(SeqDataset(te, classes, cache), batch_size=256)

    best_f1, best_state, bad_epochs = -1.0, None, 0
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
        vr = evaluate_classification(pd.DataFrame({"y": [classes[i] for i in trues],
                                                     "p": [classes[i] for i in preds]}), "y", "p")
        print(f"  [gate={gate}] epoch {ep+1}/{epochs}  train_loss={tot_loss/len(tr):.3f}  "
              f"val_acc={vr['accuracy']:.3f}  val_macroF1={vr['macro_f1']:.3f}  ({time.time()-t0:.1f}s)")
        if vr["macro_f1"] > best_f1:
            best_f1, best_state, bad_epochs = vr["macro_f1"], {k: v.cpu().clone() for k, v in model.state_dict().items()}, 0
        else:
            bad_epochs += 1
            if bad_epochs >= patience:
                print(f"  [gate={gate}] early stop (no val macro-F1 improvement in {patience} epochs)"); break

    model.load_state_dict(best_state); model.eval()
    preds, trues = [], []
    with torch.no_grad():
        for xb, yb in dl_te:
            preds += model(xb.to(device)).argmax(1).cpu().tolist(); trues += yb.tolist()
    te_r = te.assign(pred=[classes[i] for i in preds])
    result = evaluate_classification(te_r.assign(y=[classes[i] for i in trues]), "y", "pred", by="source")

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    torch.save(best_state, OUT_DIR / f"sttr_{gate}.pt")
    print(format_report(result, f"\n[gate={gate}] TEST"))
    return {"gate": gate, "best_val_macro_f1": best_f1, "test_accuracy": result["accuracy"],
            "test_macro_f1": result["macro_f1"], "test_weighted_f1": result["weighted_f1"],
            "n_train": len(tr), "n_val": len(va), "n_test": len(te), "classes": classes}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gate", default="both", choices=["simple", "adaptive", "both"])
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    gates = ["simple", "adaptive"] if args.gate == "both" else [args.gate]
    summary = [run(g, args.epochs, args.limit, args.device, seed=args.seed) for g in gates]
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    json.dump(summary, open(OUT_DIR / "summary.json", "w"), indent=2)
    print("\n=== summary ===")
    print(pd.DataFrame(summary)[["gate", "best_val_macro_f1", "test_accuracy", "test_macro_f1", "test_weighted_f1"]]
          .to_string(index=False))
    print("\nreference (existing architecture comparison, docs/paper/results_phase3.md): "
          "LSTM macro-F1 ~0.785, Transformer ~0.766, GRU (final-hidden readout) 0.777")


if __name__ == "__main__":
    main()
