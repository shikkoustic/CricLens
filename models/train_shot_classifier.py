"""Shot classifier: RNN vs LSTM vs GRU vs Transformer over the resampled pose sequences (PROGRESS.md step 3).

Input: data/processed/sequences.parquet + .npz (models/resample_sequences.py), joined to
data/processed/pose_index.parquet for the shot label and the match-grouped split. 8 shots trained
(the taxonomy's 'other' class is excluded, matching datasets/taxonomy.py); 'scoop' has only 96 clips
(PROGRESS.md) and is kept in training but its small-n metrics should be read separately, which
models/evaluate.py's per-class report already gives for free.

Spatial normalisation (deliberately left out of resample_sequences.py, done here instead): per frame,
centre on the hip midpoint (COCO joints 11, 12) so the model sees pose shape, not the batter's position
on screen; scale by a single per-clip robust size (median hip-to-shoulder-midpoint distance across the
window) so it isn't sensitive to camera distance or source resolution. Confidence (channel 2) is kept
un-normalised as a soft per-joint reliability signal.

Evaluation: models/evaluate.py on val (model selection) and test (final numbers), per-source and
per-handedness breakdowns, using the existing match-grouped, leakage-checked splits.

    python models/train_shot_classifier.py --arch all --epochs 30              # all 4 architectures
    python models/train_shot_classifier.py --arch lstm --epochs 2 --limit 200  # fast smoke test
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
OUT_DIR = ROOT / "models/shot_classifier"


def load_labelled_sequences(limit: int | None = None) -> pd.DataFrame:
    idx = pd.read_parquet(ROOT / "data/processed/pose_index.parquet")
    seq = pd.read_parquet(ROOT / "data/processed/sequences.parquet")
    d = idx[idx.train_ready & (idx.shot != "other")][["clip_id", "shot", "split", "source", "handedness"]]
    d = d.merge(seq[["clip_id", "seq_path"]], on="clip_id")
    return d.head(limit) if limit else d


def normalise(seq: np.ndarray) -> np.ndarray:
    """seq: (T, 17, 3). Per-frame hip-centred, per-clip scale-normalised (x, y); confidence untouched."""
    xy = seq[:, :, :2]
    hip = np.nanmean(xy[:, [HIP_L, HIP_R]], axis=1, keepdims=True)  # (T, 1, 2)
    centred = xy - hip
    sho = np.nanmean(xy[:, [SHO_L, SHO_R]], axis=1)  # (T, 2)
    scale = np.nanmedian(np.linalg.norm(sho - hip[:, 0], axis=-1))
    scale = scale if np.isfinite(scale) and scale > 1e-3 else 1.0
    out = seq.copy()
    out[:, :, :2] = np.nan_to_num(centred / scale, nan=0.0)
    out[:, :, 2] = np.nan_to_num(out[:, :, 2], nan=0.0)
    return out.reshape(seq.shape[0], -1).astype(np.float32)  # (T, 51)


class SeqDataset(Dataset):
    def __init__(self, df: pd.DataFrame, classes: list[str]):
        self.df = df.reset_index(drop=True)
        self.c2i = {c: i for i, c in enumerate(classes)}

    def __len__(self):
        return len(self.df)

    def __getitem__(self, i):
        r = self.df.iloc[i]
        seq = normalise(np.load(ROOT / r.seq_path)["seq"])
        return torch.from_numpy(seq), self.c2i[r.shot]


class RecurrentClassifier(nn.Module):
    def __init__(self, cell: str, in_dim: int, n_classes: int, hidden: int = 128, layers: int = 2):
        super().__init__()
        rnn_cls = {"rnn": nn.RNN, "lstm": nn.LSTM, "gru": nn.GRU}[cell]
        self.rnn = rnn_cls(in_dim, hidden, num_layers=layers, batch_first=True, dropout=0.2 if layers > 1 else 0)
        self.head = nn.Sequential(nn.LayerNorm(hidden), nn.Linear(hidden, n_classes))

    def forward(self, x):
        out, *_ = self.rnn(x)
        return self.head(out.mean(1))  # mean-pool over time: robust to which frame the model attends to


class TransformerClassifier(nn.Module):
    def __init__(self, in_dim: int, n_classes: int, d_model: int = 128, heads: int = 4, layers: int = 2, t_len: int = 32):
        super().__init__()
        self.proj = nn.Linear(in_dim, d_model)
        self.pos = nn.Parameter(torch.randn(1, t_len, d_model) * 0.02)
        enc = nn.TransformerEncoderLayer(d_model, heads, dim_feedforward=d_model * 2, dropout=0.2, batch_first=True)
        self.enc = nn.TransformerEncoder(enc, layers)
        self.head = nn.Sequential(nn.LayerNorm(d_model), nn.Linear(d_model, n_classes))

    def forward(self, x):
        h = self.enc(self.proj(x) + self.pos[:, : x.shape[1]])
        return self.head(h.mean(1))


def make_model(arch: str, in_dim: int, n_classes: int, t_len: int) -> nn.Module:
    return TransformerClassifier(in_dim, n_classes, t_len=t_len) if arch == "transformer" else \
        RecurrentClassifier(arch, in_dim, n_classes)


def run(arch: str, epochs: int, limit: int | None, device: str, batch_size: int = 64, patience: int = 6) -> dict:
    d = load_labelled_sequences(limit)
    classes = sorted(d.shot.unique())
    tr, va, te = (d[d.split == s] for s in ("train", "val", "test"))
    print(f"[{arch}] train={len(tr)} val={len(va)} test={len(te)} classes={classes}")

    z0 = np.load(ROOT / d.seq_path.iloc[0])
    t_len, in_dim = z0["seq"].shape[0], z0["seq"].shape[1] * z0["seq"].shape[2]
    model = make_model(arch, in_dim, len(classes), t_len).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)
    # class-balanced loss: shot counts range 96 (scoop) to 3402 (drive) -- 35x imbalance
    counts = tr.shot.value_counts().reindex(classes).to_numpy()
    weight = torch.tensor(counts.sum() / (len(classes) * counts), dtype=torch.float32, device=device)
    crit = nn.CrossEntropyLoss(weight=weight)

    dl_tr = DataLoader(SeqDataset(tr, classes), batch_size=batch_size, shuffle=True, num_workers=0)
    dl_va = DataLoader(SeqDataset(va, classes), batch_size=256, num_workers=0)

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
        vr = evaluate_classification(pd.DataFrame({"y": [classes[i] for i in trues], "p": [classes[i] for i in preds]}), "y", "p")
        print(f"  epoch {ep+1}/{epochs}  train_loss={tot_loss/len(tr):.3f}  val_acc={vr['accuracy']:.3f}  "
              f"val_macroF1={vr['macro_f1']:.3f}  ({time.time()-t0:.1f}s)")
        if vr["macro_f1"] > best_f1:
            best_f1, best_state, bad_epochs = vr["macro_f1"], {k: v.cpu().clone() for k, v in model.state_dict().items()}, 0
        else:
            bad_epochs += 1
            if bad_epochs >= patience:
                print(f"  early stop (no val macro-F1 improvement in {patience} epochs)"); break

    model.load_state_dict(best_state); model.eval()
    dl_te = DataLoader(SeqDataset(te, classes), batch_size=256, num_workers=0)
    preds, trues = [], []
    with torch.no_grad():
        for xb, yb in dl_te:
            preds += model(xb.to(device)).argmax(1).cpu().tolist(); trues += yb.tolist()
    te = te.assign(pred=[classes[i] for i in preds])
    result = evaluate_classification(te.assign(y=[classes[i] for i in trues]), "y", "pred", by="source")
    result_h = evaluate_classification(te.dropna(subset=["handedness"]), "shot", "pred", by="handedness") \
        if "handedness" in te and te.handedness.notna().any() else None

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    torch.save(best_state, OUT_DIR / f"{arch}.pt")
    print(format_report(result, f"\n[{arch}] TEST"))
    if result_h:
        print(format_report(result_h, f"[{arch}] TEST by handedness"))
    return {"arch": arch, "best_val_macro_f1": best_f1, "test_accuracy": result["accuracy"],
            "test_macro_f1": result["macro_f1"], "test_weighted_f1": result["weighted_f1"],
            "n_train": len(tr), "n_val": len(dl_va.dataset), "n_test": len(te), "classes": classes}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arch", default="all", choices=["rnn", "lstm", "gru", "transformer", "all"])
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--limit", type=int, default=None, help="cap total clips, for a fast smoke test")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()
    archs = ["rnn", "lstm", "gru", "transformer"] if args.arch == "all" else [args.arch]
    summary = [run(a, args.epochs, args.limit, args.device) for a in archs]
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    json.dump(summary, open(OUT_DIR / "summary.json", "w"), indent=2)
    print("\n=== summary ===")
    print(pd.DataFrame(summary)[["arch", "best_val_macro_f1", "test_accuracy", "test_macro_f1", "test_weighted_f1"]]
          .to_string(index=False))


if __name__ == "__main__":
    main()
