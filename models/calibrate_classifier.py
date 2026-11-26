"""Fit a temperature for the shot classifier, and pick the point below which the app should
say it isn't sure.

The app prints a confidence beside every call. Measured on the test split the raw softmax is
overconfident by ~5 points (ECE 0.047 over the five strokes the app offers): a clip shown at
75% is right about 65% of the time. A number that reads as a guarantee and isn't one is worse
than no number, so this fits the standard one-parameter fix (Guo et al., 2017, "On Calibration
of Modern Neural Networks") on the validation split and measures what it buys on test.

Two things are fitted, both on val, never on test:
  temperature  -- one scalar dividing the logits before the softmax
  abstain_below -- the confidence under which the app stops naming one stroke and offers two

Calibration is done in the space the app actually shows, which is the softmax over the five
strokes in app.shots.SCORED, not over all eight: masking changes the denominator, so a
temperature fitted on the 8-way distribution would be the wrong correction for what the user
reads.

    python models/calibrate_classifier.py          # writes app/calibration.json

Writes app/calibration.json; app/pipeline.py applies it when the file is present.
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from app.shots import SCORED, SCORED_IDX, SHOTS  # noqa: E402
from models.train_shot_classifier import make_model, normalise  # noqa: E402

BINS = 10


def logits_for(split, clf, seq, idx):
    d = idx[(idx.split == split) & idx.train_ready & idx.shot.isin(SCORED)
            & idx.clip_id.isin(seq.index)]
    L, Y = [], []
    with torch.no_grad():
        for r in d.itertuples():
            x = torch.from_numpy(normalise(np.load(ROOT / seq.loc[r.clip_id, "seq_path"])["seq"]))[None]
            L.append(clf(x)[0].numpy()[SCORED_IDX])
            Y.append(SCORED.index(r.shot))
    return np.array(L, dtype=np.float64), np.array(Y)


def softmax(z):
    e = np.exp(z - z.max(axis=1, keepdims=True))
    return e / e.sum(axis=1, keepdims=True)


def ece(P, Y, bins=BINS):
    conf, pred = P.max(1), P.argmax(1)
    acc = (pred == Y).astype(float)
    out, rows = 0.0, []
    for i in range(bins):
        lo, hi = i / bins, (i + 1) / bins
        m = (conf > lo) & (conf <= hi)
        if not m.sum():
            continue
        out += m.mean() * abs(acc[m].mean() - conf[m].mean())
        rows.append((lo, hi, int(m.sum()), float(conf[m].mean()), float(acc[m].mean())))
    return out, rows


def fit_temperature(L, Y):
    """Minimise NLL over log T, so T stays positive without a constrained optimiser."""
    logt = torch.zeros(1, requires_grad=True)
    z, y = torch.tensor(L), torch.tensor(Y)
    opt = torch.optim.LBFGS([logt], lr=0.1, max_iter=200)
    nll = torch.nn.CrossEntropyLoss()

    def step():
        opt.zero_grad()
        loss = nll(z / torch.exp(logt), y)
        loss.backward()
        return loss
    opt.step(step)
    return float(torch.exp(logt).item())


def main():
    clf = make_model("lstm", 51, len(SHOTS), 32)
    clf.load_state_dict(torch.load(ROOT / "models/shot_classifier/lstm.pt", map_location="cpu"))
    clf.eval()
    seq = pd.read_parquet(ROOT / "data/processed/sequences.parquet").set_index("clip_id")
    idx = pd.read_parquet(ROOT / "data/processed/pose_index.parquet")

    Lv, Yv = logits_for("val", clf, seq, idx)
    Lt, Yt = logits_for("test", clf, seq, idx)
    print(f"val {len(Yv)} clips, test {len(Yt)} clips (the five strokes the app offers)\n")

    T = fit_temperature(Lv, Yv)
    print(f"fitted temperature T = {T:.3f}  (>1 means the raw model was overconfident)\n")

    for name, P in (("before", softmax(Lt)), ("after", softmax(Lt / T))):
        e, rows = ece(P, Yt)
        acc, conf = (P.argmax(1) == Yt).mean(), P.max(1).mean()
        print(f"test, {name:>6}:  accuracy {acc:.3f}   mean confidence {conf:.3f}   "
              f"ECE {e:.3f}   gap {conf - acc:+.3f}")
    print()

    # ---- where should the app stop naming one stroke? -----------------------------------------
    # Chosen on val: the lowest threshold whose *accepted* clips are right at least TARGET of the
    # time, so the number the app prints is one it can stand behind, and coverage is the cost.
    TARGET = 0.85
    Pv = softmax(Lv / T)
    cv, okv = Pv.max(1), (Pv.argmax(1) == Yv)
    print(f"{'threshold':>10} {'coverage':>9} {'accuracy when it answers':>26}   (val)")
    chosen = None
    for th in np.arange(0.40, 0.96, 0.05):
        m = cv >= th
        if not m.sum():
            continue
        a = okv[m].mean()
        print(f"{th:>10.2f} {m.mean():>9.1%} {a:>26.3f}")
        if chosen is None and a >= TARGET:
            chosen = float(round(th, 2))
    if chosen is None:
        chosen = 0.60
        print(f"\nno threshold reached {TARGET:.0%} on val; falling back to {chosen}")
    else:
        print(f"\nabstain below {chosen}: the first point where val accuracy clears {TARGET:.0%}")

    Pt = softmax(Lt / T)
    ct, okt = Pt.max(1), (Pt.argmax(1) == Yt)
    m = ct >= chosen
    print(f"on test: answers on {m.mean():.1%} of clips, right {okt[m].mean():.3f} of the time "
          f"(vs {okt.mean():.3f} if it always answers)")

    out = {"model": "lstm", "classes": list(SCORED), "temperature": round(T, 4),
           "abstain_below": chosen, "fitted_on": "val", "n_val": int(len(Yv)), "n_test": int(len(Yt)),
           "test_ece_before": round(ece(softmax(Lt), Yt)[0], 4),
           "test_ece_after": round(ece(Pt, Yt)[0], 4),
           "test_coverage": round(float(m.mean()), 4),
           "test_accuracy_when_answering": round(float(okt[m].mean()), 4),
           "reliability_after": [{"lo": r[0], "hi": r[1], "n": r[2], "conf": round(r[3], 4),
                                  "acc": round(r[4], 4)} for r in ece(Pt, Yt)[1]]}
    (ROOT / "app/calibration.json").write_text(json.dumps(out, indent=1) + "\n")
    print("\nwrote app/calibration.json")


if __name__ == "__main__":
    main()
