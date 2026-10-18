"""Left/right handedness bias audit (PROGRESS.md step 3).

CricketVision's scored clips are 3,134 right-handed / 1,705 left-handed (docs/DATASHEET.md), and the
full pose_index carries handedness only for that subset (models/evaluate.py's by="handedness" breakdown
already reports each trained model's raw performance gap between the two -- see the training scripts'
test output). This script asks the sharper question: is a performance gap, if any, because the model
has learned right-hander-specific patterns that don't transfer to mirror-image left-handers, or because
there's simply less left-handed data?

Two tests, neither requiring a full retrain:
  1. TEST-TIME MIRRORING. Flip left-handed test clips to look like right-handers (mirror_sequence) and
     re-run the already-trained model on the flipped version. If accuracy/Spearman on left-handers
     improves after mirroring, the model is right-hander-biased -- it does better on a mirrored left-hander
     than on the real thing, which means what it learned doesn't generalise across the mirror symmetry
     that batting actually has (a cover drive and its mirror image are the same shot).
  2. CONSISTENCY CHECK. Mirror RIGHT-handed test clips (the model's presumably-strong side) and check
     performance drops -- this confirms the model is actually using handedness-specific visual pattern,
     not something invariant to it already (if performance doesn't drop, the model may already be
     roughly mirror-invariant and the audit in (1) would need a different explanation for any gap found).

Mirroring a COCO pose: negate the (already hip-centred) x-coordinate, then swap each left/right joint
pair (the nose, index 0, has no pair and is just negated). Unit-tested below: mirroring twice must be
the identity, and mirroring must move the joint that was at "left shoulder" to the "right shoulder" slot.

    python models/handedness_audit.py                    # requires trained models (train_shot_classifier.py,
                                                           # train_technique_scorer.py) to have run first
    python -c "from models.handedness_audit import _test_mirror; _test_mirror()"   # unit test only
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from models.evaluate import evaluate_classification, evaluate_rating, format_report  # noqa: E402
from models.train_shot_classifier import normalise, make_model, load_labelled_sequences  # noqa: E402
from models.train_technique_scorer import VAERegressor, load_scored_sequences, PARTS, TARGETS  # noqa: E402

# COCO-17: 0=nose (no pair), then 8 left/right pairs
LR_PAIRS = [(1, 2), (3, 4), (5, 6), (7, 8), (9, 10), (11, 12), (13, 14), (15, 16)]


def mirror_sequence(seq: np.ndarray) -> np.ndarray:
    """seq: (T, 17, 3) raw (x, y, confidence), NOT yet spatially normalised. Horizontal flip: negate x
    around the per-frame hip midpoint (so it doesn't need frame width), then swap left/right joint pairs."""
    out = seq.copy()
    hip_mid_x = np.nanmean(seq[:, [11, 12], 0], axis=1, keepdims=True)  # (T, 1)
    out[:, :, 0] = 2 * hip_mid_x - seq[:, :, 0]  # reflect x around the hip midpoint
    for l, r in LR_PAIRS:
        out[:, [l, r]] = out[:, [r, l]]
    return out


def _test_mirror():
    rng = np.random.default_rng(0)
    seq = rng.normal(0, 50, (5, 17, 3))
    seq[:, :, 2] = rng.uniform(0.3, 1.0, (5, 17))  # confidence stays positive-ish, not that it matters here
    m = mirror_sequence(seq)
    m2 = mirror_sequence(m)
    assert np.allclose(seq, m2, atol=1e-6), "mirroring twice must be the identity"
    # the joint at the "left shoulder" (5) slot after mirroring must equal the ORIGINAL right shoulder (6),
    # reflected in x, with y and confidence untouched
    hip_mid_x = np.nanmean(seq[:, [11, 12], 0], axis=1)
    expected_x = 2 * hip_mid_x - seq[:, 6, 0]
    assert np.allclose(m[:, 5, 0], expected_x, atol=1e-6), "left slot after mirror should hold the reflected right joint"
    assert np.allclose(m[:, 5, 1], seq[:, 6, 1], atol=1e-6)
    assert np.allclose(m[:, 5, 2], seq[:, 6, 2], atol=1e-6)
    # nose (0, no pair) should just be reflected in place
    assert np.allclose(m[:, 0, 1], seq[:, 0, 1], atol=1e-6)
    print("mirror_sequence: unit tests OK")


def _predict_shots(model, df, classes, mirror=False, device="cpu"):
    model.eval(); preds = []
    with torch.no_grad():
        for r in df.itertuples():
            raw = np.load(ROOT / r.seq_path)["seq"]
            if mirror:
                raw = mirror_sequence(raw)
            x = torch.from_numpy(normalise(raw)).unsqueeze(0).to(device)
            preds.append(classes[model(x).argmax(1).item()])
    return preds


def _predict_scores(model, df, mu, sd, mirror=False, device="cpu"):
    model.eval(); preds = []
    with torch.no_grad():
        for r in df.itertuples():
            raw = np.load(ROOT / r.seq_path)["seq"]
            if mirror:
                raw = mirror_sequence(raw)
            x = torch.from_numpy(normalise(raw)).unsqueeze(0).to(device)
            _, mvu, _, _ = model(x)
            sc = model.reg_head(mvu).cpu().numpy()[0]
            preds.append((sc * sd) + mu)
    return np.array(preds)


def audit_shot_classifier(arch: str = "gru", device: str = "cpu") -> None:
    d = load_labelled_sequences()
    classes = sorted(d.shot.unique())
    te = d[(d.split == "test") & d.handedness.notna()].copy()
    if te.empty:
        print("no handedness-labelled test clips for the shot classifier -- skipping"); return
    z0 = np.load(ROOT / d.seq_path.iloc[0])
    in_dim, t_len = z0["seq"].shape[1] * z0["seq"].shape[2], z0["seq"].shape[0]
    model = make_model(arch, in_dim, len(classes), t_len).to(device)
    state_path = ROOT / f"models/shot_classifier/{arch}.pt"
    if not state_path.exists():
        print(f"no trained {arch} classifier at {state_path} -- run train_shot_classifier.py first"); return
    model.load_state_dict(torch.load(state_path, map_location=device))

    te["pred_normal"] = _predict_shots(model, te, classes, mirror=False, device=device)
    te["pred_mirrored"] = _predict_shots(model, te, classes, mirror=True, device=device)

    print(f"\n=== Shot classifier ({arch}) handedness audit, n={len(te)} ===")
    print(format_report(evaluate_classification(te, "shot", "pred_normal", by="handedness"), "BASELINE (no mirroring)"))
    for hand in ("Left", "Right"):
        sub = te[te.handedness == hand]
        if sub.empty:
            continue
        r_norm = evaluate_classification(sub, "shot", "pred_normal")
        r_mirr = evaluate_classification(sub, "shot", "pred_mirrored")
        print(f"\n{hand}-handed (n={len(sub)}): normal acc={r_norm['accuracy']:.3f} macroF1={r_norm['macro_f1']:.3f}"
              f"  |  mirrored acc={r_mirr['accuracy']:.3f} macroF1={r_mirr['macro_f1']:.3f}"
              f"  |  mirroring {'HELPS' if r_mirr['accuracy'] > r_norm['accuracy'] else 'hurts/no change'}")


def audit_technique_scorer(device: str = "cpu") -> None:
    d, mu, sd = load_scored_sequences()
    te = d[(d.split == "test") & d.handedness.notna()].copy()
    if te.empty:
        print("no handedness-labelled test clips for the technique scorer -- skipping"); return
    z0 = np.load(ROOT / d.seq_path.iloc[0])
    in_dim, t_len = z0["seq"].shape[1] * z0["seq"].shape[2], z0["seq"].shape[0]
    model = VAERegressor(in_dim, t_len, len(TARGETS)).to(device)
    state_path = ROOT / "models/technique_scorer/vae_regressor.pt"
    if not state_path.exists():
        print(f"no trained scorer at {state_path} -- run train_technique_scorer.py first"); return
    model.load_state_dict(torch.load(state_path, map_location=device))

    pred_n = _predict_scores(model, te, mu, sd, mirror=False, device=device)
    pred_m = _predict_scores(model, te, mu, sd, mirror=True, device=device)
    for i, p in enumerate(TARGETS):
        te[f"pred_normal_{p}"] = pred_n[:, i]; te[f"pred_mirrored_{p}"] = pred_m[:, i]

    true_cols = {p: f"score_{p}" for p in PARTS}
    print(f"\n=== Technique scorer handedness audit, n={len(te)} ===")
    print(format_report(evaluate_rating(te, true_cols, {p: f"pred_normal_{p}" for p in PARTS}, by="handedness",
                                         overall_cols=("score_overall", "pred_normal_overall")), "BASELINE"))
    for hand in ("Left", "Right"):
        sub = te[te.handedness == hand]
        if sub.empty:
            continue
        r_norm = evaluate_rating(sub, true_cols, {p: f"pred_normal_{p}" for p in PARTS})
        r_mirr = evaluate_rating(sub, true_cols, {p: f"pred_mirrored_{p}" for p in PARTS})
        print(f"\n{hand}-handed (n={len(sub)}): normal rho={r_norm['mean_spearman']:.3f}"
              f"  |  mirrored rho={r_mirr['mean_spearman']:.3f}"
              f"  |  mirroring {'HELPS' if r_mirr['mean_spearman'] > r_norm['mean_spearman'] else 'hurts/no change'}")


if __name__ == "__main__":
    _test_mirror()
    audit_shot_classifier("gru")
    audit_technique_scorer()
