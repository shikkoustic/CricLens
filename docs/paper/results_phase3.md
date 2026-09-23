# Phase 3 results: shot classifier, technique scorer, handedness audit

**Status: training in progress (2026-09-23).** This file is filled in as each run completes.
Code: `models/train_shot_classifier.py`, `models/train_technique_scorer.py`, `models/handedness_audit.py`.
Evaluated with `models/evaluate.py` on the existing match-grouped, leakage-checked splits
(`check_split_integrity` passes: 3,618 groups checked, 0 leaking).

## Setup common to both models
- Input: `data/processed/sequences.parquet` -- every clip's analysis window resampled to T=32 frames
  over normalised time (`models/resample_sequences.py`).
- Spatial normalisation (`models/train_shot_classifier.py::normalise`): per-frame hip-centred (COCO
  joints 11/12), per-clip scale by median shoulder-hip distance. Confidence channel kept, un-normalised.
- **Trained on data from before the `kaggle/pose-pad` zero-padding re-run.** `pose-pad-c0` (5,136 clips)
  finished 2026-09-23: usability essentially unchanged (91.65% -> 91.67%, +1 clip, 0 lost), mean window
  confidence improved slightly (0.8306 -> 0.8336, improved for 76% of clips), bone-length-consistency
  plausibility check improved for exactly half of clips (a wash there). Small but real, no regressions --
  as predicted from the earlier severity analysis (76% of clamped frames lose only 5-10% of the padded
  box). `pose-pad-c1` is now running; once both finish, `pose_index.parquet` gets rebuilt and both models
  above will be re-trained on the corrected joints for the final numbers.

## 1. Shot classifier (RNN vs LSTM vs GRU vs Transformer)
8 shots (taxonomy's `other` excluded, `scoop` kept at n=96/70/11/15 total/train/val/test -- reported
separately per PROGRESS.md). Class-balanced loss (35x count imbalance, drive 3,402 vs scoop 96).

<!-- TABLE: arch | best_val_macro_f1 | test_accuracy | test_macro_f1 | test_weighted_f1 -->

**Comparison points** (docs/paper/reading_log.md): Kang's honest EfficientNet-B0+GRU on CricShot10
(random split) = 92.25%; CricShotNet on CricShot10k (random split) = 89%. Ours is pose-only, multi-source,
on match-grouped (non-leaking) splits -- a harder, more honest setting, so a materially lower number here
is expected and is itself evidence for the leakage argument (docs/paper/dossier.md A3), not a weakness.

## 2. Technique scorer (GRU-VAE + regression) -- DONE, first pass

Trained on the 4,872 scored train-ready clips (3,718 / 657 / 497 train/val/test). Predicts all 6 targets
(5 parts + overall) jointly. Early-stopped at epoch 19/60 (no val mean-Spearman improvement in 8 epochs).

| part | n | Spearman | R-l2 | partial-rho (vs overall) |
|---|---|---|---|---|
| head | 497 | 0.576 | 2.81 | **0.052** |
| shoulder | 497 | 0.575 | 2.95 | **-0.011** |
| hands | 497 | 0.576 | 3.05 | **-0.009** |
| hips | 497 | 0.558 | 3.01 | **0.062** |
| feet | 497 | 0.570 | 2.82 | **0.010** |
| **mean** | | **0.571** | **2.93** | **0.021** |

`score_overall` (predicted vs true) Spearman: **0.577** -- I3D-AE-LSTM's headline number is 0.84, so this
first pass trails it. Read that gap with real caveats: this run used un-tuned loss weights
(kl_weight=0.01, reg_weight=5.0, no sweep), a single architecture (GRU only, no comparison), only 19
epochs before early stopping, joints from *before* the `kaggle/pose-pad` zero-padding correction, and a
match-grouped test split (I3D-AE-LSTM's was "through trial and error", i.e. random, on data we
independently found contains 1,156 duplicated strokes -- so some of the gap is very plausibly leakage in
the comparison point, not a weaker model). Further tuning is the obvious next step before treating 0.577
as a final number.

**The `partial_spearman_vs_overall` result is the one that matters most, and it needed no tuning to be
clean: mean 0.021 -- indistinguishable from zero.** Four of five parts sit in [-0.011, 0.062], all
consistent with noise at n=497. This is the model-side confirmation of
`docs/paper/finding_label_collinearity.md`'s label-side finding: even after training explicitly to predict
all five parts jointly, **the model shows no evidence of independently discriminating any body part from
overall shot quality.** During training, raw val Spearman rose steadily (0.48 -> ~0.60 by epoch 11) while
val partial Spearman stayed flat in [0.00, 0.05] throughout all 19 epochs -- the model learned to track
overall quality and never picked up part-specific signal, exactly as the label collinearity predicts.
This is now confirmed on two independent grounds (labels themselves, and a real trained model) rather than
inferred from a third party's aggregate numbers.

## 3. Handedness bias audit

Test-time mirroring (`models/handedness_audit.py::mirror_sequence`, unit-tested: mirroring twice is the
identity, left/right joint pairs correctly swapped) on left-handed test clips, using the already-trained
models -- no retraining required for this first pass.

### Technique scorer -- DONE

| | n | normal mean Spearman | mirrored mean Spearman | mirroring... |
|---|---|---|---|---|
| Right-handed | 375 | 0.574 | 0.580 | **helps, barely** (+0.006, noise-level) |
| Left-handed | 119 | 0.562 | 0.508 | **hurts** (-0.054) |

**Not what the hypothesis predicted, and worth reporting as-is.** The pre-registered hypothesis was: if
the model is right-hander-biased, mirroring a left-hander to look right-handed should recover some of the
gap. Instead mirroring makes left-handers *worse*, while barely moving right-handers. Two things follow:

1. **The raw gap is small to begin with** (0.562 vs 0.574, on n=119 vs n=375 -- well within noise range)
   and does not look like a dramatic right-hander bias in this first place. Plausibly this is because the
   taxonomy's off/leg/straight side axis (`datasets/taxonomy.py`) is already handedness-independent by
   design, and left-handers, though a minority (1,705 vs 3,134 in the scored data), are not vanishingly
   scarce.
2. **Mirroring is not a free fix, and actively hurts the side it was meant to help.** A candidate
   explanation: our spatial normalisation's scale reference (median shoulder-hip distance) and the
   mirror reflection point (hip midpoint) interact with genuine left/right asymmetries in real batting
   technique (which side of the pitch the bowler's angle favours, camera-angle conventions in broadcast
   footage) that a naive geometric mirror does not fully undo -- mirroring may be introducing distortion
   rather than removing bias. This is a testable follow-up (train ONE model on mirror-canonicalised data
   throughout, not just at test time) rather than a conclusion to draw from this result alone.

**Honest framing for the paper**: this audit does not find evidence of a strong handedness bias worth
correcting via mirroring in the technique scorer as currently built, which is itself a useful negative
result -- it means the small gap is not the easy, common failure mode (naive right-hander bias) that a
reviewer would first suspect.

### Shot classifier -- pending (waiting on training to finish)

## Reproduce
```
python models/train_shot_classifier.py --arch all --epochs 30
python models/train_technique_scorer.py --epochs 60
python models/handedness_audit.py
```
