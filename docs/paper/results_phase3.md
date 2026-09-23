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
- Trained on data *before* the `kaggle/pose-pad` zero-padding re-run lands (see PROGRESS.md next steps);
  clamping was measured mild (76% of affected frames lose only 5-10% of the padded box), so the effect
  of re-running on the corrected joints is expected to be small. Flagged here for the honest record --
  will re-run both once `pose_index.parquet` is rebuilt.

## 1. Shot classifier (RNN vs LSTM vs GRU vs Transformer)
8 shots (taxonomy's `other` excluded, `scoop` kept at n=96/70/11/15 total/train/val/test -- reported
separately per PROGRESS.md). Class-balanced loss (35x count imbalance, drive 3,402 vs scoop 96).

<!-- TABLE: arch | best_val_macro_f1 | test_accuracy | test_macro_f1 | test_weighted_f1 -->

**Comparison points** (docs/paper/reading_log.md): Kang's honest EfficientNet-B0+GRU on CricShot10
(random split) = 92.25%; CricShotNet on CricShot10k (random split) = 89%. Ours is pose-only, multi-source,
on match-grouped (non-leaking) splits -- a harder, more honest setting, so a materially lower number here
is expected and is itself evidence for the leakage argument (docs/paper/dossier.md A3), not a weakness.

## 2. Technique scorer (GRU-VAE + regression)
Trained on the 4,872 scored train-ready clips (3,718 / 657 / 497 train/val/test). Predicts all 6 targets
(5 parts + overall) jointly.

<!-- TABLE: per-part rho, R-l2, partial_spearman_vs_overall; mean_spearman; mean_partial_spearman_vs_overall;
     test_overall_spearman vs I3D-AE-LSTM's 0.84 -->

**Reading these numbers**: per `docs/paper/finding_label_collinearity.md`, raw per-part Spearman close to
the mean Spearman is *expected* given the labels correlate at 0.95-0.99 -- it is not evidence of
independent per-part discrimination. `partial_spearman_vs_overall` is the number that tests that claim.
During training, raw val Spearman rose steadily (0.48 -> ~0.58 by epoch 10) while val partial Spearman
stayed flat near zero throughout -- consistent with the model learning to track overall quality well
without learning independent part-specific signal, exactly as the label-collinearity finding predicts.

## 3. Handedness bias audit
Test-time mirroring (`models/handedness_audit.py::mirror_sequence`, unit-tested: mirroring twice is the
identity, left/right joint pairs correctly swapped) on left-handed test clips, using the already-trained
models -- no retraining required for this first pass.

<!-- TABLE: handedness x {normal, mirrored} accuracy/macro-F1 (classifier) and mean Spearman (scorer) -->

**Interpretation guide**: if mirrored-left-hander performance beats un-mirrored, the model has learned
right-hander-specific patterns that don't transfer across the mirror symmetry batting actually has (a
cover drive and its mirror image are the same shot) -- evidence for training-time mirror-canonicalisation
as a fix, not just a training-data-scarcity story (1,705 left-handed vs 3,134 right-handed clips).

## Reproduce
```
python models/train_shot_classifier.py --arch all --epochs 30
python models/train_technique_scorer.py --epochs 60
python models/handedness_audit.py
```
