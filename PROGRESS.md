# CricLens progress (resume point)

Last updated: 2026-09-23. Preprocessing, IVA experiments, and phase-3 model training (with a proper
seed-controlled variance sweep) are done; loss-weight tuning is next.

## How Kaggle work runs
Kaggle jobs run on Kaggle's servers and keep going when the laptop sleeps. Long jobs are split into
~30-minute chunks (`python kaggle/make_chunks.py <job> <n>`), each a separate kernel with its own saved output,
so a failure or pause costs at most one chunk. State: `kaggle/progress.json`. Kaggle allows 2 GPU sessions per
account; runners retry every 5 min when both are busy. Resume = re-run the runner:

    kaggle/run_chunks.sh <job> <n> [first last]     # e.g. kaggle/run_chunks.sh pose-rest 10 0 4

## Done
| # | Stage | Result |
|---|---|---|
| 1 | Dataset: 6 public sources, 480p clips, dedupe, match-grouped splits | 22,420 unique clips |
| 2 | Detection image set (ball/bat/stumps) | 23,799 images |
| 3 | Pose method: ViTPose-Base; per-clip batter; window = contact -0.8 s to +0.6 s, stops at camera cuts | chosen over YOLO11-pose |
| 4 | Pose on CricketVision (hand-drawn batter boxes) | 7,087 clips, 0 failures |
| 5 | Ball/bat/stumps detector (YOLO11s) | test mAP50 0.81 |
| 6 | Stumps clue for the batter finder | far stumps too small at 480p; adds nothing yet |
| 7 | Batter finder (gradient boosting on tracked-person features) | 91.2% on unseen matches (rules 57%) |
| 8 | Pose on the other 15,333 clips via the batter finder | 0 failures |
| 9 | 400-clip visual audit | striker right 92.5% (CricShot10k 89%, mostly keeper confusion) |
| 10 | Training index `data/processed/pose_index.parquet` | **16,460 train-ready** (train 11,891 / val 2,307 / test 2,262), ~98-99% right striker |
| 11 | IVA D1: quality profile of all clips (Kaggle CPU) | darkness not a problem; CricShot10k blurred/compressed |
| 12 | IVA D2: enhancement on 800 flagged clips + bat check | no enhancement helps pose; CLAHE raises bat detection (~89% of extra detections real) |
| 13 | IVA D3: degradation/restoration + padding study | adaptive median best for impulse noise; zero padding cuts edge-case pose error 20-60% |
| 14 | Zero-padding re-run: crop clamp removed from pose scripts, `kaggle/pose-pad` re-estimates joints for every clip that was clamped | **done** -- 10,271 clips, 0 failures; usable rate ~unchanged (+1 clip net), confidence improved for ~78% of clips |
| 15 | Frame-rate normalisation: every clip's window resampled to T=32 frames over normalised time (`models/resample_sequences.py`) | all 22,420 clips, 0 failures, 99.9% real (non-imputed) joint data; `data/processed/sequences.parquet` |
| 16 | Evaluation harness (`models/evaluate.py`): shared accuracy/F1/confusion-matrix and Spearman/R-ℓ2, with per-source/handedness breakdowns and a split-leakage check | built before any model, so every model is scored the same way |
| 17 | **Finding**: CricketVision's five body-part scores are ~one signal, not five (`docs/paper/finding_label_collinearity.md`) | raw corr 0.95-0.99, PC1 = 97.5% of variance, `score_overall` alone explains 97-99% of each part; changes how the scorer must be evaluated |
| 18 | Seed-controlled variance sweep (3 seeds x 2 models, `models/seed_sweep_driver.sh`) | full mean+/-std results in `docs/paper/results_phase3.md`; label-collinearity finding now confirmed across labels + 3 independently seeded trained models |
| 19 | Technique scorer loss-weight tuning (3x3 grid, `models/tune_technique_scorer_driver.sh`) | untuned defaults were the worst combo in the grid; new default kl_weight=0.001/reg_weight=10.0 gives mean Spearman 0.603+/-0.010 (up from 0.592+/-0.023) -- modest gain, ~half the run-to-run variance |
| 20 | IVA D4: pitch calibration (HSV + morphology + connected components + Hough crease lines, stumps fallback), full 22,420 clips (Kaggle CPU) | **20.0% of clips calibrated** (3,028 crease, 1,450 stumps-fallback); median stride 174.7cm, median swing 14.5 m/s; yield ranges 3.5% (ipl2023, tiny crease at 480p) to 38.2% (cricketvision); `data/processed/iva/pitch_calibration.parquet` |

Details: `docs/iva/iva_results.md`; syllabus mapping: `docs/iva/syllabus_alignment.md`.

## Next steps (in order)
1. **Phase 3 + seed sweep done** (2026-09-23), full results in `docs/paper/results_phase3.md`. Final,
   variance-characterised numbers (mean +/- pop. std across 3 seeds, corrected post-pose-pad joints):
   - shot classifier: **LSTM and Transformer are statistically tied for best** (macro-F1 0.785 +/- 0.009 vs
     0.766 +/- 0.014; accuracy 0.785 +/- 0.006 vs 0.781 +/- 0.008), GRU a little behind, RNN clearly last.
     LSTM is the reasonable default (cheaper to train, same ballpark accuracy). The single-run "GRU wins" /
     "Transformer wins" readings from the two earlier unseeded runs are superseded -- both were reading
     architecture rankings out of noise comparable in size to the gaps themselves.
   - technique scorer: mean Spearman 0.592 +/- 0.023, overall-score Spearman 0.600 +/- 0.024, **mean
     partial-Spearman-vs-overall 0.032 +/- 0.010** (every part within ~1-2 std of zero across all 3 seeds).
     **The label-collinearity finding is now confirmed across the labels themselves and 3 independently
     seeded trained models**: CricketVision's five part scores carry no independently learnable
     part-specific signal beyond overall quality, at least not one this setup can extract -- this is the
     headline result to lead with for the scorer, ahead of the raw Spearman number.
   - handedness audit (first-pass, not yet seed-swept): mirroring direction flipped between the two unseeded
     runs -- **not a reproducible effect; the "mirroring hurts left-handers" claim is retracted.** What held
     across both runs: a modest raw right>left performance gap (0.01-0.06 depending on model/run).
2. **Loss-weight tuning done** (2026-09-23, #19 above) -- technique scorer's new defaults (kl_weight=0.001,
   reg_weight=10.0) give a modest, seed-confirmed gain and much tighter variance. Model training for phase 3
   is now considered final: shot classifier (LSTM or Transformer) and technique scorer (tuned weights) both
   have variance-characterised numbers ready to quote in the paper.
3. **Pitch calibration done** (2026-09-23, #20 above) -- HSV/morphology/CC/Hough crease detection with a
   stumps-detection fallback, run on all 22,420 clips on Kaggle CPU. 20.0% of clips get a real stride/swing
   measurement in cm/m/s; ship as a per-clip coaching-report add-on, not a pipeline-wide dependency, given
   the yield. `data/processed/iva/pitch_calibration.parquet`.
4. Later: bat U-Net + bat angle from shape moments (Hough/HSV for pitch calibration's morphology and
   connected-components code in `models/pitch_calibration.py` is directly reusable for bat mask cleanup),
   TrackNet ball tracking, 3D pose lifting for camera angles (PoseC3D found this hurts on FineGym -- verify
   before investing here), coaching LLM, web app.

## Open decisions
- Practice/nets/shadow-batting videos (no bowler or ball) are parked; focus is match clips for now.
