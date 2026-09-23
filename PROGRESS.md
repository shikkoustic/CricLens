# CricLens progress (resume point)

Last updated: 2026-09-19. Preprocessing and the first IVA experiments are finished; model training has not started.

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
| 14 | Zero-padding re-run: crop clamp removed from pose scripts, `kaggle/pose-pad` re-estimates joints for every clip that was clamped | 10,271 clips queued; launched on Kaggle 2026-09-23, in progress |
| 15 | Frame-rate normalisation: every clip's window resampled to T=32 frames over normalised time (`models/resample_sequences.py`) | all 22,420 clips, 0 failures, 99.9% real (non-imputed) joint data; `data/processed/sequences.parquet` |
| 16 | Evaluation harness (`models/evaluate.py`): shared accuracy/F1/confusion-matrix and Spearman/R-ℓ2, with per-source/handedness breakdowns and a split-leakage check | built before any model, so every model is scored the same way |
| 17 | **Finding**: CricketVision's five body-part scores are ~one signal, not five (`docs/paper/finding_label_collinearity.md`) | raw corr 0.95-0.99, PC1 = 97.5% of variance, `score_overall` alone explains 97-99% of each part; changes how the scorer must be evaluated |

Details: `docs/iva/iva_results.md`; syllabus mapping: `docs/iva/syllabus_alignment.md`.

## Next steps (in order)
1. **Phase 3 is done, first pass** (2026-09-23), full results in `docs/paper/results_phase3.md`:
   - shot classifier: transformer best, test accuracy 0.788 / macro-F1 0.792 (RNN 0.754, LSTM 0.775, GRU 0.755)
   - technique scorer: mean Spearman 0.571; mean partial-Spearman-vs-overall 0.021 (~zero -- confirms
     finding #17 on a real trained model, not just the labels)
   - handedness audit (mirroring): no support for the hypothesis on either model; small raw gap, mirroring
     doesn't close it and sometimes widens it -- reproducible across both models
2. Once `kaggle/pose-pad` finishes (both chunks): rebuild `pose_index.parquet` (`models/pose_index.py`,
   picks up the overlay automatically), re-run `models/resample_sequences.py`, and re-train both models on
   the corrected joints for final numbers (expect a small change -- `pose-pad-c0`'s effect was mild: usable
   rate 91.65%->91.67%, confidence 0.831->0.834).
3. Tune the technique scorer before treating 0.571/0.577 as final: this was one architecture, un-tuned loss
   weights (kl_weight=0.01, reg_weight=5.0), 19 epochs. A sweep is the obvious next step.
4. IVA module next: pitch calibration (HSV pitch segmentation, morphology, connected components, Canny + Hough
   crease lines) for stride in cm and swing speed in m/s.
5. Later: bat U-Net + bat angle from shape moments, TrackNet ball tracking, 3D pose lifting for camera angles
   (PoseC3D found this hurts on FineGym -- verify before investing here), coaching LLM, web app.

## Open decisions
- Practice/nets/shadow-batting videos (no bowler or ball) are parked; focus is match clips for now.
