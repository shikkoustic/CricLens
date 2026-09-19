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

Details: `docs/iva/iva_results.md`; syllabus mapping: `docs/iva/syllabus_alignment.md`.

## Next steps (in order)
1. **Zero-padding re-run** on the 4,461 clips whose batter crop leaves the frame (pose only, reuse saved
   boxes; ~40 min Kaggle GPU). Change crop padding in the pose scripts from clamping to
   `cv2.copyMakeBorder(..., BORDER_CONSTANT)`, then rebuild `pose_index.parquet` with `models/pose_index.py`.
2. **Frame-rate normalisation**: clips are 25 or 30 fps; resample joint sequences to one rate before training.
3. **Phase 3: train the models** on `pose_index.parquet` (train-ready clips): shot classifier (RNN vs LSTM vs
   GRU vs Transformer), technique scorer (VAE + regression on CricketVision scores), left/right bias audit.
   Note: `scoop` has only 96 clips; merge or report separately.
4. IVA module next: pitch calibration (HSV pitch segmentation, morphology, connected components, Canny + Hough
   crease lines) for stride in cm and swing speed in m/s.
5. Later: bat U-Net + bat angle from shape moments, TrackNet ball tracking, 3D pose lifting for camera angles,
   coaching LLM, web app.

## Open decisions
- Practice/nets/shadow-batting videos (no bowler or ball) are parked; focus is match clips for now.
