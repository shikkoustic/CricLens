# CricLens progress (resume point)

Last updated: 2026-09-24. Preprocessing, IVA experiments and phase-3 model training are done. The web
app now runs end to end on a laptop and has been measured on 48 test clips (#22); the open work is the
batter finder's confident wrong picks and the bat U-Net domain gap.

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
| 7 | Batter finder (gradient boosting on tracked-person features) | 92.9% on unseen matches, all 6 sources (rules 63%) |
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
| 21 | IVA D5: bat segmentation U-Net + bat angle from shape moments | test IoU 0.81 on its own (mostly product-photo) test set, but **fails completely on our real clips** (max confidence 0.04-0.10 on 9 real frames, incl. one cropped around a clearly-visible bat) -- genuine domain gap, not usable yet; needs real-footage training data or heavy augmentation |
| 22 | Web app end to end on a laptop (`app/`), measured with `scripts/app_smoke.py` on 48 test clips (6 per shot, all 6 sources) | **48/48 analysed, 0 failures, shot accuracy 36/48 = 75.0%** -- in line with the LSTM's 78.0% test accuracy, so the serving path reproduces the trained model. Median 28s a clip with the local LLM, 12s without |

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
4. **Bat U-Net trained but doesn't transfer to real footage** (2026-09-24, #21 above) -- needs real-footage
   bat labels (a small hand-labelled sample from our own clips) or heavy augmentation (motion blur,
   compression, occlusion) before it's usable. Decide whether that's worth the effort vs. relying on the
   existing YOLO box detector (mAP50 0.81, already works) without pixel-level angle.
5. **Web app runs end to end** (2026-09-24, #22 above). `bash scripts/setup_app.sh` then
   `python -m app.server`. Upload a clip or pick a sample: overlay video, shot with its probabilities,
   technique dial and part bars, stride/swing where calibration works, and coaching text written by a
   local Qwen2.5-1.5B (template fallback if it is unavailable). Re-measure any change with
   `python scripts/app_smoke.py`. What the 48-clip run established, beyond the headline 75.0%:
   - **The reliability warnings carry real information.** They fire on 15/48 clips and cover 7 of the 12
     wrong shot calls; clips that stay silent are right 28/33 = 85%, against 75% overall. The batter
     finder's own confidence drives most of this: below 0.9 it is right 3/8, at or above it 33/40.
   - **The collinearity finding holds at serving time.** Across 48 clips the five body-part percentiles
     spread by a median of 4 points and never by the 15 that would make `app/coach.py` name a specific
     weak body part, so the app never once made a part-specific claim the data cannot support.
   - The app's live pitch calibration agrees with the D4 batch results on 42/47 clips, so what users see
     is the same measurement the write-up characterises.
6. **Fixed: the batter finder only ever learned one broadcast style.** It used to train on
   `kaggle/chunks/pose-cv-c*` alone (CricketVision) and apply the result to all six sources, which on an
   amittalmale drive picked a fielder at finder_p 0.902. Retrained across all six sources
   (`models/train_batter_finder.py`, 2026-10-17): visually reviewed 2,559 clips across the five
   non-CricketVision sources (`models/finder_review.py`, `kaggle/finder-review-sample.csv`), combined with
   the existing 400-clip pass/fail audit (`kaggle/pose-rest-audit/verdicts.csv`, which still caught 30
   keeper-confusion errors the faster visual pass missed -- the two reviews overlapped on 76 clips and
   disagreed on 5, so the hand audit took precedence there). Confirmed-wrong clips contribute negatives
   only (the actual batter track wasn't re-identified, just the wrong pick ruled out). New model: 9,004
   clips (6,121 CricketVision + 2,883 confirmed from the other five), 92.9% grouped 5-fold top-1, every
   source above 91% (cricshot10k lowest at 92.7%, consistent with where the keeper-confusion errors
   concentrate). `models/batter_finder.joblib` needs re-uploading to Kaggle before the next pose-rest run
   picks it up.
   Tried and rejected: `segment_pitch()` from D4 as a geometric prior -- it segments the whole playing
   surface, not the pitch strip, so it does not separate a fielder on grass from the striker.
7. **Open: normalised-time resampling erases duration.** `models/resample_sequences.py` stretches every
   window to 32 frames, so a 0.1s window (the observed minimum) and a full 1.4s swing reach the model as
   the same shape. Worth reporting as a limitation, and feeding duration back in as a feature is a
   cheap, honest novelty candidate for the paper.
8. **CricketVision can be re-cut longer; the other five sources cannot.** Those five ship pre-cropped
   clips -- amittalmale's raw archive holds the same 1.2s files, `clip_start`/`clip_end` are null, and
   cricshot10k's median clip is 0.8s -- so their follow-through is simply not recorded anywhere.
   CricketVision is different: all 8,441 clips carry `clip_start`/`clip_end` against 202 source videos,
   and `datasets/cricketvision.py` streams those videos from Dropbox (`VIDEOS_URL`), cuts, then deletes
   them. `PAD = 0.3` is the context kept either side of buildup/follow-through; raising the trailing pad
   to ~1.2s takes the median clip from 1.99s to ~2.9s, which is the "1-1.3s past follow-through" worth
   having. These are also **the only clips with technique scores**, so they are the subset that matters.
   Cost: re-stream and re-cut, re-run pose on Kaggle, rebuild `pose_index.parquet`. This gates step 7 --
   duration-awareness cannot be demonstrated on clips that never contained a follow-through.
9. Later: TrackNet ball tracking, 3D pose lifting for camera angles (PoseC3D found this hurts on FineGym --
   verify before investing here).

## Paper: where the novelty stands (2026-09-24)

The supervisor's brief is: survey the field, reproduce the existing pipeline as a baseline, add a
novelty, and show the novelty beats the baseline. Against that, what we hold today:

- **A methodological result, already evidenced.** I3D-AE-LSTM (WACV 2025) -- the direct competitor,
  and the source of the CricketVision dataset -- reports *part-wise* action quality assessment on these
  exact labels. Our finding (#17, #18) is that those five part scores are one signal: PC1 97.5% of
  variance, and mean partial-Spearman-vs-overall 0.032 +/- 0.010 across three seeds. The app run (#22)
  showed it holds at inference too -- over 48 clips the five predicted part percentiles never spread
  the 15 points that would justify naming a weak body part. Proposing partial correlation against the
  overall score as the reporting protocol for part-wise AQA is a contribution in its own right.
- **The improvement to measure.** Every AQA pipeline in the survey resamples to a fixed length over
  normalised time, ours included, which erases duration: a 0.1s window and a full 1.4s swing reach the
  model as the same 32 frames. A duration-aware or contact-anchored representation is the obvious
  candidate for "ours beats the baseline", and it needs the longer CricketVision clips (step 8).
- **Supporting dataset-integrity material.** 1,612 verified duplicates removed, CricketVision shipping
  1,156 strokes duplicated across annotator folders, and CricShot10's *published* split leaking 75
  duplicate clusters across its own train/val/test.

### Update 2026-10-17: contrastive regression, first result (not yet beating baseline)

Per a supervisor conversation, the LLM coaching-feedback step is dropped from the app and the paper
focuses purely on shot classification + technique scoring; the novelty direction settled on is
**contrastive regression against fixed exemplars** (CoRe/TSA/MCoRe family, docs/paper/dossier.md B10),
not the duration-aware representation above (still a valid open item, just not what got built first).

Built: `kaggle/exemplars.csv` -- 50 clips (10 per shot type, 5 score tiers each), quality-filtered and
visually confirmed clean at window start/contact (`models/exemplar_review.py`), train-split only so
val/test never leak into the reference set. `models/train_technique_scorer_contrastive.py` -- Siamese
GRU encoder (shared weights) predicts the score diff between a query and each same-shot exemplar; at
inference, predicted score = mean(exemplar_score + predicted_diff) across that shot's 10 exemplars.

First full run (60 epochs, early-stopped at 11): **does not yet beat the VAE baseline** on raw metrics.

| | mean Spearman | mean R-l2 | partial-rho vs overall |
|---|---|---|---|
| VAE baseline (`models/technique_scorer/summary.json`) | 0.601 | 2.47 | 0.019 |
| Contrastive (this run) | 0.565 | 2.60 | 0.034 |

Early stop after just 3 productive epochs suggests underfitting/over-regularisation at this scale (only
50 exemplars x ~3,662 train clips = ~37k pairs, GRU hidden=128, no hyperparameter search yet) rather
than the idea being wrong -- partial-rho vs overall, the metric that actually matters for the paper's
"parts carry independent signal" claim, did move the right direction. Needs: a learning-rate/hidden-size
sweep, more patience before early-stopping (the val curve was still falling when it stopped), and
probably a seed-variance check (per #19's protocol) before treating either number as settled. Not yet
committed to beating the baseline -- flagging honestly rather than overclaiming.

### Update 2026-10-17 (later): a bigger finding -- both models' PREDICTIONS are more collinear than the LABELS

`models/prediction_collinearity_check.py` ran the same PCA/correlation check #17/#18 ran on the labels
(PC1 97.5% of variance across the 5 part scores), but on each trained model's actual test-set
predictions instead. Both come back **worse than the labels, not better**:

| | PC1 share | mean pairwise correlation |
|---|---|---|
| labels (test split) | 0.979 | 0.95-0.99 |
| VAE baseline predictions | 1.000 | 0.999 |
| contrastive predictions | 0.998 | 0.997 |

Neither architecture learns anything resembling independent per-part signal -- both collapse to
predicting one learned "quality" scalar and copying it across all 5 outputs, more collinear than even
the (already collinear) labels force them to be. This is a stronger, more fundamental version of the
#17/#18 finding: it is not just that the labels can't support independent part discrimination, every
model tried on them inherits and amplifies that collinearity regardless of training objective. The
nonzero partial-rho-vs-overall numbers above are real but are computed on a sliver of residual variance
against predictions that are ~99.7-100% explained by one factor, so they should be read as "barely
any part-specific signal exists" rather than "the model is tracking parts independently."

Likely cause: with MSE loss, the cheapest way to fit 5 correlated targets is to predict their shared
factor well and leave each part's small independent variance essentially unmodelled (near the mean),
which *increases* prediction correlation past the label correlation rather than preserving it.
**Next experiment, queued**: train on *residual* targets (`score_part - score_overall`) instead of
absolute part scores, so the model can't satisfy the loss by just reproducing the shared factor -- the
shared factor is removed from the target by construction, forcing it to fit only the part-specific
deviation. `score_overall` stays a separate, absolute target. If this still doesn't produce
independently-moving part predictions, that itself is worth reporting as a dataset-level limitation,
not an architecture one.

## Open decisions
- Practice/nets/shadow-batting videos (no bowler or ball) are parked; focus is match clips for now.
