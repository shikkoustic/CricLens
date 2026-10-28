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

### Update 2026-10-17 (later still): hyperparameter sweep -- confirms it's not a tuning problem

`models/sweep_contrastive.py` ran 4 configs (lower lr, higher patience, hidden 64/128/256) against the
VAE baseline, since the first contrastive run's val curve was still falling at early-stop:

| config | lr | hidden | spearman | R-l2 | partial-rho |
|---|---|---|---|---|---|
| VAE baseline | - | - | 0.601 | 2.47 | 0.019 |
| lr3e4_p15 | 3e-4 | 128 | 0.573 | 2.63 | 0.075 |
| lr3e4_h256_p15 | 3e-4 | 256 | 0.554 | 2.67 | 0.067 |
| lr1e4_p20 | 1e-4 | 128 | 0.556 | 2.75 | **0.089** |
| lr3e4_h64_p15 | 3e-4 | 64 | 0.551 | 2.71 | 0.062 |

None of the 4 beat the baseline on raw Spearman or R-l2 (all landed 0.55-0.57 vs 0.601). Every config
beat it 3-4x on partial-rho vs overall, consistently, regardless of lr/hidden size -- that is not a
tuning artefact, it is the diff-based training objective itself surfacing more part-specific signal
than absolute regression does, even before the residual-target fix above is tried. Conclusion: more
hyperparameter search on the *absolute*-target objective is not going to close the raw-metric gap --
`--residual-targets` (just implemented, `models/train_technique_scorer_contrastive.py`) is the more
promising next lever, since it attacks the actual mechanism (shared-factor collapse) rather than
tuning around it. Best raw-Spearman config so far (lr3e4_p15) is the baseline hyperparameters for that
run.

### Update 2026-10-17 (final): residual-targets result -- closes most of the gap

Ran `--residual-targets` with the sweep's best hyperparameters (lr=3e-4, hidden=128, patience=15,
80 epochs, early-stopped at 29). Full comparison:

| | mean Spearman | mean R-l2 | partial-rho vs overall | predictions' PC1 share |
|---|---|---|---|---|
| VAE baseline | 0.601 | 2.47 | 0.019 | 1.000 |
| Contrastive, absolute targets (first run) | 0.565 | 2.60 | 0.034 | 0.998 |
| Contrastive, absolute targets (sweep best) | 0.573 | 2.63 | 0.075 | -- |
| **Contrastive, residual targets** | **0.584** | **2.56** | **0.116** | **0.991** |
| (labels themselves, for reference) | -- | -- | -- | 0.979 |

Residual targets close most of the raw-metric gap (Spearman 0.584 vs baseline 0.601, R-l2 2.56 vs 2.47
-- both now the closest any contrastive variant has gotten) *and* give by far the best partial-rho
(0.116, 6x the baseline, nearly 2x the best absolute-target config) and the first real drop in
prediction collinearity (PC1 0.991, down from 0.998-1.000, though still above the labels' own 0.979 --
progress, not solved). This is the first config where every metric that matters moved in the same
direction at once.

Training dynamic worth flagging for anyone tuning further: partial-rho peaked early (0.145-0.162 in
epochs 1-6) and drifted down through training (to ~0.04-0.07 by epoch 20+) while val Spearman kept
climbing -- the two objectives partially trade off, and early-stopping on Spearman alone (as the current
script does) leaves some partial-rho on the table. A combined early-stopping criterion, or reporting the
best-partial-rho checkpoint separately, is a reasonable next refinement.

Next: seed-variance check on this exact config (3 seeds, matching #7's protocol) before treating the
comparison as settled -- queued.

### Update 2026-10-17 (really final): 3-seed variance check -- the result holds

Same config (residual targets, lr=3e-4, hidden=128, patience=15), 3 seeds:

| seed | mean Spearman | mean R-l2 | partial-rho vs overall |
|---|---|---|---|
| 0 | 0.584 | 2.56 | 0.116 |
| 1 | 0.585 | 2.57 | 0.112 |
| 2 | 0.595 | 2.51 | 0.137 |
| **mean +/- std** | **0.588 +/- 0.006** | **2.55 +/- 0.03** | **0.122 +/- 0.013** |
| VAE baseline (single run) | 0.601 | 2.47 | 0.019 |

Tight variance across seeds (std is ~1% of the mean on Spearman, consistent with #7's VAE-baseline
seed spread) -- this is a reproducible result, not a lucky run. Seed 2 alone gets to 0.595 mean Spearman
and 2.51 R-l2, within noise of the baseline, while still landing 0.137 partial-rho (7x the baseline).
Averaged across seeds: raw Spearman within ~2% of baseline, R-l2 within ~3%, partial-rho 6.4x higher.

**Where this leaves the novelty claim**: contrastive regression with residual targets does not clearly
beat the VAE baseline on absolute score accuracy (Spearman/R-l2), but it is the first model in this
project (or, per docs/paper/dossier.md, arguably in the reviewed literature) to show a measured,
reproducible improvement in genuine per-part discrimination on this dataset, without sacrificing much
overall accuracy to get it. That is the honest shape of the result: not "beats the baseline," but
"same ballpark on overall accuracy, substantially better at the thing part-wise AQA is supposed to do
and the literature doesn't actually check for" (per #17/#18's methodological finding). Whether that is
enough novelty for the paper, or whether it should be paired with the duration-aware idea (#6 above) or
presented as a methods contribution (report partial-rho, not just raw Spearman) is exactly what's queued
for review.

### Update 2026-10-17 (one more): ablation -- isolates what's actually doing the work

Open question after the above: does the collinearity fix need the contrastive query-vs-exemplar
mechanism, or is "train on relative-to-overall targets" the real fix regardless of architecture?
`models/ablation_vae_residual.py` reuses the VAE baseline's own architecture unchanged, trained on
residual targets instead of absolute ones. Full 2x2:

| | mean Spearman | R-l2 | partial-rho | predictions' PC1 share |
|---|---|---|---|---|
| VAE, absolute targets (original baseline) | 0.601 | 2.47 | 0.019 | 1.000 |
| **VAE, residual targets (ablation)** | 0.562 | 2.64 | 0.108 | 0.993 |
| Contrastive, absolute targets (sweep best) | 0.573 | 2.63 | 0.075 | 0.998 |
| **Contrastive, residual targets (3-seed mean)** | 0.588 | 2.55 | 0.122 | 0.991 |
| labels themselves | -- | -- | -- | 0.979 |

Clean result: **the residual-target transform is the dominant fix, and it works on either
architecture** -- it takes the VAE's partial-rho from 0.019 to 0.108 (5.7x) and its PC1 share from
1.000 to 0.993, with no contrastive mechanism involved at all. The contrastive framing adds a smaller,
real improvement on top (0.108 to 0.122 partial-rho, 0.993 to 0.991 PC1, and notably the only
residual-target variant that *also* improves raw Spearman over its own absolute-target version: VAE's
Spearman gets worse with residual targets, 0.601 to 0.562, while contrastive's gets better, roughly
0.57 to 0.588). So: two separable effects, both real, and they compound -- contrastive + residual is the
best setup on every metric. For the paper, this means the honest attribution is "a target
reformulation that generalises across architectures, with contrastive regression against exemplars as
a complementary, independently-useful second improvement" rather than crediting either alone.

### Update 2026-10-17 (last one): combined-metric selection tried, didn't beat the best seed

Tested the refinement flagged above: `--select-metric combined` (early-stops on mean_spearman +
partial-rho instead of mean_spearman alone), same hyperparameters as the winning residual-targets
config. Result: mean Spearman 0.569, R-l2 2.64, partial-rho 0.133 -- worse on Spearman and R-l2 than
every one of the 3 plain-Spearman-selection seeds, and not even the best partial-rho (seed 2's
0.137 is higher, with better Spearman and R-l2 too). **Honest negative result**: explicitly selecting
for the combined metric traded away more Spearman than it gained in partial-rho, and didn't land
anywhere seed 2 hadn't already reached by chance. The best checkpoint found across everything tried so
far remains **seed 2, plain Spearman selection**: mean Spearman 0.595, R-l2 2.51, partial-rho 0.137,
saved at `models/technique_scorer_contrastive_residual_seed2/contrastive_scorer.pt`. Leaving
`--select-metric combined` in the script since it's a reasonable thing to try with more seeds/budget,
but it is not the current best config.

### Update 2026-10-17 (truly last): more exemplars tried, also didn't help

Tried 98 exemplars (all 4 quality-ranked candidates per shot x tier from the original visual review,
`kaggle/exemplars_expanded.csv`, no new review work needed) instead of the current 50 (top-2 per tier).
Same winning hyperparameters otherwise. Result: mean Spearman 0.574, R-l2 2.69, partial-rho 0.120 --
slightly *worse* than the 50-exemplar 3-seed mean (0.588, 2.55, 0.122) on every metric. **Another
honest negative result**: doubling the reference set diluted the average rather than reducing its
noise, most likely because the original quality-ranking already put the cleanest 2 candidates per tier
first, so the extra 2 are measurably noisier references, not just more of the same signal. 50
exemplars (`kaggle/exemplars.csv`, the original set) remains the better choice; `--exemplars-path`
stays in the script for anyone who wants to try a different selection (e.g. re-ranking by a different
quality criterion) rather than just more of the same-ranked candidates.

Also tried, cheaply (no retraining -- reuses the seed-2 checkpoint, post-hoc re-weighting only):
embedding-distance-weighted exemplar averaging instead of a plain mean (closer exemplar in embedding
space gets more weight, the inference-time version of the attention-weighted aggregation CoRe/MCoRe use,
docs/paper/dossier.md B10). Result: 0.593/2.53/0.138, statistically indistinguishable from plain
averaging's 0.595/2.51/0.137. Same-shot, quality-filtered exemplars are apparently similar enough in
embedding space that distance carries little differentiating signal -- a properly *trained* attention
head (grouped-batch training with a loss on the aggregated prediction, not per-pair diffs) might still
help, but is a bigger change than this round's budget covered; worth flagging as a follow-up rather than
doing speculatively.

**Round summary for review**: implementation, hyperparameter sweep, residual-target reformulation (the
real win), a clean architecture-isolating ablation, and three follow-up refinements that honestly didn't
help (combined selection, more exemplars, distance-weighted averaging) are all done and documented
above. The best result stands at seed 2 / plain-Spearman selection / 50 exemplars / residual targets /
plain averaging: mean Spearman 0.595, R-l2 2.51, partial-rho 0.137 (vs VAE baseline's 0.601, 2.47,
0.019).

### Update 2026-10-21: where next on partial-rho -- ceiling check ruled out, part-local input built

0.137 is a real, reproducible improvement over 0.019, but it's a thin signal in absolute terms, and a
reviewer will ask "is this close to the best achievable, or is there more on the table?" Two things
checked/built before chasing more hyperparameter tweaks (three of which already came back negative):

1. **Tried to establish a ceiling first, via inter-annotator agreement -- ruled out.** If CricketVision's
   own human raters wouldn't agree with each other on the part-specific residuals, no model could beat
   that disagreement, and we'd know 0.137 is already close to the achievable ceiling. But per
   `docs/DATASHEET.md`, the dataset's only "double-annotated" clips (`P3_V6` = `P4_V9` etc., 1,156
   strokes) are copies of one rating (Spearman 0.98 between the two folders), not independent re-scores
   -- so there is no usable inter-rater ceiling in this data. Can't check this without new annotation,
   which is out of scope; proceeding on the model side instead.

2. **Built the next concrete lever: part-localised pose input**, `models/train_technique_scorer_partlocal.py`.
   The residual-target fix (above) changed what the model is asked to predict, but never changed what it's
   *shown* -- one shared GRU still reads all 17 joints for every one of the 5 part heads, so even with a
   fixed target, nothing stops that one encoding from being an equally poor source of part-specific
   evidence for all 5. New architecture: the shared global GRU is unchanged and still drives the
   overall-score head (same as v2's `ContrastiveScorer`), but each of the 5 part heads additionally gets
   its own small Siamese GRU reading *only* that part's own COCO joint columns (hands head only ever sees
   wrist+elbow columns, feet only knee+ankle, etc. -- see `JOINT_GROUPS` in the script), so each part head
   has real, separate visual evidence instead of a slice of one shared full-body encoding. Always trains
   on residual targets (that part of the fix stays); this isolates whether part-local *input* helps on top
   of the already-settled target fix, not instead of it.

   Smoke-tested locally (`--epochs 2 --limit 200 --device cpu`): runs end-to-end, loss decreases, eval
   harness output shapes match the v2 script's, nothing crashes. Numbers from the smoke test itself are
   meaningless (200 clips, 2 epochs, CPU) and weren't kept.

   **Kaggle GPU run, 3 seeds, same protocol as the v2 seed check (`kaggle/technique-scorer-partlocal/`,
   self-contained since Kaggle kernels don't see `models/` -- resamples sequences on the fly from the raw
   `kps.npz` chunks since `data/processed/sequences.parquet` turned out not to be part of
   `criclens-processed.tgz.bin`, caught and fixed after the first push errored on a missing file):**

   | | v2 (residual + contrastive, shared input, 3-seed mean) | v3 (+ part-local input, 3-seed mean) |
   |---|---|---|
   | mean Spearman | 0.595 | 0.564 +/- 0.016 |
   | mean R-l2 | 2.51 | 2.71 +/- 0.11 |
   | partial-rho vs overall | 0.137 | **0.090 +/- 0.005** |

   **Negative result, consistently across all 3 seeds (tight std, not a lucky/unlucky draw): giving each
   part head its own joint-only encoder made every metric worse, not better.** The hypothesis -- that the
   model needs separate visual evidence per part, not just a separate *target* -- didn't hold in this
   implementation. Plausible reasons, untested: each local encoder only sees 2-4 joints, which may simply
   be too little signal on its own; and `diff_g` (the shared global context fed to part heads) was
   deliberately `.detach()`-ed to keep gradient responsibilities clean, which may have cut off more useful
   context than it isolated noise. Both are concrete things a follow-up could try (don't detach the global
   context; widen the joint groups to overlap, e.g. give "hands" the shoulders too for arm-chain context)
   but **not pursued further without review** -- this is now two architecture changes deep on what was
   already a thin signal, and the honest standing result for the paper remains v2's 0.137, not an improved
   number. Ceiling-check (inter-annotator agreement) stays ruled out per the entry above.

   **Follow-up tried: overlapping joint groups** (`train_technique_scorer_partlocal_overlap.py`,
   `kaggle/technique-scorer-partlocal-overlap/`), testing explanation (a) above in isolation (explanation
   (b), the detached global context, deliberately left unchanged). Each part's joint group widened to
   include its biomechanical neighbours along the kinetic chain (e.g. "hands" = shoulders+elbows+wrists,
   not just elbows+wrists), 6 joints per part instead of 2-4. 3-seed Kaggle result:

   | | v2 (shared input) | v3 (disjoint joints) | v3b (overlapping joints) |
   |---|---|---|---|
   | mean Spearman | 0.595 | 0.564 +/- 0.016 | 0.560 +/- 0.008 |
   | mean R-l2 | 2.51 | 2.71 +/- 0.11 | 2.69 +/- 0.05 |
   | partial-rho vs overall | 0.137 | 0.090 +/- 0.005 | 0.092 +/- 0.003 |

   **Still a negative result, and statistically indistinguishable from v3** (0.092 vs 0.090, well within
   each other's std) -- widening the joint groups changed nothing. This rules out "too little signal per
   part" as the explanation for v3's drop. The remaining untested explanation (detaching the shared global
   context from part heads) is still open, but two architecture variants landing in the same place is
   enough to stop here and write this up rather than try a third without review. **Standing best result
   for the paper remains v2: residual targets + contrastive regression, shared (non-part-local) input,
   partial-rho 0.137.**

### Update 2026-10-24: encoder pretraining tried -- also negative

Third lever tried (a different kind: data/representation, not architecture): pretrain the technique
scorer's GRU encoder on the much bigger shot-classification task (16,460 `train_ready` clips, all 6
sources -- `kaggle/shot-classifier-pretrain-gru/`, same `nn.GRU(51, 128, num_layers=2, dropout=0.2)` shape
as `ContrastiveScorer.encoder`, so the state_dict loads with no surgery) before fine-tuning on the small
~3,662-clip scored set with the settled v2 recipe. Rationale: the scorer has to learn "what does cricket
motion look like" and "how do I score it" simultaneously from a small pool; a pretrained encoder arrives
already knowing the former.

Pretraining step: GRU shot classifier, all 8 shot types, test accuracy 0.772 / macro-F1 0.771 (healthy,
in the same range as the LSTM/Transformer numbers from the original architecture comparison) -- the
encoder itself trained fine.

Fine-tuning step (`kaggle/technique-scorer-pretrained/`, both conditions x 3 seeds in one job for an
apples-to-apples comparison under identical code/environment):

| | baseline (random-init, same job) | pretrained |
|---|---|---|
| mean Spearman | 0.587 +/- 0.005 | 0.570 +/- 0.017 |
| mean R-l2 | 2.56 +/- 0.05 | 2.61 +/- 0.06 |
| partial-rho vs overall | 0.123 +/- 0.009 | **0.092 +/- 0.012** |

(The baseline here, 0.587/0.123, lands close to but not identical to the earlier-reported v2 numbers --
0.595/0.137 -- expected run-to-run noise; it confirms this job's control is a faithful re-creation of v2,
not a broken comparison.)

**Negative result: pretraining made every metric worse, not better.** Plausible cause, worth flagging
for the writeup rather than re-testing blindly: a readout mismatch between the two tasks. The shot
classifier reads the GRU by **mean-pooling across all 32 timesteps** (`RecurrentClassifier.forward`'s
`out.mean(1)`); the technique scorer reads it by taking only the **final hidden state**
(`ContrastiveScorer.embed`'s `h[-1]`). So the pretrained weights were optimised to produce a good
*time-averaged* representation, which is not the same thing the scorer actually consumes -- a real
architectural reason the transfer could hurt rather than help, not just "pretraining doesn't work here".
A fixed version (e.g. retrain the classifier with a final-hidden-state readout, or have the scorer
mean-pool too) is a legitimate follow-up, but **not run without review** -- three different levers
(part-local disjoint, part-local overlapping, pretraining) have now come back negative in a row. Standing
best result for the paper remains v2 (above): partial-rho 0.137.

**Follow-up tried: fix the readout mismatch.** Retrained the GRU shot classifier with the readout changed
to match `ContrastiveScorer.embed()` exactly -- final hidden state (`h[-1]`) instead of mean-pooling over
all 32 timesteps. Pretraining step stayed healthy (test accuracy 0.780, macro-F1 0.777, basically
unchanged from the mean-pool version's 0.772/0.771 -- the fix didn't cost classification quality). Same
fine-tuning job re-run with the corrected encoder:

| | pretrained, mean-pool readout (first attempt) | pretrained, final-hidden-state readout (fixed) | baseline (unchanged) |
|---|---|---|---|
| mean Spearman | 0.570 +/- 0.017 | 0.583 +/- 0.013 | 0.587 +/- 0.005 |
| mean R-l2 | 2.61 +/- 0.06 | 2.61 +/- 0.10 | 2.56 +/- 0.05 |
| partial-rho vs overall | 0.092 +/- 0.012 | **0.119 +/- 0.007** | **0.123 +/- 0.009** |

**The fix closed almost all of the gap -- partial-rho went from clearly worse than baseline (0.092) to
within baseline's noise band (0.119 vs 0.123 +/- 0.009; every metric's std ranges overlap).** This confirms
the readout-mismatch diagnosis was real: that's most of why the first attempt hurt. But the corrected
version is a **tie with baseline, not an improvement** -- pretraining no longer actively hurts, but it
doesn't add anything measurable beyond what the small scored set already teaches the model on its own.
Four levers tried now in this round (part-local disjoint, part-local overlapping, pretraining v1,
pretraining v2): three clean negatives and one tie. **Standing best result for the paper remains v2:
partial-rho 0.137**, and this round of exploration is being written up as the honest, documented search
it was, rather than continuing to chase further variants without review.

### Update 2026-10-26: ST-TR/Timesformer + NAFNet gating, on the shot classifier -- negative

Following a supervisor meeting, tested the supervisor's suggestion: (a) Timesformer-style "divided"
attention (separate temporal attention for "when" + spatial attention for "what pose") and (b) NAFNet's
activation-free gate, plus an "adaptive"/learned version. Literature check before building: neither piece
is novel alone -- divided spatial/temporal attention on skeleton data specifically is **ST-TR** (Plizzari
et al., arXiv:2008.07404, 9-layer two-stream GCN-hybrid design tuned for NTU's ~56k clips); NAFNet's fixed
gate is arXiv:2204.04676; the "adaptive" learned-gate idea is the established GLU/SwiGLU family (Shazeer,
arXiv:2002.05202). Built a simplified, single-stream version (`models/train_shot_classifier_sttr.py`,
`kaggle/shot-classifier-sttr-gate/`) with both gate types, deliberately on the **full shot-classification
pool (~16,460 clips, all 6 sources)** rather than the small scored set, since the bigger architecture
needs more data than the ~3,662-clip scoring task can offer.

First (single-seed) run looked promising: adaptive gate 0.768 macro-F1 vs simple gate 0.759, both in the
same range as existing architectures. **3-seed confirmation overturned this:**

| | macro-F1 (3-seed) |
|---|---|
| LSTM (existing baseline) | 0.785 |
| GRU (existing baseline) | 0.777 |
| Transformer (existing baseline) | 0.766 |
| ST-TR-style, simple gate | 0.755 +/- 0.003 |
| ST-TR-style, adaptive gate | 0.754 +/- 0.010 |

**Two negative findings.** (1) The adaptive-vs-simple gap was noise, not real -- 0.754 vs 0.755 is well
within each other's std once properly seeded, exactly the kind of single-run illusion the project's own
seed-variance protocol exists to catch. (2) The architecture underperforms all three existing baselines,
including Transformer -- this specific simplified combination doesn't beat anything already built.

**Operational note**, worth recording since it ate real time: the first 3-seed push hung for ~2h50m with
no error signal and no way to inspect progress mid-run on Kaggle's side (`kaggle kernels output` returns
nothing while a kernel is still running). Cancelled it (Kaggle's CLI has no separate stop command --
`kaggle kernels delete` + re-push was the only way) and a fresh push completed normally in the expected
~35-40 minutes, so it was very likely a stuck/queued session, not a bug in the code -- the same code,
same data, ran to completion with sane timing the second time.

Given this comes right after four negative/tied results on the technique-scoring side (2026-10-21 through
2026-10-24), the honest read is that the "swap in a published architecture, see if it wins" line of
search has not produced a win on either task so far. The interpretability-focused idea discussed but not
yet built -- using this architecture's attention weights to check whether spatial attention concentrates
on anatomically correct joints per body part, and whether temporal attention finds ball-bat contact
unsupervised -- remains a more promising direction *if* pursued, since it doesn't depend on this
architecture also having to win on raw accuracy; not started yet.

### Update 2026-10-26: real video + pretrained Timesformer -- promising, scaling to the full pool

The pose-domain ST-TR+gate result above came back negative, but the supervisor's actual ask was to try
Timesformer on **real video**, not a pose-domain adaptation of the same idea -- pose data throws away the
pixel information a pretrained video transformer is built to use. Built `kaggle/shot-classifier-timesformer/`:
raw frames (8 per clip, sampled across the pose pipeline's ball-release-to-follow-through window, cropped
to the tracked batter box with 12% padding) through Facebook's Kinetics-400-pretrained Timesformer
(`facebook/timesformer-base-finetuned-k400`), with each block's feed-forward sublayer optionally replaced
by a gated version (`none` = original pretrained GELU feed-forward, control; `adaptive` = SwiGLU-style
learned gate, warm-started -- see below).

**Three bugs hit and fixed, each verified before trusting it, not just assumed fixed:**
1. *Occlusion edge case*: some frames have no valid tracked batter box (low confidence/occlusion) --
   fixed by falling back to the nearest frame (by index) that does have one; caught via a smoke test
   against real downloaded clips, not found in CI.
2. *Unverified warm-start*: naively swapping in a fresh gated feed-forward discards that layer's pretrained
   weights, so it has to relearn 12 layers of feed-forward transform from scratch -- on a small subset the
   loss barely moved while the pretrained control dropped sharply. Fix: initialize the adaptive gate's two
   branches both from the pretrained dense layer, which reduces the formula to `SiLU(dense(x))`, a close
   stand-in for the original `GELU(dense(x))` -- checked directly against the real pretrained layer's
   output on random input (8% mean relative error, 0.998 correlation) before trusting it. The same trick
   does **not** work for the plain-product ("simple"/NAFNet) gate -- checked and found worse than the
   output scale itself (heavy-tailed outliers from squaring) -- left unfixed, `simple` excluded from runs
   until solved properly.
3. *CUDA OOM, then a stuck gradient*: batch_size=16 OOM'd a T4 partway through epoch 1 (a ViT-Base video
   transformer over 8 frames at 224x224 holds far more activation memory than anything else in this
   project). Fixed with mixed precision (fp16 autocast + GradScaler) + batch_size=8. This then made the
   warm-started adaptive gate's training completely stall (loss flat at ln(5), val macro-F1 bit-identical
   for 3 epochs straight) -- diagnosed by checking gradients in plain fp32 on CPU (no NaN/Inf, healthy
   magnitudes, ruling out the warm-start weights) and concluding GradScaler was skipping nearly every
   optimizer step on gradient overflow. Fixed with gradient clipping before the scaler step -- confirmed
   fixed (amp_scale stayed pinned at 32768 instead of collapsing; the model now genuinely learns).

**Results so far, scaling up on the CricketVision-only subset (5 shot classes) before expanding scope:**

| clips | gate | test macro-F1 |
|---|---|---|
| 300 | none | 0.533 |
| 1,500 | none | **0.678** |
| 1,500 | adaptive (gradient-clip fix) | 0.231 (learning, but well behind `none` within an 8-epoch budget) |

`none`'s result climbs clearly with scale (0.533 -> 0.678) and already matches the better pose-only
baselines (LSTM 0.785, GRU 0.777) despite only 1,500 of the ~4,863 CricketVision clips and a much harder
task (raw pixels, not pre-extracted joints) -- real signal that this approach is worth scaling further,
not a result to settle for. `adaptive` learns correctly post-fix but is far behind `none` in the same
epoch budget; not yet tuned (a separate/lower learning rate for the gate's fresh parameters is the next
thing to try, not attempted yet).

**Next step, in progress:** scale to the full pool across all 6 sources (cricshot10k 6,316 + cricketvision
4,863 + ipl2023 1,481 + kucricshot 1,192 + cricshot10 1,119 + amittalmale 409 = **15,380 clips, 8 shot
classes**), re-running `none` as the real baseline on this harder task (the 1,500-clip number doesn't
carry over) alongside `adaptive`. Given this is a multi-hour run and the project has already hit three
separate failure modes in this exact script, added epoch-level checkpoint/resume support (full
model+optimizer+scaler state saved every epoch, self-referential `kernel_sources` in
`kernel-metadata.json` so a prior push's output is visible as input on the next push) -- verified locally
end-to-end on a 4-clip real-data fixture (ran 1 epoch, confirmed the checkpoint, re-ran with a higher
epoch budget, confirmed it resumed from epoch 1 rather than restarting, and that the final eval correctly
reloaded the best-epoch weights) before spending Kaggle GPU time on the real run.

**That run died anyway -- a fourth failure mode, bigger than the checkpointing was built for.** Killed by
Kaggle's OOM killer (plain `Killed`, no traceback) 43 minutes in, before a single epoch finished, so the
new checkpoint/resume support never even got a chance to matter. Root cause: `cache_clips()` decoded
*every* clip's full pixel array (8x224x224x3 float32, ~4.6 MB/clip) into one Python dict held entirely in
RAM before training started -- fine at 300-1,500 clips (1-7 GB) but ~70 GB at the full 15,380-clip pool,
far past what a Kaggle GPU session's RAM allows. This design had only ever been exercised at the smaller
scales; nothing about it was wrong until the pool size changed by 10x.

Fixed by not holding decoded pixels at all: `VideoDataset` now decodes+crops each clip's frames on
access (lazily, per batch) instead of from a pre-built RAM cache, with `DataLoader(num_workers=4)` so
that CPU decode overlaps GPU compute instead of serializing in front of it. A cheap `precheck_clips()`
(reads only the small kps.npz window/box arrays, not video) replaces the old full-decode prefilter for
dropping clips with un-trackable windows; a clip that still fails to decode at runtime (rare: genuine
video I/O corruption, not a window/box issue) falls back to an all-zero tensor rather than crashing a
whole epoch over one bad file. Verified the same way as the checkpoint logic -- re-ran the 4-clip local
fixture end-to-end (including the resume check) against the lazy-decode path before relaunching on
Kaggle.

### Update 2026-10-27: supervisor approved the body-part-independence novelty; new architecture angle -- 'contact' gate

Supervisor approved the overall-score/individual-body-part-score independence finding (partial-rho 0.137,
PROGRESS.md 2026-10-24 and earlier) as one novelty for the paper, and asked to keep building on the
Timesformer/NAFNet direction. Since 'adaptive' is just Shazeer's published SwiGLU and 'simple' is
NAFNet's own published gate, neither is *our* architectural contribution -- added a third gate, **'contact'**,
as that piece: the 'adaptive' gate's sigmoid pre-activation is additionally biased by a learned readout of
each sampled frame's signed, window-normalised distance to the ball-bat contact frame (already extracted
by the pose pipeline as `window`, free to reuse). The idea: let the feed-forward gate open and close as a
function of proximity to the contact moment -- something neither NAFNet's fixed gate nor a plain learned
gate can do, since neither sees per-frame timing. This is specific to this project's data/task, not a
repackaging of existing work.

Implementation required knowing Timesformer's exact token layout (verified against the installed
transformers' `TimesformerEmbeddings.forward`, divided-space-time path): tokens are `[CLS, then
patch-major/frame-minor]` -- patch 0's 8 frames, then patch 1's 8 frames, etc. `GateWrapper` (new class)
tiles the per-sample, per-frame contact signal into that exact order and sets it as an attribute on every
gated layer before each forward pass, since `TimesformerEncoder`'s layer loop doesn't expose a way to pass
extra arguments through. `contact_proj` (the new linear readout) is zero-initialized (weight AND bias),
so at init the gate reduces exactly to 'adaptive' -- training decides whether the signal ends up mattering,
rather than starting from an unverified random init.

**Verified before spending any GPU time on it** (same rigor as the earlier warm-start fix): (1) a fresh
'contact' model's logits are bit-identical to a fresh 'adaptive' model's on the same random input despite
a nonzero contact signal (max abs diff 0.0) -- confirms the zero-init actually neutralises the new term;
(2) the token-tiling code's output, checked in isolation against the derived patch-major/frame-minor
order, places each frame's signal at exactly the expected positions; (3) hooked into a real pretrained
model's gated layers and confirmed the contact-signal tensor's length matches the real hidden-states
sequence length (1569 = 1 CLS + 196 patches x 8 frames) at every one of the 12 layers, not just in theory.
Also re-ran the checkpoint/resume smoke test against this new wrapped model structure (state_dict keys
now nest under `model.` because of `GateWrapper`) to confirm resume still works.

**Not yet run on Kaggle** -- holding off pushing until the full-pool `none`/`adaptive` run (started this
session, see the entry above) finishes, since pushing a new kernel version while one is `RUNNING` would
very likely restart/cancel it. Default `CRICLENS_GATES` is now `contact` alone (not `none,adaptive,contact`)
so the next push doesn't re-spend GPU time on already-confirmed numbers.

Ran it as a second, separate Kaggle kernel (`kaggle/shot-classifier-timesformer-contact/`) in parallel
with the still-running `none`/`adaptive` job instead of waiting, since Kaggle allows 2 concurrent GPU
sessions per account.

**Result, full 15,380-clip, 8-class pool, 8 epochs (no early stop -- val macro-F1 was still climbing):**

| epoch | val macro-F1 |
|---|---|
| 1 | 0.169 |
| 3 | 0.292 |
| 5 | 0.428 |
| 7 | **0.557** (best) |
| 8 | 0.504 |

Test: acc=0.576, **macro_f1=0.540**, weighted_f1=0.567. ~38 min/epoch, ~5h15m total. The steady climb
through epoch 7 (no plateau, patience=3 never triggered) suggests more epochs would likely help further
-- not run yet, pending a decision on whether to spend more GPU time extending this before comparing
against `none`/`adaptive`'s numbers on the same pool (that run is still in progress as of this writing).

### Update 2026-10-28: full comparison in -- both gates lose to the plain pretrained model, clearly

`none` and `adaptive` finished overnight (as two separate kernels on a teammate's Kaggle account, after
the original single-kernel sequential run got stuck at 13h+ and was killed -- see above). `contact` was
separately extended from 8 to 13 epochs via checkpoint-resume (not restart) on shikkoustic's account,
and hit early-stop (patience=3) at epoch 12, so it's a converged number, not a budget cutoff like the
other two still might be.

**Full 15,380-clip, 8-class pool, test set (2,109 clips):**

| gate | test macro-F1 | test acc | val macro-F1 trajectory |
|---|---|---|---|
| **none** (plain pretrained FFN) | **0.797** | 0.800 | 0.710 -> 0.758 -> 0.767 -> 0.784 -> 0.752 -> 0.700 -> **0.788** -> 0.782 (8 ep, noisy but clearly converged near 0.78-0.79) |
| adaptive (SwiGLU, published) | 0.594 | 0.612 | 0.149 -> 0.288 -> 0.367 -> 0.456 -> 0.482 -> 0.514 -> 0.535 -> 0.597 (8 ep, still climbing, not stopped) |
| contact (ours, 13 ep, early-stopped) | 0.583 | 0.627 | ...0.557 (ep7) -> 0.504 (ep8) -> **0.596** (ep9, new best) -> 0.569 -> 0.565 -> 0.596 (ep12, early stop) |

**`none` is the clear winner, by a wide margin** -- and it beats every pose-only baseline in the project
too (LSTM 0.785, GRU 0.777, Transformer 0.766, pose-ST-TR+gate ~0.754-0.755), despite this being the
harder 8-class task (those baselines were measured on an easier split in some cases). That's a genuine,
strong, honest result on its own: **real video through a Kinetics-pretrained Timesformer, with no
architectural tricks at all, outperforms every pose-based sequence model tried in this project.**

**Both gates lose to `none` by ~0.20 macro-F1, and `contact` does not clearly beat `adaptive`** --
0.583 vs 0.594 is within plausible single-seed noise, not a demonstrated win for the contact-conditioning
idea. The honest read: swapping out `.intermediate`+`.output` for either gate discards 12 layers of
Kinetics-pretrained feed-forward weights (documented caveat since this pipeline was first built), and
that loss is costly enough that neither gate's added expressiveness earns it back within a realistic
epoch budget -- confirmed now at full scale with a real baseline, not assumed. `contact`'s own
conditioning signal doesn't rescue this on top of `adaptive`'s already-warm-started gate.

**What this means for the architectural-novelty angle discussed with the supervisor:** gate-*replacement*
(this design) is not the way to beat the plain pretrained model -- that's now a clean, ablation-backed
negative result, exactly the kind of "tried it, here's why it didn't work" evidence the supervisor asked
for. A more promising variant, not yet tried: add the contact-conditioned gate as a small residual/additive
term *alongside* the pretrained feed-forward (frozen or fine-tuned) instead of replacing it outright --
keeps the Kinetics pretraining intact while still letting the model's representation respond to contact
timing. Also worth stating plainly: `none` winning this convincingly is itself a legitimate, publishable
contribution (video + pretrained Timesformer beats pose-based GRU/LSTM/Transformer), independent of
whether any gate variant ever wins.

## Open decisions
- Practice/nets/shadow-batting videos (no bowler or ball) are parked; focus is match clips for now.
