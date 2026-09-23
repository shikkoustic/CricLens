# Finding: CricketVision's five body-part scores are one signal, not five

**Status: confirmed with real data and reproducible.** Script: `models/label_correlation_analysis.py`.
Raw output tables: `docs/paper/label_correlation/*.csv`. Run on the 4,872 train-ready clips that carry
CricketVision scores.

## The question

`docs/paper/dossier.md` (Part IV, Gap 6) flagged this from Moodley & van der Haar's WACV 2025 results:
their model's five per-body-part Spearman correlations differ by only **0.003** (head 0.84146, shoulder
0.84040, hands 0.83865, hips 0.84027, feet 0.83844). That pattern is consistent with two very different
explanations:

- **(a)** the model genuinely, independently assesses all five body parts, and they happen to be almost
  equally learnable, or
- **(b)** the model has learned one overall-quality signal, and that signal correlates similarly with all
  five labels because **the labels themselves are collinear** — in which case no model can be shown to
  independently assess five body parts by this evaluation, however good it is.

This can be tested directly on the labels, without training anything.

## The result

On the 4,872 scored train-ready clips:

**1. Raw correlation between the five part scores: 0.951–0.987.** Every pair of anatomically distinct
body parts — head and feet included — correlates above 0.95.

**2. One principal component explains 97.5% of the joint variance** across the five (standardised) scores.
The next four components together explain the remaining 2.5%.

**3. `score_overall` alone explains 96.7–99.2% of each individual part's variance** (R²), a linear fit of
one part against the overall score.

**4. Partial correlations — controlling for the other three parts — collapse to near zero or flip sign**
(mean |partial r| = 0.267, several pairs at -0.19 to -0.04). No stable positive structure survives once
the shared factor is removed.

**5. Residuals after regressing out `score_overall`** show the same thing from a different angle: some
pairs anti-correlate as strongly as -0.59 (hands/hips), -0.48 (shoulder/feet), -0.41 (head/feet). This is
not the residual of independent measurements — it looks like annotators distributing a fixed overall
impression across five slots, so that giving more to one part leaves less for another.

## What this means

**CricketVision's five body-part scores carry, to first approximation, one number:** overall stroke
quality. There is a small amount of real per-part structure in the residuals (points 4 and 5 are not
exactly zero, and the sign pattern is consistent enough to not be pure noise), but it is minor next to
the shared factor, and what's there looks partly like a scoring artefact (annotators trading points between
parts) rather than independent biomechanical measurement.

**Consequence for the field's one entry in this space:** I3D-AE-LSTM's near-identical per-part Spearman
(spread 0.003) is not evidence of independent per-part assessment — it is the expected result of predicting
one latent quality signal and evaluating it against five collinear labels. Explanation (a) above is not
supported; explanation (b) is. **Nobody has yet demonstrated a model that independently discriminates
cricket technique per body part**, because the only dataset with per-body-part cricket scores does not,
on inspection, contain five independent measurements.

## What this changes for CricLens

This is not a reason to abandon per-body-part scoring — coaches want per-part feedback regardless of the
statistics, and the small residual structure (point 4/5) is real and possibly usable. It does change what
the paper can honestly claim, and sets three requirements:

1. **Report this finding.** It is a genuine, checkable contribution in its own right — a data-quality
   audit of the field's only per-part cricket AQA dataset, parallel to the leakage audit already planned
   for the splits.
2. **Evaluate discriminative power directly, not just per-part Spearman against raw labels.** Report
   per-part Spearman **and** partial Spearman (predicted part vs true part, controlling for predicted/true
   overall) — the second number is the one that actually tests whether a model captures part-specific
   signal beyond the shared factor. A model that matches or improves on I3D-AE-LSTM's raw per-part
   Spearman while also showing non-trivial partial correlation would be a real advance; one that doesn't
   is just fitting the dominant factor, however good the headline number looks.
3. **Consider score_overall as the primary target**, with per-part scores as a secondary, explicitly
   caveated output — rather than presenting five outputs as if they were five equally-weighted contributions.

## Reproduce

```
python models/label_correlation_analysis.py
```
No GPU, no model, ~2 seconds. Re-run after any change to `manifest.parquet` or the score columns.
