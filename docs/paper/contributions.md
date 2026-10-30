# CricLens: what the paper's contributions are

Written to settle one question before the write-up starts: **what exactly are we claiming, in what
order, with what evidence.** Everything below is drawn from work already done and documented
elsewhere (`PROGRESS.md`, `docs/paper/finding_label_collinearity.md`, `docs/paper/results_phase3.md`,
`docs/paper/scope_report.md`); this file adds no new numbers, it ranks and frames the ones that exist.

## The one-sentence version

**Per-part cricket technique scoring, as currently framed by the field's only dataset, is measuring
one thing five times, not five things once -- and once you fix the evaluation and the training
target to account for that, you can recover a modest but real, reproducible amount of genuine
per-part signal, though not a large one.**

Everything else in the paper (the benchmark, the shot classifier, the handedness audit) supports or
surrounds that claim. None of it should headline over it.

## Ranked contributions

### 1. Primary: the label-collinearity finding + the evaluation protocol it implies

**Claim:** CricketVision's five body-part scores (head/shoulder/hands/hips/feet) are, to first
approximation, one latent "stroke quality" score split five ways by annotators, not five independent
biomechanical measurements. Evidence, strongest first:

- Raw pairwise correlation between the five labels: **0.951-0.987** (`finding_label_collinearity.md`).
  Anatomically unrelated pairs (head/feet) correlate as strongly as adjacent ones.
- One principal component explains **97.5%** of the joint variance across the five scores.
- `score_overall` alone explains **96.7-99.2%** of each individual part's variance (R²).
- Confirmed independently on **trained model predictions**, not just labels: both a VAE regressor and
  a contrastive-regression model, trained with different objectives and architectures, predict the
  five parts *more* collinear with each other (PC1 99.7-100%) than the labels themselves force them to
  be (97.9-98%) (`models/prediction_collinearity_check.py`). The field's one prior entry in this space
  (I3D-AE-LSTM, WACV 2025) reports five per-part Spearman scores within **0.003** of each other -- which
  this finding reinterprets as the expected signature of predicting one shared factor, not evidence of
  independent per-part assessment.
- Confirmed across **three independently seeded trained models** (`results_phase3.md`): mean
  partial-Spearman-vs-overall = 0.032 +/- 0.010, indistinguishable from zero at every seed.

**Why this is the headline, not a footnote:** it is checkable without training anything (a label-only
statistical audit), it directly reinterprets a published result, and it motivates every other modeling
decision downstream (why absolute-score Spearman is the wrong primary metric; why residual targets
help; why "five independent scores" was never the right description of the task). A reviewer can
verify it from `models/label_correlation_analysis.py` in under 2 seconds with no GPU.

**What to propose as a field-level contribution, not just a CricLens finding:** report **partial
Spearman vs. overall** (predicted part vs. true part, controlling for predicted/true overall) alongside
raw per-part Spearman for any part-wise AQA system. Raw per-part Spearman cannot distinguish "the model
independently learned this part" from "the model learned overall quality and every part correlates
with overall quality." This is a protocol contribution usable by anyone working on part-wise AQA, not
specific to cricket.

### 2. Secondary: a target reformulation that recovers real (if modest) per-part signal, and
   generalises across architectures

**Claim:** training on **residual targets** (`part_score - overall_score` instead of the absolute part
score) is the dominant fix for the shared-factor collapse in (1), and it works on both architectures
tested, not just one — so it is a target-design fix, not an artifact of a particular model.

Evidence (`PROGRESS.md`, 2026-10-17 entries; ablation table reproduced below):

| | mean Spearman | R-l2 | partial-rho vs overall | predictions' PC1 share |
|---|---|---|---|---|
| VAE, absolute targets (original baseline) | 0.601 | 2.47 | 0.019 | 1.000 |
| VAE, residual targets (ablation) | 0.562 | 2.64 | 0.108 | 0.993 |
| Contrastive regression, absolute targets (tuned) | 0.573 | 2.63 | 0.075 | 0.998 |
| **Contrastive regression, residual targets (3-seed mean)** | **0.588 +/- 0.006** | **2.55 +/- 0.03** | **0.122 +/- 0.013** | **0.991** |
| (labels themselves, for reference) | -- | -- | -- | 0.979 |

Two separable, additive effects: residual targets alone (applied to the unchanged VAE architecture)
take partial-rho from 0.019 to 0.108 — a 5.7x improvement with zero architecture change. Contrastive
regression against same-shot exemplars (CoRe/TSA-family: predict the score *difference* to each of 50
quality-filtered reference clips, average the result) adds a further, smaller improvement on top
(0.108 -> 0.122) and is the only variant where residual targets *also* improve raw Spearman over its
own absolute-target version (0.573 -> 0.588), rather than trading it away as the VAE does (0.601 ->
0.562).

**Honest framing, stated plainly, not as a late caveat:** this does not beat the absolute-score VAE
baseline on raw Spearman (0.588 vs 0.601 — within noise, not a win) or R-l2 (2.55 vs 2.47). It is a
**same-ballpark-accuracy, substantially-better-at-independent-part-discrimination** result. That is
the correct scope of the claim: not "state of the art technique scoring," but "the first model on this
dataset, and arguably in the reviewed literature per `docs/paper/dossier.md`, to show a measured,
reproducible improvement in the specific thing part-wise AQA is supposed to do and the field doesn't
check for."

**Robustness already established**, worth citing together as "we checked the obvious alternative
explanations before claiming this":
- 3-seed variance check on the winning config: 0.588 +/- 0.006 Spearman, 0.122 +/- 0.013 partial-rho —
  tight, not a lucky run.
- Inter-annotator-agreement ceiling check: ruled out as infeasible (CricketVision's "double-annotated"
  clips are copies of one rating, Spearman 0.98 between folders, not independent re-scores — no usable
  ceiling exists in this data; `docs/DATASHEET.md`).
- Five follow-up variants tried and **honestly reported as negative/tied**, which is itself evidence the
  positive result isn't an artifact of under-exploring the space: part-local disjoint joint inputs,
  part-local overlapping joint inputs, encoder pretraining on shot classification (twice — once with a
  genuine readout-mismatch bug, once fixed to a tie), combined-metric early-stopping selection, a larger
  (98-clip) exemplar set, and distance-weighted exemplar averaging. None beat the standing config.

### 3. Supporting: the leakage-audited multi-source benchmark

6 sources, 22,420 deduplicated clips, match-grouped splits (no clip, no match, leaks across
train/val/test). Concretely falsifies two things in the prior literature this project can point to by
name: CricShot10's published split leaks 75 duplicate clusters across its own train/val/test, and
CricketVision ships 1,156 strokes duplicated across annotator folders. **The shot classifier's own
number should be reported as lower than Kang (92.25%, CricShot10, random split) and CricShotNet (89%,
CricShot10k, random split) and framed as the point**: ours (LSTM, 78.5% macro-F1, match-grouped,
8-way, multi-source) is harder and is, as far as this project's literature review found, the first
number in this space measured on a split that provably cannot leak.

This is dataset/benchmark housekeeping, not a modeling result, but it is citable on its own
(manifests, dedupe clusters, and grouped splits are a releasable artifact — SoccerNet-style) and it is
what makes contributions 1 and 2 trustworthy rather than contaminated by the same leakage problem being
called out in others' work.

### 4. Supporting: the batter finder

Gradient-boosted classifier on tracked-person features beats hand-written rules by a wide margin
(92.9% grouped top-1 across all 6 sources vs. 57-63% for rules) and was caught making a real mistake
during development (training only on one broadcast style initially picked a fielder on an
amittalmale clip at 0.902 confidence) and fixed by retraining across sources with a mixed
visual + hand-audit review process. This is an engineering contribution, not a modeling novelty, but
it is the reason the downstream pose/shot/scoring pipeline is trustworthy at all, and the "trained on
one style, silently wrong on others" failure mode is worth reporting as a cautionary methods point.

### 5. Supporting, negative-but-honest: the architecture-search trail

Worth a short subsection, explicitly framed as a negative-results contribution rather than buried:
seven architecture-level attempts to improve either the shot classifier or the technique scorer beyond
what's in (2) came back negative or tied, each checked against a literature source and reported with
its own citation (`PROGRESS.md`, 2026-10-21 through 2026-10-26):

| Attempt | Target | Result |
|---|---|---|
| Part-local joint inputs (disjoint) | technique scorer | partial-rho 0.090 vs v2's 0.137 — worse |
| Part-local joint inputs (overlapping) | technique scorer | partial-rho 0.092 — statistically same as disjoint, still worse |
| Encoder pretraining on shot classification (mean-pool readout) | technique scorer | partial-rho 0.092 — worse (later found: readout mismatch) |
| Encoder pretraining, readout fixed | technique scorer | tie with baseline, no net gain |
| ST-GCN encoder (replacing the pose-GRU) | technique scorer | loss vs GRU, confirmed over 3 seeds |
| ST-TR-style divided attention + NAFNet/adaptive gating | shot classifier | 0.754-0.755 macro-F1, below LSTM/GRU/Transformer baselines |

**Why this belongs in the paper, framed correctly:** it is evidence the field's standard moves
(architecture swaps, pretraining, part-local inputs) do not trivially transfer to this small,
collinear-label, 6-source cricket setting, which is itself informative for anyone building on
CricketVision next, and it preempts the obvious reviewer question ("did you just not try hard enough
on the modeling side?"). It should **not** be presented as a sequence of failures apologetically — it
should be framed as a documented negative-result search, which several venues (and this course's own
emphasis on reproducibility) treat as legitimate content, not padding.

### 6. Supporting: the handedness audit, including a retraction

4,400 right- / 2,492 left-handed labelled strokes — existing fairness work in action recognition covers
skin tone, gender, age, not handedness, so this is a genuinely open axis. Result, stated as found:
a modest, direction-consistent right>left performance gap (0.01-0.06 depending on run) that
**test-time mirroring does not reliably close** — it helped left-handers in one unseeded run and hurt
them in the seed-matched one, "flipped sign on retraining," which the project explicitly retracted
rather than quietly dropping. Reporting the retraction alongside the finding is itself a small
methodological point worth keeping: a single-run bias-mitigation result that reverses under a second
run is exactly the kind of claim the seed-variance protocol exists to catch.

## What this is *not* a paper about

- **Not a shot-classification paper.** LSTM and Transformer are statistically tied (macro-F1
  0.785 +/- 0.009 vs 0.766 +/- 0.014); no architecture tried beats the two incumbents. The classifier is
  a necessary supporting experiment (the technique scorer's exemplar matching and the coaching app both
  need a shot label) and a benchmark data point (§3), not a contribution in its own right.
- **Not a state-of-the-art AQA-accuracy paper.** The honest Spearman number (0.588-0.595, best
  configuration) does not beat the VAE baseline (0.601) on the metric the rest of the field reports,
  and both trail I3D-AE-LSTM's 0.84 on overall-score Spearman (with the caveat, already documented,
  that I3D-AE-LSTM's comparison is on a split this project independently found contains duplicate
  strokes). Do not lead with raw Spearman; it invites a comparison this project does not win.
- **Not a coaching-LLM paper.** The LLM-phrasing layer was prototyped, then dropped from both the app
  and the paper's scope per a supervisor conversation (2026-10-17) — mentioned, if at all, as a demo
  detail in the app section, not a contribution.
- **Not (yet) a 3D-pose, ball-tracking, or bat-angle paper.** Those remain enrichment
  (`scope_report.md` §2) with no measured research payoff, and the bat U-Net has a confirmed domain-gap
  failure on real footage (test IoU 0.81 on product-photo data, confidence 0.04-0.10 on real clips).

## Open items before this framing is submission-ready

1. **The seed-2 contrastive-residual checkpoint (the number this whole section leans on) needs to
   exist as a reproducible artifact, not just prose in `PROGRESS.md`.** See the app-integration PR
   stack: it was never saved anywhere retrievable and is being regenerated now.
2. **Statistical significance, not just descriptive variance, on the headline partial-rho gain.** The
   3-seed mean +/- std (0.122 +/- 0.013 vs baseline's single-run 0.019) reads as a clear gap, but a
   paired or bootstrap test against the baseline (ideally also seed-swept) would make "6x improvement"
   a number a reviewer can't wave away as "different seeds, different luck."
3. **Decide, explicitly, whether §3 (the benchmark) is a releasable artifact** (manifests + dedupe
   clusters + splits published alongside the paper) or stays internal. This changes how strong a claim
   §3 can make.
4. **The abstract/intro should lead with §1, use §2 as "and we show what follows once you fix the
   evaluation," and relegate §3-6 to a methods/supporting-results framing** — not the chronological
   order this work was discovered in, which is prose-log order, not argument order.
