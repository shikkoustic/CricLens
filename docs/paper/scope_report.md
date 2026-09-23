# CricLens — scoping report

For deciding what this project actually ships. Written to be discussed, not executed.
Companion documents: `aligned_papers.md` (literature), `dossier.md` (per-paper notes),
`claude_project_context.md` (background).

---

## 1. The ambitious vision, as the repository declares it

Reconstructed from `CLAUDE.md` and `PROGRESS.md`. This is the full plan — not everything here
will survive contact with a deadline.

> CricLens analyses cricket batting videos: it finds the batter, extracts body joints from ball
> release to follow-through, classifies the shot, scores technique per body part, and writes
> coaching feedback, served as a web app.

Sixteen components:

| # | Component | Status | Cost | Serves |
|---|---|---|---|---|
| 1 | Dataset: 6 sources, dedupe, match-grouped splits | **BUILT** | — | paper + everything |
| 2 | Batter finder (learned striker selection) | **BUILT** — 91.2% vs 57% rules | — | paper |
| 3 | Pose extraction (ViTPose, windowed) | **BUILT** — 16,460 train-ready | — | everything |
| 4 | Ball / bat / stumps detector (YOLO11s) | **BUILT** — mAP50 0.81 | — | bat angle, demo |
| 5 | Image-quality profile + enhancement study (D1–D3) | **BUILT** | — | methods justification |
| 6 | Zero-padding re-run (4,461 edge clips) | **NEXT** | ~40 min GPU | blocks training |
| 7 | Frame-rate normalisation (25 vs 30 fps) | **NEXT** | ~hours, CPU | blocks training |
| 8 | **Shot classifier** | **NOT STARTED** | days | paper core |
| 9 | **Per-body-part technique scorer** | **NOT STARTED** | 1–2 weeks | **paper core** |
| 10 | Left/right handedness bias audit | **NOT STARTED** | ~1 day | paper, cheap win |
| 11 | Pitch calibration → stride in cm, swing speed in m/s | **PLANNED** | ~1 week | enrichment |
| 12 | Bat U-Net + bat angle from shape moments | **PLANNED** | ~1 week | enrichment |
| 13 | TrackNet ball tracking | **PLANNED** | 1–2 weeks | enrichment |
| 14 | 3D pose lifting for camera angles | **PLANNED** | ~1 week | **risky — see §4** |
| 15 | Coaching LLM | **PLANNED** | ~1 week | demo, CV, paper §5 |
| 16 | Web app | **PLANNED** | 1–2 weeks | **CV only** |

## 2. The central scoping insight

**The paper and the CV want different things, and only two components serve both.**

- **The paper needs:** 6, 7 (prerequisites) → **8, 9, 10** (the contribution). Nothing else is required.
- **The CV / demo needs:** **15, 16** — a working system someone can try. Research value: near zero.
- **Items 11, 12, 13, 14 serve neither strongly.** They are enrichment: interesting, expensive, and not
  load-bearing for either goal.

That is the decision in one line: **8 + 9 + 10 is the paper. 15 + 16 is the portfolio piece.
11–14 are optional, and 14 is actively risky.**

## 3. What the literature says about each core component

### Component 8 — shot classifier
The field's consensus input is I3D over ~96 frames. But for a pose-based system the real choice is
ST-GCN versus PoseC3D. PoseC3D's own evidence is decisive for our setting: dropping one limb keypoint
per frame costs PoseC3D **<1%** and costs a GCN **14.3%**. With six sources of noisy broadcast poses,
that robustness gap is the deciding argument.

Comparison points: Kang's honest **92.25%** on CricShot10 (10 classes) and CricShotNet's **89%** on
CricShot10k (15 classes) — both on random, ungrouped splits. Our number will be on match-grouped splits
and should be expected to be **lower**, and that is the point: it will be the first honest one.

### Component 9 — technique scorer (the paper's centre)
The AQA methodology arc is clear: direct regression (MTL-AQA, 0.9044) → score distributions
(USDL, 0.9273) → contrastive regression against an exemplar (CoRe 0.9512, TSA 0.9203, TPT 0.9451,
FineParser 0.9435).

Cricket's only entry, I3D-AE-LSTM, is at the *first* stage — plain regression, no contrastive component,
no distributions, no learned temporal parsing. **Target: SRCC 0.84 on CricketVision scores.**

Three design questions to settle before building:
- Regression, score distribution, or contrastive regression? (CoRe-style is the consensus; CricketVision's
  high-scoring strokes are ready-made exemplars.)
- Explicit phase segmentation, learned queries, or neither? (TPT suggests learned queries beat explicit
  segmentation; FineDiving shows perfect phase boundaries are worth only ~1 Spearman point.)
- Pose-only or pose+RGB? (Their own ablation: pose-only 0.79, +RGB 0.84 — RGB costs a lot for 0.05.)

### Component 10 — handedness bias audit
4,400 right- and 2,492 left-handed labelled strokes. Existing fairness work in action recognition covers
skin tone, gender and age — **not handedness**. Cheapest genuine contribution available: measure the
left/right performance gap, test whether mirror-canonicalisation closes it.

## 4. Known risks — test, don't assume

- **Component 14 (3D lifting) may hurt.** PoseC3D found lifted 3D poses performed *worse* than the
  original 2D poses on fine-grained action recognition (90.0 / 90.2 vs 92.0 on FineGym). Verify before
  investing a week.
- **Phase segmentation has a low ceiling.** FineDiving with ground-truth boundaries gains ~1 Spearman
  point. CricTAL sits at 60–66% mAP@0.5 regardless of architecture. Do not make this the project.
- **Our five-part taxonomy is what ExpertAF criticises** — "a fixed taxonomy of errors designed separately
  for each activity." Defensible (it is the coaching guideline's own taxonomy, formalised with two cricket
  experts) but must be defended explicitly.
- **The per-part scores may not be discriminative.** Published per-part correlations span 0.003 across
  five body parts. If our own label correlation matrix confirms collinearity, component 9's framing has
  to change — possibly into the paper's headline finding.

## 5. Open scoping questions

1. **Is the paper about the scorer, the benchmark, or both?** A scorer paper needs a strong number
   against 0.84. A benchmark paper needs the leakage audit, honest re-baselining and released manifests.
   Both is more work but a stronger story.
2. **How much does the paper need the shot classifier?** It is the natural companion to the scorer and
   the thing every prior cricket paper does — but if the contribution is rating, the classifier may be
   a supporting experiment rather than a headline.
3. **Does the coaching LLM belong in the paper or only in the demo?** MTL-AQA established score +
   commentary as a legitimate research framing, and CoachMe shows reference-based generation works. But
   evaluating generated text properly is its own project.
4. **Do we publish the dataset artefacts?** Manifests, dedupe clusters and grouped splits — SoccerNet-style.
   This converts housekeeping into a citable contribution, at the cost of doing it properly.
5. **What happens to the image-processing work?** Under the new framing it is Methods justification rather
   than a contribution. Does it stay as an ablation section, an appendix, or a separate short paper?
6. **Which of 11–14 survive, if any?** Bat angle (12) is the most defensible — it produces a coaching
   output the scorer cannot. Pitch calibration (11) gives real-world units, which is a genuine
   differentiator, but needs the calibration to actually work on broadcast footage.

## 6. A defensible minimum

If everything went wrong and only the essentials shipped, this would still be a real paper:

1. The leakage-audited multi-source benchmark (already built)
2. The batter finder, with its downstream effect measured (already built; measurement outstanding)
3. A modern AQA scorer on CricketVision, compared honestly against 0.84
4. The handedness audit
5. The preprocessing decisions as a Methods ablation

That is four weeks of work on top of what exists, not four months. Everything beyond it — ball tracking,
3D lifting, pitch calibration, the LLM coach, the web app — is upside.
