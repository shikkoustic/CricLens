# The existing pipeline, inferred across 53 papers

What the field actually does today for: **shot/action classification → pose estimation → quality rating**.
This is the "current methodology" a paper must reproduce before claiming to improve on it.

Sources: the 53 papers in `reading_log.md`. Numbers are as reported by their authors.

---

## The pipeline at a glance

```
[0] Clip acquisition and trimming
        ↓
[1] Person detection  →  SUBJECT SELECTION  →  tracking
        ↓                 (mostly skipped)
[2] Pose estimation on the subject's box
        ↓
[3] Temporal handling: windowing / phase segmentation
        ↓
[4] Representation: RGB stream, pose stream, or both
        ↓
   ┌────┴────┐
[5a] CLASSIFY   [5b] RATE
   └────┬────┘
        ↓
[6] Evaluation
        ↓
[7] Feedback generation  (new, rare)
```

---

## Stage 0 — Clip acquisition and trimming

Every paper assumes a **short, pre-trimmed clip containing one action**. Nobody in the AQA line works on
untrimmed video; that is a separate literature (temporal action localisation).

| Domain | Typical clip |
|---|---|
| Cricket (CricketVision) | 28 frames, middle 12 used, **4 per phase** |
| Cricket (CricShot10 / 10k) | 15–25 frames, ~0.9–2.5 s |
| Diving / gymnastics (AQA) | **96–103 frames**, split into 9–10 overlapping 16-frame snippets, stride 10 |

**Note the mismatch.** AQA methods are designed for ~100-frame sequences; cricket clips are 4–28 frames.
I3D-AE-LSTM had to adapt the baselines it compared against (changing C3D stride from (2,2,2) to (1,2,2),
bypassing augmentation) purely because its clips were too short. **Anyone porting AQA methods to cricket
inherits this problem.**

## Stage 1 — Person detection, subject selection, tracking

**This is the stage the field skips, and the one that matters most for broadcast footage.**

| Approach | Who | Cost of getting it wrong |
|---|---|---|
| Pre-cropped clips | most cricket classification papers | unmeasured |
| Hand-drawn / annotated boxes | CricketVision, FineParser (masks) | — |
| **One heuristic rule** | I3D-AE-LSTM: *"the person with the largest y-coordinate"* | unmeasured |
| Detector only, no subject prior | — | **75.8** Mean-Top1 (PoseC3D, FineGym) |
| Tracking from one GT box | PoseC3D (Siamese-RPN) | **85.3** |
| GT box every frame | PoseC3D | **92.0** |

Detection: YOLO variants almost universally. Tracking: ByteTrack (associates low-confidence boxes in a
second pass, 80.3 MOTA / 77.3 IDF1 on MOT17), or Siamese-RPN.

**The evidence.** PoseC3D's ablation is the only quantification of this stage in the whole corpus: a
**16.2-point swing** between detector-only and ground-truth boxes. Their conclusion: *"the prior of the
interested person is extremely important… other persons like the audience or referee are unrelated."*
Their remedy — ground-truth boxes — does not exist at scale in broadcast footage.

## Stage 2 — Pose estimation

Top-down is universal: detect person → crop → estimate joints on the crop.

| Model | Used by | Notes |
|---|---|---|
| **ViTPose** | I3D-AE-LSTM, CricLens | 80.9 AP COCO; most robust under corruption (PoseBench: 78.84 clean → 65.02 corrupted) |
| OpenPose | CricTAL | same group as I3D-AE-LSTM, different paper, no cross-comparison |
| HRNet | PoseC3D | 2D beats 3D on FineGym (92.0 vs 87.0) |
| MediaPipe | Sensors 2023 cricket | lightweight |
| YOLOPose | Sci. Reports 2026 cricket | low latency |

**Settled questions in the literature:**
- **2D beats 3D.** PoseC3D: estimated 2D keypoints outperform sensor-collected and estimated 3D.
- **Lifted 3D does not help.** FrameLift 90.0 / VideoLift 90.2 vs HRNet-2D 92.0 — lifting *hurt*.
- **Heatmaps beat coordinates**, especially for weak pose estimators (~2% on FineGym).
- **Top-down beats bottom-up** (HRNet top-down 93.6 vs bottom-up 93.0 on NTU-60).

Post-processing: interpolation for missing joints — **cubic spline** (I3D-AE-LSTM) or **linear** (CricTAL);
normalisation to [0,1] or L2 / min-max per frame.

## Stage 3 — Temporal handling

Three strategies, in increasing sophistication:

**(a) None — pool the whole clip.** MTL-AQA, USDL. Simple, and still competitive (0.9044, 0.9273).

**(b) Explicit phase segmentation.** Segment into semantic steps, then compare step-wise.
- TSA / FineDiving — procedure segmentation + cross-attention → 0.9203
- MCoRe — multi-stage → 0.9232
- CricTAL — cricket phases, TCN, sliding window (best w=7), 35-frame classification window → mAP@0.5 64.45%
- MS-TCN — the standard segmentation architecture: multi-stage dilated convolutions + smoothing loss

**(c) Learned temporal parts, no phase labels.** Temporal Parsing Transformer: learnable queries recover
atomic patterns, with a ranking loss (temporal order) and sparsity loss (distinctiveness) → **0.9451**,
beating explicitly-segmented TSA.

**The calibration that matters.** FineDiving with *ground-truth* phase boundaries gains only
**~1 Spearman point** (0.8925 → 0.9029). And CricTAL's mAP sits in a 60–66% band *regardless of
architecture*. **Phase segmentation has a low ceiling — (c) is the smart default.**

## Stage 4 — Representation

| Stream | Standard choice | Notes |
|---|---|---|
| **RGB** | **I3D pretrained on Kinetics** | Universal across MTL-AQA, USDL, CoRe, TSA, TPT, FineParser. C3D is the older default; SlowFast underperforms |
| **Pose / skeleton** | ST-GCN or **PoseC3D** | PoseC3D far more robust: dropping one limb keypoint per frame costs it **<1%**, costs a GCN **14.3%** |
| **Both** | weighted fusion `F = αFv + βFp` | I3D-AE-LSTM |
| Cricket classification | **EfficientNet-B0/V2-S + GRU** | Kang 92.25%, CricShotNet 89% |

**The most useful single number here:** I3D-AE-LSTM's own ablation — pose-only **0.79**, adding the entire
I3D RGB stream **0.84**. The RGB pipeline buys 0.05.

## Stage 5a — Classification head

CNN feature extractor per frame → recurrent aggregator (GRU/LSTM) → softmax. The cricket standard.

| Method | Dataset | Accuracy |
|---|---|---|
| EfficientNet-B0 + GRU (Kang) | CricShot10, 10 cls | **92.25%** |
| EfficientNetV2-S + GRU (CricShotNet) | CricShot10k, 15 cls | **89%** |
| VGG16-GRU (Sen et al.) | CricShot10 | 93% claimed, **57.7%** reproduced |
| 3D CNN + optical flow + YOLOPose | IPL-2023, 7 cls | 91.37 / 92% |
| ST-GCN (skeleton) | Kinetics-Skeleton | 30.7% top-1 |

**Every one of these uses a random, ungrouped split.** Kang's re-implementation study found reproductions
of 46.0 / 55.6 / 57.7% against claims of 96 / 99.2 / 93%.

## Stage 5b — Rating head

The clearest methodological arc in the corpus:

**Generation 1 — direct regression.** Features → MLP → score. MTL-AQA **0.9044**; I3D-AE-LSTM **0.84**
(cricket, 5 outputs).

**Generation 2 — score distributions.** USDL: treat the label as a Gaussian over the score range, train
with KL divergence, because judges are subjective. MUSDL adds one path per judge. **0.9273**.

**Generation 3 — contrastive regression (current consensus).** Predict the *difference* against an exemplar
with shared attributes, not an absolute score.
- CoRe + group-aware regression tree (depth 5, 10 exemplars, voting) → **0.9512**
- TSA — contrastive + procedure-aware → 0.9203
- TPT — contrastive over learned parts → 0.9451
- FineParser — contrastive + spatial and temporal parsing → **0.9435** (SOTA, human-centric)

**Multi-output rating (per body part) is almost unique to cricket.** Everywhere else a single score is
predicted. I3D-AE-LSTM's five outputs are the exception — and its five per-part correlations differ by
only **0.003**, which is worth scrutiny.

## Stage 6 — Evaluation

- **Spearman rank correlation (SRCC)** — universal for rating. Fisher z-averaging across action classes.
- **Relative ℓ2 distance (R-ℓ2)** — introduced by CoRe, normalised by the score range; now standard alongside SRCC.
- **AIoU@0.5 / @0.75** — for temporal parsing quality.
- **Accuracy / precision / recall / F1** — classification. **mAP** — localisation.
- **Splits: random.** I3D-AE-LSTM 75/15/10 "through trial and error"; FineDiving 75/25; Kang 70/15/15
  seed 27. **No paper in the corpus uses subject- or match-grouped splits.** That is where dataset-hygiene
  work (perceptual-hash dedupe, leakage audits) enters — from outside this literature.

## Stage 7 — Feedback generation (new and rare)

- **MTL-AQA** (2019) established score + commentary as a joint task.
- **ExpertAF** (2025): video + 3D pose → commentary + retrieved expert demo + generated corrected pose;
  weakly supervised from expert commentary via an LLM; up to **3×** over baselines in human evaluation.
- **CoachMe** (2025): **reference-based** — compare the learner against a reference along temporal and
  physical axes. G-Eval 1.83 vs GPT-4o 1.39.
- **Hard constraint:** frontier VLMs shown to be **near chance** at judging action quality, with a bias
  toward predicting "correct execution" regardless of evidence. **Ground the language model in measured
  pose features; do not ask it to watch the video and judge.**

---

## The canonical pipeline, as one recipe

If you asked the field "what should I build?", the answer today is:

1. Trimmed clip, single performer, ~100 frames
2. Person box assumed given (**the weak point**)
3. Top-down 2D pose, ViTPose or HRNet; no 3D lifting
4. I3D Kinetics features over 9–10 overlapping 16-frame snippets; optional pose stream
5. Learned temporal parts (TPT-style) rather than explicit phase labels
6. Contrastive regression against exemplars (CoRe/FineParser-style)
7. Report SRCC **and** R-ℓ2 on a random split
8. Optional: an LLM producing commentary from grounded features

## Where CricLens diverges, and why that is the paper

| Stage | Field default | CricLens | Justification |
|---|---|---|---|
| 1 | assume the box | **learned batter finder**, 91.2% vs 57% rules | PoseC3D shows a 16.2-pt downstream swing |
| 3 | ~100-frame clips | contact-anchored window, cut-aware | broadcast clips are short and contain cuts |
| 4 | I3D RGB + pose | pose-first | their own ablation: RGB worth 0.05 |
| 5b | single score | five body-part scores | the coaching guideline's own taxonomy |
| 6 | random split | **match-grouped**, dedupe-verified | 1,612 duplicates found; CricShot10's own split leaks |

Five divergences, each with evidence behind it. That is the shape of a methods contribution.
