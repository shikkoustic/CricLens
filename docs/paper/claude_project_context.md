# CricLens — project context
(Paste into the Context / project-knowledge field of the claude.ai CricLens project.)

## 1. Who I am and what I have been asked to do

I am Shikhar Srivastava, an undergraduate at Bennett University (2026). CricLens is one
project submitted to two courses, and it is now also being written up as a **journal research
paper** (explicitly a journal paper, not a conference paper).

My supervisor's brief, in his words and order:
1. **Read research papers in the same problem domain** — not just cricket, but the wider
   domain of video analysis and automated analytics generation. Cite **at least 20-30 papers**.
2. **From those papers, identify the pipeline / methodology that exists today** — what is the
   standard way this problem is solved right now.
3. **Find a novelty** — something new we can do that the existing pipelines do not.
4. **Implement the existing pipeline first**, as a baseline, so there is something to compare
   against.
5. **Then introduce the novelty**, optimise the results with it, and in the paper demonstrate
   that our new pipeline outperforms the existing methodology.

So the paper needs: a credible related-work section, a reproduced baseline, a novel
contribution, and a measured improvement over that baseline.

## 2. What CricLens is

Given a broadcast clip of a cricket delivery, CricLens:
1. finds the batter (striker) among everyone in frame — bowler, keeper, umpire, fielders
2. extracts 17 2D body joints per frame across a window from roughly ball release to
   follow-through
3. (next phase) classifies the shot played
4. (next phase) scores technique per body part — head, shoulders, hands, hips, feet
5. (later) generates coaching feedback, served as a web app

**Steps 1-2 are complete and validated. Steps 3-5 have not been started.**

Two courses, one project:
- **IMD (Intelligent Model Design)** — the deep learning: pose estimation, detectors, the
  learned batter finder, sequence models for shot classification, a technique scorer, a bias
  audit, and an LLM coach.
- **IVA / CSET344 (Image and Video Processing)** — the classical CV. Self-imposed rule: a
  syllabus technique is only included if the app genuinely needs it, and its effect must be
  measured. Several experiments concluded "this does not help, do not use it", and those
  negative results are kept.

Repo: github.com/shikkoustic/CricLens (private). Everything must run free — heavy compute
goes to Kaggle in ~30-minute resumable chunks; no paid APIs.

## 3. Data collected so far

**22,420 deduplicated clips at 480p**, assembled from six public sources:

| Source | Clips | What it carries |
|---|---|---|
| CricketVision (UJ-AQA-CricketVision, WACV 2025) | 8,441 strokes | stroke type, front/back foot, handedness, **1-10 technique scores for head / shoulders / hands / hips / feet per phase**, phase timestamps, batter bounding boxes |
| CricShot10k (IEEE Access 2026) | 10,091 | 15 shot classes |
| CricShot10 (Sensors 2021) | 1,750 | 10 shot classes |
| Cricket Shots IPL 2023 (Kaggle) | 1,922 | 7 classes |
| amittalmale (Kaggle) | 562 | 8 classes |
| KUCricShot (ICCIT 2023) | 1,266 | 4 classes |

All broadcast-derived: academic, non-commercial use only, never redistributed.

**Deduplication.** Two-stage perceptual hashing — 64-bit pHash LSH for recall, then 256-bit
pHash sequence alignment at the best time offset for verification. Removed **1,612 verified
duplicates**. Stage 1 alone was insufficient: deliveries from one match share camera angle and
pitch, which chained whole matches into clusters of up to 297 clips.

**Splits.** StratifiedGroupKFold grouped by source match (or duplicate cluster where no match
id exists), stratified on shot class. **No match or duplicate cluster appears in two splits.**

**Data problems we found ourselves (these matter for the paper's argument):**
- CricketVision ships **1,156 strokes duplicated across annotator folders** — identical frames,
  same stroke index, filed under different video names.
- The CricShot10 HuggingFace mirror's **own published split leaks**: 157 duplicate clusters
  inside it, **75 spanning its train/val/test folders**.

**Label taxonomy.** Unified to 8 shots (drive, defence, flick_glance, pull_hook, cut, sweep,
lofted, scoop) plus "other", with an off/leg/straight side axis that is handedness-independent.
Original labels always retained. Foot and handedness only from explicit annotation, never
guessed from shot type.

**Detection data.** 23,799-image ball/bat/stumps detection set and a 5,999-image polygon set,
assembled from Roboflow and Kaggle sources with cross-source near-duplicate merging.

## 4. The pipeline as built

- Camera-cut detection via HSV histogram differences (the analysis window must never cross a cut)
- YOLO11m + ByteTrack tracking every person in the clip
- A learned **batter finder**: HistGradientBoostingClassifier over per-track features (upright
  shape, position, size stability — the bowler grows as he runs at camera — Striker/bat
  detections from a player-type model, plus per-clip relative features).
  **91.2% top-1 on unseen matches, versus 57% for the hand-written rule it replaced.**
- **ViTPose-Base** (chosen over YOLO11-pose), fp16, on the 12%-padded batter box
- Analysis window = [contact − 0.8s, contact + 0.6s], never crossing a camera cut
- A clip is "usable" when ≥80% of window frames have the batter at mean joint confidence ≥0.5
- Ball/bat/stumps detector: YOLO11s, **test mAP50 0.81**

**Validation.** A 400-clip manual audit: **92.5% correct striker** overall — 100% on four
sources, 89% on CricShot10k where the wicketkeeper is the main confusion. Plus a "twin clip"
check using deliveries that appear in both CricketVision and another source.

**Output.** `pose_index.parquet`, 22,420 rows, **16,460 train-ready** (train 11,891 / val 2,307 /
test 2,262), estimated 98-99% correct striker after filtering on finder confidence ≥0.9.
7,072 clips carry the CricketVision per-body-part technique scores.

## 5. Image-processing findings (the IVA half)

- **No enhancement method improves pose.** Global histogram equalisation, contrast stretching
  and homomorphic filtering all actively hurt it (global HE costs 0.036 wrist confidence).
- **But CLAHE raises bat detection inside the batter box from 11% to 19%**, and a manual check
  of 101 disagreement frames confirmed ~89% of the extra detections are the real bat.
  Conclusion: CLAHE before bat detection only, never before pose.
- **Zero padding beats clamping** when the batter's crop runs off the frame — edge-case pose
  error down 20-60%, winning in 97% of paired cases. Affects 4,461 clips.
- Adaptive median crushes salt-and-pepper noise (joint error 0.029 → 0.005), beating plain median.
- **NL-means has the best SSIM and the worst pose error** — image-quality metrics do not
  predict task accuracy.
- Darkness is a non-problem (14 clips). CricShot10k is genuinely degraded (Laplacian sharpness
  86 vs 341-713 elsewhere).

## 6. Literature research done so far

53 papers were downloaded and read — 18 in full, 35 at the level of abstract, method, results
and conclusion. A 20-paper shortlist was produced, ranked by alignment with CricLens.

### Tier 1 — the ten that matter most
1. **I3D-AE-LSTM**, Moodley & van der Haar, WACV 2025 — the direct competitor. Built the
   CricketVision dataset we use. Part-wise cricket AQA, **SRCC 0.84**.
   https://openaccess.thecvf.com/content/WACV2025/html/Moodley_I3D-AE-LSTM_A_2-Stream_Autoencoder_for_Action_Quality_Assessment_using_a_WACV_2025_paper.html
2. **PoseC3D**, Duan et al., CVPR 2022 — https://arxiv.org/abs/2104.13586
3. **Modern DL Approaches for Cricket Shot Classification**, Kang 2025 — https://arxiv.org/abs/2510.09187
4. **CricTAL**, Moodley & van der Haar, ICCVW 2025 —
   https://openaccess.thecvf.com/content/ICCV2025W/SAUAFG/html/Moodley_CricTAL_Introducing_Temporal_Activity_Localisation_using_pose_estimation_to_identify_ICCVW_2025_paper.html
5. **A Decade of Action Quality Assessment**, Yin, Parmar et al., IJCV — https://arxiv.org/abs/2502.02817
6. **FineDiving + TSA**, Xu et al., CVPR 2022 — https://arxiv.org/abs/2204.03646
7. **CoRe (Group-aware Contrastive Regression)**, Yu et al., ICCV 2021 — https://arxiv.org/abs/2108.07797
8. **PoseBench**, Ma et al. 2024 — https://arxiv.org/abs/2406.14367
9. **LIME-Eval**, Li, Zhao & Guo 2024 — https://arxiv.org/abs/2410.08810
10. **CoachMe**, Yeh et al., ACL 2025 — https://aclanthology.org/2025.acl-long.1413/

### Tier 2 — cite precisely
11. **FineParser**, CVPR 2024 — https://openaccess.thecvf.com/content/CVPR2024/papers/Xu_FineParser_A_Fine-grained_Spatio-temporal_Action_Parser_for_Human-centric_Action_Quality_CVPR_2024_paper.pdf
12. **USDL / MUSDL**, CVPR 2020 — https://arxiv.org/abs/2006.07665
13. **Temporal Parsing Transformer**, ECCV 2022 — https://arxiv.org/abs/2207.09270
14. **ExpertAF**, CVPR 2025 — https://arxiv.org/abs/2408.00672
15. **Can VLMs Judge Action Quality?**, CVPRW 2026 — https://arxiv.org/abs/2604.08294
16. **Data Leakage Detection and De-duplication**, 2023 — https://arxiv.org/abs/2304.02296
17. **MTL-AQA**, Parmar & Morris, CVPR 2019 — https://arxiv.org/abs/1904.04346
18. **The Pros and Cons**, Doughty et al., CVPR 2019 — https://arxiv.org/abs/1812.05538
19. **AQA Survey + Benchmark**, Zhou et al., Pattern Recognition 2026 — https://arxiv.org/abs/2412.11149
20. **ViTPose**, Xu et al., NeurIPS 2022 — https://arxiv.org/abs/2204.12484

### Three papers that could not be fetched automatically
- **CricShot10k** (IEEE Access 2026) — IEEE blocks automated access
- **I3D-AE-LSTM journal extension** (Expert Systems with Applications) — Elsevier paywall
- **A Comprehensive Review of Computer Vision in Sports** (Applied Sciences 12(9):4429) — MDPI blocks

## 7. The competitive landscape

The direct competitors are **Moodley & van der Haar (University of Johannesburg)**, who built
the CricketVision dataset whose scores we train on:
- **I3D-AE-LSTM** (WACV 2025): ViTPose + I3D two-stream autoencoder → MLP predicting five
  body-part scores. **SRCC 0.84** (pose-only 0.79). Trained on the execution phase only,
  **4 frames per sample**. Batter chosen by "the person with the largest y-coordinate".
  Splits "through trial and error, 75, 15, 10" — random and ungrouped. Per-body-part
  correlations: head 0.84146, shoulder 0.84040, hands 0.83865, hips 0.84027, feet 0.83844 —
  **a spread of 0.003 across five body parts.**
- **CricTAL** (ICCVW 2025): pose-only phase localisation, best TCN **90.4% accuracy,
  mAP@0.5 64.45%** — every architecture lands in a narrow 60-66% mAP band.

Other key numbers:
- **Kang 2025** re-implemented seven published cricket classifiers: 96% → 46.0%, 99.2% → 55.6%,
  93% → 57.7%. His own EfficientNet-B0 + GRU: **92.25%** — but on a stratified *random*,
  ungrouped split of CricShot10, the dataset where we found 75 cross-split duplicate clusters.
- **PoseC3D** ablates person-box quality on FineGym: **Detection 75.8 / Tracking 85.3 /
  GT boxes 92.0** Mean-Top1 — a 16.2-point swing from which person-box the pose model gets.
  It also found that **lifted 3D poses performed worse than the original 2D poses**, and that
  dropping one limb keypoint per frame costs PoseC3D <1% but costs a GCN 14.3%.
- **FineDiving/TSA**: with *ground-truth* phase boundaries, Spearman rises only ~1 point
  (0.8925 → 0.9029) — so perfect phase segmentation has a low ceiling.
- **A Decade of AQA** survey states *"the biggest AQA dataset has more than 20000 samples"* —
  CricLens has 22,420 clips / 16,460 train-ready, i.e. frontier scale.

## 8. Candidate novelty directions

1. **Task-conditioned preprocessing** — restoration should be selected per *downstream
   consumer*, not per image-quality score. Evidence in hand: CLAHE helps the bat detector and
   hurts the pose model; best-SSIM is worst-for-pose. PoseBench studies robustness at
   *training* time only; LIME-Eval argues for task-based evaluation but stays in low-light and
   uses detection as a proxy for human preference. **Nobody asks which restoration to apply per
   consumer.** (Serves the IVA course.)
2. **Subject selection as a measured pipeline stage** — PoseC3D shows person-box quality is
   worth ~16 points downstream, but its remedy is ground-truth boxes, unavailable at scale in
   broadcast footage. No learned selector recovers that gap automatically, and it has never
   been measured for *quality assessment* rather than recognition. (Serves the IMD course.)
3. **A leakage-audited, multi-source cricket benchmark** with honest re-baselining — Kang
   documented the reproducibility crisis but never diagnosed near-duplicate leakage as the cause.
4. **Testing whether per-body-part scores are genuinely discriminative** — the published
   per-part correlations differ by 0.003, suggesting one number predicted five times. If true,
   nobody has yet demonstrated working per-body-part cricket assessment.
5. **A handedness bias audit** — we hold 4,400 right- and 2,492 left-handed labelled strokes;
   existing fairness work covers skin tone, gender and age, not handedness.

## 9. Direction of travel — what happens next

Immediate, before any training:
- Compute the correlation matrix between the five CricketVision part scores in
  `manifest.parquet`. If they are highly collinear, novelty #4 is real and becomes the paper's
  headline. This costs one afternoon and no GPU.

Data fixes that block training:
- Zero-padding re-run on the 4,461 clips whose batter crop leaves the frame (pose only, reuse
  saved boxes, ~40 min Kaggle GPU).
- Frame-rate normalisation — clips are 25 or 30 fps; resample joint sequences to one rate.

Then, per my supervisor's brief:
- **Reproduce the existing pipeline**: I3D-AE-LSTM on CricketVision (target SRCC 0.84);
  EfficientNet-B0 + GRU for shot classification (target ~92%); a modern AQA baseline
  (CoRe or FineParser); ST-GCN vs PoseC3D as skeleton baselines. The AQA-Benchmark code from
  Zhou et al. (paper #19) is the fastest honest route.
- **Then add the novelty** and show the improvement.

Known risks to test rather than assume:
- Our planned 3D-lifting step may hurt — PoseC3D found lifted 3D poses worse than 2D.
- Heavy investment in phase segmentation has a low ceiling (~1 Spearman point).
- Our five-body-part taxonomy is exactly the "fixed, hand-designed error taxonomy" that
  ExpertAF criticises — defensible because it is the coaching guideline's own taxonomy
  (GCSE batting phases, formalised with two cricket experts), but it must be defended explicitly.

## 10. How to work with papers in this project

When I name a paper or give you a link:
- **Fetch and read the paper yourself from the URL.** Use the links in section 6, or search for
  the paper. Do not ask me to upload a PDF unless fetching genuinely fails.
- If a source blocks you (IEEE Xplore, Elsevier/ScienceDirect and MDPI are known to block
  automated access), say so plainly and *then* ask me to upload it — I can get it through my
  university library.
- Read the actual paper before answering. Do not answer from general knowledge about action
  quality assessment.
- Quote the paper and cite the section or page. If something is not stated in the paper, say
  "the paper doesn't say" rather than filling the gap.
- Distinguish clearly between what a paper claims, what it actually demonstrates, and what I am
  inferring.
