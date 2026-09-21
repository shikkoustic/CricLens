# CricLens paper dossier — full-text reading notes

One entry per paper, read from the publisher PDF. Format: citation → problem → pipeline → data →
numbers → limitations → relevance to CricLens. All numbers are as reported by the authors.

Reading status is tracked in `reading_log.md`. Abstract-level context and the gap analysis live in
`related_work.md`.

---

# Cluster A — Cricket (the direct problem)

## A1. Moodley & van der Haar — I3D-AE-LSTM — WACV 2025, pp. 5470–5478
Tevin Moodley, Dustin van der Haar (University of Johannesburg). Dataset: `github.com/dvanderhaar/uj-aqa-cricketvision`.

- **Problem.** Body-part-level action quality assessment of cricket batting strokes.
- **Pipeline.** Crop to annotated batter bbox → ViTPose-large (stated "COCO, 25 keypoints" — internally
  inconsistent, COCO is 17) with YOLO person detection @ conf 0.5 → cubic spline interpolation for short
  frames/keypoints → normalise [0,1] → **two autoencoder streams** (I3D over frames; LSTM 256→128 over
  pose) → weighted fusion `F = αFv + βFp` → LSTM aggregation → MLP with 5 linear outputs (head, shoulders,
  hands, hips, feet), scores 0–9.
- **Data.** UJ-AQA-CricketVision: 8,540 samples, YouTube footage of First-Class International Test matches,
  VIA-annotated by 2 annotators against a guideline built with 2 cricket experts. 5,571 right-handed /
  2,969 left-handed. Scoring: head & shoulders by angle 0–15°, hands & hips on a 0–4 number line, feet by
  angle 0–15°, all relative to lines drawn through the head. Competency classes from the execution phase:
  poor 1,478 / average 2,415 / good 2,689 / excellent 1,958.
- **Scope.** 28 frames/sample, middle 12 used, **4 frames per phase; trained on the execution phase only**
  (ball pitch → bat-ball contact). Buildup and follow-through deferred to future work.
- **Numbers.** Headline **SRC 0.84**. Ablations: pose-only ViTPose **0.79**; C3D+ViTPose 0.83;
  SlowFast+ViTPose 0.80; I3D+ViTPose **0.84**. Baselines re-run on their data: I3D-DAE (Zhang 2024) **0.60**,
  C3D-LSTM (Parmar & Morris) **0.68**. Per-part SRC: head 0.84146 / shoulder 0.84040 / hands 0.83865 /
  hips 0.84027 / feet 0.83844. Also claims **0.9767 on MTL-AQA** (sequences truncated to 65 frames).
- **Protocol issues.**
  1. Batter selection is one heuristic line: *"we found the person with the largest y-coordinate in the
     frames and used those keypoints as the batter for every sample."*
  2. Splits: *"Through trial and error, the train, validation, and test splits were 75, 15, 10."* — random,
     **no grouping by match, batter or source video**.
  3. Score smoothing: replicating scores across frames before a windowed average *"improved SRC performance
     by up to 20%"*.
  4. Per-part SRC spread is **0.003 across five body parts** — consistent with near-collinear labels.
- **Stated weaknesses.** Left-skewed error distribution (systematic overestimation); fails on strokes
  scoring 0–2 due to data scarcity.
- **Relevance.** The direct baseline. Its subject selection, split protocol and per-part collinearity are
  CricLens's three openings.

## A2. Moodley & van der Haar — CricTAL — ICCV 2025 Workshops (SAUAFG), pp. 2738–2745
- **Problem.** Temporal activity localisation of the three stroke phases, pose-only, to feed downstream AQA.
- **Pipeline.** **OpenPose** keypoints → (25,3) as (x, y, confidence) → L2 or min-max normalisation per
  frame → linear interpolation for missing keypoints/frames → two framings: non-overlapping *classification
  window* of **35 frames** (= mean frames per stroke) and overlapping *sliding window* centred per frame
  (sizes 7/11/14/21/31/41 tried; **7 best**). Models: LSTM, RNN, TCN, Transformer.
- **Numbers.**

  | Model | Approach | Acc | mAP@0.5 | avg mAP@[0.3:0.7] |
  |---|---|---|---|---|
  | LSTM | classification window | 94.5% | 63.83% | — |
  | RNN | classification window | 76.7% | 63.87% | **66.10%** |
  | Transformer | classification window | 80.1% | 63.87% | 63.81% |
  | LSTM | sliding window | 93.3% | 62.45% | 63.28% |
  | **TCN** | sliding window | 90.4% | **64.45%** | 65.72% |
  | Transformer | sliding window | 89.3% | 60.88% | 62.23% |

- **Key observation.** Frame accuracy is high (90–94%) but **localisation sits in a narrow 60–66% mAP@0.5
  band for every architecture** — the model class barely matters, suggesting the bottleneck is the
  representation or the label definition. Their §4.1 concedes that misclassified boundaries directly
  undermine downstream phase-wise scoring.
- **Inconsistency.** ViTPose in the WACV paper, OpenPose here, same dataset, no cross-comparison.
- **Useful.** Their Table 2 collects prior cricket video understanding: Gupta & Balan 2020 (C3D+GRU stroke
  localisation, mTIoU 0.71); Abbas et al. 2022 (YOLO+RetinaNet delivery segmentation, 90.0%); Raval et al.
  2023 (K-means + scoreboard heuristic replay detection, 94–96%); Shingrakhia et al. 2021 (SGRNN-AM +
  HRF-DBN highlight summarisation, 96.3%).

## A3. Kang — Modern Deep Learning Approaches for Cricket Shot Classification — arXiv:2510.09187 (10 Oct 2025)
Sungwoo Kang, Korea University. Code: `github.com/hpicsk/CricShot10_Baselines`.

- **Problem.** Unified re-implementation and benchmarking of seven cricket shot classifiers.
- **Setup.** CricShot10, **1,888 samples / 10 classes** (~180–200 per class), **stratified random split,
  fixed seed 27**, 70/15/15 → 1,320 / 284 / 284. Single A100. Models implemented "strictly following the
  details described in the respective papers".
- **The reproduction gap.**

  | Original claim | Re-implemented |
  |---|---|
  | Kumar/Balaji, modified LRCN — 96% (on a reduced **5-class** subset) | **46.0%** |
  | Bhat et al. — 99.2% | **55.6%** |
  | Sen et al. (Sensors 2021), VGG16-GRU — 93% | **57.7%** |
  | ViT + RNN — 98.9% | **10.6%** |
  | Attention network — 99.19% | **40.5%** |
  | Kang's own EfficientNet-B0 + GRU | **92.25%** |

- **The opening.** Kang attributes the gap to *"differences in dataset splits, evaluation code, or minor
  implementation details not specified in the papers"* — **he never diagnoses near-duplicate leakage and
  never audits the data.** His own 92.25% is on a stratified random, ungrouped split of CricShot10, the
  dataset where the CricLens manifest build found **157 verified duplicate clusters, 75 spanning its own
  train/val/test folders**.
- **Relevance.** Legitimises a leakage-audited re-baselining contribution, and leaves the root cause open
  for us to supply.

---

# Cluster B — Action Quality Assessment (method core)

## B1. Parmar & Morris — MTL-AQA — CVPR 2019 (arXiv:1904.04346)
Paritosh Parmar, Brendan Tran Morris (University of Nevada, Las Vegas).

- **Problem.** Does multitask learning improve AQA? Learn spatio-temporal features explaining three tasks:
  fine-grained action recognition, **commentary generation**, and score estimation.
- **Pipeline.** Two architectures — C3D-AVG and MSCADC — with shared features and three heads. Factorised
  action recognition: position, armstand, rotation type, #somersaults, #twists.
- **Data.** MTL-AQA: **1,412 diving samples from 16 events**, individual + synchronous, male + female,
  3m springboard + 10m platform. Annotations: difficulty degree, 7 individual judge scores, final score,
  dive class, and natural-language commentary.
- **Numbers.** C3D-AVG-MTL **0.9044** rank correlation (SOTA at the time); C3D-AVG-STL 0.8960;
  MSCADC-MTL 0.8612; MSCADC-STL 0.8472.
- **Findings.** MTL generalises better than STL; **representations from action-recognition models are not
  sufficient for AQA and must be learned**.
- **Relevance.** The precedent for coupling scores with generated language — i.e. CricLens's coaching
  output. Also the origin of the "score + commentary" framing our paper will extend.

## B2. Tang et al. — USDL / MUSDL — CVPR 2020 (arXiv:2006.07665)
Yansong Tang, Zanlin Ni, Jiahuan Zhou, Danyang Zhang, Jiwen Lu, Ying Wu, Jie Zhou (Tsinghua + Northwestern).

- **Problem.** Regression ignores the **intrinsic ambiguity of score labels** from multiple subjective judges.
- **Pipeline.** Video → sliding window → N overlapping 16-frame clips → **I3D** (Kinetics-pretrained) → 3 FC
  layers (shared weights) → per-clip score vectors → temporal average pooling → softmax → predicted score
  distribution. Ground truth is a **Gaussian centred on the label with std σ** (σ = the uncertainty
  hyper-parameter). Loss = **KL divergence** between predicted and generated distributions. Inference takes
  the argmax of the distribution. **MUSDL** adds a multi-path head — one path per judge — with a shared I3D
  backbone, then applies the sport's rule (discard top-2 and bottom-2 judges, multiply by difficulty degree).
- **Data.** AQA-7 (1,189 samples / 7 sports; 803 train / 303 test, trampoline excluded), MTL-AQA (1,412),
  JIGSAWS (surgical).
- **Numbers.** MTL-AQA: USDL 0.9066, MUSDL **0.9273** (vs C3D-AVG-MTL 0.9044). JIGSAWS avg: USDL 0.63,
  MUSDL 0.70.
- **Relevance.** **Directly applicable to CricLens's label problem.** CricketVision scores are subjective,
  single-annotation-team, with no inter-rater ceiling (our datasheet shows the apparent duplicate
  annotations are copies, not independent ratings). USDL is the principled treatment; it also gives us a
  way to report calibrated uncertainty in coaching feedback.

## B3. Yu et al. — CoRe + GART — ICCV 2021 (arXiv:2108.07797)
Xumin Yu, Yongming Rao, Wenliang Zhao, Jiwen Lu, Jie Zhou (Tsinghua).

- **Problem.** Regressing an absolute score from one video suffers from large inter-video score variation.
- **Pipeline.** Reformulate as **contrastive regression**: predict the *relative* score against an exemplar
  video with shared attributes (category, difficulty). I3D (Kinetics) backbone; **group-aware regression
  tree (GART)** of depth 5, node feature dim 256, which coarse-to-fine bins the relative score. 103 frames
  per clip → 10 overlapping 16-frame snippets. **10 exemplars per test video with multi-exemplar voting.**
- **New metric.** Relative ℓ2-distance (**R-ℓ2**), normalised by the action's score range — a stricter
  companion to Spearman.
- **Numbers.** MTL-AQA w/o DD **0.9341** (R-ℓ2 0.365); w/ DD **0.9512** (R-ℓ2 0.260). JIGSAWS avg **0.85**
  (vs MUSDL 0.70) — a very large jump. AQA-7: +8.95%, +2.32%, +8.83%, −6.82%, +3.01%, +2.25% per class
  vs USDL. Ablation on MTL-AQA: I3D+MLP 0.9381 → +GART 0.9403 → +CoRe 0.9512.
- **Relevance.** The dominant modern paradigm, and a natural fit for cricket: CricketVision's high-scoring
  strokes are ready-made exemplars, and comparison-to-a-reference is exactly how a coach reasons (cf. CoachMe).

## B4. Xu et al. — FineDiving + TSA — CVPR 2022 (arXiv:2204.03646)
Jinglin Xu, Yongming Rao, Xumin Yu, Guangyi Chen, Jie Zhou, Jiwen Lu (Tsinghua).

- **Problem.** Holistic video features give non-transparent, poorly interpretable scores.
- **Pipeline.** **Temporal Segmentation Attention (TSA)**: procedure segmentation → procedure-aware
  cross-attention between query and exemplar steps → fine-grained contrastive regression. I3D backbone,
  96 frames → 9 snippets of 16 frames, stride 10. 75/25 train/test.
- **Data.** **FineDiving: 3,000 video samples, 52 action types, 29 sub-action types**, with step-level
  temporal boundaries — the first AQA dataset with fine-grained procedure annotation.
- **Numbers.** w/o dive number: TSA **0.8925** / R-ℓ2 0.4782 / AIoU@0.5 80.71 / AIoU@0.75 30.17
  (USDL 0.8302, MUSDL 0.8427, CoRe 0.8631). w/ DN: TSA **0.9203** / R-ℓ2 0.3420 / AIoU@0.5 82.51
  (CoRe 0.9061).
- **Critical ablation for us.** With **ground-truth step boundaries** (TSA†) the score reaches 0.9029 (w/o
  DN) and 0.9310 (w/ DN) — i.e. **perfect phase segmentation buys only ≈ +1 point of Spearman**. That is a
  direct, citable counterweight to CricTAL's premise that better boundaries are the path forward.

## B5. Bai et al. — Temporal Parsing Transformer — ECCV 2022 (arXiv:2207.09270)
Yang Bai, Desen Zhou, Songyang Zhang, Jian Wang, Errui Ding, Yu Guan, Yang Long, Jingdong Wang
(Durham / Baidu VIS / Shanghai AI Lab / Warwick).

- **Problem.** Get part-level temporal structure **without any part-level labels**.
- **Pipeline.** I3D clip features → transformer decoder with a set of **learnable queries** representing
  atomic temporal patterns → fixed number of temporally ordered part representations → contrastive
  regression on part representations. Two novel losses on the decoder's cross-attention: a **ranking loss**
  (queries must respect temporal order) and a **sparsity loss** (parts must be discriminative).
- **Numbers.** MTL-AQA w/o DD **0.9451** / R-ℓ2 0.3222 (CoRe 0.9341, TSA-Net 0.9422).
  AQA-7 avg **0.8715** (CoRe 0.8401, TSA-Net 0.8476); per class: Diving 0.8969, Gym Vault 0.8043,
  BigSki 0.7336, BigSnow 0.6965, Sync3m 0.9456, Sync10m 0.9545.
- **Relevance.** **The strongest argument against needing CricTAL-style phase labels at all.** If learnable
  queries recover atomic phases unsupervised and beat explicitly segmented TSA, then CricLens does not need
  hand-annotated buildup/execution/follow-through boundaries — a significant simplification, and a
  defensible design choice to test.

## B6. Xu et al. — FineParser — CVPR 2024
Jinglin Xu, Sibo Yin, Guohao Zhao, Zishuo Wang, Yuxin Peng (USTB + Peking University).

- **Problem.** Human-centric AQA: parse the performer in **space as well as time**.
- **Pipeline.** Four modules — **SAP** (spatial action parser, extracts human-centric foreground action
  regions), **TAP** (temporal action parser, divides into steps), **SVE** (static visual encoder,
  ResNet34), **FineReg** (fine-grained contrastive regression). I3D backbone for SAP/TAP; 96 frames →
  9 snippets of 16, stride 10.
- **Data.** **FineDiving-HM** — FineDiving re-annotated with human-centric **foreground action masks**
  (largest class 107B with 35,287 mask instances; smallest 101 instances).
- **Numbers.** FineDiving-HM: FineParser **ρ 0.9435 / R-ℓ2 0.2602** (TSA 0.9324/0.3022, CoRe 0.9308/0.3148,
  MUSDL 0.9241, USDL 0.8830, I3D+MLP 0.8776, C3D-AVG 0.8371, MSCADC 0.7688, C3D-LSTM 0.6969).
  Temporal parsing: **AIoU@0.5 0.9946 / AIoU@0.75 0.9467** vs TSA 0.9239 / 0.5007 — a very large gain.
  Spatial parsing: MAE 0.0408, Fβ 0.1273, Sm 0.8357.
- **Honest note for our gap analysis.** FineParser **does** address spatial localisation of the performer —
  so "AQA ignores where the subject is" is too strong a claim. The accurate claim is narrower and still
  holds: FineParser segments *the* performer in a single-performer scene from mask supervision; it never
  has to decide **which of several people** is the subject, and no AQA paper measures the cost of getting
  that choice wrong. CricLens's contribution is subject *selection* under ambiguity, not subject
  *segmentation*.

## B7. An, Qi & Ma — MCoRe (Multi-stage Contrastive Regression) — arXiv:2401.02841 (Jan 2024)
Qi An, Mengshi Qi, Huadong Ma (Beijing University of Posts and Telecommunications).
- Stage-level segmentation + graph-based contrastive regression. **FineDiving: SRCC 0.9232, R-ℓ2 0.3265,
  AIoU@0.5/0.75 = 98.26 / 79.17.**
- Relevance: confirms the "segment into stages, then contrast" recipe is the field's consensus.

## B8. Yin, Parmar et al. — A Decade of Action Quality Assessment — IJCV (read arXiv:2502.02817 v1, Feb 2025)
Hao Yin, Paritosh Parmar (co-first), Daoliang Xu, Yang Zhang, Tianyou Zheng, Weiwei Fu.
Resources: `haoyin116.github.io/Survey_of_AQA/`.

- PRISMA review of **200+ papers**; arXiv v1 counts **26 publicly available AQA datasets across 9 domains**
  (the published IJCV version is cited elsewhere as 214 papers / 33 datasets — **check which figure the
  final version uses before citing**).
- **Challenges it names, verbatim in structure:**
  - *Action*: inherent complexity; **"there is currently no sample that can be used as a gold standard"**;
    constructing gold-standard action samples is called out as a promising future direction.
  - *Dataset*: **"Scant Scale"** — *"the biggest AQA dataset has more than 20000 samples"*;
    **"Narrow Action"**; **"Coarse Annotation"** — *"a single score cannot allow the model to learn and
    utilize features sufficiently… The limitation of fine-grained labels has indeed restricted the
    advancement of AQA models in interpretability."*
  - *Methodology*: computational complexity & latency; **blackbox models / interpretability**;
    **fragile robustness** (multimodal methods ignore performance under missing modalities).
  - *Future directions*: large-scale multi-action real-world datasets; lightweight, efficient, real-time
    solutions — explicitly motivated by *"improving the lives of socio-economically disadvantaged
    communities"* who cannot run heavy models; interpretability via neural-symbolic or causal methods.
- **Why this matters enormously for our framing.** The survey's own headline number is that the largest
  AQA dataset has ~20,000 samples. **CricLens has 22,420 clips (16,460 train-ready), with part-level
  scores on 7,072.** So we can credibly claim scale comparable to the largest dataset in the field, in a
  domain the survey lists as under-served, addressing three of its four named dataset challenges
  (scale, annotation granularity, real-world sourcing).

## B9. Zhou, Cai, Wang, Shum & Liang — A Comprehensive Survey of AQA: Method and Benchmark — Pattern Recognition (2026), DOI 10.1016/j.patcog.2026.113933 (read arXiv:2412.11149 v2)
Kanglei Zhou (Tsinghua), Ruizhi Cai, Liyuan Wang, Hubert P.H. Shum (Durham), Xiaohui Liang (Beihang).
Resources: `ZhouKanglei.github.io/AQA-Survey`, `Awesome-AQA`, **AQA-Benchmark (code)**.

- PRISMA-based; a **modality-driven hierarchical taxonomy** and, crucially, **a unified benchmark that
  re-runs representative AQA methods under standardised protocols**, reporting both accuracy and
  **computational efficiency**.
- Motivation stated: *"heterogeneous datasets and evaluation protocols hinder reproducibility; methods are
  often validated in narrowly defined settings, limiting generalization."*
- **Practical value to us: their benchmark code is the fastest honest route to reproducing USDL / CoRe /
  TSA baselines on CricLens data** rather than reimplementing each from scratch.

---

# Cluster C — Skill assessment and feedback generation

## C1. Doughty, Damen & Mayol-Cuevas — Who's Better? Who's Best? — CVPR 2018 (arXiv:1703.09913)
University of Bristol.
- **Problem.** Assess skill from video without absolute scores, via **pairwise ranking**.
- **Pipeline.** Siamese network with shared weights; novel ranking loss that learns discriminative features
  when a video pair differs in skill and shared features when the pair is comparable.
- **Data.** Introduces **EPIC-Skills**: Surgery (JIGSAWS — Knot-Tying, Needle-Passing, Suturing),
  Dough-Rolling (CMU-MMAC), Drawing (new), Chopstick-Using (new).
- **Numbers.** 70–83% correctly ordered pairs across four tasks.
- **Relevance.** The alternative supervision signal when absolute scores are unreliable — which is exactly
  the situation with CricketVision's single-annotation-team labels (see B2).

## C2. Doughty, Mayol-Cuevas & Damen — The Pros and Cons — CVPR 2019 (arXiv:1812.05538)
- **Problem.** In long videos, most of the footage is irrelevant to skill.
- **Pipeline.** **Rank-aware temporal attention**: separate attention branches trained to attend to
  high-skill evidence ("pros") and low-skill evidence ("cons"), with a rank-aware loss and a disparity
  loss (`Ldisp`) that forces the attention branches to beat uniform attention.
- **Data.** EPIC-Skills + new **BEST** dataset (long videos).
- **Numbers.** Rank-aware attention **80.3%** (EPIC-Skills) / **81.2%** (BEST) pairwise accuracy — **+4.3%
  and +5.4%** over their CVPR 2018 work. Baselines: Last Segment 76.8/61.0, Softmax Attention 74.5/72.3.
  Rank-aware loss alone gives ~5% average improvement on BEST, up to +10.4% on some tasks.
- **Relevance.** **A ready-made interpretable-feedback mechanism.** "Which part of the stroke cost you
  marks" is precisely what a cons-attention map over the swing gives you — without needing per-phase
  labels. Strong alternative to CricTAL-style supervised phase boundaries.

## C3. Grauman et al. — Ego-Exo4D — CVPR 2024 (arXiv:2311.18259 v4)
Kristen Grauman, Andrew Westbury, Lorenzo Torresani, Kris Kitani, Jitendra Malik et al. (FAIR Meta + academia).
- **Data.** **1,286 hours**, 5,035 takes (1–42 min each), **740 participants**, time-synchronised ego +
  multiple exo views, precisely localised in a metric gravity-aligned frame. **200,000+ hours of annotator
  effort.** Annotations include **expert commentary** critiquing performance.
- **Four benchmark families.** ego-exo relation; ego(-exo) keystep recognition; **ego(-exo) proficiency
  estimation**; ego body/hand pose.
- **Proficiency task.** Two variants — *demonstrator* proficiency (4 classes: Novice, Early Expert,
  Intermediate Expert, Late Expert; derived from participant surveys + expert commentary, top-1 accuracy)
  and *demonstration* proficiency (temporal localisation of good execution / tips, scored with an
  **L1-distance-based mAP** rather than tIoU). Baselines use TimeSFormer and Omnivore features.
  *(Exact baseline accuracies did not extract cleanly from the PDF text layer — re-check Table 10 in the
  published version before quoting a number.)*
- **Relevance.** The reference point for modern skill assessment, and the source of ExpertAF's supervision.
  Note it is *purpose-captured multi-view* data — the opposite of CricLens's uncontrolled broadcast setting,
  which is a difference worth stating explicitly in our introduction.

## C4. Ashutosh, Nagarajan, Pavlakos, Kitani & Grauman — ExpertAF — CVPR 2025 (arXiv:2408.00672 v3)
UT Austin, FAIR Meta, CMU.
- **Problem.** *"Current methods for skill-assessment from video only provide scores or compare
  demonstrations, leaving the burden of knowing what to do differently on the user."*
- **Pipeline.** Input: a demonstration `V = {RGB video, 3D pose sequence P, skill level S}`. Outputs three
  things: (1) **free-form expert commentary** `T̂ = Ft(V)`, (2) **expert demonstration retrieval**
  `V̄ = Fr(V)` (argmin over a retrieval set), and (3) **expert pose generation** `P = Fg(V)`. Training data
  is **weakly supervised**, built by combining Ego-Exo4D expert commentary with a strong LLM.
- **Domains.** Basketball, soccer, rock climbing (bouldering).
- **Numbers.** Outperforms off-the-shelf video models **by as much as 3×** in direct human evaluation.
- **The sentence that matters most for us.** *"All the prior work assumes a fixed taxonomy of errors, and
  the taxonomy is designed separately for each activity."* They cite Fitness-AQA and Action Quality as
  producing outputs like *"knees inward error, shallow squat error"*. **CricLens's five-body-part scoring
  is exactly such a fixed, hand-designed taxonomy** — so we must either defend it (it comes from a
  coaching guideline, i.e. it is the domain's own taxonomy, not an arbitrary one) or go beyond it. Worth
  addressing head-on in the paper rather than being caught by a reviewer.

## C5. Yeh, Su, Chen, Lin, Ku, Chiu, Hu & Ku — CoachMe — ACL 2025 (Long Papers, pp. 29131+; arXiv:2509.11698)
Academia Sinica, National Tsing Hua University, National Taiwan University. `motionxperts.github.io`.
- **Problem.** Generate corrective coaching instructions, not just descriptions.
- **Pipeline.** **Reference-based**: compare the learner's motion against a reference along **temporal and
  physical** axes, mimicking a coach identifying deviations from a standard motion and focusing on the
  crucial body parts.
- **Data.** Figure Skating (FS) — ground-truth timestamps provided by a figure-skating coach — and Boxing (BX).
- **Numbers (FS).** CoachMe with reference + aligned segment: BLEU@1 22.2, Rouge 20.0, BertScore 11.7,
  **G-Eval 1.83**; with ground-truth segments BLEU@1 24.7 / BertScore 26.5 / G-Eval 1.73; Basic CoachMe
  (no reference) G-Eval 1.53; **GPT-4o 1.39**, LLaMa 3.2 1.31, MiniCpm 1.37, ViLa 1.27. Human ratings (FS):
  CoachMe 26.6% Good vs GPT-4o 20.3%. Reported as **+31.6% over GPT-4o on figure skating, +58.3% on boxing**.
- **Relevance.** **The single most transferable design for CricLens's coaching module.** A reference-based
  comparison needs no new annotation — CricketVision's high-scoring strokes are the references, matched by
  shot type and handedness. It also composes naturally with CoRe-style contrastive regression (B3): the
  same exemplar serves both the score and the text.

## C6. Monte e Freitas, Henriques, Rei & Martins — Can Vision Language Models Judge Action Quality? — CVPR 2026 Workshops (SAUAFG) (arXiv:2604.08294 v1)
Sword Health, Instituto Superior Técnico / Universidade de Lisboa, INESC-ID.
- **Setup.** Gemini 3, Qwen3-VL and InternVL3.5 families, across body-weight exercises (overhead press, 339
  videos; squat, 224 videos), **FineFS** (figure skating), **MTL-AQA** (353 diving videos, 5–7 judges), over
  tasks including visual question answering and technical guideline verification, with and without skeleton
  input, grounding instructions, reasoning structures and in-context learning.
- **Findings.** All evaluated VLMs perform **only marginally above random chance**. Two systematic biases:
  (i) a tendency to **predict correct execution regardless of visual evidence**, and (ii) **sensitivity to
  superficial linguistic framing**. Contrastive task reformulation yields minimal improvement — *"pointing
  to a fundamental difficulty with fine-grained movement quality assessment."*
- **Methodological detail worth borrowing.** On FineFS they **mask the on-screen scoreboards before
  evaluation "to prevent data leakage"** — an explicit precedent for the leakage-hygiene argument.
- **Relevance.** This is the citation that justifies CricLens's whole architecture: **do not ask a VLM to
  judge the stroke; compute pose-derived measurements and let the language model verbalise them.**

## C7. Bianchi & Liotta — SkillFormer — arXiv:2505.08665 v5 (Oct 2025)
Free University of Bozen-Bolzano. Parameter-efficient unified multi-view architecture for proficiency
estimation on Ego-Exo4D; **4.5× fewer trainable parameters** than baselines, outperforming them in the Exos
and Ego+Exos settings. Companion: **ProfVLM** (arXiv:2509.26278), a lightweight video-language model for
the same task.

## C8. Liao, Vakanski & Xian — A Deep Learning Framework for Assessing Physical Rehabilitation Exercises — IEEE TNSRE (arXiv:1901.10435)
University of Idaho. The clinical sibling of sports AQA: per-exercise movement quality scores from
sensor-captured skeletons, on UI-PRMD. Establishes the "continuous quality score from a skeleton sequence"
formulation that CricLens's part-wise scorer inherits.

## C9. Ismail-Fawaz, Devanne, Berretti, Weber & Forestier — A Standardized Benchmark for Skeleton-Based Rehabilitation Assessment — arXiv:2507.21018
IRIMAS Université de Haute-Alsace, University of Florence, Monash. A standardised benchmark on KIMORE and
UI-PRMD — the same "the field's protocols are inconsistent, here is a unified one" move that Kang (A3) and
Zhou et al. (B9) make. Useful precedent for framing our leakage-audited benchmark as a contribution type
that journals accept.

---

# Cluster D — Pose, tracking, skeleton and temporal machinery

## D1. Xu, Zhang, Zhang & Tao — ViTPose — NeurIPS 2022 (arXiv:2204.12484 v3)
University of Sydney, JD Explore Academy.
- Plain, non-hierarchical ViT backbone + lightweight decoder. Scales **100M → 1B parameters**.
- **80.9 AP on MS COCO test-dev.** Variants ViTPose-B / L / H.
- Relevance: the pose backbone CricLens uses; cite for the architecture choice, and pair with F1 (PoseBench)
  which independently finds ViT backbones most robust under corruption.

## D2. Yan, Xiong & Lin — ST-GCN — AAAI 2018 (arXiv:1801.07455)
CUHK.
- Spatial-temporal graph convolution over body joints; spatial edges follow the skeleton, temporal edges
  link each joint to itself across time.
- **Kinetics-Skeleton: 30.7% top-1 / 52.8% top-5** (vs Deep LSTM 16.4/35.3, Temporal Conv 20.3/40.0,
  Feature Encoding 14.9/25.8). **NTU-RGB+D: 81.5% X-Sub / 88.3% X-View.**
- Relevance: the reference skeleton baseline our shot classifier must beat.

## D3. Duan, Zhao, Chen, Lin & Dai — PoseC3D ("Revisiting Skeleton-based Action Recognition") — CVPR 2022 (arXiv:2104.13586 v2)
CUHK, UT Austin, Shanghai AI Lab, NTU, SenseTime.
- **3D heatmap volumes** instead of joint graphs, processed by a 3D CNN.
- **Robustness (the number that matters).** Randomly dropping one limb keypoint per frame costs
  PoseConv3D **<1% Mean-Top1**; the GCN loses **14.3%**. Even with noise-robust training the GCN still
  loses 1.4%, plus a 1.1% penalty on clean input.
- **Bounding-box ablation (FineGYM Mean-Top1) — critically important for our Gap 1:**

  | Human proposals | GYM Mean-Top1 |
  |---|---|
  | Detection (no prior about which person) | **75.8** |
  | Tracking (GT box in frame 1 + Siamese-RPN) | **85.3** |
  | GT boxes in every frame | **92.0** |

  Their words: *"the prior of the interested person is extremely important: even weak prior knowledge
  (1 GT box per video) can improve the performance by a large margin"*, and *"other persons like the
  audience or referee are unrelated."*
- Other ablations: 2D keypoints beat 3D (HRNet-2D 92.0 vs VIBE-3D 87.0 on GYM); **lifted 3D poses do not
  help and perform worse than the original 2D poses**; heatmaps beat coordinates (Heatmap-HRNet 93.6 vs
  Coordinate-HRNet 93.2, and a ~2% gap for low-quality pose estimators).
- **Two consequences for CricLens.** (i) PoseC3D is the right skeleton backbone for noisy broadcast poses,
  and the robustness gap is testable on our data. (ii) **The lifted-3D result is a warning about our
  planned 3D-lifting step** — on FineGYM it actively hurt. We should test before committing to it.

## D4. Pavllo, Feichtenhofer, Grangier & Auli — VideoPose3D — CVPR 2019 (arXiv:1811.11742 v2)
ETH Zürich, Facebook AI Research, Google Brain. Dilated temporal convolutions over 2D keypoint sequences,
plus semi-supervised back-projection training. The standard 2D→3D lifting route — **but see D3's finding
that lifted 3D poses did not help fine-grained action recognition.**

## D5. Zhang et al. — ByteTrack — ECCV 2022 (arXiv:2110.06864)
- Associates **every** detection box, including low-confidence ones, in a second matching stage — which is
  what keeps an occluded track alive.
- **MOT17 test: 80.3 MOTA, 77.3 IDF1, 63.1 HOTA at 30 FPS on a V100.**
- Relevance: the tracker in CricLens's pose pipeline; its low-score second association is why the batter's
  track survives occlusion by bowler and keeper.

## D6. Abu Farha & Gall — MS-TCN — CVPR 2019 (arXiv:1903.01945)
University of Bonn. Multi-stage dilated temporal convolutions; each stage refines the previous stage's
predictions. Classification loss + a **smoothing loss that penalises over-segmentation**. Evaluated on
50Salads, GTEA, Breakfast; up to **+12.6% accuracy** over the prior state of the art. MS-TCN++ followed in
2020. The architecture class CricTAL (A2) applies to cricket.

## D7. Shao, Zhao, Dai & Lin — FineGym — CVPR 2020 (arXiv:2004.06704)
CUHK-SenseTime Joint Lab. **29K videos, 99 fine-grained gymnastic classes**, three-level hierarchy
(event → set → element) with temporal annotations at both action and sub-action level. The template for
how a fine-grained sports taxonomy should be structured — directly relevant to CricLens's 8-shot +
off/leg/straight design.

---

# Cluster E — Sports video understanding at scale

## E1. Giancola, Amine, Dghaily & Ghanem — SoccerNet — CVPR Workshops 2018 (arXiv:1804.04527)
**500 complete games** from six European leagues, 2014–2017, **764 hours**, 6,637 temporal annotations at
one-minute resolution for Goal / Card / Substitution. Baselines: **67.8% mAP** for video classification,
**49.7%** for event spotting.

## E2. Deliège et al. — SoccerNet-v2 — CVPR Workshops 2021 (arXiv:2011.13367)
**~300k annotations** over the same 500 broadcasts; adds action spotting, camera-shot segmentation with
boundary detection, and a novel replay-grounding task.

## E3. Cioppa et al. — SoccerNet-Tracking — arXiv:2204.06918. Multi-object tracking in soccer broadcasts.

**Why E1–E3 matter to us:** they are the model for *how to release a broadcast-sports benchmark* — public
manifests, defined splits, multiple tasks on one corpus. If CricLens publishes its manifests, dedupe
clusters and match-grouped splits, this is the precedent to cite for the format.

## E4. Huang, Liao, Chen, İk & Peng — TrackNet — 2019 (arXiv:1907.03698)
National Chiao Tung University. Heatmap-based network taking **consecutive frames** to learn flight
patterns, for balls that are *"small, blurry, and sometimes with afterimage tracks or even invisible"*.
- **Precision 99.7% / recall 97.3% / F1 98.5%** on the primary labelled dataset;
  **95.3% / 75.7% / 84.3%** under 10-fold cross-validation with additional partially labelled videos.
- **TrackNetV4** (arXiv:2409.14543) adds motion attention maps.
- Relevance: the reference for CricLens's planned ball-tracking module. Note the large gap between the
  headline numbers and the cross-validated ones — a useful caution about single-split reporting.

## E5–E7. Surveys.
- Zhu et al., *A Survey of Deep Learning in Sports Applications: Perception, Comprehension, and Decision*
  (arXiv:2307.03353) — the three-layer framing CricLens spans end to end.
- Wu et al., *A Survey on Video Action Recognition in Sports* (arXiv:2206.01038).
- Naik et al., *A Comprehensive Review of Computer Vision in Sports*, Applied Sciences 12(9):4429 (2022)
  — **not retrievable from this environment (MDPI blocks automated access); fetch manually.**

---

# Cluster F — Robustness, preprocessing and dataset hygiene (the IVA contribution)

## F1. Ma, Zhang, Cao & Tao — PoseBench — arXiv:2406.14367 v2 (Sep 2024)
University of Sydney, JD Explore Academy, NTU Singapore. `xymsh.github.io/PoseBench/`.
- **60 model variants from 15 methods**, across COCO-C, OCHuman-C and AP10K-C (human + animal), with
  **10 corruption types in four groups**: blur & noise, compression & colour alteration, lighting, and
  occlusion/dropout, at multiple severities. Metrics include mAP and **mRR (mean resilience rate)**.
- **Findings.** *"Human pose models are particularly vulnerable to compression and blur."* ViT backbones
  (ViTPose, HRFormer) dominate both clean and corrupted performance — **ViTPose-H: 78.84 mAP clean,
  65.02 mAP corrupted**, beating DEKR HRNet-W32 (68.64 clean) at every severity. Pre-training,
  post-processing and large transformer backbones help robustness; **input resolution does not**.
- **The crucial scoping fact for CricLens.** PoseBench's §4.3 "Robustness Enhancement" examines
  **training-time factors only** — backbone, pre-training, input resolution, post-processing, and
  **data augmentation** (the four augmentation groups are applied during *training*). **It never evaluates
  test-time image restoration: it does not ask which filter to apply to an already-degraded frame, and it
  does not propagate the choice to a downstream task.** Gap 2 survives this paper intact.
- Two of its findings corroborate CricLens D1/D3 independently: vulnerability to compression and blur
  (we found blur, impulse noise and compression hurt pose most, darkness and low contrast barely), and the
  superiority of ViT-based pose models.

## F2. Wang, Jin, Liu, Liu, Qian & Luo — When Human Pose Estimation Meets Robustness — CVPR 2021 (arXiv:2105.06152)
HUST, HKU, SenseTime. Builds **COCO-C, MPII-C, OCHuman-C** with 15 corruption types × 5 severities across
noise, blur, weather and digital categories, plus adversarial algorithms.

## F3. Hoang et al. — Improving the Robustness of 3D Human Pose Estimation — CVPR Workshops 2024 (arXiv:2312.06797)
**Human3.6M-C and HumanEva-I-C** — temporary occlusion, motion blur and pixel noise for video-based 3D
pose lifters.

## F4. Schiappa, Biyani, Kamtam, Vyas, Palangi, Vineet & Rawat — Large-scale Robustness Analysis of Video Action Recognition Models — arXiv:2207.01398 v2
UCF CRCV, IIT Kanpur, Microsoft Research.
- Findings: models are **very robust to temporal perturbations** on Kinetics/UCF/HMDB but not on SSv2 —
  i.e. *"the importance of temporal information varies based on the dataset and activities"*. Transformers
  are generally more robust to compression perturbations. **"High model capacity does not necessarily mean
  more robustness"** — contradicting earlier work. Introduces **UCF101-DS**, a real-world distribution-shift
  dataset (no simulation).
- Relevance: supports measuring robustness on *real* degraded clips, not only synthetic corruption — which
  is exactly CricLens's D2 design (800 flagged real clips) alongside the synthetic D3 study.

## F5. Li, Zhao & Guo — LIME-Eval — arXiv:2410.08810 v2 (Oct 2024)
Tianjin University.
- **Argument.** The common practice of evaluating low-light enhancement by *retraining* a detector on
  enhanced images is **prone to overfitting** and unreliable. They introduce **LIME-Bench** (an online
  platform collecting human preferences for low-light enhancement) and **LIME-Eval**, which uses detectors
  **pre-trained on standard-lighting data, without annotations**, judging enhancement via an energy-based
  assessment of output confidence maps.
- **How CricLens differs and extends it.** LIME-Eval stays inside low-light enhancement and uses detection
  as a *proxy for human preference*. CricLens evaluates restoration by its effect on the **actual end task**
  (pose → part-wise quality assessment) and finds **the correct choice differs between two consumers inside
  the same pipeline** (CLAHE helps bat detection, hurts pose). Note our protocol already satisfies
  LIME-Eval's own recommendation: we use **fixed pre-trained** models (ViTPose, our YOLO detector) and never
  retrain on enhanced images, so the overfitting critique does not apply to us.

## F6. Adimoolam, Poullis & Averkiou — Data Leakage Detection and De-duplication in Large Scale Geospatial Image Datasets — arXiv:2304.02296 v2
CYENS Centre of Excellence, Concordia, Cyprus University of Technology.
- On the AICrowd Mapping Challenge dataset: **~89% of training images are duplicates** (exact or
  augmented), and **~93–97% of validation images are also present in the training split**.
- Contributes a **perceptual-hashing de-duplication and leakage-detection pipeline** for large image
  datasets.
- **The closest published methodological precedent for CricLens's dedupe contribution** — same technique
  (perceptual hashing), same argument (leakage inflates reported performance and harms generalisation),
  different domain. Cite it to establish that "audit the dataset, then re-baseline" is an accepted
  contribution type.

## F7. Theiner & Ewerth — TVCalib — WACV 2023 (arXiv:2207.11709)
Sports field registration in soccer treated as **camera calibration** rather than plain homography
estimation, optimising camera pose and focal length from **segment correspondences** (lines, point clouds)
by minimising segment reprojection error. Companion: **PnLCalib** (CVIU) via points-and-lines optimisation.
The reference for CricLens's planned IVA pitch-calibration module (HSV pitch segmentation → morphology →
connected components → Canny + Hough crease lines → pixels-to-centimetres).

## F8. Băltăret,u, Benschop & van Gemert — Identifying Ethical Biases in Action Recognition Models — arXiv:2604.17971 (Apr 2026)
TU Delft. Audits human action recognition models using **synthetic video with controlled appearance
variation** (skin tone, ethnicity), isolating attributes to expose behavioural bias; concludes
*"model accuracy alone does not capture the full picture."* Covers skin tone and ethnicity —
**not handedness**, which remains open for CricLens's left/right audit.

---

# Cluster A (continued) — the rest of the cricket literature

## A4. Moodley, van der Haar & Noorbhai — Automated recognition of the cricket batting backlift technique — Scientific Reports 12 (2022)
University of Johannesburg + Habib Noorbhai.
- Binary classification: **lateral backlift (LBBT)** vs **straight backlift (SBBT)**. Architectures compared:
  AlexNet, Inception V3, Inception-ResNet-V2, **Xception**.
- **Xception best: loss 0.03, ~98.2% accuracy**; precision 100%, recall 95%, F1 97–98%.
- **Essential context the abstract hides: the dataset is tiny — "40 images, resulting in a total of 200
  images".** A 200-image, two-class problem. Cite the finding, but note the scale; it is not evidence that
  cricket technique analysis is solved.

## A5. Sen, Deb, Dhar & Koshiba — CricShotClassify — Sensors 21(8):2846 (2021)
- Introduces **CricShot10**: 10 cricket shot classes, uneven clip lengths, unpredictable illumination.
- CNN + **GRU** for long temporal dependency; transfer learning comparison across VGG16, InceptionV3,
  Xception, DenseNet169 with all layers frozen → **VGG16-GRU best at 86%**; fine-tuning the final 4 or
  final 8 VGG16 layers → **93%**.
- Kang (A3) reproduces this at **57.7%**.

## A6. Rajarajeswari, Prashant, Srinivasan & Dubey — Deep learning based cricket batting shot classification and performance analysis — Scientific Reports (2026)
- **3D CNN + optical flow + YOLOPose**; performance analysis by **cosine similarity** against a curated
  set of professional players' shots.
- **Data: the Kaggle "Cricket Shots IPL 2023" dataset — ~1,050 videos, ~150 per class across 7 classes.**
  (That is one of CricLens's six sources; we retain 1,922 clips from it.)
- **Reported accuracy 91.37% / 92%** (and they quote a prior 99.77% figure from the Sensors 2023 work).
- No leakage control, no grouped split, small dataset. The closest published attempt at "analytics
  generation" for cricket — and a direct comparison point for CricLens, on shared source data.

## A7. Cricket HPE + ML — Sensors 23(15):6839 (2023)
MediaPipe features + RF / SVM / KNN / DT / LR / LSTM; reports **99.77% with Random Forest**, k-fold 95.0%
(σ 0.07). Cite as the clearest example of the credibility problem Kang (A3) documents: this accuracy on
broadcast cricket video is not plausible without clip-level leakage or a trivially separable split.

## A8. Gupta & Balan — Cricket stroke extraction — arXiv:1901.03107; and **Fine-Grain Annotation of Cricket Videos** — arXiv:1511.07607
Early cricket stroke localisation and fine-grained annotation. Gupta & Balan's later TAL-inspired pipeline
(C3D + GRU, constrained and unconstrained variants) reports **weighted mTIoU 0.9376 (highlights) and 0.715
(generic)** for single-category temporal localisation, as reported in CricTAL's Table 2.

---

# Part IV — Corrections to the earlier gap analysis

Reading in full **overturned one of my earlier claims** and sharpened two others. The corrected position:

### ✗ CORRECTION — Gap 1 as originally stated was wrong
I previously wrote that *"no AQA paper measures how subject-selection error propagates downstream."*
**That is false.** PoseC3D (D3) measures exactly this on FineGYM: Detection **75.8** → Tracking **85.3** →
GT boxes **92.0** Mean-Top1 — a **16.2-point swing** attributable purely to which person-box the pose model
is given. FineParser (B6) also performs spatial parsing of the performer, from mask supervision.

**The claim that survives, and it is stronger as motivation than as novelty:**
- The field's own best evidence says subject selection is worth **~16 points** of downstream accuracy.
- PoseC3D's remedy is **ground-truth boxes** — unavailable at scale in broadcast footage. FineParser's is
  **mask annotation** — likewise.
- Every cricket paper resolves it with a heuristic: I3D-AE-LSTM takes *"the person with the largest
  y-coordinate"*; others use pre-cropped clips.
- **Nobody has built and evaluated a *learned* subject selector that closes the Detection→GT gap
  automatically, and nobody has measured that gap in a *quality assessment* setting** (PoseC3D measured it
  for action *recognition*).
- CricLens's finder: **91.2% top-1 on unseen matches vs 57% for hand-written rules.** The paper's
  contribution is the automatic recovery of the gap PoseC3D quantified, plus its first measurement on AQA.

### ✓ CONFIRMED and narrowed — Gap 2 (task-conditioned restoration)
PoseBench's "robustness enhancement" is **training-time only** (backbone, pre-training, resolution,
post-processing, data augmentation). It never selects a test-time restoration filter, and never propagates
the choice downstream. LIME-Eval argues for task-based evaluation but stays inside low-light enhancement
and uses detection as a proxy for **human preference**. **Neither asks which restoration to apply per
downstream consumer.** CricLens's D2/D3 result — CLAHE helps bat detection (11%→19%) and hurts pose; zero
padding cuts edge-case pose error 20–60%; best-SSIM (NL-means) is worst-for-pose — is unclaimed.
*Bonus:* our protocol already complies with LIME-Eval's own methodological warning, since we use fixed
pre-trained models and never retrain on enhanced images.

### ✓ CONFIRMED and strengthened — Gap 3 (leakage)
I3D-AE-LSTM: random 75/15/10, *"through trial and error"*, no grouping — on a dataset our build found to
carry 1,156 strokes duplicated across annotator folders. Kang: stratified random seed 27 on CricShot10,
where we found 157 duplicate clusters with 75 spanning its own official splits. Published precedent for
the fix and for the contribution type: F6 (geospatial, 89% duplicates / 93–97% leakage, pHash pipeline).
The VLM-judge paper (C6) even masks scoreboards *"to prevent data leakage"*.

### ✓ CONFIRMED — Gap 6 (per-part scores may not be discriminative)
Unchanged and still the sharpest available result: per-part SRC spread of **0.003** across five body parts.
Testable today from `manifest.parquet` at zero compute cost.

### ✓ CONFIRMED — Gap 7 (RGB stream earns little)
Their own ablation: pose-only 0.79 → +I3D 0.84. Reinforced by the survey's call (B8) for
**lightweight, efficient, real-time** AQA for resource-poor settings.

### ⚠ NEW RISK — our planned 3D lifting may not help
PoseC3D (D3): *"lifted 3D poses do not provide any additional information, performs even worse than the
original 2D poses in action recognition"* (FrameLift 90.0 / VideoLift 90.2 vs HRNet-2D 92.0 on FineGYM).
**Test before committing** to the 3D-lifting step in `PROGRESS.md` step 5.

### ⚠ NEW RISK — our five-part taxonomy is the thing ExpertAF criticises
ExpertAF (C4): *"All the prior work assumes a fixed taxonomy of errors, and the taxonomy is designed
separately for each activity."* CricLens's head/shoulders/hands/hips/feet scoring is exactly such a
taxonomy. **Defence available and worth stating explicitly:** it is not arbitrary — it is the coaching
guideline's own taxonomy (GCSE batting phases, formalised with two cricket experts), which makes it
domain-grounded rather than researcher-invented. Say so before a reviewer says it for us.

### ⚠ NEW CALIBRATION — perfect phase boundaries buy about 1 point
FineDiving/TSA (B4): with **ground-truth** step transitions, ρ rises 0.8925 → 0.9029 (w/o DN) and
0.9203 → 0.9310 (w/ DN). So even a perfect phase segmenter is worth roughly **+1 point of Spearman**.
CricTAL sits at 60–66% mAP@0.5 with every architecture. **Conclusion: heavy investment in better phase
localisation has a low ceiling.** Prefer the Temporal Parsing Transformer route (B5) — learnable queries
recovering atomic phases with no phase labels at all, which beat explicitly segmented TSA on MTL-AQA
(0.9451 vs 0.9422).
