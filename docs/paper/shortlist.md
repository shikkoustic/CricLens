# The 20 papers that actually matter — ranked

From the 53 read. **Tier 1** is what you should read yourself before writing a word; **Tier 2** is what you
should know well enough to cite precisely. The other 33 are background — cite them from `reading_log.md`.

Ranking is by *alignment with CricLens*, not by fame. Full notes for every entry are in `dossier.md`.

---

# TIER 1 — read these ten yourself

## 1. Moodley & van der Haar — I3D-AE-LSTM — WACV 2025
**Why #1: it is your paper's direct competitor, on your exact data.** They built CricketVision — the
dataset whose scores you train on — and published part-wise cricket AQA with **SRCC 0.84**.

ViTPose + I3D, two autoencoder streams, MLP predicting head / shoulders / hands / hips / feet.
8,540 clips of Test-match YouTube footage, VIA-annotated by two annotators against a guideline built with
two cricket experts. Trained on the **execution phase only — 4 frames per sample**.

**Three weaknesses you can build a paper on:** batter selection is one line (*"the person with the largest
y-coordinate"*); splits are random and ungrouped (*"through trial and error, 75, 15, 10"*) on a dataset
your own build found carries 1,156 duplicated strokes; and their five per-part correlations differ by
**0.003** total (0.83844–0.84146), which is what you see when five near-collinear labels are predicted as
one number. Their own ablation also shows pose-only gets **0.79** — the whole RGB stream is worth 0.05.

**Take away:** your baseline number, your three openings, and the scoring guideline's definitions.

## 2. Duan et al. — PoseC3D — CVPR 2022
**Why #2: it contains the single most useful experimental table in the entire literature for you.**

Bounding-box ablation on FineGym: **Detection 75.8 → Tracking 85.3 → GT boxes 92.0** Mean-Top1. A
16.2-point swing from nothing but *which person-box* the pose model gets. Their conclusion: *"the prior of
the interested person is extremely important."* This is the number that justifies your batter finder's
existence — and it also corrected a claim I had made (see `dossier.md` Part IV).

Also: dropping one limb keypoint per frame costs PoseConv3D **<1%** but costs a GCN **14.3%** — i.e. use
heatmap volumes, not ST-GCN, on noisy broadcast poses. And a warning: **lifted 3D poses performed *worse*
than the original 2D poses** (90.0/90.2 vs 92.0), which puts your planned 3D-lifting step at risk.

**Take away:** the motivation for Gap 1, your skeleton backbone choice, and a roadmap risk.

## 3. Kang — Modern DL Approaches for Cricket Shot Classification — arXiv:2510.09187 (2025)
**Why #3: it proves the cricket literature is not credible, and leaves the cause unexplained for you.**

Re-implements seven published models on one benchmark: 96% → **46.0%**, 99.2% → **55.6%**, 93% → **57.7%**,
98.9% → **10.6%**. His own EfficientNet-B0 + GRU gets **92.25%**.

He blames *"differences in dataset splits, evaluation code, or minor implementation details"* — and never
audits the data. His own 92.25% is on a **stratified random, ungrouped split of CricShot10**, the dataset
where your manifest build found **157 duplicate clusters, 75 spanning its own official train/val/test
folders**. He named the symptom; you can supply the diagnosis.

**Take away:** your permission to make leakage-audited re-baselining a contribution, plus the honest
numbers to compare against.

## 4. Moodley & van der Haar — CricTAL — ICCVW 2025
**Why #4: it is the other half of your competitor's pipeline, and it reveals a ceiling.**

Pose-only temporal localisation of buildup / execution / follow-through, using **OpenPose** (note: ViTPose
in their WACV paper — no cross-comparison anywhere). Best: TCN, **90.4% accuracy, mAP@0.5 64.45%**.

The revealing part: frame accuracy is 90–94%, but **localisation sits in a 60–66% mAP@0.5 band for every
architecture tried** — LSTM, RNN, TCN, Transformer. The model class barely matters, which says the
bottleneck is the representation or the label definition, not the network.

**Take away:** the state of phase segmentation in cricket, and evidence that throwing architectures at it
won't help.

## 5. Yin, Parmar et al. — A Decade of Action Quality Assessment — IJCV
**Why #5: it hands you your introduction.**

PRISMA review of 200+ papers. It names your gaps as the field's open problems: **"Scant Scale"**
(*"the biggest AQA dataset has more than 20000 samples"*), **"Narrow Action"**, **"Coarse Annotation"**
(*"a single score cannot allow the model to learn and utilize features sufficiently"*), blackbox
interpretability, and fragile robustness. It also says plainly: *"there is currently no sample that can be
used as a gold standard."*

**Take away:** you have **22,420 clips / 16,460 train-ready**. Against the survey's own headline, that is
frontier-scale, in a domain it lists as under-served. That sentence belongs in your abstract.

## 6. Xu et al. — FineDiving + TSA — CVPR 2022
**Why #6: it calibrates how much phase segmentation is actually worth.**

Introduces FineDiving (3,000 clips, 52 action / 29 sub-action types with step boundaries) and TSA
(segmentation → cross-attention → contrastive regression): **0.9203** with dive numbers.

The ablation that matters: with **ground-truth** step transitions, ρ moves 0.8925 → 0.9029 and
0.9203 → 0.9310. **A perfect phase segmenter is worth about one point of Spearman.** Read alongside #4
and the conclusion is clear — don't spend your project on better phase boundaries.

**Take away:** a calibrated expectation, and the procedure-aware recipe if you want it.

## 7. Yu et al. — CoRe + GART — ICCV 2021
**Why #7: this is the paradigm your scorer should use.**

Reformulates AQA as regressing the *relative* score against an exemplar with shared attributes, via a
group-aware regression tree. MTL-AQA **0.9512**; JIGSAWS jumps **0.70 → 0.85**. Introduces **R-ℓ2**, the
stricter companion metric to Spearman that you should report.

Why it fits cricket: CricketVision's high-scoring strokes are ready-made exemplars, matched by shot type
and handedness. And comparison-against-a-reference is literally how a coach reasons.

**Take away:** your model design, and a second evaluation metric.

## 8. Ma et al. — PoseBench — arXiv:2406.14367
**Why #8: it defines the boundary of your IVA novelty claim.**

60 models, 10 corruption types across blur/noise, compression/colour, lighting, occlusion. Findings:
**human pose models are particularly vulnerable to compression and blur** (exactly what your D1/D3 found),
and ViT backbones are most robust — **ViTPose-H 78.84 mAP clean, 65.02 corrupted**, justifying your
backbone choice independently.

**The scoping fact:** its "robustness enhancement" section covers **training-time factors only** —
backbone, pre-training, resolution, post-processing, augmentation. It never selects a test-time
restoration filter and never propagates the choice downstream. **Your Gap 2 survives this paper.**

**Take away:** the paper you must position against, and corroboration of two of your own findings.

## 9. Li, Zhao & Guo — LIME-Eval — arXiv:2410.08810
**Why #9: it is the closest published argument to your IVA contribution — and it stops short of it.**

Argues enhancement must be judged by downstream task performance, not reference image metrics. But it also
shows that the common protocol — *retraining* a detector on enhanced images — **overfits** and is
unreliable; so they propose evaluation with detectors pre-trained on standard-lighting data.

Two things follow. First, your protocol already complies: you use **fixed pre-trained** ViTPose and
detector and never retrain on enhanced frames. Say so. Second, they stay inside low-light enhancement and
use detection as a proxy for human preference — **nobody asks which restoration to apply per downstream
consumer**, which is exactly your CLAHE-helps-bat-detection-but-hurts-pose result.

**Take away:** the citation your N1 novelty is built on, and a protocol defence.

## 10. Yeh et al. — CoachMe — ACL 2025
**Why #10: it is the design for your coaching module, and it needs no new annotation.**

Reference-based: compare the learner's motion to a reference along temporal and physical axes, mimicking a
coach identifying deviations and focusing on the crucial body parts. Figure skating G-Eval **1.83** vs
GPT-4o **1.39**; reported as +31.6% on skating and +58.3% on boxing.

It composes perfectly with #7: the same exemplar serves both the relative score and the generated text.

**Take away:** how to build the coaching output without collecting a single new label.

---

# TIER 2 — know these well enough to cite precisely

## 11. Xu et al. — FineParser — CVPR 2024
Current SOTA for human-centric AQA: spatial action parser + temporal action parser + static visual encoder
+ contrastive regression. FineDiving-HM **ρ 0.9435 / R-ℓ2 0.2602**; temporal parsing **AIoU@0.5 0.9946** vs
TSA 0.9239. Introduces foreground action masks. *Read it to know where the ceiling is — and note it
segments a single obvious performer from mask supervision, never chooses among several people.*

## 12. Tang et al. — USDL / MUSDL — CVPR 2020
Treats a score as a **distribution**, not a point, because judges are subjective. I3D + Gaussian label +
KL loss; multi-path variant, one path per judge. MTL-AQA **0.9273**.
*Directly applicable to your label problem: CricketVision scores are single-annotation-team with no
inter-rater ceiling, so a point estimate overstates what the label can support.*

## 13. Bai et al. — Temporal Parsing Transformer — ECCV 2022
Learnable queries recover atomic temporal parts **with no part-level labels**, via ranking + sparsity
losses on cross-attention. MTL-AQA **0.9451** — beating explicitly-segmented TSA.
*The strongest argument that you don't need annotated phase boundaries at all. Pair with #4 and #6.*

## 14. Ashutosh et al. — ExpertAF — CVPR 2025
Video + 3D pose → expert commentary + retrieved expert demonstration + generated corrected pose, weakly
supervised from Ego-Exo4D commentary via an LLM. Up to **3×** better than baselines in human evaluation.
*Contains the line you must pre-empt: "All the prior work assumes a fixed taxonomy of errors, designed
separately for each activity." Your five body parts are such a taxonomy — defend it as the coaching
guideline's own, not researcher-invented.*

## 15. Monte e Freitas et al. — Can VLMs Judge Action Quality? — CVPRW 2026
Gemini 3, Qwen3-VL, InternVL3.5 perform **only marginally above chance** on AQA, with two systematic
biases: predicting correct execution regardless of evidence, and sensitivity to linguistic framing.
*This is why you ground the LLM in measured pose features instead of showing it the video. One citation
defends your whole architecture.*

## 16. Adimoolam et al. — Data Leakage Detection and De-duplication — arXiv:2304.02296
AICrowd Mapping Challenge: **89% of training images are duplicates, 93–97% of validation images appear in
training**. Contributes a perceptual-hashing dedupe and leakage pipeline.
*Your methodological precedent: same technique, same argument, different domain. Proves "audit then
re-baseline" is a publishable contribution type.*

## 17. Parmar & Morris — MTL-AQA — CVPR 2019
1,412 dives, 16 events, with difficulty, seven judges' scores, dive class **and natural-language
commentary**. C3D-AVG-MTL **0.9044**. Finds that action-recognition representations are *not* sufficient
for AQA and must be learned.
*The origin of the "score + language" framing your project extends.*

## 18. Doughty et al. — The Pros and Cons — CVPR 2019
Rank-aware temporal attention with separate branches for high-skill ("pros") and low-skill ("cons")
evidence. **80.3 / 81.2%** pairwise accuracy, +4.3 / +5.4 over their CVPR 2018 work.
*A ready-made interpretable-feedback mechanism: "which part of your stroke cost you marks" falls out of the
cons-attention map, with no phase labels.*

## 19. Zhou et al. — AQA Survey and Benchmark — Pattern Recognition 2026
PRISMA survey **plus a unified benchmark with public code** that re-runs representative AQA methods under
standardised protocols, reporting accuracy *and* computational efficiency.
*Practical value: the fastest honest route to reproducing USDL / CoRe / TSA on your data instead of
reimplementing each from scratch. Start your baseline work here.*

## 20. Xu et al. — ViTPose — NeurIPS 2022
Plain ViT backbone + lightweight decoder, scaling 100M → 1B parameters, **80.9 AP on COCO test-dev**.
*Your pose backbone. Cite together with #8, which independently finds ViT backbones most robust under
corruption — so your choice is defensible on both accuracy and robustness.*

---

# How to use this

- **Read Tier 1 in order.** 1–4 are the competitive landscape; 5–7 are the method; 8–10 are your two
  novelty claims.
- **Tier 2 you can read at the "setup + results + conclusion" level.** That is enough to cite precisely.
- **Everything else** — ST-GCN, ByteTrack, MS-TCN, FineGym, SoccerNet, TrackNet, TVCalib, the rehab and
  robustness papers, the remaining cricket work — goes in Related Work as context, citable straight from
  `reading_log.md` without further reading.

That is 20 papers read properly plus ~15 cited as context, which comfortably clears your sir's 20–30 target
while every citation is one you can actually defend in a viva.
