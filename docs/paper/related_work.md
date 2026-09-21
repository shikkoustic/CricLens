# CricLens: literature survey for the journal paper

Purpose: the reading list and gap analysis behind the CricLens journal submission (IVA / CSET344 + IMD).
Compiled 2026-09-21. **Verification status:** entries were gathered through search and abstract-level
sources; arXiv and CVF Open Access were unreachable from the build environment, so every citation must be
checked against the publisher PDF (authors, page numbers, exact metric values) before submission. Numbers
quoted below are as reported by the authors, not reproduced by us.

---

## 0. How the problem is framed

The paper's problem is **not** "classify a cricket shot". Framed for a journal audience it is:

> Automated, fine-grained performance analytics from *uncontrolled, in-the-wild* video: locating the
> subject of interest, segmenting the action into semantically meaningful phases, and producing
> per-body-part quality scores and interpretable feedback — under the image degradations, camera cuts,
> viewpoint changes and annotation scarcity that real broadcast footage imposes.

That places CricLens at the intersection of five literatures: (A) cricket video analysis, (B) action
quality assessment, (C) skill assessment and feedback generation, (D) pose estimation / skeleton action
recognition, (E) large-scale sports video understanding, and (F) robustness, preprocessing and dataset
hygiene. Clusters A–C are the "same problem"; D–E are the machinery; F is where the IVA half of the
project makes its contribution.

---

## A. Cricket-specific video analysis — the direct competitors

**A1. Moodley & van der Haar (2025). "I3D-AE-LSTM: A 2-Stream Autoencoder for Action Quality Assessment
Using a Newly Created Cricket Batsman Video Dataset." WACV 2025, pp. 5470–5478.**
The single closest paper to CricLens. Introduces the UJ-AQA-CricketVision dataset (8,540 stroke clips,
three annotated phases: buildup / execution / follow-through) and a 2-stream autoencoder over I3D RGB
features and pose keypoints, feeding an MLP regressor that predicts quality for **head, shoulders, hands,
hips and feet**. Reports Spearman rank correlation **0.84**. *This is the baseline we must reproduce and
beat* — it is also the source of the score labels CricLens trains on.

**A2. Moodley et al. (2025). "CricTAL: Introducing Temporal Activity Localisation using pose estimation to
identify cricket stroke phases." ICCV 2025 Workshops (SAUAFG).**
Pose-**only** temporal activity localisation for cricket, producing aligned, interpretable phase
boundaries to feed phase-wise AQA. Best TCN model: test accuracy **90.4%**, mAP@0.5 **64.45%**. Directly
overlaps CricLens's analysis-window logic (contact −0.8 s to +0.6 s) but solves it with a learned model.

**A3. Moodley et al. (2025). "I3D-AE-LSTM: Combining action representations using a 2-stream autoencoder
for Action Quality Assessment." Expert Systems with Applications (journal extension of A1).**
Worth reading as a *template*: it is exactly the conference-to-journal extension route your sir is asking
you to take.

**A4. Moodley & van der Haar (2022). "Automated recognition of the cricket batting backlift technique in
video footage using deep learning architectures." Scientific Reports 12.**
Lateral (LBBT) vs straight (SBBT) backlift classification with AlexNet / Inception V3 / Inception-ResNet-V2
/ Xception; Xception best at ~98.2% accuracy, precision 100%, recall 95%. Establishes that
*cricket-specific technique constructs* are learnable from video, but on a tiny curated two-class set.

**A5. CricShot10k (2026). "CricShot10k: A Large-Scale Video Dataset for Cricket Shot Classification."
IEEE Access.**
10,086 clips / 15 classes, built by exploiting recurring broadcast patterns to auto-split long videos.
First to include women's and U-19 matches and footage back to 1996. Their CricShotNet (cropping +
segmentation layers) with EfficientNetV2-S + GRU(128) reaches **89%**. One of CricLens's six sources —
and, per our own D1 quality profile, the most degraded (Laplacian sharpness 86 vs 341–713 elsewhere).

**A6. Sen et al. (2021). "CricShotClassify: An Approach to Classifying Batting Shots from Cricket Videos
Using a Convolutional Neural Network and Gated Recurrent Unit." Sensors 21(8):2846.**
The CNN+GRU baseline that most later cricket work compares against; source of the CricShot10 dataset.

**A7. Shot-Net (2019).** Early CNN for cricket shot classification — historical baseline.

**A8. Shot-ViT (2024). "Cricket Batting Shots Classification with Vision Transformer Network."**
ViT applied to the same task; shows the field's drift from CNN+RNN to transformers.

**A9. Kang, S. (2025). "Modern Deep Learning Approaches for Cricket Shot Classification: A Comprehensive
Baseline Study." arXiv:2510.09187.**
**The most useful paper for our framing.** Re-implements seven approaches across four paradigms on a
unified benchmark and finds a massive credibility gap: published accuracies of 96%, 99.2% and 93%
reproduce at **46.0%, 55.6% and 57.7%**. Best honest result: EfficientNet-B0 + GRU at 92.25%. This paper
legitimises a "leakage-audited re-baselining" contribution — which CricLens already has evidence for.

**A10. (2026). "Deep learning based cricket batting shot classification and performance analysis using
computer vision." Scientific Reports.**
3D CNN + optical flow + YOLOPose, with comparison against professional players' shots. The nearest thing
in the literature to "analytics generation" for cricket, but on a small dataset with no leakage control.

**A11. (2025). "Cricket Shot Classification and Pose Correction using Detectron2 and XGBoost Classifier."
Procedia Computer Science.** Pose → classical classifier → "pose correction" feedback. Closest existing
attempt at the coaching-feedback output CricLens targets; shallow evaluation.

**A12. (2023). "Enhancing Cricket Performance Analysis with Human Pose Estimation and Machine Learning."
Sensors 23(15):6839.**
MediaPipe features + RF/SVM/KNN/DT/LR/LSTM; reports **99.77%** with Random Forest. Cite this as a
*cautionary* example: an accuracy that high on broadcast cricket video is a strong indicator of
clip-level leakage or a trivially separable split — exactly what A9 documents across the field.

**A13. (2022). "A survey on event detection based video summarization for cricket." Multimedia Tools and
Applications.** Coverage of the cricket-video pipeline upstream of shot analysis.

**A14. Gupta & Karel (2015). "Fine-Grain Annotation of Cricket Videos." arXiv:1511.07607.**
Early fine-grained cricket annotation; useful for the "annotation is the bottleneck" argument.

---

## B. Action Quality Assessment — the methodological core

**B1. Parmar & Morris (2019). "What and How Well You Performed? A Multitask Learning Approach to Action
Quality Assessment." CVPR 2019.** MTL-AQA, 1,412 diving samples from 16 competitions; jointly learns
fine-grained action recognition, **commentary generation** and score estimation; C3D-AVG-MTL reaches
rank correlation 90.44%. The canonical "score + language" formulation — the precedent for CricLens's
coaching-text output.

**B2. Tang et al. (2020). "Uncertainty-aware Score Distribution Learning for Action Quality Assessment."
CVPR 2020.** USDL/MUSDL: predict a *distribution* over scores instead of a point estimate, because judge
labels are intrinsically ambiguous. Directly relevant — CricketVision's 1–10 part scores are subjective
single-annotator labels, and our datasheet shows the apparent "duplicate annotations" are copies, not
independent ratings, so we have **no inter-rater ceiling**. USDL is the principled response.

**B3. Yu et al. (2021). "Group-aware Contrastive Regression for Action Quality Assessment." ICCV 2021.**
CoRe: regress the *relative* score difference between a query and exemplar videos rather than an absolute
score. The dominant paradigm in modern AQA.

**B4. Xu et al. (2022). "FineDiving: A Fine-grained Dataset for Procedure-aware Action Quality
Assessment." CVPR 2022.** Introduces FineDiving and the TSA module: procedure segmentation →
procedure-aware cross-attention → fine-grained contrastive regression. The "phases matter" argument, which
maps one-to-one onto cricket's buildup/execution/follow-through.

**B5. Bai et al. (2022). "Action Quality Assessment with Temporal Parsing Transformer." ECCV 2022.**
Decomposes holistic features into temporally ordered part representations via learnable queries — an
alternative to explicit phase labels.

**B6. Xu et al. (2024). "FineParser: A Fine-grained Spatio-temporal Action Parser for Human-centric Action
Quality Assessment." CVPR 2024.** Four modules: spatial action parser, temporal action parser, static
visual encoder, fine-grained contrastive regression. State of the art for *human-centric* AQA and the
architecture most worth borrowing from.

**B7. (2024). "Multi-Stage Contrastive Regression for Action Quality Assessment." arXiv:2401.02841.**

**B8. (2025). "Action Quality Assessment via Hierarchical Pose-guided Multi-stage Contrastive
Regression." arXiv:2501.03674.** Pose-guided AQA — the closest methodological neighbour to a pose-first
CricLens scorer.

**B9. Yin, H. et al. (2025/26). "A Decade of Action Quality Assessment: Largest Systematic Survey of
Trends, Challenges, and Future Directions." International Journal of Computer Vision 134(2).**
PRISMA review of 214 papers across 33 datasets and 7 research trends. **Explicitly names our gaps as the
field's open problems:** dataset scale and subject/action diversity, annotation granularity,
interpretability, real-time inference, unified evaluation metrics, and multimodal robustness. Cite this
in the introduction to justify the contribution.

**B10. (2024). "A Comprehensive Survey of Action Quality Assessment: Method and Benchmark."
arXiv:2412.11149.** Taxonomy by modality, learning paradigm and evaluation granularity.

**B11. (2022). "Skeleton-based deep pose feature learning for action quality assessment on figure skating
videos." Journal of Visual Communication and Image Representation.** Pose-only AQA precedent.

**B12. (2020). "A Comprehensive Review of Skeleton-based Movement Assessment Methods." arXiv:2007.10737.**

---

## C. Skill assessment and feedback generation — where the project is heading

**C1. Doughty, Damen & Mayol-Cuevas (2018). "Who's Better? Who's Best? Pairwise Deep Ranking for Skill
Determination." CVPR 2018.** Formulates skill as pairwise ranking rather than absolute scoring;
70–83% correctly ordered pairs across four task datasets. The alternative supervision signal when
absolute scores are unreliable — relevant given B2's concern about our labels.

**C2. Doughty et al. (2019). "The Pros and Cons: Rank-aware Temporal Attention for Skill Determination in
Long Videos." CVPR 2019.** Rank-specific attention modules attend separately to high-skill ("pros") and
low-skill ("cons") video segments; +4% pairwise accuracy overall, up to +12% per task. **A natural source
of interpretable per-phase feedback** — highly transferable to "which part of your stroke cost you marks".

**C3. Grauman et al. (2024). "Ego-Exo4D: Understanding Skilled Human Activity from First- and Third-Person
Perspectives." CVPR 2024.** 1,286 hours, ego + synchronised exo views, with a **proficiency estimation**
benchmark (Novice / Early Expert / Intermediate / Late Expert) and expert commentary. The reference point
for modern skill assessment at scale.

**C4. Ashutosh et al. (2025). "ExpertAF: Expert Actionable Feedback from Video." CVPR 2025.**
Takes video + 3D pose and generates (i) free-form expert commentary on what is good and what to improve,
and (ii) a *visual* expert demonstration with corrections applied. Weakly-supervised training data built
from Ego-Exo4D commentary + an LLM. **The state of the art for the exact output CricLens promises**, and
the paper to position our coaching module against.

**C5. Yeh et al. (2025). "CoachMe: Decoding Sport Elements with a Reference-Based Coaching Instruction
Generation Model." ACL 2025.** Compares a learner's motion against a *reference* motion along temporal and
physical axes, mimicking a coach's reasoning; beats GPT-4o by 31.6% (G-Eval, figure skating) and 58.3%
(boxing). Reference-based comparison is directly implementable with CricketVision's high-scoring strokes
as references.

**C6. Monte e Freitas et al. (2026). "Can Vision Language Models Judge Action Quality? An Empirical
Evaluation." CVPR 2026 Workshops (SAUAFG).** Frontier VLMs perform only marginally above chance on AQA;
adding skeletons, grounding instructions or in-context learning gives isolated but inconsistent gains; two
systematic biases found (predicting "correct execution" regardless of evidence, and sensitivity to
linguistic framing). **Strong justification for CricLens's design: ground the LLM in measured pose
features rather than asking a VLM to judge the video.**

**C7. (2025). "SkillFormer: Unified Multi-View Video Understanding for Proficiency Estimation."
arXiv:2505.08665** and **"ProfVLM: A Lightweight Video-Language Model for Multi-View Proficiency
Estimation." arXiv:2509.26278.** Recent proficiency-estimation architectures on Ego-Exo4D.

**C8. Liao, Vakanski & Xian (2020). "A Deep Learning Framework for Assessing Physical Rehabilitation
Exercises." IEEE TNSRE.** Plus the **skeleton rehabilitation benchmark (arXiv:2507.21018)** on KIMORE and
UI-PRMD. The clinical sibling of sports AQA; cite for the "per-joint quality score" formulation and for
the argument that pose-only assessment generalises beyond sport.

---

## D. Pose estimation, tracking and skeleton action recognition — the machinery

**D1. Xu et al. (2022). "ViTPose: Simple Vision Transformer Baselines for Human Pose Estimation."
NeurIPS 2022** (and ViTPose++, TPAMI 2023). 80.9 AP on COCO test-dev. The pose backbone CricLens uses;
cite for the architecture choice.

**D2. Yan, Xiong & Lin (2018). "Spatial Temporal Graph Convolutional Networks for Skeleton-Based Action
Recognition." AAAI 2018.** ST-GCN — the reference skeleton model and the obvious comparison point for
our shot classifier.

**D3. Duan et al. (2022). "Revisiting Skeleton-based Action Recognition" (PoseC3D). CVPR 2022.**
3D heatmap volumes instead of graphs; **more robust to pose-estimation noise and better at cross-dataset
generalisation** than GCNs. Given that CricLens's joints come from noisy broadcast footage across six
sources, this robustness claim is directly testable and worth an ablation.

**D4. Pavllo et al. (2019). "3D human pose estimation in video with temporal convolutions and
semi-supervised training." CVPR 2019.** VideoPose3D — the standard 2D→3D lifting route, and the
mechanism behind CricLens's planned camera-angle normalisation.

**D5. Zhang et al. (2022). "ByteTrack: Multi-Object Tracking by Associating Every Detection Box."
ECCV 2022.** The tracker CricLens uses; its low-confidence second association stage is what keeps the
batter's track alive through occlusion by the bowler and keeper.

**D6. Abu Farha & Gall (2019). "MS-TCN: Multi-Stage Temporal Convolutional Network for Action
Segmentation." CVPR 2019** (and MS-TCN++, 2020). The standard phase-segmentation architecture — the
model class CricTAL (A2) applies to cricket, and our alternative to heuristic window selection.

**D7. Shao et al. (2020). "FineGym: A Hierarchical Video Dataset for Fine-grained Action Understanding."
CVPR 2020.** 29k videos, 99 classes, three-level event/set/element hierarchy with two levels of temporal
annotation. The model for how a fine-grained sports taxonomy should be structured — relevant to our
8-shot + off/leg/straight side taxonomy.

**D8. VideoMAE / Video Swin / TimeSformer / SlowFast / I3D.** RGB backbone family; VideoMAE-H tops
Kinetics-400 at 86.6% top-1 but at 633M parameters, vs I3D at 12.2M. Cite when arguing the
compute-vs-accuracy case for a pose-only pipeline.

---

## E. Large-scale sports video understanding

**E1. Giancola et al. (2018). "SoccerNet: A Scalable Dataset for Action Spotting in Soccer Videos."
CVPRW 2018** and **E2. Deliège et al. (2021). "SoccerNet-v2." CVPRW 2021** (~300k annotations, 500
untrimmed broadcasts, plus camera-shot segmentation and replay grounding). The gold standard for how a
broadcast-sports benchmark should be constructed and released — the model to emulate if we publish the
CricLens manifests.

**E3. Cioppa et al. (2022). "SoccerNet-Tracking." arXiv:2204.06918.** Player tracking in broadcast.

**E4. Huang et al. (2019). "TrackNet: A Deep Learning Network for Tracking High-speed and Tiny Objects in
Sports Applications."** Heatmap-based multi-frame ball tracking; 98.2% F1 tennis, 68.7% badminton.
Plus **TrackNetV4 (arXiv:2409.14543)** with motion attention maps. The reference for CricLens's planned
ball-tracking module.

**E5. Zhu et al. (2023). "A Survey of Deep Learning in Sports Applications: Perception, Comprehension,
and Decision." arXiv:2307.03353.** The three-layer framing (perception → comprehension → decision) is a
clean structure for our introduction — CricLens spans all three.

**E6. Naik et al. (2022). "A Comprehensive Review of Computer Vision in Sports: Open Issues, Future
Trends and Research Directions." Applied Sciences 12(9):4429.**

**E7. Wu et al. (2022). "A Survey on Video Action Recognition in Sports: Datasets, Methods and
Applications." arXiv:2206.01038.**

---

## F. Robustness, preprocessing and dataset hygiene — the IVA contribution

**F1. Wang et al. (2024). "PoseBench: Benchmarking the Robustness of Pose Estimation Models under
Corruptions." arXiv:2406.14367.** 60 models, 3 datasets, 10 corruption types in four categories
(blur/noise, compression/colour loss, severe lighting, masks). **The methodological precedent for our
D3 degradation study** — and the paper to position against, because PoseBench measures pose accuracy
under corruption but does not ask which *restoration* to apply, nor propagate the effect to a downstream
analytics task.

**F2. Wang et al. (2021). "When Human Pose Estimation Meets Robustness: Adversarial Algorithms and
Benchmarks." CVPR 2021.** COCO-C / MPII-C / OCHuman-C: 15 corruption types × 5 severities.

**F3. Hoang et al. (2024). "Improving the Robustness of 3D Human Pose Estimation: A Benchmark Dataset and
Learning from Noisy Input." CVPRW 2024.** Human3.6M-C / HumanEva-I-C — occlusion, motion blur, pixel noise
for video-based 3D lifters.

**F4. Schiappa et al. (2022). "Large-scale Robustness Analysis of Video Action Recognition Models."
arXiv:2207.01398.** The same question at the action-recognition level.

**F5. (2024). "LIME-Eval: Rethinking Low-light Image Enhancement Evaluation via Object Detection."
arXiv:2410.08810.** Argues enhancement should be judged by downstream detection performance, not by
reference image-quality metrics. **The single most important citation for CricLens's D2/D3 finding** that
NL-means achieves the best SSIM yet the *worst* pose error, and that global HE improves apparent contrast
while degrading wrist confidence by 0.036. We extend this argument from detection to pose-driven AQA.

**F6. (2023). "Data Leakage Detection and De-duplication in Large Scale Geospatial Image Datasets."
arXiv:2304.02296**, and **F7. "Near-Duplicate Leakage in Fine-Art Classification"**, and
**F8. "Leakage-Safe Re-Evaluation of Deep Video Models for Violence Detection."**
Three independent demonstrations that near-duplicate leakage silently inflates published results (one
reports 287 byte-identical test clips producing a meaningless F1 of 1.000; another isolates a 3.60 pp
accuracy inflation). These justify CricLens's two-stage pHash dedupe and match-grouped splits as a
*methodological contribution*, not just housekeeping — and they pair with A9 and A12 to make the case
that cricket shot classification specifically has this problem.

**F9. Theiner & Ewerth (2023). "TVCalib: Camera Calibration for Sports Field Registration in Soccer."
WACV 2023**, and **F10. "PnLCalib: Sports field registration via points and lines optimization." CVIU.**
Field registration by detecting markings (segmentation or Hough lines) and optimising camera pose. The
reference for the planned IVA pitch-calibration module (HSV pitch segmentation → morphology → connected
components → Canny + Hough crease lines → pixels-to-centimetres).

**F11. Sony AI / FHIBE and "Identifying Ethical Biases in Action Recognition Models" (arXiv:2604.17971).**
Fairness evaluation for pose estimation and action recognition. Sparse on *handedness* specifically —
which is precisely the gap CricLens's left/right audit can fill (we hold 4,400 right- and 2,492
left-handed labelled strokes).

---

## G. What the existing pipeline looks like (the thing to reproduce first)

Synthesising A–C, the consensus pipeline for video-based sports technique analysis today is:

1. **Input**: a pre-trimmed, pre-cropped clip containing exactly one subject. *Subject selection is
   assumed away* — MTL-AQA, FineDiving and Ego-Exo4D all have one obvious performer; cricket papers use
   either pre-cropped datasets or hand-drawn boxes (A1).
2. **Feature extraction**: I3D / C3D RGB features, optionally a second pose stream (A1, B1, B6).
3. **Temporal structure**: either heuristic uniform sampling, or a learned phase segmenter — TSA (B4),
   temporal parsing queries (B5), MS-TCN/TCN (D6, A2).
4. **Scoring head**: direct regression (B1), score-distribution learning (B2), or contrastive regression
   against exemplars (B3, B6, B8).
5. **Evaluation**: Spearman rank correlation against judge scores, on a random split.
6. **Feedback (recent, still rare)**: LLM/VLM commentary from video + pose (C4, C5) — with C6 showing
   ungrounded VLMs are near chance.

**Minimum reproduction set for the "implement the existing pipeline" step:**
- A1 (I3D-AE-LSTM) on CricketVision — the direct baseline, same dataset we already hold, SRCC 0.84 target.
- A5/A9 (EfficientNet-B0/V2-S + GRU) for shot classification — A9's honest ~92% is the number to match.
- B3 or B6 as the modern AQA baseline on our part-wise scores.
- D2/D3 (ST-GCN and PoseC3D) as the skeleton baselines for shot classification.

---

## H. The gaps, and candidate novelty

Five gaps are visible across the literature. CricLens is unusually well-placed on four of them because of
work already done.

**Gap 1 — Subject selection is an unsolved, unmeasured stage.**
No AQA paper identifies *which* person to assess; every cricket paper uses pre-cropped clips or
hand-drawn boxes (A1, A5, A6, A12). Nobody has measured how subject-selection error propagates into shot
classification accuracy or technique scores.
*CricLens has:* a learned batter finder at 91.2% top-1 on unseen matches vs 57% for hand-written rules,
validated by a 400-clip audit (92.5% striker-correct) and by "twin clips" appearing in two sources.

**Gap 2 — Preprocessing is chosen by image-quality metrics, not by task effect.**
F1–F4 measure robustness *to* corruption; F5 argues for task-based evaluation but only for detection. No
AQA work conditions its preprocessing on measured downstream effect, and several cricket papers apply
enhancement with no ablation at all.
*CricLens has:* D1–D3 — a quality profile of all 22,420 clips; a measured result that *no* enhancement
helps pose while CLAHE raises bat detection 11%→19% (~89% of extra detections verified real by eye); and
a padding study where zero padding cuts edge-case pose error 20–60% over clamping, affecting 4,461 clips.
Plus the headline negative result: **NL-means has the best SSIM and the worst pose error** — image quality
metrics do not predict task accuracy.

**Gap 3 — Cricket results are not credible, and the field knows it.**
A9 documents 96%→46% reproduction gaps; A12 reports 99.77%. Our own datasheet independently found the
CricShot10 mirror's published split leaks 75 duplicate clusters spanning train/val/test.
*CricLens has:* 22,420 clips from six sources, 1,612 verified duplicates removed by two-stage pHash, and
match-grouped stratified splits — the first leakage-controlled, multi-source cricket benchmark.

**Gap 4 — Cricket technique assessment has never been tested across sources.**
A1's SRCC 0.84 is single-source, single-annotation-team. Cross-broadcast generalisation is untested.
*CricLens has:* the only corpus where a model can be trained on CricketVision scores and evaluated on five
other broadcast sources.

**Gap 5 — Handedness bias is unexamined in sports AQA.**
F11 covers skin tone, gender and age; handedness is absent, despite left/right asymmetry being structural
in cricket.
*CricLens has:* 4,400 right- and 2,492 left-handed labelled strokes, and a handedness-independent
off/leg/straight side axis in the taxonomy.

### Candidate novelty statements, ranked

**N1 (recommended core — serves IVA and IMD together): Task-conditioned preprocessing for pose-driven
analytics.** Claim: restoration should be selected per *downstream consumer*, not per image-quality score,
and the correct choice differs between consumers in the same pipeline. Evidence already in hand: CLAHE
helps bat detection and hurts pose; zero padding beats clamping; best-SSIM is worst-for-pose. Deliverable:
a per-clip degradation profiler that routes each clip to a measured policy, ablated end-to-end against a
fixed-preprocessing baseline. Extends F5 from detection to fine-grained quality assessment.

**N2 (recommended core): Subject selection as a first-class, measured pipeline stage.** Claim: the
"assume one performer" convention hides a real error source in in-the-wild sport. Deliverable: the learned
batter finder plus a propagation study — annotated box vs learned finder vs hand-written rules vs
whole-frame pose — measured at the end task (shot accuracy and part-wise SRCC), not just at top-1
selection accuracy. No prior AQA work reports this.

**N3 (evaluation framing): A leakage-audited, multi-source cricket benchmark** with published manifests,
dedupe clusters and match-grouped splits (SoccerNet-style, E1/E2), plus honest re-baselining of prior
methods on it — following A9's lead and F6–F8's argument.

**N4 (model): Pose-only, phase-aware, part-wise scoring.** Combine CricTAL-style learned phase
segmentation (A2, D6) with part-wise contrastive regression (B3/B6/B8) over skeletons alone, and test
whether it matches A1's 2-stream RGB+pose SRCC 0.84 at a fraction of the compute — with PoseC3D (D3)
versus ST-GCN (D2) as the robustness ablation. Pose-only is also the privacy- and bandwidth-friendly
option for a deployed web app.

**N5 (audit section): Handedness bias audit and mirror canonicalisation.** Quantify the left/right
performance gap and test whether canonicalising all strokes to one handedness closes it. Fills Gap 5.

**Suggested paper shape:** N3 as the benchmark, N1 + N2 as the twin methodological contributions
(one IVA-flavoured, one IMD-flavoured), N4 as the model that consumes both, N5 as an audit section.
That is a coherent journal-length contribution rather than four thin ones.
