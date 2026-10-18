# Which of the 53 papers actually map to CricLens's problem

Re-ranked after the shift in framing: this is a **research project on cricket batting analysis**, not a
demonstration of course syllabi. Papers are grouped by *how close their problem is to ours*, not by how
famous they are.

**30 of the 53 map onto the problem. 23 are machinery or background.**

The distinction that matters most, and the one that separates the cricket literature into two halves:

- **CLASSIFY** — "which shot was this?" A label. Nearly all cricket papers stop here.
- **RATE** — "how well was it played?" A score, ideally per body part. Almost nobody does this in cricket.

CricLens does both. That combination is rare, and it is the reason the project has room to contribute.

---

## Group A — Same sport, same task: cricket technique quality (3 papers)
**This is the entire world of cricket technique rating. It is one research group.**

| Paper | Venue | What it does | Result |
|---|---|---|---|
| **I3D-AE-LSTM** — Moodley & van der Haar | WACV 2025 | RATE. Five body-part scores from ViTPose + I3D two-stream autoencoder | SRCC **0.84** (pose-only 0.79) |
| **CricTAL** — Moodley & van der Haar | ICCVW 2025 | Phase localisation (buildup/execution/follow-through) to feed rating | TCN **90.4%** acc, mAP@0.5 **64.45%** |
| **Backlift recognition** — Moodley, van der Haar & Noorbhai | Sci. Reports 2022 | Technique *type* recognition (lateral vs straight backlift) | Xception **98.2%** — but on only **200 images** |

Plus a fourth we could not retrieve: the **I3D-AE-LSTM journal extension** in *Expert Systems with
Applications* (Elsevier paywall).

**What this tells us:** cricket technique rating is a field of one lab, three papers, one dataset. That is
both an opportunity (little competition) and a warning (few reviewers will take "nobody has done this" on
trust — we must show *why* it is hard, not just that it is new).

## Group B — Same sport, adjacent task: cricket shot classification (6 papers)
**These CLASSIFY. None of them RATE, with one partial exception.**

| Paper | Venue | Data | Result |
|---|---|---|---|
| **Kang — baseline study** | arXiv 2025 | CricShot10, 1,888 clips / 10 classes, random split | Reproductions **46.0 / 55.6 / 57.7%** vs claimed 96 / 99.2 / 93%; own **92.25%** |
| **CricShotClassify** — Sen et al. | Sensors 2021 | CricShot10 (introduced), 10 classes | VGG16-GRU **93%** fine-tuned, 86% frozen |
| **Rajarajeswari et al.** | Sci. Reports 2026 | Kaggle IPL-2023, ~1,050 clips / 7 classes | **91.37 / 92%** — *and* cosine-similarity comparison against professional players |
| **Cricket HPE + ML** | Sensors 2023 | small | MediaPipe + RF **99.77%** (implausible — cite as cautionary) |
| **Cricket stroke extraction** — Gupta & Balan | arXiv 2019 | cricket | stroke localisation, mTIoU 0.9376 / 0.715 |
| **Fine-grain annotation of cricket videos** | arXiv 2015 | cricket | early fine-grained annotation |

**The one to look at closely is Rajarajeswari et al. (Sci. Reports 2026)** — it classifies *and* compares
the player against professionals by cosine similarity. That is the closest anyone outside Moodley's group
has come to rating, and it uses the IPL-2023 Kaggle set, which is one of our six sources. It is the natural
comparison point for a "classification + rating in one system" claim.

## Group C — Same task, different sport: action quality assessment (8 papers)
**This is where the methodology actually lives.** Cricket has three papers; AQA as a field has hundreds.
Our method will be built from this group and tested on cricket.

| Paper | Venue | Contribution | Best result |
|---|---|---|---|
| **MTL-AQA** — Parmar & Morris | CVPR 2019 | Multitask: recognition + **commentary** + score. 1,412 dives | 0.9044 |
| **USDL / MUSDL** — Tang et al. | CVPR 2020 | Scores as **distributions**, not points — handles judge subjectivity | 0.9273 |
| **CoRe + GART** — Yu et al. | ICCV 2021 | **Contrastive regression** against an exemplar; introduces R-ℓ2 metric | 0.9512 |
| **FineDiving + TSA** — Xu et al. | CVPR 2022 | Procedure-aware: segment into steps, then compare step-wise | 0.9203 |
| **Temporal Parsing Transformer** — Bai et al. | ECCV 2022 | Learnable queries recover phases **with no phase labels** | 0.9451 |
| **FineParser** — Xu et al. | CVPR 2024 | Parses the performer in **space and time**; current SOTA | 0.9435 |
| **MCoRe** — An, Qi & Ma | arXiv 2024 | Multi-stage contrastive regression | 0.9232 |
| **Hierarchical pose-guided MCoRe** — Qi et al. | IEEE TIP | Pose-guided multi-stage contrastive regression | 0.9266–0.9310 |

## Group D — Same goal, broader framing: skill assessment and feedback (7 papers)
**Where the project is heading in phase 5 — turning a score into advice.**

| Paper | Venue | Contribution |
|---|---|---|
| **Who's Better? Who's Best?** — Doughty et al. | CVPR 2018 | Skill as **pairwise ranking**, not absolute score. 70–83% correct pairs |
| **The Pros and Cons** — Doughty et al. | CVPR 2019 | Attention branches for high- and low-skill evidence. 80.3 / 81.2% |
| **Ego-Exo4D** — Grauman et al. | CVPR 2024 | 1,286 h multi-view skilled activity + proficiency benchmark |
| **ExpertAF** — Ashutosh et al. | CVPR 2025 | Video + 3D pose → commentary + corrected demonstration. Up to **3×** over baselines |
| **CoachMe** — Yeh et al. | ACL 2025 | **Reference-based** coaching instructions. G-Eval 1.83 vs GPT-4o 1.39 |
| **Can VLMs Judge Action Quality?** | CVPRW 2026 | Frontier VLMs are **near chance** at AQA |
| **SkillFormer / ProfVLM** | 2025 | Efficient multi-view proficiency estimation |

## Group E — The surveys that define the consensus (3 papers)
| Paper | Venue | Why it matters |
|---|---|---|
| **A Decade of AQA** — Yin, Parmar et al. | IJCV | 200+ papers, PRISMA. Names scale, annotation granularity, interpretability and robustness as the field's open problems |
| **AQA Survey + Benchmark** — Zhou et al. | Pattern Recognition 2026 | Unified taxonomy **plus public benchmark code** re-running the main methods |
| **Skeleton-based Movement Assessment Review** | arXiv 2020 | The pose-only branch of the same field |

## Group F — Same task, clinical domain (2 papers)
**Rehabilitation assessment is sports AQA with a different label source** — useful because it is pose-only
by necessity and interpretability is non-negotiable there.

- **Liao, Vakanski & Xian**, IEEE TNSRE — deep learning framework for rehab exercise assessment (UI-PRMD)
- **Ismail-Fawaz et al.**, arXiv 2025 — standardised skeleton rehab benchmark (KIMORE, UI-PRMD)

---

## The other 23 — machinery and background, not problem-aligned

These are tools we *use* or context we *cite*, not papers solving our problem. They belong in Related Work
or Methods as one-line citations, and need no further reading.

- **Pose / skeleton / tracking:** ViTPose, ST-GCN, PoseC3D, VideoPose3D, ByteTrack, MS-TCN, FineGym, VideoMAE
- **Sports video at scale:** SoccerNet, SoccerNet-v2, SoccerNet-Tracking, TrackNet, TrackNetV4, and two
  sports surveys
- **Robustness / preprocessing:** PoseBench, HPE robustness (CVPR'21), 3D HPE robustness, video action
  recognition robustness, LIME-Eval
- **Dataset hygiene, calibration, fairness:** geospatial leakage/dedup, TVCalib, ethical bias in action
  recognition

**One exception worth keeping close: PoseC3D.** It is machinery, but it carries the person-box ablation
(Detection 75.8 / Tracking 85.3 / GT 92.0 on FineGym) that motivates our batter finder, plus the finding
that lifted 3D poses hurt. Treat it as a Group-C-adjacent paper despite being a backbone paper.

---

## The consensus pipeline, as Groups A and C agree on it

Reading the twelve method papers together, the field has converged on this. **This is the "existing
pipeline" the professor asked us to implement first.**

1. **Input:** a trimmed clip containing one performer, already cropped or with the performer's box known.
   *Subject selection is assumed away* — the single largest mismatch with broadcast cricket.
2. **Backbone:** I3D pretrained on Kinetics, applied to ~96–103 frames split into 9–10 overlapping
   16-frame snippets. Universal across MTL-AQA, USDL, CoRe, TSA, TPT, FineParser.
3. **Temporal structure:** one of three choices —
   (a) none, just pool the whole clip (MTL-AQA, USDL);
   (b) explicit step segmentation (TSA, MCoRe, and CricTAL in cricket);
   (c) learned queries with no phase labels (TPT) — which currently wins.
4. **Scoring head:** evolution from direct regression (MTL-AQA) → score *distributions* (USDL) →
   **contrastive regression against an exemplar** (CoRe, TSA, TPT, FineParser). Contrastive regression is
   the current consensus.
5. **Evaluation:** Spearman rank correlation, increasingly with **R-ℓ2** alongside it, on a random split.
6. **Feedback (new, rare):** LLM or VLM generating commentary — grounded in pose, because ungrounded VLMs
   are near chance.

**Where cricket sits relative to that consensus:** Moodley's I3D-AE-LSTM is roughly at step 4's *first*
stage — direct regression from fused features, no contrastive component, no score distributions, no
learned temporal parsing. The cricket literature is about three years behind the AQA methodology it
should be using. **That gap is itself an opportunity: applying the modern AQA recipe to cricket, properly
evaluated, is a defensible contribution even before any novel component is added.**
