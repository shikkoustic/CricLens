# CricLens reading log — 53 papers

Status key: **F** = read in full from the publisher PDF · **T** = targeted read (abstract, method, results,
conclusion) · **✗** = could not retrieve.
All 53 PDFs were downloaded and text-extracted on 2026-09-21; every entry below was read at least at level T.
Detailed notes: `dossier.md`. Gap analysis: `related_work.md`.

## A. Cricket

| # | Paper | Venue / Year | Data | Method | Headline result | St |
|---|---|---|---|---|---|---|
| A1 | Moodley & van der Haar, **I3D-AE-LSTM** | WACV 2025, 5470–5478 | UJ-AQA-CricketVision, 8,540 clips (5,571 R / 2,969 L) | ViTPose + I3D, 2-stream AE, MLP, 5 part scores | **SRCC 0.84** (pose-only 0.79) | F |
| A2 | Moodley & van der Haar, **CricTAL** | ICCVW 2025 (SAUAFG), 2738–2745 | same | OpenPose + LSTM/RNN/TCN/Transformer, 35-frame window | TCN **90.4% acc, mAP@0.5 64.45%** | F |
| A3 | Kang, **Cricket baseline study** | arXiv:2510.09187, 2025 | CricShot10, 1,888 clips / 10 cls, random seed-27 split | 7 models re-implemented | 96%→**46.0%**, 99.2%→**55.6%**, 93%→**57.7%**; own SOTA **92.25%** | F |
| A4 | Moodley, van der Haar & Noorbhai, **Backlift** | Sci. Reports 12, 2022 | **200 images**, 2 classes (LBBT/SBBT) | AlexNet / InceptionV3 / IRv2 / Xception | Xception **98.2%**, P 100 / R 95 / F1 97–98 | T |
| A5 | Sen et al., **CricShotClassify** | Sensors 21(8):2846, 2021 | CricShot10 (introduced), 10 classes | CNN+GRU; VGG16-GRU transfer | frozen **86%**, fine-tuned **93%** | T |
| A6 | Rajarajeswari et al. | Sci. Reports, 2026 | Kaggle IPL-2023, ~1,050 clips / 7 cls | 3D CNN + optical flow + YOLOPose; cosine similarity vs pros | **91.37 / 92%** | T |
| A7 | Cricket HPE + ML | Sensors 23(15):6839, 2023 | small | MediaPipe + RF/SVM/KNN/LSTM | **99.77%** (implausible — cite as cautionary) | T |
| A8 | Gupta & Balan, stroke extraction | arXiv:1901.03107 / 1511.07607 | cricket | C3D + GRU localisation | mTIoU **0.9376** / **0.715** | T |
| A9 | CricShot10k | IEEE Access 2026 | 10,086 clips / 15 cls | CricShotNet, EfficientNetV2-S + GRU | **89%** | ✗ IEEE bot-block |
| A10 | I3D-AE-LSTM journal extension | Expert Systems w/ Applications | — | — | — | ✗ Elsevier paywall |

## B. Action Quality Assessment

| # | Paper | Venue / Year | Data | Method | Headline result | St |
|---|---|---|---|---|---|---|
| B1 | Parmar & Morris, **MTL-AQA** | CVPR 2019 | MTL-AQA, 1,412 dives / 16 events | C3D-AVG + MSCADC, multitask (recognition + commentary + score) | **0.9044** | T |
| B2 | Tang et al., **USDL / MUSDL** | CVPR 2020 | AQA-7 (1,189), MTL-AQA, JIGSAWS | I3D + Gaussian score distribution, KL loss; multi-path per judge | MTL-AQA **0.9273**; JIGSAWS 0.70 | F |
| B3 | Yu et al., **CoRe + GART** | ICCV 2021 | AQA-7, MTL-AQA, JIGSAWS | contrastive regression vs exemplar + group-aware regression tree (d=5), 10-exemplar voting; introduces **R-ℓ2** | MTL-AQA **0.9512** w/ DD; JIGSAWS **0.85** | F |
| B4 | Xu et al., **FineDiving + TSA** | CVPR 2022 | **FineDiving: 3,000 clips, 52 action / 29 sub-action types** | procedure segmentation → cross-attention → contrastive regression | **0.9203** w/ DN; **GT boundaries only +1 pt** | F |
| B5 | Bai et al., **Temporal Parsing Transformer** | ECCV 2022 | MTL-AQA, AQA-7, JIGSAWS | learnable queries → ordered part reps, **no part labels**; ranking + sparsity loss | MTL-AQA **0.9451**; AQA-7 avg **0.8715** | F |
| B6 | Xu et al., **FineParser** | CVPR 2024 | **FineDiving-HM** (+ foreground masks) | SAP + TAP + SVE + FineReg | **0.9435 / R-ℓ2 0.2602**; AIoU@0.5 **0.9946** | F |
| B7 | An, Qi & Ma, **MCoRe** | arXiv:2401.02841, 2024 | FineDiving | multi-stage contrastive regression | **0.9232**, AIoU 98.26/79.17 | T |
| B8 | Yin, Parmar et al., **Decade of AQA** | IJCV (arXiv:2502.02817) | survey, 200+ papers, 26 datasets / 9 domains | PRISMA | *"biggest AQA dataset has more than 20000 samples"* | F |
| B9 | Zhou et al., **AQA survey + benchmark** | Pattern Recognition 2026, DOI 10.1016/j.patcog.2026.113933 | survey + **unified benchmark code** | PRISMA + standardised re-runs | accuracy **and** efficiency reported | T |
| B10 | Qi et al., hierarchical pose-guided MCoRe | IEEE TIP | FineDiving | pose-guided multi-stage contrastive regression | **0.9266–0.9310** | T |

## C. Skill assessment and feedback

| # | Paper | Venue / Year | Data | Method | Headline result | St |
|---|---|---|---|---|---|---|
| C1 | Doughty et al., **Who's Better? Who's Best?** | CVPR 2018 | **EPIC-Skills** (surgery, dough, drawing, chopsticks) | Siamese pairwise deep ranking | **70–83%** correct pairs | T |
| C2 | Doughty et al., **The Pros and Cons** | CVPR 2019 | EPIC-Skills + **BEST** | rank-aware temporal attention (pros / cons branches) | **80.3 / 81.2%** (+4.3 / +5.4) | T |
| C3 | Grauman et al., **Ego-Exo4D** | CVPR 2024 | **1,286 h, 5,035 takes, 740 participants**, ego + multi-exo | 4 benchmarks incl. proficiency (4 classes) | expert commentary; L1-based mAP | T |
| C4 | Ashutosh et al., **ExpertAF** | CVPR 2025 | Ego-Exo4D (basketball, soccer, climbing) | video + 3D pose → commentary + retrieved demo + generated pose; weakly supervised via LLM | **up to 3×** over baselines (human eval) | F |
| C5 | Yeh et al., **CoachMe** | ACL 2025 | Figure skating + boxing | **reference-based** motion comparison (temporal + physical) | G-Eval **1.83** vs GPT-4o **1.39**; +31.6% / +58.3% | F |
| C6 | Monte e Freitas et al., **Can VLMs judge AQA?** | CVPRW 2026 (SAUAFG) | press (339), squat (224), FineFS, MTL-AQA (353) | Gemini 3 / Qwen3-VL / InternVL3.5 | **marginally above chance**; 2 systematic biases | F |
| C7 | Bianchi & Liotta, **SkillFormer** / **ProfVLM** | arXiv 2505.08665 / 2509.26278 | Ego-Exo4D | multi-view proficiency, parameter-efficient | **4.5× fewer** trainable params | T |
| C8 | Liao, Vakanski & Xian | IEEE TNSRE (arXiv:1901.10435) | UI-PRMD | skeleton → continuous quality score | — | T |
| C9 | Ismail-Fawaz et al., rehab benchmark | arXiv:2507.21018 | KIMORE, UI-PRMD | standardised benchmark | — | T |

## D. Machinery

| # | Paper | Venue / Year | Headline result | St |
|---|---|---|---|---|
| D1 | Xu et al., **ViTPose** | NeurIPS 2022 | **80.9 AP** COCO test-dev; scales 100M→1B | T |
| D2 | Yan et al., **ST-GCN** | AAAI 2018 | Kinetics-Skeleton **30.7 / 52.8%**; NTU **81.5 / 88.3%** | T |
| D3 | Duan et al., **PoseC3D** | CVPR 2022 | keypoint dropout: **<1%** loss vs GCN **14.3%**; **boxes: Detection 75.8 / Tracking 85.3 / GT 92.0**; lifted 3D **hurts** | F |
| D4 | Pavllo et al., **VideoPose3D** | CVPR 2019 | dilated temporal conv 2D→3D + semi-supervised back-projection | T |
| D5 | Zhang et al., **ByteTrack** | ECCV 2022 | MOT17 **80.3 MOTA / 77.3 IDF1 / 63.1 HOTA**, 30 FPS | T |
| D6 | Abu Farha & Gall, **MS-TCN** | CVPR 2019 | multi-stage dilated TCN + smoothing loss; up to **+12.6%** | T |
| D7 | Shao et al., **FineGym** | CVPR 2020 | **29K videos, 99 classes**, 3-level hierarchy | T |
| D8 | Tong et al., **VideoMAE** | NeurIPS 2022 | masked video pretraining | T |

## E. Sports video at scale

| # | Paper | Venue / Year | Headline result | St |
|---|---|---|---|---|
| E1 | Giancola et al., **SoccerNet** | CVPRW 2018 | **500 games, 764 h, 6,637 annotations**; 67.8% mAP / 49.7% spotting | T |
| E2 | Deliège et al., **SoccerNet-v2** | CVPRW 2021 | **~300k annotations**, + replay grounding | T |
| E3 | Cioppa et al., **SoccerNet-Tracking** | arXiv:2204.06918 | MOT in soccer broadcasts | T |
| E4 | Huang et al., **TrackNet** | 2019 | **P 99.7 / R 97.3 / F1 98.5**; cross-val **95.3 / 75.7 / 84.3** | T |
| E5 | **TrackNetV4** | arXiv:2409.14543 | + motion attention maps | T |
| E6 | Zhu et al., DL in sports survey | arXiv:2307.03353 | perception → comprehension → decision | T |
| E7 | Wu et al., sports action recognition survey | arXiv:2206.01038 | — | T |
| E8 | Naik et al., CV in sports review | Applied Sciences 12(9):4429 | — | ✗ MDPI block |

## F. Robustness, preprocessing, dataset hygiene

| # | Paper | Venue / Year | Headline result | St |
|---|---|---|---|---|
| F1 | Ma et al., **PoseBench** | arXiv:2406.14367 | **60 models, 10 corruptions**; ViTPose-H 78.84→65.02 mAP; vulnerable to **compression and blur**; robustness studied **at training time only** | F |
| F2 | Wang et al., HPE robustness | CVPR 2021 | **COCO-C / MPII-C / OCHuman-C**, 15 corruptions × 5 severities | T |
| F3 | Hoang et al., 3D HPE robustness | CVPRW 2024 | Human3.6M-C, HumanEva-I-C | T |
| F4 | Schiappa et al., video AR robustness | arXiv:2207.01398 | robust to temporal perturbation except SSv2; **capacity ≠ robustness**; **UCF101-DS** real shifts | T |
| F5 | Li, Zhao & Guo, **LIME-Eval** | arXiv:2410.08810 | retraining-based enhancement evaluation **overfits**; LIME-Bench (human preference) + energy-based LIME-Eval | F |
| F6 | Adimoolam et al., geospatial leakage | arXiv:2304.02296 | **89% train duplicates, 93–97% val leakage**; pHash dedupe pipeline | T |
| F7 | Theiner & Ewerth, **TVCalib** | WACV 2023 | field registration as camera calibration from segments | T |
| F8 | Băltăret,u et al., ethical bias in AR | arXiv:2604.17971 | synthetic controlled-appearance audit; skin tone / ethnicity, **not handedness** | T |

## Not retrieved (3)
1. **CricShot10k** (IEEE Access 2026) — IEEE Xplore blocks automated access. *Needed: it is one of our six sources.*
2. **I3D-AE-LSTM journal extension** (Expert Systems with Applications) — Elsevier paywall. *Needed: fuller version of our main baseline.*
3. **A Comprehensive Review of Computer Vision in Sports** (Applied Sciences 12(9):4429) — MDPI blocks automated access. *Low priority.*
