# CricLens dataset datasheet

Everything here is assembled from public datasets; nothing was recorded or hand-labelled.
All video comes from TV broadcast footage, so the whole dataset is for **academic,
non-commercial use only**, whatever license tag an uploader attached. Clips are not
redistributed; the repo ships code, manifests and derived labels only.

Final counts per split and the dedupe results are in `data/processed/manifest_report.txt`.

**Summary (build of 2026-09-11):** 24,032 decoded clips -> **22,420 canonical clips** after
removing 1,612 verified duplicates; splits train 16,206 / val 3,137 / test 3,077; no match or
duplicate cluster appears in two splits. 7,072 canonical clips carry CricketVision technique
scores (6,892 with handedness: 4,400 right, 2,492 left).

## 1. Shot-clip sources

| Source | Clips | Labels | Native format | License / access |
|---|---|---|---|---|
| CricketVision (WACV 2025) | 8,441 strokes from 202 source videos | stroke type (6), front/back foot, handedness, 1-10 scores for head/shoulders/hands/hips/feet per phase, phase timestamps, batter bbox, dismissal | 720p 25 fps full-length videos, cut to strokes by us | No license stated; academic use. Dropbox |
| CricShot10k (IEEE Access 2026) | 10,091 | 15 shot classes; match id from filename (`vid<match>_<n>`) | 30 fps, ~0.9 s | Research / non-commercial. Google Drive |
| CricShot10 (Sensors 2021) | 1,750 | 10 shot classes; no match id | mostly 720p 25 fps, ~2.5 s | CC0 repo, data by request; HF mirror `rokmr/cricket-shot` |
| Cricket Shots IPL 2023 (Kaggle) | 1,922 | 7 classes; match id from filename | 720p 25 fps, ~1 s | Apache-2.0 per uploader |
| amittalmale/cricket-shots (Kaggle) | 562 | 8 classes; source highlights video from filename | batter-cropped 640x360, 1.2 s | CC0 per uploader |
| KUCricShot (ICCIT 2023) | 1,266 of 1,278 | 4 classes | 720p 30 fps, ~1 s | Not stated; academic use. Google Drive |

## 2. Processing

- Every clip is re-encoded to H.264 yuv420p, short side 480 px (never upscaled), native fps
  kept, audio removed (audio is out of scope), `+faststart`.
- CricketVision: the 24 GB Dropbox archive is streamed (no HTTP range support) with a custom
  reader (`datasets/zipstream.py`); each source video is cut into strokes from Buildup start
  - 0.3 s to FollowThrough end + 0.3 s, then deleted. Per-part score = mean over the 3 phases;
  overall = mean of the 5 parts.
- Labels are unified (`datasets/taxonomy.py`) to 8 shots + off/leg/straight side. The original
  label is always kept in `label_orig`. Foot and handedness come only from explicit
  annotations (CricketVision), never guessed from shot type.

| Unified shot | Source labels |
|---|---|
| drive | Cover/Straight Drive, cover, straight, drive, OffDrive, OnDrive, on/off drive, Drive Shot |
| defence | Defensive, defense, Block, defence, Defensive Shot |
| flick_glance | Flick, flick, Glance, glance, Flick Shot |
| pull_hook | Pull, Hook, pull, hook, Pull Shot |
| cut | Late/Square/Upper Cut, late_cut, square_cut, cut, Cut |
| sweep | Sweep, Reverse Sweep, sweep |
| lofted | Lofted Legside/Offside, lofted, slog |
| scoop | Scoop |
| other (kept, excluded from the shot head) | Down The Wicket, misc, unorthodox |

- Deduplication (`datasets/build_manifest.py`), two stages on frames sampled at 5 fps:
  1. recall: 64-bit pHash LSH; candidate pair if >= 60% of the shorter clip's frames have a
     match within Hamming distance 6;
  2. verification: 256-bit pHash sequences aligned at the best time offset (overlap >= 60% of
     the shorter clip) must differ by <= 20 bits per frame.
  Stage 1 alone is not enough: broadcast deliveries from one match share camera angle and
  pitch, so it merged different balls and chained whole matches into clusters of up to 297
  clips (checked by eye: different scoreboards). Calibration on sampled pairs: same stroke
  ~0 bits, different balls in one match ~58, unrelated ~84. One canonical clip is kept per
  verified cluster (preference: CricketVision, CricShot10k, CricShot10, IPL 2023, KUCricShot).
- Splits: StratifiedGroupKFold (7 folds: 1 test, 1 val, 5 train), grouped by source match
  (or duplicate cluster when no match id exists), stratified on the unified shot.

## 3. Detection / segmentation images

`data/processed/detection/` (built by `datasets/build_detection.py`), classes ball, bat, stumps.

| Set | Images | train / val / test | Use |
|---|---|---|---|
| det (boxes) | 23,799 | 19,850 / 1,691 / 2,258 | ball/bat/stumps detector |
| seg (polygons) | 5,999 | 4,720 / 707 / 572 | bat (and ball) segmentation U-Net |

Sources: Roboflow `powerinflow` (polygons), `cricket-ball-segmentation` (polygons),
`cricket-dataset-z2wkt`, `cricket-balls-l1us5`, `cricket-bat-detection` (boxes, CC BY 4.0 /
public domain, attribution required) and Kaggle `kushagra3204/cricket-ball-dataset-for-yolo`
(CC0). Roboflow augmented copies are grouped with their original frame and near-identical
frames across sources are merged, so a frame never appears in two splits. People are dropped
(the pose model finds the batter).

Pretrained models from the CricShot10k authors (`data/raw/cricshot10k/models/`): ball
detection, bat detection, striker bat segmentation, player-type detection.

## 4. Known issues and decisions

- **CricketVision stroke-id bug**: 183 of 203 annotation files ship a StrokeType option table
  that drops Sweep and renumbers Block to 5, while their values still use 6 for Block. We
  decode all files with the full 7-option table; the resulting counts match the dataset
  README (back foot 2,941 vs 2,943; front foot 5,266 vs 5,273).
- **CricketVision contains some source videos twice** under different annotator folders
  (e.g. `P3_V6` = `P4_V9`, `P5_V35` = `P6_V31`: identical frames, same stroke index): 1,156
  strokes. Their two annotation sets match almost exactly (Spearman 0.98, mean abs diff 0.08,
  97.7% same stroke label), so they are copies of one annotation, **not** independent ratings,
  and cannot serve as an inter-rater ceiling. The manifest keeps one clip per stroke. Without
  this dedupe, the same stroke would sit in train and test under two different video names.
- CricketVision's "Cut" class is published as "Cut / Square Drive", so some square drives sit
  under `cut`.
- 225 CricketVision strokes have no stroke label (`shot = other`); 198 have no handedness.
- **The CricShot10 HF mirror's own split leaks**: 157 verified duplicate clusters inside
  CricShot10, of which 75 span its train/val/test folders (38 involve its test folder), mostly
  as the same clip under different file names. Results reported on that split are optimistic.
  We ignore it and re-split. CricShot10 has no match ids, so its clips are grouped only by
  duplicate cluster.
- 12 KUCricShot files could not be fetched from Google Drive after 3 retries.
- Roboflow `naveen-akash/lbw-9pb1e` (2,019 bat polygons) was skipped: its owner never
  generated a version, so it cannot be downloaded through the API.
- Everything is broadcast footage (mostly behind-the-bowler); there is no phone/nets footage.
  Camera-angle robustness is handled later in pose space.

## 5. Reproduce

```
datasets/download.sh cricketvision_ann cricshot10 cricshot10k_models cricshot10k_shots kucricshot kaggle_sets roboflow_sets
python datasets/cricketvision.py parse && python datasets/cricketvision.py stream
python datasets/ingest_clips.py cricshot10k cricshot10 ipl2023 amittalmale kucricshot
python datasets/build_detection.py
python datasets/build_manifest.py
```
Kaggle needs `~/.kaggle/access_token`; Roboflow needs `ROBOFLOW_API_KEY` in `.env`.
