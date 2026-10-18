# IVA experiments D1-D3 (CricLens)

## D1 - dataset quality profile (22,420 clips, Kaggle CPU)
- Darkness is not a problem: 14 clips are dark by the Lab-2 classes.
- CricShot10k is heavily degraded: Laplacian sharpness 86 vs 341-713 elsewhere, contrast 25 vs 32-41, blockiness 1.26 vs 1.09-1.14.
- Global contrast (Lab 2) labels 71% of clips "low contrast" because grass dominates the frame; the Immerkaer noise estimate confuses crowd/advert texture with noise.
- Image-quality measures correlate weakly with pose outcomes (|Spearman| < 0.3).

## D2 - enhancement on flagged real clips (800 clips, 6+5 methods)
- No method improves pose. Strong global changes hurt: global HE wrist conf -0.036, contrast stretch -0.024, homomorphic -0.021; jitter up.
- CLAHE beats global HE (Lab 2); local gamma is the gentlest of Lab 3's illumination methods (SSIM 0.985); homomorphic the harshest.
- CLAHE raises bat detections inside the batter box (A: 11% -> 19%, B: 7% -> 22%).

## Bat check (101 disagreement frames judged by eye)
- CLAHE-only detections: 89, of which ~79 are the real bat (~89% real). Wrong ones: keeper/stumps in close-up keeper shots (3 frames from one clip), stumps, edge artefacts.
- Baseline-only detections: 12, of which ~10 real.
- Conclusion: CLAHE's extra bat detections are mostly genuine - use CLAHE (clip 2.0, 8x8, on L of Lab) before bat detection only.

## D3a - deliberate degradation and restoration (135 clean clips; error vs clean pose, / batter height)
| degradation | none | best restoration |
|---|---|---|
| salt & pepper 5% | err 0.029, PSNR 18.6, SSIM 0.40 | adaptive median err 0.005, PSNR 36.7, SSIM 0.98 (median3: 0.010) |
| Gaussian noise s=25 | 0.022 | Gaussian smoothing 0.020 (median3 0.021, NL-means 0.027 = worst despite best SSIM 0.84) |
| shot noise | 0.025 | Gaussian smoothing 0.022 (NL-means 0.027) |
| blur s=3 | 0.030 | unsharp 0.028 |
| dark (gamma 2.2) | 0.011 | inverse gamma 0.002 (CLAHE 0.014, local gamma 0.019) |
| low contrast | 0.013 | stretch 0.013 (CLAHE 0.016, global HE 0.021) |
| JPEG q15 | 0.027 | median3 0.024 |
Findings: blur, impulse noise and compression hurt pose most; darkness and low contrast barely. Adaptive median > median for impulse noise (Lab 4 Q4). Gaussian smoothing > median > NL-means for Gaussian noise on pose, although NL-means has the best SSIM: image-quality scores do not predict task accuracy.

## D3b - padding at the frame edge (Lab 4 Q2)
Synthetic edge through a fully visible batter, error on joints still in frame (vs uncut reference):
| scenario | clamp (current) | zero | replicate | reflect |
|---|---|---|---|---|
| bottom 85% | 0.0180 | **0.0073** | 0.0152 | 0.0150 |
| bottom 70% | 0.0216 | **0.0139** | 0.0187 | 0.0218 |
| right 85% | 0.0187 | **0.0076** | 0.0085 | 0.0117 |
| right 70% | 0.0233 | **0.0190** | 0.0248 | 0.0459 |
Zero padding beats clamping in 97% of paired cases; replicate in 74%; reflect only 58% (mirrors body parts into fake limbs).
Clamping changes the crop's aspect ratio and distorts the pose input; zero padding keeps the geometry and the black area carries no false body evidence.
Real edge clips (150): differences are small on proxy metrics (confidence 0.827 -> 0.832).
Scale: the padded crop leaves the frame in 44.6% of clips, and in >=30% of window frames for 19.9% (4,461 clips); CricShot10k 41.5%.

## D4 - pitch calibration: pixels -> cm (22,420 clips, Kaggle CPU, ~4.5 min/chunk x 4)
Method: `models/pitch_calibration.py` / `kaggle/pitch-calib/pitch_calib.py`. HSV threshold (tan pitch vs
green outfield) -> morphological close/open -> connected components (keep the blob touching bottom-
centre of frame, not just the largest -- a wide shot's crowd/stand region can win on area alone) ->
white-line threshold restricted to that region -> thin/elongated-contour filter (crease markings are a
few px wide and long; kit, pads and gloves are not) -> probabilistic Hough transform. Two near-vertical,
laterally-offset lines are the two return creases, always 2.64 m apart by law -> a local cm/px scale at
the crease. Falls back to `kaggle/chunks/stumps-c0`'s existing stumps detections (known 71.1 cm height)
when no crease pair is found, cross-checked against any crease estimate (reject if they disagree >40%)
and against absolute physical plausibility (stride 10-250 cm, swing <=40 m/s) before trusting either.

Prototyped and validated locally first on 9 hand-picked clips across all 3 main sources (overlays
checked visually, per the project's hard rule to check outputs before trusting a run at scale). That
caught and fixed two real bugs before the full run: a scoreboard-bar false positive (fixed by hard-
cropping the HUD band inside the white-line mask itself, not just the pitch mask that fed its ROI), and
a false crease pairing that produced a 12 m "stride" -- two fragments of one horizontal seam (the pitch/
scoreboard boundary) satisfied the "near-parallel, laterally offset" pair test the same way a real
return-crease pair would. Fixed by requiring pair candidates to be near-vertical first: a genuine return
crease is always closer to vertical than horizontal in frame, since the camera looks down the pitch
length, so a horizontal artifact can never pass that filter regardless of how its fragments pair up.

**Full-dataset result**: 4,478 / 22,420 clips (**20.0%**) got a usable calibration (3,028 crease-based,
the syllabus technique; 1,450 stumps-fallback); 467 more (2.1%) were attempted but rejected by the
plausibility/cross-check (correctly caught, not silently trusted). 1,635 clips (7.3%) yielded a real
stride measurement (needs both ankles visible at both window ends -- a stricter condition than swing);
4,433 (19.8%) yielded a swing-speed measurement. Median stride 174.7 cm, median swing 14.5 m/s across
the successfully-calibrated clips -- both squarely in plausible ranges for real batting strokes.

Yield varies sharply by source (crease detection needs the crease to be visible and resolvable at each
source's native resolution and camera framing):
| source | usable calib | median stride (cm) | median swing (m/s) |
|---|---|---|---|
| cricketvision | **38.2%** | 195.4 | 15.2 |
| kucricshot | 30.6% | 179.3 | 14.9 |
| cricshot10 | 25.1% | 175.1 | 13.0 |
| cricshot10k | 13.7% | 166.0 | 14.4 |
| amittalmale | 5.0% | 177.8 | 12.0 |
| ipl2023 | 3.5% | 203.4 | 11.0 |

CricketVision (hand-annotated batter boxes, closer/more consistent camera framing) calibrates 11x more
often than IPL2023 (wide broadcast shots where the crease is a handful of pixels at 480p) -- expected,
and consistent with D1's finding that per-source quality varies a lot in this dataset. **Honest reading**:
this is a real, working classical-CV measurement, not a demo restricted to hand-picked clips, but its
yield is source-dependent and its absolute numbers (a few thousand clips) are not yet enough coverage to
be a primary per-clip feature for the shot classifier or technique scorer -- it is usable today as a
coaching-report add-on (stride/swing numbers alongside the technique score) for the ~1 in 5 clips it
calibrates, and as validated ground truth if a future pass tries to estimate these from pose alone on the
other 80%. Output: `data/processed/iva/pitch_calibration.parquet`.

## D5 - bat segmentation U-Net: trains well, does not transfer to real footage
`kaggle/bat-seg/bat_seg.py`: ResNet18-encoder U-Net on `data/processed/detection/seg/`'s 5,999
polygon-labelled bat images. **Test IoU 0.81, Dice 0.83** (572 held-out test images) -- a strong number,
but on the SAME distribution as training: most of `seg/`'s images (Roboflow `powerinflow` etc.) are
clean product/equipment photography (a bat on its own against a plain background), not match footage.

**Checked against our own clips before trusting it (CLAUDE.md's rule): it fails completely.** Ran the
trained model on 9 real clip frames (including one cropped tightly around a clearly-visible bat, using
the batter box we already have from pose extraction) -- max predicted probability 0.04-0.10 across all 9,
nowhere near the 0.5 threshold, i.e. the model finds no bat at all even when one is plainly in frame.
This is a genuine domain gap (motion blur, 480p compression, players' bodies partially occluding the bat,
natural lighting -- none of which the mostly-clean training images represent), not a scale/cropping
issue. **Bat angle from shape moments is implemented and correct on the training distribution (verified
on real bat photos), but the segmentation model behind it is not yet usable on the project's actual
clips.** Needs either real-footage bat annotations or heavy domain-randomisation augmentation
(motion blur, JPEG/H.264 compression, synthetic occlusion) before it can feed the coaching pipeline --
not attempted yet, flagged as the next step in PROGRESS.md rather than silently shipped as "done".

## Decisions for the pipeline
1. Pose: no image enhancement; switch crop padding from clamp to zero padding; re-run pose on the 4,461 edge clips (boxes reused).
2. Bat detection: CLAHE on luminance first.
3. User uploads (later): median / adaptive median when impulse noise is detected, Gaussian smoothing for heavy Gaussian noise, inverse gamma for dark footage; no global HE / homomorphic.
4. Pitch calibration: crease-line detection first (syllabus technique), stumps-detection fallback, cross-checked and plausibility-bounded before trusting either -- gets a real cm/px scale for ~20% of clips; ship it as a per-clip coaching add-on, not a universal pipeline dependency.
