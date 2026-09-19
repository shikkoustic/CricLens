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

## Decisions for the pipeline
1. Pose: no image enhancement; switch crop padding from clamp to zero padding; re-run pose on the 4,461 edge clips (boxes reused).
2. Bat detection: CLAHE on luminance first.
3. User uploads (later): median / adaptive median when impulse noise is detected, Gaussian smoothing for heavy Gaussian noise, inverse gamma for dark footage; no global HE / homomorphic.
