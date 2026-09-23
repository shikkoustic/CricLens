# CricLens: CSET344 (Image and Video Processing) syllabus alignment

Rule: a syllabus technique is listed only if CricLens genuinely needs it. Done = implemented with measured results
(details in iva_results.md). Planned = real use, not built yet.

## Module 1: Image fundamentals & spatial enhancement
| Syllabus topic | How CricLens uses it | Why it's genuine | Status |
|---|---|---|---|
| Sampling & quantization / spatial resolution | All 22,420 clips resampled to 480p; pose accuracy compared with original resolution | Joints differ by only ~0.7% of body height, so 480p is safe and saves storage/GPU | Done |
| Colour -> grayscale conversion | Grayscale frames for duplicate detection and quality measures | Brightness, contrast, sharpness and hashing are defined on intensity | Done |
| Distance measures (Euclidean) | Joint jitter, stride, head movement, joint comparison between frame versions | Every posture measurement is a distance between joints | Done |
| Thresholding (grayscale -> binary) | Letterbox/pillarbox black-bar detection | Black bars faked uneven lighting and had to be masked before enhancement | Done |
| Image arithmetic | Unsharp masking: original + k * (original - blurred) | Tested as the fix for CricShot10k blur | Done |
| Histogram processing | Brightness/contrast profile of all 22,420 clips; dark/bright/low/high-contrast classes | Showed darkness is not a real problem (14 clips); found CricShot10k degradation | Done |
| Histogram equalization vs CLAHE | Compared on 800 real clips for pose and bat detection | Global HE hurt pose; CLAHE raised bat detection 11% -> 19% (~89% of extra detections real) -> adopted before bat detection | Done |
| Gamma / power law, contrast stretching | Tested on dark/low-contrast clips and as repairs in the degradation study | Inverse gamma fully repaired darkened footage | Done |
| Noise models (Gaussian, salt-and-pepper, shot) | Added deliberately to clean clips to test robustness | Measures how real-world noise breaks pose estimation | Done |
| Mean, median, Gaussian filters | Repairs per noise type, compared on real joint error | Gaussian smoothing best for Gaussian noise; median helped JPEG artefacts | Done |
| Adaptive median filter | Repair for salt-and-pepper noise | Joint error 0.029 -> 0.005, clearly better than plain median | Done |
| Convolution & padding (zero / replicate) | Padding the batter crop when it runs off the frame | Zero padding cut pose error 20-60% on edge cases; affects 4,461 clips -> adopted | Done (re-run pending) |
| Laplacian / Sobel | Sharpness measures (Laplacian variance, Tenengrad) in the quality profile | How blur was detected across the dataset | Done |
| Canny edge detection | Crease and pitch lines | Needed for calibration | Done (D4) |

## Module 2: Edges, colour, morphology
| Syllabus topic | How CricLens uses it | Why it's genuine | Status |
|---|---|---|---|
| Colour models (CIELAB / HSV) | Enhancement on the luminance channel only; HSV pitch-strip segmentation (tan pitch vs green outfield) | Keeps colours intact; pitch segmentation is step 1 of calibration | Done |
| Hough line transform | Detect crease lines | Creases give known real distances for pixels -> cm | Done (D4, 20.0% of clips calibrated) |
| Morphological operations (opening, closing, hole filling) | Clean the pitch mask | Raw colour thresholds are noisy | Done (D4) |
| Connected components | Keep the pitch region, drop logos and crowd blobs | Picks the one real pitch area | Done (D4, bottom-centre blob, not just largest) |
| Shape moments | Bat angle from the bat mask | Bat angle is a coaching output | Planned (with bat model) |

## Module 3: Motion & frequency domain
| Syllabus topic | How CricLens uses it | Why it's genuine | Status |
|---|---|---|---|
| Homomorphic filtering (frequency domain) | Compared with CLAHE and local gamma for shadows across the pitch | Harshest method, hurt pose -> not adopted | Done |
| Motion detection (histogram / frame difference) | Camera-cut detection inside clips | The analysis window must never cross a cut | Done |
| DCT | Perceptual hashing to find duplicate clips across datasets | Removed 1,612 duplicates so no delivery is in both train and test | Done |
| Lossy compression / quantization | JPEG compression as a degradation; blockiness at 8x8 DCT block edges | Explains CricShot10k damage; median filter partly repairs it | Done |
| Optical flow | Estimate camera pan during a shot | Camera movement corrupts swing speed measured from joints | Planned |

## Module 4: Video processing & recognition
| Syllabus topic | How CricLens uses it | Why it's genuine | Status |
|---|---|---|---|
| Video sampling / frame-rate conversion | Resample 25 and 30 fps clips to one rate | Otherwise sequence models see shots at different speeds | Planned (next data fix) |
| Human action recognition from video | Shot classification from joint sequences | The core task of the project | Planned (training phase) |

## Deliberately left out (no genuine need yet)
Hough circles, Harris corners, SIFT, HOG, PCA, GLCM, bilateral filter, Viola-Jones, depth cameras.
