"""D1: image-quality profile of every clip (CPU only, no GPU quota).

Measures, on 5 frames sampled evenly from each clip, the properties the IVA labs deal with:
  brightness / exposure  - mean and median grey level, share of clipped dark and blown-out pixels
  contrast               - grey-level standard deviation (RMS contrast) and the p5-p95 spread
  histogram shape        - entropy, and the dark / bright / low-contrast / high-contrast class
  uneven illumination    - spread of brightness across a 4x4 grid (shadow across the pitch)
  noise                  - Immerkaer's sigma estimate, plus residual after a 3x3 median filter
  sharpness / blur       - variance of the Laplacian and the Tenengrad (Sobel energy) score
  compression artefacts  - blockiness at the 8x8 JPEG/H.264 block boundaries
  colourfulness          - Hasler-Susstrunk metric
Output: quality_<chunk>.csv (one row per clip) and sample frames of the worst cases.
"""
import glob, json, os, sys, time
from concurrent.futures import ProcessPoolExecutor

import cv2
import numpy as np
import pandas as pd

CHUNK = int(os.environ.get("CRICLENS_CHUNK", "0"))
NCHUNKS = int(os.environ.get("CRICLENS_NCHUNKS", "1"))
LIMIT = int(os.environ.get("CRICLENS_LIMIT", "0"))
N_FRAMES = 5
OUT = "/kaggle/working"
MANIFEST = glob.glob("/kaggle/input/**/manifest.parquet", recursive=True)[0]
CLIPS = [c for c in glob.glob("/kaggle/input/**/clips", recursive=True) if os.path.isdir(f"{c}/cricketvision")][0]
os.makedirs(f"{OUT}/frames", exist_ok=True)


def noise_sigma(g: np.ndarray) -> float:
    """Immerkaer: convolve with a Laplacian-like mask; the mean absolute response scales with noise."""
    m = np.array([[1, -2, 1], [-2, 4, -2], [1, -2, 1]], np.float32)
    h, w = g.shape
    return float(np.abs(cv2.filter2D(g.astype(np.float32), -1, m)).sum() * np.sqrt(0.5 * np.pi) / (6 * (w - 2) * (h - 2)))


def blockiness(g: np.ndarray) -> float:
    """Compression artefacts: how much stronger the differences are across 8-pixel block edges."""
    d = np.abs(np.diff(g.astype(np.float32), axis=1))
    on = d[:, 7::8].mean() if d.shape[1] > 8 else 0.0
    off = np.delete(d, np.s_[7::8], axis=1).mean() if d.shape[1] > 8 else 1.0
    return float(on / (off + 1e-6))


def colourfulness(bgr: np.ndarray) -> float:
    b, g, r = cv2.split(bgr.astype(np.float32))
    rg, yb = r - g, 0.5 * (r + g) - b
    return float(np.sqrt(rg.std() ** 2 + yb.std() ** 2) + 0.3 * np.sqrt(rg.mean() ** 2 + yb.mean() ** 2))


def frame_metrics(bgr: np.ndarray) -> dict:
    g = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    hist = cv2.calcHist([g], [0], None, [256], [0, 256]).ravel()
    p = hist / max(hist.sum(), 1)
    h, w = g.shape
    cells = [g[y:y + h // 4, x:x + w // 4].mean() for y in range(0, h - h // 4 + 1, h // 4) for x in range(0, w - w // 4 + 1, w // 4)]
    med = cv2.medianBlur(g, 3)
    return {
        "brightness": float(g.mean()), "brightness_med": float(np.median(g)),
        "contrast_std": float(g.std()), "contrast_p5p95": float(np.percentile(g, 95) - np.percentile(g, 5)),
        "dark_clipped": float((g < 16).mean()), "blown_out": float((g > 240).mean()),
        "entropy": float(-(p[p > 0] * np.log2(p[p > 0])).sum()),
        "illum_uneven": float(np.std(cells)),
        "noise_sigma": noise_sigma(g), "noise_residual": float(np.abs(g.astype(np.float32) - med).mean()),
        "sharp_laplacian": float(cv2.Laplacian(g, cv2.CV_64F).var()),
        "sharp_tenengrad": float((cv2.Sobel(g, cv2.CV_64F, 1, 0, 3) ** 2 + cv2.Sobel(g, cv2.CV_64F, 0, 1, 3) ** 2).mean()),
        "blockiness": blockiness(g), "colourfulness": colourfulness(bgr),
    }


def clip_metrics(args) -> dict:
    clip_id, path = args
    cap = cv2.VideoCapture(path)
    n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    rows, first = [], None
    for q in np.linspace(0.05, 0.95, N_FRAMES):
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(q * max(n - 1, 0)))
        ok, f = cap.read()
        if ok:
            rows.append(frame_metrics(f))
            first = first if first is not None else f
    cap.release()
    if not rows:
        return {"clip_id": clip_id, "readable": False}
    d = pd.DataFrame(rows)
    out = {"clip_id": clip_id, "readable": True, "frames_read": len(rows),
           "height": first.shape[0], "width": first.shape[1]}
    out.update({k: float(d[k].median()) for k in d.columns})
    out["brightness_spread"] = float(d.brightness.max() - d.brightness.min())   # lighting change across the clip
    # lab-style classes (Lab 2: dark / bright / low contrast / high contrast)
    b, c = out["brightness"], out["contrast_std"]
    out["class"] = ("dark" if b < 70 else "bright" if b > 185 else
                    "low_contrast" if c < 35 else "high_contrast" if c > 75 else "normal")
    return out


if __name__ == "__main__":
    m = pd.read_parquet(MANIFEST)
    m = m[m.is_canonical].sort_values("clip_id").iloc[CHUNK::NCHUNKS]
    if LIMIT:
        m = m.head(LIMIT)
    jobs = [(r.clip_id, f"{CLIPS}/{r.source}/{r.clip_id}.mp4") for r in m.itertuples()]
    print(f"chunk {CHUNK}/{NCHUNKS}: {len(jobs)} clips", flush=True)
    t0, rows = time.time(), []
    with ProcessPoolExecutor(max(1, os.cpu_count() - 1)) as ex:
        for i, r in enumerate(ex.map(clip_metrics, jobs, chunksize=16)):
            rows.append(r)
            if i % 1000 == 0:
                print(f"{i}/{len(jobs)} {i / max(time.time() - t0, 1):.1f} clips/s", flush=True)
    d = pd.DataFrame(rows).merge(m[["clip_id", "source", "split"]], on="clip_id", how="left")
    d.to_csv(f"{OUT}/quality_{CHUNK}.csv", index=False)
    ok = d[d.readable]
    # sample frames of the worst cases, for the report
    for name, sub in (("darkest", ok.nsmallest(8, "brightness")), ("noisiest", ok.nlargest(8, "noise_sigma")),
                      ("blurriest", ok.nsmallest(8, "sharp_laplacian")), ("uneven_light", ok.nlargest(8, "illum_uneven"))):
        for j, r in enumerate(sub.itertuples()):
            cap = cv2.VideoCapture(f"{CLIPS}/{r.source}/{r.clip_id}.mp4")
            cap.set(cv2.CAP_PROP_POS_FRAMES, 2); okf, f = cap.read(); cap.release()
            if okf:
                cv2.imwrite(f"{OUT}/frames/{name}_{CHUNK}_{j}_{r.clip_id[:40]}.jpg", f)
    s = {"chunk": CHUNK, "clips": len(d), "readable": int(ok.shape[0]), "seconds": time.time() - t0,
         "class_counts": ok["class"].value_counts().to_dict(),
         "medians": {k: round(float(ok[k].median()), 2) for k in ("brightness", "contrast_std", "noise_sigma", "sharp_laplacian", "illum_uneven", "blockiness")}}
    json.dump(s, open(f"{OUT}/summary.json", "w"), indent=1)
    print(json.dumps(s, indent=1), flush=True)
