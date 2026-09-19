"""D2: does classical enhancement (IVA labs 1-4) help the analysis on the clips that need it?

Group A (chunk 0): CricShot10k clips flagged blurry + blocky/low-contrast, and clean controls.
    A0 baseline | A1 median 3x3 (deblocking) | A2 unsharp mask | A3 CLAHE | A4 contrast stretch | A5 median->unsharp->CLAHE
Group B (chunk 1): clips flagged for uneven lighting, and clean controls.
    B0 baseline | B1 global histogram equalisation | B2 CLAHE | B3 homomorphic filtering | B4 local gamma correction
Intensity methods act on the luminance (L of CIELAB) so colours are kept, and only inside the picture area:
letterbox/pillarbox black bars are found by thresholding row/column means and masked out.

The batter box per frame is reused from the pose run, so only the pose model is re-run.
Per clip x method: joint confidence (all / wrists / ankles), jitter, bat-detection rate inside the batter
box (CricLens detector), Striker confirmation rate (player-type model), and for the batter crop: sharpness,
contrast, and change vs baseline (SSIM, PSNR, histogram intersection).
"""
import glob, json, os, subprocess, sys, time, traceback

GROUP = "AB"[int(os.environ.get("CRICLENS_CHUNK", "0"))]
LIMIT = int(os.environ.get("CRICLENS_LIMIT", "0"))
subprocess.run([sys.executable, "-m", "pip", "install", "-q", "ultralytics", "-U", "transformers>=4.48", "accelerate"], check=False)
import cv2, numpy as np, pandas as pd, torch
from skimage.metrics import structural_similarity, peak_signal_noise_ratio
from transformers import AutoProcessor, VitPoseForPoseEstimation
from ultralytics import YOLO

DEV = "cuda" if torch.cuda.is_available() else "cpu"
OUT = "/kaggle/working"; os.makedirs(f"{OUT}/examples", exist_ok=True)
D2 = glob.glob("/kaggle/input/**/samples.csv", recursive=True)[0].rsplit("/", 1)[0]
CLIPS = [c for c in glob.glob("/kaggle/input/**/clips", recursive=True) if os.path.isdir(f"{c}/cricketvision")][0]
KPS = glob.glob("/kaggle/input/**/kps", recursive=True)
KPS = [k for k in KPS if glob.glob(f"{k}/*.npz")][0]
PAD = 0.12
proc = AutoProcessor.from_pretrained("usyd-community/vitpose-base-simple")
vit = VitPoseForPoseEstimation.from_pretrained("usyd-community/vitpose-base-simple", torch_dtype=torch.float16 if DEV == "cuda" else torch.float32).to(DEV).eval()
det = YOLO(f"{D2}/detector_best.pt"); BAT = [i for i, n in det.names.items() if n == "bat"]
ptype = YOLO(glob.glob("/kaggle/input/**/Player_Type_Detection_Model.pt", recursive=True)[0])
STRIKER = [i for i, n in ptype.names.items() if n.lower() == "striker"]
Y = dict(device=0 if DEV == "cuda" else "cpu", verbose=False, half=DEV == "cuda")
CLAHE = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))


# ---------------- enhancement methods ----------------
def content_rect(frames):
    """Lab 1/2: black bars = rows/columns whose mean grey level stays below a threshold in every sampled frame."""
    g = np.stack([cv2.cvtColor(f, cv2.COLOR_BGR2GRAY) for f in frames[:: max(1, len(frames) // 5)]]).max(0)
    rows, cols = np.where(g.mean(1) > 18)[0], np.where(g.mean(0) > 18)[0]
    h, w = g.shape
    return (int(cols[0]), int(rows[0]), int(cols[-1]) + 1, int(rows[-1]) + 1) if len(rows) and len(cols) else (0, 0, w, h)


def on_luma(bgr, rect, fn):
    lab = cv2.cvtColor(bgr, cv2.COLOR_BGR2LAB)
    x0, y0, x1, y1 = rect
    L = lab[y0:y1, x0:x1, 0]
    lab[y0:y1, x0:x1, 0] = np.clip(fn(L), 0, 255).astype(np.uint8)
    return cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)


def unsharp(bgr, sigma=1.5, amount=1.0):     # Lab 1 arithmetic: original + amount * (original - blurred)
    return cv2.addWeighted(bgr, 1 + amount, cv2.GaussianBlur(bgr, (0, 0), sigma), -amount, 0)


def stretch(L):                                # Lab 2 piecewise-linear contrast stretch between p2 and p98
    lo, hi = np.percentile(L, (2, 98))
    return (L.astype(np.float32) - lo) * 255.0 / max(hi - lo, 1)


def homomorphic(L, gl=0.5, gh=1.5, c=1.0, d0_frac=0.04):
    """Lab 3: log -> FFT -> Gaussian high-emphasis filter (damp slow illumination, boost reflectance) -> exp."""
    f = np.log1p(L.astype(np.float32))
    F = np.fft.fftshift(np.fft.fft2(f))
    h, w = L.shape
    v, u = np.meshgrid(np.arange(h) - h / 2, np.arange(w) - w / 2, indexing="ij")
    d0 = d0_frac * min(h, w)
    H = (gh - gl) * (1 - np.exp(-c * (u ** 2 + v ** 2) / (2 * d0 ** 2))) + gl
    g = np.expm1(np.real(np.fft.ifft2(np.fft.ifftshift(H * F))))
    lo, hi = np.percentile(g, (1, 99))
    return (g - lo) * 255.0 / max(hi - lo, 1e-6)


def local_gamma(L):
    """Lab 3: per-pixel gamma from a smooth illumination estimate (dark regions brightened, bright ones calmed)."""
    illum = cv2.GaussianBlur(L.astype(np.float32), (0, 0), max(L.shape) / 30)
    gamma = np.power(2.0, (128.0 - illum) / 128.0)
    return 255.0 * np.power(L.astype(np.float32) / 255.0, gamma)


METHODS = {
    "A": {"A0_baseline": lambda f, r: f,
          "A1_median3": lambda f, r: cv2.medianBlur(f, 3),
          "A2_unsharp": lambda f, r: unsharp(f),
          "A3_clahe": lambda f, r: on_luma(f, r, CLAHE.apply),
          "A4_stretch": lambda f, r: on_luma(f, r, stretch),
          "A5_median_unsharp_clahe": lambda f, r: on_luma(unsharp(cv2.medianBlur(f, 3)), r, CLAHE.apply)},
    "B": {"B0_baseline": lambda f, r: f,
          "B1_global_he": lambda f, r: on_luma(f, r, cv2.equalizeHist),
          "B2_clahe": lambda f, r: on_luma(f, r, CLAHE.apply),
          "B3_homomorphic": lambda f, r: on_luma(f, r, homomorphic),
          "B4_local_gamma": lambda f, r: on_luma(f, r, local_gamma)},
}[GROUP]


# ---------------- measurement ----------------
def iou(a, b):
    x0, y0, x1, y1 = max(a[0], b[0]), max(a[1], b[1]), min(a[2], b[2]), min(a[3], b[3])
    i = max(0, x1 - x0) * max(0, y1 - y0)
    return i / ((a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - i + 1e-9)


def padded(b, H, W):
    pw, ph = PAD * (b[2] - b[0]), PAD * (b[3] - b[1])
    return [int(max(0, b[0] - pw)), int(max(0, b[1] - ph)), int(min(W, b[2] + pw)), int(min(H, b[3] + ph))]


def pose(frames, boxes):
    H, W = frames[0].shape[:2]
    k = np.full((len(frames), 17, 3), np.nan, np.float32)
    idx = [i for i in range(len(frames)) if np.isfinite(boxes[i]).all()]
    for s in range(0, len(idx), 64):
        ch = idx[s:s + 64]; crops, offs = [], []
        for i in ch:
            x0, y0, x1, y1 = padded(boxes[i], H, W)
            crops.append(cv2.cvtColor(frames[i][y0:y1, x0:x1], cv2.COLOR_BGR2RGB)); offs.append((x0, y0))
        bx = [[[0, 0, c.shape[1], c.shape[0]]] for c in crops]
        inp = proc(images=crops, boxes=bx, return_tensors="pt").to(DEV)
        if DEV == "cuda":
            inp["pixel_values"] = inp["pixel_values"].half()
        with torch.no_grad():
            o = vit(**inp)
        o.heatmaps = o.heatmaps.float()
        for i, (ox, oy), r in zip(ch, offs, proc.post_process_pose_estimation(o, boxes=bx)):
            kp = r[0]["keypoints"].float().cpu().numpy(); kp[:, 0] += ox; kp[:, 1] += oy
            k[i, :, :2] = kp; k[i, :, 2] = r[0]["scores"].float().cpu().numpy()
    return k


def measure(frames, boxes, k, orig_crop_gray, mid):
    H, W = frames[0].shape[:2]
    hs = boxes[:, 3] - boxes[:, 1]
    a = np.linalg.norm(k[2:, :, :2] - 2 * k[1:-1, :, :2] + k[:-2, :, :2], axis=-1) / hs[1:-1, None]
    row = {"conf_all": float(np.nanmean(k[:, :, 2])), "conf_wrists": float(np.nanmean(k[:, [9, 10], 2])),
           "conf_ankles": float(np.nanmean(k[:, [15, 16], 2])), "jitter": float(np.nanmedian(a)) if np.isfinite(a).any() else np.nan}
    sel = list(range(0, len(frames), 3))
    hits = 0
    for i, r in zip(sel, det.predict([frames[i] for i in sel], conf=0.25, **Y)):
        if not np.isfinite(boxes[i]).all():
            continue
        x0, y0, x1, y1 = padded(boxes[i], H, W)
        hits += any(int(c) in BAT and x0 <= (b[0] + b[2]) / 2 <= x1 and y0 <= (b[1] + b[3]) / 2 <= y1
                    for b, c in zip(r.boxes.xyxy.cpu().numpy(), r.boxes.cls.cpu().numpy()))
    row["bat_rate"] = hits / max(1, sum(np.isfinite(boxes[i]).all() for i in sel))
    first = [i for i in range(min(len(frames), 12)) if np.isfinite(boxes[i]).all()][::3][:4]
    sh = 0
    for i, r in zip(first, ptype.predict([frames[i] for i in first], conf=0.25, **Y)) if first else []:
        sh += any(int(c) in STRIKER and iou(b, boxes[i]) > 0.3 for b, c in zip(r.boxes.xyxy.cpu().numpy(), r.boxes.cls.cpu().numpy()))
    row["striker_rate"] = sh / max(1, len(first))
    x0, y0, x1, y1 = padded(boxes[mid], H, W)
    g = cv2.cvtColor(frames[mid][y0:y1, x0:x1], cv2.COLOR_BGR2GRAY)
    row["crop_sharpness"] = float(cv2.Laplacian(g, cv2.CV_64F).var())
    row["crop_contrast"] = float(g.std())
    if orig_crop_gray is not None and orig_crop_gray.shape == g.shape and min(g.shape) >= 7:
        row["ssim_vs_orig"] = float(structural_similarity(orig_crop_gray, g, data_range=255))
        row["psnr_vs_orig"] = float(peak_signal_noise_ratio(orig_crop_gray, g, data_range=255)) if not np.array_equal(orig_crop_gray, g) else 99.0
        h1 = cv2.calcHist([orig_crop_gray], [0], None, [64], [0, 256]); h2 = cv2.calcHist([g], [0], None, [64], [0, 256])
        row["hist_intersection"] = float(cv2.compareHist(cv2.normalize(h1, h1, 1, 0, cv2.NORM_L1), cv2.normalize(h2, h2, 1, 0, cv2.NORM_L1), cv2.HISTCMP_INTERSECT))
    return row, g


samples = pd.read_csv(f"{D2}/samples.csv")
samples = samples[samples.group == GROUP]
if LIMIT:
    samples = samples.groupby("kind").head(LIMIT)
man = pd.read_parquet(glob.glob("/kaggle/input/**/manifest.parquet", recursive=True)[0]).set_index("clip_id")
rows, t0 = [], time.time()
print(f"group {GROUP}: {len(samples)} clips, methods {list(METHODS)}, device {DEV}", flush=True)
for n, s in enumerate(samples.itertuples()):
    try:
        z = np.load(f"{KPS}/{s.clip_id}.npz"); boxes_all = z["boxes"]; lo, c, hi = [int(v) for v in z["window"]]
        a, b = max(lo, c - 25), min(hi, c + 20)
        cap = cv2.VideoCapture(f"{CLIPS}/{s.source}/{s.clip_id}.mp4"); frames, i = [], 0
        while i <= b:
            ok, f = cap.read()
            if not ok:
                break
            if i >= a:
                frames.append(f)
            i += 1
        cap.release()
        if len(frames) < 5:
            continue
        boxes = boxes_all[a:a + len(frames)].astype(np.float64)
        rect = content_rect(frames); H, W = frames[0].shape[:2]
        bars = rect != (0, 0, W, H)
        mid = min(c - a, len(frames) - 1)
        orig_gray, example = None, []
        for name, fn in METHODS.items():
            ef = [fn(f.copy(), rect) for f in frames]
            k = pose(ef, boxes)
            row, g = measure(ef, boxes, k, orig_gray, mid)
            if orig_gray is None:
                orig_gray = g
            rows.append({"clip_id": s.clip_id, "source": s.source, "kind": s.kind, "method": name, "black_bars": bars, **row})
            if n < 6:
                x0, y0, x1, y1 = padded(boxes[mid], H, W)
                crop = ef[mid][y0:y1, x0:x1]
                example.append(cv2.putText(cv2.resize(crop, (160, int(160 * crop.shape[0] / max(crop.shape[1], 1)))), name.split("_", 1)[0], (3, 14), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 255), 1))
        if example:
            hmax = max(e.shape[0] for e in example)
            cv2.imwrite(f"{OUT}/examples/{GROUP}_{n}_{s.kind}.jpg", np.hstack([cv2.copyMakeBorder(e, 0, hmax - e.shape[0], 0, 4, cv2.BORDER_CONSTANT) for e in example]))
        if n % 25 == 0:
            print(f"{n}/{len(samples)} {time.time() - t0:.0f}s", flush=True)
    except Exception:
        print("clip failed", s.clip_id, traceback.format_exc()[-500:], flush=True)

d = pd.DataFrame(rows); d.to_csv(f"{OUT}/d2_{GROUP}.csv", index=False)
base = [m for m in METHODS if "baseline" in m][0]
metrics = ["conf_all", "conf_wrists", "conf_ankles", "jitter", "bat_rate", "striker_rate", "crop_sharpness", "crop_contrast"]
p = d.pivot_table(index=["clip_id", "kind"], columns="method", values=metrics)
summary = {"group": GROUP, "clips": int(d.clip_id.nunique()), "seconds": time.time() - t0, "results": {}}
for kind in ("flagged", "control"):
    pk = p.xs(kind, level="kind") if kind in p.index.get_level_values("kind") else None
    if pk is None:
        continue
    for mth in METHODS:
        r = {}
        for mt in metrics:
            delta = pk[(mt, mth)] - pk[(mt, base)]
            r[mt] = round(float(pk[(mt, mth)].mean()), 4)
            r[f"{mt}_delta"] = round(float(delta.mean()), 4)
            r[f"{mt}_improved_share"] = round(float(((delta < 0) if mt == "jitter" else (delta > 0)).mean()), 3)
        sub = d[(d.kind == kind) & (d.method == mth)]
        for mt in ("ssim_vs_orig", "psnr_vs_orig", "hist_intersection"):
            if mt in sub:
                r[mt] = round(float(sub[mt].mean()), 4)
        summary["results"][f"{kind}/{mth}"] = r
json.dump(summary, open(f"{OUT}/summary.json", "w"), indent=1)
print(json.dumps(summary, indent=1), flush=True)
