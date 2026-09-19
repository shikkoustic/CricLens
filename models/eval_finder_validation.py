"""Score the batter finder on footage it was not trained on (the validation run's output).

1. Twin clips: KUCricShot / CricShot10 copies of CricketVision deliveries. The CricketVision copy's
   batter comes from its hand-drawn box; the finder must pick the same person in the other copy.
   Boxes are compared in normalised coordinates (share of frame width/height) at the best time
   offset between the two copies; correct = IoU > 0.5. Twins whose CricketVision pick was itself
   wrong are skipped.
2. Random clips: contact sheets (window start / contact / end) for a visual check, per source.
"""
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
VAL = ROOT / "kaggle/pose-rest-val/out"
FINDER_DATA = ROOT / "data/kaggle_upload/criclens-finder"


def iou(a, b):
    x0, y0, x1, y1 = max(a[0], b[0]), max(a[1], b[1]), min(a[2], b[2]), min(a[3], b[3])
    i = max(0.0, x1 - x0) * max(0.0, y1 - y0)
    return i / ((a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - i + 1e-9)


def norm_boxes(npz: Path, clip_path: Path) -> np.ndarray:
    cap = cv2.VideoCapture(str(clip_path)); w, h = cap.get(cv2.CAP_PROP_FRAME_WIDTH), cap.get(cv2.CAP_PROP_FRAME_HEIGHT); cap.release()
    return np.load(npz)["boxes"] / np.array([w, h, w, h])


def best_aligned_iou(a: np.ndarray, b: np.ndarray, max_shift: int = 60) -> float:
    best = 0.0
    for off in range(-max_shift, max_shift + 1):
        ious = [iou(a[i], b[i + off]) for i in range(len(a)) if 0 <= i + off < len(b)
                and np.isfinite(a[i]).all() and np.isfinite(b[i + off]).all()]
        if len(ious) >= 5:
            best = max(best, float(np.median(ious)))
    return best


def twins() -> pd.DataFrame:
    pairs = pd.read_csv(FINDER_DATA / "cv_twin_pairs.csv")
    m = pd.read_parquet(ROOT / "data/processed/manifest.parquet").set_index("clip_id")
    cvq = pd.concat([pd.read_csv(f) for f in ROOT.glob("kaggle/chunks/pose-cv-c*/out/per_clip.csv")]).set_index("clip_id")
    rows = []
    for r in pairs.itertuples():
        vn, cn = VAL / "kps" / f"{r.clip_id}.npz", next(ROOT.glob(f"kaggle/chunks/pose-cv-c*/out/kps/{r.cv_twin}.npz"), None)
        if not vn.exists() or cn is None or r.cv_twin not in cvq.index:
            continue
        cv_ok = cvq.at[r.cv_twin, "cv_iou_final"] > 0.5
        a = norm_boxes(vn, ROOT / m.at[r.clip_id, "path"]); b = norm_boxes(cn, ROOT / m.at[r.cv_twin, "path"])
        rows.append({"clip_id": r.clip_id, "source": r.source, "cv_pick_ok": cv_ok, "iou_vs_cv": best_aligned_iou(a, b)})
    d = pd.DataFrame(rows)
    d["finder_correct"] = d["iou_vs_cv"] > 0.5
    return d


def sheets() -> None:
    val = pd.read_csv(FINDER_DATA / "validation_clips.csv")
    rnd = val[val.kind == "random"]
    for src, g in rnd.groupby("source"):
        tiles = [cv2.imread(str(f)) for cid in g.clip_id for f in [VAL / "frames" / f"{cid}.jpg"] if f.exists()]
        if not tiles:
            continue
        tiles = [cv2.resize(t, (1200, int(t.shape[0] * 1200 / t.shape[1]))) for t in tiles]
        out = VAL / f"check_{src}.jpg"
        cv2.imwrite(str(out), np.vstack(tiles), [cv2.IMWRITE_JPEG_QUALITY, 80]); print("sheet:", out, len(tiles))


if __name__ == "__main__":
    t = twins()
    ok = t[t.cv_pick_ok]
    print(f"twin clips scored: {len(ok)} (skipped {len(t) - len(ok)} whose CricketVision pick was wrong)")
    print(f"finder picks the same batter as CricketVision: {ok.finder_correct.mean():.1%}")
    print(ok.groupby("source").finder_correct.agg(["mean", "size"]).round(3).to_string())
    t.to_csv(VAL / "twin_eval.csv", index=False)
    sheets()
