"""Visual review for picking contrastive-regression exemplar clips (PROGRESS.md: technique scorer rewrite).

For each candidate clip, renders three frames (window start, contact, end) from the CricketVision video
with the COCO-17 skeleton overlaid on the already-chosen batter, so a reviewer can confirm the pose is
clean and the shot is clearly visible before it gets baked into the model as a fixed reference.

    python models/exemplar_review.py sheets --candidates /path/to/exemplar_candidates.csv
"""
import sys
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "kaggle/exemplar-review"

SKELETON = [
    (5, 7), (7, 9), (6, 8), (8, 10),       # arms
    (5, 6), (5, 11), (6, 12), (11, 12),    # torso
    (11, 13), (13, 15), (12, 14), (14, 16),  # legs
    (0, 5), (0, 6),                        # head to shoulders
]


def read_frame(video_path: Path, frame_idx: int):
    cap = cv2.VideoCapture(str(video_path))
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
    ok, f = cap.read()
    cap.release()
    return f if ok else None


def draw_skeleton(f: np.ndarray, kp: np.ndarray, color=(0, 255, 255)) -> None:
    for a, b in SKELETON:
        if kp[a, 2] > 0.3 and kp[b, 2] > 0.3:
            cv2.line(f, tuple(kp[a, :2].astype(int)), tuple(kp[b, :2].astype(int)), color, 2)
    for x, y, c in kp:
        if c > 0.3:
            cv2.circle(f, (int(x), int(y)), 3, color, -1)


def render_clip(clip_id: str, source: str, label: str) -> np.ndarray | None:
    video = ROOT / "data/interim/clips" / source / f"{clip_id}.mp4"
    kps_candidates = list(ROOT.glob(f"kaggle/chunks/pose-cv-c*/out/kps/{clip_id}.npz"))
    if not video.exists() or not kps_candidates:
        return None
    z = np.load(kps_candidates[0])
    lo, contact, hi = z["window"]
    kps = z["kps"]
    frames = []
    for idx in (int(lo), int(contact), int(hi)):
        f = read_frame(video, idx)
        if f is None:
            return None
        draw_skeleton(f, kps[idx - int(lo)] if idx - int(lo) < len(kps) else kps[-1])
        frames.append(f)
    h = min(fr.shape[0] for fr in frames)
    frames = [cv2.resize(fr, (int(fr.shape[1] * h / fr.shape[0]), h)) for fr in frames]
    strip = np.hstack(frames)
    cv2.rectangle(strip, (0, 0), (min(260, strip.shape[1]), 20), (0, 0, 0), -1)
    cv2.putText(strip, label, (3, 15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
    return strip


def sheets(candidates_csv: str) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    cand = pd.read_csv(candidates_csv)
    for shot, grp in cand.groupby("shot"):
        rows, missing = [], 0
        strips = []
        for r in grp.itertuples():
            label = f"{r.tier[:4]} s={r.score_overall:.1f} {r.clip_id[:28]}"
            img = render_clip(r.clip_id, r.source, label)
            if img is None:
                missing += 1
                continue
            strips.append(img)
            rows.append(r.clip_id)
        if not strips:
            print(f"{shot}: nothing rendered ({missing} missing)")
            continue
        w = max(s.shape[1] for s in strips)
        strips = [cv2.copyMakeBorder(s, 0, 4, 0, w - s.shape[1], cv2.BORDER_CONSTANT, value=(255, 255, 255))
                  for s in strips]
        sheet = np.vstack(strips)
        cv2.imwrite(str(OUT / f"{shot}.jpg"), sheet, [cv2.IMWRITE_JPEG_QUALITY, 85])
        print(f"{shot}: {len(strips)} rendered ({missing} missing) -> {OUT / f'{shot}.jpg'}")


if __name__ == "__main__":
    sheets(sys.argv[sys.argv.index("--candidates") + 1])
