"""Large-scale visual review for retraining the batter finder across all 6 sources (PROGRESS.md item 6).

Unlike `models/audit_striker.py` (pass/fail only, 400 clips sampled during the original Kaggle run), this
renders EVERY tracked person's box with their track id, so a reviewer can record which id is actually the
batter -- a usable correction, not just a flag. Needs the clip's video locally (data/interim/clips/) and its
already-computed kps.npz (already present for all clips from the original pose-rest run).

    python models/finder_review.py sheets --sample /path/to/sample.csv   # build numbered sheets
    python models/finder_review.py summary                              # after verdicts.csv is filled in
"""
import sys
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "kaggle/finder-review"
PER_SHEET, COLS = 24, 4
COLORS = [(0, 255, 255), (255, 120, 0), (0, 255, 0), (255, 0, 255), (0, 128, 255), (180, 180, 0)]


def read_frame(video_path: Path, frame_idx: int):
    cap = cv2.VideoCapture(str(video_path))
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
    ok, f = cap.read()
    cap.release()
    return f if ok else None


def render_clip(clip_id: str, source: str) -> np.ndarray | None:
    video = ROOT / "data/interim/clips" / source / f"{clip_id}.mp4"
    npz_candidates = list(ROOT.glob(f"kaggle/chunks/pose-rest-c*/out/kps/{clip_id}.npz"))
    if not video.exists() or not npz_candidates:
        return None
    z = np.load(npz_candidates[0])
    lo, contact, hi = z["window"]
    f = read_frame(video, int(contact))
    if f is None:
        return None
    tids, tfr, tbx = z["track_ids"], z["track_frames"], z["track_boxes"]
    chosen = int(z["chosen_tid"])
    near = np.abs(tfr - contact) <= 2
    seen = {}
    for tid, fr, bx in zip(tids[near], tfr[near], tbx[near]):
        if tid not in seen or abs(fr - contact) < abs(seen[tid][0] - contact):
            seen[tid] = (fr, bx)
    for tid, (fr, bx) in seen.items():
        c = COLORS[int(tid) % len(COLORS)]
        x0, y0, x1, y1 = bx.astype(int)
        thick = 3 if tid == chosen else 1
        cv2.rectangle(f, (x0, y0), (x1, y1), c, thick)
        cv2.putText(f, str(int(tid)), (x0 + 2, max(14, y0 + 14)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, c, 2)
    if chosen in seen:
        cv2.putText(f, f"chosen: {chosen}", (6, f.shape[0] - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                    (0, 255, 255), 2)
    else:
        cv2.putText(f, "chosen: not visible near contact", (6, f.shape[0] - 8), cv2.FONT_HERSHEY_SIMPLEX,
                    0.45, (0, 0, 255), 2)
    return f


def sheets(sample_csv: str) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    sample = pd.read_csv(sample_csv)
    rows, tiles = [], []
    missing = 0
    for r in sample.itertuples():
        img = render_clip(r.clip_id, r.source)
        if img is None:
            missing += 1
            continue
        h, w = img.shape[:2]
        img = cv2.resize(img, (260, int(260 * h / w)))
        cv2.rectangle(img, (0, 0), (46, 18), (0, 0, 0), -1)
        cv2.putText(img, f"#{len(rows)}", (3, 14), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 255), 1)
        rows.append({"tile": len(rows), "clip_id": r.clip_id, "source": r.source})
        tiles.append(img)
    h = max(t.shape[0] for t in tiles)
    tiles = [cv2.copyMakeBorder(t, 0, h - t.shape[0] + 3, 0, 3, cv2.BORDER_CONSTANT, value=(255, 255, 255))
             for t in tiles]
    for s in range(0, len(tiles), PER_SHEET):
        chunk = tiles[s:s + PER_SHEET]
        while len(chunk) % COLS:
            chunk.append(np.full_like(tiles[0], 255))
        grid = np.vstack([np.hstack(chunk[rw:rw + COLS]) for rw in range(0, len(chunk), COLS)])
        cv2.imwrite(str(OUT / f"sheet_{s // PER_SHEET:03d}.jpg"), grid, [cv2.IMWRITE_JPEG_QUALITY, 80])
    pd.DataFrame(rows).to_csv(OUT / "tiles.csv", index=False)
    n_sheets = -(-len(rows) // PER_SHEET)
    print(f"{len(rows)} clips rendered ({missing} missing video/pose data) on {n_sheets} sheets in {OUT}")


def summary() -> None:
    v = pd.read_csv(OUT / "verdicts.csv")
    n = len(v)
    ok = (v.correct_tid == v.chosen_tid).sum() if "chosen_tid" in v else v.ok.sum()
    print(f"{ok}/{n} = {ok/n:.1%} already correct; {n-ok} corrections recorded")
    if "error" in v:
        print(v[v.correct_tid != v.chosen_tid].error.value_counts())


if __name__ == "__main__":
    if sys.argv[1] == "sheets":
        sheets(sys.argv[sys.argv.index("--sample") + 1])
    else:
        summary()
