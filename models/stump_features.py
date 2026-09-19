"""Stumps clues for every tracked person (the user's rule: the striker stands IN FRONT of the far
stumps; the keeper and slips are BEHIND them; the non-striker and umpire are BESIDE the near stumps).

Inputs : kaggle/chunks/stumps-c0/out/stumps_dets.csv  (stumps boxes on 4 early frames per clip)
         kaggle/chunks/pose-cv-c*/out/kps/*.npz        (every tracked person's boxes per frame)
Output : kaggle/stumps/out/stump_features.csv  (clip_id, tid, st_*), read by train_batter_finder.py
Measures are in units of the person's height, from their feet (box bottom centre) to the far
stumps' base (bottom centre of the stumps box that is highest in the frame = farthest away).
"""
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DETS = ROOT / "kaggle/chunks/stumps-c0/out/stumps_dets.csv"
OUT = ROOT / "kaggle/stumps/out/stump_features.csv"


def person_stump_features(track_boxes: dict[int, dict[int, np.ndarray]], stumps: dict[int, np.ndarray]) -> list[dict]:
    """track_boxes: tid -> {frame: [x0,y0,x1,y1]}; stumps: frame -> array of stumps boxes (k,4)."""
    rows = []
    for tid, fb in track_boxes.items():
        dxs, dys, seen = [], [], 0
        for f, sb in stumps.items():
            if not len(sb):
                continue
            seen += 1
            near = [g for g in fb if abs(g - f) <= 2]
            if not near:
                continue
            b = fb[min(near, key=lambda g: abs(g - f))]
            far = sb[np.argmin(sb[:, 3])]                     # stumps whose base is highest = far end
            th = max(b[3] - b[1], 1.0)
            dxs.append(((b[0] + b[2]) / 2 - (far[0] + far[2]) / 2) / th)
            dys.append((b[3] - far[3]) / th)                  # > 0: feet in front of (below) the stumps' base
        dx, dy = (float(np.median(dxs)), float(np.median(dys))) if dxs else (np.nan, np.nan)
        rows.append({"tid": tid, "st_found": seen / max(len(stumps), 1), "st_dx": dx, "st_dy": dy,
                     "st_far": float(bool(dxs)),
                     "st_in_front": float(bool(dxs) and abs(dx) < 0.6 and 0.0 <= dy <= 0.6),
                     "st_behind": float(bool(dxs) and abs(dx) < 0.8 and dy < -0.05),
                     "st_beside": float(bool(dxs) and abs(dx) >= 0.6)})
    return rows


def main() -> None:
    dets = pd.read_csv(DETS)
    st = dets[dets.cls == "stumps"]
    by_clip = {c: {f: g[["x0", "y0", "x1", "y1"]].to_numpy() for f, g in cg.groupby("frame")} for c, cg in st.groupby("clip_id")}
    frames_by_clip = dets.groupby("clip_id")["frame"].unique().to_dict()
    out = []
    for npz in sorted(ROOT.glob("kaggle/chunks/pose-cv-c*/out/kps/*.npz")):
        z = np.load(npz); cid = npz.stem
        tracks: dict[int, dict[int, np.ndarray]] = {}
        for t, f, b in zip(z["track_ids"], z["track_frames"], z["track_boxes"]):
            tracks.setdefault(int(t), {})[int(f)] = b
        stumps = by_clip.get(cid, {int(f): np.zeros((0, 4)) for f in frames_by_clip.get(cid, [])})
        for r in person_stump_features(tracks, stumps):
            out.append({"clip_id": cid, **r})
    OUT.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(out).to_csv(OUT, index=False)
    print(f"{len(out)} tracked people from {len({r['clip_id'] for r in out})} clips -> {OUT}")


if __name__ == "__main__":
    main()
