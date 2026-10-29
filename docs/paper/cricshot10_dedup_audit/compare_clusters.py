"""Audit every CricShot10 duplicate cluster: pairwise, time-aligned frame comparison.

For each pair of clips in a cluster, extract small grayscale frame signatures, find the best
temporal alignment offset (clips may be trimmed differently, different lengths), then compute the
mean absolute pixel difference at that alignment. Low diff = same footage (just re-exported/
trimmed differently); high diff even at best alignment = visually similar shot but different
footage (different match) -- NOT a true duplicate.
"""
import sys
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

ROOT = Path("/home/user/CricLens")
CLIPS = ROOT / "data/interim/clips/cricshot10"


def load_gray(cid: str, size=(64, 36)) -> np.ndarray:
    p = CLIPS / f"{cid}.mp4"
    cap = cv2.VideoCapture(str(p))
    frames = []
    while True:
        ok, f = cap.read()
        if not ok:
            break
        g = cv2.cvtColor(cv2.resize(f, size), cv2.COLOR_BGR2GRAY)
        frames.append(g.astype(np.float32))
    cap.release()
    return np.stack(frames) if frames else np.zeros((0, size[1], size[0]), np.float32)


def best_alignment_diff(a: np.ndarray, b: np.ndarray) -> tuple[float, int, int]:
    """Try every offset -len(b)+1 .. len(a)-1, return (min mean abs diff, offset, n overlapping frames)."""
    na, nb = len(a), len(b)
    if na == 0 or nb == 0:
        return 255.0, 0, 0
    best = (1e9, 0, 0)
    for off in range(-nb + 1, na):
        if off >= 0:
            ia, ib = a[off:], b[: na - off]
        else:
            ia, ib = a[: nb + off], b[-off:]
        n = min(len(ia), len(ib))
        if n < 3:
            continue
        ia, ib = ia[:n], ib[:n]
        d = float(np.mean(np.abs(ia - ib)))
        if d < best[0]:
            best = (d, off, n)
    return best


def main():
    clusters = pd.read_csv("/tmp/claude-0/-home-user-CricLens/dup_verify/all_clusters.csv")
    rows = []
    for dc, grp in clusters.groupby("dup_cluster"):
        clips = grp.clip_id.tolist()
        cache = {c: load_gray(c) for c in clips}
        for i in range(len(clips)):
            for j in range(i + 1, len(clips)):
                a, b = clips[i], clips[j]
                diff, off, n = best_alignment_diff(cache[a], cache[b])
                rows.append({"dup_cluster": dc, "clip_a": a, "clip_b": b,
                             "mean_diff": diff, "offset": off, "n_overlap": n,
                             "len_a": len(cache[a]), "len_b": len(cache[b])})
        print(f"done cluster {dc} ({len(clips)} clips)", flush=True)
    out = pd.DataFrame(rows)
    out.to_csv("/tmp/claude-0/-home-user-CricLens/dup_verify/pair_diffs.csv", index=False)
    print(f"\n{len(out)} pairs compared")
    print(out.mean_diff.describe())


if __name__ == "__main__":
    main()
