"""Large visual audit of striker selection + pose: numbered sheets from the frames the final run saved
(every 40th clip per worker). Each tile = window-start frame | contact frame, with the chosen batter's
box (yellow) and skeleton (red). Verdicts are recorded in kaggle/pose-rest-audit/verdicts.csv
(tile, clip_id, source, striker_ok, pose_ok) and summarised with 95% Wilson intervals.

    python models/audit_striker.py sheets     # build sheets + tile index
    python models/audit_striker.py summary    # after verdicts.csv is filled in
"""
import math
import sys
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
AUD = ROOT / "kaggle/pose-rest-audit"
PER_SHEET, COLS = 20, 2


def sheets() -> None:
    AUD.mkdir(parents=True, exist_ok=True)
    m = pd.read_parquet(ROOT / "data/processed/manifest.parquet").set_index("clip_id")
    files = sorted(ROOT.glob("kaggle/chunks/pose-rest-c*/out/frames/*.jpg"))
    rows, tiles = [], []
    for i, f in enumerate(files):
        img = cv2.imread(str(f))
        if img is None:
            continue
        w3 = img.shape[1] // 3
        pair = np.hstack([img[:, :w3], img[:, w3:2 * w3]])          # window start | contact
        pair = cv2.resize(pair, (800, int(pair.shape[0] * 800 / pair.shape[1])))
        cv2.rectangle(pair, (0, 0), (70, 26), (0, 0, 0), -1)
        cv2.putText(pair, f"#{len(rows)}", (4, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 255, 255), 2)
        rows.append({"tile": len(rows), "clip_id": f.stem, "source": m.at[f.stem, "source"] if f.stem in m.index else "?"})
        tiles.append(pair)
    h = max(t.shape[0] for t in tiles)
    tiles = [cv2.copyMakeBorder(t, 0, h - t.shape[0] + 4, 0, 4, cv2.BORDER_CONSTANT, value=(255, 255, 255)) for t in tiles]
    for s in range(0, len(tiles), PER_SHEET):
        chunk = tiles[s:s + PER_SHEET]
        while len(chunk) % COLS:
            chunk.append(np.full_like(tiles[0], 255))
        grid = np.vstack([np.hstack(chunk[r:r + COLS]) for r in range(0, len(chunk), COLS)])
        cv2.imwrite(str(AUD / f"sheet_{s // PER_SHEET:02d}.jpg"), grid, [cv2.IMWRITE_JPEG_QUALITY, 82])
    pd.DataFrame(rows).to_csv(AUD / "tiles.csv", index=False)
    print(f"{len(rows)} clips on {math.ceil(len(rows) / PER_SHEET)} sheets in {AUD};",
          pd.DataFrame(rows).source.value_counts().to_dict())


def wilson(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    if n == 0:
        return (float("nan"), float("nan"))
    p = k / n; d = 1 + z * z / n; c = p + z * z / (2 * n); r = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n))
    return ((c - r) / d, (c + r) / d)


def summary() -> None:
    v = pd.read_csv(AUD / "verdicts.csv")
    idx = pd.read_parquet(ROOT / "data/processed/pose_index.parquet")
    pop = idx[idx.batter_chosen_by == "batter_finder"].source.value_counts()
    lines, est = [], 0.0
    for src, g in v.groupby("source"):
        k, n = int(g.striker_ok.sum()), len(g); lo, hi = wilson(k, n); pk = int(g.pose_ok.sum())
        est += pop.get(src, 0) * k / n
        lines.append(f"{src:13} striker right {k}/{n} = {k/n:.0%} (95% CI {lo:.0%}-{hi:.0%}) | pose right {pk}/{n} = {pk/n:.0%} | clips in run: {pop.get(src, 0)}")
    k, n = int(v.striker_ok.sum()), len(v); lo, hi = wilson(k, n)
    lines += [f"ALL          striker right {k}/{n} = {k/n:.1%} (95% CI {lo:.0%}-{hi:.0%}) | pose right {int(v.pose_ok.sum())}/{n}",
              f"estimated clips with the right striker (weighted by source size): ~{est:,.0f} of {int(pop.sum()):,}"]
    (AUD / "audit_report.txt").write_text("\n".join(lines) + "\n"); print("\n".join(lines))


if __name__ == "__main__":
    {"sheets": sheets, "summary": summary}[sys.argv[1]]()
