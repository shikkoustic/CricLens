"""D1 report: combine the Kaggle quality chunks into tables, figures and the D2 target list.

Inputs : kaggle/chunks/quality-c*/out/quality_*.csv and frames/
Outputs: data/processed/quality/
  quality_all.csv        one row per clip, all measurements
  report.txt             class counts, per-source medians, flagged-clip counts
  flags.csv              clips needing help (dark / low contrast / uneven light / noisy / blurry / blocky)
  fig_distributions.png  per-source histograms of brightness, contrast, noise, sharpness, uneven light
  worst_examples.jpg     sample frames of the worst clips per problem
Flag thresholds are percentile-based (worst 5% of the whole dataset) plus the Lab-2 brightness classes,
so they describe this dataset rather than an arbitrary cut-off.
"""
from pathlib import Path

import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "data/processed/quality"
METRICS = [("brightness", "Brightness (mean grey)"), ("contrast_std", "Contrast (grey std)"),
           ("noise_sigma", "Noise sigma (Immerkaer)"), ("sharp_laplacian", "Sharpness (Laplacian var)"),
           ("illum_uneven", "Uneven lighting (4x4 std)"), ("blockiness", "Blockiness (8x8)")]


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    d = pd.concat([pd.read_csv(f) for f in sorted(ROOT.glob("kaggle/chunks/quality-c*/out/quality_*.csv"))], ignore_index=True)
    d.to_csv(OUT / "quality_all.csv", index=False)
    ok = d[d.readable == True].copy()
    # dataset-relative flags: worst 5% on each problem
    th = {"dark": ok.brightness.quantile(0.05), "low_contrast": ok.contrast_std.quantile(0.05),
          "uneven_light": ok.illum_uneven.quantile(0.95), "noisy": ok.noise_sigma.quantile(0.95),
          "blurry": ok.sharp_laplacian.quantile(0.05), "blocky": ok.blockiness.quantile(0.95)}
    ok["f_dark"] = ok.brightness <= th["dark"]
    ok["f_low_contrast"] = ok.contrast_std <= th["low_contrast"]
    ok["f_uneven_light"] = ok.illum_uneven >= th["uneven_light"]
    ok["f_noisy"] = ok.noise_sigma >= th["noisy"]
    ok["f_blurry"] = ok.sharp_laplacian <= th["blurry"]
    ok["f_blocky"] = ok.blockiness >= th["blocky"]
    flags = [c for c in ok.columns if c.startswith("f_")]
    ok["n_flags"] = ok[flags].sum(axis=1)
    ok[ok.n_flags > 0][["clip_id", "source", "split", "n_flags"] + flags + [m for m, _ in METRICS]].to_csv(OUT / "flags.csv", index=False)

    lines = [f"clips profiled: {len(d)} | readable: {len(ok)} | unreadable: {len(d) - len(ok)}", "",
             "Lab-2 brightness/contrast classes:", ok["class"].value_counts().to_string(), "",
             "class by source (share):", pd.crosstab(ok.source, ok["class"], normalize="index").round(3).to_string(), "",
             "median measurements by source:", ok.groupby("source")[[m for m, _ in METRICS]].median().round(2).to_string(), "",
             "flag thresholds (worst 5% of dataset): " + ", ".join(f"{k} {v:.2f}" for k, v in th.items()), "",
             "flagged clips by problem and source:",
             ok.groupby("source")[flags].sum().astype(int).to_string(), "",
             f"clips with >=1 problem: {int((ok.n_flags > 0).sum())} ({(ok.n_flags > 0).mean():.1%}) | with >=2: {int((ok.n_flags >= 2).sum())}"]
    (OUT / "report.txt").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))

    fig, axes = plt.subplots(2, 3, figsize=(15, 8))
    for ax, (m, title) in zip(axes.ravel(), METRICS):
        lo, hi = ok[m].quantile([0.005, 0.995])
        for src, g in ok.groupby("source"):
            ax.hist(g[m].clip(lo, hi), bins=50, histtype="step", density=True, label=src)
        ax.set_title(title); ax.set_yticks([])
    axes[0, 0].legend(fontsize=8)
    fig.suptitle("D1: image-quality distributions across 22,420 clips, by dataset"); fig.tight_layout()
    fig.savefig(OUT / "fig_distributions.png", dpi=110)

    rows = []
    for name in ("darkest", "noisiest", "blurriest", "uneven_light"):
        fs = sorted(ROOT.glob(f"kaggle/chunks/quality-c*/out/frames/{name}_*.jpg"))[:4]
        tiles = [cv2.resize(cv2.imread(str(f)), (320, 180)) for f in fs if cv2.imread(str(f)) is not None]
        if tiles:
            while len(tiles) < 4:
                tiles.append(np.zeros_like(tiles[0]))
            row = np.hstack(tiles)
            cv2.putText(row, name, (6, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
            rows.append(row)
    if rows:
        cv2.imwrite(str(OUT / "worst_examples.jpg"), np.vstack(rows))


if __name__ == "__main__":
    main()
