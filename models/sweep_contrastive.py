"""Small hyperparameter sweep for the contrastive technique scorer (PROGRESS.md 2026-10-17 update).

The first full run early-stopped after only 3 productive epochs with the val curve still falling,
suggesting the default lr/patience/hidden size were mismatched to this pair-dataset's scale rather than
the approach being wrong. Runs a handful of configs sequentially (re-using one cached process so the
~37k-pair dataset and exemplar sequences load only once per config) and prints a comparison table
against the VAE baseline (models/technique_scorer/summary.json).

    python models/sweep_contrastive.py
"""
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from models.train_technique_scorer_contrastive import run  # noqa: E402

CONFIGS = [
    {"tag": "lr3e4_p15", "lr": 3e-4, "hidden": 128, "patience": 15},
    {"tag": "lr3e4_h256_p15", "lr": 3e-4, "hidden": 256, "patience": 15},
    {"tag": "lr1e4_p20", "lr": 1e-4, "hidden": 128, "patience": 20},
    {"tag": "lr3e4_h64_p15", "lr": 3e-4, "hidden": 64, "patience": 15},
]

BASELINE = json.load(open(ROOT / "models/technique_scorer/summary.json"))


def main():
    results = []
    for cfg in CONFIGS:
        print(f"\n{'='*70}\nconfig: {cfg}\n{'='*70}")
        r = run(epochs=80, limit=None, device="cpu", batch_size=64, **cfg)
        results.append(r)

    print(f"\n{'='*70}\nSWEEP SUMMARY\n{'='*70}")
    print(f"{'tag':20s} {'lr':>8s} {'hidden':>6s} {'spearman':>9s} {'r_l2':>6s} {'partial':>8s}")
    print(f"{'VAE baseline':20s} {'-':>8s} {'-':>6s} {BASELINE['test_mean_spearman']:9.3f} "
          f"{BASELINE['test_mean_r_l2']:6.2f} {BASELINE['test_mean_partial_spearman_vs_overall']:8.3f}")
    for cfg, r in zip(CONFIGS, results):
        print(f"{cfg['tag']:20s} {cfg['lr']:8.0e} {cfg['hidden']:6d} {r['test_mean_spearman']:9.3f} "
              f"{r['test_mean_r_l2']:6.2f} {r['test_mean_partial_spearman_vs_overall']:8.3f}")


if __name__ == "__main__":
    main()
