#!/usr/bin/env bash
# Loss-weight grid sweep for the technique scorer's VAE (PROGRESS.md next step 2).
# kl_weight=0.01, reg_weight=5.0 were never swept before 2026-09-23's seed-controlled sweep confirmed
# results are stable enough to make tuning meaningful. Single fixed seed (0) during the search so the
# grid isn't confounded with run-to-run variance; the winning combo then gets re-run across the existing
# 3 seeds to confirm the gain is real, not noise (docs/paper/results_phase3.md's seed sweep found
# mean-Spearman moves +/-0.023 from seed alone).
set -uo pipefail
cd "$(dirname "$0")/.."
. .venv/bin/activate
mkdir -p models/tune_sweep
for kl in 0.001 0.01 0.1; do
  for reg in 2.0 5.0 10.0; do
    tag="kl${kl}_reg${reg}"
    echo "=== kl_weight=$kl reg_weight=$reg ==="
    OMP_NUM_THREADS=2 python models/train_technique_scorer.py --epochs 60 --seed 0 \
      --kl-weight "$kl" --reg-weight "$reg" \
      > "models/tune_sweep/${tag}.log" 2>&1
    cp models/technique_scorer/summary.json "models/tune_sweep/${tag}.json"
  done
done
echo "TUNE SWEEP DONE: 3x3 kl_weight/reg_weight grid"
