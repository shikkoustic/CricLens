#!/usr/bin/env bash
# Runs the shot classifier and technique scorer across 3 seeds each, sequentially, to characterise
# training-run variance (docs/paper/results_phase3.md found unseeded reruns moved several points on
# their own -- larger than some effects being measured). Tags each run's summary.json by seed so nothing
# gets overwritten. One completion line at the end for a single background-wait notification.
set -uo pipefail
cd "$(dirname "$0")/.."
. .venv/bin/activate
mkdir -p models/seed_sweep
for seed in 0 1 2; do
  echo "=== shot classifier seed=$seed ==="
  python models/train_shot_classifier.py --arch all --epochs 30 --seed "$seed" \
    > "models/seed_sweep/shot_clf_seed${seed}.log" 2>&1
  cp models/shot_classifier/summary.json "models/seed_sweep/shot_clf_seed${seed}.json"
  echo "=== technique scorer seed=$seed ==="
  OMP_NUM_THREADS=2 python models/train_technique_scorer.py --epochs 60 --seed "$seed" \
    > "models/seed_sweep/tech_scorer_seed${seed}.log" 2>&1
  cp models/technique_scorer/summary.json "models/seed_sweep/tech_scorer_seed${seed}.json"
done
echo "SEED SWEEP DONE: 3 seeds x 2 models"
