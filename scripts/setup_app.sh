#!/usr/bin/env bash
# Fetch everything the web app (app/) needs that isn't in git, onto a fresh checkout of this branch.
# Requires a working `kaggle` CLI (~/.kaggle/kaggle.json or ~/.kaggle/access_token) with access to
# shikkoustic/criclens-all and shikkoustic/criclens-app-weights (both private, shared with collaborators).
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"; cd "$ROOT"
[ -d .venv ] && source .venv/bin/activate

echo "== 1/3: processed data (pose index, stumps detections, batter finder) via fetch_data.sh =="
bash scripts/fetch_data.sh

echo "== 2/3: player-type detection model (top-level file, not in the processed tarball) =="
mkdir -p data/raw/cricshot10k/models
TMP="$ROOT/data/_download"; rm -rf "$TMP"; mkdir -p "$TMP"
kaggle datasets download shikkoustic/criclens-all -f models/Player_Type_Detection_Model.pt -p "$TMP"
for z in $(find "$TMP" -name "*.zip"); do unzip -q -o "$z" -d "$(dirname "$z")" && rm "$z"; done
find "$TMP" -name "Player_Type_Detection_Model.pt" -exec mv {} data/raw/cricshot10k/models/ \;
rm -rf "$TMP"

echo "== 3/3: app-specific weights not yet folded into criclens-all (shot classifier, technique scorer, =="
echo "==      pitch calibration results, plus guaranteed-path copies of the batter finder + detector)   =="
TMP="$ROOT/data/_download"; mkdir -p "$TMP"
kaggle datasets download shikkoustic/criclens-app-weights -p "$TMP" --unzip
mkdir -p models/shot_classifier models/technique_scorer data/processed/iva kaggle/train-detector/out/detector/weights
mv "$TMP/lstm.pt" models/shot_classifier/
mv "$TMP/shot_classifier_summary.json" models/shot_classifier/summary.json
mv "$TMP/vae_regressor.pt" models/technique_scorer/
mv "$TMP/technique_scorer_summary.json" models/technique_scorer/summary.json
mv "$TMP/pitch_calibration.parquet" data/processed/iva/
[ -f models/batter_finder.joblib ] || mv "$TMP/batter_finder.joblib" models/
[ -f kaggle/train-detector/out/detector/weights/best.pt ] || mv "$TMP/detector_best.pt" kaggle/train-detector/out/detector/weights/best.pt
rm -rf "$TMP"

echo "== done. Run the app: =="
echo "   python -m app.server"
