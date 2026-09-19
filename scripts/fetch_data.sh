#!/usr/bin/env bash
# Download CricLens data from the private Kaggle dataset shikkoustic/criclens-all (must be shared with you).
#   scripts/fetch_data.sh            processed data only (~240 MB): pose joints, manifests, models, results
#   scripts/fetch_data.sh --all      everything (~6 GB): also the 480p clips and the detection image sets
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"; cd "$ROOT"
[ -d .venv ] && source .venv/bin/activate
DS=shikkoustic/criclens-all
TMP="$ROOT/data/_download"; rm -rf "$TMP"; mkdir -p "$TMP"
unzip_all() { for z in $(find "$TMP" -name "*.zip"); do unzip -q -o "$z" -d "$(dirname "$z")" && rm "$z"; done; }
if [ "${1:-}" = "--all" ]; then
  kaggle datasets download "$DS" -p "$TMP" --unzip; unzip_all
  mkdir -p data/interim data/processed
  rm -rf data/interim/clips && mv "$TMP/clips" data/interim/clips
  rm -rf data/processed/detection && mv "$TMP/detection" data/processed/detection
  mkdir -p data/raw/cricshot10k/models && mv "$TMP/models/Player_Type_Detection_Model.pt" data/raw/cricshot10k/models/
  TAR="$TMP/processed/criclens-processed.tar.gz"
else
  kaggle datasets download "$DS" -f processed/criclens-processed.tar.gz -p "$TMP"; unzip_all
  TAR=$(find "$TMP" -name "criclens-processed.tar.gz" | head -1)
fi
tar -xzf "$TAR" -C "$ROOT"
rm -rf "$TMP"
echo "done: data/processed, kaggle/chunks/*/out (pose joints), models/batter_finder.joblib, detector weights"
