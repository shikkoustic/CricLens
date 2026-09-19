#!/usr/bin/env bash
# Download CricLens data from the private Kaggle datasets (they must be shared with your Kaggle account).
#   scripts/fetch_data.sh            processed data only (~240 MB): pose joints, manifests, models, results
#   scripts/fetch_data.sh --clips    also the 480p clips (~4 GB) into data/interim/clips/
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"; cd "$ROOT"
[ -d .venv ] && source .venv/bin/activate
TMP="$ROOT/data/_download"; mkdir -p "$TMP"
kaggle datasets download -d shikkoustic/criclens-processed -p "$TMP" --unzip
tar -xzf "$TMP/criclens-processed.tar.gz" -C "$ROOT"
echo "processed data extracted: data/processed, kaggle/chunks/*/out, models/batter_finder.joblib, detector weights"
if [ "${1:-}" = "--clips" ]; then
  kaggle datasets download -d shikkoustic/criclens-clips -p "$TMP/clips" --unzip
  mkdir -p data/interim && rm -rf data/interim/clips && mv "$TMP/clips/clips" data/interim/clips
  echo "clips in data/interim/clips/<source>/"
fi
rm -rf "$TMP"
