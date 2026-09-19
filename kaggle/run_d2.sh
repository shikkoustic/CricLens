#!/usr/bin/env bash
# D2 pipeline: wait for dataset -> smoke both groups -> if both pass, run the full groups in parallel.
set -uo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"; cd "$ROOT"; source .venv/bin/activate
kq() { perl -e 'alarm shift; exec @ARGV' 120 kaggle "$@"; }
until kq datasets status shikkoustic/criclens-d2 2>&1 | grep -q ready; do sleep 30; done
echo "dataset ready $(date +%H:%M)"
for k in 0 1; do until out=$(kq kernels push -p kaggle/d2-smoke-c$k 2>&1) && [[ "$out" == *successfully* ]]; do echo "push smoke c$k refused, retry"; sleep 120; done; done
sleep 60
for k in 0 1; do
  while true; do s=$(kq kernels status shikkoustic/criclens-d2-smoke-c$k 2>&1 | tail -1); case "$s" in *COMPLETE*|*ERROR*|*CANCEL*) break;; esac; sleep 45; done
  rm -rf kaggle/d2-smoke-c$k/out; mkdir -p kaggle/d2-smoke-c$k/out
  perl -e 'alarm 900; exec @ARGV' kaggle kernels output shikkoustic/criclens-d2-smoke-c$k -p kaggle/d2-smoke-c$k/out -o >/dev/null 2>&1
  if [ ! -f kaggle/d2-smoke-c$k/out/summary.json ]; then echo "SMOKE c$k FAILED ($s)"; tail -c 3000 kaggle/d2-smoke-c$k/out/*.log; exit 1; fi
  echo "smoke c$k ok"
done
./kaggle/run_chunks.sh d2 2 0 0 > data/raw/logs/run_d2_c0.log 2>&1 &
sleep 20
./kaggle/run_chunks.sh d2 2 1 1 > data/raw/logs/run_d2_c1.log 2>&1
wait
echo "D2 finished $(date +%H:%M)"; tail -2 data/raw/logs/run_d2_c0.log | cut -c1-200; tail -2 data/raw/logs/run_d2_c1.log | cut -c1-200
