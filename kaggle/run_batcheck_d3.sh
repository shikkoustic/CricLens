#!/usr/bin/env bash
set -uo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"; cd "$ROOT"; source .venv/bin/activate
kq() { perl -e 'alarm shift; exec @ARGV' 120 kaggle "$@"; }
push() { until out=$(kq kernels push -p "$1" 2>&1) && [[ "$out" == *successfully* ]]; do echo "push $1 waiting: $(echo "$out" | tail -1 | head -c 100)"; sleep 120; done; echo "pushed $1 $(date +%H:%M)"; }
waitk() { while true; do s=$(kq kernels status "shikkoustic/$1" 2>&1 | tail -1); case "$s" in *COMPLETE*|*ERROR*|*CANCEL*) echo "$1: $s"; break;; esac; sleep 45; done; }
fetch() { rm -rf "$2"; mkdir -p "$2"; perl -e 'alarm 1200; exec @ARGV' kaggle kernels output "shikkoustic/$1" -p "$2" -o >/dev/null 2>&1; }
until kq datasets status shikkoustic/criclens-d3 2>&1 | grep -q ready; do sleep 30; done; echo "dataset ready $(date +%H:%M)"
push kaggle/batcheck; push kaggle/d3-smoke-c0
sleep 60; waitk criclens-d3-smoke-c0; push kaggle/d3-smoke-c1
waitk criclens-batcheck; fetch criclens-batcheck kaggle/batcheck/out; echo "batcheck: $(cat kaggle/batcheck/out/summary.json 2>/dev/null)"
waitk criclens-d3-smoke-c1
for k in 0 1; do fetch criclens-d3-smoke-c$k kaggle/d3-smoke-c$k/out
  [ -f kaggle/d3-smoke-c$k/out/summary.json ] || { echo "D3 SMOKE c$k FAILED"; tail -c 3000 kaggle/d3-smoke-c$k/out/*.log; exit 1; }; echo "smoke c$k ok"; done
./kaggle/run_chunks.sh d3 2 0 0 > data/raw/logs/run_d3_c0.log 2>&1 &
sleep 20; ./kaggle/run_chunks.sh d3 2 1 1 > data/raw/logs/run_d3_c1.log 2>&1; wait
echo "D3 finished $(date +%H:%M)"; tail -1 data/raw/logs/run_d3_c0.log | cut -c1-200; tail -1 data/raw/logs/run_d3_c1.log | cut -c1-200
