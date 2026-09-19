#!/usr/bin/env bash
# Resumable chunk runner. Pushes chunks of a job to Kaggle one after another, waits, downloads each
# chunk's output into kaggle/chunks/<job>-c<k>/out, and records progress in kaggle/progress.json.
# Safe to stop at any time: re-running skips finished chunks, downloads chunks that completed while
# we were away (never re-runs them), and re-attaches to chunks still running on Kaggle.
# Several runners can work on different chunk ranges at once (progress writes are locked):
#   kaggle/run_chunks.sh pose-cv 8          # all chunks
#   kaggle/run_chunks.sh pose-cv 8 4 7      # chunks 4..7 only
set -uo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"; source "$ROOT/.venv/bin/activate"
JOB="$1"; N="$2"; FROM="${3:-0}"; TO="${4:-$((N-1))}"; PROG="$ROOT/kaggle/progress.json"
[ -f "$PROG" ] || echo '{}' > "$PROG"
kq() { perl -e 'alarm shift; exec @ARGV' 120 kaggle "$@"; }   # a dropped connection (sleep, Wi-Fi) must never hang a runner
mark() { python3 - "$PROG" "$1" "$2" <<'PY'
import fcntl, json, sys, time
p, key, val = sys.argv[1:]
with open(p + ".lock", "w") as lk:
    fcntl.flock(lk, fcntl.LOCK_EX)
    d = json.load(open(p)); d[key] = {"status": val, "at": time.strftime("%Y-%m-%d %H:%M")}
    json.dump(d, open(p, "w"), indent=1)
PY
}
state() { python3 -c "import json; print(json.load(open('$PROG')).get('$1',{}).get('status',''))"; }
for k in $(seq "$FROM" "$TO"); do
  C="$JOB-c$k"; DIR="$ROOT/kaggle/chunks/$C"
  REF=$(python3 -c "import json,sys; print(json.load(open(sys.argv[1]))['id'])" "$DIR/kernel-metadata.json")
  [ "$(state "$C")" = "done" ] && { echo "skip $C (done)"; continue; }
  s=$(kq kernels status "$REF" 2>&1 | tail -1)
  case "$s" in
    *RUNNING*|*QUEUED*) echo "re-attaching to $C (still running on Kaggle)";;
    *COMPLETE*) echo "$C already complete on Kaggle, downloading";;
    *) until out=$(kq kernels push -p "$DIR" 2>&1) && [[ "$out" == *successfully* ]]; do
         echo "push of $C refused ($(echo "$out" | tail -1 | head -c 160)); retrying in 5 min"; sleep 300; done
       echo "$out" | tail -1;;
  esac
  mark "$C" running; sleep 30
  while true; do
    s=$(kq kernels status "$REF" 2>&1 | tail -1)
    case "$s" in *COMPLETE*|*ERROR*|*CANCEL*) break;; esac; sleep 60
  done
  for try in 1 2 3; do  # downloads can be cut off (sleep, Wi-Fi): retry until the summary arrives
    rm -rf "$DIR/out"; mkdir -p "$DIR/out"
    perl -e 'alarm shift; exec @ARGV' 1800 kaggle kernels output "$REF" -p "$DIR/out" -o >/dev/null 2>&1
    if [ -f "$DIR/out/summary.json" ]; then
      [ -f "$DIR/out/kps.tar" ] && tar -xf "$DIR/out/kps.tar" -C "$DIR/out" && rm "$DIR/out/kps.tar"
      break
    fi
    echo "download of $C incomplete (try $try), retrying in 60 s"; sleep 60
  done
  if [[ "$s" == *COMPLETE* ]] && [ -f "$DIR/out/summary.json" ]; then mark "$C" done; echo "$C done: $(tr -d '\n ' < "$DIR/out/summary.json" | head -c 300)"
  else mark "$C" failed; echo "$C FAILED ($s) - see $DIR/out/*.log"; exit 1; fi
done
echo "chunks $FROM..$TO of $JOB done"
