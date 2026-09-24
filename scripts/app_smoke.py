#!/usr/bin/env python3
"""Run the sample clips through a running app server and report what broke.

    python -m app.server &                  # or CRICLENS_LLM=off for a faster run
    python scripts/app_smoke.py             # all samples
    python scripts/app_smoke.py -n 12       # first 12

Unlike models/evaluate.py, which scores the trained models on saved joints, this exercises the
whole serving path on raw clips: decode, batter finder, pose, window, classifier, scorer,
calibration, render. It reports shot accuracy against the manifest label, but the point is the
failure modes around it -- wrong batter, truncated windows, crashes -- which only appear here.
Writes per-clip rows to app/runtime/smoke.json for follow-up.
"""
import argparse
import json
import statistics
import sys
import time
import urllib.error
import urllib.request
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "app/runtime/smoke.json"
FOLLOW_THROUGH_S = 0.6  # the window the models were trained on: contact -0.8s to +0.6s
FINDER_MIN = 0.9  # train_ready used this confidence floor for "this really is the striker"


def api(base, path, payload=None, timeout=600):
    url = f"{base}{path}"
    data = json.dumps(payload).encode() if payload is not None else None
    req = urllib.request.Request(url, data=data, headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.load(r)


def run_one(base, sample):
    job = api(base, "/api/jobs/sample", {"sample_id": sample["id"]})["job_id"]
    while True:
        d = api(base, f"/api/jobs/{job}")
        if d["status"] in ("done", "error"):
            break
        time.sleep(2)
    row = {"id": sample["id"], "source": sample["source"], "truth": sample["label"],
           "job": job, "status": d["status"]}
    if d["status"] == "error":
        row["error"] = d.get("error")
        return row
    r = d["result"]
    b, w, s, t, m = r["batter"], r["batter"]["window"], r["shot"], r["technique"], r["metrics"]
    row.update(
        pred=s["label"], p=round(s["confidence"], 3), finder_p=round(b["finder_p"], 3),
        pose_conf=round(b["pose_confidence"], 3), found=round(b["found_in_window"], 3),
        start_s=w["start_s"], contact_s=w["contact_s"], end_s=w["end_s"],
        after_contact_s=round(w["end_s"] - w["contact_s"], 3),
        seconds=r["video"]["seconds"], truncated=r["video"]["truncated"],
        calibrated=m["calibrated"], technique=round(t["overall"], 2),
        warnings=r["warnings"], coach_source=r["coach"]["source"],
    )
    return row


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("-n", type=int, default=0, help="only the first N samples")
    ap.add_argument("--base", default="http://127.0.0.1:8000")
    a = ap.parse_args()

    try:
        health = api(a.base, "/api/health")
    except urllib.error.URLError as e:
        sys.exit(f"no server at {a.base} ({e}); start it with: python -m app.server")
    while not health["ready"]:
        if health["error"]:
            sys.exit(f"server failed to load models: {health['error']}")
        print(f"waiting: {health['stage']}", flush=True)
        time.sleep(5)
        health = api(a.base, "/api/health")

    samples = api(a.base, "/api/samples")
    if a.n:
        samples = samples[:a.n]
    print(f"{len(samples)} samples, llm={health['llm']}\n", flush=True)

    rows = []
    for i, s in enumerate(samples, 1):
        t0 = time.time()
        try:
            row = run_one(a.base, s)
        except Exception as e:  # a hung or crashed job must not lose the rows already collected
            row = {"id": s["id"], "source": s["source"], "truth": s["label"],
                   "status": "client_error", "error": repr(e)}
        row["secs"] = round(time.time() - t0, 1)
        rows.append(row)
        ok = "ok " if row.get("pred") == row.get("truth") else "MISS" if row["status"] == "done" else "ERR "
        print(f"[{i:>3}/{len(samples)}] {ok} {row['source']:<14} truth={str(row.get('truth')):<12} "
              f"pred={str(row.get('pred')):<12} p={row.get('p')} finder_p={row.get('finder_p')} "
              f"after_contact={row.get('after_contact_s')}s {row['secs']}s", flush=True)
        OUT.parent.mkdir(parents=True, exist_ok=True)
        OUT.write_text(json.dumps(rows, indent=1))

    done = [r for r in rows if r["status"] == "done"]
    errs = [r for r in rows if r["status"] != "done"]
    print(f"\n{'='*70}\nran {len(rows)}: {len(done)} done, {len(errs)} failed")
    for r in errs:
        print(f"  FAIL {r['source']}/{r['id'][:40]}: {r.get('error')}")
    if not done:
        return
    hit = [r for r in done if r["pred"] == r["truth"]]
    print(f"\nshot accuracy: {len(hit)}/{len(done)} = {len(hit)/len(done):.1%}")

    low = [r for r in done if r["finder_p"] < FINDER_MIN]
    print(f"batter finder below {FINDER_MIN}: {len(low)}/{len(done)} = {len(low)/len(done):.1%}")
    for group, name in ((low, f"finder_p<{FINDER_MIN}"), ([r for r in done if r["finder_p"] >= FINDER_MIN], "confident")):
        if group:
            acc = sum(r["pred"] == r["truth"] for r in group) / len(group)
            print(f"    {name:<16} n={len(group):<3} shot accuracy {acc:.1%}")

    short = [r for r in done if r["after_contact_s"] < FOLLOW_THROUGH_S]
    print(f"\nwindow ends <{FOLLOW_THROUGH_S}s after contact: {len(short)}/{len(done)} = {len(short)/len(done):.1%}")
    if short:
        acc = sum(r["pred"] == r["truth"] for r in short) / len(short)
        print(f"    their shot accuracy {acc:.1%}; median follow-through "
              f"{statistics.median(r['after_contact_s'] for r in short):.2f}s")
    print(f"clips carrying no warning despite one of the above: "
          f"{sum(1 for r in low + short if not r['warnings'])}")

    print(f"\ncalibrated (stride/swing): {sum(r['calibrated'] for r in done)}/{len(done)}")
    print(f"median runtime {statistics.median(r['secs'] for r in done):.1f}s")
    print(f"predicted-label spread: {dict(Counter(r['pred'] for r in done))}")
    print(f"\nrows -> {OUT}")


if __name__ == "__main__":
    main()
