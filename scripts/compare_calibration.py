#!/usr/bin/env python3
"""Check the app's live pitch calibration against the Kaggle batch job's saved results.

    python scripts/compare_calibration.py app/runtime/smoke/<run>.json

app/pipeline.py calls calibrate_clip() on the frames it decoded, while D4 wrote
data/processed/iva/pitch_calibration.parquet from a separate pass over the same clips. The two
should agree on which clips calibrate at all; if they disagree the app is not measuring what the
reported 20% yield describes, and the stride/swing numbers shown to users are not the ones the
write-up characterises.
"""
import json
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent


def main(rows_path: str) -> None:
    rows = [r for r in json.loads(Path(rows_path).read_text()) if r["status"] == "done"]
    if not rows:
        sys.exit("no completed rows in that file")
    # sample id is "<source>--<clip_id>"; clip_id itself may contain "--", so split once.
    for r in rows:
        r["clip_id"] = r["id"].split("--", 1)[1]

    batch = pd.read_parquet(ROOT / "data/processed/iva/pitch_calibration.parquet")
    batch["batch_ok"] = batch.calib_method.isin(["crease", "stumps"])
    by_clip = batch.set_index("clip_id")[["calib_method", "batch_ok", "stride_cm", "swing_speed_mps"]]

    seen = [r for r in rows if r["clip_id"] in by_clip.index]
    missing = len(rows) - len(seen)
    agree = both = app_only = batch_only = 0
    for r in seen:
        b = by_clip.loc[r["clip_id"]]
        a, bo = bool(r["calibrated"]), bool(b.batch_ok)
        agree += a == bo
        both += a and bo
        app_only += a and not bo
        batch_only += bo and not a

    print(f"{len(seen)} clips matched against the batch results ({missing} not in the parquet)")
    print(f"  agree            {agree}/{len(seen)} = {agree/len(seen):.0%}")
    print(f"  both calibrated  {both}")
    print(f"  app only         {app_only}")
    print(f"  batch only       {batch_only}   <- clips the write-up counts but the app does not")
    print(f"\napp yield   {sum(r['calibrated'] for r in seen)/len(seen):.1%}")
    print(f"batch yield {by_clip.loc[[r['clip_id'] for r in seen], 'batch_ok'].mean():.1%} on the same clips")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else str(ROOT / "app/runtime/smoke.json"))
