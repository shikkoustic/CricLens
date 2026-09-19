"""Build a small pilot pack: ~300 canonical clips (480p) + original-resolution versions from the
kept archives, so the pose pilot can compare resolutions and batter selection on Kaggle."""
import json, os, shutil, subprocess
from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "data/kaggle_upload/criclens-pilot"
PER_SOURCE = {"cricketvision": 80, "cricshot10k": 60, "cricshot10": 40, "ipl2023": 40, "kucricshot": 40, "amittalmale": 40}
ARCH = {"cricshot10k": ROOT / "data/raw/cricshot10k/archives", "cricshot10": ROOT / "data/raw/cricshot10",
        "ipl2023": ROOT / "data/raw/ipl2023", "amittalmale": ROOT / "data/raw/amittalmale"}

m = pd.read_parquet(ROOT / "data/processed/manifest.parquet")
c = m[m["is_canonical"] & (m["shot"] != "other")]
pick = pd.concat([  # per source: shuffle, then an even number of clips per shot type
    g.sample(frac=1, random_state=5).groupby("shot").head(max(1, n // g["shot"].nunique())).head(n)
    for src, g in c.groupby("source") for n in [PER_SOURCE[src]]])
shutil.rmtree(OUT, ignore_errors=True)
(OUT / "clips480").mkdir(parents=True); (OUT / "orig").mkdir(); (OUT / "models").mkdir()
rows = []
for r in pick.itertuples():
    os.link(ROOT / r.path, OUT / "clips480" / f"{r.clip_id}.mp4")
    orig = None
    if r.source == "kucricshot":
        src = ROOT / "data/raw/kucricshot" / r.src_relpath
        orig = OUT / "orig" / f"{r.clip_id}{src.suffix}"; os.link(src, orig)
    elif r.source in ARCH:
        arch = ARCH[r.source] / r.unit
        tmp = OUT / "_x"; tmp.mkdir(exist_ok=True)
        subprocess.run(["bsdtar", "-xf", str(arch), "-C", str(tmp), r.src_relpath], check=True)
        src = tmp / r.src_relpath
        orig = OUT / "orig" / f"{r.clip_id}{src.suffix}"; shutil.move(src, orig)
    row = {"clip_id": r.clip_id, "source": r.source, "shot": r.shot, "fps": r.fps,
           "orig": orig.name if orig else None}
    for ph in ("buildup", "execution", "followthrough"):  # CricketVision batter boxes (source px, 1280x720)
        v = getattr(r, f"{ph}_bbox", None)
        row[f"{ph}_bbox"] = json.dumps(list(v)) if v is not None and not (isinstance(v, float)) else None
    for k in ("buildup_start", "clip_start", "execution_start", "execution_end", "handedness"):
        row[k] = getattr(r, k, None)
    rows.append(row)
shutil.rmtree(OUT / "_x", ignore_errors=True)
pd.DataFrame(rows).to_csv(OUT / "pilot.csv", index=False)
shutil.copy(ROOT / "data/raw/cricshot10k/models/Player_Type_Detection_Model.pt", OUT / "models/")
json.dump({"title": "criclens-pilot", "id": "shikkoustic/criclens-pilot", "licenses": [{"name": "other"}]},
          open(OUT / "dataset-metadata.json", "w"))
print(pd.DataFrame(rows).groupby("source").agg(n=("clip_id", "size"), with_orig=("orig", "count")).to_string())
