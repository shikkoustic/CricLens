"""Point every Kaggle job at a different Kaggle account.

Kernel ids get the new owner. Dataset sources that live in the combined private dataset
(shikkoustic/criclens-all, shared with collaborators) are switched to it, so jobs work with one share:
criclens-clips, criclens-pilot and criclens-detection -> criclens-all. Jobs that need processed data
(pose joints, batter finder, stumps detections) should extract criclens-processed.tgz.bin from it.

    python kaggle/set_owner.py <your-kaggle-username>
Then regenerate chunk folders, e.g.  python kaggle/make_chunks.py pose-rest 10
"""
import json, sys
from pathlib import Path

user = sys.argv[1]
MERGED = {"shikkoustic/criclens-clips", "shikkoustic/criclens-pilot", "shikkoustic/criclens-detection"}
n = 0
for meta in Path(__file__).parent.rglob("kernel-metadata.json"):
    if "/out/" in str(meta):
        continue
    m = json.loads(meta.read_text())
    m["id"] = f"{user}/{m['id'].split('/', 1)[1]}"
    src = []
    for s in m.get("dataset_sources", []):
        s = "shikkoustic/criclens-all" if s in MERGED else s
        if s not in src:
            src.append(s)
    m["dataset_sources"] = src
    meta.write_text(json.dumps(m, indent=2)); n += 1
print(f"{n} kernel-metadata.json files now owned by {user}, sources switched to shikkoustic/criclens-all")
