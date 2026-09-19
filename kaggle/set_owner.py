"""Point every Kaggle job at a different Kaggle account (kernel ids only; dataset sources are left alone,
because the private datasets stay on the original account and are shared with collaborators).

    python kaggle/set_owner.py <your-kaggle-username>
Then regenerate chunk folders, e.g.  python kaggle/make_chunks.py pose-rest 10
"""
import json, sys
from pathlib import Path

user = sys.argv[1]
n = 0
for meta in Path(__file__).parent.rglob("kernel-metadata.json"):
    if "/out/" in str(meta):
        continue
    m = json.loads(meta.read_text())
    m["id"] = f"{user}/{m['id'].split('/', 1)[1]}"
    meta.write_text(json.dumps(m, indent=2)); n += 1
print(f"{n} kernel-metadata.json files now owned by {user}")
