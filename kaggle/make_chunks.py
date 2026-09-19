"""Create one Kaggle kernel folder per chunk of a job, so each ~30 min chunk saves its own output.

    python kaggle/make_chunks.py pose-cv 4    -> kaggle/chunks/pose-cv-c0 ... pose-cv-c3
"""
import json, sys
from pathlib import Path

HERE = Path(__file__).parent
job, n = sys.argv[1], int(sys.argv[2])
tpl = HERE / job
meta = json.loads((tpl / "kernel-metadata.json").read_text())
code = (tpl / meta["code_file"]).read_text()
for k in range(n):
    d = HERE / "chunks" / f"{job}-c{k}"
    d.mkdir(parents=True, exist_ok=True)
    (d / meta["code_file"]).write_text(
        f'import os; os.environ["CRICLENS_CHUNK"] = "{k}"; os.environ["CRICLENS_NCHUNKS"] = "{n}"\n' + code)
    m = dict(meta, id=f"{meta['id']}-c{k}", title=f"{meta['title']}-c{k}")
    (d / "kernel-metadata.json").write_text(json.dumps(m, indent=2))
    print("wrote", d)
