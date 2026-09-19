"""Ingest pre-trimmed shot-clip datasets into the standard 480p format.

For each source: extract archives to a temp dir, find every video, read its label from
the folder (or archive) name, transcode to data/interim/clips/<source>/, then delete the
extracted originals (the downloaded archive is kept). Writes one row per clip to
data/processed/<source>_clips.parquet, including clips that failed to decode.

    python datasets/ingest_clips.py cricshot10k cricshot10 ...
"""
from __future__ import annotations

import re
import shutil
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
import video  # noqa: E402
from taxonomy import MAP, unify  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
RAW, INTERIM, PROCESSED = ROOT / "data/raw", ROOT / "data/interim", ROOT / "data/processed"
VIDEO_EXT = {".mp4", ".avi", ".mov", ".mkv", ".webm", ".m4v", ".mpg", ".mpeg"}

SOURCES = {  # where the downloaded archives (or loose folders) live
    "cricshot10k": {"root": RAW / "cricshot10k/archives", "glob": "*.rar", "prefix": "cs10k"},
    "cricshot10": {"root": RAW / "cricshot10", "glob": "*.tar.gz", "prefix": "cs10"},
    "ipl2023": {"root": RAW / "ipl2023", "glob": "*.zip", "prefix": "ipl"},
    "kucricshot": {"root": RAW / "kucricshot", "glob": None, "prefix": "ku"},
    "amittalmale": {"root": RAW / "amittalmale", "glob": "*.zip", "prefix": "amt"},
}


def _norm(s: str) -> str:
    return re.sub(r"[^a-z0-9]", "", s.lower())


def label_for(source: str, path: Path, fallback: str | None) -> str | None:
    keys = {_norm(k): k for k in MAP[source]}
    for part in reversed(path.parts[:-1]):
        if _norm(part) in keys:
            return keys[_norm(part)]
    if fallback and _norm(fallback) in keys:
        return keys[_norm(fallback)]
    stem = _norm(path.stem)  # e.g. ipl2023 "10_gt_dc_cut_1"
    hits = [k for n, k in keys.items() if n and n in stem]
    return max(hits, key=len) if hits else None


def group_for(source: str, path: Path) -> str | None:
    """Source-match id, used to keep clips from one match inside one split."""
    m = re.match(r"^(?:vid)?(\d+)[_-]", path.stem)  # cricshot10k "vid100_41", ipl2023 "10_gt_dc_cut_1"
    if source in ("cricshot10k", "ipl2023") and m:
        return f"{source}:{int(m.group(1))}"
    if source == "amittalmale":  # "<highlights video title>_<start>_<end>.avi"
        return f"{source}:{re.sub(r'_\d+_\d+$', '', path.stem)}"
    return None


def _slug(s: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", s.lower()).strip("-")


def _one(source: str, prefix: str, src: Path, rel: Path, fallback: str | None) -> dict:
    label = label_for(source, rel, fallback)
    clip_id = f"{prefix}_{_slug(label or 'unlabelled')}_{_slug(rel.with_suffix('').as_posix())[-60:]}"
    dst = INTERIM / "clips" / source / f"{clip_id}.mp4"
    row = {"clip_id": clip_id, "source": source, "src_relpath": rel.as_posix(), "label_orig": label,
           "group": group_for(source, rel), "path": str(dst.relative_to(ROOT))}
    row["shot"], row["side"] = unify(source, label) if label else ("other", "unknown")
    try:
        sp = video.probe(src)
        row.update({f"src_{k}": v for k, v in sp.items()})
        if not dst.exists():
            video.transcode(src, dst, size=video.target_size(sp["width"], sp["height"]))
        row.update(video.probe(dst))
        row["ok"] = True
    except Exception as e:  # corrupt/undecodable source: keep the row, flag it
        row["ok"], row["error"] = False, repr(e)[:300]
    return row


def ingest(source: str, workers: int = 6) -> pd.DataFrame:
    cfg = SOURCES[source]
    root, prefix = cfg["root"], cfg["prefix"]
    units = sorted(root.glob(cfg["glob"])) if cfg["glob"] else [root]
    table = PROCESSED / f"{source}_clips.parquet"
    prev = pd.read_parquet(table) if table.exists() and "unit" in pd.read_parquet(table, columns=None).columns else None
    done_units = set(prev["unit"]) if prev is not None else set()
    rows = prev.to_dict("records") if prev is not None else []
    for unit in units:
        if unit.name in done_units:  # archive already ingested in an earlier run
            continue
        if unit.is_file():
            tmp = INTERIM / "_extract" / source / unit.name.split(".")[0]
            shutil.rmtree(tmp, ignore_errors=True)
            tmp.mkdir(parents=True)
            subprocess.run(["bsdtar", "-xf", str(unit), "-C", str(tmp)], check=True)
            base, fallback = tmp, unit.name.split(".")[0]
        else:
            base, tmp, fallback = unit, None, None
        vids = [p for p in base.rglob("*") if p.suffix.lower() in VIDEO_EXT and not p.name.startswith("._")]
        with ThreadPoolExecutor(workers) as ex:
            new = list(ex.map(lambda p: _one(source, prefix, p, p.relative_to(base), fallback), vids))
        for r in new:
            r["unit"] = unit.name
        rows += new
        if tmp is not None:
            shutil.rmtree(tmp, ignore_errors=True)
        ok = sum(r["ok"] for r in rows)
        print(f"{source}: {unit.name}: {len(vids)} videos (running total {len(rows)}, ok {ok})", flush=True)
        PROCESSED.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(rows).to_parquet(table, index=False)  # checkpoint after every archive
    return pd.DataFrame(rows)


if __name__ == "__main__":
    for s in sys.argv[1:]:
        d = ingest(s)
        print(d.groupby(["label_orig"], dropna=False).size().to_string())
