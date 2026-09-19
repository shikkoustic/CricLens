"""Merge the public cricket detection/segmentation sets into two YOLO datasets.

Unified classes: 0 ball, 1 bat, 2 stumps (people are dropped: the pose model finds the batter).
  det/  every source, boxes (polygons are reduced to their bounding box)  -> ball/bat/stumps detector
  seg/  polygon sources only                                             -> bat/ball segmentation
Roboflow exports contain augmented copies of one frame ("<stem>_jpg.rf.<hash>.jpg"); copies
are grouped by their original stem, and near-identical frames across sources (pHash
distance <= 4) are merged, so a frame and its copies always land in the same split.
Images are hard-linked (no extra disk). Output: data/processed/detection/{det,seg}/.
"""
from __future__ import annotations

import os
import random
import re
import shutil
import subprocess
from collections import Counter, defaultdict
from pathlib import Path

import imagehash
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data/raw/detection"
EXTRACT = ROOT / "data/interim/_extract_det"
OUT = ROOT / "data/processed/detection"
CLASSES = ["ball", "bat", "stumps"]
UNIFY = {"ball": 0, "cricketball": 0, "bat": 1, "cricketbat": 1, "cricketbats": 1, "stump": 2, "stumps": 2}


def _key(name: str) -> str:  # "cricketBall", "cricket-ball", "Cricket Ball" -> "cricketball"
    return re.sub(r"[^a-z]", "", name.lower())
SPLIT = {"train": 0.8, "val": 0.1, "test": 0.1}
SEED = 13
IMG_EXT = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def _sources() -> list[tuple[str, Path]]:
    out = []
    for d in sorted(p for p in RAW.iterdir() if p.is_dir()):
        zips = list(d.glob("*.zip"))
        if zips:  # kaggle_ball and any Roboflow set that arrived zipped
            dst = EXTRACT / d.name
            if not dst.exists():
                dst.mkdir(parents=True)
                for z in zips:
                    subprocess.run(["bsdtar", "-xf", str(z), "-C", str(dst)], check=True)
            out.append((d.name, dst))
        else:
            out.append((d.name, d))
    return out


def _names(src_root: Path) -> list[str] | None:
    for y in list(src_root.rglob("data.yaml")) + list(src_root.rglob("*.yaml")):
        txt = y.read_text()
        m = re.search(r"names:\s*\[(.*?)\]", txt, re.S)
        if m:
            return [n.strip().strip("'\"") for n in m.group(1).split(",")]
        items = re.findall(r"^\s*-\s*(.+?)\s*$", txt.split("names:", 1)[-1], re.M)
        if items:
            return [i.strip("'\"") for i in items]
        items = re.findall(r"^\s*\d+\s*:\s*(.+?)\s*$", txt, re.M)
        if items:
            return [i.strip("'\"") for i in items]
    classes = next(src_root.rglob("classes.txt"), None)
    return classes.read_text().split() if classes else None


def _base_stem(stem: str) -> str:
    return re.sub(r"_(jpg|jpeg|png)\.rf\.[0-9a-f]+$", "", stem, flags=re.I)


def collect() -> list[dict]:
    items = []
    for name, root in _sources():
        names = _names(root)
        if not names:
            print(f"skip {name}: no class names found")
            continue
        remap = {i: UNIFY.get(_key(n)) for i, n in enumerate(names)}
        for img in root.rglob("*"):
            if img.suffix.lower() not in IMG_EXT:
                continue
            lab = Path(str(img.parent).replace("/images", "/labels")) / (img.stem + ".txt")
            det, seg = [], []
            for line in (lab.read_text().splitlines() if lab.exists() else []):
                v = line.split()
                if len(v) < 5 or remap.get(int(float(v[0]))) is None:
                    continue
                c, xs = remap[int(float(v[0]))], list(map(float, v[1:]))
                if len(xs) == 4:
                    det.append((c, *xs))
                else:
                    px, py = xs[0::2], xs[1::2]
                    x0, x1, y0, y1 = min(px), max(px), min(py), max(py)
                    det.append((c, (x0 + x1) / 2, (y0 + y1) / 2, x1 - x0, y1 - y0))
                    seg.append((c, *xs))
            items.append({"source": name, "img": img, "stem": img.stem, "group": f"{name}:{_base_stem(img.stem)}",
                          "det": det, "seg": seg})
        print(f"{name}: {sum(1 for i in items if i['source'] == name)} images, classes {names} -> {remap}")
    return items


def dedupe_groups(items: list[dict]) -> None:
    parent: dict[str, str] = {}

    def find(x: str) -> str:
        parent.setdefault(x, x)
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    buckets: dict[tuple[int, int], list[tuple[int, str]]] = defaultdict(list)
    for it in items:
        try:
            h = int(str(imagehash.phash(Image.open(it["img"]).convert("L"))), 16)
        except Exception:
            it["bad"] = True
            continue
        it["hash"] = h
        for b in range(4):
            key = (b, (h >> (16 * b)) & 0xFFFF)
            for h2, g2 in buckets[key]:
                if bin(h ^ h2).count("1") <= 4:
                    parent[find(it["group"])] = find(g2)
            buckets[key].append((h, it["group"]))
    for it in items:
        it["group"] = find(it["group"])


def write(items: list[dict]) -> None:
    items = [i for i in items if not i.get("bad")]
    groups = sorted({i["group"] for i in items})
    random.Random(SEED).shuffle(groups)
    n = len(groups)
    cut1, cut2 = int(n * SPLIT["train"]), int(n * (SPLIT["train"] + SPLIT["val"]))
    split_of = {g: ("train" if k < cut1 else "val" if k < cut2 else "test") for k, g in enumerate(groups)}
    shutil.rmtree(OUT, ignore_errors=True)
    stats = Counter()
    for kind in ("det", "seg"):
        for it in items:
            rows = it[kind]
            if kind == "seg" and not rows:
                continue
            sp = split_of[it["group"]]
            fname = f"{it['source']}__{it['stem']}"
            (OUT / kind / "images" / sp).mkdir(parents=True, exist_ok=True)
            (OUT / kind / "labels" / sp).mkdir(parents=True, exist_ok=True)
            dst = OUT / kind / "images" / sp / (fname + it["img"].suffix.lower())
            if not dst.exists():
                os.link(it["img"], dst)
            (OUT / kind / "labels" / sp / (fname + ".txt")).write_text(
                "".join(" ".join([str(r[0])] + [f"{x:.6f}" for x in r[1:]]) + "\n" for r in rows))
            stats[(kind, sp, "images")] += 1
            for r in rows:
                stats[(kind, sp, CLASSES[r[0]])] += 1
        (OUT / kind / "data.yaml").write_text(
            f"path: {OUT / kind}\ntrain: images/train\nval: images/val\ntest: images/test\n"
            f"names: {CLASSES}\n")
    lines = ["kind split   images   ball    bat  stumps"]
    for kind in ("det", "seg"):
        for sp in SPLIT:
            lines.append(f"{kind:4} {sp:5} {stats[(kind, sp, 'images')]:7} {stats[(kind, sp, 'ball')]:6} "
                         f"{stats[(kind, sp, 'bat')]:6} {stats[(kind, sp, 'stumps')]:7}")
    lines.append(f"groups: {n} (augmented copies + cross-source near-duplicates merged)")
    (OUT / "report.txt").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    its = collect()
    dedupe_groups(its)
    write(its)
