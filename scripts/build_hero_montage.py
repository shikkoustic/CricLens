"""Build the home page's hero video: a short crossfaded montage of real strokes from the
dataset, not one static clip.

The clips are broadcast-derived (academic, non-commercial use only -- CLAUDE.md), so the output
goes to data/ (gitignored, never committed) and is served locally at runtime by app/server.py's
/api/hero.mp4, the same pattern every other dataset clip in the app already uses (see
/api/samples/{id}/video). Nothing here is redistributed; it only ever runs on a machine that
already has the dataset.

The five clips below were hand-picked (not re-derived from a filter each run) for clean broadcast
quality and a clearly-readable shot across five different strokes -- a judgement call, so it's a
fixed list rather than a query. Re-run after `scripts/fetch_data.sh` or whenever you want a
different selection; edit CLIPS to swap any of them out.

    python scripts/build_hero_montage.py
"""
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "data/interim/hero_montage.mp4"

# (path under data/interim/clips/, the stroke it shows)
CLIPS = [
    ("cricshot10/cs10_cover_cricketshot-train-cover-cover-0012.mp4", "drive"),
    ("cricshot10/cs10_square-cut_cricketshot-train-square-cut-square-cut-0096.mp4", "cut"),
    ("cricshot10/cs10_pull_cricketshot-train-pull-pull-0052.mp4", "pull_hook"),
    ("cricshot10/cs10_sweep_cricketshot-train-sweep-sweep-0069.mp4", "sweep"),
    ("cricshot10k/cs10k_lofted-legside_lofted-legside-vid375-54.mp4", "lofted"),
]
W, H = 854, 480
SEG = 1.8   # seconds taken from the start of each clip
XF = 0.35   # crossfade duration between consecutive clips


def main() -> None:
    missing = [c for c, _ in CLIPS if not (ROOT / "data/interim/clips" / c).exists()]
    if missing:
        raise SystemExit(f"missing clip(s), run scripts/fetch_data.sh first: {missing}")

    inputs = []
    for rel, _ in CLIPS:
        inputs += ["-i", str(ROOT / "data/interim/clips" / rel)]

    filters, labels = [], []
    for i in range(len(CLIPS)):
        lbl = f"v{i}"
        filters.append(
            f"[{i}:v]trim=0:{SEG},setpts=PTS-STARTPTS,"
            f"scale={W}:{H}:force_original_aspect_ratio=increase,crop={W}:{H},"
            f"setsar=1,fps=25[{lbl}]"
        )
        labels.append(lbl)

    # chain crossfades: v0 x v1 -> c1, c1 x v2 -> c2, ...
    offset = SEG - XF
    chain = labels[0]
    for i in range(1, len(CLIPS)):
        out_lbl = f"c{i}"
        filters.append(f"[{chain}][{labels[i]}]xfade=transition=fade:duration={XF}:offset={offset}[{out_lbl}]")
        chain = out_lbl
        offset += SEG - XF

    OUT.parent.mkdir(parents=True, exist_ok=True)
    cmd = [
        "ffmpeg", "-y", "-v", "error", *inputs,
        "-filter_complex", ";".join(filters),
        "-map", f"[{chain}]",
        "-an", "-c:v", "libx264", "-crf", "20", "-preset", "slow", "-pix_fmt", "yuv420p",
        "-movflags", "+faststart", str(OUT),
    ]
    subprocess.run(cmd, check=True)
    print(f"wrote {OUT} ({len(CLIPS)} clips, ~{len(CLIPS) * (SEG - XF) + XF:.1f}s)")


if __name__ == "__main__":
    main()
