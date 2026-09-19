"""ffprobe/ffmpeg helpers: probe clips and write the standard training format.

Standard format: H.264 (yuv420p), short side 480 px (never upscaled), native fps kept,
audio stripped (audio is out of scope), faststart for web playback.
"""
from __future__ import annotations

import json
import subprocess
from pathlib import Path

SHORT_SIDE = 480


def probe(path: str | Path) -> dict:
    out = subprocess.run(
        ["ffprobe", "-v", "error", "-select_streams", "v:0",
         "-show_entries", "stream=width,height,r_frame_rate,avg_frame_rate,nb_frames:format=duration",
         "-of", "json", str(path)],
        capture_output=True, text=True, check=True,
    ).stdout
    d = json.loads(out)
    s = d["streams"][0]
    num, den = (s.get("avg_frame_rate") or s["r_frame_rate"]).split("/")
    fps = float(num) / float(den) if float(den) else 0.0
    dur = float(d.get("format", {}).get("duration") or 0)
    nb = s.get("nb_frames")
    return {
        "width": int(s["width"]), "height": int(s["height"]), "fps": round(fps, 3),
        "duration": round(dur, 3), "n_frames": int(nb) if nb and nb.isdigit() else round(dur * fps),
    }


def target_size(w: int, h: int, short: int = SHORT_SIDE) -> tuple[int, int]:
    s = min(w, h)
    if s <= short:
        return w - w % 2, h - h % 2
    k = short / s
    return int(round(w * k / 2) * 2), int(round(h * k / 2) * 2)


def transcode(src: str | Path, dst: str | Path, start: float | None = None, end: float | None = None,
              size: tuple[int, int] | None = None, crf: int = 23) -> None:
    """Re-encode src (optionally the [start, end] window, in seconds) to the standard format."""
    dst = Path(dst)
    dst.parent.mkdir(parents=True, exist_ok=True)
    if size is None:
        p = probe(src)
        size = target_size(p["width"], p["height"])
    cmd = ["ffmpeg", "-nostdin", "-loglevel", "error", "-y"]
    if start is not None:
        cmd += ["-ss", f"{start:.3f}"]  # before -i + re-encode = fast and frame-accurate
    cmd += ["-i", str(src)]
    if end is not None:
        cmd += ["-t", f"{(end - (start or 0)):.3f}"]
    cmd += ["-map", "0:v:0", "-an", "-vf", f"scale={size[0]}:{size[1]}:flags=bicubic",
            "-c:v", "libx264", "-preset", "veryfast", "-crf", str(crf), "-pix_fmt", "yuv420p",
            "-movflags", "+faststart", str(dst.with_suffix(".tmp.mp4"))]
    subprocess.run(cmd, check=True)
    dst.with_suffix(".tmp.mp4").rename(dst)
