"""Pitch calibration at full dataset scale (CPU only, no GPU quota): pixels -> real-world cm, for stride
length and swing speed. Method and rationale: models/pitch_calibration.py's docstring (this script
re-implements the same functions self-contained, since Kaggle kernels only see the pushed file + input
datasets, not the repo). Validated locally first (docs/iva/iva_results.md D4) on a hand-picked sample
before this full run, per CLAUDE.md's rule to check outputs before trusting a run at scale.
Output: pitch_calib_<chunk>.csv (one row per clip with kps_exists) and overlay frames for a QA sample.
"""
import glob, json, os, subprocess, time
from concurrent.futures import ProcessPoolExecutor

import cv2
import numpy as np
import pandas as pd

CHUNK = int(os.environ.get("CRICLENS_CHUNK", "0"))
NCHUNKS = int(os.environ.get("CRICLENS_NCHUNKS", "1"))
LIMIT = int(os.environ.get("CRICLENS_LIMIT", "0"))
OUT = "/kaggle/working"
os.makedirs(f"{OUT}/overlays", exist_ok=True)

# criclens-processed.tgz.bin holds data/processed/ (pose_index.parquet) and the per-chunk kps/ + stumps
# detections; the dataset only carries it packed (CLAUDE.md), so every job that needs processed data
# extracts it first (same convention as kaggle/pose-pad/pose_pad.py)
PROC = "/tmp/proc"
_tgz = glob.glob("/kaggle/input/**/criclens-processed.tgz.bin", recursive=True)[0]
os.makedirs(PROC, exist_ok=True)
subprocess.run(["tar", "-xzf", _tgz, "-C", PROC, "--wildcards",
                 "data/processed/*", "kaggle/chunks/pose-*/out/kps/*", "kaggle/chunks/stumps-c0/out/*"],
                check=True)
POSE_IDX = f"{PROC}/data/processed/pose_index.parquet"
STUMPS_DETS = f"{PROC}/kaggle/chunks/stumps-c0/out/stumps_dets.csv"
INPUT_ROOT = PROC
CLIPS = [c for c in glob.glob("/kaggle/input/**/clips", recursive=True) if os.path.isdir(f"{c}/cricketvision")][0]

STUMPS_HEIGHT_CM = 71.1
RETURN_CREASE_SEP_CM = 264.0
_LO1, _HI1 = np.array([5, 15, 60]), np.array([45, 110, 240])
_LO2, _HI2 = np.array([0, 0, 90]), np.array([40, 35, 220])
_WHITE_LO, _WHITE_HI = np.array([0, 0, 195]), np.array([179, 30, 255])
L_ANKLE, R_ANKLE, L_WRIST, R_WRIST = 15, 16, 9, 10


def segment_pitch(bgr):
    h, w = bgr.shape[:2]
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, _LO1, _HI1) | cv2.inRange(hsv, _LO2, _HI2)
    mask[: int(h * 0.06)] = 0
    mask[int(h * 0.86):] = 0
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, k, iterations=2)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, k, iterations=1)
    n, lbl, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    if n <= 1:
        return np.zeros_like(mask)
    bc_label = lbl[h - 1, w // 2]
    best = int(bc_label) if bc_label != 0 else 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
    return np.where(lbl == best, 255, 0).astype(np.uint8)


def detect_crease_lines(bgr, pitch_mask):
    h = bgr.shape[0]
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    white = cv2.inRange(hsv, _WHITE_LO, _WHITE_HI)
    roi = cv2.dilate(pitch_mask, np.ones((25, 25), np.uint8))
    white = cv2.bitwise_and(white, white, mask=roi)
    white[: int(h * 0.06)] = 0
    white[int(h * 0.86):] = 0
    contours, _ = cv2.findContours(white, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    keep = np.zeros_like(white)
    for c in contours:
        if cv2.contourArea(c) < 15:
            continue
        (_, _), (rw, rh), _ = cv2.minAreaRect(c)
        short, long_ = min(rw, rh), max(rw, rh)
        if short < 1 or long_ / max(short, 1e-3) < 4 or short > 14 or long_ < 20:
            continue
        cv2.drawContours(keep, [c], -1, 255, -1)
    lines = cv2.HoughLinesP(keep, 1, np.pi / 180, threshold=20, minLineLength=20, maxLineGap=10)
    return lines.reshape(-1, 4) if lines is not None else None


def _line_angle(seg):
    x0, y0, x1, y1 = seg
    return float(np.degrees(np.arctan2(y1 - y0, x1 - x0))) % 180


def crease_pair_scale(lines):
    if lines is None or len(lines) < 2:
        return None
    best = None
    for i in range(len(lines)):
        for j in range(i + 1, len(lines)):
            a, b = lines[i], lines[j]
            da = abs(_line_angle(a) - _line_angle(b)); da = min(da, 180 - da)
            if da > 15:
                continue
            ca = np.array([(a[0] + a[2]) / 2, (a[1] + a[3]) / 2])
            cb = np.array([(b[0] + b[2]) / 2, (b[1] + b[3]) / 2])
            dx, dy = abs(ca[0] - cb[0]), abs(ca[1] - cb[1])
            sep_px = float(np.linalg.norm(ca - cb))
            if not (20 <= sep_px <= 400) or dx < 2 * dy:
                continue
            if best is None or sep_px > best:
                best = sep_px
    return RETURN_CREASE_SEP_CM / best if best else None


def stumps_scale(stumps_boxes):
    if stumps_boxes.empty:
        return None
    b = stumps_boxes.loc[stumps_boxes.y1.idxmax()]
    h_px = float(b.y1 - b.y0)
    return STUMPS_HEIGHT_CM / h_px if h_px > 5 else None


def calibrate_clip(frames, stumps_boxes):
    stumps_est = stumps_scale(stumps_boxes)
    for i, frame in enumerate(frames):
        mask = segment_pitch(frame)
        lines = detect_crease_lines(frame, mask)
        crease_est = crease_pair_scale(lines)
        if crease_est is not None:
            if stumps_est is None or abs(crease_est - stumps_est) / stumps_est <= 0.4:
                return crease_est, "crease", i, mask, lines
    if stumps_est is not None:
        return stumps_est, "stumps", None, None, None
    return None, "none", None, None, None


def measure_stride_and_swing(kps, window, fps, cm_per_px):
    lo, _, hi = (int(v) for v in window)
    seg = kps[lo:hi + 1]
    out = {"stride_cm": np.nan, "swing_speed_mps": np.nan}
    disp = {}
    for ank in (L_ANKLE, R_ANKLE):
        p0, p1 = seg[0, ank, :2], seg[-1, ank, :2]
        if np.all(np.isfinite(p0)) and np.all(np.isfinite(p1)):
            disp[ank] = float(np.linalg.norm(p1 - p0))
    if disp:
        out["stride_cm"] = max(disp.values()) * cm_per_px
    speeds = []
    for wr in (L_WRIST, R_WRIST):
        pts = seg[:, wr, :2]
        for t in range(len(pts) - 1):
            if np.all(np.isfinite(pts[t])) and np.all(np.isfinite(pts[t + 1])):
                d_px = float(np.linalg.norm(pts[t + 1] - pts[t]))
                speeds.append(d_px * fps * cm_per_px / 100.0)
    if speeds:
        out["swing_speed_mps"] = max(speeds)
    return out


def read_sample_frames(video_path, frame_idxs):
    cap = cv2.VideoCapture(video_path)
    frames, i, want = [], 0, sorted(set(frame_idxs))
    while want:
        ok, f = cap.read()
        if not ok:
            break
        if i == want[0]:
            frames.append(f)
            want.pop(0)
        i += 1
    cap.release()
    return frames


def process_clip(args):
    clip_id, source, kps_path, fps, stumps_boxes, save_overlay = args
    video_path = f"{CLIPS}/{source}/{clip_id}.mp4"
    sample_idxs = sorted(stumps_boxes.frame.unique().tolist()) if len(stumps_boxes) else [0, 4, 8, 12]
    frames = read_sample_frames(video_path, sample_idxs)
    stumps_only = stumps_boxes[stumps_boxes.cls == "stumps"]
    cm_per_px, method, frame_i, mask, lines = calibrate_clip(frames, stumps_only)

    result = {"clip_id": clip_id, "source": source, "cm_per_px": cm_per_px, "calib_method": method}
    if cm_per_px is not None:
        try:
            z = np.load(kps_path)
            m = measure_stride_and_swing(z["kps"], z["window"], fps, cm_per_px)
        except Exception:
            m = {"stride_cm": np.nan, "swing_speed_mps": np.nan}
        if not (10 <= m["stride_cm"] <= 250 or np.isnan(m["stride_cm"])):
            m["stride_cm"] = np.nan
        if not (m["swing_speed_mps"] <= 40 or np.isnan(m["swing_speed_mps"])):
            m["swing_speed_mps"] = np.nan
        if np.isnan(m["stride_cm"]) and np.isnan(m["swing_speed_mps"]):
            method = f"{method}(implausible)"
        result["calib_method"] = method
        result.update(m)
    else:
        result.update({"stride_cm": np.nan, "swing_speed_mps": np.nan})

    if save_overlay and frames:
        f = frames[frame_i] if frame_i is not None else frames[0]
        if mask is None:
            mask = segment_pitch(f)
            lines = detect_crease_lines(f, mask)
        overlay = f.copy()
        overlay[mask > 0] = (0.6 * overlay[mask > 0] + 0.4 * np.array([0, 80, 0])).astype(np.uint8)
        if lines is not None:
            for x0, y0, x1, y1 in lines:
                cv2.line(overlay, (int(x0), int(y0)), (int(x1), int(y1)), (0, 0, 255), 2)
        cv2.imwrite(f"{OUT}/overlays/{clip_id[:50]}.jpg", overlay)
    return result


if __name__ == "__main__":
    idx = pd.read_parquet(POSE_IDX)
    idx = idx[idx.kps_exists].sort_values("clip_id").iloc[CHUNK::NCHUNKS]
    if LIMIT:
        idx = idx.head(LIMIT)
    stumps = pd.read_csv(STUMPS_DETS)
    stumps_by_clip = {c: g for c, g in stumps.groupby("clip_id")}
    empty = stumps.iloc[0:0]

    jobs = []
    for i, r in enumerate(idx.itertuples()):
        kps_path = f"{INPUT_ROOT}/{r.kps_path}"
        jobs.append((r.clip_id, r.source, kps_path, r.fps,
                     stumps_by_clip.get(r.clip_id, empty), i < 40))
    print(f"chunk {CHUNK}/{NCHUNKS}: {len(jobs)} clips", flush=True)

    t0, rows = time.time(), []
    with ProcessPoolExecutor(max(1, os.cpu_count() - 1)) as ex:
        for i, res in enumerate(ex.map(process_clip, jobs, chunksize=8)):
            rows.append(res)
            if i % 1000 == 0:
                print(f"{i}/{len(jobs)} {i / max(time.time() - t0, 1):.1f} clips/s", flush=True)
    d = pd.DataFrame(rows)
    d.to_csv(f"{OUT}/pitch_calib_{CHUNK}.csv", index=False)

    s = {"chunk": CHUNK, "clips": len(d), "seconds": time.time() - t0,
         "method_counts": d.calib_method.value_counts().to_dict(),
         "n_stride": int(d.stride_cm.notna().sum()), "n_swing": int(d.swing_speed_mps.notna().sum()),
         "stride_median_cm": float(d.stride_cm.median()) if d.stride_cm.notna().any() else None,
         "swing_median_mps": float(d.swing_speed_mps.median()) if d.swing_speed_mps.notna().any() else None}
    json.dump(s, open(f"{OUT}/summary.json", "w"), indent=1)
    print(json.dumps(s, indent=1), flush=True)
