"""Pitch calibration: pixels -> real-world cm, for stride length and swing speed (PROGRESS.md IVA step).

Method (docs/iva/syllabus_alignment.md Module 1/2): HSV pitch-strip segmentation -> morphological
clean-up -> connected components (keep the blob under the batter) -> Canny-style white-line
thresholding restricted to that region -> elongation filter (crease markings are thin and long; kit,
pads and gloves are not) -> probabilistic Hough transform for the crease lines. Two detected return-
crease lines give an exact real-world reference (Law 8: return creases are always 2.64 m apart) -> a
local cm-per-pixel scale at the crease.

Crease-line detection is unreliable frame-by-frame on 480p broadcast footage (the marking is a handful
of pixels wide) -- see docs/iva/iva_results.md D4 for the measured yield. `kaggle/chunks/stumps-c0`'s
existing stumps detections (from the ball/bat/stumps YOLO detector, already run for the batter finder)
give a second, independent reference: stumps are always 71.1 cm tall (Law 8), and are a bigger, higher-
contrast target than a painted line. calibrate_clip() tries the crease pair first (the syllabus
technique) and falls back to stumps, recording which one was used so the yield of each is honestly
reported rather than blurred together.

The resulting cm-per-pixel is a LOCAL scale at the crease depth, not a full ground-plane homography: it
assumes the batter's feet and hands stay close to that depth for the duration of one stroke, which holds
for stride length (feet are on the ground throughout) and is a documented approximation for swing speed
(the bat/hands leave the ground plane during the shot -- see the caveat in run_on_clip()'s docstring).

    python models/pitch_calibration.py --clips <clip_id> [<clip_id> ...] --overlay-dir <dir>
"""
import argparse
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
STUMPS_HEIGHT_CM = 71.1  # Law 8: stumps + bails, ground to top of bails
STUMPS_WIDTH_CM = 22.86  # Law 8: outer edge to outer edge of the three stumps
RETURN_CREASE_SEP_CM = 264.0  # Law 8: the two return creases are always 2.64 m apart

_LO1, _HI1 = np.array([5, 15, 60]), np.array([45, 110, 240])   # tan/dried-pitch HSV range
_LO2, _HI2 = np.array([0, 0, 90]), np.array([40, 35, 220])     # low-saturation/dusty worn-pitch range
_WHITE_LO, _WHITE_HI = np.array([0, 0, 195]), np.array([179, 30, 255])  # crease-line white


def segment_pitch(bgr: np.ndarray) -> np.ndarray:
    """HSV threshold -> morphological close/open -> connected components. Returns a binary mask of the
    single pitch blob under the batter (the component touching bottom-centre of the frame -- more robust
    than "largest area", which a wide shot's crowd/stand region can win)."""
    h, w = bgr.shape[:2]
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, _LO1, _HI1) | cv2.inRange(hsv, _LO2, _HI2)
    mask[: int(h * 0.06)] = 0   # fixed broadcaster-logo corner
    mask[int(h * 0.86):] = 0    # fixed scoreboard bar
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, k, iterations=2)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, k, iterations=1)
    n, lbl, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    if n <= 1:
        return np.zeros_like(mask)
    bc_label = lbl[h - 1, w // 2]
    best = int(bc_label) if bc_label != 0 else 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
    return np.where(lbl == best, 255, 0).astype(np.uint8)


def detect_crease_lines(bgr: np.ndarray, pitch_mask: np.ndarray) -> np.ndarray | None:
    """White, thin, elongated marks inside the pitch region -> probabilistic Hough. Returns (N, 4) array
    of [x0, y0, x1, y1] segments, or None."""
    h = bgr.shape[0]
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    white = cv2.inRange(hsv, _WHITE_LO, _WHITE_HI)
    roi = cv2.dilate(pitch_mask, np.ones((25, 25), np.uint8))
    white = cv2.bitwise_and(white, white, mask=roi)
    # re-apply the HUD crop by hand: dilating the pitch ROI toward the frame edge can push it back into
    # the scoreboard bar even though segment_pitch() already zeroed that band -- ticker text there
    # (bright, low-saturation) otherwise passes the same elongation filter as a real crease line
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
            continue  # not thin+long enough to be a crease line (kit/pads/gloves are blobby)
        cv2.drawContours(keep, [c], -1, 255, -1)
    lines = cv2.HoughLinesP(keep, 1, np.pi / 180, threshold=20, minLineLength=20, maxLineGap=10)
    return lines.reshape(-1, 4) if lines is not None else None


def _line_angle(seg: np.ndarray) -> float:
    x0, y0, x1, y1 = seg
    return float(np.degrees(np.arctan2(y1 - y0, x1 - x0))) % 180


def crease_pair_scale(lines: np.ndarray) -> float | None:
    """Two near-parallel, near-VERTICAL segments, laterally offset from each other by a plausible pixel
    distance = the two return creases, 2.64 m apart by law -> cm/px. The camera always looks down the
    pitch length, so a genuine return crease is always closer to vertical than horizontal in the image;
    requiring that up front rejects the dominant false-positive mode -- a horizontal seam (scoreboard-bar
    edge, popping-crease fragments) broken by Hough into two collinear pieces, which are "near-parallel"
    and "laterally offset" by the same tests a real pair would pass, just along the wrong axis. The
    lateral-vs-depth offset check (dx >= 2*dy) then rejects a near-end/far-end mismatch (vertically
    separated by perspective, not by the crease width) among genuinely vertical candidates. None if no
    plausible pair exists."""
    if lines is None or len(lines) < 2:
        return None
    vertical = [ln for ln in lines if abs(_line_angle(ln) - 90) < 40]
    best = None
    for i in range(len(vertical)):
        for j in range(i + 1, len(vertical)):
            a, b = vertical[i], vertical[j]
            da = abs(_line_angle(a) - _line_angle(b))
            da = min(da, 180 - da)
            if da > 15:
                continue
            ca = np.array([(a[0] + a[2]) / 2, (a[1] + a[3]) / 2])
            cb = np.array([(b[0] + b[2]) / 2, (b[1] + b[3]) / 2])
            dx, dy = abs(ca[0] - cb[0]), abs(ca[1] - cb[1])
            sep_px = float(np.linalg.norm(ca - cb))
            if not (20 <= sep_px <= 400) or dx < 2 * dy:  # implausible distance, or mostly vertical offset
                continue
            if best is None or sep_px > best:
                best = sep_px
    return RETURN_CREASE_SEP_CM / best if best else None


def stumps_scale(stumps_boxes: pd.DataFrame) -> float | None:
    """The nearest (largest, lowest-in-frame) stumps detection's box height -> cm/px, using the fixed
    71.1 cm stumps height. A wide shot can show both ends' stumps at very different apparent sizes
    (perspective); the near/striker's end is the one at the same depth as the crease-based scale, so it
    is the one to prefer, not just the highest-confidence detection (models/stump_features.py uses the
    same "highest y1 = nearest" convention)."""
    if stumps_boxes.empty:
        return None
    b = stumps_boxes.loc[stumps_boxes.y1.idxmax()]
    h_px = float(b.y1 - b.y0)
    return STUMPS_HEIGHT_CM / h_px if h_px > 5 else None


def calibrate_clip(frames: list[np.ndarray], stumps_boxes: pd.DataFrame) -> tuple[float | None, str, int | None]:
    """Try every sampled frame for a crease pair (the syllabus technique); cross-check against the
    stumps-based scale where both exist and reject the crease result if they disagree by more than 40%
    (a real pairing and a real stumps box, at the same crease depth, should roughly agree -- a big
    disagreement means one of them locked onto the wrong thing, e.g. a near/far crease-line mismatch).
    Falls back to stumps alone, then reports failure. Returns (cm_per_px, method) with method in
    {"crease", "stumps", "none"}. The third element is the index into `frames` that produced the crease
    result (for overlay/QA), or None."""
    stumps_est = stumps_scale(stumps_boxes)
    for i, frame in enumerate(frames):
        mask = segment_pitch(frame)
        lines = detect_crease_lines(frame, mask)
        crease_est = crease_pair_scale(lines)
        if crease_est is not None:
            if stumps_est is None or abs(crease_est - stumps_est) / stumps_est <= 0.4:
                return crease_est, "crease", i
    if stumps_est is not None:
        return stumps_est, "stumps", None
    return None, "none", None


L_ANKLE, R_ANKLE, L_WRIST, R_WRIST = 15, 16, 9, 10


def measure_stride_and_swing(kps: np.ndarray, window: np.ndarray, fps: float, cm_per_px: float) -> dict:
    """Stride: the ankle with the larger displacement between window start and end (the front/striding
    foot), converted to cm -- geometrically sound, feet stay on the ground plane throughout.

    Swing speed: peak frame-to-frame wrist displacement in the window, converted to m/s via the SAME
    local ground-plane scale. This is an approximation, not a full 3D reconstruction: the hands/bat
    leave the crease depth during the shot (more so for lofted shots), so this reads as "peak hand speed
    at the calibrated depth" rather than a metrologically exact bat speed -- documented here rather than
    silently assumed, per CLAUDE.md's rule against overclaiming an unmeasured technique."""
    lo, _, hi = (int(v) for v in window)
    seg = kps[lo:hi + 1]  # (Wf, 17, 3)
    out = {"stride_cm": np.nan, "swing_speed_mps": np.nan}
    disp = {ank: np.nan for ank in (L_ANKLE, R_ANKLE)}
    for ank in (L_ANKLE, R_ANKLE):
        p0, p1 = seg[0, ank, :2], seg[-1, ank, :2]
        if np.all(np.isfinite(p0)) and np.all(np.isfinite(p1)):
            disp[ank] = float(np.linalg.norm(p1 - p0))
    valid = {k: v for k, v in disp.items() if np.isfinite(v)}
    if valid:
        out["stride_cm"] = max(valid.values()) * cm_per_px

    speeds = []
    for wr in (L_WRIST, R_WRIST):
        pts = seg[:, wr, :2]
        for t in range(len(pts) - 1):
            if np.all(np.isfinite(pts[t])) and np.all(np.isfinite(pts[t + 1])):
                d_px = float(np.linalg.norm(pts[t + 1] - pts[t]))
                speeds.append(d_px * fps * cm_per_px / 100.0)  # px/frame -> px/s -> cm/s -> m/s
    if speeds:
        out["swing_speed_mps"] = max(speeds)
    return out


def read_sample_frames(video_path: Path, frame_idxs: list[int]) -> list[np.ndarray]:
    cap = cv2.VideoCapture(str(video_path))
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


def save_overlay(frame: np.ndarray, mask: np.ndarray, lines: np.ndarray | None, path: Path) -> None:
    overlay = frame.copy()
    overlay[mask > 0] = (0.6 * overlay[mask > 0] + 0.4 * np.array([0, 80, 0])).astype(np.uint8)
    if lines is not None:
        for x0, y0, x1, y1 in lines:
            cv2.line(overlay, (int(x0), int(y0)), (int(x1), int(y1)), (0, 0, 255), 2)
    path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(path), overlay)


def run_on_clip(clip_id: str, source: str, stumps_dets: pd.DataFrame, overlay_dir: Path | None = None) -> dict:
    idx = pd.read_parquet(ROOT / "data/processed/pose_index.parquet")
    row = idx[idx.clip_id == clip_id].iloc[0]
    video_path = ROOT / "data/interim/clips" / source / f"{clip_id}.mp4"
    z = np.load(ROOT / row.kps_path)
    stumps_boxes = stumps_dets[(stumps_dets.clip_id == clip_id) & (stumps_dets.cls == "stumps")]

    sample_idxs = sorted(stumps_dets[stumps_dets.clip_id == clip_id].frame.unique().tolist()) or [0, 4, 8, 12]
    frames = read_sample_frames(video_path, sample_idxs)
    cm_per_px, method, frame_i = calibrate_clip(frames, stumps_boxes)

    result = {"clip_id": clip_id, "source": source, "cm_per_px": cm_per_px, "calib_method": method}
    if cm_per_px is not None:
        m = measure_stride_and_swing(z["kps"], z["window"], float(z["fps"]), cm_per_px)
        # a stride outside 10-250cm or a swing over 40 m/s is not a real batting motion -- the scale
        # itself was wrong (most often a false crease-line pairing with no stumps detection to catch it,
        # see docs/iva/iva_results.md D4), so report the failure honestly rather than an impossible number
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

    if overlay_dir is not None and frames:
        f = frames[frame_i if frame_i is not None else 0]
        mask = segment_pitch(f)
        lines = detect_crease_lines(f, mask)
        save_overlay(f, mask, lines, overlay_dir / f"{clip_id}.jpg")
    return result


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--clips", nargs="+", required=True)
    ap.add_argument("--overlay-dir", default=None)
    args = ap.parse_args()

    idx = pd.read_parquet(ROOT / "data/processed/pose_index.parquet")
    stumps_dets = pd.read_csv(ROOT / "kaggle/chunks/stumps-c0/out/stumps_dets.csv")
    overlay_dir = Path(args.overlay_dir) if args.overlay_dir else None

    rows = []
    for cid in args.clips:
        source = idx[idx.clip_id == cid].iloc[0].source
        rows.append(run_on_clip(cid, source, stumps_dets, overlay_dir))
    out = pd.DataFrame(rows)
    print(out.to_string(index=False))


if __name__ == "__main__":
    main()
