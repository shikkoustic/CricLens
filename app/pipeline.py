"""End-to-end inference on one raw clip: batter -> pose -> shot -> technique -> stride/swing.

Mirrors the training-time data path exactly so the models see what they were trained on:
480p frames, batter chosen by the trained batter finder (kaggle/pose-rest/pose_rest.py's logic, ported
here because that file is a Kaggle script with side effects at import), ViTPose-Base with zero-padded
crops, the contact-anchored window (0.8 s before, 0.6 s after, never crossing a camera cut), T=32
resampling (models/resample_sequences.py), hip-centred normalisation (models/train_shot_classifier.py),
then the LSTM shot classifier, the tuned VAE technique scorer and pitch calibration.
"""
import json
import os
import subprocess
import sys
from pathlib import Path

import cv2
import joblib
import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from models.pitch_calibration import calibrate_clip, measure_stride_and_swing  # noqa: E402
from models.resample_sequences import resample_clip  # noqa: E402
from models.train_shot_classifier import make_model, normalise  # noqa: E402
from models.train_technique_scorer import VAERegressor  # noqa: E402

PRE, FOLLOW, PAD, GAP = 0.8, 0.6, 0.12, 3
TARGET_H = 480
MAX_SECONDS = float(os.environ.get("CRICLENS_MAX_SECONDS", "10"))
SHOTS = ["cut", "defence", "drive", "flick_glance", "lofted", "pull_hook", "scoop", "sweep"]
SHOT_NAMES = {"cut": "Cut", "defence": "Defence", "drive": "Drive", "flick_glance": "Flick / Glance",
              "lofted": "Lofted shot", "pull_hook": "Pull / Hook", "scoop": "Scoop", "sweep": "Sweep"}
PARTS = ["head", "shoulder", "hands", "hips", "feet"]
SKEL = [(5, 7), (7, 9), (6, 8), (8, 10), (5, 6), (5, 11), (6, 12), (11, 12), (11, 13), (13, 15), (12, 14),
        (14, 16), (0, 5), (0, 6)]
CPU = dict(device="cpu", verbose=False)


class AnalysisError(Exception):
    """A clip we can't analyse, with a message fit to show the user."""


def _iou(a, b):
    x0, y0, x1, y1 = max(a[0], b[0]), max(a[1], b[1]), min(a[2], b[2]), min(a[3], b[3])
    i = max(0, x1 - x0) * max(0, y1 - y0)
    return i / ((a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - i + 1e-9)


def _pct(value, quantiles, levels):
    if value is None or quantiles is None or not np.isfinite(value):
        return None
    return float(np.interp(value, quantiles, levels))


class Analyzer:
    def __init__(self, status=lambda msg: None):
        from transformers import AutoProcessor, VitPoseForPoseEstimation
        from ultralytics import YOLO
        self._YOLO = YOLO
        self.person_model = os.environ.get("CRICLENS_PERSON_MODEL", "yolo11m.pt")
        status("Loading detectors"); YOLO(self.person_model)  # fetch weights once, up front
        self.ptype = YOLO(str(ROOT / "data/raw/cricshot10k/models/Player_Type_Detection_Model.pt"))
        self.striker_cls = [i for i, n in self.ptype.names.items() if n.lower() in ("striker", "batsman", "batter")]
        self.bat_cls = [i for i, n in self.ptype.names.items() if n.lower() == "bat"]
        self.detector = YOLO(str(ROOT / "kaggle/train-detector/out/detector/weights/best.pt"))
        self.finder = joblib.load(ROOT / "models/batter_finder.joblib")
        status("Loading pose model")
        vit = "usyd-community/vitpose-base-simple"
        self.proc = AutoProcessor.from_pretrained(vit)
        self.vit = VitPoseForPoseEstimation.from_pretrained(vit).eval()
        status("Loading shot and technique models")
        self.classifier = make_model("lstm", 51, len(SHOTS), 32)
        self.classifier.load_state_dict(torch.load(ROOT / "models/shot_classifier/lstm.pt", map_location="cpu"))
        self.classifier.eval()
        self.scorer = VAERegressor(51, 32, 6)
        self.scorer.load_state_dict(torch.load(ROOT / "models/technique_scorer/vae_regressor.pt", map_location="cpu"))
        self.scorer.eval()
        self.ref = json.load(open(ROOT / "app/reference.json"))

    # ---- video ------------------------------------------------------------------------------------
    @staticmethod
    def read(path):
        cap = cv2.VideoCapture(str(path))
        fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
        fps = fps if 5 <= fps <= 120 else 25.0
        frames, limit = [], int(MAX_SECONDS * fps)
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        while len(frames) < limit:
            ok, f = cap.read()
            if not ok:
                break
            h, w = f.shape[:2]
            if h != TARGET_H:  # the training clips are all standardised to 480p
                f = cv2.resize(f, (int(round(w * TARGET_H / h / 2) * 2), TARGET_H), interpolation=cv2.INTER_AREA)
            frames.append(f)
        cap.release()
        return frames, float(fps), total > len(frames)

    @staticmethod
    def camera_cuts(frames):
        hs = []
        for f in frames:
            h = cv2.calcHist([cv2.cvtColor(f, cv2.COLOR_BGR2HSV)], [0, 1], None, [32, 32], [0, 180, 0, 256])
            hs.append(cv2.normalize(h, h).flatten())
        return [i for i in range(1, len(hs)) if cv2.compareHist(hs[i - 1], hs[i], cv2.HISTCMP_BHATTACHARYYA) > 0.5]

    # ---- batter -----------------------------------------------------------------------------------
    def track_people(self, frames, tick):
        det = self._YOLO(self.person_model)  # fresh tracker state per clip
        tracks = {}
        for i, f in enumerate(frames):
            r = det.track(f, persist=True, classes=[0], conf=0.2, tracker="bytetrack.yaml", **CPU)[0]
            if r.boxes.id is not None:
                for b, t in zip(r.boxes.xyxy.numpy(), r.boxes.id.numpy().astype(int)):
                    tracks.setdefault(int(t), {})[i] = b
            tick(i / len(frames))
        return tracks

    def cue_hits(self, frames, idx):
        out = {}
        for i, r in zip(idx, self.ptype.predict([frames[i] for i in idx], conf=0.25, **CPU)):
            pb, pc = r.boxes.xyxy.numpy(), r.boxes.cls.numpy().astype(int)
            out[i] = ([b for b, c in zip(pb, pc) if c in self.striker_cls],
                      [((b[0] + b[2]) / 2, (b[1] + b[3]) / 2) for b, c in zip(pb, pc) if c in self.bat_cls])
        return out

    def detect_objects(self, frames, idx):
        """Ball/bat/stumps detector on the same early frames the dataset-wide stumps job used."""
        rows = []
        for i, r in zip(idx, self.detector.predict([frames[i] for i in idx], conf=0.25, **CPU)):
            for b, c, s in zip(r.boxes.xyxy.numpy(), r.boxes.cls.numpy().astype(int), r.boxes.conf.numpy()):
                rows.append({"frame": i, "cls": self.detector.names[c], "x0": b[0], "y0": b[1], "x1": b[2], "y1": b[3],
                             "conf": float(s)})
        return pd.DataFrame(rows, columns=["frame", "cls", "x0", "y0", "x1", "y1", "conf"])

    @staticmethod
    def track_features(tracks, cues, H, W, n):
        rows = []
        med = {t: np.median(np.array(list(fb.values())), axis=0) for t, fb in tracks.items() if fb}
        for tid, fb in tracks.items():
            fi = sorted(fb); B = np.array([fb[i] for i in fi]); w, h = B[:, 2] - B[:, 0], B[:, 3] - B[:, 1]
            cxs, cys = (B[:, 0] + B[:, 2]) / 2, (B[:, 1] + B[:, 3]) / 2
            k = max(1, len(h) // 4); ci = [i for i in fi if i in cues]
            mb = med[tid]; mw, mh = mb[2] - mb[0], mb[3] - mb[1]
            keeper = any(o != tid and (m[3] - m[1]) < 1.3 * (m[2] - m[0]) and m[3] < mb[3]
                         and abs((m[0] + m[2]) / 2 - (mb[0] + mb[2]) / 2) < 1.5 * mw and (mb[1] - m[3]) < 1.0 * mh
                         for o, m in med.items())
            rows.append({
                "tid": tid, "n_tracks": len(tracks), "cover": len(fb) / n, "first": fi[0] / n,
                "cx": float(np.median(cxs)) / W, "cy": float(np.median(cys)) / H,
                "w": float(np.median(w)) / W, "h": float(np.median(h)) / H,
                "aspect": float(np.median(h / np.maximum(w, 1))),
                "growth": float(np.log(np.median(h[-k:]) / max(np.median(h[:k]), 1))),
                "move": float(np.mean(np.hypot(np.diff(cxs), np.diff(cys))) / max(np.median(h), 1)) if len(fi) > 1 else 0.0,
                "striker": float(np.mean([any(_iou(fb[i], s) > 0.3 for s in cues[i][0]) for i in ci])) if ci else 0.0,
                "bat": float(np.mean([any(fb[i][0] - 0.25 * (fb[i][2] - fb[i][0]) <= bx <= fb[i][2] + 0.25 * (fb[i][2] - fb[i][0])
                                          and fb[i][1] <= by <= fb[i][3] for bx, by in cues[i][1]) for i in ci])) if ci else 0.0,
                "size_rank": int(sorted(-(m[3] - m[1]) for m in med.values()).index(-mh)),
                "keeper_behind": int(keeper)})
        return rows

    @staticmethod
    def stump_features(tracks, stumps):
        rows = []
        for tid, fb in tracks.items():
            dxs, dys, seen = [], [], 0
            for f, sb in stumps.items():
                if not len(sb):
                    continue
                seen += 1
                near = [g for g in fb if abs(g - f) <= 2]
                if not near:
                    continue
                b = fb[min(near, key=lambda g: abs(g - f))]
                far = sb[np.argmin(sb[:, 3])]
                th = max(b[3] - b[1], 1.0)
                dxs.append(((b[0] + b[2]) / 2 - (far[0] + far[2]) / 2) / th)
                dys.append((b[3] - far[3]) / th)
            dx, dy = (float(np.median(dxs)), float(np.median(dys))) if dxs else (np.nan, np.nan)
            rows.append({"tid": tid, "st_found": seen / max(len(stumps), 1), "st_dx": dx, "st_dy": dy,
                         "st_far": float(bool(dxs)), "st_in_front": float(bool(dxs) and abs(dx) < 0.6 and 0.0 <= dy <= 0.6),
                         "st_behind": float(bool(dxs) and abs(dx) < 0.8 and dy < -0.05),
                         "st_beside": float(bool(dxs) and abs(dx) >= 0.6)})
        return rows

    def choose_batter(self, tracks, cues, stumps, H, W, n):
        rows = self.track_features(tracks, cues, H, W, n)
        if not rows:
            return None, float("nan")
        d = pd.DataFrame(rows).merge(pd.DataFrame(self.stump_features(tracks, stumps)), on="tid", how="left")
        for c in ("cy", "h", "striker", "bat"):
            d[f"{c}_rel"] = d[c] - d[c].mean()
        p = self.finder["model"].predict_proba(d.reindex(columns=self.finder["features"]).to_numpy(float))[:, 1]
        j = int(np.argmax(p))
        return int(d["tid"].iloc[j]), float(p[j])

    @staticmethod
    def fill_boxes(fb, n):
        out = np.full((n, 4), np.nan)
        for i, b in fb.items():
            out[i] = b
        ok = np.where(np.isfinite(out[:, 0]))[0]
        for a, b in zip(ok[:-1], ok[1:]):
            if 1 < b - a <= GAP + 1:
                for j in range(a + 1, b):
                    out[j] = out[a] + (out[b] - out[a]) * (j - a) / (b - a)
        return out

    # ---- pose -------------------------------------------------------------------------------------
    def vitpose(self, frames, boxes, tick):
        k = np.full((len(frames), 17, 3), np.nan, np.float32)
        idx = [i for i in range(len(frames)) if np.isfinite(boxes[i, 0])]
        for s in range(0, len(idx), 16):
            ch, bx = idx[s:s + 16], []
            for i in ch:
                x0, y0, x1, y1 = boxes[i]; pw, ph = PAD * (x1 - x0), PAD * (y1 - y0)
                x0, y0, x1, y1 = x0 - pw, y0 - ph, x1 + pw, y1 + ph  # unclamped = zero padding (D3)
                bx.append([[x0, y0, x1 - x0, y1 - y0]])
            inp = self.proc(images=[cv2.cvtColor(frames[i], cv2.COLOR_BGR2RGB) for i in ch], boxes=bx, return_tensors="pt")
            with torch.no_grad():
                o = self.vit(**inp)
            for i, r in zip(ch, self.proc.post_process_pose_estimation(o, boxes=bx)):
                k[i, :, :2] = r[0]["keypoints"].numpy(); k[i, :, 2] = r[0]["scores"].numpy()
            tick(min(1.0, (s + 16) / max(len(idx), 1)))
        for j in range(17):
            for c in range(3):
                k[:, j, c] = pd.Series(k[:, j, c]).interpolate(limit=GAP, limit_area="inside").to_numpy()
        return k

    @staticmethod
    def window(k, boxes, fps, n, cuts):
        wr = k[:, [9, 10], :2]; hgt = boxes[:, 3] - boxes[:, 1]
        with np.errstate(all="ignore"):
            sp = np.nanmax(np.linalg.norm(np.diff(wr, axis=0), axis=-1), axis=1) / hgt[1:]
        sp = np.convolve(np.nan_to_num(sp), np.ones(3) / 3, mode="same")
        contact = int(np.argmax(sp)) + 1 if sp.size and sp.max() > 0 else n // 2
        contact = min(max(contact, 0), n - 1)
        lo, hi = max(0, contact - int(PRE * fps)), min(n - 1, contact + int(FOLLOW * fps))
        for c in cuts:
            if c <= contact:
                lo = max(lo, c)
            else:
                hi = min(hi, c - 1)
        return lo, contact, hi

    # ---- main -------------------------------------------------------------------------------------
    def analyze(self, video_path, out_dir: Path, progress=lambda stage, frac: None) -> dict:
        out_dir.mkdir(parents=True, exist_ok=True)
        warnings = []
        progress("read", 0.02)
        frames, fps, truncated = self.read(video_path)
        n = len(frames)
        if n < 8:
            raise AnalysisError("That video is too short or couldn't be decoded. Try an MP4 of one delivery.")
        if truncated:
            warnings.append(f"Only the first {MAX_SECONDS:.0f} seconds were analysed -- trim the clip to one delivery for best results.")
        H, W = frames[0].shape[:2]

        progress("batter", 0.05)
        cuts = self.camera_cuts(frames)
        tracks = self.track_people(frames, lambda f: progress("batter", 0.05 + 0.25 * f))
        stance = list(range(0, max(3, int(0.6 * n)), max(1, n // 12)))
        cues = self.cue_hits(frames, stance)
        sample_idx = [i for i in (0, 4, 8, 12) if i < n]
        objs = self.detect_objects(frames, sample_idx)
        stumps = {i: objs[(objs.frame == i) & (objs.cls == "stumps")][["x0", "y0", "x1", "y1"]].to_numpy() for i in sample_idx}
        tracks = {t: fb for t, fb in tracks.items() if len(fb) >= 3}
        tid, finder_p = self.choose_batter(tracks, cues, stumps, H, W, n)
        if tid is None:
            raise AnalysisError("No batter found. The app expects a broadcast-style view of one delivery with the batter visible.")
        boxes = self.fill_boxes(tracks[tid], n)

        progress("pose", 0.32)
        k = self.vitpose(frames, boxes, lambda f: progress("pose", 0.32 + 0.33 * f))
        lo, contact, hi = self.window(k, boxes, fps, n, cuts)
        kw = k[lo:hi + 1]
        found = float(np.isfinite(kw[:, 0, 0]).mean()) if len(kw) else 0.0
        conf = float(np.nanmean(kw[:, :, 2])) if np.isfinite(kw[:, :, 2]).any() else 0.0
        if found < 0.8 or conf < 0.5:
            warnings.append("The batter was only partly visible or tracked with low confidence around the shot, "
                            "so the results below are less reliable than usual.")
        if finder_p < 0.5:
            warnings.append("The app wasn't fully sure which player is the batter -- check the highlighted player in the video.")

        progress("shot", 0.68)
        rs = resample_clip(k, np.array([lo, contact, hi]), fps)
        if rs is None:
            raise AnalysisError("The shot window was too short to analyse. Try a clip with the full swing visible.")
        x = torch.from_numpy(normalise(rs["seq"])).unsqueeze(0)
        with torch.no_grad():
            probs = torch.softmax(self.classifier(x), 1)[0].numpy()
        order = np.argsort(-probs)
        shot = SHOTS[int(order[0])]

        progress("technique", 0.74)
        mu, sd = np.array(self.ref["scorer_mu"]), np.array(self.ref["scorer_sd"])
        with torch.no_grad():
            pred = self.scorer(x)[3][0].numpy() * sd + mu
        scores = {t: float(np.clip(v, 0, 10)) for t, v in zip(self.ref["targets"], pred)}
        levels = self.ref["quantile_levels"]
        score_ref = shot if shot in self.ref["score"] else "all"
        sq = self.ref["score"][score_ref]

        progress("metrics", 0.78)
        cal_frames = [frames[i] for i in sample_idx]
        cm_per_px, method, _ = calibrate_clip(cal_frames, objs[objs.cls == "stumps"])
        metrics = {"calibrated": False, "method": method, "stride_cm": None, "swing_mps": None}
        if cm_per_px is not None:
            m = measure_stride_and_swing(k, np.array([lo, contact, hi]), fps, cm_per_px)
            stride = m["stride_cm"] if 10 <= (m["stride_cm"] or 0) <= 250 else None
            swing = m["swing_speed_mps"] if (m["swing_speed_mps"] or 99) <= 40 else None
            if stride is not None or swing is not None:
                mref = shot if shot in self.ref["stride_cm"] else "all"
                metrics = {"calibrated": True, "method": method, "cm_per_px": cm_per_px, "reference_shot": mref,
                           "stride_cm": stride, "swing_mps": swing,
                           "stride_pct": _pct(stride, self.ref["stride_cm"].get(mref), levels),
                           "swing_pct": _pct(swing, self.ref["swing_mps"].get(mref), levels)}

        progress("render", 0.82)
        media = self.render(frames, k, boxes, (lo, contact, hi), SHOT_NAMES[shot], fps, out_dir)

        return {
            "video": {"fps": fps, "frames": n, "seconds": n / fps, "width": W, "height": H, "truncated": truncated},
            "batter": {"finder_p": finder_p, "found_in_window": found, "pose_confidence": conf,
                       "window": {"start": lo, "contact": contact, "end": hi,
                                  "start_s": lo / fps, "contact_s": contact / fps, "end_s": hi / fps}},
            "shot": {"label": shot, "display": SHOT_NAMES[shot], "confidence": float(probs[order[0]]),
                     "probs": [{"label": SHOTS[i], "display": SHOT_NAMES[SHOTS[i]], "p": float(probs[i])} for i in order]},
            "technique": {"overall": scores["overall"], "overall_pct": _pct(scores["overall"], sq["overall"], levels),
                          "reference_shot": score_ref,
                          "parts": [{"part": p, "score": scores[p], "pct": _pct(scores[p], sq[p], levels)} for p in PARTS]},
            "metrics": metrics, "media": media, "warnings": warnings}

    # ---- overlay ----------------------------------------------------------------------------------
    @staticmethod
    def render(frames, k, boxes, win, label, fps, out_dir: Path) -> dict:
        lo, contact, hi = win
        H, W = frames[0].shape[:2]
        t = max(1, H // 240)
        drawn = []
        for i, f in enumerate(frames):
            g = f.copy()
            inside = lo <= i <= hi
            if np.isfinite(boxes[i]).all():
                x0, y0, x1, y1 = boxes[i].astype(int)
                cv2.rectangle(g, (x0, y0), (x1, y1), (80, 200, 255) if inside else (160, 160, 160), 1, cv2.LINE_AA)
                for u, v in SKEL:
                    if np.isfinite(k[i, [u, v], 2]).all() and min(k[i, u, 2], k[i, v, 2]) > 0.3:
                        cv2.line(g, tuple(k[i, u, :2].astype(int)), tuple(k[i, v, :2].astype(int)),
                                 (60, 90, 255) if inside else (200, 200, 200), 2 * t, cv2.LINE_AA)
                for j in range(17):
                    if np.isfinite(k[i, j, 2]) and k[i, j, 2] > 0.3:
                        cv2.circle(g, tuple(k[i, j, :2].astype(int)), 2 * t, (255, 255, 255), -1, cv2.LINE_AA)
            if inside:
                cv2.rectangle(g, (0, 0), (W - 1, 4 * t), (60, 90, 255), -1)
            if i == contact:
                cv2.putText(g, "CONTACT", (10, H - 14), cv2.FONT_HERSHEY_SIMPLEX, 0.6 * t, (255, 255, 255), 2 * t, cv2.LINE_AA)
            drawn.append(g)

        tmp = out_dir / "overlay_raw.avi"
        vw = cv2.VideoWriter(str(tmp), cv2.VideoWriter_fourcc(*"MJPG"), fps, (W, H))
        for g in drawn:
            vw.write(g)
        vw.release()
        import imageio_ffmpeg
        subprocess.run([imageio_ffmpeg.get_ffmpeg_exe(), "-y", "-loglevel", "error", "-i", str(tmp), "-c:v", "libx264",
                        "-pix_fmt", "yuv420p", "-crf", "23", "-movflags", "+faststart", str(out_dir / "overlay.mp4")],
                       check=True)
        tmp.unlink(missing_ok=True)

        keyframes = []
        for name, i in (("Backlift", lo), ("Contact", contact), ("Follow-through", hi)):
            fn = f"key_{name.lower().replace('-', '')}.jpg"
            cv2.imwrite(str(out_dir / fn), drawn[i], [cv2.IMWRITE_JPEG_QUALITY, 88])
            keyframes.append({"label": name, "file": fn, "time_s": i / fps})
        return {"video": "overlay.mp4", "keyframes": keyframes}
