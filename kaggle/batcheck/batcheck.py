# ---- shared by batcheck / d3 (inlined into each kernel file at build time) ----
import glob, json, os, subprocess, sys, time, traceback
subprocess.run([sys.executable, "-m", "pip", "install", "-q", "ultralytics", "-U", "transformers>=4.48", "accelerate"], check=False)
import cv2, numpy as np, pandas as pd, torch
from transformers import AutoProcessor, VitPoseForPoseEstimation

DEV = "cuda" if torch.cuda.is_available() else "cpu"
OUT = "/kaggle/working"
D3 = glob.glob("/kaggle/input/**/bat_check.csv", recursive=True)[0].rsplit("/", 1)[0]
D2 = glob.glob("/kaggle/input/**/detector_best.pt", recursive=True)[0].rsplit("/", 1)[0]
CLIPS = [c for c in glob.glob("/kaggle/input/**/clips", recursive=True) if os.path.isdir(f"{c}/cricketvision")][0]
PAD = 0.12
_proc = AutoProcessor.from_pretrained("usyd-community/vitpose-base-simple")
_vit = VitPoseForPoseEstimation.from_pretrained("usyd-community/vitpose-base-simple",
                                                torch_dtype=torch.float16 if DEV == "cuda" else torch.float32).to(DEV).eval()
CLAHE = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))


def npz_for(clip_id):
    return np.load(glob.glob(f"/kaggle/input/**/kps/{clip_id}.npz", recursive=True)[0])


def read_frames(source, clip_id, a, b):
    cap = cv2.VideoCapture(f"{CLIPS}/{source}/{clip_id}.mp4"); fr, i = [], 0
    while i <= b:
        ok, f = cap.read()
        if not ok:
            break
        if i >= a:
            fr.append(f)
        i += 1
    cap.release()
    return fr


def padded(b, H=None, W=None):
    pw, ph = PAD * (b[2] - b[0]), PAD * (b[3] - b[1])
    x0, y0, x1, y1 = int(round(b[0] - pw)), int(round(b[1] - ph)), int(round(b[2] + pw)), int(round(b[3] + ph))
    if H is not None:
        x0, y0, x1, y1 = max(0, x0), max(0, y0), min(W, x1), min(H, y1)
    return [x0, y0, x1, y1]


def pose_crops(crops_bgr):
    """ViTPose on whole crops; keypoints (x, y, score) in crop coordinates."""
    out = np.full((len(crops_bgr), 17, 3), np.nan, np.float32)
    for s in range(0, len(crops_bgr), 64):
        ch = crops_bgr[s:s + 64]
        imgs = [cv2.cvtColor(c, cv2.COLOR_BGR2RGB) for c in ch]
        bx = [[[0, 0, c.shape[1], c.shape[0]]] for c in ch]
        inp = _proc(images=imgs, boxes=bx, return_tensors="pt").to(DEV)
        if DEV == "cuda":
            inp["pixel_values"] = inp["pixel_values"].half()
        with torch.no_grad():
            o = _vit(**inp)
        o.heatmaps = o.heatmaps.float()
        for j, r in enumerate(_proc.post_process_pose_estimation(o, boxes=bx)):
            out[s + j, :, :2] = r[0]["keypoints"].float().cpu().numpy(); out[s + j, :, 2] = r[0]["scores"].float().cpu().numpy()
    return out


def jitter(k, heights):
    a = np.linalg.norm(k[2:, :, :2] - 2 * k[1:-1, :, :2] + k[:-2, :, :2], axis=-1) / np.asarray(heights)[1:-1, None]
    return float(np.nanmedian(a)) if np.isfinite(a).any() else np.nan


def on_luma(bgr, fn):
    lab = cv2.cvtColor(bgr, cv2.COLOR_BGR2LAB)
    lab[..., 0] = np.clip(fn(lab[..., 0]), 0, 255).astype(np.uint8)
    return cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
# ---- end shared ----
"""Bat check: are CLAHE's extra bat detections real? Frames where baseline and CLAHE disagree are drawn side by
side (left: original + its bat boxes in green, right: CLAHE + its bat boxes in magenta) for a visual count."""
from ultralytics import YOLO
det = YOLO(f"{D2}/detector_best.pt"); BAT = [i for i, n in det.names.items() if n == "bat"]
Y = dict(device=0 if DEV == "cuda" else "cpu", verbose=False, half=DEV == "cuda", conf=0.25)
os.makedirs(f"{OUT}/sheets", exist_ok=True)


def bats_in(res, box, H, W):
    x0, y0, x1, y1 = padded(box, H, W)
    return [b for b, c in zip(res.boxes.xyxy.cpu().numpy(), res.boxes.cls.cpu().numpy())
            if int(c) in BAT and x0 <= (b[0] + b[2]) / 2 <= x1 and y0 <= (b[1] + b[3]) / 2 <= y1]


clips = pd.read_csv(f"{D3}/bat_check.csv"); tiles, rows = [], []
for s in clips.itertuples():
    try:
        z = npz_for(s.clip_id); boxes = z["boxes"]; lo, c, hi = [int(v) for v in z["window"]]
        a, b = max(lo, c - 25), min(hi, c + 20)
        fr = read_frames(s.source, s.clip_id, a, b); H, W = fr[0].shape[:2]
        sel = [i for i in range(0, len(fr), 3) if np.isfinite(boxes[a + i]).all()]
        base = [fr[i] for i in sel]; enh = [on_luma(f, CLAHE.apply) for f in base]
        rb, re_ = det.predict(base, **Y), det.predict(enh, **Y)
        n_clip = 0
        for i, img0, img1, r0, r1 in zip(sel, base, enh, rb, re_):
            box = boxes[a + i]; b0, b1 = bats_in(r0, box, H, W), bats_in(r1, box, H, W)
            if bool(b0) == bool(b1) or n_clip >= 3:
                continue
            x0, y0, x1, y1 = box; w, h = x1 - x0, y1 - y0
            cx0, cy0, cx1, cy1 = int(max(0, x0 - 0.35 * w)), int(max(0, y0 - 0.2 * h)), int(min(W, x1 + 0.35 * w)), int(min(H, y1 + 0.1 * h))
            pair = []
            for img, bb, col in ((img0, b0, (0, 255, 0)), (img1, b1, (255, 0, 255))):
                t = img.copy()
                for q in bb:
                    cv2.rectangle(t, (int(q[0]), int(q[1])), (int(q[2]), int(q[3])), col, 2)
                t = t[cy0:cy1, cx0:cx1]; t = cv2.resize(t, (int(t.shape[1] * 300 / max(t.shape[0], 1)), 300))
                pair.append(t)
            tile = np.hstack([pair[0], np.full((300, 4, 3), 255, np.uint8), pair[1]])
            cv2.putText(tile, f"#{len(rows)}", (4, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
            rows.append({"tile": len(rows), "clip_id": s.clip_id, "group": s.group, "frame": a + i, "base_bats": len(b0), "clahe_bats": len(b1)})
            tiles.append(tile); n_clip += 1
    except Exception:
        print("clip failed", s.clip_id, traceback.format_exc()[-400:], flush=True)
for k in range(0, len(tiles), 12):
    ch = tiles[k:k + 12]; w = max(t.shape[1] for t in ch)
    ch = [cv2.copyMakeBorder(t, 0, 6, 0, w - t.shape[1], cv2.BORDER_CONSTANT, value=(255, 255, 255)) for t in ch]
    cols = 3; grid_rows = []
    for r in range(0, len(ch), cols):
        rr = ch[r:r + cols]
        while len(rr) < cols:
            rr.append(np.full_like(ch[0], 255))
        grid_rows.append(np.hstack(rr))
    cv2.imwrite(f"{OUT}/sheets/sheet_{k // 12:02d}.jpg", np.vstack(grid_rows), [cv2.IMWRITE_JPEG_QUALITY, 85])
pd.DataFrame(rows).to_csv(f"{OUT}/tiles.csv", index=False)
s = {"tiles": len(rows), "clahe_only": int(sum(r["clahe_bats"] > 0 for r in rows)), "base_only": int(sum(r["base_bats"] > 0 for r in rows))}
json.dump(s, open(f"{OUT}/summary.json", "w")); print(s)
