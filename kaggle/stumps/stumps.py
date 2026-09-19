"""Detect stumps / bat / ball on 4 early frames (0, 15, 30, 45% of the clip) of every canonical clip,
with the CricLens detector trained in criclens-train-detector. The striker stands in front of the far
stumps at the start of the delivery, so these frames feed the batter finder's stumps clues.
Output: stumps_dets.csv (clip_id, frame, H, W, cls, box, conf), summary.json
"""
import glob, json, os, subprocess, sys, time
subprocess.run([sys.executable, "-m", "pip", "install", "-q", "ultralytics"], check=False)
import cv2, pandas as pd, torch
from ultralytics import YOLO

W = glob.glob("/kaggle/input/**/detector/weights/best.pt", recursive=True)[0]
MAN = glob.glob("/kaggle/input/**/manifest.parquet", recursive=True)[0]
ROOT = [c for c in glob.glob("/kaggle/input/**/clips", recursive=True) if os.path.isdir(f"{c}/cricketvision")][0]
det = YOLO(W); NAMES = det.names
m = pd.read_parquet(MAN); m = m[m.is_canonical]
dev = 0 if torch.cuda.is_available() else "cpu"
rows, t0 = [], time.time()
for n, r in enumerate(m.itertuples()):
    cap = cv2.VideoCapture(f"{ROOT}/{r.source}/{r.clip_id}.mp4"); N = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frames, idx = [], []
    for q in (0.0, 0.15, 0.30, 0.45):
        i = int(q * max(N - 1, 0)); cap.set(cv2.CAP_PROP_POS_FRAMES, i); ok, f = cap.read()
        if ok:
            frames.append(f); idx.append(i)
    cap.release()
    if not frames:
        continue
    H, Wd = frames[0].shape[:2]
    for i, res in zip(idx, det.predict(frames, conf=0.2, verbose=False, half=dev == 0, device=dev)):
        for b, c, s in zip(res.boxes.xyxy.cpu().numpy(), res.boxes.cls.cpu().numpy().astype(int), res.boxes.conf.cpu().numpy()):
            rows.append((r.clip_id, i, H, Wd, NAMES[int(c)], *map(float, b), float(s)))
    if n % 2000 == 0:
        print(n, len(m), len(rows), f"{time.time() - t0:.0f}s", flush=True)
d = pd.DataFrame(rows, columns=["clip_id", "frame", "H", "W", "cls", "x0", "y0", "x1", "y1", "conf"])
d.to_csv("/kaggle/working/stumps_dets.csv", index=False)
s = {"clips": len(m), "detections": len(d), "clips_with_stumps": int(d[d.cls == "stumps"].clip_id.nunique()),
     "seconds": time.time() - t0}
json.dump(s, open("/kaggle/working/summary.json", "w"), indent=1); print(s)
