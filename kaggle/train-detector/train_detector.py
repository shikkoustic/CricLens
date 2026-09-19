"""Train the CricLens ball / bat / stumps detector (YOLO11s, both T4s) on the merged public sets.

Used for: stumps + bat clues in the batter finder, and the app's ball/bat/stumps detection.
Outputs: /kaggle/working/detector/weights/best.pt, per-class val/test metrics in metrics.json.
"""
import glob, json, os, subprocess, sys
subprocess.run([sys.executable, "-m", "pip", "install", "-q", "ultralytics"], check=False)
import torch
from ultralytics import YOLO

root = glob.glob("/kaggle/input/**/det/images/train", recursive=True)[0].rsplit("/images/", 1)[0]
yaml = "/kaggle/working/det.yaml"
open(yaml, "w").write(f"path: {root}\ntrain: images/train\nval: images/val\ntest: images/test\nnames: ['ball', 'bat', 'stumps']\n")
dev = [0, 1] if torch.cuda.device_count() > 1 else 0
model = YOLO("yolo11s.pt")
model.train(data=yaml, epochs=60, imgsz=640, batch=32, device=dev, workers=4, patience=15, cos_lr=True,
            project="/kaggle/working", name="detector", exist_ok=True, plots=True, cache=False, amp=True)
best = YOLO("/kaggle/working/detector/weights/best.pt")
out = {}
for split in ("val", "test"):
    m = best.val(data=yaml, split=split, imgsz=640, batch=32, device=0, plots=False)
    out[split] = {"mAP50": float(m.box.map50), "mAP50-95": float(m.box.map),
                  "per_class_mAP50": {n: float(v) for n, v in zip(["ball", "bat", "stumps"], m.box.ap50)}}
json.dump(out, open("/kaggle/working/metrics.json", "w"), indent=1)
print(json.dumps(out, indent=1))
