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
