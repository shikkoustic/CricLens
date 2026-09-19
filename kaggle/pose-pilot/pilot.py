"""CricLens pose pilot (~283 clips): ViTPose-Base (top-down) vs YOLO11m-pose (one-stage), all PyTorch.

Questions:
  1. batter selection: does the pipeline pick the striker? (IoU vs CricketVision annotated boxes)
  2. which pose model: temporal steadiness (jitter), cross-model agreement, speed, visual check
  3. resolution: do keypoints from our 480p clips match those from the original files?
Both pose models get the SAME batter box per frame, so differences are pose quality only.
Outputs in /kaggle/working: summary.json, per_clip.csv, frames/*.jpg (side-by-side), kps/*.npz
"""
import glob, json, os, subprocess, sys, time, traceback


def pip(*a):
    subprocess.run([sys.executable, "-m", "pip", "install", "-q", *a], check=False)


pip("ultralytics")
pip("-U", "transformers>=4.48", "accelerate")
import cv2, numpy as np, pandas as pd, torch
from ultralytics import YOLO
from transformers import AutoProcessor, VitPoseForPoseEstimation

DEV = "cuda" if torch.cuda.is_available() else "cpu"
IN = glob.glob("/kaggle/input/**/pilot.csv", recursive=True)[0].rsplit("/", 1)[0]
OUT = "/kaggle/working"
for d in ("frames", "kps"):
    os.makedirs(f"{OUT}/{d}", exist_ok=True)
print("device", DEV, torch.cuda.get_device_name(0) if DEV == "cuda" else "", "| input", IN, flush=True)

det = YOLO("yolo11m.pt")              # person boxes (COCO class 0)
ypose = YOLO("yolo11m-pose.pt")       # one-stage pose
ptype = YOLO(glob.glob(f"{IN}/**/Player_Type_Detection_Model.pt", recursive=True)[0])
VIT = "usyd-community/vitpose-base-simple"
proc = AutoProcessor.from_pretrained(VIT)
vit = VitPoseForPoseEstimation.from_pretrained(VIT).to(DEV).eval()
print("player-type classes:", ptype.names, flush=True)
BAT = [i for i, n in ptype.names.items() if any(k in n.lower() for k in ("bats", "striker", "batter"))]
print("batter class ids:", BAT, flush=True)
SKEL = [(5, 7), (7, 9), (6, 8), (8, 10), (5, 6), (5, 11), (6, 12), (11, 12), (11, 13), (13, 15), (12, 14), (14, 16), (0, 5), (0, 6)]


def iou(a, b):
    x0, y0, x1, y1 = max(a[0], b[0]), max(a[1], b[1]), min(a[2], b[2]), min(a[3], b[3])
    i = max(0, x1 - x0) * max(0, y1 - y0)
    return i / ((a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - i + 1e-9)


def read(path):
    cap, fr = cv2.VideoCapture(path), []
    while True:
        ok, f = cap.read()
        if not ok:
            break
        fr.append(f)
    cap.release()
    return fr


def batter_boxes(frames):
    """Per-frame striker box (x0,y0,x1,y1): player-type batter class, else a person near the
    frame centre / upper-middle (behind-the-bowler view), kept consistent with the previous box."""
    h, w = frames[0].shape[:2]
    pt = ptype.predict(frames, verbose=False, conf=0.25, device=0 if DEV == "cuda" else "cpu") if BAT else [None] * len(frames)
    pr = det.predict(frames, verbose=False, conf=0.25, classes=[0], device=0 if DEV == "cuda" else "cpu")
    out, how, prev = [], [], None
    for p, r in zip(pt, pr):
        box, src = None, "none"
        if p is not None and len(p.boxes):
            cand = [(float(c), b) for b, k, c in zip(p.boxes.xyxy.cpu().numpy(), p.boxes.cls.cpu().numpy(), p.boxes.conf.cpu().numpy()) if int(k) in BAT]
            if cand:
                box, src = max(cand, key=lambda t: t[0])[1], "ptype"
        if box is None and len(r.boxes):
            ps = r.boxes.xyxy.cpu().numpy()
            ps = ps[(ps[:, 3] - ps[:, 1]) >= 0.08 * h]
            if len(ps):
                if prev is not None and max(iou(prev, q) for q in ps) > 0.3:
                    box = max(ps, key=lambda q: iou(prev, q)); src = "track"
                else:
                    cx, cy = (ps[:, 0] + ps[:, 2]) / 2, (ps[:, 1] + ps[:, 3]) / 2
                    box = ps[np.argmin(np.abs(cx / w - 0.5) + np.abs(cy / h - 0.45))]; src = "centre"
        out.append(None if box is None else np.asarray(box, float)); how.append(src)
        if box is not None:
            prev = np.asarray(box, float)
    return out, how


def vitpose(frames, boxes):
    k = np.full((len(frames), 17, 3), np.nan, np.float32)
    idx = [i for i, b in enumerate(boxes) if b is not None]
    for s in range(0, len(idx), 32):
        ch = idx[s:s + 32]
        imgs = [cv2.cvtColor(frames[i], cv2.COLOR_BGR2RGB) for i in ch]
        bx = [[[boxes[i][0], boxes[i][1], boxes[i][2] - boxes[i][0], boxes[i][3] - boxes[i][1]]] for i in ch]
        inp = proc(images=imgs, boxes=bx, return_tensors="pt").to(DEV)
        with torch.no_grad():
            o = vit(**inp)
        res = proc.post_process_pose_estimation(o, boxes=bx)
        for i, r in zip(ch, res):
            k[i, :, :2] = r[0]["keypoints"].cpu().numpy(); k[i, :, 2] = r[0]["scores"].cpu().numpy()
    return k


def yolopose(frames, boxes):
    k = np.full((len(frames), 17, 3), np.nan, np.float32)
    res = ypose.predict(frames, verbose=False, conf=0.2, device=0 if DEV == "cuda" else "cpu")
    for i, (r, b) in enumerate(zip(res, boxes)):
        if b is None or r.keypoints is None or not len(r.boxes):
            continue
        ious = [iou(b, q) for q in r.boxes.xyxy.cpu().numpy()]
        j = int(np.argmax(ious))
        if ious[j] > 0.3:
            k[i, :, :2] = r.keypoints.xy[j].cpu().numpy(); k[i, :, 2] = r.keypoints.conf[j].cpu().numpy()
    return k


def jitter(k, boxes):
    """Median |second difference| of joint positions / box height: lower = steadier."""
    hs = np.array([b[3] - b[1] if b is not None else np.nan for b in boxes])
    a = np.linalg.norm(k[2:, :, :2] - 2 * k[1:-1, :, :2] + k[:-2, :, :2], axis=-1) / hs[1:-1, None]
    return float(np.nanmedian(a)) if np.isfinite(a).any() else np.nan


def draw(f, k, b, color):
    g = f.copy()
    if b is not None:
        cv2.rectangle(g, tuple(map(int, b[:2])), tuple(map(int, b[2:])), (0, 255, 255), 1)
    for a_, c_ in SKEL:
        if np.isfinite(k[a_, :2]).all() and np.isfinite(k[c_, :2]).all() and min(k[a_, 2], k[c_, 2]) > 0.3:
            cv2.line(g, tuple(map(int, k[a_, :2])), tuple(map(int, k[c_, :2])), color, 2)
    return g


df = pd.read_csv(f"{IN}/pilot.csv")
rows, tv, ty, nfr = [], 0.0, 0.0, 0
for n, r in enumerate(df.itertuples()):
    try:
        fr = read(glob.glob(f"{IN}/**/{r.clip_id}.mp4", recursive=True)[0])
        if len(fr) < 3:
            continue
        t = time.time(); boxes, how = batter_boxes(fr); tb = time.time() - t
        t = time.time(); kv = vitpose(fr, boxes); tv += time.time() - t + tb
        t = time.time(); ky = yolopose(fr, boxes); ty += time.time() - t + tb
        nfr += len(fr)
        hh = np.array([b[3] - b[1] if b is not None else np.nan for b in boxes])
        row = {"clip_id": r.clip_id, "source": r.source, "frames": len(fr), "h": fr[0].shape[0],
               "batter_found": float(np.mean([b is not None for b in boxes])),
               "via_ptype": float(np.mean([x == "ptype" for x in how])),
               "batter_h_px": float(np.nanmedian(hh)),
               "vit_found": float(np.mean(np.isfinite(kv[:, 0, 0]))), "yolo_found": float(np.mean(np.isfinite(ky[:, 0, 0]))),
               "vit_jitter": jitter(kv, boxes), "yolo_jitter": jitter(ky, boxes),
               "vit_conf": float(np.nanmean(kv[:, :, 2])), "yolo_conf": float(np.nanmean(ky[:, :, 2])),
               "vit_vs_yolo": float(np.nanmedian(np.linalg.norm(kv[:, :, :2] - ky[:, :, :2], axis=-1) / hh[:, None]))}
        if isinstance(r.execution_bbox, str):  # CricketVision annotated batter box (1280x720 px, xywh)
            x, y, w, hb = json.loads(r.execution_bbox); s = fr[0].shape[0] / 720
            mid = len(boxes) // 2
            row["cv_iou"] = iou(boxes[mid], [x * s, y * s, (x + w) * s, (y + hb) * s]) if boxes[mid] is not None else 0.0
        if isinstance(r.orig, str):  # resolution check: same pipeline on the original file
            fo = read(glob.glob(f"{IN}/**/{r.orig}", recursive=True)[0])
            m = min(len(fo), len(fr))
            if m >= 3:
                bo, _ = batter_boxes(fo[:m]); kvo = vitpose(fo[:m], bo); kyo = yolopose(fo[:m], bo)
                sc = fo[0].shape[0] / fr[0].shape[0]; ho = np.array([b[3] - b[1] if b is not None else np.nan for b in bo])
                row["orig_h"] = fo[0].shape[0]
                row["vit_480_vs_orig"] = float(np.nanmedian(np.linalg.norm(kv[:m, :, :2] * sc - kvo[:, :, :2], axis=-1) / ho[:, None]))
                row["yolo_480_vs_orig"] = float(np.nanmedian(np.linalg.norm(ky[:m, :, :2] * sc - kyo[:, :, :2], axis=-1) / ho[:, None]))
        np.savez_compressed(f"{OUT}/kps/{r.clip_id}.npz", vit=kv, yolo=ky,
                            boxes=np.array([b if b is not None else [np.nan] * 4 for b in boxes], np.float32))
        if n % 12 == 0:  # side-by-side frames for a visual check: left ViTPose, right YOLO
            for j in (len(fr) // 4, len(fr) // 2, 3 * len(fr) // 4):
                cv2.imwrite(f"{OUT}/frames/{r.clip_id}_{j:03d}.jpg",
                            np.hstack([draw(fr[j], kv[j], boxes[j], (0, 0, 255)), draw(fr[j], ky[j], boxes[j], (255, 128, 0))]))
        rows.append(row)
        if n % 20 == 0:
            print(n, {k: (round(v, 3) if isinstance(v, float) else v) for k, v in row.items()}, flush=True)
    except Exception:
        print("clip failed", r.clip_id, traceback.format_exc()[-500:], flush=True)

res = pd.DataFrame(rows); res.to_csv(f"{OUT}/per_clip.csv", index=False)
g = lambda c: float(res[c].median()) if c in res and res[c].notna().any() else None
summary = {
    "device": DEV, "clips": len(res), "frames": nfr,
    "fps_vitpose_incl_detection": nfr / tv if tv else None, "fps_yolo_incl_detection": nfr / ty if ty else None,
    "batter_found_rate": float(res.batter_found.mean()), "via_player_type_model": float(res.via_ptype.mean()),
    "median_batter_height_px_480p": g("batter_h_px"),
    "cv_batter_iou_median": g("cv_iou"),
    "cv_batter_iou_gt_0.5": float((res["cv_iou"] > 0.5).mean()) if "cv_iou" in res else None,
    "vit_found": float(res.vit_found.mean()), "yolo_found": float(res.yolo_found.mean()),
    "vit_jitter_median": g("vit_jitter"), "yolo_jitter_median": g("yolo_jitter"),
    "vit_vs_yolo_median_rel": g("vit_vs_yolo"),
    "vit_480_vs_orig_median_rel": g("vit_480_vs_orig"), "yolo_480_vs_orig_median_rel": g("yolo_480_vs_orig"),
    "by_source": res.groupby("source")[["batter_found", "vit_jitter", "yolo_jitter"]].median().round(4).to_dict(),
}
json.dump(summary, open(f"{OUT}/summary.json", "w"), indent=1)
print(json.dumps(summary, indent=1), flush=True)
