"""CricLens pose re-check: fixed batter selection + ViTPose only (fp16), on the 283 pilot clips.

Fix over the first pilot: the player-type 'Striker' class also fired on scoreboard graphics
(player names/photos), and nothing rejected it. Now a batter box must be person-shaped, sit
above the scoreboard strip, and overlap a real person detection; the person holding the bat is
preferred; the same person is followed across frames; frames with no valid batter are left
missing (NaN) instead of guessed.
Checks: share of scoreboard-shaped picks (target ~0), agreement with CricketVision's annotated
batter boxes (per-clip best source scale), missing-frame rate, speed.
"""
import glob, json, os, subprocess, sys, time, traceback
subprocess.run([sys.executable, "-m", "pip", "install", "-q", "ultralytics", "-U", "transformers>=4.48", "accelerate"], check=False)
import cv2, numpy as np, pandas as pd, torch
from ultralytics import YOLO
from transformers import AutoProcessor, VitPoseForPoseEstimation

DEV = "cuda" if torch.cuda.is_available() else "cpu"
IN = glob.glob("/kaggle/input/**/pilot.csv", recursive=True)[0].rsplit("/", 1)[0]
OUT = "/kaggle/working"
for d in ("frames", "kps"):
    os.makedirs(f"{OUT}/{d}", exist_ok=True)
det = YOLO("yolo11m.pt")
ptype = YOLO(glob.glob(f"{IN}/**/Player_Type_Detection_Model.pt", recursive=True)[0])
STRIKER = [i for i, n in ptype.names.items() if n.lower() in ("striker", "batsman", "batter")]
BATCLS = [i for i, n in ptype.names.items() if n.lower() == "bat"]
VIT = "usyd-community/vitpose-base-simple"
proc = AutoProcessor.from_pretrained(VIT)
vit = VitPoseForPoseEstimation.from_pretrained(VIT, torch_dtype=torch.float16 if DEV == "cuda" else torch.float32).to(DEV).eval()
print("device", DEV, "| ptype", ptype.names, "striker", STRIKER, "bat", BATCLS, flush=True)
SKEL = [(5, 7), (7, 9), (6, 8), (8, 10), (5, 6), (5, 11), (6, 12), (11, 12), (11, 13), (13, 15), (12, 14), (14, 16), (0, 5), (0, 6)]
G = dict(device=0 if DEV == "cuda" else "cpu", verbose=False, half=DEV == "cuda")


def iou(a, b):
    x0, y0, x1, y1 = max(a[0], b[0]), max(a[1], b[1]), min(a[2], b[2]), min(a[3], b[3])
    i = max(0, x1 - x0) * max(0, y1 - y0)
    return i / ((a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - i + 1e-9)


def plausible(b, H):
    w, h, cy = b[2] - b[0], b[3] - b[1], (b[1] + b[3]) / 2
    return h > 0.9 * w and cy < 0.82 * H and h > 0.06 * H   # person-shaped, above the scoreboard strip


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
    H, W = frames[0].shape[:2]
    pts = ptype.predict(frames, conf=0.25, **G)
    prs = det.predict(frames, conf=0.25, classes=[0], **G)
    out, how, prev = [], [], None
    for p, r in zip(pts, prs):
        persons = [b for b in r.boxes.xyxy.cpu().numpy() if plausible(b, H)]
        pb = p.boxes.xyxy.cpu().numpy(); pc = p.boxes.cls.cpu().numpy().astype(int); pf = p.boxes.conf.cpu().numpy()
        strikers = [b for b, c in zip(pb, pc) if c in STRIKER and plausible(b, H)]
        bats = [((b[0] + b[2]) / 2, (b[1] + b[3]) / 2) for b, c in zip(pb, pc) if c in BATCLS]
        box, src = None, "none"
        # 1) striker detection confirmed by an overlapping person box
        for s in sorted(strikers, key=lambda s: -iou(s, prev) if prev is not None else 0):
            m = max(persons, key=lambda q: iou(s, q), default=None)
            if m is not None and iou(s, m) > 0.3:
                box, src = m, "striker+person"; break
        # 2) follow the same person as the previous frame
        if box is None and prev is not None and persons:
            m = max(persons, key=lambda q: iou(prev, q))
            if iou(prev, m) > 0.3:
                box, src = m, "track"
        # 3) first frames: the person holding the bat, else the upper-middle person
        if box is None and prev is None and persons:
            def score(q):
                cx, cy = (q[0] + q[2]) / 2 / W, (q[1] + q[3]) / 2 / H
                holds = any(q[0] - 0.2 * (q[2] - q[0]) <= bx <= q[2] + 0.2 * (q[2] - q[0]) and q[1] <= by <= q[3] for bx, by in bats)
                return abs(cx - 0.5) + abs(cy - 0.45) - (0.5 if holds else 0)
            box, src = min(persons, key=score), "bat/centre"
        out.append(None if box is None else np.asarray(box, float)); how.append(src)
        if box is not None:
            prev = np.asarray(box, float)
    return out, how


def vitpose(frames, boxes):
    k = np.full((len(frames), 17, 3), np.nan, np.float32)
    idx = [i for i, b in enumerate(boxes) if b is not None]
    for s in range(0, len(idx), 48):
        ch = idx[s:s + 48]
        bx = [[[boxes[i][0], boxes[i][1], boxes[i][2] - boxes[i][0], boxes[i][3] - boxes[i][1]]] for i in ch]
        inp = proc(images=[cv2.cvtColor(frames[i], cv2.COLOR_BGR2RGB) for i in ch], boxes=bx, return_tensors="pt").to(DEV)
        if DEV == "cuda":
            inp["pixel_values"] = inp["pixel_values"].half()
        with torch.no_grad():
            o = vit(**inp)
        o.heatmaps = o.heatmaps.float()  # post-processing smooths heatmaps with scipy, which rejects float16
        for i, r in zip(ch, proc.post_process_pose_estimation(o, boxes=bx)):
            k[i, :, :2] = r[0]["keypoints"].float().cpu().numpy(); k[i, :, 2] = r[0]["scores"].float().cpu().numpy()
    return k


df = pd.read_csv(f"{IN}/pilot.csv")
rows, t_all, n_all = [], 0.0, 0
for n, r in enumerate(df.itertuples()):
    try:
        fr = read(glob.glob(f"{IN}/**/{r.clip_id}.mp4", recursive=True)[0])
        if len(fr) < 3:
            continue
        H = fr[0].shape[0]
        t = time.time(); boxes, how = batter_boxes(fr); k = vitpose(fr, boxes); t_all += time.time() - t; n_all += len(fr)
        valid = [b for b in boxes if b is not None]
        bad = [not plausible(b, H) for b in valid]
        hs = np.array([b[3] - b[1] if b is not None else np.nan for b in boxes])
        a = np.linalg.norm(k[2:, :, :2] - 2 * k[1:-1, :, :2] + k[:-2, :, :2], axis=-1) / hs[1:-1, None]
        row = {"clip_id": r.clip_id, "source": r.source, "frames": len(fr),
               "found": len(valid) / len(fr), "scoreboard_like": float(np.mean(bad)) if bad else 0.0,
               "via_striker": float(np.mean([h == "striker+person" for h in how])),
               "jitter": float(np.nanmedian(a)) if np.isfinite(a).any() else np.nan,
               "conf": float(np.nanmean(k[:, :, 2])) if len(valid) else np.nan}
        if isinstance(r.execution_bbox, str) and valid:  # CricketVision box, best source scale per clip
            x, y, w, h = json.loads(r.execution_bbox)
            row["cv_iou"] = max(iou(b, [x * H / sh, y * H / sh, (x + w) * H / sh, (y + h) * H / sh])
                                for sh in (360, 480, 720, 1080) for b in valid)
        np.savez_compressed(f"{OUT}/kps/{r.clip_id}.npz", vit=k,
                            boxes=np.array([b if b is not None else [np.nan] * 4 for b in boxes], np.float32))
        if n % 12 == 0:
            for j in (len(fr) // 4, len(fr) // 2, 3 * len(fr) // 4):
                g = fr[j].copy(); b = boxes[j]
                if b is not None:
                    cv2.rectangle(g, tuple(map(int, b[:2])), tuple(map(int, b[2:])), (0, 255, 255), 1)
                    for u, v in SKEL:
                        if min(k[j, u, 2], k[j, v, 2]) > 0.3:
                            cv2.line(g, tuple(map(int, k[j, u, :2])), tuple(map(int, k[j, v, :2])), (0, 0, 255), 2)
                cv2.imwrite(f"{OUT}/frames/{r.clip_id}_{j:03d}.jpg", g)
        rows.append(row)
        if n % 25 == 0:
            print(n, row, flush=True)
    except Exception:
        print("clip failed", r.clip_id, traceback.format_exc()[-400:], flush=True)

res = pd.DataFrame(rows); res.to_csv(f"{OUT}/per_clip.csv", index=False)
if res.empty:
    raise SystemExit("no clip succeeded - see 'clip failed' lines above")
summary = {
    "device": DEV, "clips": len(res), "frames": n_all, "fps_pipeline_fp16": n_all / t_all,
    "batter_found_rate": float(res.found.mean()), "scoreboard_like_rate": float(res.scoreboard_like.mean()),
    "via_striker_confirmed": float(res.via_striker.mean()),
    "cv_iou_median": float(res.cv_iou.median()), "cv_iou_gt_0.5": float((res.cv_iou > 0.5).mean()),
    "jitter_median": float(res.jitter.median()),
    "by_source": res.groupby("source")[["found", "scoreboard_like", "jitter"]].mean().round(3).to_dict(),
}
json.dump(summary, open(f"{OUT}/summary.json", "w"), indent=1)
print(json.dumps(summary, indent=1), flush=True)
