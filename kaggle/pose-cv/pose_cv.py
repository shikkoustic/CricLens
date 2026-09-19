"""CricLens pose extraction, CricketVision run (7,087 canonical clips), both T4 GPUs.

Same method as the v3 pilot (one batter per clip, ball release -> contact + 0.6 s, stop at camera
cuts, ViTPose-Base fp16). CricketVision's annotated batter box picks the batter track. Every
tracked person is also saved with its batter-cue features and whether it is the annotated batter:
that table trains the batter-finder used later for the ~15k clips that have no boxes.
The main process spawns one worker per GPU; each worker takes every 2nd clip.

Method notes (from v3):

1. camera cuts split the clip into segments (HSV-histogram jump between frames)
2. every person is tracked (YOLO11m + ByteTrack); each track gets a batter score from
   Striker detections + bat in hand (player-type model, stance frames), upright shape, position,
   size stability (the bowler grows as he runs at the camera), coverage; CricketVision clips add
   their annotated batter box. The best track is the batter for the whole clip.
3. ViTPose-Base on the batter box (padded 12%), fp16; gaps <= 3 frames interpolated
4. analysis window: contact = CricketVision execution end, else peak wrist speed; window =
   [contact - 0.8 s (~ball release), contact + 0.6 s], never crossing a camera cut
A clip is usable when >= 80% of window frames have the batter with mean joint confidence >= 0.5.
Pilot mode also reports agreement with CricketVision boxes for the choice made WITHOUT them.
"""
import glob, json, os, subprocess, sys, time, traceback
SHARD = int(sys.argv[sys.argv.index("--shard") + 1]) if "--shard" in sys.argv else None
if SHARD is None:  # main process: install once, launch one worker per GPU, merge results
    subprocess.run([sys.executable, "-m", "pip", "install", "-q", "ultralytics", "lap", "-U", "transformers>=4.48", "accelerate"], check=False)
    import torch as _t
    ngpu = max(1, _t.cuda.device_count())
    procs = [subprocess.Popen([sys.executable, os.path.abspath(__file__), "--shard", str(g), "--of", str(ngpu)],
                              env={**os.environ, "CUDA_VISIBLE_DEVICES": str(g)}) for g in range(ngpu)]
    for p_ in procs:
        p_.wait()
    import pandas as pd, json as _j, glob as _g
    parts = _g.glob("/kaggle/working/per_clip_*.csv")
    if not parts:
        for f in _g.glob("/kaggle/working/log_*.txt"):
            print(open(f).read()[-3000:])
        sys.exit("no worker produced results")
    res = pd.concat([pd.read_csv(f) for f in parts], ignore_index=True)
    trk = pd.concat([pd.read_csv(f) for f in _g.glob("/kaggle/working/tracks_*.csv")], ignore_index=True)
    res.to_csv("/kaggle/working/per_clip.csv", index=False); trk.to_csv("/kaggle/working/track_features.csv", index=False)
    summ = {"clips": len(res), "usable_rate": float(res.usable.mean()),
            "found_in_window": float(res.found_in_window.mean()), "conf_in_window": float(res.conf_in_window.mean()),
            "batter_correct_with_annotation": float((res.cv_iou_final > 0.5).mean()),
            "batter_correct_auto_rules": float((res.cv_iou_auto > 0.5).mean()),
            "tracks_saved": len(trk), "clips_with_positive_track": int(trk.groupby("clip_id").label.max().sum()),
            "failed": int(sum(open(f).read().count("clip failed") for f in _g.glob("/kaggle/working/log_*.txt")))}
    _j.dump(summ, open("/kaggle/working/summary.json", "w"), indent=1); print(_j.dumps(summ, indent=1))
    sys.exit(0)
NOF = int(sys.argv[sys.argv.index("--of") + 1])
LOG = open(f"/kaggle/working/log_{SHARD}.txt", "a")
import cv2, numpy as np, pandas as pd, torch
from ultralytics import YOLO
from transformers import AutoProcessor, VitPoseForPoseEstimation

PRE, FOLLOW, PAD, GAP = 0.8, 0.6, 0.12, 3
DEV = "cuda" if torch.cuda.is_available() else "cpu"
MANIFEST = glob.glob("/kaggle/input/**/manifest.parquet", recursive=True)[0]
CLIPDIR = glob.glob("/kaggle/input/**/clips/cricketvision", recursive=True)[0]
IN = glob.glob("/kaggle/input/**/Player_Type_Detection_Model.pt", recursive=True)[0].rsplit("/", 1)[0]
OUT = "/kaggle/working"
for d in ("frames", "kps"):
    os.makedirs(f"{OUT}/{d}", exist_ok=True)
ptype = YOLO(f"{IN}/Player_Type_Detection_Model.pt")
STRIKER = [i for i, n in ptype.names.items() if n.lower() in ("striker", "batsman", "batter")]
BATCLS = [i for i, n in ptype.names.items() if n.lower() == "bat"]
VIT = "usyd-community/vitpose-base-simple"
proc = AutoProcessor.from_pretrained(VIT)
vit = VitPoseForPoseEstimation.from_pretrained(VIT, torch_dtype=torch.float16 if DEV == "cuda" else torch.float32).to(DEV).eval()
G = dict(device=0 if DEV == "cuda" else "cpu", verbose=False, half=DEV == "cuda")
SKEL = [(5, 7), (7, 9), (6, 8), (8, 10), (5, 6), (5, 11), (6, 12), (11, 12), (11, 13), (13, 15), (12, 14), (14, 16), (0, 5), (0, 6)]
print("device", DEV, "| ptype", ptype.names, flush=True)


def iou(a, b):
    x0, y0, x1, y1 = max(a[0], b[0]), max(a[1], b[1]), min(a[2], b[2]), min(a[3], b[3])
    i = max(0, x1 - x0) * max(0, y1 - y0)
    return i / ((a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - i + 1e-9)


def read(path):
    cap = cv2.VideoCapture(path); fps = cap.get(cv2.CAP_PROP_FPS) or 25; fr = []
    while True:
        ok, f = cap.read()
        if not ok:
            break
        fr.append(f)
    cap.release()
    return fr, fps


def camera_cuts(frames):
    hs = []
    for f in frames:
        h = cv2.calcHist([cv2.cvtColor(f, cv2.COLOR_BGR2HSV)], [0, 1], None, [32, 32], [0, 180, 0, 256])
        hs.append(cv2.normalize(h, h).flatten())
    return [i for i in range(1, len(hs)) if cv2.compareHist(hs[i - 1], hs[i], cv2.HISTCMP_BHATTACHARYYA) > 0.5]


def track_people(frames):
    det = YOLO("yolo11m.pt")  # fresh tracker state per clip
    tracks = {}
    for i, f in enumerate(frames):
        r = det.track(f, persist=True, classes=[0], conf=0.2, tracker="bytetrack.yaml", **G)[0]
        if r.boxes.id is None:
            continue
        for b, t in zip(r.boxes.xyxy.cpu().numpy(), r.boxes.id.cpu().numpy().astype(int)):
            tracks.setdefault(int(t), {})[i] = b
    return tracks


def cue_hits(frames, idx):
    res = ptype.predict([frames[i] for i in idx], conf=0.25, **G)
    out = {}
    for i, r in zip(idx, res):
        pb, pc = r.boxes.xyxy.cpu().numpy(), r.boxes.cls.cpu().numpy().astype(int)
        out[i] = ([b for b, c in zip(pb, pc) if c in STRIKER], [((b[0] + b[2]) / 2, (b[1] + b[3]) / 2) for b, c in zip(pb, pc) if c in BATCLS])
    return out


def cv_box(r, H):
    """CricketVision annotated batter box at the execution keyframe; the source resolution is not
    stored, so each candidate height is returned and the caller keeps the best-matching one."""
    if not isinstance(getattr(r, "execution_bbox", None), str):
        return None
    x, y, w, h = json.loads(r.execution_bbox)
    return [[x * H / s, y * H / s, (x + w) * H / s, (y + h) * H / s] for s in (360, 480, 720, 1080)]


def choose(tracks, cues, H, W, n, ann=None):
    best, scores = None, {}
    for tid, fb in tracks.items():
        if len(fb) < max(3, 0.25 * n):
            continue
        fi = sorted(fb); B = np.array([fb[i] for i in fi]); w, h = B[:, 2] - B[:, 0], B[:, 3] - B[:, 1]
        cx, cy = np.median((B[:, 0] + B[:, 2]) / 2) / W, np.median((B[:, 1] + B[:, 3]) / 2) / H
        aspect = np.median(h / np.maximum(w, 1))
        ci = [i for i in fi if i in cues]
        striker = np.mean([any(iou(fb[i], s) > 0.3 for s in cues[i][0]) for i in ci]) if ci else 0
        bat = np.mean([any(fb[i][0] - 0.25 * (fb[i][2] - fb[i][0]) <= bx <= fb[i][2] + 0.25 * (fb[i][2] - fb[i][0])
                           and fb[i][1] <= by <= fb[i][3] for bx, by in cues[i][1]) for i in ci]) if ci else 0
        k = max(1, len(h) // 4)
        growth = abs(np.log(np.median(h[-k:]) / max(np.median(h[:k]), 1)))
        s = (2.0 * striker + 1.5 * bat + 0.5 * len(fb) / n - 0.6 * abs(cx - 0.5) - 0.4 * abs(cy - 0.45)
             - 0.6 * growth - (2.0 if aspect < 1.2 or cy > 0.82 else 0.0))
        if ann is not None:
            s += 4.0 * max(iou(fb[i], a) for i in fi for a in ann)
        scores[tid] = s
        if best is None or s > scores[best]:
            best = tid
    return best, scores


def track_features(clip_id, tracks, cues, H, W, n, ann):
    """One row per tracked person: cues a batter-finder can use, and label = is the annotated batter."""
    rows = []
    med = {t: np.median(np.array(list(fb.values())), axis=0) for t, fb in tracks.items() if fb}
    for tid, fb in tracks.items():
        fi = sorted(fb); B = np.array([fb[i] for i in fi]); w, h = B[:, 2] - B[:, 0], B[:, 3] - B[:, 1]
        cxs, cys = (B[:, 0] + B[:, 2]) / 2, (B[:, 1] + B[:, 3]) / 2
        k = max(1, len(h) // 4); ci = [i for i in fi if i in cues]
        mb = med[tid]; mw, mh = mb[2] - mb[0], mb[3] - mb[1]
        # a crouched person just above (behind) this one = wicketkeeper behind the striker
        keeper = any(o != tid and (m[3] - m[1]) < 1.3 * (m[2] - m[0]) and m[3] < mb[3] and abs((m[0] + m[2]) / 2 - (mb[0] + mb[2]) / 2) < 1.5 * mw
                     and (mb[1] - m[3]) < 1.0 * mh for o, m in med.items())
        rows.append({
            "clip_id": clip_id, "tid": tid, "n_tracks": len(tracks), "cover": len(fb) / n, "first": fi[0] / n,
            "cx": float(np.median(cxs)) / W, "cy": float(np.median(cys)) / H, "w": float(np.median(w)) / W, "h": float(np.median(h)) / H,
            "aspect": float(np.median(h / np.maximum(w, 1))), "growth": float(np.log(np.median(h[-k:]) / max(np.median(h[:k]), 1))),
            "move": float(np.mean(np.hypot(np.diff(cxs), np.diff(cys))) / max(np.median(h), 1)) if len(fi) > 1 else 0.0,
            "striker": float(np.mean([any(iou(fb[i], s_) > 0.3 for s_ in cues[i][0]) for i in ci])) if ci else 0.0,
            "bat": float(np.mean([any(fb[i][0] - 0.25 * (fb[i][2] - fb[i][0]) <= bx <= fb[i][2] + 0.25 * (fb[i][2] - fb[i][0])
                                      and fb[i][1] <= by <= fb[i][3] for bx, by in cues[i][1]) for i in ci])) if ci else 0.0,
            "size_rank": int(sorted((-(m[3] - m[1]) for m in med.values())).index(-mh)) if tid in med else -1,
            "keeper_behind": int(keeper),
            "ann_iou": float(max((iou(b, a) for b in fb.values() for a in ann), default=0.0)) if ann else np.nan})
    if ann and rows:
        best = max(rows, key=lambda r: r["ann_iou"])
        for r in rows:
            r["label"] = int(r is best and r["ann_iou"] > 0.5)
    return rows


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


def vitpose(frames, boxes):
    H, W = frames[0].shape[:2]
    k = np.full((len(frames), 17, 3), np.nan, np.float32)
    idx = [i for i in range(len(frames)) if np.isfinite(boxes[i, 0])]
    for s in range(0, len(idx), 48):
        ch = idx[s:s + 48]; bx = []
        for i in ch:
            x0, y0, x1, y1 = boxes[i]; pw, ph = PAD * (x1 - x0), PAD * (y1 - y0)
            x0, y0, x1, y1 = max(0, x0 - pw), max(0, y0 - ph), min(W - 1, x1 + pw), min(H - 1, y1 + ph)
            bx.append([[x0, y0, x1 - x0, y1 - y0]])
        inp = proc(images=[cv2.cvtColor(frames[i], cv2.COLOR_BGR2RGB) for i in ch], boxes=bx, return_tensors="pt").to(DEV)
        if DEV == "cuda":
            inp["pixel_values"] = inp["pixel_values"].half()
        with torch.no_grad():
            o = vit(**inp)
        o.heatmaps = o.heatmaps.float()
        for i, r in zip(ch, proc.post_process_pose_estimation(o, boxes=bx)):
            k[i, :, :2] = r[0]["keypoints"].float().cpu().numpy(); k[i, :, 2] = r[0]["scores"].float().cpu().numpy()
    return k


def window(k, boxes, fps, n, cuts, r):
    if isinstance(getattr(r, "execution_end", None), float) and np.isfinite(r.execution_end):
        contact, how = int(round((r.execution_end - r.clip_start) * fps)), "annotated"
    else:
        wr = k[:, [9, 10], :2]; hgt = boxes[:, 3] - boxes[:, 1]
        sp = np.nanmax(np.linalg.norm(np.diff(wr, axis=0), axis=-1), axis=1) / hgt[1:]
        sp = np.convolve(np.nan_to_num(sp), np.ones(3) / 3, mode="same")
        contact, how = (int(np.argmax(sp)) + 1 if np.isfinite(sp).any() and sp.max() > 0 else n // 2), "wrist-speed"
    contact = min(max(contact, 0), n - 1)
    lo, hi = max(0, contact - int(PRE * fps)), min(n - 1, contact + int(FOLLOW * fps))
    for c in cuts:  # never cross a camera cut
        if c <= contact:
            lo = max(lo, c)
        else:
            hi = min(hi, c - 1)
    return lo, contact, hi, how


def process(path, r):
    fr, fps = read(path); n = len(fr)
    if n < 3:
        return None
    H, W = fr[0].shape[:2]
    cuts = camera_cuts(fr); tracks = track_people(fr)
    stance = list(range(0, max(3, int(0.6 * n)), max(1, n // 12)))  # player-type cues on a sample of early frames
    cues = cue_hits(fr, stance)
    ann = cv_box(r, H)
    auto_tid, _ = choose(tracks, cues, H, W, n)                       # automatic choice (for honest eval)
    tid, scores = choose(tracks, cues, H, W, n, ann) if ann else (auto_tid, None)
    boxes = fill_boxes(tracks[tid], n) if tid is not None else np.full((n, 4), np.nan)
    k = vitpose(fr, boxes)
    for j in range(17):  # interpolate short joint gaps
        for c in range(3):
            s = pd.Series(k[:, j, c]); k[:, j, c] = s.interpolate(limit=GAP, limit_area="inside").to_numpy()
    lo, contact, hi, how = window(k, boxes, fps, n, cuts, r)
    win = slice(lo, hi + 1); kw = k[win]
    found = np.isfinite(kw[:, 0, 0]).mean() if len(kw) else 0.0
    conf = float(np.nanmean(kw[:, :, 2])) if np.isfinite(kw[:, :, 2]).any() else 0.0
    hs = boxes[win, 3] - boxes[win, 1]
    a = np.linalg.norm(kw[2:, :, :2] - 2 * kw[1:-1, :, :2] + kw[:-2, :, :2], axis=-1) / hs[1:-1, None] if len(kw) > 2 else np.array([np.nan])
    row = {"clip_id": r.clip_id, "source": r.source, "frames": n, "fps": fps, "cuts": len(cuts),
           "win_frames": hi - lo + 1, "win_sec": (hi - lo + 1) / fps, "contact_by": how,
           "found_in_window": float(found), "conf_in_window": conf,
           "jitter": float(np.nanmedian(a)) if np.isfinite(a).any() else np.nan,
           "batter_h_px": float(np.nanmedian(hs)) if np.isfinite(hs).any() else np.nan,
           "usable": bool(found >= 0.8 and conf >= 0.5)}
    if ann:
        for name, t in (("cv_iou_auto", auto_tid), ("cv_iou_final", tid)):
            row[name] = max((iou(b, a_) for b in tracks[t].values() for a_ in ann), default=0.0) if t is not None else 0.0
    tids = np.array([t for t in tracks for _ in tracks[t]], np.int32)
    tfr = np.array([i for t in tracks for i in tracks[t]], np.int32)
    tbx = np.array([tracks[t][i] for t in tracks for i in tracks[t]], np.float32).reshape(-1, 4)
    np.savez_compressed(f"{OUT}/kps/{r.clip_id}.npz", kps=k, boxes=boxes.astype(np.float32),
                        window=np.array([lo, contact, hi]), fps=fps, chosen_tid=-1 if tid is None else tid,
                        track_ids=tids, track_frames=tfr, track_boxes=tbx)
    row["_tracks"] = track_features(r.clip_id, tracks, cues, H, W, n, ann)
    return row, fr, k, boxes, (lo, contact, hi)


def draw(f, k, b, label):
    g = f.copy()
    if np.isfinite(b).all():
        cv2.rectangle(g, tuple(map(int, b[:2])), tuple(map(int, b[2:])), (0, 255, 255), 1)
        for u, v in SKEL:
            if min(k[u, 2], k[v, 2]) > 0.3:
                cv2.line(g, tuple(map(int, k[u, :2])), tuple(map(int, k[v, :2])), (0, 0, 255), 2)
    cv2.putText(g, label, (6, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2)
    return g


m = pd.read_parquet(MANIFEST)
CHUNK, NCHUNKS = int(os.environ.get("CRICLENS_CHUNK", "0")), int(os.environ.get("CRICLENS_NCHUNKS", "1"))
df = m[(m.source == "cricketvision") & m.is_canonical].sort_values("clip_id").iloc[CHUNK::NCHUNKS].iloc[SHARD::NOF]
LIMIT = int(os.environ.get("CRICLENS_LIMIT", "0"))  # smoke test: only the first N clips per worker
if LIMIT:
    df = df.head(LIMIT)
df["bbox_json"] = df["execution_bbox"].map(lambda v: json.dumps([float(x) for x in v]) if v is not None and not isinstance(v, float) else None)
df = df.drop(columns=["execution_bbox"]).rename(columns={"bbox_json": "execution_bbox"})
rows, trows, t0, nf = [], [], time.time(), 0
print(f"worker {SHARD}/{NOF}: {len(df)} clips on", torch.cuda.get_device_name(0) if DEV == "cuda" else "cpu", flush=True)
for n_, r in enumerate(df.itertuples()):
    if time.time() - t0 > 10.5 * 3600:
        print("time budget reached, stopping", flush=True); break
    try:
        out = process(f"{CLIPDIR}/{r.clip_id}.mp4", r)
        if out is None:
            continue
        row, fr, k, boxes, (lo, c, hi) = out; trows += row.pop("_tracks"); rows.append(row); nf += len(fr)
        if n_ % 150 == 0:
            cv2.imwrite(f"{OUT}/frames/{r.clip_id}.jpg", np.hstack([draw(fr[i], k[i], boxes[i], lab)
                        for i, lab in ((lo, "window start"), (c, "contact"), (hi, "window end"))]))
        if n_ % 100 == 0:
            print(f"[{SHARD}] {n_}/{len(df)} {nf/(time.time()-t0):.1f} fps", {k_: (round(v, 3) if isinstance(v, float) else v) for k_, v in row.items()}, flush=True)
    except Exception:
        LOG.write(f"clip failed {r.clip_id}\n{traceback.format_exc()[-600:]}\n"); LOG.flush()
pd.DataFrame(rows).to_csv(f"{OUT}/per_clip_{SHARD}.csv", index=False)
pd.DataFrame(trows).to_csv(f"{OUT}/tracks_{SHARD}.csv", index=False)
print(f"worker {SHARD} done: {len(rows)} clips, {nf} frames, {nf/(time.time()-t0):.1f} fps", flush=True)
