"""Pose re-run without box clamping (zero padding), for every clip whose padded batter box was clamped in any frame.

Why: the pose jobs padded the batter box by 12% and then clamped it to the frame. ViTPose's processor turns the box
into a 3:4 crop scaled by 1.25 around the box centre, and fills anything outside the image with black. Clamping
therefore moved the crop centre and shrank it whenever the batter was near an edge, which distorts the pose. Passing
the unclamped box gives true zero padding (D3: zero padding beats clamping in 97% of paired cases). Boxes, windows,
tracks and the batter choice are reused unchanged; only the joints are re-estimated. For frames that were never
clamped the result equals the old one, so after this job the whole dataset is as if pose ran without clamping.

Inputs : shikkoustic/criclens-all (clips/ + criclens-processed.tgz.bin, extracted to /tmp/proc)
Outputs: kps/<clip>.npz (old keys + new `kps`, old joints kept as `kps_clamp`), per_clip.csv (new window metrics
         next to the old ones), summary.json, kps.tar
Local smoke test (CPU):  CRICLENS_LOCAL=<repo root> CRICLENS_CLIPS=<dir with <source>/<clip>.mp4> CRICLENS_LIMIT=3 python pose_pad.py
"""
import glob, json, os, subprocess, sys, time, traceback

CHUNK = int(os.environ.get("CRICLENS_CHUNK", "0"))
NCHUNKS = int(os.environ.get("CRICLENS_NCHUNKS", "1"))
LIMIT = int(os.environ.get("CRICLENS_LIMIT", "0"))
LOCAL = os.environ.get("CRICLENS_LOCAL")
OUT = os.environ.get("CRICLENS_OUT", "/kaggle/working")
PROC = LOCAL or "/tmp/proc"
PAD, GAP = 0.12, 3
BONES = [(5, 7), (7, 9), (6, 8), (8, 10), (11, 13), (13, 15), (12, 14), (14, 16)]
SHARD = int(sys.argv[sys.argv.index("--shard") + 1]) if "--shard" in sys.argv else None

import numpy as np, pandas as pd


def clips_dir():
    if os.environ.get("CRICLENS_CLIPS"):
        return os.environ["CRICLENS_CLIPS"]
    return [c for c in glob.glob("/kaggle/input/**/clips", recursive=True) if os.path.isdir(f"{c}/cricketvision")][0]


def clamped_frames(boxes, W, H):
    """Frames whose 12%-padded box was clamped by the old code (x0/y0 < 0, x1 > W-1, y1 > H-1)."""
    ok = np.isfinite(boxes).all(1)
    pw, ph = PAD * (boxes[:, 2] - boxes[:, 0]), PAD * (boxes[:, 3] - boxes[:, 1])
    return ok & ((boxes[:, 0] - pw < 0) | (boxes[:, 1] - ph < 0) | (boxes[:, 2] + pw > W - 1) | (boxes[:, 3] + ph > H - 1))


def todo_list():
    """Every clip with at least one clamped frame, sorted, then this chunk's share."""
    idx = pd.read_parquet(f"{PROC}/data/processed/pose_index.parquet")
    q = pd.read_csv(f"{PROC}/data/processed/quality/quality_all.csv", usecols=["clip_id", "height", "width"])
    idx = idx.merge(q, on="clip_id").sort_values("clip_id").reset_index(drop=True)
    keep = []
    for r in idx.itertuples():
        z = np.load(f"{PROC}/{r.kps_path}")
        n = int(clamped_frames(z["boxes"], r.width, r.height).sum())
        if n:
            keep.append({"clip_id": r.clip_id, "source": r.source, "kps_path": r.kps_path, "W": r.width, "H": r.height,
                         "clamped_frames": n})
    d = pd.DataFrame(keep)
    return d[d.index % NCHUNKS == CHUNK].reset_index(drop=True)


def window_metrics(k, boxes, lo, hi):
    """Same definitions as the pose jobs, plus bone-length consistency (as in D3's real-edge comparison)."""
    kw, bw = k[lo:hi + 1], boxes[lo:hi + 1]
    hs = bw[:, 3] - bw[:, 1]
    found = float(np.isfinite(kw[:, 0, 0]).mean()) if len(kw) else 0.0
    conf = float(np.nanmean(kw[:, :, 2])) if np.isfinite(kw[:, :, 2]).any() else 0.0
    if len(kw) > 2:
        a = np.linalg.norm(kw[2:, :, :2] - 2 * kw[1:-1, :, :2] + kw[:-2, :, :2], axis=-1) / hs[1:-1, None]
        jit = float(np.nanmedian(a)) if np.isfinite(a).any() else np.nan
    else:
        jit = np.nan
    L = np.stack([np.linalg.norm(kw[:, p, :2] - kw[:, q, :2], axis=-1) for p, q in BONES], 1) / hs[:, None]
    with np.errstate(all="ignore"):
        bone_cv = float(np.nanmean(np.nanstd(L, 0) / (np.nanmean(L, 0) + 1e-6)))
    return {"found_in_window": found, "conf_in_window": conf, "jitter": jit, "bone_cv": bone_cv,
            "usable": bool(found >= 0.8 and conf >= 0.5)}


if SHARD is None and not LOCAL:  # main process on Kaggle: unpack data, pick clips, one worker per GPU, merge
    subprocess.run([sys.executable, "-m", "pip", "install", "-q", "-U", "transformers>=4.48", "accelerate"], check=False)
    tgz = glob.glob("/kaggle/input/**/criclens-processed.tgz.bin", recursive=True)[0]
    os.makedirs(PROC, exist_ok=True)
    subprocess.run(["tar", "-xzf", tgz, "-C", PROC, "--wildcards", "data/processed/*", "kaggle/chunks/pose-*/out/kps/*"], check=True)
    todo = todo_list()
    if LIMIT:
        todo = todo.head(LIMIT)
    todo.to_csv("/tmp/todo.csv", index=False)
    print(f"chunk {CHUNK}/{NCHUNKS}: {len(todo)} clips, {int(todo.clamped_frames.sum())} clamped frames", flush=True)
    import torch as _t
    ngpu = max(1, _t.cuda.device_count())
    procs = [subprocess.Popen([sys.executable, os.path.abspath(__file__), "--shard", str(g), "--of", str(ngpu)],
                              env={**os.environ, "CUDA_VISIBLE_DEVICES": str(g)}) for g in range(ngpu)]
    for p_ in procs:
        p_.wait()
    parts = glob.glob(f"{OUT}/per_clip_*.csv")
    if not parts:
        for f in glob.glob(f"{OUT}/log_*.txt"):
            print(open(f).read()[-3000:])
        sys.exit("no worker produced results")
    res = pd.concat([pd.read_csv(f) for f in parts], ignore_index=True).sort_values("clip_id")
    res.to_csv(f"{OUT}/per_clip.csv", index=False)
    for f in parts:
        os.remove(f)
    failed = sum(open(f).read().count("clip failed") for f in glob.glob(f"{OUT}/log_*.txt"))
    summ = {"chunk": CHUNK, "clips": len(res), "planned": len(todo), "failed": failed,
            "usable_old": float(res.old_usable.mean()), "usable_new": float(res.usable.mean()),
            "became_usable": int((res.usable & ~res.old_usable).sum()), "lost_usable": int((~res.usable & res.old_usable).sum())}
    for m in ("conf_in_window", "jitter", "bone_cv"):
        summ[f"{m}_old_mean"] = float(res[f"old_{m}"].mean()); summ[f"{m}_new_mean"] = float(res[m].mean())
    summ["conf_improved_share"] = float((res.conf_in_window > res.old_conf_in_window).mean())
    summ["bone_cv_improved_share"] = float((res.bone_cv < res.old_bone_cv).mean())
    json.dump(summ, open(f"{OUT}/summary.json", "w"), indent=1)
    print(json.dumps(summ, indent=1), flush=True)
    subprocess.run(["tar", "-cf", f"{OUT}/kps.tar", "-C", OUT, "kps"], check=True)
    import shutil as _sh
    _sh.rmtree(f"{OUT}/kps")
    sys.exit(0)

# ---- worker (on Kaggle one per GPU; locally a single CPU worker) ----
import cv2, torch
from transformers import AutoProcessor, VitPoseForPoseEstimation

NOF = int(sys.argv[sys.argv.index("--of") + 1]) if "--of" in sys.argv else 1
SHARD = SHARD or 0
os.makedirs(f"{OUT}/kps", exist_ok=True)
LOG = open(f"{OUT}/log_{SHARD}.txt", "a")
DEV = "cuda" if torch.cuda.is_available() else "cpu"
VIT = "usyd-community/vitpose-base-simple"
proc = AutoProcessor.from_pretrained(VIT)
vit = VitPoseForPoseEstimation.from_pretrained(VIT, torch_dtype=torch.float16 if DEV == "cuda" else torch.float32).to(DEV).eval()
CLIPS = clips_dir()


def read(path):
    cap = cv2.VideoCapture(path); fr = []
    while True:
        ok, f = cap.read()
        if not ok:
            break
        fr.append(f)
    cap.release()
    return fr


def vitpose_unclamped(frames, boxes):
    """The pose jobs' vitpose() with the clamp removed: the processor zero-fills outside the image."""
    k = np.full((len(frames), 17, 3), np.nan, np.float32)
    idx = [i for i in range(min(len(frames), len(boxes))) if np.isfinite(boxes[i, 0])]
    for s in range(0, len(idx), 48):
        ch = idx[s:s + 48]; bx = []
        for i in ch:
            x0, y0, x1, y1 = boxes[i]; pw, ph = PAD * (x1 - x0), PAD * (y1 - y0)
            x0, y0, x1, y1 = x0 - pw, y0 - ph, x1 + pw, y1 + ph
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


if LOCAL:
    todo = todo_list()
    todo = todo[todo.apply(lambda r: os.path.exists(f"{CLIPS}/{r.source}/{r.clip_id}.mp4"), axis=1)]
    todo = todo.head(LIMIT) if LIMIT else todo
else:
    todo = pd.read_csv("/tmp/todo.csv")
todo = todo[todo.index % NOF == SHARD]
rows, t0, nf = [], time.time(), 0
for n, r in enumerate(todo.itertuples()):
    try:
        z = dict(np.load(f"{PROC}/{r.kps_path}"))
        boxes = z["boxes"]; lo, c, hi = [int(v) for v in z["window"]]
        fr = read(f"{CLIPS}/{r.source}/{r.clip_id}.mp4")
        k = vitpose_unclamped(fr, boxes)
        if len(k) < len(boxes):  # decoder returned fewer frames than the original run: keep the length
            k = np.concatenate([k, np.full((len(boxes) - len(k), 17, 3), np.nan, np.float32)])
        k = k[:len(boxes)]
        for j in range(17):  # interpolate short joint gaps, as the pose jobs do
            for ch_ in range(3):
                k[:, j, ch_] = pd.Series(k[:, j, ch_]).interpolate(limit=GAP, limit_area="inside").to_numpy()
        new, old = window_metrics(k, boxes, lo, hi), window_metrics(z["kps"], boxes, lo, hi)
        cl = clamped_frames(boxes, r.W, r.H)
        same = ~cl & np.isfinite(k[:, 0, 0]) & np.isfinite(z["kps"][:, 0, 0])
        drift = float(np.nanmax(np.abs(k[same, :, :2] - z["kps"][same, :, :2]))) if same.any() else np.nan
        np.savez_compressed(f"{OUT}/kps/{r.clip_id}.npz", **{**z, "kps": k, "kps_clamp": z["kps"], "padding": "zero"})
        rows.append({"clip_id": r.clip_id, "source": r.source, "clamped_frames": int(cl.sum()), "frames": len(boxes),
                     "unclamped_max_drift_px": drift, **new, **{f"old_{m}": v for m, v in old.items()}})
        nf += len(fr)
        if n % 50 == 0:
            print(f"[{SHARD}] {n}/{len(todo)} {nf / (time.time() - t0):.1f} fps", rows[-1], file=LOG, flush=True)
    except Exception:
        print("clip failed", r.clip_id, traceback.format_exc()[-800:], file=LOG, flush=True)
pd.DataFrame(rows).to_csv(f"{OUT}/per_clip_{SHARD}.csv", index=False)
print(f"worker {SHARD} done: {len(rows)} clips, {nf} frames, {nf / max(1e-9, time.time() - t0):.1f} fps", file=LOG, flush=True)
if LOCAL:
    print(pd.DataFrame(rows).T.to_string())
