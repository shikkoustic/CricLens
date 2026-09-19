"""D3: robustness to deliberate degradation (chunk 0) and padding at the frame edge (chunk 1).

Chunk 0 - clean clips are degraded on purpose, then restored with the matching lab filter. The pose on the clean
crop is the reference, so the real joint error is measurable (normalised by batter height), next to PSNR/SSIM.
Chunk 1 - (a) synthetic: a frame edge is placed through a batter who is fully in view (right edge / bottom edge,
two severities); crops are made by clamping at the edge (current code), zero padding, replicate padding or reflect
padding; error vs the uncut reference on the joints still inside the frame. (b) real clips whose batter box already
leaves the frame: no reference, so confidence, jitter and bone-length consistency are compared.
"""
PART = int(os.environ.get("CRICLENS_CHUNK", "0"))
LIMIT = int(os.environ.get("CRICLENS_LIMIT", "0"))
os.makedirs(f"{OUT}/examples", exist_ok=True)
from skimage.metrics import structural_similarity, peak_signal_noise_ratio
BONES = [(5, 7), (7, 9), (6, 8), (8, 10), (11, 13), (13, 15), (12, 14), (14, 16)]


# ---------------- degradations (Labs 2-4) ----------------
def gauss_noise(x, s, rng): return np.clip(x.astype(np.float32) + rng.normal(0, s, x.shape), 0, 255).astype(np.uint8)
def salt_pepper(x, p, rng):
    y = x.copy(); m = rng.random(x.shape[:2]); y[m < p / 2] = 0; y[m > 1 - p / 2] = 255; return y
def shot_noise(x, peak, rng): return np.clip(rng.poisson(x.astype(np.float32) / 255 * peak) / peak * 255, 0, 255).astype(np.uint8)
def dark(x): return (255 * np.power(x / 255.0, 2.2)).astype(np.uint8)
def low_contrast(x): return np.clip(127.5 + 0.35 * (x.astype(np.float32) - 127.5), 0, 255).astype(np.uint8)
def jpeg(x, q): return cv2.imdecode(cv2.imencode(".jpg", x, [cv2.IMWRITE_JPEG_QUALITY, q])[1], cv2.IMREAD_COLOR)


# ---------------- restorations (Labs 2-4) ----------------
def adaptive_median(x, smax=7):
    """Lab 4 adaptive median: grow the window until the median is not an impulse, keep the pixel if it is not one."""
    out = np.empty_like(x)
    for ch in range(x.shape[2]):
        z = x[..., ch]; res = cv2.medianBlur(z, smax); done = np.zeros(z.shape, bool)
        for s in range(3, smax + 1, 2):
            k = np.ones((s, s), np.uint8)
            zmed, zmin, zmax = cv2.medianBlur(z, s), cv2.erode(z, k), cv2.dilate(z, k)
            ok = (zmed > zmin) & (zmed < zmax) & ~done
            keep = (z > zmin) & (z < zmax)
            res[ok & keep] = z[ok & keep]; res[ok & ~keep] = zmed[ok & ~keep]; done |= ok
        out[..., ch] = res
    return out
def gauss_smooth(x, s=1.0): return cv2.GaussianBlur(x, (0, 0), s)
def nlmeans(x, h=12): return cv2.fastNlMeansDenoisingColored(x, None, h, h, 7, 21)
def unsharp(x, s=1.5, a=1.0): return cv2.addWeighted(x, 1 + a, cv2.GaussianBlur(x, (0, 0), s), -a, 0)
def inv_gamma(x): return (255 * np.power(x / 255.0, 1 / 2.2)).astype(np.uint8)
def stretch(L):
    lo, hi = np.percentile(L, (2, 98)); return (L.astype(np.float32) - lo) * 255.0 / max(hi - lo, 1)
def local_gamma(L):
    il = cv2.GaussianBlur(L.astype(np.float32), (0, 0), max(L.shape) / 30); return 255.0 * np.power(L / 255.0, np.power(2.0, (128.0 - il) / 128.0))

PLAN = [  # (degradation name, fn, [(restoration name, fn), ...])
    ("gauss_noise_10", lambda x, r: gauss_noise(x, 10, r), []),
    ("gauss_noise_25", lambda x, r: gauss_noise(x, 25, r), [("gauss_smooth", gauss_smooth), ("median3", lambda x: cv2.medianBlur(x, 3)), ("nlmeans", nlmeans)]),
    ("salt_pepper_2", lambda x, r: salt_pepper(x, 0.02, r), []),
    ("salt_pepper_5", lambda x, r: salt_pepper(x, 0.05, r), [("median3", lambda x: cv2.medianBlur(x, 3)), ("adaptive_median", adaptive_median)]),
    ("shot_noise", lambda x, r: shot_noise(x, 30, r), [("gauss_smooth", gauss_smooth), ("nlmeans", nlmeans)]),
    ("blur_1.5", lambda x, r: cv2.GaussianBlur(x, (0, 0), 1.5), []),
    ("blur_3", lambda x, r: cv2.GaussianBlur(x, (0, 0), 3), [("unsharp", unsharp)]),
    ("dark_gamma2.2", lambda x, r: dark(x), [("inverse_gamma", inv_gamma), ("clahe", lambda x: on_luma(x, CLAHE.apply)), ("local_gamma", lambda x: on_luma(x, local_gamma))]),
    ("low_contrast", lambda x, r: low_contrast(x), [("stretch", lambda x: on_luma(x, stretch)), ("clahe", lambda x: on_luma(x, CLAHE.apply)), ("global_he", lambda x: on_luma(x, cv2.equalizeHist))]),
    ("jpeg_q15", lambda x, r: jpeg(x, 15), [("median3", lambda x: cv2.medianBlur(x, 3))]),
]


def compare(k, ref, heights):
    vis = ref[:, :, 2] > 0.3
    err = np.linalg.norm(k[:, :, :2] - ref[:, :, :2], axis=-1) / np.asarray(heights)[:, None]
    e = err[vis]
    return {"err": float(np.nanmean(e)), "pck05": float(np.nanmean(e < 0.05)), "pck10": float(np.nanmean(e < 0.10)),
            "conf": float(np.nanmean(k[:, :, 2])), "jitter": jitter(k, heights)}


def img_quality(clean, other):
    idx = np.linspace(0, len(clean) - 1, 3).astype(int); ps, ss = [], []
    for i in idx:
        a, b = cv2.cvtColor(clean[i], cv2.COLOR_BGR2GRAY), cv2.cvtColor(other[i], cv2.COLOR_BGR2GRAY)
        ps.append(99.0 if np.array_equal(a, b) else peak_signal_noise_ratio(a, b, data_range=255))
        ss.append(structural_similarity(a, b, data_range=255) if min(a.shape) >= 7 else np.nan)
    return {"psnr": float(np.mean(ps)), "ssim": float(np.nanmean(ss))}


def window(z):
    lo, c, hi = [int(v) for v in z["window"]]; return max(lo, c - 20), min(hi, c + 15)


samples = pd.read_csv(f"{D3}/samples.csv")
rows, t0 = [], time.time()
if PART == 0:
    samples = samples[samples.part == "degrade"]
    if LIMIT: samples = samples.head(LIMIT)
    for n, s in enumerate(samples.itertuples()):
        try:
            z = npz_for(s.clip_id); boxes = z["boxes"]; a, b = window(z)
            fr = read_frames(s.source, s.clip_id, a, b); H, W = fr[0].shape[:2]
            idx = [i for i in range(len(fr)) if np.isfinite(boxes[a + i]).all()]
            crops = []
            for i in idx:
                x0, y0, x1, y1 = padded(boxes[a + i], H, W); crops.append(fr[i][y0:y1, x0:x1])
            if len(crops) < 5: continue
            hts = [c.shape[0] / (1 + 2 * PAD) for c in crops]
            ref = pose_crops(crops)
            rows.append({"clip_id": s.clip_id, "source": s.source, "degradation": "clean", "restoration": "-", **compare(ref, ref, hts), "psnr": 99.0, "ssim": 1.0})
            import zlib; rng = np.random.default_rng(zlib.crc32(s.clip_id.encode()))
            for dname, dfn, rest in PLAN:
                deg = [dfn(c, rng) for c in crops]
                rows.append({"clip_id": s.clip_id, "source": s.source, "degradation": dname, "restoration": "none",
                             **compare(pose_crops(deg), ref, hts), **img_quality(crops, deg)})
                for rname, rfn in rest:
                    rec = [rfn(c) for c in deg]
                    rows.append({"clip_id": s.clip_id, "source": s.source, "degradation": dname, "restoration": rname,
                                 **compare(pose_crops(rec), ref, hts), **img_quality(crops, rec)})
                    if n < 3:
                        m = len(crops) // 2
                        cv2.imwrite(f"{OUT}/examples/{n}_{dname}_{rname}.jpg", np.hstack([crops[m], deg[m], rec[m]]))
            if n % 20 == 0: print(f"{n}/{len(samples)} {time.time() - t0:.0f}s", flush=True)
        except Exception:
            print("clip failed", s.clip_id, traceback.format_exc()[-500:], flush=True)
    d = pd.DataFrame(rows); d.to_csv(f"{OUT}/d3_degrade.csv", index=False)
    g = d.groupby(["degradation", "restoration"])[["err", "pck05", "pck10", "conf", "jitter", "psnr", "ssim"]].mean().round(4)
    summary = {"part": "degrade", "clips": int(d.clip_id.nunique()), "seconds": time.time() - t0,
               "table": {f"{k[0]}|{k[1]}": v for k, v in g.to_dict("index").items()}}
else:
    STRATS = {"clamp": None, "zero": cv2.BORDER_CONSTANT, "replicate": cv2.BORDER_REPLICATE, "reflect": cv2.BORDER_REFLECT_101}

    def crop_with(frame, box, strat):
        H, W = frame.shape[:2]; x0, y0, x1, y1 = padded(box)
        if strat == "clamp":
            cx0, cy0, cx1, cy1 = max(0, x0), max(0, y0), min(W, x1), min(H, y1)
            return frame[cy0:cy1, cx0:cx1], (cx0, cy0)
        t, l, bt, r = max(0, -y0), max(0, -x0), max(0, y1 - H), max(0, x1 - W)
        big = cv2.copyMakeBorder(frame, t, bt, l, r, STRATS[strat], value=(0, 0, 0))
        return big[y0 + t:y1 + t, x0 + l:x1 + l], (x0, y0)

    def run_strats(frames, bxs):
        res = {}
        for st in STRATS:
            cs, offs = zip(*[crop_with(f, b, st) for f, b in zip(frames, bxs)])
            k = pose_crops(list(cs))
            for j, (ox, oy) in enumerate(offs):
                k[j, :, 0] += ox; k[j, :, 1] += oy
            res[st] = k
        return res

    syn = samples[samples.part == "degrade"]; real = samples[samples.part == "edge"]
    if LIMIT: syn, real = syn.head(LIMIT), real.head(LIMIT)
    for n, s in enumerate(syn.itertuples()):
        try:
            z = npz_for(s.clip_id); boxes = z["boxes"]; a, b = window(z)
            fr = read_frames(s.source, s.clip_id, a, b); idx = [i for i in range(len(fr)) if np.isfinite(boxes[a + i]).all()]
            if len(idx) < 5: continue
            frames = [fr[i] for i in idx]; bxs = [boxes[a + i] for i in idx]; hts = [bb[3] - bb[1] for bb in bxs]
            ref = run_strats(frames, bxs)["clamp"]          # batter fully inside: every strategy is identical here
            for edge in ("right", "bottom"):
                for frac in (0.85, 0.70):
                    cut_frames, vis_mask = [], []
                    for f, bb, rk in zip(frames, bxs, ref):
                        x0, y0, x1, y1 = padded(bb)
                        if edge == "right":
                            cut = int(x0 + frac * (x1 - x0)); cut_frames.append(f[:, :cut]); vis_mask.append(rk[:, 0] < cut)
                        else:
                            cut = int(y0 + frac * (y1 - y0)); cut_frames.append(f[:cut]); vis_mask.append(rk[:, 1] < cut)
                    vis = np.array(vis_mask) & (ref[:, :, 2] > 0.3)
                    for st, k in run_strats(cut_frames, bxs).items():
                        err = np.linalg.norm(k[:, :, :2] - ref[:, :, :2], axis=-1) / np.asarray(hts)[:, None]
                        hidden = ~np.array(vis_mask) & (ref[:, :, 2] > 0.3)
                        rows.append({"clip_id": s.clip_id, "case": "synthetic", "edge": edge, "cut_frac": frac, "strategy": st,
                                     "err_visible": float(np.nanmean(err[vis])) if vis.any() else np.nan,
                                     "pck05_visible": float(np.nanmean(err[vis] < 0.05)) if vis.any() else np.nan,
                                     "conf_visible": float(np.nanmean(k[:, :, 2][vis])) if vis.any() else np.nan,
                                     "conf_hidden": float(np.nanmean(k[:, :, 2][hidden])) if hidden.any() else np.nan,
                                     "jitter": jitter(k, hts)})
            if n < 2:
                m = len(frames) // 2; bb = bxs[m]; x0, y0, x1, y1 = padded(bb); cut = int(x0 + 0.7 * (x1 - x0))
                tiles = [crop_with(frames[m][:, :cut], bb, st)[0] for st in STRATS]
                hmax = max(t.shape[0] for t in tiles)
                cv2.imwrite(f"{OUT}/examples/padding_{n}.jpg", np.hstack([cv2.copyMakeBorder(t, 0, hmax - t.shape[0], 0, 6, cv2.BORDER_CONSTANT, value=(255, 255, 255)) for t in tiles]))
        except Exception:
            print("clip failed", s.clip_id, traceback.format_exc()[-500:], flush=True)
    for s in real.itertuples():
        try:
            z = npz_for(s.clip_id); boxes = z["boxes"]; lo, c, hi = [int(v) for v in z["window"]]
            fr = read_frames(s.source, s.clip_id, lo, hi); idx = [i for i in range(len(fr)) if np.isfinite(boxes[lo + i]).all()]
            if len(idx) < 5: continue
            frames = [fr[i] for i in idx]; bxs = [boxes[lo + i] for i in idx]; hts = [bb[3] - bb[1] for bb in bxs]
            for st, k in run_strats(frames, bxs).items():
                L = np.stack([np.linalg.norm(k[:, p, :2] - k[:, q, :2], axis=-1) for p, q in BONES], 1) / np.asarray(hts)[:, None]
                rows.append({"clip_id": s.clip_id, "case": "real_edge", "strategy": st, "conf_all": float(np.nanmean(k[:, :, 2])),
                             "jitter": jitter(k, hts), "bone_cv": float(np.nanmean(np.nanstd(L, 0) / (np.nanmean(L, 0) + 1e-6))),
                             "implausible_share": float(np.nanmean((L > 0.6).any(1)))})
        except Exception:
            print("clip failed", s.clip_id, traceback.format_exc()[-500:], flush=True)
    d = pd.DataFrame(rows); d.to_csv(f"{OUT}/d3_padding.csv", index=False)
    syn_t = d[d.case == "synthetic"].groupby(["edge", "cut_frac", "strategy"])[["err_visible", "pck05_visible", "conf_visible", "conf_hidden", "jitter"]].mean().round(4)
    real_t = d[d.case == "real_edge"].groupby("strategy")[["conf_all", "jitter", "bone_cv", "implausible_share"]].mean().round(4)
    summary = {"part": "padding", "synthetic_clips": int(d[d.case == "synthetic"].clip_id.nunique()), "real_clips": int(d[d.case == "real_edge"].clip_id.nunique()),
               "seconds": time.time() - t0, "synthetic": {f"{k[0]}|{k[1]}|{k[2]}": v for k, v in syn_t.to_dict("index").items()},
               "real": real_t.to_dict("index")}
json.dump(summary, open(f"{OUT}/summary.json", "w"), indent=1)
print(json.dumps(summary, indent=1)[:4000], flush=True)
