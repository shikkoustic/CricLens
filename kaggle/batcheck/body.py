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
