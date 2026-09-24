"""Bat segmentation U-Net + bat angle from shape moments (PROGRESS.md's next step).

Why a U-Net and not another YOLO head: the ball/bat/stumps detector (kaggle/train-detector) already
gives boxes, but a coaching-relevant "bat angle" (is the swing coming down straight, is the face
open/closed) needs the bat's actual silhouette, not a box -- shape moments only make sense on a mask.
data/processed/detection/seg/ (5,999 polygon-labelled images, Roboflow `powerinflow` +
`cricket-ball-segmentation`) has bat+ball polygons in YOLO-seg format; this script rasterises the bat
polygons into binary masks and trains a small U-Net (pretrained ResNet18 encoder, small dataset) to
predict them. Bat angle: cv2.moments() on the predicted mask -> orientation from the second-order
central moments (standard shape-moments formula, docs/iva/syllabus_alignment.md's "Shape moments" row).

Output: /kaggle/working/unet_best.pt, metrics.json (val/test IoU, Dice), sample overlay frames.
"""
import glob, json, os, subprocess, sys, time

subprocess.run([sys.executable, "-m", "pip", "install", "-q", "segmentation-models-pytorch"], check=False)
import cv2
import numpy as np
import segmentation_models_pytorch as smp
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

LIMIT = int(os.environ.get("CRICLENS_LIMIT", "0"))
OUT = "/kaggle/working"
os.makedirs(f"{OUT}/overlays", exist_ok=True)
DEV = "cuda" if torch.cuda.is_available() else "cpu"
IMG = 384
BAT_CLASS = 1  # datasets/build_detection.py: classes ball=0, bat=1, stumps=2 (stumps not in seg/)

SEG_ROOT = glob.glob("/kaggle/input/**/seg/images/train", recursive=True)[0].rsplit("/images/", 1)[0]


def list_split(split):
    imgs = sorted(glob.glob(f"{SEG_ROOT}/images/{split}/*"))
    if LIMIT:
        # shuffle before truncating for a smoke test: the sources are alphabetically clustered (e.g.
        # cricket-ball-segmentation sorts first and is ball-only close-ups, no bat at all), so a plain
        # head-slice can accidentally sample a batch with zero bat ground truth
        rng = np.random.RandomState(0)
        imgs = list(rng.choice(imgs, size=min(LIMIT, len(imgs)), replace=False))
    return imgs


def label_path(img_path):
    return img_path.replace("/images/", "/labels/").rsplit(".", 1)[0] + ".txt"


def load_mask(img_path, h, w):
    mask = np.zeros((h, w), np.uint8)
    lp = label_path(img_path)
    if not os.path.exists(lp):
        return mask
    for line in open(lp):
        parts = line.split()
        if not parts or int(float(parts[0])) != BAT_CLASS:
            continue
        pts = np.array(parts[1:], np.float32).reshape(-1, 2)
        pts[:, 0] *= w
        pts[:, 1] *= h
        cv2.fillPoly(mask, [pts.astype(np.int32)], 1)
    return mask


class SegDataset(Dataset):
    def __init__(self, img_paths):
        self.paths = img_paths

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, i):
        p = self.paths[i]
        img = cv2.imread(p)
        h, w = img.shape[:2]
        mask = load_mask(p, h, w)
        img = cv2.resize(img, (IMG, IMG))
        mask = cv2.resize(mask, (IMG, IMG), interpolation=cv2.INTER_NEAREST)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        img = (img - np.array([0.485, 0.456, 0.406])) / np.array([0.229, 0.224, 0.225])
        x = torch.from_numpy(img.transpose(2, 0, 1)).float()
        y = torch.from_numpy(mask[None]).float()
        return x, y


def dice_iou(pred, target, eps=1e-6):
    """Thresholded IoU/Dice for reporting -- NOT differentiable (the >0.5 breaks the autograd graph),
    use soft_dice_loss for training."""
    pred = (pred > 0.5).float()
    inter = (pred * target).sum((1, 2, 3))
    union = pred.sum((1, 2, 3)) + target.sum((1, 2, 3)) - inter
    iou = (inter + eps) / (union + eps)
    dice = (2 * inter + eps) / (pred.sum((1, 2, 3)) + target.sum((1, 2, 3)) + eps)
    return iou.mean().item(), dice.mean().item()


def soft_dice_loss(prob, target, eps=1e-6):
    """Differentiable Dice on raw sigmoid probabilities (no thresholding), for backprop."""
    inter = (prob * target).sum((1, 2, 3))
    denom = prob.sum((1, 2, 3)) + target.sum((1, 2, 3))
    return 1 - ((2 * inter + eps) / (denom + eps)).mean()


def bat_angle(mask: np.ndarray) -> float | None:
    """Orientation of the bat's long axis in degrees, from the mask's second-order central moments
    (standard shape-moments formula: theta = 0.5*atan2(2*mu11, mu20-mu02)). None if the mask is empty."""
    m = cv2.moments((mask > 0.5).astype(np.uint8), binaryImage=True)
    if m["m00"] < 20:  # too small/empty a mask to trust an orientation from
        return None
    mu20, mu02, mu11 = m["mu20"] / m["m00"], m["mu02"] / m["m00"], m["mu11"] / m["m00"]
    return float(np.degrees(0.5 * np.arctan2(2 * mu11, mu20 - mu02)))


def run(epochs: int, batch_size: int = 16, patience: int = 8) -> dict:
    tr, va, te = list_split("train"), list_split("val"), list_split("test")
    print(f"train={len(tr)} val={len(va)} test={len(te)}", flush=True)

    model = smp.Unet(encoder_name="resnet18", encoder_weights="imagenet", in_channels=3, classes=1).to(DEV)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)
    bce = nn.BCEWithLogitsLoss()

    dl_tr = DataLoader(SegDataset(tr), batch_size=batch_size, shuffle=True, num_workers=4)
    dl_va = DataLoader(SegDataset(va), batch_size=batch_size, num_workers=2)

    def evaluate(dl):
        model.eval(); ious, dices = [], []
        with torch.no_grad():
            for xb, yb in dl:
                pred = torch.sigmoid(model(xb.to(DEV)))
                i, d = dice_iou(pred.cpu(), yb)
                ious.append(i); dices.append(d)
        return float(np.mean(ious)), float(np.mean(dices))

    best_iou, best_state, bad = -1.0, None, 0
    for ep in range(epochs):
        model.train(); t0 = time.time(); tot = 0.0
        for xb, yb in dl_tr:
            xb, yb = xb.to(DEV), yb.to(DEV)
            opt.zero_grad()
            pred = model(xb)
            loss = bce(pred, yb) + soft_dice_loss(torch.sigmoid(pred), yb)
            loss.backward(); opt.step()
            tot += loss.item() * len(yb)
        val_iou, val_dice = evaluate(dl_va)
        print(f"  epoch {ep+1}/{epochs} loss={tot/len(tr):.3f} val_iou={val_iou:.3f} val_dice={val_dice:.3f} "
              f"({time.time()-t0:.1f}s)", flush=True)
        if val_iou > best_iou:
            best_iou, best_state, bad = val_iou, {k: v.cpu().clone() for k, v in model.state_dict().items()}, 0
        else:
            bad += 1
            if bad >= patience:
                print(f"  early stop (no val IoU improvement in {patience} epochs)", flush=True); break

    model.load_state_dict(best_state)
    dl_te = DataLoader(SegDataset(te), batch_size=batch_size, num_workers=2)
    test_iou, test_dice = evaluate(dl_te)
    torch.save(best_state, f"{OUT}/unet_best.pt")

    # bat-angle sample + overlay QA on a handful of test images
    model.eval(); angles = []
    for i, p in enumerate(te[:24]):
        img = cv2.imread(p); h, w = img.shape[:2]
        x = SegDataset([p])[0][0].unsqueeze(0).to(DEV)
        with torch.no_grad():
            pred = torch.sigmoid(model(x))[0, 0].cpu().numpy()
        pred_full = cv2.resize(pred, (w, h))
        ang = bat_angle(pred_full)
        if ang is not None:
            angles.append(ang)
        overlay = img.copy()
        m8 = (pred_full > 0.5).astype(np.uint8) * 255
        overlay[m8 > 0] = (0.5 * overlay[m8 > 0] + 0.5 * np.array([0, 0, 255])).astype(np.uint8)
        cv2.imwrite(f"{OUT}/overlays/{os.path.basename(p)}", overlay)

    summary = {"n_train": len(tr), "n_val": len(va), "n_test": len(te),
               "best_val_iou": best_iou, "test_iou": test_iou, "test_dice": test_dice,
               "sample_bat_angles_deg": angles}
    json.dump(summary, open(f"{OUT}/summary.json", "w"), indent=1)
    print(json.dumps(summary, indent=1), flush=True)
    return summary


if __name__ == "__main__":
    epochs = 3 if LIMIT else 40
    run(epochs)
