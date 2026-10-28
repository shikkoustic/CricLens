"""Shot classifier v3: real video + pretrained Timesformer + NAFNet/SwiGLU-style gated feed-forward --
Kaggle GPU run (self-contained, no repo imports).

Distinct from kaggle/shot-classifier-sttr-gate/ (which applied divided attention to POSE/joint data, not
pixels, and came back negative, 3-seed confirmed -- PROGRESS.md 2026-10-26). This is the real thing: raw
video frames, cropped to the batter's tracked bounding box (already computed by the pose pipeline, kps.npz
'boxes'), through Facebook's Kinetics-400-pretrained Timesformer (arXiv:2102.05095, HuggingFace
`facebook/timesformer-base-finetuned-k400`), with each block's feed-forward sublayer optionally replaced
by a gated version: 'simple' (NAFNet, arXiv:2204.04676), 'adaptive' (SwiGLU-style, Shazeer,
arXiv:2002.05202), or 'contact' (this project's own architectural novelty, added 2026-10-27: the
'adaptive' gate additionally conditioned on each sampled frame's distance to the ball-bat contact
frame -- see GatedFeedForward's docstring). 'none' keeps the original pretrained GELU feed-forward,
as the control.

PROGRESS.md 2026-10-28: on the full pool, 'adaptive' and 'contact' both lost to 'none' by ~0.20 test
macro-F1 -- replacing the pretrained feed-forward costs more than either gate earns back. 'contact_residual'
(added 2026-10-28) is the fix: an additive branch alongside the UNCHANGED pretrained feed-forward (a
forward hook on TimesformerOutput, not a replacement -- see ContactResidualGate's docstring), so it starts
bit-identical to 'none' rather than needing a warm-start approximation.

IMPORTANT caveat, stated honestly: swapping intermediate+output discards that layer's PRETRAINED
feed-forward weights (replaced with a freshly-initialized gated block) -- only the attention layers keep
Kinetics pretraining when gate != 'none'. Verified against the actual HF source
(TimesformerLayer.forward): `layer_output = self.intermediate(x); layer_output = self.output(layer_output);
hidden_states = hidden_states + drop_path(layer_output)` -- so a GatedFFN placed at `.intermediate`
outputting hidden_size directly, with `.output` replaced by nn.Identity(), preserves the residual
connection exactly.

Heavier than every previous job in this project: raw pixels (8 frames, 224x224, cropped per-frame to the
batter box with 12% padding and zero-padding beyond frame edges, same convention as the pose pipeline's
zero-padding fix) through a ~121M-parameter transformer, not 51 numbers/frame through a GRU. Validated at
300 then 1,500 clips (CricketVision-only, 5-class) before this run, which uses the full 6-source, 8-class
pool (~15,380 clips) -- CRICLENS_LIMIT can still cap it for a quick smoke test.

Inputs : shikkoustic/criclens-all (criclens-processed.tgz.bin -> data/processed/pose_index.parquet +
          kaggle/chunks/pose-*/out/kps/<clip>.npz for window/box data; clips/<source>/<clip_id>.mp4 read
          directly from the dataset's unpacked clips/ folder, same convention as kaggle/pose-pad/pose_pad.py)
Outputs: timesformer_<gate>.pt, summary.json
"""
import glob
import json
import os
import subprocess
import sys
import time

if not os.environ.get("CRICLENS_LOCAL"):
    subprocess.run([sys.executable, "-m", "pip", "install", "-q", "-U", "transformers>=4.48"], check=False)

import cv2
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from transformers import TimesformerForVideoClassification

OUT = os.environ.get("CRICLENS_OUT", "/kaggle/working")
LOCAL = os.environ.get("CRICLENS_LOCAL")
LOCAL_CLIPS = os.environ.get("CRICLENS_LOCAL_CLIPS")  # dir containing <source>/<clip_id>.mp4, for smoke tests
_limit_env = os.environ.get("CRICLENS_LIMIT")  # unset => no cap: use the full 6-source, 8-class pool
LIMIT = int(_limit_env) if _limit_env else None  # (~15,380 clips). Calibration runs on the CricketVision-only
# subset (300, then 1,500 clips) already showed macro-F1 climbing with scale (0.533 -> 0.678), so this run
# uses everything rather than another capped subset.
os.makedirs(OUT, exist_ok=True)
if LOCAL:
    PROC = LOCAL
else:
    PROC = "/tmp/proc"
    os.makedirs(PROC, exist_ok=True)
    _tgz = glob.glob("/kaggle/input/**/criclens-processed.tgz.bin", recursive=True)[0]
    subprocess.run(["tar", "-xzf", _tgz, "-C", PROC, "--wildcards",
                     "data/processed/*", "kaggle/chunks/pose-*/out/kps/*"], check=True)


def clips_dir():
    if LOCAL_CLIPS:
        return LOCAL_CLIPS
    return [c for c in glob.glob("/kaggle/input/**/clips", recursive=True) if os.path.isdir(f"{c}/cricketvision")][0]


NUM_FRAMES = 8
IMG_SIZE = 224
PAD = 0.12
IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], np.float32)


EXCLUDED_SHOTS = {"other", "sweep", "scoop"}  # PROGRESS.md 2026-10-28: 'scoop' has only 96 clips total
# (70 train) and 'sweep' only 838 -- both far too thin next to the other 6 classes (1,898-3,402 each),
# and macro-F1 weights every class equally regardless of size, so a class the model can barely learn
# from a handful of examples drags the metric down out of proportion to how often it matters in practice.
# Dropping both for a more honestly-balanced 6-class task (1,898-3,402 range, <2x spread) rather than
# padding the class count for its own sake.


def load_labelled_clips(limit=None):
    """6 sources, 6 balanced-ish shot types (14,446 clips, EXCLUDED_SHOTS dropped) -- verified before
    using it that every non-CricketVision source's kps.npz has the same 'boxes'/'window' fields in the
    same format (checked ipl2023 directly), and that clips/<source>/<clip_id>.mp4 exists in the dataset
    for every source (confirmed via the dataset file listing), so VideoDataset's per-row r.source lookup
    just works."""
    idx = pd.read_parquet(f"{PROC}/data/processed/pose_index.parquet")
    d = idx[idx.train_ready & ~idx.shot.isin(EXCLUDED_SHOTS)][
        ["clip_id", "shot", "split", "source", "handedness", "kps_path"]]
    limit = limit or (LIMIT or None)
    return d.head(limit) if limit else d


def padded_crop(frame, box, pad=PAD):
    """Crop `frame` to `box` (x0,y0,x1,y1), padded by `pad` fraction of box size, zero-padding any part
    of the padded box that falls outside the frame -- same convention as the pose pipeline's zero-padding
    fix (PROGRESS.md: zero padding cuts edge-case pose error 20-60% over clamping)."""
    h, w = frame.shape[:2]
    x0, y0, x1, y1 = box
    bw, bh = x1 - x0, y1 - y0
    x0, x1 = x0 - pad * bw, x1 + pad * bw
    y0, y1 = y0 - pad * bh, y1 + pad * bh
    ix0, iy0, ix1, iy1 = int(np.floor(x0)), int(np.floor(y0)), int(np.ceil(x1)), int(np.ceil(y1))
    out_w, out_h = max(ix1 - ix0, 1), max(iy1 - iy0, 1)
    out = np.zeros((out_h, out_w, 3), np.uint8)
    sx0, sy0, sx1, sy1 = max(ix0, 0), max(iy0, 0), min(ix1, w), min(iy1, h)
    if sx1 > sx0 and sy1 > sy0:
        out[sy0 - iy0:sy1 - iy0, sx0 - ix0:sx1 - ix0] = frame[sy0:sy1, sx0:sx1]
    return cv2.resize(out, (IMG_SIZE, IMG_SIZE))


def read_clip_pixels(video_path, boxes, window, num_frames=NUM_FRAMES):
    """Sample num_frames evenly across [lo, hi], crop each to its tracked batter box, return
    ((num_frames, 3, H, W) float32 ImageNet-normalised RGB, (num_frames,) float32 contact_signal), or
    (None, None) if the clip can't be read. Some frames have no valid tracked box (occlusion/low
    confidence, same tracking gaps the pose pipeline already deals with) -- fall back to the nearest
    frame (by index distance) with a finite box.

    contact_signal[t] is sampled frame idxs[t]'s signed distance to the ball-bat contact frame `c`,
    normalised by the window length -- the 'contact' gate (GatedFeedForward) uses this to condition its
    feed-forward gate on proximity to the event the window is already anchored on; other gates ignore
    it (computed here regardless since it's nearly free alongside idxs, and keeps one source of truth
    for the frame-sampling formula instead of recomputing idxs wherever this signal is needed)."""
    lo, c, hi = (int(v) for v in window)
    if hi <= lo:
        return None, None
    valid = np.isfinite(boxes).all(axis=1)
    if not valid.any():
        return None, None
    valid_idx = np.flatnonzero(valid)
    idxs = np.linspace(lo, hi, num_frames).round().astype(int)
    idxs = np.clip(idxs, 0, len(boxes) - 1)
    contact_signal = ((idxs - c) / max(hi - lo, 1)).astype(np.float32)
    cap = cv2.VideoCapture(str(video_path))
    frames = []
    for i in idxs:
        box_i = i if valid[i] else valid_idx[np.argmin(np.abs(valid_idx - i))]
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(i))
        ok, f = cap.read()
        if not ok:
            cap.release()
            return None, None
        crop = padded_crop(cv2.cvtColor(f, cv2.COLOR_BGR2RGB), boxes[box_i])
        frames.append(crop)
    cap.release()
    arr = np.stack(frames).astype(np.float32) / 255.0  # (T, H, W, 3)
    arr = (arr - IMAGENET_MEAN) / IMAGENET_STD
    return arr.transpose(0, 3, 1, 2), contact_signal  # (T, 3, H, W), (T,)


def precheck_clips(df):
    """Filters out clips whose window/box data can never decode (same two checks read_clip_pixels makes)
    WITHOUT touching video or holding pixels -- just the tiny kps.npz arrays (~10-15 KB each, fine to
    open 15k+ of in sequence). Caught in production: the full pixel-caching approach this replaced
    (`cache_clips`, decode-everything-into-a-RAM-dict-upfront) held all ~15,380 clips' 8x224x224x3
    float32 arrays simultaneously (~4.6 MB/clip, ~70 GB total) and got SIGKILLed by Kaggle's OOM killer
    43 minutes into the full-pool run, before a single epoch finished -- it had only ever been exercised
    at 300-1,500 clips (1-7 GB), which fit. Fix: don't hold decoded pixels for more than one batch at a
    time (see VideoDataset below); this prefilter only needs the lightweight metadata, so it stays cheap
    at any scale."""
    failed = []
    for r in df.itertuples():
        z = np.load(f"{PROC}/{r.kps_path}")
        lo, c, hi = (int(v) for v in z["window"])
        if hi <= lo or not np.isfinite(z["boxes"]).all(axis=1).any():
            failed.append(r.clip_id)
    return failed


class VideoDataset(Dataset):
    """Decodes and crops each clip's frames on access rather than caching them -- see precheck_clips for
    why. A clip that fails to decode at runtime (rare: real video I/O corruption, not the window/box
    issues precheck_clips already removed) falls back to an all-zero tensor rather than crashing a whole
    epoch over one bad file; DataLoader(num_workers>0) overlaps this CPU decode with GPU compute."""
    def __init__(self, df, classes, clips_root):
        self.df = df.reset_index(drop=True)
        self.c2i = {c: i for i, c in enumerate(classes)}
        self.clips_root = clips_root

    def __len__(self):
        return len(self.df)

    def __getitem__(self, i):
        r = self.df.iloc[i]
        z = np.load(f"{PROC}/{r.kps_path}")
        video_path = f"{self.clips_root}/{r.source}/{r.clip_id}.mp4"
        pix, contact_signal = read_clip_pixels(video_path, z["boxes"], z["window"])
        if pix is None:
            pix = np.zeros((NUM_FRAMES, 3, IMG_SIZE, IMG_SIZE), np.float32)
            contact_signal = np.zeros(NUM_FRAMES, np.float32)
        return torch.from_numpy(pix), torch.from_numpy(contact_signal), self.c2i[r.shot]


class GatedFeedForward(nn.Module):
    """Same simple/adaptive gate as kaggle/shot-classifier-sttr-gate/shot_sttr.py, outputting hidden_size
    directly so it can take TimesformerLayer's `.intermediate` slot with `.output` set to nn.Identity().

    WARM-START FIX (PROGRESS.md 2026-10-26, after the first small-subset run): naively replacing
    intermediate+output discards that layer's pretrained Kinetics feed-forward weights entirely, so a
    freshly-initialized gate has to relearn 12 layers' worth of feed-forward transforms from scratch --
    on a tiny subset (236 train clips, 5 epochs) the loss barely moved (1.89 -> 1.64) while the
    pretrained-FFN control dropped 1.66 -> 0.14. Fix: initialize the gate so it starts out computing
    (approximately) the same thing the original pretrained dense1->activation->dense2 did, then let
    training adapt the gate parameters away from that starting point instead of from random noise.
    'adaptive' is accordingly plain GLU (a * sigmoid(b), the base case in Shazeer's GLU-variants paper,
    arXiv:2002.05202 -- not the SwiGLU variant used before) specifically because sigmoid's bounded output
    allows a clean "gate open" initialization (bias=4 -> sigmoid~0.98) that a Swish gate does not.

    'contact' (PROGRESS.md 2026-10-27, the architectural novelty angle, not published prior art like the
    other two gates): same plain GLU, but the sigmoid gate's pre-activation is additionally biased by a
    small linear readout of `contact_signal` -- each token's sampled frame's signed, window-normalised
    distance to the ball-bat contact frame (set externally, once per forward pass, by GateWrapper; see
    its docstring for the token-layout details). The idea: let the feed-forward gate open and close as a
    function of how close this moment is to contact, something neither NAFNet's fixed gate nor a plain
    learned gate can condition on, since neither sees per-frame timing at all. `contact_proj` is
    zero-initialized (weight AND bias), so at init `contact_proj(contact_signal) == 0` for every input --
    the gate reduces exactly to 'adaptive' at the start of training (the already-verified GELU stand-in
    above), and training is what decides whether the contact signal ends up mattering."""

    def __init__(self, hidden_size, intermediate_size, gate="simple", dropout=0.0,
                 pretrained_dense=None, pretrained_out=None):
        super().__init__()
        self.gate = gate
        if gate == "simple":
            self.proj_in = nn.Linear(hidden_size, intermediate_size * 2)
        else:
            self.proj_a = nn.Linear(hidden_size, intermediate_size)
            self.proj_b = nn.Linear(hidden_size, intermediate_size)
        if gate == "contact":
            self.contact_proj = nn.Linear(1, intermediate_size)
            nn.init.zeros_(self.contact_proj.weight); nn.init.zeros_(self.contact_proj.bias)
            self.contact_signal = None  # (batch, seq_len, 1), set per-forward-pass by GateWrapper
        self.proj_out = nn.Linear(intermediate_size, hidden_size)
        self.drop = nn.Dropout(dropout)
        if pretrained_dense is not None:
            self._warm_start(pretrained_dense, pretrained_out)

    @torch.no_grad()
    def _warm_start(self, dense, out):
        """Checked empirically (not just assumed) before committing GPU time: on a random input through
        a real pretrained layer, out(SiLU(dense(x))) vs the true out(GELU(dense(x))) gives mean abs diff
        0.29 against an output scale of 3.47 (~8% relative error, correlation 0.998 between SiLU(dense(x))
        and GELU(dense(x))) -- a close stand-in. The first version of this (b~constant, a*b~=a, skipping
        the nonlinearity entirely) was NOT checked this way and turned out badly wrong (~100% relative
        error) -- GELU is a real nonlinear transform, not something a constant gate can skip past."""
        n = dense.out_features
        if self.gate == "simple":
            # Tried the same approach (a=b=dense -> a*a = dense(x)^2) with a 0.5 output rescale, same
            # as 'adaptive' below -- checked it the same way and it does NOT work: mean abs diff 6.25
            # against an output scale of 3.47 (worse than the scale itself), because squaring produces
            # heavy-tailed outliers (max diff 76 on the same test) that no scalar rescale fixes -- that's
            # a shape mismatch against GELU, not a magnitude one. NAFNet's plain product has no bounded
            # gate the way sigmoid does, so there's no equally clean warm-start for it. Left as plain
            # random init (pretrained_dense is still used for proj_out only, everything else default
            # nn.Linear init) rather than ship an unverified rescale -- 'simple' is excluded from
            # CRICLENS_GATES below until this is solved properly.
            self.proj_out.weight.copy_(out.weight); self.proj_out.bias.copy_(out.bias)
        else:
            # a=b=dense -> a*sigmoid(b) = dense(x)*sigmoid(dense(x)) = SiLU(dense(x)), the close GELU
            # stand-in verified above. Covers both 'adaptive' and 'contact' -- contact_proj's separate
            # zero-init (above) is what keeps 'contact' at this same starting point despite the extra term.
            self.proj_a.weight.copy_(dense.weight); self.proj_a.bias.copy_(dense.bias)
            self.proj_b.weight.copy_(dense.weight); self.proj_b.bias.copy_(dense.bias)
            self.proj_out.weight.copy_(out.weight); self.proj_out.bias.copy_(out.bias)

    def forward(self, x):
        if self.gate == "simple":
            a, b = self.proj_in(x).chunk(2, dim=-1)
            h = a * b
        elif self.gate == "contact":
            b = self.proj_b(x)
            if self.contact_signal is not None:
                b = b + self.contact_proj(self.contact_signal.to(x.dtype))
            h = self.proj_a(x) * torch.sigmoid(b)
        else:
            h = self.proj_a(x) * torch.sigmoid(self.proj_b(x))
        return self.proj_out(self.drop(h))


class GateWrapper(nn.Module):
    """Thin wrapper so run()'s forward call can always pass `contact_signal` (a per-sample, per-frame
    tensor) whether or not the active gate uses it -- only 'contact' gate layers read it, via a plain
    attribute set here each forward pass, since TimesformerEncoder's layer loop doesn't expose a way to
    thread extra args through to each layer.intermediate call.

    Token layout verified against the installed transformers' TimesformerEmbeddings.forward (divided
    space-time attention path): the sequence is [CLS, then patch-major/frame-minor -- patch 0's frames
    0..T-1, patch 1's frames 0..T-1, ...]. So tiling `contact_signal` (shape batch x T) to
    (batch, num_patches x T) by repeating it num_patches times, then prepending one zero for CLS,
    reproduces that exact order."""

    def __init__(self, model, gate_layers, num_patches):
        super().__init__()
        self.model = model
        self.gate_layers = gate_layers
        self.num_patches = num_patches

    def forward(self, pixel_values, contact_signal=None):
        if self.gate_layers and contact_signal is not None:
            b, t = contact_signal.shape
            tok = contact_signal.unsqueeze(1).expand(b, self.num_patches, t).reshape(b, self.num_patches * t, 1)
            tok = torch.cat([torch.zeros(b, 1, 1, device=tok.device, dtype=tok.dtype), tok], dim=1)
            for g in self.gate_layers:
                g.contact_signal = tok
        return self.model(pixel_values=pixel_values)


class ContactResidualGate(nn.Module):
    """PROGRESS.md 2026-10-28: 'contact' and 'adaptive' (both REPLACE intermediate+output, discarding
    the pretrained Kinetics feed-forward weights) lost to plain 'none' by ~0.20 test macro-F1 on the
    full pool -- confirmed, not assumed, with a real baseline. The likely reason: replacing 12 layers'
    worth of pretrained feed-forward transforms costs more than either gate's added expressiveness earns
    back. This gate doesn't replace anything: it's a small additive branch registered as a forward hook
    on `layer.output` (TimesformerOutput), so the pretrained `intermediate`+`output` path runs completely
    unchanged and fully used, exactly as in 'none'. The hook reads TimesformerOutput.forward's own only
    argument (dropout(GELU(dense(x))), intermediate_size-dim -- verified against the installed
    transformers source, same as GatedFeedForward's 'contact' did for the layout) plus the contact-timing
    signal, and ADDS a small correction to the FFN's hidden_size-dim output, before the layer's own
    residual sum. `proj_out` is zero-initialized, so at init this is not an approximation of 'none' the
    way the replacement gates needed a warm-start to approximate GELU -- it is BIT-IDENTICAL to 'none',
    verified the same way (see the init-equality check run before this was trusted on Kaggle GPU time).
    `bottleneck` keeps the added branch small relative to a full FFN (~128*3072*2 + 128*768 =~ 0.88M
    params/layer here vs ~4.7M/layer for a full intermediate+output pair)."""

    def __init__(self, intermediate_size, hidden_size, bottleneck=128):
        super().__init__()
        self.proj_a = nn.Linear(intermediate_size, bottleneck)
        self.proj_b = nn.Linear(intermediate_size, bottleneck)
        self.contact_proj = nn.Linear(1, bottleneck)
        self.proj_out = nn.Linear(bottleneck, hidden_size)
        nn.init.zeros_(self.proj_out.weight); nn.init.zeros_(self.proj_out.bias)
        self.contact_signal = None  # (batch, seq_len, 1), set externally per forward pass by GateWrapper

    def forward(self, intermediate_act):
        b = self.proj_b(intermediate_act)
        if self.contact_signal is not None:
            b = b + self.contact_proj(self.contact_signal.to(intermediate_act.dtype))
        h = self.proj_a(intermediate_act) * torch.sigmoid(b)
        return self.proj_out(h)


def _residual_hook(gate_module):
    def hook(module, inputs, output):
        return output + gate_module(inputs[0])
    return hook


def build_model(n_classes, gate, device):
    model = TimesformerForVideoClassification.from_pretrained(
        "facebook/timesformer-base-finetuned-k400", num_labels=n_classes, ignore_mismatched_sizes=True)
    gate_layers = []
    cfg = model.config
    if gate == "contact_residual":
        for layer in model.timesformer.encoder.layer:
            g = ContactResidualGate(cfg.intermediate_size, cfg.hidden_size)
            layer.output.register_forward_hook(_residual_hook(g))
            layer.residual_gate = g  # plain attribute assignment registers it as a submodule (standard
            # nn.Module.__setattr__ behaviour) -- needed so g's params show up in .parameters()/.to(device)
            gate_layers.append(g)
    elif gate != "none":
        for layer in model.timesformer.encoder.layer:
            dense, out = layer.intermediate.dense, layer.output.dense  # pretrained weights, before replacing
            g = GatedFeedForward(cfg.hidden_size, cfg.intermediate_size, gate=gate,
                                  dropout=cfg.hidden_dropout_prob, pretrained_dense=dense, pretrained_out=out)
            layer.intermediate = g
            layer.output = nn.Identity()
            gate_layers.append(g)
    num_patches = (model.config.image_size // model.config.patch_size) ** 2
    return GateWrapper(model, gate_layers, num_patches).to(device)


def classify_eval(preds, trues, n_classes):
    n = len(trues)
    acc = sum(p == t for p, t in zip(preds, trues)) / n
    f1s, supports = [], []
    for c in range(n_classes):
        tp = sum(p == c and t == c for p, t in zip(preds, trues))
        fp = sum(p == c and t != c for p, t in zip(preds, trues))
        fn = sum(p != c and t == c for p, t in zip(preds, trues))
        sup = sum(t == c for t in trues)
        prec = tp / (tp + fp) if (tp + fp) else 0.0
        rec = tp / (tp + fn) if (tp + fn) else 0.0
        f1 = 2 * prec * rec / (prec + rec) if (prec + rec) else 0.0
        f1s.append(f1); supports.append(sup)
    macro_f1 = float(np.mean(f1s))
    weighted_f1 = float(np.average(f1s, weights=supports)) if sum(supports) else 0.0
    return acc, macro_f1, weighted_f1


def find_checkpoint(gate):
    """Checks OUT first (a checkpoint from an earlier call within the same process/output dir -- this is
    what a local resume test exercises, since there's no /kaggle/input locally), then falls back to a
    recursive glob under /kaggle/input. Self-referential kernel_sources (kernel-metadata.json lists this
    same kernel as its own source) makes a prior push's /kaggle/working output show up under
    /kaggle/input on the next push -- same generic-glob pattern as find_pretrained_encoder(), so this
    doesn't need to know Kaggle's exact input-folder naming. Returns None (start fresh) if neither has a
    checkpoint, which is the correct behaviour on the very first push too."""
    own = f"{OUT}/checkpoint_{gate}.pt"
    if os.path.exists(own):
        return own
    hits = glob.glob(f"/kaggle/input/**/checkpoint_{gate}.pt", recursive=True)
    return hits[0] if hits else None


def run(gate, epochs, device, d, clips_root, classes, batch_size=8, patience=3, lr=1e-4, seed=0, num_workers=4):
    """batch_size=16 OOM'd a T4 (14.56 GiB) partway through epoch 1 of the 'none' condition -- a
    ViT-Base video transformer over 8 frames at 224x224 holds a lot more activation memory per sample
    than anything else in this project. Fixed two ways, not just by shrinking the batch further: mixed
    precision (fp16 autocast + GradScaler) roughly halves activation memory AND is faster on a T4's
    tensor cores, so batch_size=8 with amp should be safer AND not meaningfully slower than the 16 that
    OOM'd.

    Checkpoints every epoch (full resumable state: model, optimizer, scaler, best-so-far) and resumes
    from one if found -- the full 15,380-clip, 8-class pool is a multi-hour job, and this script has
    already hit three different failure modes (an opaque stuck session, an OOM, and a stuck-gradient
    bug) in three prior runs, so losing hours of progress to a fourth is a real risk worth engineering
    around rather than risking again."""
    torch.manual_seed(seed); np.random.seed(seed)
    tr, va, te = (d[d.split == s] for s in ("train", "val", "test"))
    print(f"[gate={gate}, seed={seed}] train={len(tr)} val={len(va)} test={len(te)}", flush=True)

    model = build_model(len(classes), gate, device)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    counts = tr.shot.value_counts().reindex(classes).to_numpy()
    weight = torch.tensor(counts.sum() / (len(classes) * counts), dtype=torch.float32, device=device)
    crit = nn.CrossEntropyLoss(weight=weight)
    amp = device == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=amp)

    # VideoDataset now decodes+crops per access (see precheck_clips/VideoDataset docstrings) instead of
    # reading from a RAM cache, so worker processes overlap that CPU decode with GPU compute.
    dl_tr = DataLoader(VideoDataset(tr, classes, clips_root), batch_size=batch_size, shuffle=True,
                        num_workers=num_workers, persistent_workers=num_workers > 0)
    dl_va = DataLoader(VideoDataset(va, classes, clips_root), batch_size=batch_size, num_workers=num_workers)
    dl_te = DataLoader(VideoDataset(te, classes, clips_root), batch_size=batch_size, num_workers=num_workers)

    start_ep, best_f1, best_state, bad = 0, -1.0, None, 0
    ckpt_path = find_checkpoint(gate)
    if ckpt_path:
        ckpt = torch.load(ckpt_path, map_location=device)
        model.load_state_dict(ckpt["model"]); opt.load_state_dict(ckpt["opt"]); scaler.load_state_dict(ckpt["scaler"])
        start_ep, best_f1, best_state, bad = ckpt["epoch"], ckpt["best_f1"], ckpt["best_state"], ckpt["bad"]
        print(f"[gate={gate}, seed={seed}] RESUMED from {ckpt_path}: epoch={start_ep} best_f1={best_f1:.3f}", flush=True)

    n_batches = len(dl_tr)
    for ep in range(start_ep, epochs):
        model.train(); t0 = time.time(); tot_loss = 0.0; seen = 0; t_last_print = t0
        for bi, (xb, cb, yb) in enumerate(dl_tr, 1):
            xb, cb, yb = xb.to(device), cb.to(device), yb.to(device)
            opt.zero_grad()
            with torch.amp.autocast("cuda", enabled=amp):
                loss = crit(model(pixel_values=xb, contact_signal=cb).logits, yb)
            scaler.scale(loss).backward()
            scaler.unscale_(opt)  # needed before clipping -- grads are still fp16-scaled otherwise
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(opt); scaler.update()
            tot_loss += loss.item() * len(yb); seen += len(yb)
            # Kaggle's API only exposes a kernel's log once it finishes (confirmed -- `kaggle kernels
            # logs`/`-f`/`output` all return nothing for a RUNNING kernel), so the one place to actually
            # watch progress mid-run is the kernel's own page in a browser, which streams stdout live.
            # This print (time-gated, not every batch, to avoid spamming that view) is what shows up there.
            now = time.time()
            if now - t_last_print >= 60 or bi == n_batches:
                rate = seen / (now - t0)
                eta_min = (n_batches * batch_size - seen) / rate / 60 if rate > 0 else float("nan")
                print(f"    [gate={gate}, seed={seed}] epoch {ep+1}/{epochs}  batch {bi}/{n_batches}  "
                      f"running_loss={tot_loss/seen:.3f}  ({rate:.1f} clips/s, ~{eta_min:.1f} min left this epoch)",
                      flush=True)
                t_last_print = now
        model.eval(); preds, trues = [], []
        with torch.no_grad(), torch.amp.autocast("cuda", enabled=amp):
            for xb, cb, yb in dl_va:
                preds += model(pixel_values=xb.to(device), contact_signal=cb.to(device)).logits.argmax(1).cpu().tolist()
                trues += yb.tolist()
        acc, macro_f1, weighted_f1 = classify_eval(preds, trues, len(classes))
        scale_note = f"  amp_scale={scaler.get_scale():.0f}" if amp else ""
        print(f"  [gate={gate}, seed={seed}] epoch {ep+1}/{epochs}  train_loss={tot_loss/len(tr):.3f}  "
              f"val_acc={acc:.3f}  val_macroF1={macro_f1:.3f}  ({time.time()-t0:.1f}s){scale_note}", flush=True)
        if macro_f1 > best_f1:
            best_f1, best_state, bad = macro_f1, {k: v.cpu().clone() for k, v in model.state_dict().items()}, 0
        else:
            bad += 1
        torch.save({"model": model.state_dict(), "opt": opt.state_dict(), "scaler": scaler.state_dict(),
                    "epoch": ep + 1, "best_f1": best_f1, "best_state": best_state, "bad": bad},
                   f"{OUT}/checkpoint_{gate}.pt")
        if bad >= patience:
            print(f"  [gate={gate}, seed={seed}] early stop", flush=True)
            break

    model.load_state_dict(best_state); model.eval()
    preds, trues = [], []
    with torch.no_grad(), torch.amp.autocast("cuda", enabled=amp):
        for xb, cb, yb in dl_te:
            preds += model(pixel_values=xb.to(device), contact_signal=cb.to(device)).logits.argmax(1).cpu().tolist()
            trues += yb.tolist()
    acc, macro_f1, weighted_f1 = classify_eval(preds, trues, len(classes))
    print(f"\n[gate={gate}, seed={seed}] TEST  acc={acc:.3f}  macro_f1={macro_f1:.3f}  weighted_f1={weighted_f1:.3f}", flush=True)

    torch.save(best_state, f"{OUT}/timesformer_{gate}_seed{seed}.pt")
    result = {"gate": gate, "seed": seed, "best_val_macro_f1": best_f1, "test_accuracy": acc,
              "test_macro_f1": macro_f1, "test_weighted_f1": weighted_f1,
              "n_train": len(tr), "n_val": len(va), "n_test": len(te), "classes": classes}
    del model, opt, scaler, best_state  # AdamW state is ~2x model size; free it before the next gate's run
    if device == "cuda":
        torch.cuda.empty_cache()
    return result


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    epochs = int(os.environ.get("CRICLENS_EPOCHS", "8"))  # patience=3 should stop earlier if it converges
    # 'none' and 'adaptive' are the full-pool run already in flight/completed as of PROGRESS.md
    # 2026-10-27 -- default here to 'contact' only (the architectural-novelty gate) so the next push
    # doesn't burn GPU time re-running already-confirmed numbers.
    gates = os.environ.get("CRICLENS_GATES", "adaptive").split(",")
    seed = int(os.environ.get("CRICLENS_SEED", "0"))
    print(f"device={device}  epochs={epochs}  gates={gates}  seed={seed}", flush=True)

    d_all = load_labelled_clips()
    classes = sorted(d_all.shot.unique())
    failed = precheck_clips(d_all)
    d_all = d_all[~d_all.clip_id.isin(failed)]
    print(f"{len(d_all)} clips pass precheck ({len(failed)} dropped), classes={classes}", flush=True)
    root = clips_dir()

    summary = [run(g, epochs, device, d_all, root, classes, seed=seed) for g in gates]
    json.dump(summary, open(f"{OUT}/summary.json", "w"), indent=2)
    print("\n=== summary ===", flush=True)
    for s in summary:
        print(f"[{s['gate']}] val_macroF1={s['best_val_macro_f1']:.3f}  test_acc={s['test_accuracy']:.3f}  "
              f"test_macroF1={s['test_macro_f1']:.3f}", flush=True)
    print("\nreference (pose-only architectures): LSTM 0.785, GRU 0.777, Transformer 0.766, "
          "pose-ST-TR+gate ~0.754-0.755 (all macro-F1)", flush=True)


if __name__ == "__main__":
    main()
