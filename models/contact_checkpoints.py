"""Five textbook batting checkpoints, measured from the pose at the moment of contact.

Why these exist alongside the technique scorer: the scorer returns a number, and
docs/paper/finding_label_collinearity.md showed its five part scores are one signal wearing five
hats, so "your hands scored 6.1" cannot honestly be turned into advice. These are the opposite
kind of measurement -- plain geometry a coach already talks about, computed the same way for
every clip, and reported next to what well-rated players of the same stroke do.

They are a comparison, not a predictor, and the code should not be read as claiming otherwise:
measured against CricketVision's coach scores on 2,499 clips, each one correlates weakly
(|rho| 0.02-0.09 pooled, up to 0.25 within a single stroke). What is worth saying is that the
signs all land where coaching orthodoxy says they should, independently of each other -- head
nearer the front knee, more knee flex and a tighter front elbow all go with higher scores. See
models/validate_checkpoints.py for the table.

Everything is scale-free (divided by torso length) except the two angles, so a batter filmed
from the boundary and one filmed from behind the arm are measured on the same scale.
"""
import numpy as np

# COCO-17 joint order, as ViTPose returns it
NOSE, LSH, RSH, LEL, REL, LWR, RWR, LHIP, RHIP, LKN, RKN, LAN, RAN = 0, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16
MIN_CONF = 0.3          # below this the joint is treated as unseen
MIN_TORSO_PX = 5.0      # a batter smaller than this is too far away to measure

# label, unit, and which direction coaching says is better -- the UI needs all three
CHECKPOINTS = {
    "head_over_knee":  {"name": "Head over front knee", "unit": "torso", "better": "low",
                        "hint": "How far the head is from directly above the front knee at contact."},
    "front_knee_flex": {"name": "Front knee flex", "unit": "deg", "better": "low",
                        "hint": "The angle at the front knee. Straighter legs mean a taller, stiffer position."},
    "front_elbow":     {"name": "Front elbow", "unit": "deg", "better": "low",
                        "hint": "The angle at the front elbow -- the “high elbow” a coach asks for."},
    "spine_lean":      {"name": "Spine lean", "unit": "deg", "better": "low",
                        "hint": "How far the upper body leans off vertical at contact."},
    "stride":          {"name": "Stride", "unit": "torso", "better": "mid",
                        "hint": "Distance between the feet at contact, measured in torso lengths."},
}


def _angle(a, b, c):
    """Interior angle at b, in degrees."""
    v1, v2 = a - b, c - b
    n = np.linalg.norm(v1) * np.linalg.norm(v2)
    if n == 0:
        return None
    return float(np.degrees(np.arccos(np.clip(float(np.dot(v1, v2)) / n, -1.0, 1.0))))


def front_side(k, lo, contact):
    """Which side is the front leg: 'L' or 'R'.

    Inferred from the stride rather than taken from a handedness label, because the app has no
    handedness for an uploaded clip -- and because the label is missing for 15,528 of the 22,420
    clips anyway. The front foot is the one that travels toward the ball between the start of the
    window and contact, so take whichever ankle moves further horizontally. Falls back to the
    leading ankle at contact when neither moves.
    """
    lo, contact = int(lo), int(min(contact, len(k) - 1))
    a, b = k[lo], k[contact]
    moved = {}
    for side, idx in (("L", LAN), ("R", RAN)):
        if a[idx, 2] >= MIN_CONF and b[idx, 2] >= MIN_CONF:
            moved[side] = abs(float(b[idx, 0] - a[idx, 0]))
    if len(moved) == 2 and abs(moved["L"] - moved["R"]) > 1.0:
        return max(moved, key=moved.get)
    mid_hip_x = (b[LHIP, 0] + b[RHIP, 0]) / 2
    return "L" if abs(b[LAN, 0] - mid_hip_x) > abs(b[RAN, 0] - mid_hip_x) else "R"


def measure(k, lo, contact) -> dict | None:
    """The five checkpoints at the contact frame, or None if the pose there is too poor to use.

    k is (frames, 17, 3) in pixels with confidence; lo and contact are frame indices into it.
    A checkpoint whose joints weren't seen comes back as None rather than a guess.
    """
    i = int(min(max(contact, 0), len(k) - 1))
    f = k[i]
    if not np.isfinite(f[:, :2]).all(axis=1).any():
        return None
    xy, c = f[:, :2], f[:, 2]

    def seen(*js):
        return all(np.isfinite(xy[j]).all() and c[j] >= MIN_CONF for j in js)

    if not seen(LSH, RSH, LHIP, RHIP):
        return None
    msh, mhip = (xy[LSH] + xy[RSH]) / 2, (xy[LHIP] + xy[RHIP]) / 2
    torso = float(np.linalg.norm(msh - mhip))
    if torso < MIN_TORSO_PX:
        return None

    side = front_side(k, lo, i)
    fkn, fan, fhip = (LKN, LAN, LHIP) if side == "L" else (RKN, RAN, RHIP)
    bkn, ban = (RKN, RAN) if side == "L" else (LKN, LAN)
    fsh, fel, fwr = (LSH, LEL, LWR) if side == "L" else (RSH, REL, RWR)

    out = {"_torso_px": torso, "_front": side, "_contact_frame": i}
    out["head_over_knee"] = (abs(float(xy[NOSE, 0] - xy[fkn, 0])) / torso
                             if seen(NOSE, fkn) else None)
    out["front_knee_flex"] = _angle(xy[fhip], xy[fkn], xy[fan]) if seen(fhip, fkn, fan) else None
    out["front_elbow"] = _angle(xy[fsh], xy[fel], xy[fwr]) if seen(fsh, fel, fwr) else None
    out["spine_lean"] = float(np.degrees(np.arctan2(abs(msh[0] - mhip[0]),
                                                    abs(msh[1] - mhip[1]) + 1e-6)))
    out["stride"] = (float(np.linalg.norm(xy[fan] - xy[ban])) / torso
                     if seen(fan, ban) else None)
    return out
