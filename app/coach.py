"""Coaching feedback: rule-based findings grounded in the analysis numbers, phrased as fixed templates.

Every statement traces back to a measured number -- there is no free-text generation here. An LLM
phrasing pass was prototyped (Qwen2.5-1.5B-Instruct) but dropped per a supervisor conversation
(PROGRESS.md, 2026-10-17): the paper's scope is classification + technique scoring, not coaching text,
so the app no longer carries a language model.

Body-part scores are handled carefully: docs/paper/finding_label_collinearity.md showed CricketVision's
five part scores carry almost no signal beyond overall quality, so a part is only singled out when its
percentile is clearly apart from the others, and even then framed as relative.
"""
FRONT_FOOT = {"drive", "defence", "flick_glance", "lofted", "sweep", "scoop"}
ATTACKING = {"drive", "cut", "pull_hook", "lofted", "flick_glance", "sweep", "scoop"}
PART_NAMES = {"head": "head position", "shoulder": "shoulder alignment", "hands": "hands and bat path",
              "hips": "hip rotation", "feet": "footwork"}
SHOT_DRILL = {
    "drive": "Drive off a batting tee: ball level with your front toe, head over it at contact, full face of a straight bat.",
    "cut": "Cut off a tee set wide outside off stump: move back and across, stay above the ball, hit down on it.",
    "pull_hook": "Pull throwdowns from short balls: turn the hips, stay balanced, roll the wrists to keep it down.",
    "defence": "Forward defence in front of a mirror: head over the front knee, bat angled down, loose grip.",
    "flick_glance": "Flick throwdowns on leg stump: play late under your eyes and close the face with the wrists.",
    "lofted": "Lofted drives in the nets: get to the pitch of the ball first, then a full, high follow-through.",
    "sweep": "Sweep off a tee: long front stride, head over the ball, bat horizontal at contact.",
    "scoop": "Scoop drill with soft throwdowns: get low early, open the face, keep the head still.",
}


def _band(p):
    return None if p is None else ("high" if p >= 70 else "low" if p < 35 else "mid")


def findings(r: dict) -> dict:
    shot, tech, met = r["shot"], r["technique"], r["metrics"]
    name = shot["display"]
    lower = name.lower()
    out, drills = [], []

    top, second = shot["probs"][0], shot["probs"][1]
    if top["p"] >= 0.6:
        out.append({"tone": "neutral", "title": f"Shot: {name}", "detail": f"Identified as a {lower} with {top['p']:.0%} confidence."})
    else:
        out.append({"tone": "caution", "title": f"Shot: probably a {lower}",
                    "detail": f"{top['p']:.0%} {lower} vs {second['p']:.0%} {second['display'].lower()} -- the movement "
                              f"looked like a mix of the two, so read the shot-specific comparisons below with that in mind."})

    ref_name = f"rated {lower}s" if tech["reference_shot"] != "all" else "all rated shots"
    band = _band(tech["overall_pct"])
    pct = tech["overall_pct"]
    if band == "high":
        out.append({"tone": "good", "title": "Strong overall technique",
                    "detail": f"Scored {tech['overall']:.1f}/10, better than {pct:.0f}% of {ref_name} in the coaching dataset."})
    elif band == "mid":
        out.append({"tone": "neutral", "title": "Solid overall technique",
                    "detail": f"Scored {tech['overall']:.1f}/10, around the middle of {ref_name} ({pct:.0f}th percentile)."})
    elif band == "low":
        out.append({"tone": "improve", "title": "Technique below typical",
                    "detail": f"Scored {tech['overall']:.1f}/10, lower than {100 - pct:.0f}% of {ref_name}."})
        if shot["label"] in SHOT_DRILL:
            drills.append(SHOT_DRILL[shot["label"]])

    parts = [p for p in tech["parts"] if p["pct"] is not None]
    if parts:
        lo = min(parts, key=lambda p: p["pct"]); hi = max(parts, key=lambda p: p["pct"])
        if hi["pct"] - lo["pct"] >= 15:
            out.append({"tone": "improve", "title": f"Relative weak spot: {PART_NAMES[lo['part']]}",
                        "detail": f"The lowest of the five body-part ratings here ({lo['score']:.1f}/10), while "
                                  f"{PART_NAMES[hi['part']]} rated best ({hi['score']:.1f}/10)."})
        else:
            out.append({"tone": "neutral", "title": "Even across body parts",
                        "detail": "Head, shoulders, hands, hips and feet all rated about the same, so the overall "
                                  "score is the number to focus on."})

    if met.get("calibrated"):
        sref = f"{lower}s" if met.get("reference_shot") not in (None, "all") else "shots"
        if met.get("stride_cm") is not None:
            b, s = _band(met.get("stride_pct")), met["stride_cm"]
            if b == "low":
                out.append({"tone": "improve", "title": "Short stride",
                            "detail": f"Front-foot movement of about {s:.0f} cm, shorter than most {sref} in the dataset."})
                if shot["label"] in FRONT_FOOT:
                    drills.append("Cone stride drill: put a cone one full stride ahead and step to it on every throwdown, "
                                  "head over the front knee.")
            elif b == "high":
                out.append({"tone": "neutral", "title": "Big stride",
                            "detail": f"Foot movement of about {s:.0f} cm, longer than most {sref} -- fine if you stay balanced."})
                drills.append("Balance hold: freeze for two seconds after each shot in throwdowns; if you can't, "
                              "the stride is too big.")
            else:
                out.append({"tone": "good", "title": "Stride in the normal range",
                            "detail": f"Foot movement of about {s:.0f} cm, typical for {sref}."})
        if met.get("swing_mps") is not None:
            b, v = _band(met.get("swing_pct")), met["swing_mps"]
            if shot["label"] == "defence" and b == "high":
                out.append({"tone": "improve", "title": "Hard hands in defence",
                            "detail": f"Peak hand speed of {v:.1f} m/s is high for a defensive shot."})
                drills.append("Soft-hands drill: defend throwdowns so the ball drops within two metres of the bat.")
            elif shot["label"] in ATTACKING and b == "low":
                out.append({"tone": "improve", "title": "Low bat speed",
                            "detail": f"Peak hand speed of {v:.1f} m/s, slower than most {sref}."})
                drills.append("Top-hand drill: hit throwdowns with the top hand only to groove a full, fast swing.")
            else:
                out.append({"tone": "good" if b != "low" else "neutral", "title": "Hand speed",
                            "detail": f"Peak hand speed of {v:.1f} m/s ({met['swing_pct']:.0f}th percentile for {sref})."})
    if not drills:
        drills.append(SHOT_DRILL.get(shot["label"], "Shadow-bat the shot slowly in front of a mirror, checking your head "
                                                    "position at contact."))
    return {"findings": out, "drills": drills[:2]}


def template_text(f: dict) -> str:
    good = [x for x in f["findings"] if x["tone"] == "good"]
    work = [x for x in f["findings"] if x["tone"] == "improve"]
    head = f["findings"][0]["detail"] + (" " + f["findings"][1]["detail"] if len(f["findings"]) > 1 else "")
    lines = [head, ""]
    if good:
        lines += ["**What went well**"] + [f"- {x['title']}: {x['detail']}" for x in good] + [""]
    if work:
        lines += ["**Work on**"] + [f"- {x['title']}: {x['detail']}" for x in work] + [""]
    lines += ["**Try this**"] + [f"- {d}" for d in f["drills"]]
    return "\n".join(lines).strip()


class Coach:
    """No model to load; kept as a class so app/server.py's lifecycle (status callback, startup
    thread) doesn't need special-casing for a stage that happens to be instant."""

    def __init__(self, status=lambda msg: None):
        pass

    @property
    def name(self):
        return None

    def write(self, result: dict) -> dict:
        f = findings(result)
        return {"findings": f["findings"], "drills": f["drills"], "text": template_text(f), "source": "template"}
