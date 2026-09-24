"""Coaching feedback: rule-based findings grounded in the analysis numbers, phrased by a small local LLM.

The rules decide WHAT to say (so every statement traces back to a measured number); the LLM only decides
HOW to say it, and is told not to add numbers or observations. CLAUDE.md rules out paid APIs, so the LLM is
a small open model run locally (CRICLENS_LLM, default Qwen2.5-1.5B-Instruct; "off" = template only). If it
is unavailable or fails, the template text is used -- the findings are identical either way.

Body-part scores are handled carefully: docs/paper/finding_label_collinearity.md showed CricketVision's
five part scores carry almost no signal beyond overall quality, so a part is only singled out when its
percentile is clearly apart from the others, and even then framed as relative.
"""
import os
import threading

MAX_GENERATE_S = float(os.environ.get("CRICLENS_LLM_MAX_SECONDS", "45"))
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


_EXAMPLE_IN = """Analysis findings:
- Shot: Drive: Identified as a drive with 82% confidence.
- Strong overall technique: Scored 7.4/10, better than 78% of rated drives in the coaching dataset.
- Even across body parts: Head, shoulders, hands, hips and feet all rated about the same, so the overall score is the number to focus on.
- Low bat speed: Peak hand speed of 9.1 m/s, slower than most drives.

Recommended drills:
- Top-hand drill: hit throwdowns with the top hand only to groove a full, fast swing."""
_EXAMPLE_OUT = """That was a well-organised drive -- your 7.4/10 puts you ahead of most rated drives. The one thing holding it back is bat speed through the ball.

**What went well**
- Your overall technique is strong, and it's consistent from head to feet.

**Work on**
- Your hands came through at 9.1 m/s, slower than most drives. Commit to the shot and let the bat flow.

**Try this**
- Top-hand drill: hit throwdowns with the top hand only to groove a full, fast swing."""


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
    def __init__(self, status=lambda msg: None):
        self.model_id = os.environ.get("CRICLENS_LLM", "Qwen/Qwen2.5-1.5B-Instruct")
        self.tok = self.llm = None
        self.lock = threading.Lock()
        if self.model_id.lower() == "off":
            return
        try:
            status("Loading coaching language model")
            import torch
            from transformers import AutoModelForCausalLM, AutoTokenizer
            # Default float32 weights are 4 bytes a parameter: ~6 GB for a 1.5B model, which swaps on an
            # 8 GB laptop and turned one paragraph of feedback into a 10-minute wait. bfloat16 halves that,
            # and Apple's GPU shares the same memory, so it costs nothing extra to use it.
            self.device = "mps" if torch.backends.mps.is_available() else "cpu"
            self.tok = AutoTokenizer.from_pretrained(self.model_id)
            self.llm = AutoModelForCausalLM.from_pretrained(self.model_id, dtype=torch.bfloat16).eval().to(self.device)
        except Exception as e:  # the app still works on template text
            print(f"coach: LLM unavailable ({e}); using template feedback", flush=True)
            self.tok = self.llm = None

    @property
    def name(self):
        return self.model_id.split("/")[-1] if self.llm is not None else None

    def write(self, result: dict) -> dict:
        f = findings(result)
        base = {"findings": f["findings"], "drills": f["drills"], "text": template_text(f), "source": "template"}
        if self.llm is None:
            return base
        facts = "\n".join(f"- {x['title']}: {x['detail']}" for x in f["findings"])
        drills = "\n".join(f"- {d}" for d in f["drills"])
        messages = [
            {"role": "system", "content": (
                "You are a friendly, experienced cricket batting coach writing feedback on one shot from a video. "
                "Use ONLY the analysis findings provided: no new numbers, body positions or events. Talk to the batter "
                "as 'you'. If the findings say the shot type is uncertain, say so in the summary. Mention each drill "
                "once, under Try this. Under 130 words, in exactly the format of the example.")},
            {"role": "user", "content": _EXAMPLE_IN},
            {"role": "assistant", "content": _EXAMPLE_OUT},
            {"role": "user", "content": f"Analysis findings:\n{facts}\n\nRecommended drills:\n{drills}"},
        ]
        try:
            import torch
            with self.lock, torch.no_grad():
                enc = self.tok.apply_chat_template(messages, add_generation_prompt=True, return_tensors="pt",
                                                   return_dict=True).to(self.device)
                # max_time caps the stage instead of letting a slow machine hang the whole job queue:
                # whatever has been generated is returned, and short output falls through to the template.
                out = self.llm.generate(**enc, max_new_tokens=220, do_sample=False, repetition_penalty=1.05,
                                        max_time=MAX_GENERATE_S)
                text = self.tok.decode(out[0, enc["input_ids"].shape[1]:], skip_special_tokens=True).strip()
            if len(text) > 40:
                return {**base, "text": text, "source": self.name}
        except Exception as e:
            print(f"coach: generation failed ({e}); using template feedback", flush=True)
        return base
