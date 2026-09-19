"""Unified shot taxonomy across the public cricket datasets.

Each source uses its own label set. We map every original label to:
  shot  - one of SHOTS (coarse, shared by all sources) or "other" (kept, excluded from the shot head)
  side  - "off" | "leg" | "straight" | "unknown"  (relative to the batter, so handedness-independent)
The original label is always kept as `label_orig` for fine-grained, per-source experiments.
Foot (front/back) and handedness are taken only from explicit annotations (CricketVision);
they are never guessed from shot type, because e.g. pulls and cuts are played off both feet.
"""

SHOTS = ["drive", "defence", "flick_glance", "pull_hook", "cut", "sweep", "lofted", "scoop"]
SIDES = ["off", "leg", "straight", "unknown"]

MAP: dict[str, dict[str, tuple[str, str]]] = {
    "cricshot10k": {
        "Cover Drive": ("drive", "off"),
        "Straight Drive": ("drive", "straight"),
        "Defensive": ("defence", "unknown"),
        "Down The Wicket": ("other", "unknown"),  # footwork, not a stroke
        "Flick": ("flick_glance", "leg"),
        "Hook": ("pull_hook", "leg"),
        "Pull": ("pull_hook", "leg"),
        "Late Cut": ("cut", "off"),
        "Square Cut": ("cut", "off"),
        "Upper Cut": ("cut", "off"),
        "Lofted Legside": ("lofted", "leg"),
        "Lofted Offside": ("lofted", "off"),
        "Reverse Sweep": ("sweep", "off"),
        "Sweep": ("sweep", "leg"),
        "Scoop": ("scoop", "unknown"),
    },
    "cricshot10": {
        "cover": ("drive", "off"),
        "defense": ("defence", "unknown"),
        "flick": ("flick_glance", "leg"),
        "hook": ("pull_hook", "leg"),
        "late_cut": ("cut", "off"),
        "lofted": ("lofted", "unknown"),
        "pull": ("pull_hook", "leg"),
        "square_cut": ("cut", "off"),
        "straight": ("drive", "straight"),
        "sweep": ("sweep", "leg"),
    },
    "ipl2023": {
        "drive": ("drive", "unknown"),
        "pull": ("pull_hook", "leg"),
        "cut": ("cut", "off"),
        "slog": ("lofted", "unknown"),
        "sweep": ("sweep", "unknown"),
        "flick": ("flick_glance", "leg"),
        "misc": ("other", "unknown"),
    },
    "cricketvision": {
        "OffDrive": ("drive", "off"),
        "OnDrive": ("drive", "leg"),
        "Cut": ("cut", "off"),  # source class is "Cut / Square Drive": some drives are inside it
        "Glance": ("flick_glance", "leg"),
        "Hook": ("pull_hook", "leg"),
        "Sweep": ("sweep", "leg"),
        "Block": ("defence", "unknown"),
    },
    "amittalmale": {  # folder names verified after download
        "on drive": ("drive", "leg"),
        "off drive": ("drive", "off"),
        "pull": ("pull_hook", "leg"),
        "sweep": ("sweep", "leg"),
        "cut": ("cut", "off"),
        "glance": ("flick_glance", "leg"),
        "defence": ("defence", "unknown"),
        "unorthodox": ("other", "unknown"),
    },
    "kucricshot": {
        "Defensive Shot": ("defence", "unknown"),
        "Drive Shot": ("drive", "unknown"),
        "Flick Shot": ("flick_glance", "leg"),
        "Pull Shot": ("pull_hook", "leg"),
    },
}


def unify(source: str, label: str) -> tuple[str, str]:
    try:
        return MAP[source][label]
    except KeyError as e:
        raise KeyError(f"no taxonomy mapping for {source!r} label {label!r}") from e
