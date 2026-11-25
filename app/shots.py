"""The shot vocabulary, in two layers.

SHOTS is the trained classifier's output order and indexes its logits — it cannot be reordered
or trimmed without retraining (models/train_shot_classifier.py).

SCORED is what the app actually offers. The technique scorer only has an exemplar set and a
per-shot reference distribution for these five: CricketVision is the one source carrying
body-part scores, and it has too few lofted, scoop or sweep clips to build either. Predicting a
stroke the app cannot then rate properly is worse than not offering it, so the classifier's
probabilities are masked to SCORED and the sample list only offers these five.
"""
SHOTS = ["cut", "defence", "drive", "flick_glance", "lofted", "pull_hook", "scoop", "sweep"]
SCORED = ["cut", "defence", "drive", "flick_glance", "pull_hook"]
SCORED_IDX = [SHOTS.index(s) for s in SCORED]
