"""The shot classes the trained classifier predicts, in its output order.

Kept in a module of its own so the server can check a label against it without importing
app.pipeline, which pulls in torch and ultralytics and takes seconds to load. The order must
match the `classes` list in models/shot_classifier/summary.json -- it indexes the model's logits.

"other" is deliberately absent: models/train_shot_classifier.py trains on the eight named shots
only, so nothing in the app can ever return it.
"""
SHOTS = ["cut", "defence", "drive", "flick_glance", "lofted", "pull_hook", "scoop", "sweep"]
