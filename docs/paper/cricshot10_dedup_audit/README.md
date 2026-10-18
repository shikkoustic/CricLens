# CricShot10 duplicate-cluster audit (2026-10-18)

Manual re-verification of the 157 duplicate clusters the CricLens manifest pipeline
(`datasets/build_manifest.py`) flagged inside CricShot10 (Sen et al., Sensors 2021 -- the dataset
Kang's benchmark, arXiv:2510.09187, is built on). Done because a clip-level flag from a perceptual-hash
pipeline is not proof on its own; this checks whether each flagged cluster is genuinely the same match,
or two different matches that happen to look alike (same generic camera angle/shot type).

## Method

1. `all_clusters.csv`: every CricShot10 clip in a cluster with >1 member (157 clusters, 323 clips),
   with each clip's native split read from its original filename (`cricketshot/<split>/<shot>/...`).
2. `compare_clusters.py`: for every pairwise combination within a cluster (176 pairs total), extracts
   small grayscale frame signatures and finds the best temporal alignment offset (clips are trimmed
   differently), then reports the mean absolute pixel difference at that alignment. Output:
   `pair_diffs.csv`.
3. Every pair with an elevated diff (20 pairs, >=5) plus a random sample of 20 low-diff pairs were
   rendered at full resolution and checked against their on-screen scoreboard, over number, player
   names and team flags -- not just the pixel-diff number.
4. All 157 clusters were rendered as review sheets and visually scanned for any remaining anomaly.

## Result

Zero false positives. Every cluster checked -- including every one with an elevated pixel-diff score --
turned out to be the same match, confirmed by matching scoreboard text. The elevated-diff cases were
the same ball cut together from a different camera angle (wide shot vs close-up), which looks very
different pixel-wise but is the same underlying event.

| | |
|---|---|
| Duplicate clusters | 157 |
| Total clips in those clusters | 323 |
| Cluster sizes | 149 of 2, 7 of 3, 1 of 4 |
| "Extra" duplicate copies (clips removable keeping one canonical per cluster) | 166 |
| Clusters spanning native train/val/test folders | 74 |
| Of those, touching the test folder | 38 (31 test+train, 6 test+val, 1 test+train+val) |
| False positives found | 0 |

See `docs/DATASHEET.md` for how this feeds the dataset-integrity writeup, and `docs/paper/dossier.md`
A3 for how it relates to Kang's unverified "differences in dataset splits" explanation for his models'
40-90 point accuracy drop on reproduction.
