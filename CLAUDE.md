# CricLens: context for Claude

CricLens analyses cricket batting videos: it finds the batter, extracts body joints from ball release to
follow-through, and (next phase) classifies the shot, scores technique per body part, and writes coaching
feedback, served as a web app. It is one project submitted to two college courses (Bennett University, 2026):

- **IMD (Intelligent Model Design)**: the deep-learning side (pose, detectors, batter finder, sequence models,
  VAE technique scorer, bias audit, U-Net, TrackNet, LLM coach).
- **IVA / CSET344 (Image and Video Processing)**: the classical side. Only syllabus techniques with a genuine
  use are included; see `docs/iva/syllabus_alignment.md` and results in `docs/iva/iva_results.md`.

Team: Shikhar Srivastava (repo owner) + teammate. Current status and next steps: `PROGRESS.md`.
Working context, course setup and the research-paper direction: @docs/claude/context.md

## Hard rules
- **Never commit data or secrets.** Clips come from TV broadcasts (academic, non-commercial use only; never
  redistribute). `.env` holds the Roboflow key; Kaggle tokens live in `~/.kaggle/`. Both are gitignored.
- **Heavy compute runs on Kaggle, not the laptop.** Every Kaggle job is split into ~30-minute chunks
  (`kaggle/make_chunks.py`) and driven by `kaggle/run_chunks.sh`, so a laptop sleep or failure costs at most
  one chunk. Kaggle allows 2 GPU sessions per account at once; the runner queues extra ones.
- **Free to run.** No paid APIs or services.
- **Check model outputs visually**, not just metrics: render frames/overlays for a sample before trusting a run.
- **IVA: no forced techniques.** Add a syllabus method only if the app needs it, and measure its effect.

## Layout
| Path | What |
|---|---|
| `datasets/` | dataset download, clip standardisation (480p), label taxonomy, dedupe + match-grouped splits, detection set |
| `models/` | batter finder, stumps features, pose index, audits, quality report (local CPU scripts) |
| `kaggle/<job>/` | Kaggle job templates (script + `kernel-metadata.json`); `kaggle/chunks/` is generated |
| `kaggle/run_chunks.sh`, `kaggle/make_chunks.py`, `kaggle/set_owner.py` | chunked, resumable Kaggle runner |
| `kaggle/progress.json` | which chunks finished (on the owner's account) |
| `docs/DATASHEET.md` | sources, licences, processing, known dataset issues |
| `docs/iva/` | IVA experiment results and syllabus alignment |
| `scripts/fetch_data.sh` | download processed data (and optionally clips) from Kaggle |

## Data (not in git)
Everything lives in one private Kaggle dataset, `shikkoustic/criclens-all` (shared with collaborators):
`clips/<source>/*.mp4` (22,420 480p clips) + `manifest.parquet`, `detection/{det,seg}` (ball/bat/stumps YOLO
sets), `models/Player_Type_Detection_Model.pt`, and `criclens-processed.tgz.bin (a .tar.gz renamed so Kaggle keeps it packed)` at the top level (pose joints,
manifests, trained batter finder and detector, experiment results; extract at the repo root). Kaggle jobs
that need processed data extract that tar at the start. The older per-part datasets (`criclens-clips`,
`-pilot`, `-detection`, `-processed`, `-finder`, `-d2`, `-d3`) still exist on the owner's account for the
finished jobs.

Key files after `scripts/fetch_data.sh`:
- `data/processed/pose_index.parquet`: one row per clip; `train_ready` marks the 16,460 clips to train on
  (train 11,891 / val 2,307 / test 2,262), with shot label, split, handedness, CricketVision scores, and
  `kps_path` to the joint file.
- Joint files `kaggle/chunks/pose-*/out/kps/<clip>.npz`: `kps` (frames x 17 COCO joints x [x, y, conf]),
  `boxes` (batter box per frame), `window` ([start, contact, end] frame indices), `fps`.
- `data/processed/manifest.parquet`: all clips, labels, dedupe clusters, splits.

## Kaggle as a collaborator
Kernel ids in `kernel-metadata.json` name the owner's account. On another account run
`python kaggle/set_owner.py <kaggle-username>` first, then regenerate chunks with `kaggle/make_chunks.py`.
It also switches dataset sources to `shikkoustic/criclens-all`. The `kaggle/run_d2.sh` and `run_batcheck_d3.sh` scripts are
finished one-off pipelines kept for reference.
