# Teammate setup: from zero to continuing the project

Everything below takes about 20 minutes, most of it the data download.

## 0. What Shikhar does first (once)
1. GitHub: repo `CricLens` → **Settings → Collaborators → Add people** → your GitHub username. Accept the
   invite from your email or github.com/notifications.
2. Kaggle: open kaggle.com/datasets/shikkoustic/criclens-all → **Settings → Sharing** → add your Kaggle
   username. That one private dataset holds everything (clips, detection images, processed data, models).

## 1. Tools (Mac)
```bash
brew install git python@3.12 uv ffmpeg
```
Windows/Linux: install Git, Python 3.12, `uv` (https://docs.astral.sh/uv/) and ffmpeg the usual way.

## 2. Clone and install
```bash
git clone git@github.com:shikkoustic/CricLens.git
cd CricLens
uv venv --python 3.12 .venv
source .venv/bin/activate
uv pip install -r requirements.txt
```
(If you have not set up an SSH key with GitHub, use `https://github.com/shikkoustic/CricLens.git` instead.)

## 3. Your own Kaggle account
- Use **your own** Kaggle account (one account per person is Kaggle's rule). Verify it with a phone number so
  GPUs are enabled (Settings → Phone verification).
- Create an API token: kaggle.com → Settings → API → **Create New Token**. Save it with the command Kaggle
  shows, which writes `~/.kaggle/access_token`. Never paste the token into chat or commit it.
- Point the Kaggle jobs at your account:
```bash
python kaggle/set_owner.py <your-kaggle-username>
```

## 4. Get the data
```bash
scripts/fetch_data.sh          # processed data, ~240 MB: joints, manifests, models, results
scripts/fetch_data.sh --all    # optional: everything, ~6 GB (adds clips + detection images; only for video steps)
```
Check it worked:
```bash
python -c "import pandas as pd; d=pd.read_parquet('data/processed/pose_index.parquet'); print(len(d), int(d.train_ready.sum()))"
```
Expected output: `22420 16460`.

## 5. Where we left off
Read `PROGRESS.md` (status and next steps) and `CLAUDE.md` (project context and rules). With Claude Code,
just open the repo folder and ask it to read both; `CLAUDE.md` is loaded automatically.

## 6. Working together
- Pull before you start: `git pull`. Work on a branch: `git checkout -b <your-name>/<task>`, push it, and
  merge through a pull request so we don't overwrite each other.
- Commit code and docs only. Data stays on Kaggle; if you produce new processed data, tell Shikhar so it can be
  added to `criclens-processed`.
- Agree who runs which Kaggle jobs. Each account has its own 30 GPU hours a week, so splitting jobs between
  our two accounts is allowed and doubles throughput.
