#!/usr/bin/env bash
# Reproducible raw-data downloads for CricLens. Usage: datasets/download.sh <source> [...]
# Everything lands in data/raw/<source>/ on local disk. Nothing is uploaded anywhere.
set -uo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"; RAW="$ROOT/data/raw"
source "$ROOT/.venv/bin/activate"

gd() { # gd <drive-file-id> <output-path> : skip if already complete
  local id="$1" out="$2"
  [ -s "$out" ] && { echo "skip $out"; return 0; }
  mkdir -p "$(dirname "$out")"
  for try in 1 2 3; do gdown -q "$id" -O "$out.part" && mv "$out.part" "$out" && { echo "ok  $out"; return 0; }; echo "retry $try $out"; sleep 20; done
  echo "FAIL $out"; return 1
}

cricshot10k_shots() { # 15 per-class RAR archives (CricShot10k, IEEE Access 2026, research/non-commercial)
  local d="$RAW/cricshot10k/archives"
  while IFS='|' read -r id name; do gd "$id" "$d/$name"; done <<'L'
1GA3d3ismeu7tDNph-1_4nsInBlXpEQrr|Cover Drive.rar
1N4M0uimTF2578i5jWEr-d4EisSPUdc_M|Defensive.rar
1ceijlIckZ5wkMb7_uLdwPUNBnUDpW18t|Down The Wicket.rar
1vyu9q3IvCjLrx9bgH_kuIpJWOtJ0fy-b|Flick.rar
1hJAHKiB_x8ECfv7B4KiJJnNqgtC7xLwp|Hook.rar
16ykROsk0I_jYUHQuDkTWMvjqlcfIkcrh|Late Cut.rar
16LiyYIIfsSIZtP2WyMm73vAm8cCpLA4H|Lofted Legside.rar
1H9-oRr1lhQ-SlryjmfEIYAww94CmMKAs|Lofted Offside.rar
1LFr7XodyInKRGkDA7Di4BSnlN0v9ucxj|Pull.rar
1nCQiGcjoj7gQOX_dygS1V0TBCTpVUFTH|Reverse Sweep.rar
124KpsJW-XLkXUZ4tAlDLP824tEeRtRs3|Scoop.rar
1fDSYfEwVwJ1xo3dl2kVKpHjXd4LPRX7V|Square Cut.rar
1yy3e-hdLHiat2xhb09aL3pdhxqtqVJ0t|Straight Drive.rar
1fS_nvnGj6eM89PW1Wb8SuHwSrjBhtaji|Sweep.rar
1jIriwWh003bqz66IREQRGlaMIpi9fTx4|Upper Cut.rar
L
}

cricshot10k_models() { # authors' pretrained YOLO models (ball, bat, bat segmentation, player type)
  local d="$RAW/cricshot10k/models"
  gd 1EBaPnp5OHAmSNgOideWFf0QiBefsNse9 "$d/Ball_Detection_Model.pt"
  gd 18dw6se7R3cOuifSd3dhRp80bxTMyyg1A "$d/Bat_Detection_Model.pt"
  gd 1MFi1TuiNaY3DiCi1yWLcwX7O9HQ9nsBH "$d/Striker_Bat_Segmentation_Model.pt"
  gd 1XzWkOrIs9lpuE14XWv_H9gZxP6-VyGW8 "$d/Player_Type_Detection_Model.pt"
}

cricshot10() { # CricShot10 (Sensors 2021) via HF mirror rokmr/cricket-shot, 10 classes
  hf download rokmr/cricket-shot --repo-type dataset --local-dir "$RAW/cricshot10" --quiet && echo "ok  cricshot10"
}

cricketvision_ann() { # CricketVision (WACV 2025) annotations: scores, handedness, foot, phases
  local out="$RAW/cricketvision/JSON.zip"; mkdir -p "$(dirname "$out")"
  [ -s "$out" ] || curl -sL --fail -o "$out" "https://www.dropbox.com/scl/fo/3mvm1t9ru8qljgwm2m00d/AP2mRYapSp4Viz5A4mNwivw?rlkey=mdofxvrynywq20pamqal0xzwl&dl=1"
  echo "ok  $out"
}

kaggle_sets() { # needs ~/.kaggle/access_token (new KGAT_ tokens) or legacy ~/.kaggle/kaggle.json
  if [ -f ~/.kaggle/access_token ]; then chmod 600 ~/.kaggle/access_token
  elif [ -f ~/.kaggle/kaggle.json ]; then chmod 600 ~/.kaggle/kaggle.json
  else echo "missing ~/.kaggle/access_token (or kaggle.json)"; return 1; fi
  kaggle datasets download -d taarunsridhar/cricket-shots-ipl-2023 -p "$RAW/ipl2023" && echo "ok  ipl2023"
  kaggle datasets download -d amittalmale/cricket-shots -p "$RAW/amittalmale" && echo "ok  amittalmale"
  kaggle datasets download -d kushagra3204/cricket-ball-dataset-for-yolo -p "$RAW/detection/kaggle_ball" && echo "ok  kaggle_ball"
}

roboflow_sets() { # needs ROBOFLOW_API_KEY in .env (free account -> Settings -> API Keys)
  set -a; [ -f "$ROOT/.env" ] && source "$ROOT/.env"; set +a
  [ -n "${ROBOFLOW_API_KEY:-}" ] || { echo "missing ROBOFLOW_API_KEY in .env"; return 1; }
  python - "$RAW/detection" <<'PY'
import os, sys
from roboflow import Roboflow
out = sys.argv[1]
rf = Roboflow(api_key=os.environ["ROBOFLOW_API_KEY"])
SETS = [  # (workspace, project, why)
    # ("naveen-akash", "lbw-9pb1e", ...) skipped: owner never generated a version, so it is not downloadable via API
    ("powerinflow-4iw3s", "powerinflow", "ball/person/bat/stump polygons"),
    ("computer-vision-d3h0p", "cricket-ball-segmentation", "ball polygons"),
    ("cricket-ball-tracking-dataset", "cricket-dataset-z2wkt", "ball/stump boxes"),
    ("cricket-rfyd8", "cricket-balls-l1us5", "ball/bat/batsman boxes"),
    ("cricketbatting-kvxyo", "cricket-bat-detection", "bat boxes"),
]
for ws, pj, why in SETS:
    try:
        proj = rf.workspace(ws).project(pj)
        v = max(int(x.version.split("/")[-1]) for x in proj.versions())
        fmt = "yolov8" 
        proj.version(v).download(fmt, location=f"{out}/{pj}", overwrite=False)
        print(f"ok  {ws}/{pj} v{v} ({why})")
    except Exception as e:
        print(f"FAIL {ws}/{pj}: {e!r}"[:300])
PY
}

kucricshot() { # 1,278 loose mp4s in 4 public Drive folders; one request per file, paced
  local d="$RAW/kucricshot"
  python - "$d" <<'PY'
import re, sys, time, html, urllib.request, subprocess, os
out = sys.argv[1]
FOLDERS = {"Defensive Shot": "1PMGmJ77Zk0zfWQ09kXkCaWH80Q4AFNRh", "Drive Shot": "15s4gaiSNENwOE27OamSYkTkg8pE_Y-L8",
           "Flick Shot": "1KKyR-xDsMdr0lo1fvO-8xCEfPMF4EZov", "Pull Shot": "1NpaG61dWrt5KUDxciUBd75xB-VYZ_Lc4"}
for cls, fid in FOLDERS.items():
    req = urllib.request.Request(f"https://drive.google.com/embeddedfolderview?id={fid}", headers={"User-Agent": "Mozilla/5.0"})
    page = urllib.request.urlopen(req, timeout=60).read().decode("utf-8", "ignore")
    items = re.findall(r'<a href="https://drive\.google\.com/file/d/([\w-]+)[^"]*"[^>]*>.*?<div class="flip-entry-title">(.*?)</div>', page, re.S)
    os.makedirs(f"{out}/{cls}", exist_ok=True)
    def fetch(item):
        file_id, name = item
        dst = f"{out}/{cls}/{html.unescape(name)}"
        if os.path.exists(dst) and os.path.getsize(dst) > 0:
            return True
        for attempt in range(3):  # retry transient Drive errors with backoff
            if subprocess.run(["gdown", "-q", file_id, "-O", dst]).returncode == 0:
                return True
            time.sleep(5 * (attempt + 1))
        return False
    from concurrent.futures import ThreadPoolExecutor
    with ThreadPoolExecutor(8) as ex:  # files are ~280 KB, so per-request latency dominates
        got = sum(ex.map(fetch, items))
    print(f"ok  kucricshot/{cls}: {got}/{len(items)}", flush=True)
PY
}

for s in "$@"; do echo "=== $s $(date +%T)"; "$s"; done
echo "=== done $(date +%T)"
