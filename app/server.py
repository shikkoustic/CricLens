"""CricLens web app: FastAPI backend + static single-page frontend.

    python -m app.server            # http://127.0.0.1:8000  (CRICLENS_HOST / CRICLENS_PORT to change)

Models load once in the background at startup; jobs run one at a time on a single worker thread (the
pose model and the LLM are CPU-heavy, so parallel jobs would only slow each other down).
"""
import os
import queue
import re
import threading
import time
import traceback
import uuid
from pathlib import Path

import cv2
import pandas as pd
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

APP = Path(__file__).resolve().parent
ROOT = APP.parent
RUNTIME = APP / "runtime"
JOBS_DIR, THUMBS = RUNTIME / "jobs", RUNTIME / "thumbs"
SAMPLES_DIR = Path(os.environ.get("CRICLENS_SAMPLES_DIR", ROOT / "data/interim/clips"))
MAX_UPLOAD = 150 * 1024 * 1024
VIDEO_EXT = {".mp4", ".mov", ".avi", ".mkv", ".webm", ".m4v"}
MEDIA_NAME = re.compile(r"^[a-z_]+\.(mp4|jpg)$")
STAGES = [("read", "Reading video"), ("batter", "Finding the batter"), ("pose", "Tracking body joints"),
          ("shot", "Classifying the shot"), ("technique", "Scoring technique"), ("metrics", "Measuring stride & hand speed"),
          ("render", "Drawing the overlay"), ("coach", "Writing coaching feedback")]

for d in (JOBS_DIR, THUMBS):
    d.mkdir(parents=True, exist_ok=True)

state = {"ready": False, "stage": "Starting", "error": None, "llm": None}
models = {}
jobs: dict[str, dict] = {}
jobs_lock = threading.Lock()
work: queue.Queue = queue.Queue()


def load_models():
    try:
        from app.coach import Coach
        from app.pipeline import Analyzer
        models["analyzer"] = Analyzer(status=lambda m: state.update(stage=m))
        models["coach"] = Coach(status=lambda m: state.update(stage=m))
        state.update(ready=True, stage="Ready", llm=models["coach"].name)
    except Exception as e:
        traceback.print_exc()
        state.update(error=f"Model loading failed: {e}")


def update(job_id, **kw):
    with jobs_lock:
        jobs[job_id].update(kw)


def worker():
    from app.pipeline import AnalysisError
    while True:
        job_id = work.get()
        job = jobs[job_id]
        while not state["ready"] and not state["error"]:
            time.sleep(0.5)
        if state["error"]:
            update(job_id, status="error", error="The analysis models failed to load on the server.")
            continue
        update(job_id, status="running", started=time.time())
        try:
            out = JOBS_DIR / job_id
            result = models["analyzer"].analyze(job["input"], out, lambda s, f: update(job_id, stage=s, progress=f))
            update(job_id, stage="coach", progress=0.88)
            result["coach"] = models["coach"].write(result)
            update(job_id, status="done", stage="done", progress=1.0, result=result, finished=time.time())
        except AnalysisError as e:
            update(job_id, status="error", error=str(e))
        except Exception:
            traceback.print_exc()
            update(job_id, status="error", error="Something went wrong while analysing this clip.")
        finally:
            if job.get("uploaded"):
                Path(job["input"]).unlink(missing_ok=True)


def discover_samples() -> dict[str, dict]:
    labels = {}
    mf = ROOT / "data/processed/manifest.parquet"
    if mf.exists():
        m = pd.read_parquet(mf, columns=["clip_id", "shot"])
        labels = dict(zip(m.clip_id, m.shot))
    out = {}
    for p in sorted(SAMPLES_DIR.glob("*/*.mp4")):
        sid = f"{p.parent.name}--{p.stem}"
        out[sid] = {"id": sid, "path": p, "source": p.parent.name, "label": labels.get(p.stem)}
    return out


SAMPLES = discover_samples()


def new_job(input_path: Path, name: str, uploaded: bool) -> str:
    job_id = uuid.uuid4().hex[:12]
    with jobs_lock:
        jobs[job_id] = {"id": job_id, "name": name, "input": str(input_path), "uploaded": uploaded,
                        "status": "queued", "stage": "read", "progress": 0.0, "created": time.time()}
    work.put(job_id)
    return job_id


app = FastAPI(title="CricLens")


@app.on_event("startup")
def startup():
    threading.Thread(target=load_models, daemon=True).start()
    threading.Thread(target=worker, daemon=True).start()


@app.get("/api/health")
def health():
    return {**state, "stages": [{"key": k, "label": v} for k, v in STAGES]}


@app.get("/api/samples")
def samples():
    return [{"id": s["id"], "source": s["source"], "label": s["label"], "thumb": f"/api/samples/{s['id']}/thumb.jpg"}
            for s in SAMPLES.values()]


@app.get("/api/samples/{sample_id}/thumb.jpg")
def sample_thumb(sample_id: str):
    s = SAMPLES.get(sample_id)
    if s is None:
        raise HTTPException(404)
    thumb = THUMBS / f"{sample_id}.jpg"
    if not thumb.exists():
        cap = cv2.VideoCapture(str(s["path"]))
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(cap.get(cv2.CAP_PROP_FRAME_COUNT) * 0.3))
        ok, f = cap.read(); cap.release()
        if not ok:
            raise HTTPException(404)
        h, w = f.shape[:2]
        cv2.imwrite(str(thumb), cv2.resize(f, (320, int(320 * h / w))), [cv2.IMWRITE_JPEG_QUALITY, 82])
    return FileResponse(thumb, media_type="image/jpeg")


@app.get("/api/samples/{sample_id}/video")
def sample_video(sample_id: str):
    s = SAMPLES.get(sample_id)
    if s is None:
        raise HTTPException(404)
    return FileResponse(s["path"], media_type="video/mp4")


class SampleReq(BaseModel):
    sample_id: str


@app.post("/api/jobs/sample")
def create_sample_job(req: SampleReq):
    s = SAMPLES.get(req.sample_id)
    if s is None:
        raise HTTPException(404, "Unknown sample")
    return {"job_id": new_job(s["path"], f"Sample: {s['source']}", uploaded=False)}


@app.post("/api/jobs")
async def create_upload_job(file: UploadFile = File(...)):
    ext = Path(file.filename or "").suffix.lower()
    if ext not in VIDEO_EXT:
        raise HTTPException(400, "Please upload a video file (MP4, MOV, AVI, MKV or WebM).")
    dest = RUNTIME / f"upload_{uuid.uuid4().hex}{ext}"
    size = 0
    with open(dest, "wb") as fh:
        while chunk := await file.read(1 << 20):
            size += len(chunk)
            if size > MAX_UPLOAD:
                fh.close(); dest.unlink(missing_ok=True)
                raise HTTPException(413, "That file is over 150 MB. Trim it to the one delivery you want analysed.")
            fh.write(chunk)
    name = re.sub(r"[^\w .()-]", "", Path(file.filename).name)[:80] or "Uploaded clip"
    return {"job_id": new_job(dest, name, uploaded=True)}


@app.get("/api/jobs/{job_id}")
def get_job(job_id: str):
    with jobs_lock:
        job = jobs.get(job_id)
        if job is None:
            raise HTTPException(404)
        view = {k: v for k, v in job.items() if k not in ("input",)}
        if job["status"] == "queued":
            view["queue_position"] = sum(1 for j in jobs.values() if j["status"] == "queued" and j["created"] <= job["created"])
    return view


@app.get("/api/jobs/{job_id}/media/{name}")
def job_media(job_id: str, name: str):
    if job_id not in jobs or not MEDIA_NAME.match(name):
        raise HTTPException(404)
    p = JOBS_DIR / job_id / name
    if not p.is_file():
        raise HTTPException(404)
    return FileResponse(p, media_type="video/mp4" if name.endswith(".mp4") else "image/jpeg")


app.mount("/static", StaticFiles(directory=APP / "static"), name="static")


@app.get("/")
def index():
    return FileResponse(APP / "static/index.html")


def main():
    import uvicorn
    uvicorn.run(app, host=os.environ.get("CRICLENS_HOST", "127.0.0.1"), port=int(os.environ.get("CRICLENS_PORT", "8000")))


if __name__ == "__main__":
    main()
