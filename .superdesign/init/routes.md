# Routes

Static HTML served by FastAPI (`app/server.py`). No router library, no client-side routing — two
full page loads.

| URL | File served | Script | Layout |
|---|---|---|---|
| `/` | `app/static/home.html` | `app/static/home.js` | topbar + site-footer |
| `/analyze` | `app/static/analyze.html` | `app/static/app.js` | topbar only |

```python
app.mount("/static", StaticFiles(directory=APP / "static"), name="static")

@app.get("/")
def home():
    return FileResponse(APP / "static/home.html")

@app.get("/analyze")
def analyze():
    return FileResponse(APP / "static/analyze.html")
```

## What each page renders

### `/` — home
A marketing/explainer page in four stages, told top to bottom:
1. **Hero** — a looping montage of real strokes full-bleed (`/api/hero.mp4`), with the headline,
   sub, two CTAs, and a fake tracking HUD (Tracking tag, 30 FPS, telemetry chips) laid over it.
2. **"Which shot was that?"** — the eight strokes as clickable buttons; clicking opens a `<dialog>`
   that plays a real held-out clip of that stroke from `/api/samples`.
3. **"How good was it?"** — a radial technique dial, plus a playable pixel-art "net" mini-game where
   you time a drive and the dial follows your timing. (Decorative; not the real scorer.)
4. **"What do you work on?"** — a sample coach's note.
Plus a decorative pixel-art pitch strip along the foot of the window, driven by scroll.

### `/analyze` — the actual tool
Three views swapped by `hidden` on `<section class="view">`:
1. `#view-input` — upload dropzone + grid of held-out sample clips, and an "Analyse shot" button.
2. `#view-progress` — a progress bar and an eight-step checklist of pipeline stages.
3. `#view-results` — results grid: video with timeline + keyframes, shot card with probability bars,
   technique card (radial gauge + per-body-part bars), movement metrics, an optional side-by-side
   comparison against a well-rated example, the coach's note, and a collapsible details table.

## Backing API (same origin)
`/api/health`, `/api/samples`, `/api/samples/{id}/thumb.jpg`, `/api/samples/{id}/video`,
`/api/hero.mp4`, `POST /api/jobs` (upload), `POST /api/jobs/sample`, `/api/jobs/{id}`,
`/api/jobs/{id}/media/{name}`.
