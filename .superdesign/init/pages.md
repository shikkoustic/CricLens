# Page dependency trees

No bundler and no imports — each page is one HTML file plus the shared stylesheet and its own
script. The "tree" is therefore flat and the whole set fits the payload budget comfortably
(1961 lines across all five files).

## / (Home)
Entry: `app/static/home.html` (244 lines)
Dependencies:
- `app/static/styles.css` (759 lines — shared with /analyze)
- `app/static/home.js` (417 lines — theme toggle, hero clip, scroll reveals, technique dial,
  the net mini-game, the scroll-driven pitch strip, the stroke clip dialog, health polling)
- Inline in `home.html`: the `<svg id="px-sheet">` pixel-art symbol sheet (8 symbols), the
  theme-bootstrap script, the inline logo SVG
- Runtime data: `GET /api/hero.mp4` (hero montage), `GET /api/samples` + `/api/samples/{id}/video`
  (stroke dialog), `GET /api/health` (warming-up pill)
- Fonts: Google Fonts — Playfair Display + Poppins

## /analyze (Analyse a clip)
Entry: `app/static/analyze.html` (183 lines)
Dependencies:
- `app/static/styles.css` (759 lines — shared with /)
- `app/static/app.js` (358 lines — upload + sample picking, job polling, progress steps, and all
  result rendering: timeline, keyframes, shot probabilities, technique gauge and part bars,
  movement metrics, comparison, coach's note, details table)
- Runtime data: `GET /api/health`, `GET /api/samples`, `POST /api/jobs` / `POST /api/jobs/sample`,
  `GET /api/jobs/{id}`, `GET /api/jobs/{id}/media/{name}`
- Fonts: Google Fonts — Playfair Display + Poppins

## Context selection note
For a design call on either page, pass: the page HTML (full), `app/static/styles.css` (full — under
the ~900-line threshold), `.superdesign/design-system.md`, and the page's JS only when the design
depends on behaviour it drives (e.g. the net mini-game or the results renderer). `home.html`'s
pixel-art symbol sheet is one very long line — it is data, not layout, and can be summarised rather
than passed when the budget is tight.
