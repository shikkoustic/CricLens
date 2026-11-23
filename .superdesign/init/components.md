# Components — shared UI primitives

CricLens has **no component framework**: it is hand-written HTML + a single `app/static/styles.css`.
"Components" here are CSS classes applied to plain markup, so each entry is the markup pattern plus
the rules that style it. Everything is token-driven (`var(--…)`, see `theme.md`).

Source of all CSS below: `app/static/styles.css`.

## Core primitive CSS (verbatim)

```css
.wrap { max-width: 1180px; margin: 0 auto; padding: 0 20px; }
.muted { color: var(--muted); }
.small { font-size: 13px; }
.eyebrow { margin: 0 0 4px; font-size: 12px; font-weight: 600; letter-spacing: .06em; text-transform: uppercase; color: var(--muted); }

/* top bar */
.topbar { position: sticky; top: 0; z-index: 5; background: color-mix(in srgb, var(--bg) 88%, transparent); backdrop-filter: blur(8px); border-bottom: 1px solid var(--border); }
.topbar-inner { display: flex; align-items: center; justify-content: space-between; gap: 16px; height: 60px; }
.brand { display: flex; align-items: center; gap: 10px; color: inherit; text-decoration: none; min-width: 0; }
.logo { width: 28px; height: 28px; flex: none; }
.logo circle { fill: var(--ball); }
.logo path { fill: none; stroke: #fff; stroke-width: 1.6; stroke-dasharray: 2 2; }
.brand-name { font-weight: 700; font-size: 17px; }
.brand-tag { color: var(--muted); font-size: 13px; white-space: nowrap; overflow: hidden; text-overflow: ellipsis; }
.nav { display: flex; align-items: center; gap: 4px; margin-right: auto; margin-left: 28px; }
.nav a { padding: 7px 12px; border-radius: 8px; color: var(--muted); text-decoration: none; font-size: 14px; font-weight: 600; transition: background .15s, color .15s; }
.nav a:hover { color: var(--text); background: var(--surface-2); }
.nav a.on { color: var(--text); background: var(--surface); border: 1px solid var(--border); }
.status { display: inline-flex; align-items: center; gap: 8px; padding: 5px 12px; border-radius: 999px; font-size: 13px; background: var(--surface); border: 1px solid var(--border); white-space: nowrap; }
.status .dot { width: 8px; height: 8px; border-radius: 50%; background: var(--caution); }
.status-loading .dot { animation: pulse 1.2s ease-in-out infinite; }
.status-ready .dot { background: var(--good); }
.status-error .dot { background: var(--ball); }
@keyframes pulse { 50% { opacity: .3; } }

main { padding: 32px 20px 24px; }
body.home main { padding: 0 0 24px; }
.view { animation: fade .25s ease; }
@keyframes fade { from { opacity: 0; transform: translateY(4px); } }

/* cards */
.card { background: var(--surface); border: 1px solid var(--border); border-radius: var(--radius); box-shadow: var(--shadow); padding: 20px; min-width: 0; }
.card-head { display: flex; align-items: center; justify-content: space-between; gap: 12px; margin-bottom: 12px; position: relative; }
.card-title { font-size: 16px; font-weight: 650; margin-bottom: 14px; }
.card-head .card-title, .card-head .eyebrow { margin-bottom: 0; }

/* buttons */
.btn { display: inline-flex; align-items: center; justify-content: center; gap: 10px; height: 44px; padding: 0 20px;
  border-radius: var(--radius-pill); border: 1px solid var(--border); background: var(--surface);
  font: inherit; font-weight: 600; color: var(--text); text-decoration: none; cursor: pointer;
  transition: background .18s ease, border-color .18s ease, transform .06s ease, box-shadow .18s ease; }
.btn:hover { background: var(--surface-2); }
.btn:active { transform: translateY(1px); }
.btn:focus-visible { outline: 2px solid var(--accent); outline-offset: 2px; }
.btn-primary { background: var(--accent); border-color: var(--accent); color: var(--accent-text); min-width: 180px; }
.btn-primary:hover { background: color-mix(in srgb, var(--accent) 88%, #000); }
.btn:disabled { opacity: .45; cursor: not-allowed; }

/* big pills with the arrow badge sitting in the right shoulder */
.btn-lg { height: 56px; font-size: 16px; padding: 0 10px 0 28px; letter-spacing: -.005em; }
.btn-lg:not(:has(.btn-arrow)) { padding: 0 28px; }
.btn-arrow { flex: none; display: grid; place-items: center; width: 38px; height: 38px; border-radius: 50%;
  background: var(--arrow-bg, var(--accent-text)); color: var(--arrow-fg, var(--accent));
  transition: transform .22s cubic-bezier(.2,.8,.3,1); }
.btn-arrow svg { width: 15px; height: 15px; fill: none; stroke: currentColor; stroke-width: 2.1; stroke-linecap: round; stroke-linejoin: round; }
.btn:hover .btn-arrow { transform: translate(2px, -2px); }
.btn-ghost { background: transparent; }

/* buttons that sit directly on footage: cream pill, glass pill */
.btn-on-media { --arrow-bg: #16231a; --arrow-fg: #f6f3ea;
  background: var(--on-media); border-color: var(--on-media); color: #16231a; }
.btn-on-media:hover { background: #fff; }
.btn-glass { --arrow-bg: rgba(246, 243, 234, .9); --arrow-fg: #16231a;
  background: var(--on-media-glass); border-color: var(--on-media-brd); color: var(--on-media);
  backdrop-filter: blur(var(--glass-blur)) saturate(1.3); -webkit-backdrop-filter: blur(var(--glass-blur)) saturate(1.3); }
.btn-glass:hover { background: rgba(18, 29, 22, .52); border-color: rgba(246, 243, 234, .4); }
.topbar-right { display: flex; align-items: center; gap: 10px; }
.icon-btn { display: grid; place-items: center; width: 38px; height: 38px; border-radius: 10px; cursor: pointer;
  background: var(--surface); border: 1px solid var(--border); color: var(--muted); transition: color .15s, background .15s, border-color .15s; }
.icon-btn:hover { color: var(--text); background: var(--surface-2); }
.icon-btn svg { width: 18px; height: 18px; fill: none; stroke: currentColor; stroke-width: 1.9; stroke-linecap: round; }
.icon-btn .ico-moon { display: none; }
:root[data-theme="dark"] .icon-btn .ico-sun { display: none; }
:root[data-theme="dark"] .icon-btn .ico-moon { display: block; }

/* ============================================================
   Hero: the footage runs full bleed, the words sit on it
.shot-line { display: flex; align-items: center; gap: 10px; flex-wrap: wrap; margin-bottom: 14px; }
.shot-name { font-size: 26px; }
.tag { font-size: 12px; font-weight: 600; padding: 3px 9px; border-radius: 99px; background: var(--good-soft); color: var(--good); }
.tag.unsure { background: var(--caution-soft); color: var(--caution); }
.badge { font-size: 12px; padding: 3px 10px; border-radius: 99px; background: var(--accent-soft); color: var(--accent); font-weight: 600; }
```

## Markup patterns

### Button
```html
<a class="btn btn-primary btn-lg" href="/analyze">Analyse a clip
  <span class="btn-arrow"><svg viewBox="0 0 24 24" aria-hidden="true"><path d="M7 17 17 7M9 7h8v8"/></svg></span></a>
```
Variants: `.btn` (default), `.btn-primary` (accent fill), `.btn-lg`, `.btn-sm`, `.btn-ghost`,
`.btn-on-media` (white pill for use over footage), `.btn-glass` (frosted, over footage).
`.btn-arrow` is a circular icon badge that nudges on hover.

### Card
```html
<article class="card">
  <div class="card-head"><p class="eyebrow">Shot</p><span class="badge">Rule-based</span></div>
  <h2 class="card-title">Upload a clip</h2>
</article>
```

### Glass panel
`.glass-panel` — translucent surface using `--glass`, `--glass-brd`, `backdrop-filter: blur(var(--glass-blur))`.
Used for the technique dial panel and the coach bubble on the home page.

### Chip (telemetry, over footage)
```html
<span class="chip"><b>Stride</b><span class="mono">1.12 m</span></span>
```

### Eyebrow / section kicker
```html
<p class="eyebrow">First</p>
```
Uppercase, letter-spaced, `--muted`. Used above every section heading and card title.

### Status pill
```html
<div id="status" class="status status-loading" role="status" hidden><span class="dot"></span><span id="status-text"></span></div>
```
States: `.status-loading` (pulsing dot), `.status-ready`, `.status-error`. Hidden once models are ready.

### Icon button
```html
<button class="icon-btn" type="button" aria-label="Switch theme">
  <svg class="ico-sun" ...></svg><svg class="ico-moon" ...></svg>
</button>
```
38×38 square, shows sun in light theme and moon in dark via `:root[data-theme="dark"]`.

### Tag / badge
`.tag` (good) and `.tag.unsure` (caution) for shot confidence; `.badge` for small accent labels.

### Radial gauge (technique score)
```html
<div class="ring"><svg viewBox="0 0 120 120">
  <circle class="ring-bg" cx="60" cy="60" r="52"/>
  <circle id="ring-fg" class="ring-fg" cx="60" cy="60" r="52"/>
  <path class="ring-seam" d="M26 26c14 16 14 52 0 68"/><path class="ring-seam" d="M94 26c-14 16-14 52 0 68"/>
</svg><div class="ring-text"><strong id="tech-score"></strong><span>/ 10</span></div></div>
```
The two `.ring-seam` paths are cricket-ball seam stitching round the rim. The home page uses the
same idea at a larger size as `.dial` / `.dial-fg` / `.dial-seam`.

### Pixel-art sprite
```html
<svg class="px s-ball" viewBox="0 0 6 6"><use href="#px-ball"/></svg>
```
Sprites live in one inline `<svg id="px-sheet">` symbol sheet at the top of `home.html`:
`#px-batter`, `#px-player`, `#px-bat`, `#px-arm`, `#px-wicket`, `#px-bail`, `#px-bubble`, `#px-ball`.
They are coloured by `--px-ink`, `--px-cap`, `--px-skin`, `--px-kit`, `--px-kit2`, `--px-pad`,
`--px-wood`, `--px-ball`, `--px-seam`, and rendered with `image-rendering: pixelated`.
