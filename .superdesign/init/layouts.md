# Layouts — shared shell

Two pages share one shell: a sticky top bar on both, plus a site footer on the home page only.
There is no layout component file — the markup is duplicated verbatim in `app/static/home.html`
and `app/static/analyze.html`. Any shell change must be made in BOTH files.

## Top bar — markup (identical in both pages, only `.on` moves)

```html
<header class="topbar">
  <div class="wrap topbar-inner">
    <a class="brand" href="/" aria-label="CricLens home">
      <svg class="logo" viewBox="0 0 32 32" aria-hidden="true"><circle cx="16" cy="16" r="14"/><path d="M9 7c4 5 4 13 0 18M23 7c-4 5-4 13 0 18"/></svg>
      <span class="brand-name">CricLens</span>
    </a>
    <nav class="nav" aria-label="Primary">
      <a href="/" class="on" aria-current="page">Home</a>
      <a href="/analyze">Analyse</a>
    </nav>
    <div class="topbar-right">
      <div id="status" class="status status-loading" role="status" hidden><span class="dot"></span><span id="status-text"></span></div>
      <button id="theme-toggle" class="icon-btn" type="button" aria-label="Switch theme" title="Switch theme">
        <svg class="ico-sun" viewBox="0 0 24 24" aria-hidden="true"><circle cx="12" cy="12" r="4.5"/><path d="M12 2v2.6M12 19.4V22M2 12h2.6M19.4 12H22M4.9 4.9l1.9 1.9M17.2 17.2l1.9 1.9M19.1 4.9l-1.9 1.9M6.8 17.2l-1.9 1.9"/></svg>
        <svg class="ico-moon" viewBox="0 0 24 24" aria-hidden="true"><path d="M20 14.5A8.2 8.2 0 0 1 9.5 4a8.3 8.3 0 1 0 10.5 10.5z"/></svg>
      </button>
    </div>
  </div>
</header>
```

The CricLens mark is the inline `<svg class="logo">` above — a red ball with two seam arcs. It is the
only brand asset; there is no image file. `.logo circle` is filled `--ball`, the seam paths are white
dashed strokes.

## Top bar — CSS (verbatim)

```css
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
```

## Theme bootstrap (in the `<head>` of BOTH pages, before the stylesheet)

```html
<script>
  (() => { try {
    const saved = localStorage.getItem("criclens-theme");
    document.documentElement.dataset.theme =
      saved || (matchMedia("(prefers-color-scheme: dark)").matches ? "dark" : "light");
  } catch { document.documentElement.dataset.theme = "light"; }
  if (!(window.CSS && CSS.supports && CSS.supports("animation-timeline: view()")))
    document.documentElement.classList.add("no-sdt");
  })();
</script>
```
Runs before first paint so the theme never flashes. `no-sdt` marks browsers without scroll-driven
animations, which fall back to an IntersectionObserver in `home.js`.

## Site footer — home page only

```html
<footer class="site-footer">
  <div class="wrap footer-inner">
    <div>
      <p class="eyebrow">About</p>
      <p class="footer-about">CricLens analyses a single cricket delivery end to end …</p>
    </div>
    <div class="footer-by">
      <p class="eyebrow">Built by</p>
      <p><b>Shikhar Srivastava</b></p>
      <p><b>Vanshika Gupta</b></p>
    </div>
  </div>
</footer>
```

```css
.site-footer { position: relative; z-index: 2; background: var(--surface); border-top: 1px solid var(--border); padding: 44px 0 52px; }
.footer-inner { display: grid; grid-template-columns: 1.6fr .9fr; gap: 48px; }
.footer-about { margin: 0; color: var(--muted); max-width: 70ch; line-height: 1.65; }
.footer-by p { margin: 0 0 4px; }
.footer-by b { font-weight: 600; }
```

## Page container
`.wrap { max-width: 1180px; margin: 0 auto; padding: 0 20px; }` — every section's inner container.
