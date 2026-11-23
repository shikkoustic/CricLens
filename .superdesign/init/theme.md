# Theme

## Part 1 — Compact token summary

Single stylesheet: `app/static/styles.css` (759 lines). Light is the default; dark is opt-in via
`:root[data-theme="dark"]`, set by an inline head script. There is no Tailwind and no config file.

**Concept:** "the ground the game is played on" — field green and the cream of flannels. Dark is the
same match under cloud; light is the same colours on paper.

### Colour — light (`:root`) → dark (`[data-theme="dark"]`)
| Token | Light | Dark | Role |
|---|---|---|---|
| `--bg` | `#f2efe6` | `#121d16` | page background |
| `--surface` | `#fbf9f3` | `#1a271e` | cards, top bar, footer |
| `--surface-2` | `#e9e5d8` | `#233225` | hover fills, insets |
| `--text` | `#16231a` | `#f2efe6` | primary ink |
| `--muted` | `#5c6a5e` | `#a6b2a6` | secondary ink, eyebrows |
| `--border` | `#ded9c9` | `#2d3c31` | hairlines |
| `--accent` | `#1d6b45` | `#6fc58d` | primary action, gauges |
| `--accent-soft` | `#e1ece4` | `#1b3826` | accent tint |
| `--accent-text` | `#fbf9f3` | `#0b1710` | ink on accent |
| `--ball` | `#b4291c` | `#ef5a4c` | cricket-ball red, logo, seams |
| `--clay` | `#c2a473` | `#b2946a` | pitch clay |
| `--floodlight` | `#c98c14` | `#efc168` | warm highlight |
| `--good` / `--improve` / `--caution` | `#1d6b45` / `#b54708` / `#8a6100` | `#6fc58d` / `#f79b52` / `#f0c14b` | status |
| `--win` | `#e0532f` | `#ff6a45` | analysed-window marker |

### Glass + over-footage tokens
`--glass: rgba(251,249,243,.66)` → dark `rgba(35,50,39,.52)`; `--glass-2` is the opaquer step;
`--glass-brd`; `--glass-blur: 18px`.
For chrome sitting directly on video in either theme: `--on-media: #f6f3ea`,
`--on-media-muted: rgba(246,243,234,.74)`, `--on-media-glass: rgba(18,29,22,.38)`,
`--on-media-brd: rgba(246,243,234,.26)`.

### Typography
- `--font: "Poppins"` (400/500/600) — UI and body
- `--font-display: "Playfair Display"` (500/600, italic 500) — hero and section headings
- `--font-mono: ui-monospace, SFMono-Regular, "SF Mono", Menlo` — telemetry readouts
- Loaded from Google Fonts in both page heads.
- Scale: body 15px/1.55; hero `clamp(38px, 5.4vw, 64px)`; stage titles `clamp(28px, 3.6vw, 44px)`;
  eyebrow 12px uppercase `letter-spacing: .06em`.

### Radius / shadow / layout
`--radius: 16px`, `--radius-lg: 26px`, `--radius-pill: 999px`.
`--shadow` (resting) and `--shadow-lg` (lifted) — both re-specified for dark.
Container `.wrap` = `max-width: 1180px; padding: 0 20px`. Top bar height 60px.
Breakpoints used: `960px` (stage/footer split → single column), `860px`, `560px`.

### Pixel-art sprite palette
`--px-ink`, `--px-cap`, `--px-skin`, `--px-kit`, `--px-kit2`, `--px-pad`, `--px-wood`, `--px-ball`,
`--px-seam` — drive the inline `<symbol>` sheet; rendered `image-rendering: pixelated`.

### Motion
`.reveal` → `.in` fade-and-rise on scroll (IntersectionObserver, or CSS `animation-timeline: view()`
where supported — `html.no-sdt` marks the fallback). All motion is wrapped in
`@media (prefers-reduced-motion: no-preference)` or checked in JS.

## Part 2 — Raw token source

`app/static/styles.css:1:78` (verbatim):

```css
/* Light is the default. The theme is set on <html data-theme> by an inline script in each
   page head (stored choice, else the OS preference), so there is exactly one dark block.

   The palette is the ground the game is played on: field green and the cream of flannels.
   Dark is the match-under-cloud version, light is the same colours on paper. */
:root {
  --bg: #f2efe6;
  --surface: #fbf9f3;
  --surface-2: #e9e5d8;
  --text: #16231a;
  --muted: #5c6a5e;
  --border: #ded9c9;
  --accent: #1d6b45;
  --accent-soft: #e1ece4;
  --accent-text: #fbf9f3;
  --ball: #b4291c;
  --ball-soft: #fbeae7;
  --clay: #c2a473;
  --clay-soft: #f3e9d6;
  --floodlight: #c98c14;
  --good: #1d6b45;
  --good-soft: #e1ece4;
  --improve: #b54708;
  --improve-soft: #fdf0e3;
  --caution: #8a6100;
  --caution-soft: #fdf5dd;
  --win: #e0532f;

  /* glass: translucent panels that let the field show through */
  --glass: rgba(251, 249, 243, .66);
  --glass-2: rgba(251, 249, 243, .82);
  --glass-brd: rgba(22, 35, 26, .12);
  --glass-blur: 18px;
  /* type and chrome that sit directly on footage, in either theme */
  --on-media: #f6f3ea;
  --on-media-muted: rgba(246, 243, 234, .74);
  --on-media-glass: rgba(18, 29, 22, .38);
  --on-media-brd: rgba(246, 243, 234, .26);

  --radius: 16px;
  --radius-lg: 26px;
  --radius-pill: 999px;
  --shadow: 0 1px 2px rgba(22, 35, 26, .05), 0 6px 20px rgba(22, 35, 26, .06);
  --shadow-lg: 0 18px 48px rgba(22, 35, 26, .14);
  --font: "Poppins", -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, "Helvetica Neue", Arial, sans-serif;
  --font-display: "Playfair Display", "Iowan Old Style", "Palatino Linotype", Georgia, serif;
  --font-mono: ui-monospace, SFMono-Regular, "SF Mono", Menlo, monospace;
}
:root[data-theme="dark"] {
  --bg: #121d16;
  --surface: #1a271e;
  --surface-2: #233225;
  --text: #f2efe6;
  --muted: #a6b2a6;
  --border: #2d3c31;
  --accent: #6fc58d;
  --accent-soft: #1b3826;
  --accent-text: #0b1710;
  --ball: #ef5a4c;
  --ball-soft: #38191a;
  --clay: #b2946a;
  --clay-soft: #2b2419;
  --floodlight: #efc168;
  --good: #6fc58d;
  --good-soft: #163426;
  --improve: #f79b52;
  --improve-soft: #3a2616;
  --caution: #f0c14b;
  --caution-soft: #362d12;
  --win: #ff6a45;

  --glass: rgba(35, 50, 39, .52);
  --glass-2: rgba(35, 50, 39, .74);
  --glass-brd: rgba(242, 239, 230, .14);
  --shadow: 0 1px 2px rgba(0, 0, 0, .34), 0 6px 20px rgba(0, 0, 0, .28);
  --shadow-lg: 0 20px 56px rgba(0, 0, 0, .45);
}

```
