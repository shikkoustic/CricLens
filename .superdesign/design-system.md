# CricLens — Design System v2

**Direction: Apple-grade presentation, light and natural.**

Replaces v1 (cream + forest green), which the user rejected. Two inputs:
1. **Apple's design language** — one idea per screen, enormous tight-tracked type, near-black
   canvas, a single chroma used sparingly, slow smooth scroll reveals, almost no chrome.
   (Note: `extract-website` on apple.com returned nothing — the site blocks the crawler — so this
   is encoded from its observable, well-established language, not from a scrape.)
2. **The user's own brief** — obsidian slate base, electric emerald for performance, icy blue for
   secondary data. TrackMan / Hawk-Eye / Whoop territory.

---

## 1. Product context

CricLens analyses **one cricket delivery** from a short clip, entirely on the user's machine:
find the striker → track 17 joints → classify the shot (8 strokes) → score technique (0–10 plus five
body-part ratings) → measure stride and hand speed → write a coaching note.

Built by Shikhar Srivastava and Vanshika Gupta. It is instrumentation, not marketing — but it should
feel like a *product*, the way Apple makes hardware feel inevitable.

### Pages
- **`/` Home** — hero (looping montage of real strokes) → "Which shot was that?" (8 strokes,
  clickable to play a held-out clip) → "How good was it?" (technique dial + playable pixel-art net
  mini-game) → "What do you work on?" (coach's note) → footer.
- **`/analyze`** — upload/sample picker → 8-step progress → results (video + timeline + keyframes,
  shot probabilities, technique gauge + part bars, movement metrics, comparison, coach's note).

### Hard constraints — do not design these away
- **The net mini-game stays.** The user explicitly wants it kept. Re-house it; don't restyle it out.
- **Pixel-art sprites stay**, `image-rendering: pixelated`, never blurred or smoothed.
- **Honesty UI stays**: "Unsure · 53%" tag, amber warnings, "What this is based on", "Not available"
  for uncalibrated metrics. The project's own research says the numbers have limits.
- **No framework.** Hand-written HTML + one `styles.css`. No build step, no Tailwind, no JS libs.
- **Light is the default**, authored light-first in Apple's bright register. Dark remains available
  via the existing toggle and must be fully specified.

---

## 2. Colour

**Light-first, the way Apple's light pages are: white and a soft warm grey, near-black text, and
colour used sparingly where something is measured or actionable.** Bright and airy, not clinical.

### Light — the default (`:root`)
| Token | Value | Role |
|---|---|---|
| `--bg` | `#FFFFFF` | page |
| `--surface` | `#F5F5F7` | alternating bands, panels (Apple's signature grey) |
| `--surface-2` | `#EBEBEF` | insets, bar tracks, hover |
| `--text` | `#1D1D1F` | primary ink — near-black, never pure black |
| `--muted` | `#6E6E73` | secondary ink |
| `--border` | `rgba(29,29,31,.10)` | soft hairlines, used sparingly |
| `--border-strong` | `rgba(29,29,31,.20)` | focus, emphasis |
| `--accent` | `#2FA85C` | natural grass green — live, measured, primary action |
| `--accent-2` | `#34C759` | brighter green for peaks and highlights |
| `--accent-soft` | `rgba(47,168,92,.10)` | accent wash |
| `--accent-text` | `#FFFFFF` | ink on green |
| `--info` | `#0071E3` | blue — secondary data, reference values, links |
| `--ball` | `#FF6B4A` | the cricket ball, analysed window, seams |
| `--gold` | `#E8A33D` | caution, "unsure" |
| `--glow` | `rgba(47,168,92,.14)` | soft ambient glow behind media |

### Dark (`:root[data-theme="dark"]`)
| Token | Value |
|---|---|
| `--bg` | `#000000` · `--surface` `#111113` · `--surface-2` `#1C1C1F` |
| `--text` | `#F5F5F7` · `--muted` `#9A9AA0` |
| `--border` | `rgba(245,245,247,.12)` · `--border-strong` `rgba(245,245,247,.24)` |
| `--accent` | `#34C759` · `--accent-2` `#5BE07E` · `--accent-text` `#06220F` |
| `--info` | `#3B9EFF` · `--ball` `#FF7A5C` · `--gold` `#F0B84D` |
| `--glow` | `rgba(52,199,89,.16)` |

### How colour behaves
Most of the page is white, soft grey and near-black. Green appears only on live/measured/primary
things; blue only on reference or comparison data; coral only for the ball and the analysed window.
Always paired with a label — never colour alone.

### Over-footage (both themes)
`--on-media: #FFFFFF` · `--on-media-muted: rgba(255,255,255,.72)` ·
`--on-media-glass: rgba(0,0,0,.42)` · `--on-media-brd: rgba(255,255,255,.22)`

---

## 3. Typography

| Token | Family | Use |
|---|---|---|
| `--font` | **Inter** (400/500/600/700), fallback `-apple-system, "SF Pro Display"` | everything |
| `--font-mono` | **JetBrains Mono** (400/500) | numbers, telemetry, labels |

One sans family, many weights — Apple's discipline. **No serif, no display face.**

### Scale — large, tight, confident
- Hero: `clamp(48px, 8vw, 104px)` / `line-height: 1.04` / `letter-spacing: -.035em` / weight 700
- Section headline: `clamp(34px, 5vw, 64px)` / `1.06` / `-.03em` / weight 650
- Sub / lede: `clamp(17px, 1.6vw, 21px)` / `1.5` / `--muted` / max **60ch**
- Body: 16px / 1.6
- Small label: 12–13px / sentence case / `--muted`. **Mono is for numbers only** — scores,
  percentages, speeds, durations. Never for prose, headings, nav or section labels; that is what made
  the previous pass read as robotic.
- Big numbers (score, %, speed): mono, `font-variant-numeric: tabular-nums`, weight 500

Hero type is **centred**; section content is left-aligned. Headlines never exceed ~14 words.

---

## 4. Structure — one idea per screen

The old layout failed because sections were squeezed and undifferentiated. The fix is Apple's
rhythm: give each idea a whole screen and let it breathe.

- **Container** `.wrap` — `max-width: 1120px`, side padding 24px (20px under 720px).
- **Section rhythm** — each stage is its own full-viewport-ish band: `min-height: 88vh`,
  `padding: 140px 0` desktop / `88px 0` mobile. Centre the content block vertically.
- **No hairline between sections.** Separation comes from space and a background step
  (`--bg` → `--surface`), alternating band to band. Hairlines are used only *inside* components.
- **Spacing scale** — 4 · 8 · 12 · 16 · 24 · 32 · 48 · 64 · 96 · 140. Nothing off-scale.
- **Radius** — `--radius: 20px` panels, `--radius-lg: 32px` media frames, `999px` pills/buttons.
  Generous and soft, Apple-scale.
- **Elevation** — no hard borders on big surfaces. Depth comes from the background step, a soft radial
  `--glow` behind media, and gentle shadows: `0 4px 16px rgba(0,0,0,.05), 0 24px 56px rgba(0,0,0,.07)`.
  In light mode shadows are soft and wide, never dark or tight.
- **Media is the hero.** Footage and the dial are the "product shot": full-bleed or near, centred,
  with the ambient glow behind and nothing competing beside them.

### Home skeleton
```
┌ floating pill nav (glass, centred, 56px from top) ─────────────────┐
│                                                                    │
│            [ centred eyebrow · emerald mono ]                      │
│            Every delivery tells you something.      ← 104px, tight │
│            [ centred lede, 60ch, --muted ]                         │
│            ( Analyse a clip )  ( See what it does )  ← pill buttons│
│                                                                    │
│         ╭──────────── footage, rounded 28px ──────────╮            │
│         │  glow behind · telemetry strip along base   │            │
│         ╰─────────────────────────────────────────────╯            │
├──── band 2 · surface step ─────────────────────────────────────────┤
│   01 · THE SHOT        Which shot was that?                        │
│   8 strokes as soft glass tiles, emerald on the active one         │
├──── band 3 · bg ───────────────────────────────────────────────────┤
│   02 · THE SCORE       How good was it?                            │
│   huge dial left · net mini-game right, both in glass panels       │
├──── band 4 · surface step ─────────────────────────────────────────┤
│   03 · THE COACH       What do you work on?                        │
│   coach note as a floating glass card, emerald/gold tag pills      │
├──── footer ────────────────────────────────────────────────────────┤
```

### Analyse — results
Media column dominant and large; the read-out beside it as stacked glass panels in the order
**Shot → Technique → Movement → Coach**. Shot name set at hero scale with the confidence as a mono
pill. Part bars square-ended, emerald fill on `--surface-2`, mono values right-aligned.

---

## 5. Components

- **Nav** — a floating glass pill, centred, not a full-width bar: `--surface` at 60% with
  `backdrop-filter: blur(20px)`, 999px radius, 1px `--border`. Active link is a filled pill.
- **Button** — 999px pill, 52px tall, 24px side padding. Primary: emerald fill, `--accent-text` ink,
  soft emerald glow on hover. Secondary: glass with 1px `--border`. Keep the circular arrow badge.
- **Glass panel** — `--surface` at ~70%, `blur(20px)`, 1px `--border`, `--radius`, no hard shadow.
- **Stat / telemetry readout** — mono uppercase key above a large tabular-numeral value. Used for
  stride, hand speed, FPS, confidence. Keys in `--muted`, values in `--text`, unit in `--muted`.
- **Radial gauge** — thin 10px emerald arc on `--surface-2`, soft glow at the leading end, two
  dashed `--ball` seam arcs round the rim (the cricket-ball motif — keep it). Value centred, mono.
- **Bars** — 8px, square ends, emerald fill; the leading result full emerald, the rest at 35%.
  Reference/comparison values use `--info`.
- **Pixel sprites** — unchanged, pixelated.

---

## 6. Motion — this is half of the Apple feel

Slow, smooth, confident. Never snappy, never bouncy.

- **Easing**: `cubic-bezier(.28,.11,.32,1)` for everything. No springs, no `ease-in-out`.
- **Durations**: micro-interactions 240ms; element reveals 700–900ms; band transitions 1s.
- **Scroll reveal**: each band's content fades in and rises 32px, with children staggered 60–80ms
  apart, once, as it enters view.
- **Media parallax**: the hero footage scales `1.08 → 1` and lifts slightly across its first screen
  of scroll.
- **Numbers count up** from zero when their panel first enters view (dial, confidence, speeds).
- **Hover**: background step + a soft emerald glow bloom. No lift, no scale on cards.
- All behind `@media (prefers-reduced-motion: no-preference)`.

---

## 7. Fidelity rule

Use ONLY the fonts, colours, spacing and components above. Do **not** introduce cream, forest green,
serif or display faces, hairline-grid "blueprint" styling, purple/pink gradients, neon, uppercase
mono labels on prose, or any colour outside the token table. Chroma stays rare: if a surface does not represent a measurement, a
live state or an action, it is grey.
