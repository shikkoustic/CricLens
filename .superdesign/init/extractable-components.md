# Extractable components

Plain HTML/CSS, so these are markup+class patterns rather than framework components. The two worth
extracting as reusable Superdesign `DraftComponent`s are the shell pieces duplicated across both
pages — everything else is simple enough to inline in a draft.

## Layout components

## TopBar
- Source: `app/static/home.html:30:48` (identical block in `app/static/analyze.html:25:43`)
- Category: layout
- Description: Sticky top bar — CricLens ball logo + wordmark, Home/Analyse nav, warming-up status pill, theme toggle
- Extractable props: activeItem (string, default: "home" — which nav link gets `.on` + `aria-current`)
- Hardcoded: inline logo SVG (red ball with two seam arcs), nav labels and hrefs, sun/moon icon SVGs, status pill markup, all CSS classes

## SiteFooter
- Source: `app/static/home.html:175:189`
- Category: layout
- Description: Two-column footer — About blurb and the two author names
- Extractable props: none
- Hardcoded: about copy, "Shikhar Srivastava", "Vanshika Gupta", all CSS classes

## Basic components (inline in drafts — do NOT extract)
- `.btn` + variants (`btn-primary`, `btn-lg`, `btn-sm`, `btn-ghost`, `btn-on-media`, `btn-glass`) and `.btn-arrow`
- `.card`, `.card-head`, `.card-title`, `.glass-panel`
- `.eyebrow`, `.tag` / `.tag.unsure`, `.badge`, `.chip`
- `.icon-btn`, `.status` + state classes
- `.ring` / `.dial` radial gauge with `.ring-seam` cricket-seam arcs
- `.px` pixel-art sprites referencing the `#px-sheet` symbols

## Note on the mini-game
The net mini-game (`#net` in `home.html:130:148` plus its logic in `home.js`) is a bespoke
interactive, not a reusable component. The user wants it kept. Treat it as a fixed block to be
re-housed by any redesign rather than re-styled away.
