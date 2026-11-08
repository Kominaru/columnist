# Columnist brand, v2: the painted temple

## The idea
Greek and Roman temples were never white. They were painted in loud pigments, and what we call "classical" is what time bleached.
Columnist keeps the paint. An osu!mania 4K stage already is a colonnade: four lanes, standing in a row, taking weight from above.
So the default skin becomes architecture: **each lane is a column, each note is a drum of that column, the long note is one unbroken shaft,
the judgement line is the stylobate (the stone step) they all stand on.** A chart is a colonnade being built from the top down, and Columnist is the one reading it.
Pun kept on purpose: a columnist writes in columns.

## The four pigments (the lane colours)
| Slot | Pigment | Hex | Real ancient pigment |
|---|---|---|---|
| Rice | cinnabar | #e8503a | red mercury sulphide |
| LN | malachite | #2fb584 | copper carbonate green |
| SV | amethyst | #b680dc | purple (Tyrian family) |
| Hybrid | ochre | #f2b53c | yellow ochre |
Ground: **lapis** #101c44 (raised #182a63, rule #31438a), the Egyptian-blue family. Stone and text: **lime plaster** #f3eddc, dim #b9b29b, ink on plaster #0d1530.
Rules: lapis ground is default. The four pigments appear together only in the mark, the frieze and in chart or tag visuals; in layouts use plaster plus at most one pigment (cinnabar for actions, malachite for links). Plaster-on-lapis contrast is about 14:1; never put plaster-dim on a pigment fill.
These replace the site's orange/teal/violet/gold (#ff8a3d, #40c9c0, #a98bff, #ffd166); migrate by swapping the four slot variables.

## Devices (what makes it recognisable)
1. **The colonnade mark**: four columns, drums per lane exactly as in a real chart (3, 1 long, 2, 2), capitals and bases merging into one entablature and one stylobate.
2. **The carved wordmark**: COLUMNIST in Marcellus (Roman inscription capitals), tracked wide like letters cut into an architrave. This is the only place capitals are used; everything else is sentence case.
3. **The frieze**: a Doric frieze strip of triglyphs (three grooves) and metopes (painted squares) cycling cinnabar, malachite, amethyst, ochre. It is the brand's divider, page top edge, banner/card header and video lower-third (file: frieze-strip.svg). Use it once per layout, at the top or bottom, never as a repeating background.
4. **Drums as data**: in charts and diagrams, notes are rounded blocks with three vertical flutes, stacked like drums; a long note is one tall block.
5. **Fluting** (three thin dark vertical lines at 22% opacity) is the only texture. No marble photos, no gradients, no drop shadows.

## Type
Wordmark and big titles: **Marcellus** (Google Fonts, OFL). Body, UI, captions: **Hanken Grotesk** (Google Fonts, OFL), 16 px or more. Code and numbers: system mono. Never tracked caps for labels; labels are Hanken sentence case a step smaller. Graphics use outlined text, so the SVGs need no font installed.

## Voice
Less corporate, a little theatrical, always concrete. Architecture words are the seasoning, one per piece, never a metaphor every line: "stand the chart up", "read it column by column", "what's holding this chart up?", "a bad chart has a cracked shaft". Specific numbers beat adjectives ("about 0.33 stars typical error"). No "revolutionary", no "AI-powered".
- Bio (under 160 chars): "Columnist reads osu!mania 4K charts column by column: difficulty, pattern tags, similar charts. Open weights and data."
- Tagline: **A foundation model for osu!mania 4K charts.**
- Release post shape: one line of what it does with one number, the link, the og card. Reply or thread uses one column pun at most.

## Files
mark.svg/png (tile), mark-transparent.svg, favicon.svg/png, avatar.svg/png 400x400 (X, GitHub, HF), logo-dark/light .svg/.png and logo-transparent-dark.svg,
banner-1500x500 (X header, avatar corner kept clear), og-card-1200x630 (link preview, GitHub social preview, HF thumbnail),
video-title-1920x1080 (title and end card), frieze-strip.svg, palette.json, tokens.css.
Clear space: one column width around the mark. Minimum: mark 16 px (it still reads as four colour bars), lockup 160 px wide. Do not recolour the pigments, reorder the lanes, round the wordmark, or put the mark on a photo without its lapis tile.
