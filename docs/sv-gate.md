# SV as a layer, and the SV gate (Komi, 2026-11-02)

Two rules for SV Reading and SV Memo:

1. **SV goes with any other category.** An SVLN burst is SV *and* LN (Halogen's Contain), so SV
   is a layer on top of the Rice / LN patterns, not a slot that competes with them.
2. **No SV without effective scroll changes.** Reading, Memo and the `sliders/*` tags never fire
   on a chart whose scroll speed never visibly changes. Slowjams and fastjams are SV and pass.

## What changed

**Display (`categories.pick`, Trello comment, Pattern Lens).** Before, an SV category on a
Rice / LN / Hybrid map needed 0.8 to show, and Rice / LN categories on an SV-slot map needed 0.8 too.
Now SV categories show at their own cut whatever the slot. An SV-slot map takes its best
Rice / LN / Hybrid slot as its pattern slot, and that slot's categories show at their own cut. The slot
reads "LN + SV", "Hybrid + SV", and so on (`categories.slot_label`). The labels were already
compatible: sibling negatives (wiki sub-slots, sheet tabs) only apply inside one family, so an SV label
never made a chart an LN negative.

**The gate (`reading.scroll_effect`).** "Effective" is what reaches the screen: BPM x SV relative to
the main BPM, measured on the notes. So an SV that normalises a BPM change is no change, and neither is a
chart played at one constant SV. A chart passes if any one of these holds:

* 4+ rows are drawn moved: the on-screen gap to the previous row, or the row's time on screen, is off the
  chart's median by more than 25%. This catches slowjams (bunched up) and fastjams (stretched out).
* 1 s+ of off-speed scroll (more than 25% from the median) while notes are on screen. This catches stutter
  SV that lands every row at its usual gap but moves it while it falls.
* 1+ teleport or stop while notes are on screen.

The gate only rules out charts with *no* visible change. How much SV makes a chart Reading or Memo
is still the heads' call.

Applied in `mania_tags.describe` (Trello), `live.LiveModel.analyze` (chart level), the Lens window
series (`reading.sv_window_gate`: SV only in windows that contain a moved row) and the per-note causal
model (`reading.sv_note_gate`: a note is open from a moved row on, for 2 s, using only the past). When
the gate fails, the SV slot probability is set to 0 and the rest are renormalised.

## Numbers (`data/sv_gate.csv.gz`, `scripts/build_sv_gate.py`)

36,640 cached charts: 10.1% pass (4K 9.4%). By tournament slot: SV 98.9% (the 3 that fail are
"[NSV]"-style charts with no SV in the file), LN 3.0%, HB 2.4%, RC 1.6%.

Labelled positives the gate removes. All of them are user-tag votes on charts with at most a
fraction of a second of slow-down:

| label | positives | on gated charts |
|---|---|---|
| Reading (v2 labels) | 485 | 27 |
| Memo | 43 | 2 |
| tag:sliders/complex sv | 247 | 3 |
| tag:sliders/low sv | 183 | 17 |
| tag:sliders/high sv | 141 | 8 |
| staff Reading / Memo (sheets, wiki) | 31 / 33 | 0 / 0 |

Final v2 ensemble (mean of the four components, same OOF), AP with the gate on the predictions:

| task | old labels, no gate | gated labels, no gate | gated labels, gate |
|---|---|---|---|
| Reading | 0.914 | 0.931 | 0.933 |
| Memo | 0.531 | 0.555 | 0.563 |
| sliders/low sv | 0.663 | 0.678 | 0.689 |
| sliders/high sv | 0.482 | 0.496 | 0.500 |

The models already rarely fired on charts with no scroll change: 18 charts at Reading >= 0.5 and 4 at
Memo >= 0.5, out of about 33k. So the gate is mostly a guarantee, not a score change. To score with it:
`python scripts/ensemble_v2.py ... --sv-gate data/sv_gate.csv.gz`. That zeroes the SV predictions and
turns SV labels into negatives on gated charts.

For the next label round: set Reading, Memo and `tag:sliders/*` to 0 where `sve_effective == 0`.
