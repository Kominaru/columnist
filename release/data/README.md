# osu!mania pattern labels (10,205 charts)

Labels for osu!mania charts: the tournament slot (Rice / LN / Hybrid / SV), eleven pattern
categories used by tournament mappoolers, and twenty osu!(lazer) user tags. Built for the
om-bn-pipeline pattern classifier. No beatmap files are included: download a chart from
`https://osu.ppy.sh/osu/<beatmap_id>` (or any mirror) to work with the notes.

## Files

* `charts.csv`: one row per chart. `fold` (0-4) is a song-grouped split: every chart of a song is
  in the same fold, so use it for cross-validation. `group` is the song key behind it.
* `labels.csv`: one row per (chart, task, source). `label` is 1 (has it) or 0 (does not).
  `detail` says why: vote counts, the tournament and slot, or the sheet text.
* `labels_wide.csv`: one column per task. `<task>` = the benchmark label (user tags + wiki pools,
  the labels the paper evaluates on). `staff:<category>` = the tournament-sheet label.
* `sources.csv`: the Google Sheets and wiki pages the staff labels come from.

## Tasks

* `slot`: main slot of the chart in osu! wiki tournament pools (majority over pools): RC (rice),
  LN, HB (hybrid), SV.
* Categories (Komi, 2026-10-30): Chordjack, Streams, Jumpstream/Handstream, Technical, Stamina
  (rice); Release, Coordination, Density/Inverse (LN); Reading, Memo (SV); TB (tiebreaker-style).
* `tag:*`: osu!(lazer) user tags. Positive at >= 2 votes; negative at 0 votes on a chart with
  >= 8 votes in total; otherwise unknown.

## Sources and how labels were made

* `user_tags`: lazer user-tag votes from the public beatmapset pages (2026-10).
* `wiki_pool`: sub-slots of osu! wiki mappools (JHC chordjack slots, o!mLN Coordination / Density,
  SVSE Reading / Memorisation, every pool's Tiebreaker; a map in one sub-slot is a negative for the
  sibling sub-slots of that tournament).
* `wiki_slot`: the main slot of wiki pools.
* `staff_sheet`: annotations in tournament Google Sheets linked from the osu! wiki and the
  Tournaments forum ("Rice - Chordjacks", "LN Release", "SV Memory"), matched by keyword; a map
  annotated with one sub-type is a negative for the sibling sub-types annotated in the same tab.
  Quality check: a model that never saw them ranks them at AP 0.66-1.0 against base rates of
  0.13-0.67. TB annotations are not used (they usually mark a bracket tiebreaker).

Labels are noisy and incomplete by design: most cells are unknown. Tournament maps are mostly
unranked customs; ranked maps get most of the user-tag labels.

## Licence and credit

Chart metadata and user tags come from osu! (ppy). Tournament annotations belong to the
tournaments' staff; `sources.csv` lists them. Share with credit; no warranty.
