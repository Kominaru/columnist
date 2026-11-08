# Labels mined from tournament sheets and mapset text (4K, 2026-11-01)

Question: can tournament Google Sheets (linked on the osu! wiki) and beatmap descriptions give
more or better labels for Komi's categories, for 4K?

## Sources

| source | how | 4K charts with a label | new vs v2 labels |
|---|---|---|---|
| tournament sheets | `scripts/scrape_pool_sheets.py`: 183 sheets linked from mania wiki pages, 153 public; "Type / Notes / Category / Map type" cells of mappool tabs | 218 | ~160 positives, ~250 negatives |
| wiki sub-slots | slot names in `data/mappools.csv` not already used ("Technical", "Tech") | 28 | 22 Technical positives |
| diff names | `[JS]`, `[Chordjack dump]`, ... | 77 | ~60 positives |
| mapper tags | set tags, single-chart sets only | 654 | ~530 positives |
| descriptions | set description, single-chart sets only | ~400 | ~290 positives |

Mapset text comes from `scripts/fetch_set_text.py` (all 11.1k sets with a 4K chart, keyless,
`data/set_text.jsonl.gz`). Keyword table and negative rules: `osu_patterns/pool_annotations.py`.
Sheets only give negatives inside a tab that annotates two or more sub-types of a family; text
sources only give positives.

What the sheets are not: most of the 153 public sheets are score/statistics sheets (MWC/MCNC
templates), and of the mappool tabs most only repeat the slot (RC1, LN2). About 25 tabs carry a
real sub-type column. No sheet had free-text descriptions of maps.

## Quality

Checked against the out-of-fold gbm+win teacher, which never saw these labels
(`scripts/audit_pool_labels.py`, `results/pool_labels/audit.csv`). For the sheet labels, the AP
of the teacher on the new labels against their base rate:

| | Chordjack | Streams | JS/HS | Technical | Coordination | Density/Inverse |
|---|---|---|---|---|---|---|
| teacher AP | 1.00 | 0.78 | 0.68 | 0.92 | 0.94 | 1.00 |
| base rate | 0.14 | 0.25 | 0.26 | 0.48 | 0.52 | 0.55 |

So sheet labels agree with what the model already calls these skills. Text labels (positives only)
rank at AUC 0.85-0.93 against all other 4K charts for most categories (mapper tags, descriptions);
the weak ones are description JS/HS (0.78), diff-name Technical (0.75), Coordination (0.70-0.79).
Loose readings were worse and are off by default: "Jacks" slots as Chordjack (AUC 0.73, and
JHC already treats jack slots as chordjack negatives), bare "release" (usually about the song),
"Memory" (a common song/diff name).

## Does it help?

gbm+win retrained on the v2 folds with the mined labels filling unknowns
(`scripts/train_with_pool_labels.py`, `results/pool_labels/`):

| mean AP | v2 labels | + staff (sheets, wiki) | + staff + text |
|---|---|---|---|
| v2 benchmark, all charts | 0.719 | 0.721 | 0.725 |
| v2 benchmark, 4K | 0.701 | 0.702 | 0.707 |
| held-out sheet labels, 4K | 0.899 | 0.924 | 0.925 |

The +0.006 on the 4K benchmark is mostly Memo (0.274 -> 0.345, on only 43 positives) and
Density/Inverse (+0.013); Release drops 0.633 -> 0.614. On the staff labels themselves the gain is
+0.026 (Technical, Streams, JS/HS). Small, as expected: the rice heads already have ~1,000
positives each, and the sources add little where labels are scarcest (Release +1 from sheets,
Memo +2).

## Update: sheets from forum posts

`scripts/find_forum_sheets.py` lists the osu! Tournaments forum (6,578 topics, 600 with a mania
title) and collects the sheet links in each topic's first page: 356 sheets, 309 not on the wiki.
With them, 414 sheets are public (was 153) and give 19.8k map rows (was 4.1k).

* 4K charts in the corpus with a sheet label: 218 -> **730**; about 1,260 charts with a sub-type in
  total. 378 of the labelled charts were outside the corpus; their features are in
  `data/sheet_extra_features.csv.gz` (324 of them 4K). They join the fold of their song, or a
  hashed fold for a new song.
* Quality stays well above base rate but is lower than the wiki-linked sheets alone (teacher AP
  0.66-0.90 against base rates 0.13-0.67; `results/pool_labels/audit.csv`). Release now gets 27
  new positives (was 1).
* Sheet "TB" cells often mark the bracket's tiebreaker pick rather than a TB-style map. Training on
  them cost 0.03 AP on the v2 TB labels, so TB keeps only the wiki labels.

gbm+win with everything (sheets from wiki + forum, wiki sub-slots, and the 378 extra charts):

| mean AP | v2 labels | + staff | + staff + text |
|---|---|---|---|
| v2 benchmark, all charts | 0.719 | **0.732** | 0.729 |
| v2 benchmark, 4K | 0.701 | **0.714** | 0.713 |
| held-out sheet labels, 4K (1,500 labels) | 0.781 | **0.816** | 0.810 |

Per category on the 4K benchmark: Memo 0.274 -> 0.411 (43 positives, so noisy), Streams +0.015,
Reading +0.012, Coordination +0.008; Release -0.018. Without Memo, the other ten categories
together move by less than +0.001 on the v2 benchmark. The gain shows on the staff labels
(+0.035), which is the population the v2 labels cover least. Once the staff labels are in, the
mapset text no longer adds anything.

## Takeaways

* The sheets are a better **benchmark** than a training source: ~220 4K tournament maps with
  staff-assigned sub-types, the population Komi actually cares about, and disjoint from the user
  tags. Worth reporting next to the user-tag benchmark.
* Mapset text is the bigger source (~1,000 4K charts) and is cheap to refresh; use it for Memo
  and Density/Inverse, not Release.
* Forum posts tripled the staff labels (done above). Remaining sheets are private (about 60
  return 401/403) or from forum topics whose titles don't name the mode.
