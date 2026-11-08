# Columnist

A foundation model for osu!mania 4K charts: pretrained on unlabelled charts, adapted to pattern classification, difficulty estimation, per-note pattern readout and chart similarity. No snap or BPM-grid information is used.

- Website: https://columnist.quenlla.xyz (player with pattern readout, similar charts, difficulty)
- Models: https://huggingface.co/Quenlla/columnist · Data: https://huggingface.co/datasets/Quenlla/columnist-data · Code: https://github.com/Kominaru/columnist
- Licence: Apache-2.0 (code); CC BY 4.0 (weights, labels, model card). Charts remain the property of their mappers and are not redistributed. Use of osu! data complies with fair research use.
- Cards: `release/MODEL_CARD.md`, `release/DATA_CARD.md`

# osu-patterns: slot and pattern tags for osu!mania BN requests

Given an osu!mania map (a `.osu` file or a beatmapset link), predict

1. its **tournament slot**: RC (rice), LN, HB (hybrid) or SV, learned from official mappools, and
2. its **pattern tags** inside that slot (jumpstream, handstream, chordjack, speedjack, stamina,
   tech, LN release, LN coordination, inverse, SV types…), learned from osu!(lazer) user tags,

with timestamps for the jack / stream / LN / SV sections, formatted as a Trello comment.

## Pipeline

```
.osu ──mania.parse_mania──▶ notes (start, end, column), timing points
     ──mania.stats────────▶ ~80 statistics: density & stamina, chord mix, jacks / chordjacks / jack speed,
                            streams / jumpstream / handstream, snaps & rhythm variety, LN share / length /
                            release / inverse / held columns, effective scroll speed (BPM × SV) profile
     ──slots.SlotModel────▶ RC / LN / HB / SV probabilities   (gradient boosting, trained on mappools)
     ──mania_tags─────────▶ pattern tags                      (one boosted head per user tag, stacked on
                                                                rule scores + slot probabilities)
     ──mania_cli──────────▶ text / JSON / Trello markdown
```

## Data (no osu! API key needed)

| what | where from | script |
|---|---|---|
| slot labels: 8,454 mania maps, 115 tournaments | osu! wiki mappool pages (ppy/osu-wiki) | `scrape_wiki_mappools.py` → `data/mappools.csv` |
| `.osu` files | `https://osu.ppy.sh/osu/<beatmap_id>` | `fetch_osu.py` |
| ranked mania sets | logged-out `/beatmapsets/search` (paged by cursor) | `list_ranked_mania.py` |
| user-tag votes | JSON embedded in the public beatmapset page | `fetch_user_tags.py` |

Slot names are normalised in `slots.normalise_slot` (Long Note(s) → LN, Hybrid → HB, all of o!mLN → LN,
SVSE → SV, JHC → RC, JOKE/1K tournaments skipped; FreeMod / NoMod / Tiebreaker unlabelled; sub-slot names kept).
Tournament maps are mostly graveyard customs with no user tags, so tag labels come from the 3,000 most-played
ranked mania sets instead. A tag is positive at ≥2 votes and negative only on diffs with ≥8 votes in total.

## Results (2026-10-30)

**Slot**: 4,029 labelled maps (3,558 songs), 5-fold CV grouped by song so rate edits never leak:

| | accuracy | macro-F1 | RC F1 | LN F1 | HB F1 | SV F1 |
|---|---|---|---|---|---|---|
| hand rules | 0.885 | 0.844 | 0.995 | 0.843 | 0.566 | 0.970 |
| model | **0.981** | **0.979** | 0.994 | 0.977 | 0.952 | 0.991 |

The rules lose on hybrid vs LN; LN share does most of the work (HB median 38%, LN median 83%).
The remaining errors are mostly boundary maps and o!mLN pools that put hybrids in an LN-only tournament.

**Pattern tags**: 5,300 ranked diffs from 2,289 sets, CV grouped by beatmapset. Average precision of the
learned head vs the hand rule on the same maps is in `models/mania_tags.report.json`. The model beats the rule
on 19 of 20 tags (e.g. jumpstream 0.71 vs 0.53, handstream 0.51 vs 0.20, stream speed 0.53 vs 0.27,
tech 0.58 vs 0.42, low SV 0.54 vs 0.32); LN inverse is the exception (0.34 vs 0.39).

## Komi's categories (2026-10-30)

Shown under the predicted slot; several can apply at once. `osu_patterns/categories.py` defines where each
one's labels come from. Average precision with CV grouped by song, learned head vs rule:

| slot | category | labels from | positives | model AP | rule AP |
|---|---|---|---|---|---|
| Rice | Chordjack | `style/chordjack` + JHC chordjack slots | 747 | **0.83** | 0.77 |
| Rice | Streams | `skillset/streams` | 689 | **0.61** | 0.52 |
| Rice | Jumpstream/Handstream | jumpstream / handstream / chordstream / quadstream tags + JHC Quadstream | 970 | **0.78** | 0.64 |
| Rice | Technical | `skillset/tech`, `tech/complex snap` | 317 | **0.57** | 0.45 |
| Rice | Mixed Rice | derived: skills from 2+ of jacks / streams / tech | (no head) | | |
| Rice | Stamina | `streams/stamina` | 750 | **0.74** | 0.62 |
| LN | Release | `style/LN release` | 99 | **0.41** | 0.29 |
| LN | Coordination | `style/LN coordination` + o!mLN Coordination | 347 | **0.62** | 0.48 |
| LN | Density/Inverse | `style/LN density`, `gimmick/LN inverse` + o!mLN Density | 207 | **0.73** | 0.59 |
| SV | Reading | complex / high / low SV tags + SVSE Reading | 318 | **0.87** | 0.69 |
| SV | Memo | `gimmick/memory` (SV maps only) + SVSE Memorisation | 34 | 0.32 | **0.44** (rule used) |
| — | TB-worthy | every pool's Tiebreaker slot vs the other labelled pool maps | 541 | **0.88** | 0.48 |

Hybrid has no sub-categories; a hybrid shows whichever Rice and LN categories fire. A category from another
slot only shows when it is very likely (≥ 0.8). Notes: mania's `skillset/reading` user tag means dense-pattern
reading, not SV, so SV Reading uses the SV tags instead. The `style/mixed rice` tag was too noisy to learn
(AP 0.09), hence the derived rule. Memo has few labels, so it is the weakest category.

## Usage

```bash
pip install -r requirements.txt
python -m osu_patterns.mania_cli https://osu.ppy.sh/beatmapsets/2552898 --format trello
python -m osu_patterns.mania_cli path/to/map.osu            # or a folder of diffs

# rebuild everything
python scripts/fetch_osu.py data/mappools.csv data/osu
python scripts/train_slots.py --pools data/mappools.csv --osu data/osu --out models/slots.pkl
python scripts/list_ranked_mania.py data/ranked_mania.jsonl
python scripts/fetch_user_tags.py data/ranked_mania.jsonl data/user_tags.jsonl 3000
python scripts/tagged_ids.py data/user_tags.jsonl data/tagged.csv
python scripts/fetch_osu.py data/tagged.csv data/osu
python scripts/train_mania_tags.py --tags data/user_tags.jsonl --osu data/osu --out models/mania_tags.pkl
python scripts/train_categories.py --out models/categories.pkl
```

Tests: `python -m pytest tests` (synthetic charts with known jacks, streams, jumpstream, LN and SV).

The osu!standard prototype (`parser.py`, `features.py`, `tagger.py`, `cli.py`) is kept from the first
iteration; `model.TagModel` is shared.

## Next

* More labels for the thin categories (Memo, LN Release): more SV/LN tournament sub-slots, or Komi's own verdicts.
* Hook `mania_cli` into the Trello request flow so each new card gets the comment.
