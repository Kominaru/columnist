---
license: cc-by-4.0
library_name: pytorch
tags: [osu, osu-mania, rhythm-game, foundation-model]
---
# Columnist

A foundation model for osu!mania 4K charts. Pretrained without labels on unlabelled charts, then adapted to specific tasks.

## Variants
| File | Use |
|---|---|
| columnist-base | Pretrained encoder (256d, 6 layers) |
| columnist-embed | Label-free encoder behind the similar-charts search |
| columnist-joint | Fine-tuned: 31 pattern tags and categories, overall difficulty (stars ≈ 3.886 + 0.140 × score), per-window and per-row difficulty |
| columnist-lens | Per-note pattern readout used by the player page |

## Training
- Pretraining data: 62,161 unlabelled 4K charts (59,361 unique); chart ids only are released.
- Fine-tuning labels: community sheet labels, osu! user tags and tournament mappool slots; 8,453 labelled charts.
- Inputs use millisecond timing and visible scroll only. No snap or BPM-grid information.

## Results (5-fold, mean AP, categories / tags)
Fine-tuned from the base Columnist (5-fold, song-grouped, mean AP over 11 categories / 20 tags):
- columnist-base fine-tuned for tags and categories: 0.703 / 0.581
- columnist-joint (tags, categories and difficulty): 0.698 / 0.578
- Difficulty: Spearman 0.94 against official star rating, mean absolute error 0.33 stars after linear calibration; held-out tournament round ordering 0.95 accuracy.
The earlier boosted-statistics pipeline scores higher on categories (about 0.75 / 0.60); Columnist trades some of that for one shared network that also serves similarity, difficulty and per-note readout.

## Limitations
4K only; at least 64 rows per chart; long-note patterns are the weakest area; labels come from user tags and mappool slots, not ground truth. Research tool, not an official osu! tool, and not meant for Beatmap Nominator decisions on its own.

## Licence and data
Weights, labels and this card: CC BY 4.0. Code: Apache-2.0. Charts remain the property of their mappers and are not redistributed. Use of osu! data complies with fair research use.
