# Deep models for Komi's categories (2026-10-31)

Can a deep sequence model beat the boosted heads on the 71 hand-crafted statistics? **Not with
the labels we have.** Self-supervised pretraining helps the deep model a lot, but it stays well
below boosting, and adding its predictions to the boosted heads changes nothing.

## What was tried

| method | what it is | code |
|---|---|---|
| rule | hand-written score per category | `categories.heuristic_scores` |
| gbm | boosted heads on chart statistics + rule scores + slot probabilities (the current model) | `evaluate.gbm_oof` |
| gbm+win | the same plus sliding-window statistics (16 s windows, 8 s hop) pooled with max / p75 / std: multi-instance learning without DL | `windows.py`, `scripts/build_window_features.py` |
| deep:scratch | events (note starts and LN ends) → dilated 1D CNN → 4 s window tokens → 2-layer transformer → gated attention-MIL per task; trained jointly on 11 categories, 20 user tags and the slot | `osu_patterns/deep/`, `scripts/train_deep.py` |
| deep:pretrained | the same encoder, first pretrained on 36,355 ranked and labelled charts with masked events (predict the columns and gap) and masked windows (predict window statistics from context) | `scripts/pretrain_deep.py` |
| fuse:* | gbm+win plus the deep model's out-of-fold task and slot probabilities | `scripts/compare_models.py` |

Every method is scored on the same 5 song-grouped folds (`dataset.song_folds`), with the same
masked labels. The deep runs used an RTX 2060 SUPER: about 10 s per fine-tuning epoch (30 epochs
per fold) and 40 s per pretraining epoch (15 epochs).

## Results (average precision, `results/compare.csv`)

| category | pos | rule | gbm | gbm+win | deep:scratch | deep:pretrained | fuse:pretrained |
|---|---|---|---|---|---|---|---|
| Chordjack | 747 | 0.767 | 0.832 | **0.836** | 0.752 | 0.799 | 0.834 |
| Streams | 689 | 0.515 | **0.620** | 0.612 | 0.521 | 0.554 | 0.610 |
| Jumpstream/Handstream | 970 | 0.639 | 0.782 | 0.783 | 0.673 | 0.767 | **0.786** |
| Technical | 317 | 0.453 | 0.575 | **0.598** | 0.300 | 0.505 | 0.595 |
| Stamina | 750 | 0.615 | 0.737 | 0.738 | 0.665 | 0.674 | **0.741** |
| Release | 99 | 0.286 | **0.420** | 0.382 | 0.352 | 0.298 | 0.364 |
| Coordination | 347 | 0.483 | 0.621 | **0.631** | 0.507 | 0.575 | 0.630 |
| Density/Inverse | 207 | 0.585 | **0.728** | 0.722 | 0.548 | 0.549 | 0.719 |
| Reading | 318 | 0.693 | 0.870 | **0.878** | 0.762 | 0.765 | 0.855 |
| Memo | 34 | **0.440** | 0.298 | 0.402 | 0.304 | 0.312 | 0.354 |
| TB | 541 | 0.481 | **0.892** | 0.891 | 0.822 | 0.829 | 0.884 |
| **mean** | | 0.542 | 0.670 | **0.679** | 0.564 | 0.602 | 0.670 |

(The CPU run of gbm vs gbm+win in this container, on folds built without the deep models' chart
set, gave 0.671 vs 0.675. Same direction, smaller gap.)

## Reading

* **Pretraining works, but it isn't enough.** It lifts the deep model's mean AP by 0.04, with
  Technical +0.21, Jumpstream/Handstream +0.09 and Coordination +0.07. These are the pattern-grammar
  skills that masked-event prediction should teach. With about 2k labelled charts per category, it
  still trails statistics designed by someone who knows what a jumpstream is.
* **Section-level information helps a little, through the statistics.** Windowed statistics help
  Technical, Coordination, Reading and Memo, and hurt Release (only 99 positives, so the noisiest
  category). Mean +0.009. This is the cheapest improvement found, and worth shipping.
* **Fusion adds nothing.** The deep probabilities don't carry information the boosted heads lack;
  fusion only adds noise to small heads.
* **The bottleneck is labels, not the model**, as the paper already concluded for model classes.
  Release (99 positives) and Memo (34) don't improve with any method.

## If we come back to this

Ranked by expected value:
1. Ship `gbm+win` as the category model (mean +0.009, best on 6 of 11 categories).
2. Get more labels for Release / Memo / Density (Komi's own verdicts on Trello requests, more
   o!mLN / SVSE sub-slots). Every method is label-bound there.
3. Deep: longer pretraining on a larger corpus (graveyard charts; the losses were flattening but
   still falling), a frozen encoder with a small head instead of full fine-tuning, and MinaCalc
   skillset ratings as an extra pretraining target for 4K. Expect it to close the gap, not to
   overtake boosting.
4. The deep model's attention weights over windows could still serve as timestamps in the Trello
   comment. That is untested.
