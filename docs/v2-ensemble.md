# Beating gbm+win by +0.05 AP: the v2 ensemble (2026-11-01)

Komi's goal (2026-10-31): a model at least **+0.05 mean AP** better than the best model
(gbm+win) on both benchmarks, Komi's 11 categories and the 20 lazer user tags, without
pretraining on test charts.

**Result: +0.049 on categories and +0.047 on user tags** (0.768 / 0.636 against
0.719 / 0.588). That is just below the target on both. Over 500 song resamples, the 95% intervals are
[+0.036, +0.060] and [+0.040, +0.055] (`paper/experiments/v2_tables.py --boot 500`), and 38% / 25% of
the resamples reach +0.05.

The paper (`paper/main.pdf`) describes the final system in full; this note keeps the development
history, including what did not work.

## Protocol (v2)

* 36.5k charts (labelled, ranked, loved, tournament pools), 5 folds grouped by song over all of
  them (`data/v2_folds.csv`). The fold-k models never see a fold-k song, labelled or not: SSL,
  distillation and the teachers all use per-fold corpora.
* Labels as before: user tags positive at >= 2 votes, negative at 0 votes with >= 8 votes in total;
  categories add the tournament sub-slots. Evaluation uses these strict labels only.
* Mean AP over the out-of-fold predictions of all five folds pooled.

## The final model

The plain mean of four components. Each component is the mean of its runs.

| component | what it is | alone (cat / tag) |
|---|---|---|
| boosted | the five gbm variants averaged; for the 5 SV heads, gbm+win+pat+relax with reading + MinaCalc features | 0.749 / 0.616 |
| deep plain | event CNN + transformer + attention-MIL, SSL + stats distillation + teacher KD, 30 stage-1 epochs (full5X, 3 seeds) | 0.751 / 0.625 |
| deep wide | the same, plus the boosted inputs as a second path (wide & deep; wide5X, 6 seeds) | 0.762 / 0.629 |
| MLP | multi-task MLP on the boosted inputs + reading + MinaCalc, distilled from the per-fold teacher, 5 seeds | 0.747 / 0.625 |
| **mean** | | **0.768 / 0.636** |

```
python scripts/ensemble_v2.py "boosted|oof:gbm+win+pat+x+relax" \
    deep:full5X_s0+full5X_s1+full5X_s2 \
    deep:wide5X_s0+wide5X_s1+wide5X_s2+wide5X_s3+wide5X_s4+wide5X_s5 mlp:v5 \
    --oof results/v2/oof.csv results/v5_x/oof.csv
python scripts/bootstrap_v2.py <same components> --oof results/v2/oof.csv results/v5_x/oof.csv
python paper/experiments/v2_tables.py --boot 500     # the paper's tables and numbers
```

Per-task tables: `results/v2/ensemble_categories.csv`, `results/v2/ensemble_tags.csv`.

## What moved the score

| step | categories | tags |
|---|---|---|
| gbm+win (baseline) | 0.719 | 0.588 |
| + relaxed labels (1-vote positives at weight 0.5) and row-transition features | 0.728 | 0.606 |
| deep student distilled from per-fold boosted teachers on all non-test charts | 0.694 → 0.733 | |
| MLP student on the boosted inputs (beats every boosted model) | 0.735 | 0.620 |
| ensemble of boosted + deep + MLP (plain mean) | 0.750 | 0.633 |
| reading features for the SV heads (below) | 0.765 | 0.634 |
| deep: 30 pretraining epochs instead of 16, more seeds | 0.768 | 0.636 |

**Reading features** (`osu_patterns/reading.py`). The SV statistics described the speed values.
The reading features describe what the player sees. From the scroll position x(t) (the integral of
the speed), they measure how long each note is on screen, the on-screen gap vs the time gap
(stacking, spreading), how often the speed changes while notes are visible, and teleports and
freezes. They lift Memo from 0.29 to 0.43 in the boosted model and from 0.42 to 0.52 in the ensemble.
They add noise to every non-SV head, so models that use them stand in for the SV heads only
(`A|B` in `ensemble_v2.py`). That split is a design choice made after seeing the results, on 5 of
31 heads.

**Longer pretraining.** The training curves showed stage 1 (SSL + distillation) still improving
when its schedule ended at 16 epochs, while fine-tuning stopped helping after 3 epochs. With
30 epochs the best single wide & deep run went from 0.745 / 0.613 to 0.749–0.751 / 0.616–0.617.

## What gave nothing (each within ±0.004 of the reference)

Cross-task stacking, vote-share and binomial labels, mapper tags and diff names, star rating,
iterated distillation, mapset smoothing, intensity features, MinaCalc ratings on their own,
Memo-specific gimmick features (blocks revealed at once, pop-ins), LN features (release grid,
hold-and-tap, inverse gaps), mapset context and era, hitsound/keysound features, per-event
"visual" channels for the deep encoder, a larger encoder (96 channels, 3 layers), 45 pretraining
epochs, per-task early stopping (MLP alone +0.003, ensemble 0), fitted blend weights (per-task or
per-benchmark weights chosen on the other folds lose to the plain mean), tag/category
cross-blending.

## Reading

* **Labels are still the limit.** Most gains came from using labels better (relaxed labels,
  distillation onto unlabelled charts) or from information the models didn't have (reading).
  Model changes inside one family move single models by 0.005–0.01 and the ensemble by ~0.001.
* **The small heads dominate the noise.** Memo (43 positives) and quadstream (36) swing by
  ±0.05 AP between runs, which is ±0.003 on the benchmark mean.
* **Diversity matters more than any single model.** The plain mean of four families beats the best
  single model by ~0.015 on both benchmarks. Adding more seeds of one family adds ~0.001.

## If we come back to this

1. More labels for Memo, quadstream, high/low SV and LN inverse: they are the noisiest heads.
2. Ship the MLP student (0.747 / 0.625 alone, CPU-cheap) if the ensemble is too heavy for the BN
   request bot. The deep models need a GPU for training but not for inference.
3. Reading features for the SV slot model and the Trello comment ("notes appear late", "stops while
   notes are on screen") would be easy to explain to mappers.
