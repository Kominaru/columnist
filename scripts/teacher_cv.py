"""Baselines and per-fold teachers for the v2 deep experiments, on song-grouped folds over every chart.

    python scripts/teacher_cv.py --data data --out-dir results/v2

Tasks: Komi's 11 categories plus the 20 osu!(lazer) user tags (a second benchmark).
Folds: dataset.chart_folds over all charts with features, so the fold-k teacher and student never
see a chart of a fold-k test song, labelled or not.

Writes
  data/v2_folds.csv          beatmap_id, fold, group (every chart)
  data/v2_labels.csv.gz      {1, 0, NaN} per chart and task, plus the tournament slot
  data/v2_teacher.csv.gz     fold-k teacher probabilities for every chart, columns "<k>|<task>"
                             (--teacher: the mean of the listed variants; default gbm+win)
  data/v2_labels_relaxed.csv.gz, v2_weights_relaxed.csv.gz   with --relaxed
  <out-dir>/oof.csv          out-of-fold predictions of rule / gbm / gbm+win per task
  <out-dir>/ap_categories.csv, ap_tags.csv
"""
import argparse
import os
import sys
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "1")

import numpy as np
import pandas as pd
from joblib import Parallel, delayed

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from osu_patterns import categories, dataset, evaluate, mania_tags  # noqa: E402
from osu_patterns.model import _make_clf, make_labels  # noqa: E402

TAGS = [f"tag:{t}" for t in mania_tags.TAGS]


def features(d: dataset.CategoryData, allf: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Boosted-head inputs for every chart: base (stats + rule scores + slot probabilities) and base + windows."""
    stats = d.stats.loc[allf.index]
    rows = {}
    for b, s in zip(stats.index, stats.to_dict("records")):
        rows[b] = {**mania_tags.feature_vector(s, d.probs.loc[b].to_dict()),
                   **{f"h:{k}": v for k, v in categories.heuristic_scores(s).items()}}
    base = pd.DataFrame.from_dict(rows, orient="index")
    win = allf[[c for c in allf.columns if c.startswith("w")]]
    return base, base.join(win)


def fit_predict(X, y, tr, pred_idx, w=None, seed=0):
    clf = _make_clf(seed)
    clf.fit(X[tr], y[tr].astype(int), sample_weight=None if w is None else w[tr])
    return clf.predict_proba(X[pred_idx])[:, 1]


def relaxed_labels(votes: pd.Series, Ystrict: pd.DataFrame, tasks: list[str]) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Training labels that also use partially voted charts, at half weight.

    strict (weight 1): >= 2 votes positive; 0 votes on a chart with >= 8 votes negative
    relaxed (weight 0.5): 1 vote positive; 0 votes on a chart with 3-7 votes negative
    Pool sub-slot labels stay as they are. Evaluation always uses the strict labels."""
    Yr, W = Ystrict[tasks].copy(), Ystrict[tasks].notna().astype(float)
    for bid, tv in votes.items():
        if bid not in Yr.index:
            continue
        total = sum(tv.values())
        for t in tasks:
            if not np.isnan(Yr.at[bid, t]):
                continue
            names = (t[4:],) if t.startswith("tag:") else categories.BY_NAME[t].user_tags
            v = max((tv.get(n, 0) for n in names), default=0)
            if v == 1:
                Yr.at[bid, t], W.at[bid, t] = 1.0, 0.5
            elif v == 0 and total >= 3:
                Yr.at[bid, t], W.at[bid, t] = 0.0, 0.5
    return Yr, W


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data")
    ap.add_argument("--slot-model", default="models/slots.pkl")
    ap.add_argument("--out-dir", default="results/v2")
    ap.add_argument("--jobs", type=int, default=4)
    ap.add_argument("--min-pos", type=int, default=12)
    ap.add_argument("--patterns", default="data/pattern_features.csv.gz",
                    help="also score gbm+win+pat, with the bag of row transitions (if the file exists)")
    ap.add_argument("--relaxed", action="store_true", help="also score models trained on relaxed labels")
    ap.add_argument("--extra", default="", help="extra feature table (e.g. data/msd_features.csv.gz): adds the "
                    "gbm+win+pat+x variants, with the bag of row transitions and these columns")
    ap.add_argument("--max-tasks", type=int, default=0, help="smoke test on the first N tasks")
    ap.add_argument("--methods", default="", help="comma-separated boosted variants to score (default: all available)")
    ap.add_argument("--teacher", default="gbm+win", help="comma-separated variants averaged into the teacher")
    ap.add_argument("--teacher-out", default="data/v2_teacher.csv.gz")
    ap.add_argument("--staff", default="", help="tournament-staff labels (build_staff_labels.py): training labels "
                    "for every variant where the strict label is unknown (and in the relaxed files at weight 1); "
                    "evaluation keeps the strict labels")
    ap.add_argument("--tag", default="v2", help="name of the relaxed label files written with --relaxed "
                    "(data/<tag>_labels_relaxed.csv.gz); anything but v2 also leaves v2_folds / v2_labels alone")
    a = ap.parse_args()
    d = dataset.category_dataset(a.data, a.slot_model)
    allf = pd.read_csv(Path(a.data) / "all_features.csv.gz", index_col=0)
    allf = allf[~allf.index.duplicated()]

    tags = make_labels(pd.DataFrame({"tags": d.votes}), mania_tags.TAGS, 2, 8)
    tags.columns = TAGS
    Y = pd.concat([d.Y, tags], axis=1).reindex(allf.index)
    tasks = [t for t in categories.NAMES + TAGS if (Y[t] == 1).sum() >= a.min_pos and (Y[t] == 0).sum() >= a.min_pos]
    if a.max_tasks:
        tasks = tasks[:a.max_tasks]
    fold = dataset.chart_folds(d.all_groups, allf.index)
    # training labels: the strict labels, plus the staff labels where the strict label is unknown.
    # Out-of-fold scores and the AP tables always use the strict labels.
    Ytr = Y.copy()
    if a.staff:
        S = pd.read_csv(a.staff, index_col=0).reindex(Y.index)
        for c in S.columns.intersection(tasks):
            Ytr[c] = Y[c].fillna(S[c])
        print("staff labels added per task:", (Ytr[tasks].notna().sum() - Y[tasks].notna().sum()).to_dict(), flush=True)
    print(f"{len(allf)} charts, {fold.groupby(fold).size().to_dict()} per fold; "
          f"{int(Y[categories.NAMES].notna().any(axis=1).sum())} with a category label, "
          f"{int(Y[TAGS].notna().any(axis=1).sum())} with a tag label", flush=True)

    base, withwin = features(d, allf)
    Xb, Xw = base.to_numpy(np.float32), withwin.to_numpy(np.float32)
    Xp = None
    Xx = None
    if Path(a.patterns).exists():
        pat = pd.read_csv(a.patterns, index_col=0).reindex(allf.index)
        Xp = withwin.join(pat).to_numpy(np.float32)
        if a.extra:
            ext = pd.read_csv(a.extra, index_col=0)
            Xx = withwin.join(pat).join(ext[~ext.index.duplicated()].reindex(allf.index)).to_numpy(np.float32)
    f = fold.to_numpy()
    if a.relaxed:
        Yr, Wr = relaxed_labels(d.votes, Ytr, tasks)
        print("relaxed labels added per task:", (Yr.notna().sum() - Y[tasks].notna().sum()).to_dict(), flush=True)
    variants = {"gbm": (Xb, False), "gbm+win": (Xw, False)}
    if Xp is not None:
        variants["gbm+win+pat"] = (Xp, False)
    if Xx is not None:
        variants["gbm+win+pat+x"] = (Xx, False)
    if a.relaxed:
        variants["gbm+win+relax"] = (Xw, True)
        if Xp is not None:
            variants["gbm+win+pat+relax"] = (Xp, True)
        if Xx is not None:
            variants["gbm+win+pat+x+relax"] = (Xx, True)
    run = a.methods.split(",") if a.methods else list(variants)
    teach = a.teacher.split(",")
    jobs = []
    for t in tasks:
        y = Ytr[t].to_numpy()
        yr, wr = (Yr[t].to_numpy(), Wr[t].to_numpy()) if a.relaxed else (y, None)
        for k in range(5):
            te = np.where(~np.isnan(Y[t].to_numpy()) & (f == k))[0]
            for m in dict.fromkeys(run + teach):
                X, relax = variants[m]
                yt = yr if relax else y
                idx = np.arange(len(y)) if m in teach else te  # a teacher predicts every chart
                jobs.append((t, k, m, X, yt, ~np.isnan(yt) & (f != k), idx, wr if relax else None))
    print(f"{len(jobs)} boosted fits", flush=True)
    res = Parallel(n_jobs=a.jobs, verbose=5)(delayed(fit_predict)(j[3], j[4], j[5], j[6], j[7]) for j in jobs)

    ids = allf.index
    oof = {m: pd.DataFrame(np.nan, index=ids, columns=tasks) for m in ["rule", *dict.fromkeys(run + teach)]}
    teacher = {}
    for (t, k, m, _, _, _, idx, _), p in zip(jobs, res):
        te = np.where(~np.isnan(Y[t].to_numpy()) & (f == k))[0]
        if m in teach:
            key = f"{k}|{t}"
            teacher[key] = teacher.get(key, 0) + p / len(teach)
            oof[m].iloc[te, tasks.index(t)] = p[te]
        else:
            oof[m].iloc[idx, tasks.index(t)] = p
    for t in tasks:
        col = f"h:{t}" if t in categories.NAMES else f"h:{t[4:]}"
        oof["rule"][t] = base[col].to_numpy()

    out = Path(a.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    pd.concat(oof, axis=1).to_csv(out / "oof.csv", float_format="%.5g")
    for name, cols in (("categories", [t for t in tasks if t in categories.NAMES]), ("tags", [t for t in tasks if t in TAGS])):
        if not cols:
            continue
        tab = evaluate.ap_table(Y[cols], {m: P[cols] for m, P in oof.items()})
        tab.to_csv(out / f"ap_{name}.csv")
        print(tab.to_string(), flush=True)

    dd = Path(a.data)
    if a.tag != "v2":  # strict labels under the new definitions (benchmark for this tag), folds unchanged
        lab = Y[tasks].copy()
        lab["slot"] = d.slot.reindex(ids)
        lab.rename_axis("beatmap_id").to_csv(dd / f"{a.tag}_labels.csv.gz")
    if a.relaxed and a.tag != "v2":
        Yr.rename_axis("beatmap_id").to_csv(dd / f"{a.tag}_labels_relaxed.csv.gz")
        Wr.rename_axis("beatmap_id").to_csv(dd / f"{a.tag}_weights_relaxed.csv.gz")
        pd.DataFrame(teacher, index=ids).rename_axis("beatmap_id").to_csv(a.teacher_out, float_format="%.4g")
        return
    pd.DataFrame({"fold": fold, "group": d.all_groups.reindex(ids)}).rename_axis("beatmap_id").to_csv(dd / "v2_folds.csv")
    lab = Y[tasks].copy()
    lab["slot"] = d.slot.reindex(ids)
    lab.rename_axis("beatmap_id").to_csv(dd / "v2_labels.csv.gz")
    pd.DataFrame(teacher, index=ids).rename_axis("beatmap_id").to_csv(a.teacher_out, float_format="%.4g")
    if a.relaxed:  # training labels for the deep model: relaxed label and its weight per chart and task
        Yr.rename_axis("beatmap_id").to_csv(dd / "v2_labels_relaxed.csv.gz")
        Wr.rename_axis("beatmap_id").to_csv(dd / "v2_weights_relaxed.csv.gz")


if __name__ == "__main__":
    main()
