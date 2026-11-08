"""Train the production heads for the live visualizer (and any other CPU-only service).

    python scripts/train_live_heads.py --data data --out models/live_heads.pkl

Same recipe as the "gbm+win+relax" column of teacher_cv.py (chart stats + rule scores + slot
probabilities + pooled 16 s window stats, relaxed labels at half weight), but each head is fit
once on every labelled chart instead of per fold. Thresholds are the best-F1 cut on the existing
out-of-fold predictions in results/v2/oof.csv, so they are honest estimates for unseen charts.
"""
import argparse
import os
import sys
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "1")

import joblib
import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from sklearn.metrics import average_precision_score, f1_score

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))
from osu_patterns import categories, dataset, mania_tags  # noqa: E402
from osu_patterns.model import _make_clf, make_labels  # noqa: E402
from teacher_cv import TAGS, features, relaxed_labels  # noqa: E402

METHOD = "gbm+win+relax"


def best_threshold(y, p):
    grid = np.unique(np.quantile(p, np.linspace(0.05, 0.95, 37)))
    scores = [f1_score(y, p >= g) for g in grid]
    return float(grid[int(np.argmax(scores))])


def fit(X, y, w, seed=0):
    clf = _make_clf(seed)
    clf.fit(X, y.astype(int), sample_weight=w)
    return clf


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data")
    ap.add_argument("--slot-model", default="models/slots.pkl")
    ap.add_argument("--oof", default="results/v2/oof.csv")
    ap.add_argument("--out", default="models/live_heads.pkl")
    ap.add_argument("--jobs", type=int, default=4)
    ap.add_argument("--min-pos", type=int, default=12)
    a = ap.parse_args()

    d = dataset.category_dataset(a.data, a.slot_model)
    allf = pd.read_csv(Path(a.data) / "all_features.csv.gz", index_col=0)
    allf = allf[~allf.index.duplicated()]
    tags = make_labels(pd.DataFrame({"tags": d.votes}), mania_tags.TAGS, 2, 8)
    tags.columns = TAGS
    Y = pd.concat([d.Y, tags], axis=1).reindex(allf.index)
    tasks = [t for t in categories.NAMES + TAGS if (Y[t] == 1).sum() >= a.min_pos and (Y[t] == 0).sum() >= a.min_pos]
    Yr, Wr = relaxed_labels(d.votes, Y, tasks)
    _, withwin = features(d, allf)
    X = withwin.to_numpy(np.float32)
    print(f"{len(tasks)} tasks, {X.shape[1]} features", flush=True)

    jobs = []
    for t in tasks:
        y, w = Yr[t].to_numpy(), Wr[t].to_numpy()
        m = ~np.isnan(y)
        jobs.append((X[m], y[m], w[m]))
    heads = Parallel(n_jobs=a.jobs, verbose=5)(delayed(fit)(*j) for j in jobs)

    oof = pd.read_csv(a.oof, header=[0, 1], index_col=0)[METHOD]
    keys = allf["keys"].reindex(oof.index)
    thresholds, report = {}, {}
    for t in tasks:
        y = Y[t].reindex(oof.index)
        m = y.notna() & oof[t].notna()
        yy, pp = y[m].to_numpy().astype(int), oof[t][m].to_numpy()
        thresholds[t] = best_threshold(yy, pp)
        k4 = (keys[m] == 4).to_numpy()
        report[t] = {"n": int(m.sum()), "pos": int(yy.sum()), "ap": round(float(average_precision_score(yy, pp)), 3),
                     "ap_4k": round(float(average_precision_score(yy[k4], pp[k4])), 3) if yy[k4].sum() else None,
                     "threshold": round(thresholds[t], 4)}
        print(t, report[t], flush=True)
    bundle = {"columns": list(withwin.columns), "tasks": tasks, "heads": dict(zip(tasks, heads)),
              "thresholds": thresholds, "report": report, "method": METHOD}
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(bundle, a.out, compress=3)
    print("saved", a.out, os.path.getsize(a.out) // 1024, "KB")


if __name__ == "__main__":
    main()
