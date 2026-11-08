"""Section-level cuts for the live visualizer's window scores.

    python scripts/calibrate_live_windows.py --osu data/osu --n 3000

A 16 s window scored by a chart-level head is not calibrated like a whole chart (Technical, for
one, is a chart-wide property and rarely peaks inside one window). This script scores sliding
windows on labelled 4K charts and, per task, picks the cut on the *highest* window score that
best predicts the chart's label (best F1): "a chart is X when some section of it is X". The cut
is stored in models/live_heads.pkl as window_thresholds and is what the visualizer uses to say a
section "is" a pattern.

The heads saw these charts in training, so the scores are somewhat optimistic; the cut is used for
display only, not reported as accuracy.
"""
import argparse
import os
import sys
from multiprocessing import Pool
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "1")

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, f1_score

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from osu_patterns import live, mania  # noqa: E402

MODEL = None


def _init(models):
    global MODEL
    MODEL = live.LiveModel(models)


def one(args):
    bid, path = args
    try:
        m = mania.load(path)
        if m is None or m.keys != 4 or len(m.start) < 50:
            return bid, None
        a = MODEL.analyze(m, hop=4000.0, sizes=(live.WIN_MS,))
        return bid, {t: max(s) for t, s in a["windows"]["series"].items() if s and not t.startswith("slot:")}
    except Exception:  # noqa: BLE001
        return bid, None


def best_cut(y, p):
    grid = np.unique(np.quantile(p, np.linspace(0.05, 0.95, 37)))
    f = [f1_score(y, p >= g) for g in grid]
    return float(grid[int(np.argmax(f))]), float(max(f))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--osu", default="data/osu")
    ap.add_argument("--models", default="models")
    ap.add_argument("--labels", default="data/v2_labels.csv.gz")
    ap.add_argument("--n", type=int, default=3000)
    ap.add_argument("--jobs", type=int, default=4)
    ap.add_argument("--out-csv", default="results/live/window_max.csv")
    a = ap.parse_args()
    L = pd.read_csv(a.labels, index_col=0)
    L = L.drop(columns=[c for c in ("slot",) if c in L])
    have = {int(p.name.split(".")[0]): p for p in Path(a.osu).glob("*.osu.gz")}
    ids = [b for b in L.index[L.notna().any(axis=1)] if b in have]
    rng = np.random.default_rng(0)
    # every chart that is positive for a rare task, then a random fill
    rare = [b for b in ids if (L.loc[b, [c for c in L if (L[c] == 1).sum() < 600]] == 1).any()]
    rest = [b for b in ids if b not in set(rare)]
    pick = list(dict.fromkeys(rare + list(rng.permutation(rest)[: max(a.n - len(rare), 0)])))
    print(f"{len(pick)} charts ({len(rare)} rare-task positives)", flush=True)
    rows = {}
    with Pool(a.jobs, initializer=_init, initargs=(a.models,)) as pool:
        for i, (bid, r) in enumerate(pool.imap_unordered(one, [(b, have[b]) for b in pick], chunksize=8)):
            if r:
                rows[bid] = r
            if i % 250 == 0:
                print(i, flush=True)
    W = pd.DataFrame.from_dict(rows, orient="index")
    Path(a.out_csv).parent.mkdir(parents=True, exist_ok=True)
    W.rename_axis("beatmap_id").to_csv(a.out_csv, float_format="%.4g")

    bundle = joblib.load(Path(a.models) / "live_heads.pkl")
    cuts, rep = {}, {}
    for t in W.columns:
        y = L[t].reindex(W.index)
        m = y.notna() & W[t].notna()
        if (y[m] == 1).sum() < 8 or (y[m] == 0).sum() < 8:
            continue
        yy, pp = y[m].to_numpy().astype(int), W[t][m].to_numpy()
        cuts[t], f1 = best_cut(yy, pp)
        rep[t] = {"n": int(m.sum()), "pos": int(yy.sum()), "cut": round(cuts[t], 3), "f1": round(f1, 3),
                  "ap_max_window": round(float(average_precision_score(yy, pp)), 3),
                  "chart_cut": round(bundle["thresholds"][t], 3)}
        print(t, rep[t], flush=True)
    bundle["window_thresholds"] = cuts
    bundle["window_report"] = rep
    joblib.dump(bundle, Path(a.models) / "live_heads.pkl", compress=3)
    print("saved window_thresholds for", len(cuts), "tasks")


if __name__ == "__main__":
    main()
