"""Held-out comparison of Lens per-note models on the same songs and the same labels.

    python scripts/compare_note_heads.py --fold 0 \
        --oof live=results/note/f0_causal_nogrid.oof.csv columnist=results/columnist/col_f0_oof_f0.csv \
        --notes live=results/note/f0_causal_nogrid.npz columnist=models/columnist/col_f0.npz --charts 150

Part 1, chart level: average precision per task of each model's chart probabilities on the charts all of them scored
(held-out songs of the fold), against the strict v6 labels; a song-level bootstrap gives the spread of each model's
mean AP and of its difference to the first model.
Part 2, per note (what the player shows), on held-out charts, through the code the site runs: how well the note scores
pick out the notes the row rules call jumpstream / handstream / chordjack / stream / LN (AUC over all notes), how soon the
trailing bar crosses its cut after such a pattern starts (median seconds; negative = lit before the pattern arrived),
how often it flips on and off (per minute), and how often a bar is lit on a chart that has none of that pattern.
"""
import argparse
import gzip
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))
from analyze_causal import HOP, RULES, TRAIL_MS, runs, trailing  # noqa: E402
from osu_patterns import live, mania, note_model, note_model_columnist, note_model_v3  # noqa: E402


def load_note_model(path):
    z = np.load(path)
    if "enc.tok.weight" in z.files:
        return note_model_columnist.NoteModelColumnist(path)
    if "row.weight" in z.files:
        return note_model_v3.NoteModelV3(path)
    return note_model.NoteModel(path)


def chart_level(a):
    labels = pd.read_csv(Path(a.data) / "v6_labels.csv.gz", index_col=0)
    folds = pd.read_csv(Path(a.data) / "v2_folds.csv", index_col=0)
    P = {k: pd.read_csv(v, index_col=0) for k, v in (x.split("=", 1) for x in a.oof)}
    common = None
    for df in P.values():
        common = df.index if common is None else common.intersection(df.index)
    common = common.intersection(folds.index[folds.fold == a.fold])
    tasks = [t for t in next(iter(P.values())).columns if all(t in df.columns for df in P.values()) and t in labels.columns]
    Y = labels.reindex(common)[tasks]
    groups = folds.loc[common, "group"].fillna(pd.Series(common.astype(str), index=common)).to_numpy()
    ug = np.unique(groups)
    rng = np.random.default_rng(0)

    def mean_ap(idx):
        out = {}
        for k, df in P.items():
            aps = []
            for t in tasks:
                y = Y[t].to_numpy()[idx]
                ok = ~np.isnan(y)
                if ok.sum() and 5 <= y[ok].sum() < ok.sum():
                    aps.append(average_precision_score(y[ok], df.loc[common, t].to_numpy()[idx][ok]))
            out[k] = float(np.mean(aps))
        return out

    full = np.arange(len(common))
    point = mean_ap(full)
    boots = []
    by_group = {g: np.flatnonzero(groups == g) for g in ug}
    for _ in range(a.boot):
        pick = rng.choice(ug, len(ug), replace=True)
        boots.append(mean_ap(np.concatenate([by_group[g] for g in pick])))
    first = next(iter(P))
    rows = []
    for t in tasks:
        y = Y[t].to_numpy()
        ok = ~np.isnan(y)
        if 5 <= y[ok].sum() < ok.sum():
            rows.append({"task": t, "n": int(ok.sum()), "pos": int(y[ok].sum()),
                         **{k: round(average_precision_score(y[ok], df.loc[common, t].to_numpy()[ok]), 3) for k, df in P.items()}})
    table = pd.DataFrame(rows).set_index("task")
    print(f"\nChart level: {len(common)} held-out charts of fold {a.fold}, {len(tasks)} tasks\n")
    print(table.to_string())
    res = {"charts": len(common), "mean_ap": {k: round(v, 4) for k, v in point.items()}, "ci95": {}, "diff_vs_first": {}}
    for k in P:
        b = np.array([x[k] for x in boots])
        res["ci95"][k] = [round(float(np.percentile(b, 2.5)), 4), round(float(np.percentile(b, 97.5)), 4)]
        if k != first:
            d = np.array([x[k] - x[first] for x in boots])
            res["diff_vs_first"][k] = {"diff": round(point[k] - point[first], 4),
                                       "ci95": [round(float(np.percentile(d, 2.5)), 4), round(float(np.percentile(d, 97.5)), 4)]}
    print(json.dumps(res, indent=1))
    return res, table


def per_note(a):
    models = {k: load_note_model(v) for k, v in (x.split("=", 1) for x in a.notes)}
    th = {k: {t: max(v, 0.2) for t, v in m.thresholds.items()} for k, m in models.items()}
    folds = pd.read_csv(Path(a.data) / "v2_folds.csv", index_col=0)["fold"]
    ids = [b for b in folds.index[folds == a.fold] if (Path(a.osu) / f"{b}.osu.gz").exists()]
    np.random.default_rng(0).shuffle(ids)
    pool = {k: {c: ([], []) for c in RULES} for k in models}
    lat = {k: {c: [] for c in RULES} for k in models}
    flick = {k: {c: 0 for c in RULES} for k in models}
    lit_empty = {k: {c: [0, 0] for c in RULES} for k in models}
    minutes, done = 0.0, 0
    for b in ids:
        if done >= a.charts:
            break
        try:
            m = mania.parse_mania(gzip.open(Path(a.osu) / f"{b}.osu.gz", "rt", encoding="utf-8", errors="replace").read())
        except Exception:
            continue
        if m is None or m.keys != 4 or len(m.start) < 200:
            continue
        nt = m.start.astype(float)
        grid = np.arange(nt[0], m.end.max() + TRAIL_MS, HOP)
        minutes += (grid[-1] - grid[0]) / 60000
        labs = np.array(live.local_labels(m))
        res = {k: mod.scores(m) for k, mod in models.items()}
        for c, group in RULES.items():
            is_rule = np.isin(labs, list(group))
            frac = trailing(nt, is_rule.astype(float), grid)
            starts = runs(np.nan_to_num(frac) >= 0.5, int(2000 / HOP))
            for k in models:
                if c not in res[k]["tasks"]:
                    continue
                v = res[k]["note"][:, res[k]["tasks"].index(c)]
                pool[k][c][0].append(v)
                pool[k][c][1].append(is_rule)
                bar = np.nan_to_num(trailing(nt, v, grid)) >= th[k].get(c, 0.5)
                flick[k][c] += int(np.abs(np.diff(bar.astype(int))).sum())
                if not starts:
                    lit_empty[k][c][0] += int(bar.sum())
                    lit_empty[k][c][1] += len(bar)
                for s in starts:
                    lo, hi = max(0, s - int(4000 / HOP)), min(len(grid), s + int(6000 / HOP))
                    x = bar[lo:hi]
                    if x[0]:
                        continue
                    hit = np.flatnonzero(x)
                    if len(hit):
                        lat[k][c].append((lo + hit[0] - s) * HOP / 1000)
        done += 1
        if done % 25 == 0:
            print(done, "charts", flush=True)
    rep = {"charts": done, "minutes": round(minutes, 1)}
    for k in models:
        rep[k] = {}
        for c in RULES:
            v, r = pool[k][c]
            if not v:
                continue
            v, r = np.concatenate(v), np.concatenate(r)
            rep[k][c] = {"note_auc": round(float(roc_auc_score(r, v)), 3) if 0 < r.sum() < len(r) else None,
                         "latency_median_s": round(float(np.median(lat[k][c])), 2) if lat[k][c] else None,
                         "lit_before_pattern": round(float(np.mean(np.array(lat[k][c]) < 0)), 2) if lat[k][c] else None,
                         "flips_per_min": round(flick[k][c] / max(minutes, 1e-9), 2),
                         "lit_on_charts_without": round(lit_empty[k][c][0] / max(lit_empty[k][c][1], 1), 3)}
    print(json.dumps(rep, indent=1))
    return rep


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fold", type=int, default=0)
    ap.add_argument("--data", default=str(ROOT.parent / "om-bn-pipeline-auth/data"))
    ap.add_argument("--osu", default=str(ROOT.parent / "om-bn-pipeline-auth/data/osu/osu"))
    ap.add_argument("--oof", nargs="*", default=[])
    ap.add_argument("--notes", nargs="*", default=[])
    ap.add_argument("--charts", type=int, default=150)
    ap.add_argument("--boot", type=int, default=200)
    ap.add_argument("--out", default="results/columnist/compare.json")
    a = ap.parse_args()
    out = {}
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    if a.oof:
        out["chart_level"], table = chart_level(a)
        table.to_csv(Path(a.out).with_suffix(".chart.csv"))
    if a.notes:
        out["per_note"] = per_note(a)
    Path(a.out).write_text(json.dumps(out, indent=1), "utf-8")


if __name__ == "__main__":
    main()
