"""How the note model's live bars behave in time: a causal model (past notes only) against the
bidirectional one, on held-out 4K charts, with the player's trailing bars (service/viz/static/app.js).

    python scripts/analyze_causal.py results/note/f0.npz results/note/f0_causal.npz --fold 0 --charts 200

For every chart and category it builds the bar series both models would show (exponentially weighted
mean of the notes played in the last 2 s, sampled every 250 ms) and reports:
  * onset latency against the row rules: a run of notes the rules call jumpstream/handstream,
    chordjack, stream/trill or LN (>= 50% of the trailing weight for >= 2 s) starts at time s; the
    model's latency is when its bar first crosses its cut within [s - 4 s, s + 6 s], minus s.
    Negative means the bar lit before the player reached the pattern (look-ahead).
  * lag: the shift that best aligns the causal bars with the bidirectional ones (cross-correlation).
  * flicker: cut crossings per minute; agreement: correlation of the two series.
Chart-level AP comes from the fold runs' .json reports.
"""
import argparse
import gzip
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from osu_patterns import live, mania, note_model  # noqa: E402

TRAIL_MS, TAU_MS, FULL_W, HOP = 2000.0, 700.0, 3.0, 250.0
RULES = {"Jumpstream/Handstream": {"jumpstream", "handstream"}, "Chordjack": {"chordjack"},
         "Streams": {"stream", "trill"}, "Density/Inverse": {"ln"}, "Release": {"ln"}, "Coordination": {"ln"}}
CATS = ["Chordjack", "Streams", "Jumpstream/Handstream", "Technical", "Release", "Coordination",
        "Density/Inverse", "Reading", "Memo"]


def trailing(nt, v, grid):
    """The app's trailing bars: v (notes,) values at note times nt, sampled at grid."""
    out = np.full(len(grid), np.nan)
    i0 = np.searchsorted(nt, grid - TRAIL_MS)
    i1 = np.searchsorted(nt, grid + 30)
    for g, (a, b) in enumerate(zip(i0, i1)):
        if b - a < 2:
            continue
        w = np.exp(-np.maximum(0, grid[g] - nt[a:b]) / TAU_MS)
        out[g] = (w @ v[a:b]) / w.sum() * min(1.0, w.sum() / FULL_W)
    return out


def runs(on, min_len):
    """Start indices of True runs at least min_len samples long."""
    starts, k = [], 0
    while k < len(on):
        if on[k]:
            e = k
            while e < len(on) and on[e]:
                e += 1
            if e - k >= min_len:
                starts.append(k)
            k = e
        else:
            k += 1
    return starts


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("bi")
    ap.add_argument("causal")
    ap.add_argument("--fold", type=int, default=0)
    ap.add_argument("--charts", type=int, default=200)
    ap.add_argument("--out", default="results/note/causal_analysis.json")
    a = ap.parse_args()
    models = {"bidirectional": note_model.NoteModel(a.bi), "causal": note_model.NoteModel(a.causal)}
    th = {k: {t: max(v, 0.2) for t, v in m.thresholds.items()} for k, m in models.items()}

    folds = pd.read_csv(ROOT / "data/v2_folds.csv", index_col=0)["fold"]
    ids = [b for b in folds.index[folds == a.fold] if (ROOT / f"data/osu/{b}.osu.gz").exists()]
    rng = np.random.default_rng(0)
    rng.shuffle(ids)

    lat = {k: {c: [] for c in RULES} for k in models}
    lag, corr, flick, minutes, n_done = {c: [] for c in CATS}, {c: [] for c in CATS}, \
        {k: {c: 0 for c in CATS} for k in models}, 0.0, 0
    for b in ids:
        if n_done >= a.charts:
            break
        try:
            m = mania.parse_mania(gzip.open(ROOT / f"data/osu/{b}.osu.gz", "rt", encoding="utf-8", errors="replace").read())
        except Exception:
            continue
        if m is None or m.keys != 4 or len(m.start) < 200:
            continue
        nt = m.start.astype(float)
        grid = np.arange(nt[0], m.end.max() + TRAIL_MS, HOP)
        minutes += (grid[-1] - grid[0]) / 60000
        series = {}
        for k, mod in models.items():
            r = mod.scores(m)
            series[k] = {c: trailing(nt, r["note"][:, mod.tasks.index(c)], grid) for c in CATS if c in mod.tasks}
        labs = np.array(live.local_labels(m))
        for c, group in RULES.items():
            frac = trailing(nt, np.isin(labs, list(group)).astype(float), grid)
            for s in runs(np.nan_to_num(frac) >= 0.5, int(2000 / HOP)):
                lo, hi = max(0, s - int(4000 / HOP)), min(len(grid), s + int(6000 / HOP))
                for k in models:
                    x = np.nan_to_num(series[k][c][lo:hi]) >= th[k][c]
                    if x[0]:          # already lit before the window: not an onset
                        continue
                    hit = np.flatnonzero(x)
                    if len(hit):
                        lat[k][c].append((lo + hit[0] - s) * HOP / 1000)
        for c in CATS:
            x, y = np.nan_to_num(series["bidirectional"][c]), np.nan_to_num(series["causal"][c])
            if x.std() > 1e-3 and y.std() > 1e-3:
                corr[c].append(float(np.corrcoef(x, y)[0, 1]))
                sh = range(-int(4000 / HOP), int(4000 / HOP) + 1)
                cc = [np.corrcoef(x[max(0, -d):len(x) - max(0, d)], y[max(0, d):len(y) - max(0, -d)])[0, 1] for d in sh]
                lag[c].append(list(sh)[int(np.nanargmax(cc))] * HOP / 1000)
            for k in models:
                on = np.nan_to_num(series[k][c]) >= th[k][c]
                flick[k][c] += int(np.abs(np.diff(on.astype(int))).sum())
        n_done += 1
        if n_done % 25 == 0:
            print(f"{n_done} charts", flush=True)

    rep = {"charts": n_done, "latency_s": {}, "lag_s": {}, "corr": {}, "flicker_per_min": {}, "ap": {}}
    for k in models:
        rep["latency_s"][k] = {c: {"n": len(v), "median": round(float(np.median(v)), 2) if v else None,
                                   "early_frac": round(float(np.mean(np.array(v) < 0)), 2) if v else None}
                               for c, v in lat[k].items()}
        rep["flicker_per_min"][k] = {c: round(v / max(minutes, 1e-9), 2) for c, v in flick[k].items()}
        rep["ap"][k] = {t: v["ap"] for t, v in models[k].report.items()}
    rep["lag_s"] = {c: round(float(np.median(v)), 2) if v else None for c, v in lag.items()}
    rep["corr"] = {c: round(float(np.median(v)), 2) if v else None for c, v in corr.items()}
    for k in models:
        cats = [v for t, v in rep["ap"][k].items() if not t.startswith("tag:")]
        tags = [v for t, v in rep["ap"][k].items() if t.startswith("tag:")]
        rep["ap"][k]["_mean_categories"] = round(float(np.mean(cats)), 3) if cats else None
        rep["ap"][k]["_mean_tags"] = round(float(np.mean(tags)), 3) if tags else None
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(rep, indent=1), "utf-8")
    print(json.dumps(rep, indent=1))


if __name__ == "__main__":
    main()
