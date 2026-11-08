"""Does adding the text-mined labels (osu_patterns.pool_annotations) help gbm+win? 4K focus.

    python scripts/train_with_pool_labels.py --set-text data/set_text.jsonl

Same song-grouped v2 folds and gbm+win inputs as teacher_cv.py. Variants differ only in the
training labels: v2 labels, v2 + mined sheet/wiki labels, v2 + every mined source. Mined labels
only fill charts the v2 labels leave unknown. Scored out of fold on
  v2      the strict v2 labels (the benchmark), all charts and 4K charts
  mined   the mined sheet/wiki labels of 4K charts that have no v2 label (a tournament-staff
          benchmark the baseline never trains on)
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
sys.path.insert(0, str(Path(__file__).resolve().parent))
from osu_patterns import categories, dataset, evaluate, pool_annotations as pa  # noqa: E402
from teacher_cv import features, fit_predict  # noqa: E402


def mined(data: Path, set_text: list[str]):
    ex = [p for p in set_text if Path(p).exists()]
    diffs = pa.set_diffs(*ex)
    staff = pd.concat([pa.labels_from_sheets(data / "pool_sheets.jsonl.gz", False, diffs)[0],
                       pa.labels_from_wiki(pd.read_csv(data / "mappools.csv"))[0]])
    staff = staff.groupby(level=0).max()  # positive anywhere wins
    # sheet "TB" cells are often the bracket's tiebreaker pick, not a TB-style map: training on
    # them cost 0.03 AP on the v2 TB labels, so TB keeps its wiki labels only
    staff["TB"] = np.nan
    text = [pa.labels_from_set_text(p, f)[0] for p in ex for f in (("diff",), ("tags",), ("description",))]
    alltext = pd.concat([staff, *text]).groupby(level=0).max() if text else staff
    return staff, alltext


def add_charts(d, allf, fold, feats, titles, slot_model):
    """Append charts outside the corpus. A chart whose song is already in the v2 folds joins that
    song's fold; a new song gets a fold from a stable hash of its name, so no song spans folds."""
    import zlib
    from osu_patterns import slots
    new = pd.read_csv(feats, index_col=0)
    new = new.loc[new.index.difference(allf.index)]
    t = pd.read_csv(titles, index_col=0)["title"]
    sm = slots.SlotModel.load(slot_model)
    st = new[d.stats.columns]
    d.stats = pd.concat([d.stats, st])
    d.probs = pd.concat([d.probs, pd.DataFrame(sm.clf.predict_proba(st[sm.features].values),
                                               index=st.index, columns=sm.classes)])
    by_group = fold.drop_duplicates("group").set_index("group")["fold"]
    rows = {}
    for b in new.index:
        g = slots.song_key(str(t.get(b, b)) + " [x]") or str(b)
        rows[b] = {"fold": int(by_group.get(g, zlib.crc32(g.encode()) % 5)), "group": g}
    print(f"added {len(new)} charts outside the corpus", flush=True)
    return pd.concat([allf, new[allf.columns.intersection(new.columns)]]), \
        pd.concat([fold, pd.DataFrame.from_dict(rows, orient="index")])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data", type=Path)
    ap.add_argument("--slot-model", default="models/slots.pkl")
    ap.add_argument("--set-text", nargs="*", default=["data/set_text.jsonl.gz"])
    ap.add_argument("--out", default="results/pool_labels")
    ap.add_argument("--jobs", type=int, default=4)
    ap.add_argument("--seeds", type=int, default=1)
    ap.add_argument("--extra-charts", default="data/sheet_extra_features.csv.gz", help="features (build_all_features.py) of labelled sheet charts "
                    "outside the corpus; --extra-titles gives their 'artist - title' for song folds")
    ap.add_argument("--extra-titles", default="data/sheet_extra_titles.csv")
    a = ap.parse_args()
    d = dataset.category_dataset(a.data, a.slot_model)
    allf = pd.read_csv(a.data / "all_features.csv.gz", index_col=0)
    allf = allf[~allf.index.duplicated()]
    fold = pd.read_csv(a.data / "v2_folds.csv", index_col=0)
    if a.extra_charts:
        allf, fold = add_charts(d, allf, fold, a.extra_charts, a.extra_titles, a.slot_model)
    _, Xw = features(d, allf)
    X = Xw.to_numpy(np.float32)
    ids = allf.index
    k4 = (allf["keys"] == 4).to_numpy()
    cats = [c for c in categories.NAMES]
    Y = pd.read_csv(a.data / "v2_labels.csv.gz", index_col=0).reindex(ids)[cats]
    fold = fold["fold"].reindex(ids).to_numpy()
    staff, alltext = mined(a.data, a.set_text)
    staff, alltext = staff.reindex(ids)[cats], alltext.reindex(ids)[cats]
    train = {"v2": Y, "v2+staff": Y.fillna(staff), "v2+staff+text": Y.fillna(alltext)}
    for n, T in train.items():
        print(f"{n}: labels per category {T.notna().sum().to_dict()}", flush=True)
    jobs = []
    for n, T in train.items():
        for c in cats:
            y = T[c].to_numpy()
            for k in range(5):
                for s in range(a.seeds):
                    jobs.append((n, c, k, s, y, ~np.isnan(y) & (fold != k), np.where(fold == k)[0]))
    print(f"{len(jobs)} fits", flush=True)
    res = Parallel(n_jobs=a.jobs, verbose=2)(delayed(fit_predict)(X, j[4], j[5], j[6], None, j[3]) for j in jobs)
    P = {n: pd.DataFrame(0.0, index=ids, columns=cats) for n in train}
    for (n, c, k, s, *_), p in zip(jobs, res):
        P[n].iloc[np.where(fold == k)[0], cats.index(c)] += p / a.seeds
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    pd.concat(P, axis=1).to_csv(out / "oof.csv.gz", float_format="%.4g")
    # held-out tournament-staff labels: 4K charts the v2 labels say nothing about
    held = staff.where(Y.isna())
    tables = {"v2 labels, all charts": (Y, np.ones(len(ids), bool)), "v2 labels, 4K": (Y, k4),
              "mined staff labels, 4K, not in v2": (held, k4)}
    for name, (L, m) in tables.items():
        t = evaluate.ap_table(L[m], {n: p[m] for n, p in P.items()}, min_pos=5)
        print(f"\n== {name} ==\n{t.to_string()}", flush=True)
        t.to_csv(out / f"ap_{name.split(',')[0].replace(' ', '_')}{'_4k' if '4K' in name else ''}.csv")


if __name__ == "__main__":
    main()
