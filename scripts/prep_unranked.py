"""Prepare the unranked charts (list_unranked_mania.py) for pretraining in train_deep_v2.py --extra.

1. encodes them into an event cache (data/events_unranked.pkl);
2. drops every chart whose notes copy a corpus chart (data/v2_folds.csv): two fingerprints per
   chart, the column sequence of the first 300 note rows and of 300 rows from the middle, each
   taken as the smaller of the chart and its mirror image, so mirrored and rate-edited copies
   match too. The song-key filter in list_unranked_mania.py misses renamed reuploads; this
   doesn't;
3. computes the chart and window statistics (the stats distillation targets) for the kept charts.

    python scripts/prep_unranked.py --list data/unranked_4k.csv --osu data/osu

Writes data/unranked_4k_kept.csv and data/unranked_features.csv.gz.
"""
import argparse
import hashlib
import sys
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from osu_patterns import mania, windows  # noqa: E402
from osu_patterns.deep import data as ddata  # noqa: E402

N_ROWS = 300


def fingerprints(e) -> list[str]:
    heads = e.cols[:, :, 0].astype(bool)
    heads = heads[heads.any(1)]
    if len(heads) < 50:
        return []
    out = []
    for a in (0, max(0, len(heads) // 2 - N_ROWS // 2)):
        h = heads[a:a + N_ROWS]
        w = 1 << np.arange(e.keys)
        seq = min((h * w).sum(1).astype(np.uint16).tobytes(), (h[:, ::-1] * w).sum(1).astype(np.uint16).tobytes())
        out.append(hashlib.blake2b(bytes([e.keys]) + seq, digest_size=12).hexdigest())
    return out


def stats_one(p):
    bid = int(p.name.split(".")[0])
    try:
        m = mania.load(p)
        if m is None or len(m.start) < 50 or m.keys < 4:
            return bid, None
        return bid, {**mania.stats(m), **windows.pool(windows.window_stats(m))}
    except Exception:  # noqa: BLE001 - one broken chart shouldn't stop the build
        return bid, None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data")
    ap.add_argument("--list", default="data/unranked_4k.csv")
    ap.add_argument("--osu", default="data/osu")
    ap.add_argument("--events", default="data/events_all.pkl", help="the corpus event cache (train_deep_v2.py)")
    ap.add_argument("--extra-events", default="data/events_unranked.pkl")
    ap.add_argument("--out", default="data/unranked_4k_kept.csv")
    ap.add_argument("--features", default="data/unranked_features.csv.gz")
    ap.add_argument("--jobs", type=int, default=4)
    a = ap.parse_args()
    d = Path(a.data)
    lst = pd.read_csv(a.list)
    corpus = ddata.load_events(list(pd.read_csv(d / "v2_folds.csv", index_col=0).index), a.osu, a.events, jobs=a.jobs)
    extra = ddata.load_events(list(lst.beatmap_id), a.osu, a.extra_events, jobs=a.jobs)
    print(f"{len(corpus)} corpus charts, {len(extra)} of {len(lst)} unranked charts parsed", flush=True)

    seen = {f for e in corpus.values() for f in fingerprints(e)}
    kept, dup, own = [], 0, set()
    for bid in lst.beatmap_id:
        if bid not in extra:
            continue
        fp = fingerprints(extra[bid])
        if not fp or any(f in seen for f in fp):
            dup += bool(fp)
            continue
        if any(f in own for f in fp):  # copies inside the unranked list: keep one
            continue
        own.update(fp)
        kept.append(bid)
    print(f"dropped {dup} copies of corpus charts and {len(extra) - len(kept) - dup} unranked duplicates; "
          f"kept {len(kept)}", flush=True)
    out = lst[lst.beatmap_id.isin(kept)]
    out.to_csv(a.out, index=False)

    have = pd.read_csv(a.features, index_col=0) if Path(a.features).exists() else pd.DataFrame()
    todo = [Path(a.osu) / f"{b}.osu.gz" for b in kept if b not in have.index]
    rows = {}
    with Pool(a.jobs) as pool:
        for i, (bid, f) in enumerate(pool.imap_unordered(stats_one, todo, chunksize=32)):
            if f:
                rows[bid] = f
            if i % 5000 == 0:
                print(f"stats {i}/{len(todo)}", flush=True)
    feats = pd.concat([have, pd.DataFrame.from_dict(rows, orient="index")])
    feats.index.name = "beatmap_id"
    feats.sort_index().to_csv(a.features, float_format="%.5g")
    print(f"{len(out)} charts -> {a.out}; stats for {len(feats)} -> {a.features}")
    print(out.groupby("status").size().to_string())


if __name__ == "__main__":
    main()
