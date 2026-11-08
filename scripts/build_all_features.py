"""Chart statistics and pooled window statistics for every downloaded chart.

    python scripts/build_all_features.py --osu data/osu --out data/all_features.csv.gz
    python scripts/build_all_features.py --kind patterns --out data/pattern_features.csv.gz

--kind stats (default): one row per beatmap_id with the 71 chart statistics (mania.stats) followed
by the pooled sliding-window statistics (windows.pool, columns prefixed wmax: / wp75: / wstd:).
These are the boosted heads' inputs and the deep model's distillation targets.
--kind patterns: the bag of row transitions (osu_patterns.patterns).
--kind reading: what scroll speed changes do to the notes on screen (osu_patterns.reading).
"""
import argparse
import sys
from multiprocessing import Pool
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from osu_patterns import mania, patterns, reading, windows  # noqa: E402

KIND = "stats"


def one(p):
    bid = int(p.name.split(".")[0])
    try:
        if p.stat().st_size < 40:
            return bid, None
        m = mania.load(p)
        if m is None or len(m.start) < 50 or m.keys < 4:
            return bid, None
        if KIND == "patterns":
            return bid, patterns.pattern_features(m)
        if KIND == "reading":
            return bid, reading.reading_features(m)
        return bid, {**mania.stats(m), **windows.pool(windows.window_stats(m))}
    except Exception:  # noqa: BLE001 - one broken chart shouldn't stop the build
        return bid, None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--osu", default="data/osu")
    ap.add_argument("--out", default="data/all_features.csv.gz")
    ap.add_argument("--jobs", type=int, default=4)
    ap.add_argument("--kind", choices=["stats", "patterns", "reading"], default="stats")
    a = ap.parse_args()
    global KIND
    KIND = a.kind
    out = Path(a.out)
    done = pd.read_csv(out, index_col=0) if out.exists() else pd.DataFrame()
    todo = [p for p in Path(a.osu).glob("*.osu.gz") if int(p.name.split(".")[0]) not in done.index]
    rows = {}
    with Pool(a.jobs, initializer=_set_kind, initargs=(a.kind,)) as pool:
        for i, (bid, f) in enumerate(pool.imap_unordered(one, todo, chunksize=32)):
            if f:
                rows[bid] = f
            if i % 2000 == 0:
                print(i, len(todo), flush=True)
    allf = pd.concat([done, pd.DataFrame.from_dict(rows, orient="index")])
    allf.index.name = "beatmap_id"
    allf.sort_index().to_csv(out, float_format="%.5g")
    print(f"{len(rows)} new, {len(allf)} charts")


def _set_kind(k):
    global KIND
    KIND = k


if __name__ == "__main__":
    main()
