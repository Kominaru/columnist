"""Pooled sliding-window statistics for every chart with cached chart statistics.

    python scripts/build_window_features.py --data data --osu data/osu --out data/window_features.csv
"""
import argparse
import sys
from multiprocessing import Pool
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from osu_patterns import mania, windows  # noqa: E402

OSU = Path("data/osu")


def one(bid):
    p = OSU / f"{bid}.osu.gz"
    if not p.exists() or p.stat().st_size < 40:
        return bid, None
    m = mania.load(p)
    return bid, (windows.pool(windows.window_stats(m)) if m is not None and len(m.start) else None)


def main():
    global OSU
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data")
    ap.add_argument("--osu", default="data/osu")
    ap.add_argument("--out", default="data/window_features.csv")
    ap.add_argument("--jobs", type=int, default=4)
    a = ap.parse_args()
    OSU = Path(a.osu)
    d = Path(a.data)
    ids = pd.concat([pd.read_csv(d / "mania_features.csv", index_col=0),
                     pd.read_csv(d / "ranked_features.csv", index_col=0)]).index.unique()
    out = Path(a.out)
    done = pd.read_csv(out, index_col=0) if out.exists() else pd.DataFrame()
    todo = [b for b in ids if b not in done.index]
    rows = {}
    with Pool(a.jobs, initializer=_set_osu, initargs=(OSU,)) as pool:
        for i, (bid, f) in enumerate(pool.imap_unordered(one, todo, chunksize=16)):
            if f:
                rows[bid] = f
            if i % 1000 == 0:
                print(i, len(todo), flush=True)
    new = pd.DataFrame.from_dict(rows, orient="index")
    allf = pd.concat([done, new])
    allf.index.name = "beatmap_id"
    allf.to_csv(out)
    print(f"{len(new)} new, {len(allf)} charts with window features")


def _set_osu(p):
    global OSU
    OSU = p


if __name__ == "__main__":
    main()
