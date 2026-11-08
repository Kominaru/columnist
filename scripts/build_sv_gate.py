"""The SV gate for every cached chart: does the scroll speed ever visibly change? (Komi, 2026-11-02)

    python scripts/build_sv_gate.py --osu data/osu --out data/sv_gate.csv.gz

One row per beatmap_id with reading.scroll_effect's columns (sve_rows, sve_frac, sve_off_frac,
sve_jumps, sve_effective). Reading, Memo and the sliders/* tags never fire where sve_effective is 0;
ensemble_v2.py --sv-gate scores the models that way.
"""
import argparse
import gzip
import os
import sys
from multiprocessing import Pool
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from osu_patterns import mania, reading  # noqa: E402

D = None


def one(f):
    p = os.path.join(D, f)
    try:
        op = gzip.open if f.endswith(".gz") else open
        with op(p, "rt", encoding="utf-8", errors="replace") as fh:
            m = mania.parse_mania(fh.read())
        if m is None or len(m.start) < 2:
            return None
        return {"beatmap_id": int(f.split(".")[0]), **reading.scroll_effect(m)}
    except Exception:  # noqa: BLE001 - a broken file just has no row (the gate then lets it through)
        return None


def init(d):
    global D
    D = d


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--osu", default="data/osu")
    ap.add_argument("--out", default="data/sv_gate.csv.gz")
    ap.add_argument("--jobs", type=int, default=os.cpu_count())
    a = ap.parse_args()
    fs = sorted(f for f in os.listdir(a.osu) if f.endswith((".osu", ".osu.gz")))
    with Pool(a.jobs, initializer=init, initargs=(a.osu,)) as p:
        rows = [r for r in p.imap_unordered(one, fs, chunksize=64) if r]
    out = pd.DataFrame(rows).set_index("beatmap_id").sort_index()
    out.to_csv(a.out, float_format="%.4g")
    print(f"{len(out)} charts, {out.sve_effective.mean():.1%} with effective scroll changes -> {a.out}")


if __name__ == "__main__":
    main()
