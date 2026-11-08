"""Etterna MinaCalc (v527) skillset ratings for every downloaded chart, as extra features.

    cargo build --release --manifest-path tools/msdcli/Cargo.toml
    python scripts/build_msd_features.py --out data/msd_features.csv.gz

MSD at 1.0x for overall, stream, jumpstream, handstream, stamina, jackspeed, chordjack and technical,
plus each skillset divided by overall (msdrel_*: what the chart is "about", independent of how hard
it is). MinaCalc rates 4K, 6K and 7K only; other key counts get no row (NaN downstream). Long notes
count as their heads, as in Etterna.
"""
import argparse
import subprocess
import sys
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from osu_patterns import mania  # noqa: E402

SKILLS = ["overall", "stream", "jumpstream", "handstream", "stamina", "jackspeed", "chordjack", "technical"]
BIN = Path(__file__).resolve().parents[1] / "tools/msdcli/target/release/msdcli"


def chart_text(p: Path) -> str | None:
    """The calculator's input for one chart: a header "<id> <keys> <rows>", then "<seconds> <column mask>" per row."""
    try:
        m = mania.load(p)
        if m is None or len(m.start) < 50 or m.keys not in (4, 6, 7):
            return None
        t = np.round(np.asarray(m.start, float)).astype(np.int64)
        rows, inv = np.unique(t, return_inverse=True)
        mask = np.zeros(len(rows), np.int64)
        np.bitwise_or.at(mask, inv, 1 << np.asarray(m.col, np.int64))
        body = "\n".join(f"{x / 1000:.4f} {mk}" for x, mk in zip(rows - rows[0], mask))
        return f"{p.name.split('.')[0]} {m.keys} {len(rows)}\n{body}\n"
    except Exception:  # noqa: BLE001 - one broken chart shouldn't stop the build
        return None


def run_chunk(files: list[Path]) -> dict[int, list[float]]:
    text = "".join(x for x in map(chart_text, files) if x)
    r = subprocess.run([str(BIN)], input=text, capture_output=True, text=True, check=True)
    out = {}
    for line in r.stdout.splitlines():
        p = line.split()
        if len(p) == 1 + len(SKILLS):
            out[int(p[0])] = [float(v) for v in p[1:]]
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--osu", default="data/osu")
    ap.add_argument("--out", default="data/msd_features.csv.gz")
    ap.add_argument("--jobs", type=int, default=4)
    a = ap.parse_args()
    files = sorted(Path(a.osu).glob("*.osu.gz"))
    rows = {}
    with Pool(a.jobs) as pool:
        for r in pool.imap_unordered(run_chunk, [files[i:i + 200] for i in range(0, len(files), 200)]):
            rows.update(r)
    df = pd.DataFrame.from_dict(rows, orient="index", columns=[f"msd_{s}" for s in SKILLS]).rename_axis("beatmap_id")
    df = df.where(df["msd_overall"] > 0).sort_index()
    for s in SKILLS[1:]:
        df[f"msdrel_{s}"] = df[f"msd_{s}"] / df["msd_overall"]
    df.to_csv(a.out, float_format="%.4g")
    print(f"{len(df)} charts rated")


if __name__ == "__main__":
    main()
