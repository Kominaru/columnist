"""Average per-fold teacher files (columns "<k>|<task>", one row per chart) into a new teacher.

    python scripts/mean_teacher.py data/v4_teacher.csv.gz data/v3_teacher.csv.gz data/mlp3_teacher.csv.gz
"""
import sys

import pandas as pd


def main():
    out, *ins = sys.argv[1:]
    first = pd.read_csv(ins[0], index_col=0)
    rest = [pd.read_csv(p, index_col=0).reindex(index=first.index, columns=first.columns) for p in ins[1:]]
    ((first + sum(rest)) / len(ins)).to_csv(out, float_format="%.4g")


if __name__ == "__main__":
    main()
