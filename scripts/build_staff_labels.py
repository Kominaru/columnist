"""The tournament-staff labels as one table, for training and for the shared dataset.

    python scripts/build_staff_labels.py      # -> data/staff_labels.csv.gz, data/staff_evidence.csv.gz

staff_labels: {1, 0, NaN} per beatmap_id and category, from tournament Google Sheets (wiki and
forum, data/pool_sheets.jsonl.gz) and wiki sub-slot names (data/mappools.csv). TB is left out:
sheet "TB" cells usually mark the bracket's tiebreaker pick, and wiki TB slots are already in the
v2 labels. staff_evidence: one row per (beatmap_id, category, value) with the texts and the
sheets / wiki pages they came from, so every label can be traced.
"""
import argparse
import gzip
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from osu_patterns import pool_annotations as pa  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data", type=Path)
    a = ap.parse_args()
    d = a.data
    diffs = pa.set_diffs(d / "set_text.jsonl.gz")
    rows = [json.loads(l) for l in gzip.open(d / "pool_sheets.jsonl.gz", "rt", encoding="utf-8")]
    pools = pd.read_csv(d / "mappools.csv")
    sheet, _ = pa.labels_from_sheets(d / "pool_sheets.jsonl.gz", False, diffs)
    wiki, _ = pa.labels_from_wiki(pools)
    staff = pd.concat([sheet, wiki]).groupby(level=0).max()
    staff["TB"] = np.nan
    staff = staff[staff.notna().any(axis=1)].drop(columns="TB")

    # evidence: which texts in which sheet tabs / wiki pages gave each label
    tour = {}
    for r in rows:
        tour[r["sheet"]] = r["tournament"]
    ev = defaultdict(set)
    for (sid, tab), maps in pa.sheet_groups(rows, diffs).items():
        for b, texts in maps.items():
            for t in texts:
                for c in pa.classify(t):
                    ev[(b, c)].add((f"sheet:{sid}", tour.get(sid, ""), tab, t))
    for t, maps in pa.wiki_groups(pools).items():
        for b, texts in maps.items():
            for s in texts:
                for c in pa.classify(s):
                    ev[(b, c)].add(("wiki", t, "", s))
    out = []
    for b in staff.index:
        for c in staff.columns:
            v = staff.at[b, c]
            if np.isnan(v):
                continue
            src = sorted(ev.get((b, c), set()))
            if v == 0:  # negatives come from sibling annotations in the same tab
                src = sorted({x for cc in staff.columns for x in ev.get((b, cc), set())})
            out.append({"beatmap_id": b, "category": c, "label": int(v),
                        "sources": " | ".join(f"{s[0]} [{s[1]}{'/' + s[2] if s[2] else ''}] {s[3]}" for s in src)[:1000]})
    staff.rename_axis("beatmap_id").to_csv(d / "staff_labels.csv.gz", float_format="%.0f")
    pd.DataFrame(out).to_csv(d / "staff_evidence.csv.gz", index=False)
    print(f"{len(staff)} charts, {int(staff.notna().sum().sum())} labels "
          f"({int((staff == 1).sum().sum())} positive) -> {d / 'staff_labels.csv.gz'}")
    print((staff == 1).sum().to_dict())


if __name__ == "__main__":
    main()
