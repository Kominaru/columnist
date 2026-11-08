"""A shareable label dataset: chart ids, metadata, labels and where each label came from. No .osu files.

    python scripts/build_dataset_release.py --out dataset/ [--keys 4]

Writes
  charts.csv     beatmap_id, beatmapset_id, artist - title, creator, difficulty, keys, status, fold, song group
  labels.csv     one row per (beatmap_id, task, label, source, detail)
                 task: slot (RC/LN/HB/SV), Komi's categories, tag:<lazer user tag>
                 source: user_tags (lazer votes), wiki_pool (osu! wiki mappool sub-slot), staff_sheet
                         (tournament Google Sheet annotation), wiki_slot (main slot of a wiki pool)
  labels_wide.csv  the benchmark labels (v2 strict) and the staff labels, one column per task
  README.md
The v2 folds (song-grouped) are the evaluation splits used in the paper.
"""
import argparse
import gzip
import json
import shutil
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from osu_patterns import categories, mania_tags, slots  # noqa: E402

README = """# osu!mania pattern labels ({n_charts} charts{keys_note})

Labels for osu!mania charts: the tournament slot (Rice / LN / Hybrid / SV), eleven pattern
categories used by tournament mappoolers, and twenty osu!(lazer) user tags. Built for the
om-bn-pipeline pattern classifier. No beatmap files are included: download a chart from
`https://osu.ppy.sh/osu/<beatmap_id>` (or any mirror) to work with the notes.

## Files

* `charts.csv`: one row per chart. `fold` (0-4) is a song-grouped split: every chart of a song is
  in the same fold, so use it for cross-validation. `group` is the song key behind it.
* `labels.csv`: one row per (chart, task, source). `label` is 1 (has it) or 0 (does not).
  `detail` says why: vote counts, the tournament and slot, or the sheet text.
* `labels_wide.csv`: one column per task. `<task>` = the benchmark label (user tags + wiki pools,
  the labels the paper evaluates on). `staff:<category>` = the tournament-sheet label.
* `sources.csv`: the Google Sheets and wiki pages the staff labels come from.

## Tasks

* `slot`: main slot of the chart in osu! wiki tournament pools (majority over pools): RC (rice),
  LN, HB (hybrid), SV.
* Categories (Komi, 2026-10-30): Chordjack, Streams, Jumpstream/Handstream, Technical, Stamina
  (rice); Release, Coordination, Density/Inverse (LN); Reading, Memo (SV); TB (tiebreaker-style).
* `tag:*`: osu!(lazer) user tags. Positive at >= 2 votes; negative at 0 votes on a chart with
  >= 8 votes in total; otherwise unknown.

## Sources and how labels were made

* `user_tags`: lazer user-tag votes from the public beatmapset pages ({date}).
* `wiki_pool`: sub-slots of osu! wiki mappools (JHC chordjack slots, o!mLN Coordination / Density,
  SVSE Reading / Memorisation, every pool's Tiebreaker; a map in one sub-slot is a negative for the
  sibling sub-slots of that tournament).
* `wiki_slot`: the main slot of wiki pools.
* `staff_sheet`: annotations in tournament Google Sheets linked from the osu! wiki and the
  Tournaments forum ("Rice - Chordjacks", "LN Release", "SV Memory"), matched by keyword; a map
  annotated with one sub-type is a negative for the sibling sub-types annotated in the same tab.
  Quality check: a model that never saw them ranks them at AP 0.66-1.0 against base rates of
  0.13-0.67. TB annotations are not used (they usually mark a bracket tiebreaker).

Labels are noisy and incomplete by design: most cells are unknown. Tournament maps are mostly
unranked customs; ranked maps get most of the user-tag labels.

## Licence and credit

Chart metadata and user tags come from osu! (ppy). Tournament annotations belong to the
tournaments' staff; `sources.csv` lists them. Share with credit; no warranty.
"""


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data", type=Path)
    ap.add_argument("--out", default="dataset", type=Path)
    ap.add_argument("--keys", type=int, default=0, help="only charts with this key count (0: all)")
    a = ap.parse_args()
    d = a.data
    feats = pd.read_csv(d / "all_features.csv.gz", usecols=["beatmap_id", "keys"]).drop_duplicates("beatmap_id")
    keys = feats.set_index("beatmap_id")["keys"]
    folds = pd.read_csv(d / "v2_folds.csv", index_col=0)
    L = pd.read_csv(d / "v2_labels.csv.gz", index_col=0)
    staff = pd.read_csv(d / "staff_labels.csv.gz", index_col=0)
    ev = pd.read_csv(d / "staff_evidence.csv.gz")

    # metadata
    meta = {}
    for line in open(d / "ranked_mania.jsonl"):
        r = json.loads(line)
        for b in r["beatmaps"]:
            meta[b["id"]] = {"beatmapset_id": r["beatmapset_id"], "title": r["title"], "creator": r["creator"],
                             "difficulty": b["version"], "status": r["status"]}
    for line in gzip.open(d / "set_text.jsonl.gz", "rt", encoding="utf-8"):
        r = json.loads(line)
        for b, v in (r.get("diffs") or {}).items():
            meta.setdefault(int(b), {"beatmapset_id": r["beatmapset_id"], "title": r.get("title"),
                                     "creator": r.get("creator"), "difficulty": v, "status": r.get("status")})
    pools = pd.read_csv(d / "mappools.csv")
    for b, s, t in zip(pools.beatmap_id, pools.beatmapset_id, pools.title):
        if b not in meta:
            m = slots.song_key(t)
            meta[b] = {"beatmapset_id": s, "title": t.rsplit(" [", 1)[0], "creator": "",
                       "difficulty": t.rsplit(" [", 1)[-1].rstrip("]") if " [" in t else "", "status": ""}

    rows = []
    # user-tag votes behind tags and categories
    votes = {}
    for line in open(d / "user_tags.jsonl"):
        j = json.loads(line)
        for b, tv in j["beatmaps"].items():
            votes[int(b)] = tv
    tasks = [c for c in L.columns if c != "slot"]
    for b, tv in votes.items():
        total = sum(tv.values())
        for t in tasks:
            names = (t[4:],) if t.startswith("tag:") else categories.BY_NAME[t].user_tags
            v = max((tv.get(n, 0) for n in names), default=0)
            if v >= 2:
                rows.append((b, t, 1, "user_tags", f"{v} votes of {total}"))
            elif v == 0 and total >= 8:
                rows.append((b, t, 0, "user_tags", f"0 votes of {total}"))
    table = slots.label_table(pools)
    P = categories.labels_from_pools(pools, table.slot)
    for c in P.columns:
        for b, v in P[c].dropna().items():
            hits = pools[(pools.beatmap_id == b)]
            rows.append((b, c, int(v), "wiki_pool", "; ".join(sorted({f"{t} {s}" for t, s in zip(hits.tournament, hits.slot)}))[:300]))
    for b, s in table.slot.dropna().items():
        hits = pools[(pools.beatmap_id == b)]
        rows.append((b, "slot", s, "wiki_slot", "; ".join(sorted({f"{t} {x}" for t, x in zip(hits.tournament, hits.slot)}))[:300]))
    for b, c, v, s in zip(ev.beatmap_id, ev.category, ev.label, ev.sources):
        rows.append((b, c, int(v), "staff_sheet", s))
    lab = pd.DataFrame(rows, columns=["beatmap_id", "task", "label", "source", "detail"])

    ids = pd.Index(sorted(set(L.index[L.notna().any(axis=1)]) | set(staff.index)), name="beatmap_id")
    if a.keys:
        ids = ids[[keys.get(b) == a.keys for b in ids]]
    ids = ids[[b in keys.index for b in ids]]  # charts we could parse (have features)
    lab = lab[lab.beatmap_id.isin(ids)]
    # keep only label rows that agree with the benchmark/staff tables (e.g. SV-only filtering, corpus)
    wide = L.reindex(ids).join(staff.reindex(ids).add_prefix("staff:"))
    charts = pd.DataFrame({b: meta.get(b, {}) for b in ids}).T.reindex(ids)
    charts["keys"] = keys.reindex(ids).astype("Int64")
    charts = charts.join(folds.reindex(ids))
    a.out.mkdir(parents=True, exist_ok=True)
    charts.to_csv(a.out / "charts.csv")
    lab.sort_values(["beatmap_id", "task", "source"]).to_csv(a.out / "labels.csv", index=False)
    wide.to_csv(a.out / "labels_wide.csv", float_format="%.0f")
    fs = pd.read_csv(d / "forum_sheets.csv") if (d / "forum_sheets.csv").exists() else pd.DataFrame()
    used = {s.split(" ")[0].split(":", 1)[1] for s in ev.sources.dropna() for s in s.split(" | ") if s.startswith("sheet:")}
    src = [{"sheet_id": s, "url": f"https://docs.google.com/spreadsheets/d/{s}",
            "found_on": (f"https://osu.ppy.sh/community/forums/topics/{fs.set_index('sheet_id').topic_id.get(s)}"
                         if len(fs) and s in set(fs.sheet_id) else "osu! wiki")} for s in sorted(used)]
    pd.DataFrame(src).to_csv(a.out / "sources.csv", index=False)
    n = len(ids)
    (a.out / "README.md").write_text(README.format(n_charts=f"{n:,}", keys_note=f", {a.keys}K only" if a.keys else "",
                                                   date="2026-10"))
    print(f"{n} charts, {len(lab)} label rows, {len(src)} sheets -> {a.out}")
    print(lab.groupby(["source", "label"]).size().unstack(fill_value=0).to_string())


if __name__ == "__main__":
    main()
