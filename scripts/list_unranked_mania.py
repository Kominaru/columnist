"""List unranked (graveyard / WIP / pending) 4K mania charts for self-supervised pretraining.

The logged-out osu! search ignores status filters, so this pages the osu.direct mirror search
(same JSON as the osu! API v2), most-played sets first, and keeps 4K mania difficulties that look
like real charts. Charts whose song is already in the corpus (data/v2_folds.csv groups) are left
out, so no fold's test song can reach pretraining through an unranked copy; packs (many songs in
one set) are left out for the same reason, since their song can't be read from the set title.

    python scripts/list_unranked_mania.py --max-charts 40000 --out data/unranked_4k.csv
    python scripts/fetch_osu.py data/unranked_4k.csv data/osu

Columns: beatmap_id, beatmapset_id, status, playcount, stars, length_s, group (song key).
"""
import argparse
import json
import re
import sys
import time
import urllib.request
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from osu_patterns import slots  # noqa: E402

API = "https://osu.direct/api/v2/search"
STATUS = {"graveyard": -2, "wip": -1, "pending": 0}
PACK = re.compile(r"\b(pack|compilation|various artists|marathon|collection)\b", re.I)


def page(status: int, offset: int, amount: int = 100) -> list:
    url = f"{API}?mode=3&status={status}&amount={amount}&offset={offset}&sort=play_count:desc"
    for attempt in range(5):
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "om-bn-pipeline research (unranked 4K list)"})
            return json.load(urllib.request.urlopen(req, timeout=60))
        except Exception as e:  # noqa: BLE001 - mirror hiccups: back off and retry
            print(f"  retry {attempt + 1} at offset {offset}: {e}", flush=True)
            time.sleep(3 * 2 ** attempt)
    raise RuntimeError(f"giving up at offset {offset}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data")
    ap.add_argument("--out", default="data/unranked_4k.csv")
    ap.add_argument("--status", nargs="+", default=["graveyard", "pending", "wip"])
    ap.add_argument("--max-charts", type=int, default=40000)
    ap.add_argument("--min-plays", type=int, default=30, help="per-difficulty play count")
    ap.add_argument("--min-length", type=int, default=30, help="seconds of drain time")
    ap.add_argument("--min-objects", type=int, default=150)
    ap.add_argument("--keys", type=int, default=4)
    a = ap.parse_args()

    folds = pd.read_csv(Path(a.data) / "v2_folds.csv", index_col=0)
    known_ids, known_groups = set(folds.index), set(folds.group.dropna())
    rows, seen_sets, skipped = [], set(), {"song in corpus": 0, "pack": 0}
    for name in a.status:
        offset = 0
        while len(rows) < a.max_charts:
            sets = page(STATUS[name], offset)
            if not sets:
                break
            offset += len(sets)
            for s in sets:
                if s["id"] in seen_sets:
                    continue
                seen_sets.add(s["id"])
                if PACK.search(f"{s['artist']} {s['title']}"):
                    skipped["pack"] += 1
                    continue
                group = slots.song_key(f"{s['artist']} - {s['title']} [x]")
                if group in known_groups:
                    skipped["song in corpus"] += 1
                    continue
                for b in s["beatmaps"]:
                    if (b["mode"] != "mania" or b.get("convert") or round(b["cs"]) != a.keys or b["id"] in known_ids
                            or b["playcount"] < a.min_plays or b["hit_length"] < a.min_length
                            or b["count_circles"] + b["count_sliders"] < a.min_objects):
                        continue
                    rows.append({"beatmap_id": b["id"], "beatmapset_id": s["id"], "status": name,
                                 "playcount": b["playcount"], "stars": b["difficulty_rating"],
                                 "length_s": b["hit_length"], "group": group})
            top = max(b["playcount"] for s in sets for b in s["beatmaps"])
            print(f"{name} offset {offset}: {len(rows)} charts (max diff plays on page {top})", flush=True)
            if top < a.min_plays:
                break
    df = pd.DataFrame(rows).drop_duplicates("beatmap_id").head(a.max_charts)
    df.to_csv(a.out, index=False)
    print(f"{len(df)} charts from {df.beatmapset_id.nunique()} sets, {df.group.nunique()} songs -> {a.out}; "
          f"skipped sets: {skipped}")
    print(df.status.value_counts().to_string())


if __name__ == "__main__":
    main()
