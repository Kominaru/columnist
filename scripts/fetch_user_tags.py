"""Fetch osu!(lazer) user-tag votes for every beatmapset in a mappool CSV, without an API key.

The public beatmapset page embeds the same JSON the API returns (status, beatmaps[].top_tag_ids,
related_tags), so this needs no OAuth client. One request per set, ~1/s, resumable.

    python scripts/fetch_user_tags.py data/mappools.csv data/user_tags.jsonl
    python scripts/fetch_user_tags.py data/ranked_mania.jsonl data/user_tags.jsonl 3000   # most-played first
"""
import html
import json
import re
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

import pandas as pd

JSON = re.compile(r'id="json-beatmapset"[^>]*>(.*?)</script>', re.S)


def fetch(set_id: int) -> dict:
    req = urllib.request.Request(f"https://osu.ppy.sh/beatmapsets/{set_id}", headers={"User-Agent": "osu-patterns research"})
    try:
        page = urllib.request.urlopen(req, timeout=30).read().decode("utf-8", "replace")
    except urllib.error.HTTPError as e:
        if e.code == 404:
            return {"beatmapset_id": set_id, "status": "missing", "beatmaps": {}}
        raise
    raw = JSON.search(page).group(1)
    try:
        j = json.loads(raw)
    except json.JSONDecodeError:  # older pages HTML-escape the payload
        j = json.loads(html.unescape(raw))
    names = {t["id"]: t["name"] for t in j.get("related_tags") or []}
    maps = {b["id"]: {names.get(t["tag_id"], str(t["tag_id"])): t["count"] for t in b.get("top_tag_ids") or []}
            for b in j["beatmaps"] if b["mode"] == "mania"}
    return {"beatmapset_id": set_id, "status": j["status"], "beatmaps": maps}


def main(source: str, out: str, limit: int = 0, delay: float = 0.5):
    if source.endswith(".jsonl"):  # from list_ranked_mania.py: most played sets carry the most votes
        rows = sorted((json.loads(l) for l in open(source)), key=lambda r: -r["play_count"])
        sets = [r["beatmapset_id"] for r in rows]
    else:
        sets = pd.read_csv(source).beatmapset_id.drop_duplicates().tolist()
    if limit:
        sets = sets[:int(limit)]
    outp = Path(out)
    done = {json.loads(l)["beatmapset_id"] for l in outp.open()} if outp.exists() else set()
    with outp.open("a") as f:
        for i, s in enumerate(x for x in sets if x not in done):
            for attempt in range(4):
                try:
                    f.write(json.dumps(fetch(int(s))) + "\n")
                    f.flush()
                    break
                except (json.JSONDecodeError, AttributeError) as e:  # unparseable page: skip it
                    print("skip", s, e, flush=True)
                    break
                except Exception as e:  # noqa: BLE001 - network hiccups / rate limits
                    print("retry", s, e, flush=True)
                    time.sleep(10 * 2 ** attempt)
            if i % 250 == 0:
                print(i, len(sets) - len(done), flush=True)
            time.sleep(delay)


if __name__ == "__main__":
    main(*sys.argv[1:4])
