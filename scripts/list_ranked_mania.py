"""List ranked beatmapsets that contain osu!mania difficulties, without an API key.

The public search endpoint works logged out but ignores filters and sorting (it always returns
all ranked sets, newest first), so this walks every page by cursor and keeps sets with mania
beatmaps. ~1,200 pages at ~1 req/s; resumable.

    python scripts/list_ranked_mania.py data/ranked_mania.jsonl
"""
import json
import sys
import time
import urllib.parse
import urllib.request
from pathlib import Path


def main(out: str, delay: float = 0.7):
    outp = Path(out)
    state = outp.with_suffix(".cursor")
    cursor = state.read_text().strip() if state.exists() else ""
    while True:
        url = "https://osu.ppy.sh/beatmapsets/search?" + urllib.parse.urlencode(
            {"s": "ranked", **({"cursor_string": cursor} if cursor else {})})
        for attempt in range(5):
            try:
                req = urllib.request.Request(url, headers={"User-Agent": "osu-patterns research"})
                j = json.load(urllib.request.urlopen(req, timeout=30))
                break
            except Exception as e:  # noqa: BLE001
                print("retry", e, flush=True)
                time.sleep(10 * 2 ** attempt)
        with outp.open("a") as f:
            for s in j["beatmapsets"]:
                maps = [b for b in s["beatmaps"] if b["mode"] == "mania"]
                if maps:
                    f.write(json.dumps({
                        "beatmapset_id": s["id"], "title": f"{s['artist']} - {s['title']}", "creator": s["creator"],
                        "play_count": s["play_count"], "favourite_count": s["favourite_count"],
                        "ranked_date": s["ranked_date"], "status": s["status"],
                        "beatmaps": [{"id": b["id"], "version": b["version"], "keys": b["cs"],
                                      "stars": b["difficulty_rating"], "playcount": b["playcount"]} for b in maps]}) + "\n")
        cursor = j.get("cursor_string")
        if not cursor:
            break
        state.write_text(cursor)
        time.sleep(delay)
    print("done")


if __name__ == "__main__":
    main(sys.argv[1])
