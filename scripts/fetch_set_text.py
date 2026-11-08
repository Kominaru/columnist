"""Fetch the free text of beatmapsets: mapper tags, description, source and diff names.

Same keyless route as fetch_user_tags.py (the public beatmapset page embeds the API JSON).
Input: a text file with one beatmapset id per line. Output: JSONL, resumable.

    python scripts/fetch_set_text.py sets.txt set_text.jsonl --workers 4 && gzip set_text.jsonl
"""
import argparse
import html
import json
import re
import time
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

JSON = re.compile(r'id="json-beatmapset"[^>]*>(.*?)</script>', re.S)
TAG = re.compile(r"<[^>]+>")


def strip_html(s: str) -> str:
    s = re.sub(r"<br\s*/?>|</div>|</p>", "\n", s or "")
    return re.sub(r"\n{2,}", "\n", html.unescape(TAG.sub("", s))).strip()


def fetch(set_id: int) -> dict:
    req = urllib.request.Request(f"https://osu.ppy.sh/beatmapsets/{set_id}",
                                 headers={"User-Agent": "osu-patterns research"})
    try:
        page = urllib.request.urlopen(req, timeout=30).read().decode("utf-8", "replace")
    except urllib.error.HTTPError as e:
        if e.code == 404:
            return {"beatmapset_id": set_id, "status": "missing"}
        raise
    raw = JSON.search(page).group(1)
    try:
        j = json.loads(raw)
    except json.JSONDecodeError:
        j = json.loads(html.unescape(raw))
    desc = j.get("description") or {}
    return {"beatmapset_id": set_id, "status": j.get("status"), "tags": j.get("tags") or "",
            "source": j.get("source") or "", "title": f"{j.get('artist')} - {j.get('title')}",
            "creator": j.get("creator"),
            "description": strip_html(desc.get("description") if isinstance(desc, dict) else desc),
            "diffs": {b["id"]: b["version"] for b in j.get("beatmaps", []) if b.get("mode") == "mania"}}


def safe(set_id: int):
    for attempt in range(5):
        try:
            return fetch(set_id)
        except (urllib.error.URLError, TimeoutError, ConnectionError) as e:
            time.sleep(2 ** attempt * 2)
        except (json.JSONDecodeError, AttributeError) as e:
            return {"beatmapset_id": set_id, "status": "unparseable"}
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("sets"); ap.add_argument("out"); ap.add_argument("--workers", type=int, default=4)
    a = ap.parse_args()
    sets = [int(x) for x in open(a.sets).read().split()]
    out = Path(a.out)
    done = {json.loads(l)["beatmapset_id"] for l in out.open()} if out.exists() else set()
    todo = [s for s in dict.fromkeys(sets) if s not in done]
    print(f"{len(todo)} to fetch ({len(done)} done)", flush=True)
    with out.open("a") as f, ThreadPoolExecutor(a.workers) as ex:
        for i, r in enumerate(ex.map(safe, todo)):
            if r:
                f.write(json.dumps(r, ensure_ascii=False) + "\n"); f.flush()
            if i % 500 == 0:
                print(i, flush=True)


if __name__ == "__main__":
    main()
