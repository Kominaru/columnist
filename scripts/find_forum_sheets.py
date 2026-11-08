"""Google Sheets linked from mania topics in the osu! Tournaments forum (keyless).

The wiki only covers the larger tournaments; most others post their sheet in the forum.
Lists every page of forum 55, keeps topics whose title looks like mania, reads each topic's
first page and writes sheet_id,topic_id,title.

    python scripts/find_forum_sheets.py forum_sheets.csv --cache forum_cache
    python scripts/scrape_pool_sheets.py osu-wiki/wiki/Tournaments data/pool_sheets.jsonl.gz \
        --extra forum_sheets.csv
"""
import argparse
import csv
import html
import re
import subprocess
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

FORUM = "https://osu.ppy.sh/community/forums/55"
TOPIC = re.compile(r'href="https://osu\.ppy\.sh/community/forums/topics/(\d+)[^"]*"[^>]*>\s*([^<]*?)\s*<')
SHEET = re.compile(r"docs\.google\.com/spreadsheets/d/([A-Za-z0-9_-]{20,})")
MANIA = re.compile(r"mania|o!m|\bom\w*\b", re.I)
KEYS = re.compile(r"\b[4-9] ?k\b", re.I)
OTHER_MODE = re.compile(r"std|standard|bws|taiko|ctb|catch", re.I)


def get(url: str, path: Path) -> str:
    if not path.exists():
        for attempt in range(4):
            r = subprocess.run(["curl", "-sSLf", "-m", "60", "-o", str(path), url], capture_output=True)
            if r.returncode == 0:
                break
            time.sleep(2 ** attempt)
    return path.read_text(encoding="utf-8", errors="replace") if path.exists() else ""


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("out")
    ap.add_argument("--cache", type=Path, default=Path("forum_cache"))
    ap.add_argument("--workers", type=int, default=4)
    a = ap.parse_args()
    a.cache.mkdir(parents=True, exist_ok=True)
    first = get(FORUM, a.cache / "p1.html")
    last = max(int(x) for x in re.findall(r"page=(\d+)", first))
    with ThreadPoolExecutor(a.workers) as ex:
        pages = list(ex.map(lambda p: get(f"{FORUM}?page={p}", a.cache / f"p{p}.html"), range(1, last + 1)))
    titles = {}
    for t in pages:
        for tid, title in TOPIC.findall(t):
            if title.strip():
                titles.setdefault(tid, html.unescape(title))
    mania = {k: v for k, v in titles.items()
             if MANIA.search(v) or (KEYS.search(v) and not OTHER_MODE.search(v))}
    print(f"{len(titles)} topics, {len(mania)} mania")
    with ThreadPoolExecutor(a.workers) as ex:
        bodies = dict(zip(mania, ex.map(lambda k: get(f"https://osu.ppy.sh/community/forums/topics/{k}",
                                                      a.cache / f"t{k}.html"), mania)))
    seen = {}
    for tid, body in bodies.items():
        for sid in SHEET.findall(body):
            seen.setdefault(sid, (tid, mania[tid]))
    with open(a.out, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["sheet_id", "topic_id", "title"])
        for sid, (tid, title) in seen.items():
            w.writerow([sid, tid, title])
    print(f"{len(seen)} sheets -> {a.out}")


if __name__ == "__main__":
    main()
