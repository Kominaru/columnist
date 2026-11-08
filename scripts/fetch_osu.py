"""Download .osu files (gzipped) for every beatmap_id in a CSV, in file order.

    python scripts/fetch_osu.py data/unranked_4k.csv data/osu [--mirror] [--threads 3]

--mirror also uses the osu.direct mirror (byte-identical files), alternating with osu.ppy.sh and
falling back to the other host on errors, which roughly doubles the rate for large lists.
"""
import argparse
import csv
import gzip
import itertools
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

HOSTS = ["https://osu.ppy.sh/osu/{}", "https://osu.direct/api/osu/{}"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("ids")
    ap.add_argument("out")
    ap.add_argument("--mirror", action="store_true")
    ap.add_argument("--threads", type=int, default=3)
    a = ap.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    ids = list(dict.fromkeys(int(r["beatmap_id"]) for r in csv.DictReader(open(a.ids, encoding="utf-8"))))
    hosts = HOSTS if a.mirror else HOSTS[:1]

    def get(arg):
        b, first = arg
        p = out / f"{b}.osu.gz"
        if p.exists():
            return 0
        for attempt in range(4):
            url = hosts[(first + attempt) % len(hosts)].format(b)
            try:
                req = urllib.request.Request(url, headers={"User-Agent": "om-bn-pipeline research"})
                data = urllib.request.urlopen(req, timeout=30).read()
                p.write_bytes(gzip.compress(data))
                return 1
            except Exception as e:  # noqa: BLE001
                if getattr(e, "code", 0) == 404 and attempt + 1 >= len(hosts):
                    p.write_bytes(gzip.compress(b""))  # deleted map: remember it
                    return 0
                time.sleep(2 ** attempt * 3)
        print("fail", b, flush=True)
        return 0

    threads = a.threads * len(hosts)
    with ThreadPoolExecutor(threads) as ex:
        for i, _ in enumerate(ex.map(get, zip(ids, itertools.cycle(range(len(hosts)))))):
            if i % 500 == 0:
                print(i, len(ids), flush=True)


if __name__ == "__main__":
    main()
