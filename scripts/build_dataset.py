"""Build the training set: ranked osu!standard maps, their user-tag votes, and .osu files.

    OSU_CLIENT_ID=... OSU_CLIENT_SECRET=... python scripts/build_dataset.py --pages 60 --out data

Writes data/meta.jsonl (one line per difficulty) and data/osu/<beatmap_id>.osu.
Resumable: sets already in meta.jsonl are skipped. ~1 request/second.
"""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from osu_patterns.osu_api import OsuApi, beatmap_tags  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="data")
    ap.add_argument("--pages", type=int, default=20, help="search pages (~50 sets each)")
    ap.add_argument("--sort", default="plays_desc", help="plays_desc favours well-tagged maps")
    ap.add_argument("--min-votes-total", type=int, default=5, help="skip difficulties with fewer total tag votes")
    a = ap.parse_args()

    out = Path(a.out)
    (out / "osu").mkdir(parents=True, exist_ok=True)
    meta_p = out / "meta.jsonl"
    done = set()
    if meta_p.exists():
        done = {json.loads(l)["beatmapset_id"] for l in meta_p.read_text().splitlines() if l}

    api = OsuApi()
    names = api.tag_names()
    (out / "tags.json").write_text(json.dumps(names, indent=1))
    n_new = 0
    with meta_p.open("a") as f:
        for s in api.search_ranked(a.pages, sort=a.sort):
            if s["id"] in done:
                continue
            full = api.beatmapset(s["id"])
            for bid, tv in beatmap_tags(full, names).items():
                if sum(tv.values()) < a.min_votes_total:
                    continue
                try:
                    api.download_osu(bid, out / "osu" / f"{bid}.osu")
                except Exception as e:  # noqa: BLE001
                    print("download failed", bid, e)
                    continue
                b = next(x for x in full["beatmaps"] if x["id"] == bid)
                f.write(json.dumps({"beatmap_id": bid, "beatmapset_id": s["id"], "creator": full.get("creator"),
                                    "version": b.get("version"), "stars": b.get("difficulty_rating"),
                                    "playcount": b.get("playcount"), "tags": tv}) + "\n")
                n_new += 1
            done.add(s["id"])
            f.flush()
            print(f"\r{len(done)} sets, {n_new} new difficulties", end="")
    print()


if __name__ == "__main__":
    main()
