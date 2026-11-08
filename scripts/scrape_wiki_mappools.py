"""Collect osu!mania tournament mappool slots from a local clone of ppy/osu-wiki.

    git clone --depth 1 --filter=blob:none --sparse https://github.com/ppy/osu-wiki
    git -C osu-wiki sparse-checkout set wiki/Tournaments
    python scripts/scrape_wiki_mappools.py osu-wiki/wiki/Tournaments data/mappools.csv

Pool pages list maps as "- <Slot>" followed by numbered beatmap links (".../beatmapsets/S#mania/B").
"""
import csv
import re
import sys
from pathlib import Path

LINK = re.compile(r"osu\.ppy\.sh/beatmapsets/(\d+)#mania/(\d+)")
SLOT = re.compile(r"^-\s+(?:\*\*)?([^\[\]()*]+?)(?:\*\*)?\s*$")
ITEM = re.compile(r"^\s+\d+\.\s")
TITLE = re.compile(r"\[(.+?)\]\(https://osu\.ppy\.sh/beatmapsets")


def scrape(root: Path):
    for page in sorted(root.rglob("en.md")):
        tourney = str(page.parent.relative_to(root))
        stage, slot = "", None
        for line in page.read_text(encoding="utf-8").splitlines():
            if line.startswith("#"):
                stage, slot = line.lstrip("#").strip(), None
                continue
            m = SLOT.match(line)
            if m and not LINK.search(line):
                slot = m.group(1).strip()
                continue
            if slot and ITEM.match(line):
                lk = LINK.search(line)
                if lk:
                    t = TITLE.search(line)
                    yield {"tournament": tourney, "stage": stage, "slot": slot,
                           "beatmapset_id": int(lk.group(1)), "beatmap_id": int(lk.group(2)),
                           "title": t.group(1).replace("\\", "") if t else ""}
            elif line.strip() and not ITEM.match(line) and not line.startswith(" "):
                slot = None


if __name__ == "__main__":
    rows = list(scrape(Path(sys.argv[1])))
    out = Path(sys.argv[2])
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    print(f"{len(rows)} slot entries from {len({r['tournament'] for r in rows})} tournaments -> {out}")
