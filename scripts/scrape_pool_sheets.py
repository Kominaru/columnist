"""Mine per-map annotations from the Google Sheets that mania tournaments link on the osu! wiki.

Many tournament sheets carry more than the slot: a "Type" / "Notes" / "Category" / "Map type"
column ("Rice - Chordjacks", "LN Release", "SV Memory", ...). This collects every text cell on
every row that references a beatmap, keyed by its column header, so that
osu_patterns.pool_annotations can turn them into category labels.

    git clone --depth 1 --filter=blob:none --sparse https://github.com/ppy/osu-wiki
    git -C osu-wiki sparse-checkout set wiki/Tournaments
    python scripts/scrape_pool_sheets.py osu-wiki/wiki/Tournaments data/pool_sheets.jsonl.gz \
        --cache sheets_xlsx --known data/v2_folds.csv data/mappools.csv

Only public sheets can be exported (about 150 of the ~180 linked from mania pages). Score,
lobby, schedule and team tabs are skipped by name, and tabs that list the same map many times
(score logs) by shape. A bare number counts as a beatmap id only when it is a known mania id.
"""
import argparse
import csv
import gzip
import json
import re
import subprocess
import warnings
from concurrent.futures import ProcessPoolExecutor
from functools import partial
from pathlib import Path

warnings.filterwarnings("ignore")
SHEET = re.compile(r"\[([^\]]*)\]\((https?://docs\.google\.com/spreadsheets/d/([A-Za-z0-9_-]+))[^)]*\)")
BEATMAP = re.compile(r"osu\.ppy\.sh/(?:beatmapsets/\d+#\w+/|b/|beatmaps/|p/beatmap\?b=)(\d+)")
NOISE_TAB = re.compile(r"raw|score|match|eval|hiscore|result|lobby|testing|log|player|team|seed|stat|"
                       r"vs|bracket|schedule|pick|ban|^_", re.I)
SETLINK = re.compile(r"osu\.ppy\.sh/(?:beatmapsets|s)/(\d+)(?![\d#])")
NUMBERISH = re.compile(r"[\d.:%/★ -]+")


def find_sheets(wiki: Path):
    """(tournament, sheet id) for every sheet linked from a page that lists mania maps."""
    out = {}
    for page in sorted(wiki.rglob("en.md")):
        text = page.read_text(encoding="utf-8")
        if "#mania/" not in text and "mania" not in str(page).lower():
            continue
        for m in SHEET.finditer(text):
            out.setdefault(m.group(3), str(page.parent.relative_to(wiki)))
    return out


def download(sheet_id: str, cache: Path) -> Path | None:
    p = cache / f"{sheet_id}.xlsx"
    if not p.exists():
        url = f"https://docs.google.com/spreadsheets/d/{sheet_id}/export?format=xlsx"
        r = subprocess.run(["curl", "-sSLf", "-m", "180", "-o", str(p), url], capture_output=True)
        if r.returncode:
            p.unlink(missing_ok=True)
            return None
    return p if p.stat().st_size and p.read_bytes()[:2] == b"PK" else None


def _text(v):
    if v is None:
        return None
    v = str(v).strip()
    if not v or v.startswith("=") or NUMBERISH.fullmatch(v) or v in ("True", "False") or v.startswith("#"):
        return None
    return v


def extract(path: Path, known: set[str]):
    import openpyxl
    try:
        wb, wbv = openpyxl.load_workbook(path), openpyxl.load_workbook(path, data_only=True)
    except Exception:
        return []
    out = []
    for ws in wb.worksheets:
        if NOISE_TAB.search(ws.title) or ws.max_row > 2000:
            continue
        wv = wbv[ws.title]
        grid = {}
        for row in ws.iter_rows():
            for c in row:
                v = wv.cell(row=c.row, column=c.column).value
                ids = set()
                for s in (c.value, v, c.hyperlink.target if c.hyperlink else None):
                    if s is None:
                        continue
                    s = str(s).strip()
                    ids.update(BEATMAP.findall(s))
                    ids.update("s" + x for x in SETLINK.findall(s))  # set link: resolved later
                    if re.fullmatch(r"\d{5,8}(\.0)?", s) and s.split(".")[0] in known:
                        ids.add(s.split(".")[0])
                grid[(c.row, c.column)] = (ids, _text(v))
        idrows = sorted({r for (r, _), (ids, _) in grid.items() if ids})
        allids = [i for ids, _ in grid.values() for i in ids]
        if not idrows or len(idrows) > 300 or len(allids) > 3 * len(set(allids)):
            continue
        idset = set(idrows)

        def header(col, row):  # nearest short text above, skipping other map rows
            for r in range(row - 1, max(0, row - 40), -1):
                t = grid.get((r, col), (None, None))[1]
                if r not in idset and t and len(t) < 40:
                    return t
            return ""

        for r in idrows:
            ids, cells = set(), []
            for (rr, cc), (i, t) in grid.items():
                if rr == r:
                    ids |= i
                    if t:
                        cells.append((cc, header(cc, r), t))
            sets = sorted(int(i[1:]) for i in ids if i.startswith("s"))
            ids = sorted(int(i) for i in ids if not i.startswith("s"))
            out.append({"sheet": path.stem, "tab": ws.title, "row": r, "ids": ids, "sets": [] if ids else sets,
                        "cells": sorted(cells)})
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("wiki", type=Path)
    ap.add_argument("out", type=Path)
    ap.add_argument("--cache", type=Path, default=Path("sheets_xlsx"))
    ap.add_argument("--known", nargs="*", default=["data/v2_folds.csv", "data/mappools.csv"])
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--extra", nargs="*", default=[], help="CSVs with sheet_id,topic_id,title (find_forum_sheets.py)")
    a = ap.parse_args()
    known = set()
    for k in a.known:
        with open(k) as f:
            known |= {row["beatmap_id"] for row in csv.DictReader(f)}
    sheets = find_sheets(a.wiki)
    n_wiki = len(sheets)
    for e in a.extra:
        with open(e) as f:
            for row in csv.DictReader(f):
                sheets.setdefault(row["sheet_id"], f"forum/{row['topic_id']}")
    a.cache.mkdir(parents=True, exist_ok=True)
    paths = {}
    for sid in sheets:
        p = download(sid, a.cache)
        if p:
            paths[p] = sheets[sid]
    print(f"{len(sheets)} sheets ({n_wiki} from the wiki, the rest from --extra), {len(paths)} public")
    n = 0
    with (gzip.open(a.out, "wt", encoding="utf-8") if a.out.suffix == ".gz" else a.out.open("w")) as f, ProcessPoolExecutor(a.workers) as ex:
        for p, rows in zip(paths, ex.map(partial(extract, known=known), paths)):
            for r in rows:
                f.write(json.dumps({"tournament": paths[p], **r}, ensure_ascii=False) + "\n")
                n += 1
    print(f"{n} map rows -> {a.out}")


if __name__ == "__main__":
    main()
