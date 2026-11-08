"""Category labels from what tournament staff and mappers wrote about a map.

Three text sources, each turned into {1, 0, NaN} per (map, category) with the same keyword table:

* sheet: the "Type" / "Notes" / "Category" / "Map type" cells of tournament Google Sheets
  (scripts/scrape_pool_sheets.py -> data/pool_sheets.jsonl.gz), e.g. "Rice - Chordjacks",
  "LN Release", "SV Memory", "Tech/Other Rice".
* wiki: sub-slot names in the osu! wiki pools (data/mappools.csv) beyond the ones categories.py
  already uses ("Technical", "Stream", "Speed", "Full Jack", "Coordination", ...).
* settext: mapper tags, description and difficulty name of the beatmapset
  (scripts/fetch_set_text.py -> data/set_text.jsonl). Mapper tags are free text and noisy, so
  they only ever give positives, and only for single-chart sets or the diff whose name matches.

Negatives follow categories.labels_from_pools: within one annotated tab (or wiki tournament),
a map annotated with one sub-type of a parent is a negative for the sibling sub-types it was
not given, because poolers annotate the dominant skill. Streams and Jumpstream/Handstream never
negate each other (jumpstream maps nearly always stream too), and Stamina is never negated
(any long map is also stamina).
"""
from __future__ import annotations

import gzip
import json
import re
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

# category -> regex on a short annotation; a text can hit several categories ("Mixed/Stamina")
STRICT = {
    # Komi (2026-11-02): speedjacks and anchors are Chordjack, minijacks are Technical
    "Chordjack": r"chord ?jacks?|\bcj\b|\b(speed ?|anchor ?|full ?|dense ?|heavy ?)?jacks?\b|jackhammer|\banchors?\b",
    "Streams": r"\bstreams?\b|\bstreamy\b",
    "Jumpstream/Handstream": r"jump ?streams?|hand ?streams?|chord ?streams?|quad ?streams?|\bjs\b|\bhs\b|jumptrill",
    "Technical": r"\btech(nical|y)?\b|\bmini ?jacks?\b",
    "Stamina": r"stamina|\bstam\b|endurance",
    "Release": r"releases?\b",
    "Coordination": r"\bcoord(ination|\.)?",
    "Density/Inverse": r"\bln ?dens(e|ity)|\bdens(e|ity) ?ln|\binverse\b|\bfull ?ln\b",
    "Reading": r"\bsv ?reading\b|^reading$|\breading\b(?=.*sv)",
    "Memo": r"\bmemo(ry|ri[sz]ation|ri[sz]e)?\b",
    "TB": r"tie ?breaker|^tb\d?$|^tb\d?:",
}
# looser readings, kept apart so their agreement can be checked on their own
LOOSE = {
    "Streams": r"^speed$|\bspeed ?stream|\brc speed\b|rice - speed",
    "Density/Inverse": r"^density$|\bln ?speed\b",
}
# streams/JS words inside these are about something else ("Dense Chordstream" is JS, not Streams)
NOT_STREAMS = re.compile(r"jump|hand|chord|quad", re.I)
FAMILY = {"Chordjack": "RC", "Streams": "RC", "Jumpstream/Handstream": "RC", "Technical": "RC",
          "Stamina": "RC", "Release": "LN", "Coordination": "LN", "Density/Inverse": "LN",
          "Reading": "SV", "Memo": "SV", "TB": "TB"}
NEVER_NEGATE = {("Streams", "Jumpstream/Handstream"), ("Jumpstream/Handstream", "Streams")}
# "Technical Hybrid", "LN Technical", "Jack-heavy HB": a hybrid or LN slot described by a rice skill.
# They are still positives (the skill is there), but they don't make negatives.
OFF_FAMILY = {"RC": re.compile(r"\bln\b|hybrid|\bhb\b|\bsv\b", re.I),
              "LN": re.compile(r"hybrid|\bhb\b|\bsv\b|rice|\brc\b", re.I),
              "SV": re.compile(r"$^")}
META_HEADER = re.compile(r"title|artist|diff|mapper|creator|charter|map\b|name|link|id$|tags|metadata|"
                         r"source|song|version|url|md5|suggest|player|team|bpm|length|sr\b|star", re.I)
TITLEISH = re.compile(r" - .+|\[.*\]|https?://|\(.*\)$")

_STRICT = {k: re.compile(v, re.I) for k, v in STRICT.items()}
_LOOSE = {k: re.compile(v, re.I) for k, v in LOOSE.items()}


def classify(text: str, loose: bool = False) -> set[str]:
    """Categories a short annotation names."""
    text = text.strip()
    if not text or len(text) > 40 or TITLEISH.search(text):
        return set()
    out = {k for k, r in _STRICT.items() if r.search(text)}
    if loose:
        out |= {k for k, r in _LOOSE.items() if r.search(text)}
    if re.search(r"\bmini ?jacks?\b", text, re.I) and not re.search(r"chord ?jack|speed ?jack|anchor", text, re.I):
        out.discard("Chordjack")
    if "Streams" in out and NOT_STREAMS.search(text) and not re.search(r"\bstreams?\b(?!.*(jump|hand|chord|quad))", text, re.I):
        out.discard("Streams")
    return out


def _labels(groups: dict, loose: bool) -> tuple[pd.DataFrame, dict]:
    """groups: (group key) -> {beatmap_id: [annotation texts]}. Returns labels and the texts behind them."""
    pos, neg, why = defaultdict(set), defaultdict(set), defaultdict(list)
    for maps in groups.values():
        cats = {b: set().union(*[classify(t, loose) for t in ts]) if ts else set() for b, ts in maps.items()}
        on_family = {b: {c for c in cs if not any(OFF_FAMILY.get(FAMILY[c], re.compile("$^")).search(t)
                                                  for t in maps[b] if c in classify(t, loose))}
                     for b, cs in cats.items()}
        for b, cs in cats.items():
            pos[b] |= cs
            for t in maps[b]:
                if classify(t, loose):
                    why[b].append(t)
        # sibling negatives: only in groups that annotate two or more sub-types of a family
        for fam in ("RC", "LN", "SV"):
            used = {c for cs in on_family.values() for c in cs if FAMILY[c] == fam}
            if len(used) < 2:
                continue
            for b, cs in on_family.items():
                mine = {c for c in cs if FAMILY[c] == fam}
                if not mine:
                    continue
                for c in used - mine:
                    if c != "Stamina" and not any((m, c) in NEVER_NEGATE for m in mine):
                        neg[b].add(c)
    ids = sorted(set(pos) | set(neg))
    names = list(FAMILY)
    out = pd.DataFrame(np.nan, index=pd.Index(ids, name="beatmap_id"), columns=names)
    for b in ids:
        for c in neg[b]:
            out.at[b, c] = 0.0
        for c in pos[b]:
            out.at[b, c] = 1.0
    return out[out.notna().any(axis=1)], why


def _lines(p):
    return gzip.open(p, "rt", encoding="utf-8") if str(p).endswith(".gz") else open(p, encoding="utf-8")


def set_diffs(*paths) -> dict[int, dict[int, str]]:
    """beatmapset_id -> {beatmap_id: difficulty name} for mania charts, from set_text JSONL files."""
    out = {}
    for p in paths:
        if Path(p).exists():
            for line in _lines(p):
                r = json.loads(line)
                if r.get("diffs"):
                    out[r["beatmapset_id"]] = {int(b): v for b, v in r["diffs"].items()}
    return out


def resolve_set(diffs: dict[int, str], cells: list[str]) -> list[int]:
    """The chart a set-only link means: the set's only mania chart, or the one whose name is in the row."""
    if len(diffs) == 1:
        return list(diffs)
    norm = lambda s: re.sub(r"\W+", " ", s.lower()).strip()
    texts = [norm(t) for t in cells]
    hit = [b for b, v in diffs.items() if norm(v) and any(norm(v) == t or f"[{v.lower()}]" in c.lower()
                                                           for t, c in zip(texts, cells))]
    return hit if len(hit) == 1 else []


def sheet_groups(rows: list[dict], diffs: dict | None = None) -> dict:
    """One group per sheet tab: beatmap_id -> annotation texts (metadata columns dropped).
    Rows that only link a beatmapset are resolved through `diffs` (set_diffs)."""
    groups = defaultdict(lambda: defaultdict(list))
    for r in rows:
        key = (r["sheet"], r["tab"])
        texts = [t for _, h, t in r["cells"] if not META_HEADER.search(h or "")]
        ids = list(r["ids"])
        for s in r.get("sets", []) if diffs else []:
            ids += resolve_set(diffs.get(s, {}), [t for _, _, t in r["cells"]])
        for b in ids:
            groups[key][int(b)].extend(texts)
    return groups


def wiki_groups(pools: pd.DataFrame) -> dict:
    """One group per wiki tournament page: beatmap_id -> slot names."""
    groups = defaultdict(lambda: defaultdict(list))
    for t, b, s in zip(pools.tournament, pools.beatmap_id, pools.slot):
        groups[t][int(b)].append(str(s))
    return groups


def labels_from_sheets(path: str | Path, loose: bool = False, diffs: dict | None = None):
    rows = [json.loads(line) for line in _lines(path)]
    return _labels(sheet_groups(rows, diffs), loose)


def labels_from_wiki(pools: pd.DataFrame, loose: bool = False):
    return _labels(wiki_groups(pools), loose)


SETTEXT_WORDS = {  # mapper tags / descriptions: whole words only, positives only
    "Chordjack": r"\bchord ?jacks?\b|\bchordjacky\b|\bspeed ?jacks?\b|\banchor ?jacks?\b",
    "Jumpstream/Handstream": r"\bjump ?streams?\b|\bhand ?streams?\b|\bjs\b|\bhs\b",
    "Technical": r"\btechnical\b|\btech\b",
    "Stamina": r"\bstamina\b",
    # bare "release" is usually about the song ("one day after its release")
    "Release": r"\bln ?releases?\b|\breleases\b|\brelease ?(heavy|focused|patterns?|map|streams?)\b",
    "Coordination": r"\bln ?coord(ination)?\b",
    "Density/Inverse": r"\bln ?dens(e|ity)\b|\binverse\b",
    "Reading": r"\bsv ?reading\b",
    # "Memory"/"Memories" are common song and diff names, so only explicit SV memorisation counts
    "Memo": r"\bmemori[sz]ation\b|\bsv ?memo(ry)?\b|\bmemo ?sv\b",
}
_SETTEXT = {k: re.compile(v, re.I) for k, v in SETTEXT_WORDS.items()}
# in prose, "js" / "hs" are too often something else; only the full words count there
_PROSE = {**_SETTEXT, "Jumpstream/Handstream": re.compile(r"\bjump ?streams?\b|\bhand ?streams?\b", re.I)}


def labels_from_set_text(path: str | Path, fields=("tags", "diff")) -> tuple[pd.DataFrame, dict]:
    """Positives from mapper tags / difficulty names (and the description if listed in fields).

    Set-level text applies to a chart only when the set has one mania chart, since a set's tags
    describe the set; a difficulty name applies to its own chart."""
    pos, why = defaultdict(set), defaultdict(list)
    for line in _lines(path):
        r = json.loads(line)
        diffs = {int(b): v for b, v in (r.get("diffs") or {}).items()}
        set_text = " ".join(r.get(f, "") or "" for f in fields if f in ("tags", "description", "source"))
        rx_set = _PROSE if "description" in fields else _SETTEXT
        for b, v in diffs.items():
            texts = [v] if "diff" in fields else []
            if len(diffs) == 1 and set_text:
                texts.append(set_text)
            for t in texts:
                for c, rx in (rx_set if t is set_text else _SETTEXT).items():
                    m = rx.search(t)
                    if m:
                        pos[b].add(c)
                        why[b].append(m.group(0))
    out = pd.DataFrame(np.nan, index=pd.Index(sorted(pos), name="beatmap_id"), columns=list(FAMILY))
    for b, cs in pos.items():
        for c in cs:
            out.at[b, c] = 1.0
    return out, why
