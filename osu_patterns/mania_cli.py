"""Slot + pattern tags for osu!mania maps, for a Trello BN-request card.

    python -m osu_patterns.mania_cli map.osu [folder ...]
    python -m osu_patterns.mania_cli --beatmapset 1234567 --format trello     # no API key needed
    python -m osu_patterns.mania_cli https://osu.ppy.sh/beatmapsets/1234567 --format trello

Models default to models/slots.pkl, models/mania_tags.pkl and models/categories.pkl when present; without them the
rule-based slotting and heuristic tags are used.
"""
from __future__ import annotations

import argparse
import html
import json
import re
import urllib.request
from pathlib import Path

from . import mania, mania_tags
from .cli import beatmapset_id_from_text

UA = {"User-Agent": "osu-patterns"}
_JSON = re.compile(r'id="json-beatmapset"[^>]*>(.*?)</script>', re.S)


def public_beatmapset(set_id: int) -> dict:
    """The beatmapset JSON embedded in the public page (works logged out, pending maps included)."""
    page = urllib.request.urlopen(urllib.request.Request(f"https://osu.ppy.sh/beatmapsets/{set_id}", headers=UA),
                                  timeout=30).read().decode("utf-8", "replace")
    raw = _JSON.search(page).group(1)
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        return json.loads(html.unescape(raw))


def public_osu(beatmap_id: int) -> str:
    return urllib.request.urlopen(urllib.request.Request(f"https://osu.ppy.sh/osu/{beatmap_id}", headers=UA),
                                  timeout=30).read().decode("utf-8-sig", "replace")


def _ts(ms: int) -> str:
    return f"{ms // 60000:02d}:{ms // 1000 % 60:02d}:{ms % 1000:03d}"


SLOT_NAMES = {"RC": "Rice", "LN": "LN", "HB": "Hybrid", "SV": "SV"}


def to_trello(results: list[dict]) -> str:
    lines = ["**Mania slot & patterns (auto)**"]
    for r in results:
        s = r["stats"]
        conf = f" {r['slot_probs'][r['slot']]:.0%}" if r.get("slot_probs") else ""
        cats = ", ".join(r["categories"]) or "no dominant pattern"
        tb = " | **TB-worthy**" if r["tb"] else ""
        lines.append(f"\n*{r['version']}* ({r['keys']}K, {r['bpm']:.0f} BPM, {s['nps']:.1f} nps, "
                     f"LN {s['ln_ratio']:.0%}): **{r.get('slot_label') or SLOT_NAMES[r['slot']]}**{conf} | {cats}{tb}")
        tags = ", ".join(t.split("/", 1)[1] for t in r["tags"][:4])
        if tags:
            lines.append(f"- tags: {tags}")
        for kind, ts in r["evidence"].items():
            if kind == "streams" and r["slot"] != "RC":
                continue
            lines.append(f"- {kind}: " + ", ".join(f"`{_ts(t)}`" for t in ts[:2]))
    lines.append("\n_auto-tagged from the .osu file; slot from tournament pools, patterns from osu! user tags_")
    return "\n".join(lines)


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("paths", nargs="*", help=".osu files, folders, or beatmapset links")
    ap.add_argument("--beatmapset", type=int)
    ap.add_argument("--slot-model", default="models/slots.pkl")
    ap.add_argument("--tag-model", default="models/mania_tags.pkl")
    ap.add_argument("--category-model", default="models/categories.pkl")
    ap.add_argument("--format", choices=["text", "json", "trello"], default="text")
    ap.add_argument("--all-diffs", action="store_true", help="otherwise only the 2 hardest difficulties")
    a = ap.parse_args(argv)

    texts: list[str] = []
    set_ids = [a.beatmapset] if a.beatmapset else []
    for p in a.paths:
        sid = beatmapset_id_from_text(p)
        if sid:
            set_ids.append(sid)
            continue
        path = Path(p)
        for f in (sorted(path.glob("*.osu")) if path.is_dir() else [path]):
            texts.append(mania.read_text(f))
    for sid in set_ids:
        for b in public_beatmapset(sid)["beatmaps"]:
            if b["mode"] == "mania":
                texts.append(public_osu(b["id"]))

    from .slots import SlotModel
    from .model import TagModel
    slot_model = SlotModel.load(a.slot_model) if Path(a.slot_model).exists() else None
    tag_model = TagModel.load(a.tag_model) if Path(a.tag_model).exists() else None
    category_model = TagModel.load(a.category_model) if Path(a.category_model).exists() else None

    maps = [m for m in map(mania.parse_mania, texts) if m is not None and len(m.start) >= 20]
    maps.sort(key=lambda m: -len(m.start) / max(1.0, (m.end.max() - m.start.min()) / 1000))  # densest first
    if not a.all_diffs:
        maps = maps[:2]
    results = [mania_tags.describe(m, slot_model, tag_model, category_model) for m in maps]

    if a.format == "json":
        print(json.dumps([{k: v for k, v in r.items() if k != "stats"} for r in results], indent=2, default=float))
    elif a.format == "trello":
        print(to_trello(results))
    else:
        for r in results:
            print(f"[{r['version']}] {r['keys']}K  slot={r['slot']}  {r.get('slot_probs', '')}")
            print(f"  categories: {', '.join(r['categories']) or '-'}" + ("  (TB-worthy)" if r["tb"] else ""))
            for t in r["tags"]:
                print(f"  {r['tag_scores'][t]:.2f}  {t}")
            for k, ts in r["evidence"].items():
                print(f"  {k}: " + ", ".join(_ts(t) for t in ts))


if __name__ == "__main__":
    main()
