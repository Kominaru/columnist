"""Tag osu! maps from the command line.

    python -m osu_patterns.cli map.osu [more.osu | folder ...]
    python -m osu_patterns.cli --beatmapset 1234567           # downloads via the osu! API
    python -m osu_patterns.cli map.osu --model models/tagger.pkl --format trello
"""
from __future__ import annotations

import argparse
import json
import re
import tempfile
from pathlib import Path

from .parser import Beatmap, parse
from .tagger import TagResult, tag

SET_URL = re.compile(r"osu\.ppy\.sh/(?:beatmapsets|s)/(\d+)")


def beatmapset_id_from_text(text: str) -> int | None:
    """Find a beatmapset id in a Trello card name/description/link."""
    m = SET_URL.search(text or "")
    return int(m.group(1)) if m else None


def describe(bm: Beatmap, model=None, threshold: float = 0.5, max_evidence: int = 3) -> dict:
    res: TagResult = tag(bm)
    if model is not None:
        probs = model.predict(bm)
        chosen = model.decide(probs)
    else:
        probs = res.scores
        chosen = res.tags(threshold)
    ev = {}
    tr_combo = None
    if res.evidence:
        from .features import compute_transitions
        tr_combo = compute_transitions(bm).combo
    for t in chosen:
        segs = sorted(res.evidence.get(t, []), key=lambda s: -s.n)[:max_evidence]
        ev[t] = [s.timestamp(tr_combo) + f" - {s.n} notes" for s in sorted(segs, key=lambda s: s.t0)]
    return {
        "map": f"{bm.artist} - {bm.title} [{bm.version}]" if bm.version else Path(bm.path).stem,
        "version": bm.version or Path(bm.path).stem,
        "beatmap_id": bm.beatmap_id,
        "bpm": bm.main_bpm, "cs": bm.cs, "ar": bm.ar,
        "tags": chosen,
        "scores": {t: round(probs[t], 2) for t in chosen},
        "evidence": ev,
        "source": "model" if model is not None else "heuristic",
    }


def to_trello(results: list[dict]) -> str:
    """Markdown comment for a Trello card, hardest difficulty first."""
    lines = ["**Pattern tags (auto)**"]
    for r in results:
        head = [t.split("/", 1)[1] for t in r["tags"] if t.startswith("skillset/")]
        sub = [t for t in r["tags"] if not t.startswith("skillset/")]
        lines.append(f"\n*{r['version']}* ({r['bpm']:g} BPM, CS {r['cs']:g}, AR {r['ar']:g}): "
                     + (", ".join(head) if head else "no dominant skillset")
                     + (f" | {', '.join(sub)}" if sub else ""))
        for t, ts in r["evidence"].items():
            for x in ts[:2]:
                lines.append(f"- {t}: `{x}`")
    lines.append(f"\n_source: {results[0]['source'] if results else 'n/a'} tagger, scores are estimates_")
    return "\n".join(lines)


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("paths", nargs="*")
    ap.add_argument("--beatmapset", type=int)
    ap.add_argument("--model")
    ap.add_argument("--threshold", type=float, default=0.5)
    ap.add_argument("--format", choices=["text", "json", "trello"], default="text")
    ap.add_argument("--all-diffs", action="store_true", help="otherwise only the hardest 2 difficulties")
    a = ap.parse_args(argv)

    files: list[Path] = []
    for p in map(Path, a.paths):
        files += sorted(p.glob("*.osu")) if p.is_dir() else [p]
    if a.beatmapset:
        from .osu_api import OsuApi
        api = OsuApi()
        tmp = Path(tempfile.mkdtemp())
        for b in api.beatmapset(a.beatmapset)["beatmaps"]:
            if b.get("mode") == "osu":
                files.append(api.download_osu(b["id"], tmp / f"{b['id']}.osu"))

    model = None
    if a.model:
        from .model import TagModel
        model = TagModel.load(a.model)

    maps = [bm for bm in map(parse, files) if bm.mode == 0 and bm.objects]
    # hardest first, by a cheap proxy (object density * spacing); good enough to pick top diffs
    maps.sort(key=lambda bm: -len(bm.objects) / max(1, (bm.objects[-1].time - bm.objects[0].time)) * (10 - bm.cs + bm.ar))
    if not a.all_diffs:
        maps = maps[:2]
    results = [describe(bm, model, a.threshold) for bm in maps]

    if a.format == "json":
        print(json.dumps(results, indent=2))
    elif a.format == "trello":
        print(to_trello(results))
    else:
        for r in results:
            print(r["map"], f"({r['bpm']:g} BPM)")
            for t in r["tags"]:
                print(f"  {r['scores'][t]:.2f}  {t}")
                for x in r["evidence"].get(t, []):
                    print(f"          {x}")


if __name__ == "__main__":
    main()
