"""Find similar charts: a beatmap goes in, the most similar indexed 4K charts come out (Columnist embeddings).

  POST /api/similar        {"url": beatmap / beatmapset link or id, "k": 20, "same_set": false}
                           or multipart "file": a .osu
  GET  /api/similar/info   index size and which model it was built with

A beatmap that is in the index uses its stored embedding; any other chart is embedded on the spot (numpy, no torch).
A set link with several 4K difficulties answers {"choose": [...]} so the page can ask which one.
"""
from __future__ import annotations

import json
import os
import re
import threading
from pathlib import Path

from fastapi import APIRouter, File, HTTPException, UploadFile
from pydantic import BaseModel

from osu_patterns import live, mania, mania_cli
from osu_patterns.columnist_embed import SimilarIndex

router = APIRouter()
DIR = Path(os.environ.get("SIMILAR_DIR", Path(os.environ.get("MODELS_DIR", "models")) / "similar"))
_index: SimilarIndex | None = None
_lock = threading.Lock()


def index() -> SimilarIndex:
    global _index
    if _index is None:
        if not (DIR / "charts.csv").exists():
            raise HTTPException(503, "the chart index is not installed on this server")
        _index = SimilarIndex(DIR)
    return _index


class Req(BaseModel):
    url: str = ""
    beatmap_id: int | None = None
    k: int = 20
    same_set: bool = False


def _ids(text: str) -> tuple[int | None, int | None]:
    text = text.strip()
    s = re.search(r"beatmapsets/(\d+)", text)
    b = re.search(r"#\w+/(\d+)", text) or re.search(r"/(?:b|beatmaps)/(\d+)", text)
    if s:
        return int(s.group(1)), int(b.group(1)) if b else None
    if b:
        return None, int(b.group(1))
    if text.isdigit():
        return int(text), None          # a bare number is read as a beatmapset id, as in the Lens
    raise HTTPException(422, "paste a beatmap link (osu.ppy.sh/beatmapsets/…#mania/…) or a beatmap id")


def _describe(r: dict) -> dict:
    return {k: r.get(k) for k in ("beatmap_id", "beatmapset_id", "artist", "title", "version", "mapper", "stars",
                                  "length_s", "n_notes", "nps_mean", "ln_share")}


def _run(text: str | None, bid: int | None, k: int, same_set: bool) -> dict:
    ix = index()
    k = max(1, min(int(k), 50))
    row = ix.row_of.get(bid) if bid else None
    if row is not None:
        z = ix.emb[row]
        q = _describe(ix.meta.iloc[row].to_dict())
        q["indexed"] = True
        skip, excl = (None if same_set else int(ix.meta.beatmapset_id.iloc[row])), {bid}
    else:
        m = mania.parse_mania(text)
        if m is None or m.keys != 4 or len(m.start) < 30:
            raise HTTPException(422, "only 4K osu!mania charts are supported")
        with _lock:             # the NAS has few cores: one embedding at a time
            e = ix.embedder.embed_map(m)
        z = ix.normalise(e)
        meta = live.osu_meta(text)
        length = (float(m.end.max()) - float(m.start.min())) / 1000
        q = {"beatmap_id": bid, "beatmapset_id": int(meta["beatmapset_id"]) if (meta.get("beatmapset_id") or "").isdigit() else None,
             "artist": meta["artist"], "title": meta["title"], "version": meta["version"], "mapper": meta["creator"],
             "length_s": round(length), "n_notes": int(len(m.start)), "nps_mean": round(len(m.start) / max(length, 1), 2),
             "indexed": False}
        skip = q["beatmapset_id"] if not same_set else None
        excl = {bid} if bid else None
    return {"query": q, "results": ix.query(z, k, exclude=excl, skip_set=skip)}


@router.post("/api/similar")
def similar(req: Req):
    ix = index()
    set_id, bid = (None, req.beatmap_id) if req.beatmap_id else _ids(req.url)
    bare = req.url.strip().isdigit() and not req.beatmap_id
    if bare and int(req.url) in ix.row_of:      # a bare number that is an indexed beatmap id
        set_id, bid = None, int(req.url)
    if bid is None:                      # a set: which 4K difficulty?
        try:
            js = mania_cli.public_beatmapset(set_id)
        except Exception as e:  # noqa: BLE001
            if not bare:
                raise HTTPException(502, f"could not fetch beatmapset {set_id}: {e}")
            js = None               # a bare number that is no beatmapset: read it as a beatmap id
    if bid is None and js is None:
        bid = set_id
    elif bid is None:
        diffs = [b for b in js.get("beatmaps", []) if b.get("mode") == "mania" and int(b.get("cs", 0)) == 4]
        if not diffs:
            raise HTTPException(422, "no 4K difficulty in that beatmapset")
        if len(diffs) > 1:
            return {"choose": [{"beatmap_id": b["id"], "version": b["version"], "stars": b.get("difficulty_rating")}
                               for b in sorted(diffs, key=lambda b: b.get("difficulty_rating") or 0)],
                    "title": f'{js.get("artist")} - {js.get("title")}'}
        bid = int(diffs[0]["id"])
    text = None
    if bid not in ix.row_of:
        try:
            text = mania_cli.public_osu(bid)
        except Exception as e:  # noqa: BLE001
            raise HTTPException(502, f"could not fetch beatmap {bid}: {e}")
    return _run(text, bid, req.k, req.same_set)


@router.post("/api/similar/upload")
async def similar_upload(file: UploadFile = File(...), k: int = 20):
    raw = await file.read(8 * 2 ** 20)
    return _run(raw.decode("utf-8-sig", "replace"), None, k, True)


@router.get("/api/similar/info")
def info():
    ix = index()
    extra = {}
    if (DIR / "info.json").exists():
        extra = json.loads((DIR / "info.json").read_text("utf-8"))
    return {"charts": int(len(ix.meta)), "dim": int(ix.emb.shape[1]), **extra}
