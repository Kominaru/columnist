"""Difficulty and pattern tags from the joint Columnist model.

  POST /api/difficulty        {"url": beatmap / beatmapset link or id}  or multipart "file": a .osu  (/api/difficulty/upload)
Returns a star-like difficulty, the 31 pattern labels with probabilities, the hardest section and a difficulty-over-time curve.
Runs the numpy port of the joint model (models/joint/joint.npz + joint.json); no torch at serving time.
"""
from __future__ import annotations

import os
import threading
from pathlib import Path

import numpy as np
from fastapi import APIRouter, File, HTTPException, UploadFile
from pydantic import BaseModel

from osu_patterns import live, mania, mania_cli
from osu_patterns.columnist_joint import JointColumnist
from osu_patterns.deep import columnist_rows as C
from service.viz import similar

router = APIRouter()
DIR = Path(os.environ.get("JOINT_DIR", Path(os.environ.get("MODELS_DIR", "models")) / "joint"))
_model: JointColumnist | None = None
_lock = threading.Lock()


def model() -> JointColumnist:
    global _model
    if _model is None:
        if not (DIR / "joint.npz").exists():
            raise HTTPException(503, "the difficulty model is not installed on this server")
        _model = JointColumnist(DIR / "joint.npz")
    return _model


class Req(BaseModel):
    url: str = ""
    beatmap_id: int | None = None


def _curve(row: np.ndarray, gap: np.ndarray, points: int = 120) -> list[dict]:
    t = np.cumsum(gap) / 1000.0
    n = len(row)
    if n < 2:
        return []
    edges = np.linspace(0, t[-1] + 1e-6, points + 1)
    out = []
    for a, b in zip(edges[:-1], edges[1:]):
        m = (t >= a) & (t < b)
        if m.any():
            out.append({"t": round(float((a + b) / 2), 1), "d": round(float(np.sort(row[m])[::-1][:16].mean()), 2)})
    return out


def _run(text: str, bid: int | None) -> dict:
    m = mania.parse_mania(text)
    if m is None or m.keys != 4 or len(m.start) < 30:
        raise HTTPException(422, "only 4K osu!mania charts are supported")
    tok, gap, sv, _, _ = C.map_rows(m)
    gap = np.asarray(gap, dtype=np.float64)
    if len(tok) < 64:
        raise HTTPException(422, "the chart is too short (at least 64 rows)")
    mod = model()
    with _lock:                         # the NAS has few cores: one chart at a time
        out = mod.predict(tok, gap, sv)
    meta = live.osu_meta(text)
    tasks = []
    for name, p in zip(out["tasks"], out["tag_prob"]):
        tasks.append({"name": name, "prob": round(float(p), 3), "on": bool(p >= mod.thresholds.get(name, 0.5))})
    win = out["window_difficulty"]
    hard = int(np.argmax(win))
    t = np.cumsum(gap) / 1000.0
    lo, hi = hard * 512, min(len(t), (hard + 1) * 512) - 1
    return {"chart": {"beatmap_id": bid, "artist": meta["artist"], "title": meta["title"], "version": meta["version"], "mapper": meta["creator"]},
            "difficulty": round(out["difficulty"], 2), "stars": round(out["stars"], 2) if out["stars"] is not None else None,
            "hardest": {"from_s": round(float(t[lo]), 1), "to_s": round(float(t[hi]), 1)},
            "tasks": tasks, "curve": _curve(out["row_score"], gap)}


@router.post("/api/difficulty")
def difficulty(req: Req):
    set_id, bid = (None, req.beatmap_id) if req.beatmap_id else similar._ids(req.url)
    if bid is None:
        try:
            js = mania_cli.public_beatmapset(set_id)
        except Exception as e:  # noqa: BLE001
            if not req.url.strip().isdigit():
                raise HTTPException(502, f"could not fetch beatmapset {set_id}: {e}")
            js = None
        if js is None:
            bid = set_id
        else:
            diffs = [b for b in js.get("beatmaps", []) if b.get("mode") == "mania" and int(b.get("cs", 0)) == 4]
            if not diffs:
                raise HTTPException(422, "no 4K difficulty in that beatmapset")
            if len(diffs) > 1:
                return {"choose": [{"beatmap_id": b["id"], "version": b["version"], "stars": b.get("difficulty_rating")}
                                   for b in sorted(diffs, key=lambda b: b.get("difficulty_rating") or 0)],
                        "title": f'{js.get("artist")} - {js.get("title")}'}
            bid = int(diffs[0]["id"])
    try:
        text = mania_cli.public_osu(bid)
    except Exception as e:  # noqa: BLE001
        raise HTTPException(502, f"could not fetch beatmap {bid}: {e}")
    return _run(text, bid)


@router.post("/api/difficulty/upload")
async def difficulty_upload(file: UploadFile = File(...)):
    raw = await file.read(8 * 2 ** 20)
    return _run(raw.decode("utf-8-sig", "replace"), None)
