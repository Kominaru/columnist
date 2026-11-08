"""Live pattern visualizer: play a 4K osu!mania chart and watch the classifier's patterns per section.

    uvicorn service.viz.app:app --port 8770          # from the repo root
    docker compose -f service/viz/docker-compose.yml up -d --build

API
  POST /api/upload            multipart "file": .osz (charts + audio) or a single .osu
  POST /api/fetch             {"url": beatmapset / beatmap link or id}: .osz from a public mirror, else the
                              charts alone from osu.ppy.sh (no audio)
  GET  /api/chart/{sid}/{i}   notes, scroll profile, 8 s window scores, chart-level result
  GET  /api/windows/{sid}/{i}/{ms}   window scores for another window length (4000 / 8000 / 16000)
  GET  /api/audio/{sid}/{k}   audio track k of the set (each difficulty names its track: rate edits
                              and the like use their own file); /api/audio/{sid} is track 0
  GET  /api/bg/{sid}/{i}      the difficulty's background image, if there is one
  GET  /api/model             task names, thresholds, held-out AP
"""
from __future__ import annotations

import hashlib
import io
import json
import os
import re
import shutil
import threading
import time
import urllib.request
import uuid
import zipfile
from pathlib import Path

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from osu_patterns import live, mania, mania_cli
from service.viz import difficulty, similar

HERE = Path(__file__).resolve().parent
MODELS = Path(os.environ.get("MODELS_DIR", HERE.parents[1] / "models"))
WORK = Path(os.environ.get("WORK_DIR", "/tmp/osu-viz"))
MAX_UPLOAD = int(os.environ.get("MAX_UPLOAD_MB", "80")) * 1024 * 1024
KEEP_S = float(os.environ.get("KEEP_HOURS", "6")) * 3600
MIRRORS = [m for m in os.environ.get("OSZ_MIRRORS", "https://catboy.best/d/{id}n,https://api.nerinyan.moe/d/{id}?nv=1,"
                                     "https://osu.direct/api/d/{id}").split(",") if m]
UA = {"User-Agent": "osu-pattern-visualizer/1.0"}
DEFAULT_WIN = 8000.0  # the window length the player opens with; others load on demand
AUDIO_EXT = {".mp3": "audio/mpeg", ".ogg": "audio/ogg", ".wav": "audio/wav", ".m4a": "audio/mp4"}

app = FastAPI(title="osu!mania live pattern visualizer")
app.include_router(similar.router)
app.include_router(difficulty.router)
model = live.LiveModel(MODELS)
WORK.mkdir(parents=True, exist_ok=True)
_lock = threading.Lock()


# ---------------------------------------------------------------------- session storage

def _prune():
    now = time.time()
    for d in WORK.iterdir():
        if d.is_dir() and now - d.stat().st_mtime > KEEP_S:
            shutil.rmtree(d, ignore_errors=True)


def _sdir(sid: str) -> Path:
    if not re.fullmatch(r"[0-9a-f]{32}", sid):
        raise HTTPException(404, "unknown session")
    d = WORK / sid
    if not d.exists():
        raise HTTPException(404, "session expired, load the map again")
    return d


def _store(charts: list[tuple[str, str]], files: dict[str, bytes], source: str) -> dict:
    """charts: (name, .osu text). files: other files of the set by lower-cased name (audio, images)."""
    _prune()
    sid = uuid.uuid4().hex
    d = WORK / sid
    d.mkdir()
    diffs, tracks = [], []   # tracks: lower-cased audio file names, in first-use order
    for name, text in charts:
        m = mania.parse_mania(text)
        meta = live.osu_meta(text)
        if m is None or len(m.start) < 30:
            continue
        i = len(diffs)
        (d / f"{i}.osu").write_text(text, "utf-8")
        a = (meta["audio"] or "").lower()
        if a in files and a not in tracks:
            tracks.append(a)
        diffs.append({"audio": tracks.index(a) if a in files else None, "i": i, "version": meta["version"] or name, "keys": m.keys, "notes": int(len(m.start)),
                      "length_s": round((float(m.end.max()) - float(m.start.min())) / 1000), "supported": m.keys == 4,
                      "has_bg": bool(meta["bg"] and meta["bg"].lower() in files)})
        if meta["bg"] and meta["bg"].lower() in files:
            (d / f"{i}.bg").write_bytes(files[meta["bg"].lower()])
    if not diffs:
        shutil.rmtree(d, ignore_errors=True)
        raise HTTPException(422, "no osu!mania difficulty found")
    for k, a in enumerate(tracks):
        (d / f"audio{k}{Path(a).suffix.lower()}").write_bytes(files[a])
    first = live.osu_meta((d / "0.osu").read_text("utf-8"))
    info = {"sid": sid, "source": source, "artist": first["artist"], "title": first["title"],
            "creator": first["creator"], "beatmapset_id": first["beatmapset_id"], "audio": bool(tracks),
            "tracks": len(tracks),
            "preview": first["preview"],
            "diffs": sorted(diffs, key=lambda x: (not x["supported"], x["notes"]))}
    (d / "info.json").write_text(json.dumps(info), "utf-8")
    return info


def _from_osz(raw: bytes, source: str) -> dict:
    try:
        z = zipfile.ZipFile(io.BytesIO(raw))
    except zipfile.BadZipFile:
        raise HTTPException(422, "not a valid .osz file")
    charts, files = [], {}
    for zi in z.infolist():
        if zi.is_dir() or zi.file_size > MAX_UPLOAD:
            continue
        name = zi.filename.replace("\\", "/")
        low = name.lower()
        if low.endswith(".osu"):
            charts.append((name, z.read(zi).decode("utf-8-sig", "replace")))
        elif Path(low).suffix in AUDIO_EXT or Path(low).suffix in (".jpg", ".jpeg", ".png"):
            files[low] = z.read(zi)
    return _store(charts, files, source)


# ---------------------------------------------------------------------- API

@app.post("/api/upload")
async def upload(file: UploadFile = File(...)):
    raw = await file.read(MAX_UPLOAD + 1)
    if len(raw) > MAX_UPLOAD:
        raise HTTPException(413, f"file larger than {MAX_UPLOAD // 2**20} MB")
    name = (file.filename or "").lower()
    if name.endswith(".osu"):
        return _store([(file.filename, raw.decode("utf-8-sig", "replace"))], {}, "upload")
    return _from_osz(raw, "upload")


class FetchReq(BaseModel):
    url: str


def _ids(text: str) -> tuple[int | None, int | None]:
    text = text.strip()
    s = re.search(r"beatmapsets/(\d+)", text)
    b = re.search(r"#\w+/(\d+)", text) or re.search(r"/(?:b|beatmaps)/(\d+)", text)
    if s:
        return int(s.group(1)), int(b.group(1)) if b else None
    if b:
        return None, int(b.group(1))
    if text.isdigit():
        return int(text), None
    raise HTTPException(422, "paste a beatmapset link, a beatmap link or a beatmapset id")


def _get(url: str, timeout: float = 60) -> bytes:
    with urllib.request.urlopen(urllib.request.Request(url, headers=UA), timeout=timeout) as r:
        return r.read(MAX_UPLOAD + 1)


@app.post("/api/fetch")
def fetch(req: FetchReq):
    set_id, bid = _ids(req.url)
    if set_id is None:  # a beatmap link without its set: the .osu names the set
        try:
            text = mania_cli.public_osu(bid)
        except Exception as e:  # noqa: BLE001
            raise HTTPException(502, f"could not fetch beatmap {bid}: {e}")
        sid = live.osu_meta(text)["beatmapset_id"]
        set_id = int(sid) if sid and sid.lstrip("-").isdigit() and int(sid) > 0 else None
        if set_id is None:
            return _store([(str(bid), text)], {}, "osu.ppy.sh")
    for mirror in MIRRORS:
        try:
            raw = _get(mirror.format(id=set_id))
            if raw[:2] == b"PK" and len(raw) <= MAX_UPLOAD:
                return _from_osz(raw, mirror.split("/")[2])
        except Exception:  # noqa: BLE001 - try the next mirror
            continue
    try:  # no mirror: charts only, from the official site
        js = mania_cli.public_beatmapset(set_id)
        ids = [b["id"] for b in js.get("beatmaps", []) if b.get("mode") == "mania"]
        charts = [(str(i), mania_cli.public_osu(i)) for i in ids[:24]]
    except Exception as e:  # noqa: BLE001
        raise HTTPException(502, f"could not fetch beatmapset {set_id}: {e}")
    info = _store(charts, {}, "osu.ppy.sh")
    return info


@app.get("/api/session/{sid}")
def session(sid: str):
    return json.loads((_sdir(sid) / "info.json").read_text("utf-8"))


@app.get("/api/chart/{sid}/{i}")
def chart(sid: str, i: int):
    d = _sdir(sid)
    cache = d / f"{i}.json"
    if cache.exists():
        return FileResponse(cache, media_type="application/json")
    p = d / f"{i}.osu"
    if not p.exists():
        raise HTTPException(404, "no such difficulty")
    text = p.read_text("utf-8")
    m = mania.parse_mania(text)
    with _lock:  # one analysis at a time: the NAS has few cores
        res = {"meta": live.osu_meta(text), **live.chart_payload(m),
               "analysis": model.analyze(m, sizes=(DEFAULT_WIN,)), "supported": m.keys == 4}
    cache.write_text(json.dumps(res, separators=(",", ":")), "utf-8")
    return JSONResponse(res)


@app.get("/api/windows/{sid}/{i}/{size}")
def window_series(sid: str, i: int, size: int):
    """Scores for another window length, computed on first request (the chart call only does the default)."""
    if float(size) not in live.WIN_SIZES:
        raise HTTPException(404, "unsupported window length")
    d = _sdir(sid)
    cache = d / f"{i}.w{size}.json"
    if cache.exists():
        return FileResponse(cache, media_type="application/json")
    p = d / f"{i}.osu"
    if not p.exists():
        raise HTTPException(404, "no such difficulty")
    m = mania.parse_mania(p.read_text("utf-8"))
    with _lock:
        res = model._series(m, float(size), live.HOP_MS)
    cache.write_text(json.dumps(res, separators=(",", ":")), "utf-8")
    return JSONResponse(res)


@app.get("/api/audio/{sid}")
@app.get("/api/audio/{sid}/{k}")
def audio(sid: str, k: int = 0):
    d = _sdir(sid)
    for f in sorted(d.glob(f"audio{k}.*")) or sorted(d.glob("audio.*")):  # audio.* from sessions before tracks
        return FileResponse(f, media_type=AUDIO_EXT.get(f.suffix, "application/octet-stream"))
    raise HTTPException(404, "no audio for this map")


@app.get("/api/bg/{sid}/{i}")
def bg(sid: str, i: int):
    f = _sdir(sid) / f"{i}.bg"
    if not f.exists():
        raise HTTPException(404, "no background")
    head = f.read_bytes()[:4]
    return FileResponse(f, media_type="image/png" if head.startswith(b"\x89PNG") else "image/jpeg")


@app.get("/api/model")
def model_info():
    return {"tasks": model.tasks, "thresholds": model.thresholds, "report": model.report,
            "chart_only": sorted(live.CHART_ONLY), "win_ms": live.WIN_MS}


@app.get("/health")
def health():
    return {"ok": True, "tasks": len(model.tasks)}


# index.html names its assets with a content hash, so a CDN (Cloudflare caches .js/.css for hours) never
# serves an old app.js next to a new page; the page itself is sent with no-cache.
_ASSET_V = hashlib.sha1(b"".join((HERE / "static" / f).read_bytes() for f in ("app.js", "style.css"))).hexdigest()[:10]
_INDEX = ((HERE / "static" / "index.html").read_text("utf-8")
          .replace('src="app.js"', f'src="app.js?v={_ASSET_V}"').replace('href="style.css"', f'href="style.css?v={_ASSET_V}"'))


@app.get("/similar", include_in_schema=False)
def similar_page():
    return FileResponse(HERE / "static" / "similar.html", headers={"Cache-Control": "no-cache"})


@app.get("/difficulty", include_in_schema=False)
def difficulty_page():
    return FileResponse(HERE / "static" / "difficulty.html", headers={"Cache-Control": "no-cache"})


@app.get("/about", include_in_schema=False)
def about_page():
    return FileResponse(HERE / "static" / "about.html", headers={"Cache-Control": "no-cache"})


@app.get("/", include_in_schema=False)
@app.get("/index.html", include_in_schema=False)
def index():
    return HTMLResponse(_INDEX, headers={"Cache-Control": "no-cache"})


app.mount("/", StaticFiles(directory=HERE / "static", html=True), name="static")
