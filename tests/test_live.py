"""Live visualizer: per-note labels, .osu metadata, window fast path and the HTTP API."""
import io
import zipfile
from pathlib import Path

import numpy as np
import pytest

from osu_patterns import live, mania, windows

MODELS = Path(__file__).resolve().parents[1] / "models"


def chart(rows, keys=4, ln=()):
    """rows: list of (time, [columns]); ln: times whose notes are 300 ms long notes."""
    objs = []
    for t, cols in rows:
        for c in cols:
            x = int((c + 0.5) * 512 / keys)
            objs.append(f"{x},192,{t},128,0,{t + 300}:0:0:0:0:" if t in ln else f"{x},192,{t},1,0,0:0:0:0:")
    return ("osu file format v14\n\n[General]\nAudioFilename: song.mp3\nPreviewTime: 1200\nMode: 3\n\n"
            "[Metadata]\nTitle:T\nArtist:A\nCreator:C\nVersion:V\nBeatmapID:1\nBeatmapSetID:2\n\n"
            f"[Difficulty]\nHPDrainRate:8\nCircleSize:{keys}\nOverallDifficulty:8\n\n"
            '[Events]\n0,0,"bg.jpg",0,0\n\n[TimingPoints]\n0,500,4,1,0,60,1,0\n\n[HitObjects]\n' + "\n".join(objs))


def test_local_labels():
    jack = [(1000 + 125 * i, [0]) for i in range(4)]
    stream = [(2000 + 125 * i, [i % 4]) for i in range(4)]
    js = [(3000 + 125 * i, [[0, 2], [1, 3]][i % 2]) for i in range(4)]
    m = mania.parse_mania(chart(jack + stream + js + [(4000, [1])], ln={4000}))
    lab = live.local_labels(m)
    assert lab[1:4] == ["jack"] * 3
    assert set(lab[5:8]) <= {"stream", "trill"}
    assert lab[9:16].count("jumpstream") >= 6
    assert lab[-1] == "ln"


def test_osu_meta():
    meta = live.osu_meta(chart([(1000, [0])]))
    assert meta["audio"] == "song.mp3" and meta["bg"] == "bg.jpg" and meta["beatmapset_id"] == "2"
    assert meta["preview"] == 1200


def random_chart(seconds=40, seed=0):
    rng = np.random.default_rng(seed)
    rows = [(int(t), sorted(set(rng.integers(0, 4, rng.integers(1, 3)).tolist())))
            for t in np.arange(1000, 1000 + seconds * 1000, 125)]
    return chart(rows)


@pytest.mark.skipif(not (MODELS / "live_heads.pkl").exists(), reason="no live heads")
def test_window_fast_path_matches_pool():
    model = live.LiveModel(MODELS)
    m = mania.parse_mania(random_chart())
    w = windows.submap(m, 5000, 21000)
    fast = model._row(w)
    slow = windows.pool(windows.window_stats(w))
    assert all(np.isclose(fast[k], v, equal_nan=True) for k, v in slow.items())
    a = model.analyze(m)
    assert len(a["windows"]["t"]) > 20
    assert set(a["windows"]["series"]) >= {"Chordjack", "slot:RC"}
    assert "Stamina" not in a["windows"]["series"]


@pytest.mark.skipif(not (MODELS / "live_heads.pkl").exists(), reason="no live heads")
def test_api_upload_osz(tmp_path, monkeypatch):
    monkeypatch.setenv("WORK_DIR", str(tmp_path))
    from fastapi.testclient import TestClient
    from service.viz import app as viz
    client = TestClient(viz.app)
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as z:
        z.writestr("map.osu", random_chart())
        z.writestr("song.mp3", b"ID3fake")
    r = client.post("/api/upload", files={"file": ("x.osz", buf.getvalue())})
    assert r.status_code == 200, r.text
    info = r.json()
    assert info["audio"] and info["diffs"][0]["keys"] == 4
    c = client.get(f"/api/chart/{info['sid']}/0").json()
    assert len(c["notes"]["t"]) == len(c["notes"]["k"])
    assert client.get(f"/api/audio/{info['sid']}").content == b"ID3fake"
    assert client.get("/api/chart/../../etc/0").status_code == 404


@pytest.mark.parametrize("causal", [False, True])
def test_note_model_numpy_matches_torch(tmp_path, causal):
    torch = pytest.importorskip("torch")
    import json
    from osu_patterns import note_model
    from osu_patterns.deep import tokens
    from osu_patterns.deep.local import LocalNet
    m = mania.parse_mania(random_chart())
    ev = tokens.encode(m, visual=True)
    x = np.concatenate([tokens.pad_columns(ev.cols, 4), ev.cont.astype(np.float32)], 1).astype(np.float32)
    torch.manual_seed(0)
    net = LocalNet(x.shape[1], 3, c=16, causal=causal).eval()
    np.savez(tmp_path / "nm.npz", **net.export())
    (tmp_path / "nm.json").write_text(json.dumps({"tasks": ["a", "b", "c"], "thresholds": {}, "visual": True}))
    nm = note_model.NoteModel(tmp_path / "nm.npz")
    with torch.no_grad():
        chart, s = net(torch.from_numpy(x)[None], torch.ones(1, len(x), dtype=torch.bool))
    assert np.allclose(nm.event_logits(x), s[0].numpy(), atol=1e-4)
    r = nm.scores(m)
    assert r["note"].shape == (len(m.start), 3)
    assert np.allclose(list(r["chart"].values()), torch.sigmoid(chart[0]).numpy(), atol=1e-4)


def test_causal_note_model_ignores_the_future():
    torch = pytest.importorskip("torch")
    from osu_patterns.deep.local import LocalNet
    torch.manual_seed(0)
    net = LocalNet(10, 2, c=8, causal=True).eval()
    x = torch.randn(1, 200, 10)
    y = x.clone()
    y[:, 120:] = torch.randn(1, 80, 10)          # change only events 120+
    mask = torch.ones(1, 200, dtype=torch.bool)
    with torch.no_grad():
        a, b = net.events(x, mask), net.events(y, mask)
    assert torch.allclose(a[:, :120], b[:, :120], atol=1e-6)
    assert not torch.allclose(a[:, 120:], b[:, 120:])


@pytest.mark.skipif(not (MODELS / "live_heads.pkl").exists(), reason="no live heads")
def test_api_rate_edits_get_their_own_audio(tmp_path, monkeypatch):
    monkeypatch.setenv("WORK_DIR", str(tmp_path))
    from fastapi.testclient import TestClient
    from service.viz import app as viz
    client = TestClient(viz.app)
    base = random_chart()
    rate = base.replace("AudioFilename: song.mp3", "AudioFilename: Song 1.1x.MP3").replace("Version:V", "Version:V 1.1x")
    missing = base.replace("AudioFilename: song.mp3", "AudioFilename: gone.mp3").replace("Version:V", "Version:V gone")
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as z:
        z.writestr("a.osu", base)
        z.writestr("b.osu", rate)
        z.writestr("c.osu", missing)
        z.writestr("song.mp3", b"ID3normal")
        z.writestr("song 1.1x.mp3", b"ID3rate")
    info = client.post("/api/upload", files={"file": ("x.osz", buf.getvalue())}).json()
    assert info["tracks"] == 2
    by = {d["version"]: d["audio"] for d in info["diffs"]}
    assert by["V gone"] is None
    got = {v: client.get(f"/api/audio/{info['sid']}/{k}").content for v, k in by.items() if k is not None}
    assert got == {"V": b"ID3normal", "V 1.1x": b"ID3rate"}


def test_note_model_without_the_grid_ignores_the_written_bpm(tmp_path):
    torch = pytest.importorskip("torch")
    import json
    from osu_patterns import note_model
    from osu_patterns.deep import tokens
    from osu_patterns.deep.local import LocalNet
    rows = [(1000 + 42 * i, [(i * 3) % 4]) for i in range(120)]
    slow, fast = chart(rows), chart(rows).replace("0,500,4,1,0,60,1,0", "0,125,4,1,0,60,1,0")   # 120 vs 480 BPM
    ev = tokens.encode(mania.parse_mania(slow), visual=True)
    d_in = tokens.KMAX * 3 + tokens.N_CONT + (ev.cont.shape[1] - tokens.N_CONT)
    torch.manual_seed(0)
    net = LocalNet(d_in, 3, c=16, causal=True).eval()
    np.savez(tmp_path / "nm.npz", **net.export())
    (tmp_path / "nm.json").write_text(json.dumps({"tasks": ["a", "b", "c"], "thresholds": {}, "visual": True, "grid": False}))
    nm = note_model.NoteModel(tmp_path / "nm.npz")
    a, b = nm.scores(mania.parse_mania(slow)), nm.scores(mania.parse_mania(fast))
    assert np.allclose(a["note"], b["note"], atol=1e-6)
