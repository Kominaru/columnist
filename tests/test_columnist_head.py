import json

import numpy as np
import pytest

torch = pytest.importorskip("torch")
from osu_patterns import mania, note_model_columnist  # noqa: E402
from osu_patterns.deep import columnist as C  # noqa: E402


def fake_chart(n=600, seed=0):
    r = np.random.default_rng(seed)
    t = np.cumsum(r.integers(40, 300, n))
    notes = []
    for i in range(n):
        col = int(r.integers(0, 4))
        end = int(t[i] + r.integers(80, 900)) if r.random() < 0.15 else None
        notes.append((col, int(t[i]), end))
        if r.random() < 0.2:
            notes.append(((col + 1) % 4, int(t[i]), None))
    sv = [(int(x), float(r.choice([0.8, 1.0, 1.0, 1.4]))) for x in t[::90]]
    return notes, sv


def tiny(n_tasks=5, seed=0):
    torch.manual_seed(seed)
    net = C.LensColumnist(n_tasks, d=32, layers=2, heads=4, p=0.0).eval()
    for p in net.parameters():
        torch.nn.init.normal_(p, 0, 0.3) if p.dim() > 1 else torch.nn.init.normal_(p, 0, 0.1)
    return net


def batch(notes, sv):
    tok, gap, svr, _ = C.chart_rows(notes, sv)
    t = torch.from_numpy(tok.astype(np.int64))[None]
    return t, torch.from_numpy(gap.astype(np.float32))[None], torch.from_numpy(svr.astype(np.float32))[None], tok, gap, svr


def test_row_scores_ignore_the_future():
    notes, sv = fake_chart()
    net = tiny()
    tok, gap, svr, *_ = batch(notes, sv)
    valid = torch.ones_like(tok, dtype=torch.bool)
    s, _ = net(tok, gap, svr, valid)
    k = tok.shape[1] // 2
    tok2, gap2, svr2 = tok.clone(), gap.clone(), svr.clone()
    tok2[0, k:] = (tok2[0, k:] + 37) % 256
    gap2[0, k:] += 17
    svr2[0, k:] = 2.0
    s2, _ = net(tok2, gap2, svr2, valid)
    assert torch.allclose(s[0, :k], s2[0, :k], atol=1e-5)
    assert not torch.allclose(s[0, k:], s2[0, k:], atol=1e-3)


def test_windowed_scoring_equals_one_pass():
    notes, sv = fake_chart(900)
    net = tiny()
    tok, gap, svr, tk, gp, sv_ = batch(notes, sv)
    valid = torch.ones_like(tok, dtype=torch.bool)
    with torch.no_grad():
        s, g = net(tok, gap, svr, valid)
    S, G = C.chart_scores(net, tk, gp, sv_, torch.device("cpu"), win=700, ctx=C.RECEPTIVE + 16)
    assert np.abs(S - s[0].numpy()).max() < 1e-4 and np.abs(G - g[0].numpy()).max() < 1e-4


def test_numpy_inference_matches_torch(tmp_path):
    notes, sv = fake_chart(500, 3)
    net = tiny(5, 1)
    tok, gap, svr, tk, gp, sv_ = batch(notes, sv)
    with torch.no_grad():
        s, g = net(tok, gap, svr, torch.ones_like(tok, dtype=torch.bool))
    np.savez(tmp_path / "m.npz", **C.export(net))
    (tmp_path / "m.json").write_text(json.dumps({"tasks": list("abcde")}))
    nm = note_model_columnist.NoteModelColumnist(tmp_path / "m.npz")
    ns, ng = nm.row_outputs(tk, gp, sv_)
    assert np.abs(ns - s[0].numpy()).max() < 2e-3 and np.abs(ng - g[0].numpy()).max() < 2e-3


def test_map_rows_gives_every_note_its_row():
    notes, sv = fake_chart(120, 5)
    text = "osu file format v14\n[General]\nMode: 3\n[Difficulty]\nCircleSize:4\nOverallDifficulty:8\n[TimingPoints]\n0,500,4,1,0,100,1,0\n"
    text += "".join(f"{t},{-100 / m:.4f},4,1,0,100,0,0\n" for t, m in sv)
    text += "[HitObjects]\n"
    for c, t, e in notes:
        text += f"{c * 128 + 64},192,{t},{128 if e else 1},0,{str(e) + ':' if e else ''}0:0:0:0:\n"
    m = mania.parse_mania(text)
    tok, gap, svr, times, row = C.map_rows(m)
    tk, gp, sv_, tm = C.chart_rows(sorted(notes, key=lambda n: (n[1], n[0])), [(t, round(x, 3)) for t, x in sv])
    assert (tok == tk).all() and (times == tm).all() and np.allclose(svr.astype(float), sv_.astype(float), atol=1e-2)
    st = np.round(m.start).astype(int)
    assert (times[row] == st).all()
