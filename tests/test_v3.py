import numpy as np
import pytest

torch = pytest.importorskip("torch")
from osu_patterns.deep import tokens, v3  # noqa: E402


def fake(L, seed=0):
    r = np.random.default_rng(seed)
    cols = np.zeros((L, 4, 3), np.uint8)
    cols[np.arange(L), r.integers(0, 4, L), 0] = 1
    return tokens.Events(4, np.cumsum(r.uniform(0.03, 0.4, L)).astype(np.float32), cols, np.zeros((L, 7), np.float16))


def test_mirror_is_an_involution_on_the_vocabulary():
    assert (v3.MIRROR[v3.MIRROR] == np.arange(v3.VOCAB)).all()
    e = fake(50)
    assert (v3.MIRROR[v3.row_ids(e.cols)] == v3.row_ids(e.cols[:, ::-1])).all()


def test_causal_scores_ignore_the_future_and_bidirectional_ones_do_not():
    p = v3.prepare(fake(400))
    b = v3.collate([p], False, 400)
    b2 = {k: v.clone() for k, v in b.items()}
    b2["ids"][0, 200:] = (b2["ids"][0, 200:] + 7) % v3.VOCAB
    for causal, same in ((True, True), (False, False)):
        net = v3.EventNet(3, 32, causal=causal, use_glob=False, use_bins=False).eval()
        s = net(b)[-1][0, :200]
        s2 = net(b2)[-1][0, :200]
        assert bool(torch.allclose(s, s2, atol=1e-6)) == same


def test_window_mil_runs():
    b = v3.collate([v3.prepare(fake(300)), v3.prepare(fake(500, 1))], False, 500)
    net = v3.EventNet(3, 32, use_glob=False, use_bins=False, win_mil=True).eval()
    assert net(b)[1].shape == (2, 3)


def test_causal_scores_ignore_future_timing_too():
    e1, e2 = fake(300), fake(300)
    e2 = tokens.Events(4, np.concatenate([e1.t[:150], e1.t[150:] + 0.05 * np.arange(150)]).astype(np.float32), e1.cols, e1.cont)
    net = v3.EventNet(3, 32, causal=True, use_glob=False, use_bins=False).eval()
    a = net(v3.collate([v3.prepare(e1)], False, 300))[-1][0, :149]
    b = net(v3.collate([v3.prepare(e2)], False, 300))[-1][0, :149]
    assert torch.allclose(a, b, atol=1e-6)


def test_numpy_inference_matches_torch(tmp_path):
    import json
    from osu_patterns import note_model_v3
    torch.manual_seed(0)
    net = v3.EventNet(5, 32, n_conv=3, n_attn=2, causal=True, use_glob=False, use_bins=False).eval()
    for p in net.parameters():           # non-trivial weights, incl. the time-bias slopes and norms
        torch.nn.init.normal_(p, 0, 0.3) if p.dim() > 1 else torch.nn.init.normal_(p, 0, 0.1)
    e = fake(400, 3)
    pr = v3.prepare(e)
    with torch.no_grad():
        want = net(v3.collate([pr], False, 400))[-1][0].numpy()
    np.savez(tmp_path / "m.npz", **v3.export(net, []))
    (tmp_path / "m.json").write_text(json.dumps({"tasks": list("abcde")}))
    nm = note_model_v3.NoteModelV3(tmp_path / "m.npz")
    got = nm.event_logits(pr["ids"], pr["tf"], pr["t"].astype(np.float64))
    assert np.abs(got - want).max() < 2e-3


def test_multihot_variant_runs_and_masks():
    b = v3.collate([v3.prepare(fake(300))], False, 300)
    net = v3.EventNet(3, 32, use_glob=False, use_bins=False, multihot=True).eval()
    masked = torch.zeros_like(b["mask"])
    masked[0, 10] = True
    assert net(b, masked)[1].shape == (1, 3)
