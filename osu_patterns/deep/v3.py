"""Deep v3 (4K only): one event network for the chart model and the causal Lens model.

    events --(256-item row vocabulary + log-scale sin/cos timing)--> event vectors
           --(causal-optional conv stem)--> --(block-local attention, time-biased, +-LOCAL_S s)-->
           per-event features h_i
           --(per-task gated attention-MIL over events)--> chart logits
           per-event task scores s_it = w_t . h_i + b_t, and chart logit = sum_i a_it s_it
           (a_it: softmax over events), so each event's share of the chart logit localises the skill.

* Row token: every column is none / head / tail / held, so a row is one of 4^4 = 256 vocabulary items
  (like Dance Dance Convolution's steps). Mirroring is a permutation of the vocabulary.
* Time: sin/cos features of log2(gap in ms) for the previous and the next gap, their log ratio, the log
  scroll speed and the visual channels. No snap, beat or BPM-grid information anywhere.
* No window transformer: nothing global except the MIL pooling (labels are "somewhere in the chart").
* causal=True: convolutions and attention only look back, so s_it depends on what was played up to
  event i. This is the live Pattern Lens net (same class, same inputs, trained the same way).
"""
from __future__ import annotations

import math

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

from . import tokens

K = 4
VOCAB = 4 ** K
N_FREQ = 6
LOCAL_S = 2.0            # seconds of context each side that attention can use
BLOCK = 64               # events per attention block (keys: previous, own and next block)
GAP_MAX_MS = 4000.0
N_VIS = tokens.N_VIS
D_TIME = 2 * (1 + 2 * N_FREQ) + 2 + N_VIS
NEXT_COLS = slice(1 + 2 * N_FREQ, 2 * (1 + 2 * N_FREQ) + 1)   # next-gap features and the gap ratio: look one event ahead   # sin/cos of both gaps, ratio, scroll, visual
FREQ = (math.pi / 12.0) * 2.0 ** np.arange(N_FREQ)


def _perm() -> np.ndarray:
    """Vocabulary permutation that mirrors a row (column c -> K-1-c)."""
    ids = np.arange(VOCAB)
    st = np.stack([(ids // 4 ** c) % 4 for c in range(K)], 1)
    return (st[:, ::-1] * 4 ** np.arange(K)).sum(1)


def _bits_table() -> np.ndarray:
    """(VOCAB, 3K) head/tail/held bits of every vocabulary item."""
    ids = np.arange(VOCAB)
    st = np.stack([(ids // 4 ** c) % 4 for c in range(K)], 1)
    return np.concatenate([(st == 1), (st == 2), (st == 3)], 1).astype(np.float32)


MIRROR = _perm()


def row_ids(cols: np.ndarray) -> np.ndarray:
    """(L, 4, 3) head/tail/held bits -> (L,) vocabulary ids; a column is 0 none, 1 head, 2 tail, 3 held."""
    st = np.where(cols[..., 0] > 0, 1, np.where(cols[..., 1] > 0, 2, np.where(cols[..., 2] > 0, 3, 0)))
    return (st * 4 ** np.arange(K)).sum(1).astype(np.int64)


def time_features(t: np.ndarray, cont: np.ndarray) -> np.ndarray:
    """(L, D_TIME) from event times t (s) and tokens cont (log scroll speed at column 2, visual after N_CONT)."""
    ms = np.diff(t.astype(np.float64)) * 1000.0
    dp = np.concatenate([[2000.0], ms])
    dn = np.concatenate([ms, [2000.0]])
    xs = [np.log2(np.clip(g, 1.0, GAP_MAX_MS)) for g in (dp, dn)]
    parts = []
    for x in xs:
        a = x[:, None] * FREQ[None]
        parts += [x[:, None] / 12.0, np.sin(a), np.cos(a)]
    parts += [np.clip(xs[1] - xs[0], -6, 6)[:, None] / 3, cont[:, 2:3].astype(np.float64)]
    vis = cont[:, tokens.N_CONT:tokens.N_CONT + N_VIS].astype(np.float64)
    if vis.shape[1] < N_VIS:
        vis = np.zeros((len(t), N_VIS))
    return np.concatenate(parts + [vis], 1).astype(np.float32)


RATE_BIN_S = 2.0
RATE_THR = (4.0, 7.0, 10.0)          # events per second


def chart_features(t: np.ndarray, cont: np.ndarray) -> np.ndarray:
    """Whole-chart summary of timing and scroll, from the full chart (not the training crop): length, gap
    percentiles, how dense and for how long, and how the scroll speed behaves. The windowed MIL net only sees
    +-2 s, so duration and endurance ("Stamina", "TB") and the amount of SV need these. Still ms-only."""
    t = t.astype(np.float64)
    dur = max(t[-1] - t[0], 1.0)
    lg = np.log2(np.clip(np.diff(t) * 1000.0, 1.0, GAP_MAX_MS))
    f = [np.log1p(dur) / 6, np.log(len(t)) / 10, len(t) / dur / 10]
    f += list(np.percentile(lg, [5, 25, 50, 75, 95]) / 12)
    nb = int(dur // RATE_BIN_S) + 1
    rate = np.bincount(((t - t[0]) // RATE_BIN_S).astype(int), minlength=nb) / RATE_BIN_S
    f += list(np.percentile(rate, [50, 75, 90, 99]) / 10)
    for thr in RATE_THR:
        on = rate >= thr
        run = best = 0
        for x in on:
            run = run + 1 if x else 0
            best = max(best, run)
        f += [np.log1p(best * RATE_BIN_S) / 5, on.mean()]
    sp = cont[:, 2].astype(np.float64)
    d = np.abs(np.diff(sp))
    f += [sp.mean(), sp.std(), sp.min(), sp.max(), np.log1p(d.sum()), (d > 0.02).mean()]
    vis = cont[:, tokens.N_CONT:tokens.N_CONT + N_VIS].astype(np.float64)
    if vis.shape[1] < N_VIS:
        vis = np.zeros((len(t), N_VIS))
    f += list(vis.mean(0)) + list(vis.max(0)) + list((np.abs(vis) > 1e-3).mean(0))
    return np.nan_to_num(np.array(f, np.float32), posinf=0, neginf=0)


D_GLOB = 3 + 5 + 4 + 2 * len(RATE_THR) + 6 + 3 * N_VIS


def prepare(e: tokens.Events) -> dict:
    return {"ids": row_ids(e.cols), "tf": time_features(e.t, e.cont), "t": e.t.astype(np.float32),
            "glob": chart_features(e.t, e.cont)}


def collate(items: list[dict], train: bool, crop: int, mirror: bool | None = None) -> dict:
    """items: prepare() outputs. mirror: None = random when training; True / False forces it."""
    xs = []
    for p in items:
        L = len(p["t"])
        a = np.random.randint(0, L - crop + 1) if train and L > crop else max(0, (L - crop) // 2)
        sl = slice(a, a + crop)
        flip = (train and np.random.rand() < 0.5) if mirror is None else mirror
        ids = MIRROR[p["ids"][sl]] if flip else p["ids"][sl]
        xs.append((ids, p["tf"][sl], p["t"][sl] - p["t"][sl][0]))
    L = max(len(x[0]) for x in xs)
    B = len(xs)
    ids = np.zeros((B, L), np.int64)
    tf = np.zeros((B, L, D_TIME), np.float32)
    t = np.zeros((B, L), np.float32)
    mask = np.zeros((B, L), bool)
    for i, (a, b, c) in enumerate(xs):
        n = len(a)
        ids[i, :n], tf[i, :n], t[i, :n], mask[i, :n] = a, b, c, True
        t[i, n:] = c[-1] + 1e4
    return {"ids": torch.from_numpy(ids), "tf": torch.from_numpy(tf), "t": torch.from_numpy(t),
            "mask": torch.from_numpy(mask), "glob": torch.from_numpy(np.stack([p["glob"] for p in items]))}


class ConvBlock(nn.Module):
    def __init__(self, c: int, dilation: int, p_drop: float, causal: bool):
        super().__init__()
        self.norm = nn.LayerNorm(c)
        self.conv = nn.Conv1d(c, 2 * c, 3, padding=0 if causal else dilation, dilation=dilation)
        self.left = 2 * dilation if causal else 0
        self.drop = nn.Dropout(p_drop)

    def forward(self, h, m):  # h (B, L, C), m (B, L, 1)
        x = (self.norm(h) * m).transpose(1, 2)
        if self.left:
            x = F.pad(x, (self.left, 0))
        x = F.glu(self.conv(x), dim=1).transpose(1, 2)
        return h + self.drop(x) * m


def _blocks(x: torch.Tensor, nb: int, pad_value=0.0):
    """x (B, L, ...) -> own blocks (B, nb, S, ...) and the previous+own+next blocks (B, nb, 3S, ...)."""
    B, L = x.shape[:2]
    pad = nb * BLOCK - L
    if pad:
        x = F.pad(x, [0, 0] * (x.dim() - 2) + [0, pad], value=pad_value)
    own = x.reshape(B, nb, BLOCK, *x.shape[2:])
    z = torch.full_like(own[:, :1], pad_value)
    prev = torch.cat([z, own[:, :-1]], 1)
    nxt = torch.cat([own[:, 1:], z], 1)
    return own, torch.cat([prev, own, nxt], 2)


BIN_S = 4.0
MAX_BINS = 256


class BinPath(nn.Module):
    """Chart-level context at a coarse scale: event features -> mean/max per BIN_S seconds -> dilated conv over
    the bins (receptive field ~ 1 min) -> mean/max over bins. This is what sustained density (Stamina), length
    (TB) and long SV sections need; the +-2 s attention can't see them. Causal when the net is."""

    def __init__(self, c: int, p_drop: float, causal: bool):
        super().__init__()
        self.inp = nn.Sequential(nn.Linear(2 * c + 1, c), nn.GELU(), nn.LayerNorm(c))
        self.convs = nn.ModuleList(ConvBlock(c, d, p_drop, causal) for d in (1, 2, 4, 8))

    def windows(self, h, t, mask):
        """(B, W, C) window features and (B, W) which windows have events."""
        B, L, C = h.shape
        W = min(int(t.masked_fill(~mask, 0).amax() // BIN_S) + 1, MAX_BINS)
        bi = (t // BIN_S).long().clamp(0, W - 1) + torch.arange(B, device=h.device)[:, None] * W
        idx, m = bi.reshape(-1)[mask.reshape(-1)], mask.reshape(-1)
        flat = h.reshape(-1, C)[m].float()
        cnt = torch.zeros(B * W, device=h.device).index_add_(0, idx, torch.ones_like(idx, dtype=torch.float))
        mean = torch.zeros(B * W, C, device=h.device).index_add_(0, idx, flat) / cnt.clamp(min=1)[:, None]
        mx = torch.full((B * W, C), -1e4, device=h.device).index_reduce_(0, idx, flat, "amax")
        mx = torch.where(cnt[:, None] > 0, mx, torch.zeros_like(mx))
        x = self.inp(torch.cat([mean, mx, torch.log1p(cnt)[:, None] / 3], 1).to(h.dtype)).reshape(B, W, C)
        return x, (cnt > 0).reshape(B, W)

    def forward(self, h, t, mask):
        x, bm = self.windows(h, t, mask)
        mm = bm[..., None].to(x.dtype)
        x = x * mm
        for blk in self.convs:
            x = blk(x, mm)
        n = bm.sum(1, keepdim=True).clamp(min=1)
        return torch.cat([(x * mm).sum(1) / n, x.masked_fill(~bm[..., None], -1e4).amax(1)], -1)


class LocalAttention(nn.Module):
    """Multi-head self-attention within +-LOCAL_S seconds, computed block-locally (linear in events).
    The score gets a learnt per-head penalty on |dt| in seconds, so closer events weigh more."""

    def __init__(self, c: int, heads: int, p_drop: float, causal: bool):
        super().__init__()
        self.h, self.causal = heads, causal
        self.norm = nn.LayerNorm(c)
        self.qkv = nn.Linear(c, 3 * c)
        self.proj = nn.Linear(c, c)
        self.slope = nn.Parameter(torch.linspace(-1.0, 1.5, heads))   # softplus -> penalty per second
        self.drop = nn.Dropout(p_drop)
        self.ff = nn.Sequential(nn.LayerNorm(c), nn.Linear(c, 2 * c), nn.GELU(), nn.Dropout(p_drop), nn.Linear(2 * c, c))

    def forward(self, h, t, mask):
        B, L, C = h.shape
        nb = (L + BLOCK - 1) // BLOCK
        q, k, v = self.qkv(self.norm(h)).chunk(3, -1)
        d = C // self.h
        qo, _ = _blocks(q, nb)
        _, kk = _blocks(k, nb)
        _, vv = _blocks(v, nb)
        to, tk = _blocks(t, nb, 1e9)
        mo, mk = _blocks(mask, nb, False)
        io, ik = _blocks(torch.arange(L, device=h.device)[None].expand(B, -1), nb, 1 << 30)
        dt = to[..., :, None] - tk[..., None, :]                          # (B, nb, S, 3S)
        ok = mk[..., None, :] & mo[..., :, None] & (dt.abs() <= LOCAL_S)
        if self.causal:
            ok = ok & (ik[..., None, :] <= io[..., :, None])
        selfk = torch.zeros(BLOCK, 3 * BLOCK, dtype=torch.bool, device=h.device)
        selfk[:, BLOCK:2 * BLOCK] = torch.eye(BLOCK, dtype=torch.bool, device=h.device)
        ok = ok | selfk[None, None]  # an event always sees itself
        bias = -F.softplus(self.slope)[None, None, :, None, None] * dt.abs()[:, :, None]
        bias = bias.masked_fill(~ok[:, :, None], float("-inf"))
        sh = lambda x: x.reshape(B, nb, -1, self.h, d).transpose(2, 3)   # noqa: E731 (B, nb, H, n, d)
        o = F.scaled_dot_product_attention(sh(qo), sh(kk), sh(vv), attn_mask=bias.to(q.dtype))
        o = o.transpose(2, 3).reshape(B, nb * BLOCK, C)[:, :L]
        h = h + self.drop(self.proj(o)) * mask[..., None]
        return h + self.drop(self.ff(h)) * mask[..., None]


class EventNet(nn.Module):
    def __init__(self, n_tasks: int, c: int = 96, n_conv: int = 3, n_attn: int = 2, heads: int = 4, p_drop: float = 0.1,
                 causal: bool = False, d_att: int = 16, n_stats: int = 0, use_glob: bool = True, use_bins: bool = True, wide_dim: int = 0, win_mil: bool = False, multihot: bool = False):
        super().__init__()
        self.c, self.causal, self.n_tasks, self.d_att = c, causal, n_tasks, d_att
        self.multihot = multihot
        if multihot:   # the v2 way: per-column head/tail/held bits through a linear layer, not a learned row embedding
            self.row_lin = nn.Linear(3 * K, c)
            self.mask_vec = nn.Parameter(torch.zeros(c))
            self.register_buffer("bits", torch.from_numpy(_bits_table()))
        else:
            self.row = nn.Embedding(VOCAB + 1, c)                     # last item = mask token
        self.time = nn.Sequential(nn.Linear(D_TIME, c), nn.GELU(), nn.Linear(c, c))
        self.convs = nn.ModuleList(ConvBlock(c, 2 ** i, p_drop, causal) for i in range(n_conv))
        self.attn = nn.ModuleList(LocalAttention(c, heads, p_drop, causal) for _ in range(n_attn))
        self.norm = nn.LayerNorm(c)
        self.V = nn.Linear(c, d_att * n_tasks)
        self.U = nn.Linear(c, d_att * n_tasks)
        self.w = nn.Parameter(torch.randn(n_tasks, d_att) / math.sqrt(d_att))
        self.cls_w = nn.Parameter(torch.randn(n_tasks, c) / math.sqrt(c))
        self.cls_b = nn.Parameter(torch.zeros(n_tasks))
        self.slot = nn.Linear(2 * c, 4)
        self.win_mil = BinPath(c, p_drop, causal) if win_mil else None   # windows -> MIL, as deep v2 (not causal)
        self.bins = BinPath(c, p_drop, causal) if use_bins else None
        self.glob = None
        if use_glob:   # chart-level path: pooled embedding + whole-chart timing/scroll summary -> logit corrections
            self.glob = nn.Sequential(nn.Linear(2 * c * (2 if use_bins else 1) + D_GLOB + wide_dim, 2 * c), nn.GELU(), nn.Dropout(p_drop),
                                      nn.Linear(2 * c, n_tasks + 4))
        self.row_head = nn.Sequential(nn.Linear(c, c), nn.GELU(), nn.Linear(c, VOCAB))
        self.stats_head = nn.Sequential(nn.Linear(2 * c, 2 * c), nn.GELU(), nn.Linear(2 * c, max(n_stats, 1)))

    def encode(self, b, masked=None):
        ids = b["ids"]
        if masked is not None:
            ids = torch.where(masked, torch.full_like(ids, VOCAB), ids)
        m = b["mask"][..., None].float()
        tf = b["tf"]
        if self.causal:   # the next gap is the next event's time: not known when this event is played
            tf = torch.cat([tf[..., :NEXT_COLS.start], torch.zeros_like(tf[..., NEXT_COLS]), tf[..., NEXT_COLS.stop:]], -1)
        if self.multihot:
            r = self.row_lin(self.bits[ids.clamp(max=VOCAB - 1)])
            if masked is not None:
                r = torch.where(masked[..., None], self.mask_vec.expand_as(r), r)
        else:
            r = self.row(ids)
        h = (r + self.time(tf)) * m
        for blk in self.convs:
            h = blk(h, m)
        for blk in self.attn:
            h = blk(h, b["t"], b["mask"])
        return self.norm(h) * m

    def heads(self, h, mask, glob=None, t=None, wide=None):
        if self.win_mil is not None:   # pool events into 4 s windows first, then attention-MIL over windows
            hm, mm = self.win_mil.windows(h, t, mask)
        else:
            hm, mm = h, mask
        B, L, C = hm.shape
        g = torch.tanh(self.V(hm)) * torch.sigmoid(self.U(hm))
        a = torch.einsum("bltd,td->blt", g.view(B, L, self.n_tasks, self.d_att), self.w)
        a = a.masked_fill(~mm[..., None], -1e4).softmax(1)               # (B, L, T) over events (or windows)
        s = hm @ self.cls_w.T + self.cls_b                                # per-event (or window) task scores
        logits = (a * s).sum(1)
        n = mask.sum(1, keepdim=True).clamp(min=1)
        mean = (h * mask[..., None]).sum(1) / n
        mx = h.masked_fill(~mask[..., None], -1e4).amax(1)
        pooled = torch.cat([mean, mx], -1)
        slot = self.slot(pooled)
        if self.glob is not None:
            parts = [pooled] + ([self.bins(h, t, mask).to(pooled.dtype)] if self.bins is not None else [])
            g = self.glob(torch.cat(parts + [glob.to(pooled.dtype)] + ([wide.to(pooled.dtype)] if wide is not None else []), -1))
            logits, slot = logits + g[:, :self.n_tasks], slot + g[:, self.n_tasks:]
        return logits, slot, self.stats_head(pooled), a, s

    def forward(self, b, masked=None):
        h = self.encode(b, masked)
        return h, *self.heads(h, b["mask"], b["glob"], b["t"], b.get("wide"))

    def row_logits(self, h):
        return self.row_head(h)


def export(net: EventNet, tasks: list[str]) -> dict:
    """Weights as numpy arrays for osu_patterns.note_model_v3 (causal nets only)."""
    assert net.causal and net.bins is None and net.win_mil is None
    sd = {k: v.detach().cpu().numpy().astype(np.float32) for k, v in net.state_dict().items()}
    sd["n_conv"] = np.array(len(net.convs))
    sd["n_attn"] = np.array(len(net.attn))
    sd["heads"] = np.array(net.attn[0].h) if len(net.attn) else np.array(1)
    return sd
