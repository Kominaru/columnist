"""Columnist encoder (4K rows, no BPM or snap) with a per-row pattern head: the Lens per-note model.

A chart is a sequence of ROWS: the notes that fall at the same millisecond (a hold tail is its own row). Per row
a token for the state of the four columns (none / tap / hold head / hold tail, 4**4 = 256 tokens), the gap in ms
to the previous row and the visible scroll multiplier. Nothing else: no BPM, no snaps, no grid.

The encoder is the pretrained Columnist (scripts/foundation/columnist.py on the Columnist branch; the class below
is the same network, so its checkpoints load unchanged). Run causally, row i only sees rows <= i (two left-padded
convolutions and local attention that looks back at most WINDOW rows), so a row's score depends on what has been
played up to it and on nothing after it. Row features use the gap to the PREVIOUS row only.

Head: per row and task, s_it = w_t . h_i + b_t (the number the player shows). Only chart-level labels exist, so it
is trained by attention-MIL: chart logit_t = sum_i a_it s_it with a_it a softmax over rows of a gated score
g_it(h_i). The gate is part of the training signal only; a row's displayed score never uses another row.
"""
from __future__ import annotations

import math

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

from .columnist_rows import N_TOK, MASK_ID, N_GAPB, WINDOW  # noqa: E402
D_GATE = 16
RECEPTIVE = 4 * WINDOW + 8 + 1   # 4 attention layers + 2 stem convolutions of kernel 5


from .columnist_rows import chart_rows, map_rows  # noqa: E402,F401  (torch-free, re-exported)


def mirror_lut() -> torch.Tensor:
    lut = np.zeros(N_TOK, dtype=np.int64)
    for t in range(N_TOK):
        d = [(t >> (2 * c)) & 3 for c in range(4)]
        lut[t] = sum(d[3 - c] << (2 * c) for c in range(4))
    return torch.from_numpy(lut)


# ---------------------------------------------------------------------------------------------- network

def row_features(tok, gap, sv, valid):
    """tok (B,L) long, gap (B,L) ms, sv (B,L) -> (B,L,14) floats and times (B,L) s."""
    masked = tok == MASK_ID
    t = tok.clamp(max=N_TOK - 1)
    states = torch.stack([(t >> (2 * c)) & 3 for c in range(4)], -1)
    states = torch.where(masked[..., None], torch.zeros_like(states), states)
    delta = (states == 2).long() - (states == 3).long()
    held_before = (delta.cumsum(1) - delta).clamp(min=0).float().clamp(max=1)
    gap = torch.nan_to_num(gap.float(), nan=0.0, posinf=60000.0)
    g = torch.log1p(gap.clamp(min=0)) / 6.9
    g = torch.where(masked, torch.zeros_like(g), g)
    four = [f(g * (2.0 ** k) * math.pi) for k in range(4) for f in (torch.sin, torch.cos)]
    svf = torch.log(torch.nan_to_num(sv.float(), nan=1.0, posinf=10.0, neginf=0.1).clamp(0.1, 10.0))
    feats = torch.cat([g[..., None], torch.stack(four, -1), svf[..., None], held_before], -1)
    time = (gap.float().clamp(min=0) * (~masked) * valid).cumsum(1) / 1000.0
    return feats * valid[..., None], time


class Block(nn.Module):
    def __init__(self, d, heads, p):
        super().__init__()
        self.h = heads
        self.n1, self.n2 = nn.LayerNorm(d), nn.LayerNorm(d)
        self.qkv = nn.Linear(d, 3 * d)
        self.o = nn.Linear(d, d)
        self.ff = nn.Sequential(nn.Linear(d, 4 * d), nn.GELU(), nn.Linear(4 * d, d))
        self.drop = nn.Dropout(p)

    def forward(self, x, bias):
        B, L, D = x.shape
        q, k, v = self.qkv(self.n1(x)).view(B, L, 3, self.h, D // self.h).permute(2, 0, 3, 1, 4)
        a = F.scaled_dot_product_attention(q, k, v, attn_mask=bias.to(q.dtype))
        x = x + self.drop(self.o(a.transpose(1, 2).reshape(B, L, D)))
        return x + self.drop(self.ff(self.n2(x)))


class Columnist(nn.Module):
    def __init__(self, d=128, layers=4, heads=4, p=0.1):
        super().__init__()
        self.d, self.heads = d, heads
        self.tok = nn.Embedding(N_TOK + 1, d)
        self.num = nn.Linear(14, d)
        self.stem = nn.ModuleList([nn.Conv1d(d, 2 * d, 5) for _ in range(2)])
        self.blocks = nn.ModuleList(Block(d, heads, p) for _ in range(layers))
        self.norm = nn.LayerNorm(d)
        self.register_buffer("slopes", torch.tensor([0.05 * 2 ** (i * 5 / max(1, heads - 1)) for i in range(heads)]))
        self.tok_head = nn.Linear(d, N_TOK)
        self.gap_head = nn.Linear(d, N_GAPB)
        self.proj = nn.Sequential(nn.Linear(d, d), nn.GELU(), nn.Linear(d, 64))
        self.rank_head = nn.Linear(d, 1)
        self.order_head = nn.Linear(2 * d, 1)
        self.stat_head = nn.Linear(d, 4)
        self.rate_head = nn.Linear(2 * d, 1)

    def attn_bias(self, time, valid, causal):
        B, L = valid.shape
        idx = torch.arange(L, device=valid.device)
        rel = idx[None, :] - idx[:, None]
        allowed = rel.abs() <= WINDOW
        if causal:
            allowed = allowed & (rel <= 0)
        allowed = allowed[None] & valid[:, None, :]
        allowed = allowed | torch.eye(L, dtype=torch.bool, device=valid.device)[None]
        dt = (time[:, None, :] - time[:, :, None]).abs()
        bias = -self.slopes[None, :, None, None] * dt[:, None]
        return bias.masked_fill(~allowed[:, None], float("-inf"))

    def forward(self, tok, gap, sv, valid, causal=False, feats=None, time=None):
        """feats/time can be given (computed on the whole chart) to run a window of it with its true state."""
        if feats is None:
            feats, time = row_features(tok, gap, sv, valid)
        x = self.tok(tok) + self.num(feats)
        h = x.transpose(1, 2)
        for conv in self.stem:
            h = h + F.glu(conv(F.pad(h, (4, 0))), dim=1)
        x = h.transpose(1, 2)
        bias = self.attn_bias(time, valid, causal)
        for blk in self.blocks:
            x = blk(x, bias)
        return self.norm(x) * valid[..., None]


class LensColumnist(nn.Module):
    """Causal Columnist encoder + per-row task scores + the MIL gate that pools them to chart logits."""

    def __init__(self, n_tasks: int, d=128, layers=4, heads=4, p=0.1):
        super().__init__()
        self.enc = Columnist(d, layers, heads, p)
        self.n_tasks, self.d = n_tasks, d
        self.cls_w = nn.Parameter(torch.randn(n_tasks, d) / math.sqrt(d))
        self.cls_b = nn.Parameter(torch.zeros(n_tasks))
        self.V = nn.Linear(d, D_GATE * n_tasks)
        self.U = nn.Linear(d, D_GATE * n_tasks)
        self.gw = nn.Parameter(torch.randn(n_tasks, D_GATE) / math.sqrt(D_GATE))

    def load_pretrained(self, path: str, device="cpu"):
        ck = torch.load(path, map_location=device)
        self.enc.load_state_dict(ck["model"], strict=False)
        return ck.get("step")

    def rows(self, h):
        """h (B,L,d) -> per-row scores s (B,L,T) and gate logits g (B,L,T)."""
        s = h @ self.cls_w.T + self.cls_b
        a = (torch.tanh(self.V(h)) * torch.sigmoid(self.U(h))).view(*h.shape[:2], self.n_tasks, D_GATE)
        g = torch.einsum("bltk,tk->blt", a.float(), self.gw)
        return s, g

    @staticmethod
    def pool(s, g, valid):
        w = g.float().masked_fill(~valid[..., None], -1e4).softmax(1)
        return (w * s.float()).sum(1)

    def forward(self, tok, gap, sv, valid, feats=None, time=None):
        h = self.enc(tok, gap, sv, valid, causal=True, feats=feats, time=time)
        s, g = self.rows(h)
        return s, g


@torch.no_grad()
def chart_scores(net: LensColumnist, tok, gap, sv, dev, win=1024, ctx=RECEPTIVE + 16, amp=False):
    """Whole-chart per-row scores (L,T) and gate logits (L,T), by windows that carry `ctx` rows of left context.

    The features (open holds, running time) are computed once on the whole chart, and ctx covers the receptive field of
    the causal net, so this equals one pass over the whole chart."""
    net.eval()
    t_ = torch.as_tensor(np.asarray(tok), dtype=torch.long)[None]
    g_ = torch.as_tensor(np.asarray(gap, np.float32))[None]
    v_ = torch.as_tensor(np.asarray(sv, np.float32))[None]
    valid = torch.ones_like(t_, dtype=torch.bool)
    feats, time = row_features(t_, g_, v_, valid)
    L = t_.shape[1]
    new = win - ctx
    starts = list(range(0, L, new))
    S = torch.zeros(L, net.n_tasks)
    G = torch.zeros(L, net.n_tasks)
    bs = 8
    for b0 in range(0, len(starts), bs):
        part = starts[b0:b0 + bs]
        idx = []
        for a in part:
            lo = max(0, a - ctx)
            hi = min(L, a + new)
            idx.append((lo, a, hi))
        W = max(hi - lo for lo, _, hi in idx)
        T_ = torch.zeros(len(idx), W, dtype=torch.long)
        F_ = torch.zeros(len(idx), W, feats.shape[-1])
        Tm = torch.zeros(len(idx), W)
        Vl = torch.zeros(len(idx), W, dtype=torch.bool)
        for k, (lo, a, hi) in enumerate(idx):
            n = hi - lo
            T_[k, :n], F_[k, :n], Tm[k, :n], Vl[k, :n] = t_[0, lo:hi], feats[0, lo:hi], time[0, lo:hi], True
        T_, F_, Tm, Vl = (x.to(dev) for x in (T_, F_, Tm, Vl))
        with torch.autocast(device_type=dev.type, enabled=amp and dev.type == "cuda"):
            s, g = net(T_, None, None, Vl, feats=F_, time=Tm)
        s, g = s.float().cpu(), g.float().cpu()
        for k, (lo, a, hi) in enumerate(idx):
            S[a:hi], G[a:hi] = s[k, a - lo:hi - lo], g[k, a - lo:hi - lo]
    return S.numpy(), G.numpy()


def export(net: LensColumnist) -> dict:
    """Weights as numpy arrays for osu_patterns.note_model_columnist (no torch at serving time)."""
    sd = {k: v.detach().cpu().numpy().astype(np.float32) for k, v in net.state_dict().items()
          if not k.startswith(("enc.tok_head", "enc.gap_head", "enc.proj", "enc.rank_head", "enc.order_head",
                               "enc.stat_head", "enc.rate_head"))}
    sd["heads"] = np.array(net.enc.heads)
    sd["layers"] = np.array(len(net.enc.blocks))
    return sd
