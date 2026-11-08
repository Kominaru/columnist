"""Hierarchical event encoder with attention-MIL heads.

    events --(dilated 1D CNN)--> event features --(mean+max per WINDOW_S window)--> window tokens
           --(small transformer across windows)--> window embeddings h_w
           --(gated attention pooling, one per task, Ilse et al. 2018)--> chart logits

The attention weights over windows say where a category was found, which gives timestamps.
"""
from __future__ import annotations

import math

import torch
from torch import nn
import torch.nn.functional as F

from .tokens import KMAX, N_CONT

WINDOW_S = 4.0
MAX_WINDOWS = 256
D_IN = KMAX * 3 + N_CONT


class ConvBlock(nn.Module):
    def __init__(self, c: int, dilation: int, p_drop: float):
        super().__init__()
        self.conv = nn.Conv1d(c, 2 * c, 3, padding=dilation, dilation=dilation)
        self.norm = nn.GroupNorm(1, c)
        self.drop = nn.Dropout(p_drop)

    def forward(self, x, mask):  # x (B, C, L), mask (B, 1, L)
        h = self.conv(self.norm(x) * mask)
        h = F.glu(h, dim=1)
        return x + self.drop(h) * mask


class EventEncoder(nn.Module):
    def __init__(self, c: int = 64, n_conv: int = 6, n_tf: int = 2, p_drop: float = 0.1, d_in: int = D_IN):
        super().__init__()
        self.c = c
        self.inp = nn.Linear(d_in, c)
        self.keys = nn.Embedding(KMAX + 1, c)
        self.convs = nn.ModuleList(ConvBlock(c, 2 ** i, p_drop) for i in range(n_conv))
        self.win = nn.Sequential(nn.Linear(2 * c, c), nn.GELU(), nn.LayerNorm(c))
        self.pos = nn.Embedding(MAX_WINDOWS, c)
        layer = nn.TransformerEncoderLayer(c, 4, 2 * c, p_drop, batch_first=True, norm_first=True)
        self.n_tf = n_tf
        self.tf = nn.TransformerEncoder(layer, n_tf, enable_nested_tensor=False) if n_tf else None   # n_tf=0: no window transformer
        self.out_norm = nn.LayerNorm(c)
        self.mask_event = nn.Parameter(torch.zeros(c))   # used by masked pretraining only
        self.mask_window = nn.Parameter(torch.zeros(c))

    def events(self, x, keys, mask, masked=None):
        """Per-event features (B, L, C) after the convolution stack."""
        h = self.inp(x)
        if masked is not None:
            h = torch.where(masked[..., None], self.mask_event.expand_as(h), h)
        h = h + self.keys(keys)[:, None]
        h = h.transpose(1, 2)
        m = mask[:, None].float()
        for blk in self.convs:
            h = blk(h, m)
        return h.transpose(1, 2)

    def forward(self, x, keys, mask, win, n_win, masked=None, masked_windows=None, return_events=False):
        """x (B, L, D_IN); keys (B,); mask (B, L) bool; win (B, L) window index; n_win (B,).
        Returns window embeddings (B, W, C), window mask (B, W)."""
        h = self.events(x, keys, mask, masked)
        B, L, C = h.shape
        W = int(n_win.max())
        idx = (win.clamp(max=W - 1) + torch.arange(B, device=h.device)[:, None] * W).reshape(-1)
        m = mask.reshape(-1)
        flat = h.reshape(-1, C)
        s = torch.zeros(B * W, C, device=h.device).index_add_(0, idx[m], flat[m])
        cnt = torch.zeros(B * W, device=h.device).index_add_(0, idx[m], torch.ones_like(idx[m], dtype=torch.float))
        mean = s / cnt.clamp(min=1)[:, None]
        mx = torch.full((B * W, C), -1e4, device=h.device).index_reduce_(0, idx[m], flat[m], "amax")
        mx = torch.where(cnt[:, None] > 0, mx, torch.zeros_like(mx))
        w = self.win(torch.cat([mean, mx], 1)).reshape(B, W, C)
        wmask = (cnt > 0).reshape(B, W)
        if masked_windows is not None:
            w = torch.where(masked_windows[..., None], self.mask_window.expand_as(w), w)
        w = w + self.pos(torch.arange(W, device=h.device).clamp(max=MAX_WINDOWS - 1))[None]
        if self.tf is not None:
            w = self.tf(w, src_key_padding_mask=~wmask)
        if return_events:
            return self.out_norm(w), wmask, h
        return self.out_norm(w), wmask


class MILHeads(nn.Module):
    """One gated attention pool + linear classifier per binary task, plus a pooled multi-class head."""

    def __init__(self, c: int, n_tasks: int, n_classes: int = 4, d_att: int = 32):
        super().__init__()
        self.V = nn.Linear(c, d_att * n_tasks)
        self.U = nn.Linear(c, d_att * n_tasks)
        self.w = nn.Parameter(torch.randn(n_tasks, d_att) / math.sqrt(d_att))
        self.cls_w = nn.Parameter(torch.randn(n_tasks, c) / math.sqrt(c))
        self.cls_b = nn.Parameter(torch.zeros(n_tasks))
        self.n_tasks, self.d_att = n_tasks, d_att
        self.multi = nn.Linear(c, n_classes)

    def forward(self, h, wmask):
        B, W, C = h.shape
        g = torch.tanh(self.V(h)) * torch.sigmoid(self.U(h))           # (B, W, T*d)
        g = g.view(B, W, self.n_tasks, self.d_att)
        a = torch.einsum("bwtd,td->btw", g, self.w)
        a = a.masked_fill(~wmask[:, None], -1e4).softmax(-1)          # (B, T, W)
        z = torch.einsum("btw,bwc->btc", a, h)
        logits = (z * self.cls_w).sum(-1) + self.cls_b                  # (B, T)
        pooled = (h * wmask[..., None]).sum(1) / wmask.sum(1, keepdim=True).clamp(min=1)
        return logits, self.multi(pooled), a, pooled


class PatternNet(nn.Module):
    def __init__(self, n_tasks: int, c: int = 64, **kw):
        super().__init__()
        self.enc = EventEncoder(c, **kw)
        self.heads = MILHeads(c, n_tasks)

    def forward(self, batch):
        h, wmask = self.enc(batch["x"], batch["keys"], batch["mask"], batch["win"], batch["n_win"])
        return self.heads(h, wmask)
