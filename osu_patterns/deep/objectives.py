"""Self-supervised objectives on the event encoder: masked events and masked windows.

* masked events: a share of events is replaced by a mask vector; the model predicts their head and
  tail columns and the gap to the previous event (pattern grammar, local scale);
* masked windows: a share of the 4-second window tokens is replaced before the transformer; the
  model predicts that window's summary statistics from the surrounding windows (structure).
"""
from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn

from . import tokens
from .model import EventEncoder

K3 = tokens.KMAX * 3
N_WSTAT = 8


def window_targets(batch) -> torch.Tensor:
    """(B, W, N_WSTAT) summary statistics of every window, from the encoder's own inputs."""
    x, m, win = batch["x"], batch["mask"], batch["win"]
    B, L, _ = x.shape
    W = int(batch["n_win"].max())
    cols = x[..., :K3].view(B, L, tokens.KMAX, 3)
    head, tail, held = cols[..., 0], cols[..., 1], cols[..., 2]
    prev_head = torch.cat([torch.zeros_like(head[:, :1]), head[:, :-1]], 1)
    nh = head.sum(-1)
    ev = torch.stack([
        torch.ones_like(nh),                                  # events
        nh,                                                   # notes
        (nh >= 2).float(),                                    # chord rows
        (head * prev_head).sum(-1).clamp(max=1),              # jack rows
        tail.sum(-1),                                         # LN releases
        held.sum(-1),                                         # columns held
        x[..., K3],                                           # log gap to previous event
        x[..., K3 + 2],                                       # log scroll speed
    ], -1)
    ev = ev * m[..., None]
    idx = (win.clamp(max=W - 1) + torch.arange(B, device=x.device)[:, None] * W).reshape(-1)
    s = torch.zeros(B * W, ev.shape[-1], device=x.device).index_add_(0, idx, ev.reshape(-1, ev.shape[-1]))
    n = s[:, :1].clamp(min=1)
    out = torch.cat([torch.log1p(s[:, :2]), s[:, 2:] / n], 1)  # counts in log, the rest per event
    return out.view(B, W, -1)


class Pretrainer(nn.Module):
    def __init__(self, enc: EventEncoder, p_window: float = 0.15):
        super().__init__()
        self.p_window = p_window
        self.enc = enc
        c = enc.c
        self.event_head = nn.Sequential(nn.Linear(c, c), nn.GELU(), nn.Linear(c, 2 * tokens.KMAX + 1))
        n_wstat = N_WSTAT + enc.inp.in_features - K3 - tokens.N_CONT  # + the visual channels, if any
        self.window_head = nn.Sequential(nn.Linear(c, c), nn.GELU(), nn.Linear(c, n_wstat))

    def forward(self, batch, p_event=0.15, p_window=None, return_hidden=False):
        p_window = self.p_window if p_window is None else p_window
        m = batch["mask"]
        masked = (torch.rand(m.shape, device=m.device) < p_event) & m
        W = int(batch["n_win"].max())
        mw = torch.rand(m.shape[0], W, device=m.device) < p_window
        h_w, wmask, h_e = self.enc(batch["x"], batch["keys"], m, batch["win"], batch["n_win"],
                                   masked=masked, masked_windows=mw, return_events=True)
        x = batch["x"]
        cols = x[..., :K3].view(*x.shape[:2], tokens.KMAX, 3)
        tgt_bits = torch.cat([cols[..., 0], cols[..., 1]], -1)
        pe = self.event_head(h_e[masked])
        valid_cols = (batch["col_valid"][:, None].expand(-1, x.shape[1], -1))[masked]
        valid_cols = torch.cat([valid_cols, valid_cols], -1)
        l_bits = (F.binary_cross_entropy_with_logits(pe[:, :-1], tgt_bits[masked], reduction="none")
                  * valid_cols).sum() / valid_cols.sum().clamp(min=1)
        l_gap = F.smooth_l1_loss(pe[:, -1], x[..., K3][masked])
        tw = window_targets(batch)
        tw = (tw - batch["w_mean"]) / batch["w_std"]
        sel = mw & wmask
        l_win = F.smooth_l1_loss(self.window_head(h_w[sel]), tw[sel]) if sel.any() else torch.zeros((), device=m.device)
        if return_hidden:  # window embeddings of the masked pass, for heads trained in the same step
            return (l_bits, l_gap, l_win), (h_w, wmask)
        return l_bits, l_gap, l_win


def add_col_valid(batch):
    off = (tokens.KMAX - batch["keys"]) // 2
    r = torch.arange(tokens.KMAX)[None]
    batch["col_valid"] = ((r >= off[:, None]) & (r < (off + batch["keys"])[:, None])).float()
    return batch
