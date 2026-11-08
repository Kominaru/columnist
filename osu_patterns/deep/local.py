"""Note-level pattern model: a local event encoder whose per-event scores pool to the chart label.

    events --(linear)--> --(dilated 1D conv blocks, receptive field ~77 events)--> per-event features
           --(linear)--> per-event logits, one per task          (what the visualizer shows per note)
           --(auto-pool over events, per task, McFee et al. 2018)--> chart logits  (what the labels supervise)

Only chart-level labels exist, so this is multiple-instance learning. The chart logit is a softmax-
weighted mean of the event logits, w_i ~ exp(alpha_t * s_i): alpha_t -> 0 is a plain mean ("the whole
chart is X"), large alpha_t a max ("somewhere the chart is X"); alpha_t is learnt per task. Because
the chart score is a function of the event scores alone, an event scoring high for jumpstream means
"the notes around here look like what makes charts jumpstream", which is what the player shows.

There is no transformer or global context: an event only sees about +-38 events around it, so its
score can't lean on the rest of the chart. With causal=True the convolutions only look back (about
76 events), so an event's score depends on what has been played up to it and nothing after: the
player's view of the chart, for the live visualizer. Inference runs in numpy (osu_patterns.note_model).
"""
from __future__ import annotations

import torch
from torch import nn
import torch.nn.functional as F

DILATIONS = (1, 2, 4, 8, 16, 1, 2, 4)


class Block(nn.Module):
    def __init__(self, c: int, dilation: int, p_drop: float, causal: bool = False):
        super().__init__()
        self.norm = nn.LayerNorm(c)
        self.conv = nn.Conv1d(c, 2 * c, 3, padding=0 if causal else dilation, dilation=dilation)
        self.drop = nn.Dropout(p_drop)
        self.left = 2 * dilation if causal else 0   # causal: taps at t - 2d, t - d, t

    def forward(self, h, m):  # h (B, L, C), m (B, L, 1)
        x = (self.norm(h) * m).transpose(1, 2)
        if self.left:
            x = F.pad(x, (self.left, 0))
        x = F.glu(self.conv(x), dim=1).transpose(1, 2)
        return h + self.drop(x) * m


class LocalNet(nn.Module):
    def __init__(self, d_in: int, n_tasks: int, c: int = 64, p_drop: float = 0.1, dilations=DILATIONS,
                 causal: bool = False):
        super().__init__()
        self.inp = nn.Linear(d_in, c)
        self.blocks = nn.ModuleList(Block(c, d, p_drop, causal) for d in dilations)
        self.causal = causal
        self.norm = nn.LayerNorm(c)
        self.out = nn.Linear(c, n_tasks)
        self.alpha = nn.Parameter(torch.ones(n_tasks))
        self.dilations = tuple(dilations)

    def events(self, x, mask):
        """Per-event logits (B, L, T)."""
        m = mask[..., None].float()
        h = self.inp(x) * m
        for b in self.blocks:
            h = b(h, m)
        return self.out(self.norm(h))

    def pool(self, s, mask):
        """Auto-pool event logits s (B, L, T) to chart logits (B, T)."""
        a = self.alpha.clamp(0.0, 8.0)
        w = (s * a).masked_fill(~mask[..., None], -1e9).softmax(1)
        return (w * s).sum(1)

    def forward(self, x, mask):
        s = self.events(x, mask)
        return self.pool(s, mask), s

    def export(self) -> dict:
        """Weights as numpy arrays for osu_patterns.note_model."""
        sd = {k: v.detach().cpu().numpy() for k, v in self.state_dict().items()}
        sd["dilations"] = torch.tensor(self.dilations).numpy()
        sd["causal"] = torch.tensor(int(self.causal)).numpy()
        return sd
