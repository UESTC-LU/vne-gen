"""Message-passing graph encoders over packed (PyG-style) graph batches.

A layer computes edge messages from the receiving node, the sending node and the edge features, mean-
aggregates them at the receiving node and applies a residual gated update with layer normalisation.
With ``message_passing=False`` (ablation) the aggregation is skipped and every node is encoded by a
node-wise MLP only.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


def scatter_mean(x: torch.Tensor, index: torch.Tensor, num_groups: int) -> torch.Tensor:
    out = x.new_zeros((num_groups, x.size(-1))).index_add_(0, index, x)
    cnt = x.new_zeros((num_groups,)).index_add_(0, index, x.new_ones(index.size(0)))
    return out / cnt.clamp(min=1.0).unsqueeze(-1)


def scatter_max(x: torch.Tensor, index: torch.Tensor, num_groups: int) -> torch.Tensor:
    out = x.new_full((num_groups, x.size(-1)), float("-inf"))
    out = out.scatter_reduce(0, index.unsqueeze(-1).expand_as(x), x, reduce="amax", include_self=True)
    return torch.where(torch.isinf(out), torch.zeros_like(out), out)


def masked_mean(x: torch.Tensor, index: torch.Tensor, mask: torch.Tensor, num_groups: int) -> torch.Tensor:
    """Mean of ``x`` over the rows selected by ``mask`` within each group (zeros for empty groups)."""
    sel = mask.nonzero(as_tuple=True)[0]
    out = x.new_zeros((num_groups, x.size(-1)))
    cnt = x.new_zeros((num_groups,))
    if sel.numel():
        out = out.index_add_(0, index[sel], x[sel])
        cnt = cnt.index_add_(0, index[sel], x.new_ones(sel.size(0)))
    return out / cnt.clamp(min=1.0).unsqueeze(-1)


class MessagePassingLayer(nn.Module):
    def __init__(self, hidden: int, edge_dim: int, dropout: float = 0.1, message_passing: bool = True) -> None:
        super().__init__()
        self.message_passing = message_passing
        self.msg = nn.Sequential(nn.Linear(2 * hidden + edge_dim, hidden), nn.GELU(), nn.Linear(hidden, hidden))
        self.upd = nn.Sequential(nn.Linear(2 * hidden, hidden), nn.GELU(), nn.Dropout(dropout), nn.Linear(hidden, hidden))
        self.norm = nn.LayerNorm(hidden)

    def forward(self, h: torch.Tensor, edge_index: torch.Tensor, edge_attr: torch.Tensor) -> torch.Tensor:
        if self.message_passing and edge_index.numel() > 0:
            src, dst = edge_index[0], edge_index[1]
            m = self.msg(torch.cat([h[dst], h[src], edge_attr], dim=-1))
            agg = scatter_mean(m, dst, h.size(0))
        else:
            agg = torch.zeros_like(h)
        return self.norm(h + self.upd(torch.cat([h, agg], dim=-1)))


class GraphEncoder(nn.Module):
    """Three-layer (by default) message-passing encoder producing node embeddings of size ``hidden``."""

    def __init__(self, in_dim: int, edge_dim: int, hidden: int = 128, num_layers: int = 3, dropout: float = 0.1,
                 message_passing: bool = True) -> None:
        super().__init__()
        self.inp = nn.Sequential(nn.Linear(in_dim, hidden), nn.GELU(), nn.Linear(hidden, hidden))
        self.norm0 = nn.LayerNorm(hidden)
        self.layers = nn.ModuleList(
            [MessagePassingLayer(hidden, edge_dim, dropout, message_passing) for _ in range(num_layers)]
        )
        self.hidden = hidden

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor, edge_attr: torch.Tensor) -> torch.Tensor:
        h = self.norm0(self.inp(x))
        for layer in self.layers:
            h = layer(h, edge_index, edge_attr)
        return h
