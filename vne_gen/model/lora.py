"""Stage-specific low-rank adaptation (LoRA) [29] of the frozen GPT-2 backbone.

Each adapted projection computes :math:`W_z = W + B_z A_z` for the active stage :math:`z\\in\\{V,S\\}`
(virtual-node ordering / substrate-node placement).  The pretrained projection ``W`` stays frozen; only the
low-rank factors are trained.  ``shared=True`` replaces the two stage adapters by a single parameter-matched
adapter of twice the rank (the "shared adapter" ablation of Sec. VI-C).
"""

from __future__ import annotations

import math
from typing import Iterable, List

import torch
import torch.nn as nn

STAGES = ("V", "S")


class LoRAConv1D(nn.Module):
    """Wraps a Hugging Face ``Conv1D`` (``y = x @ W + b`` with ``W`` of shape ``(in, out)``)."""

    def __init__(self, base: nn.Module, rank: int, alpha: float, dropout: float = 0.0, shared: bool = False) -> None:
        super().__init__()
        self.base = base
        for p in self.base.parameters():
            p.requires_grad_(False)
        in_features, out_features = base.weight.shape
        self.shared = shared
        self.adapters: List[str] = ["shared"] if shared else list(STAGES)
        r = 2 * rank if shared else rank
        self.rank = r
        self.scaling = alpha / float(rank)
        self.A = nn.ParameterDict({z: nn.Parameter(torch.empty(r, in_features)) for z in self.adapters})
        self.B = nn.ParameterDict({z: nn.Parameter(torch.zeros(out_features, r)) for z in self.adapters})
        for z in self.adapters:
            nn.init.kaiming_uniform_(self.A[z], a=math.sqrt(5))
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
        self.active = self.adapters[0]

    def set_stage(self, stage: str) -> None:
        self.active = "shared" if self.shared else stage

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.base(x)
        A = self.A[self.active]
        B = self.B[self.active]
        lora = (self.dropout(x) @ A.t()) @ B.t()
        return out + lora * self.scaling


def inject_lora(gpt2: nn.Module, rank: int, alpha: float, dropout: float, shared: bool) -> List[LoRAConv1D]:
    """Replace the attention and MLP projections of every GPT-2 block by LoRA-adapted versions."""
    wrapped: List[LoRAConv1D] = []
    for block in gpt2.h:
        for parent, name in ((block.attn, "c_attn"), (block.attn, "c_proj"), (block.mlp, "c_fc"), (block.mlp, "c_proj")):
            base = getattr(parent, name)
            if isinstance(base, LoRAConv1D):
                continue
            mod = LoRAConv1D(base, rank, alpha, dropout, shared)
            setattr(parent, name, mod)
            wrapped.append(mod)
    return wrapped


def set_lora_stage(modules: Iterable[LoRAConv1D], stage: str) -> None:
    for m in modules:
        m.set_stage(stage)
