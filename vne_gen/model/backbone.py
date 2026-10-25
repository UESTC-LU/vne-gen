"""Causal Transformer backbone (GPT-2 small, language-pretrained) with stage-specific LoRA adapters."""

from __future__ import annotations

import logging
from typing import List, Optional

import torch
import torch.nn as nn

from .config import ModelConfig
from .lora import LoRAConv1D, inject_lora, set_lora_stage

logger = logging.getLogger(__name__)


class CausalBackbone(nn.Module):
    """Shared causal Transformer; ``forward`` consumes token embeddings and returns hidden states."""

    def __init__(self, cfg: ModelConfig, backbone_config: Optional[dict] = None) -> None:
        super().__init__()
        self.cfg = cfg
        self.kind = cfg.backbone
        self.lora_modules: List[LoRAConv1D] = []
        if cfg.backbone == "gpt2":
            from transformers import GPT2Config, GPT2Model

            model = None
            if backbone_config is not None:  # rebuild the exact architecture stored in a checkpoint
                model = GPT2Model(GPT2Config.from_dict(dict(backbone_config)))
            elif cfg.pretrained:
                try:
                    try:
                        model = GPT2Model.from_pretrained(cfg.pretrained, attn_implementation="eager")
                    except TypeError:  # older transformers without the attn_implementation argument
                        model = GPT2Model.from_pretrained(cfg.pretrained)
                except Exception as exc:  # offline machine / unknown name: fall back to random init
                    logger.warning("could not load pretrained backbone '%s' (%s); using random initialisation",
                                   cfg.pretrained, exc)
            if model is None:
                config = GPT2Config(vocab_size=8, n_positions=max(64, 4 * cfg.context_blocks), n_embd=cfg.d_model,
                                    n_layer=cfg.n_layer, n_head=cfg.n_head, attn_implementation="eager")
                model = GPT2Model(config)
            self.gpt = model
            self.d_model = int(model.config.n_embd)
            for p in self.gpt.parameters():  # the original backbone parameters remain frozen
                p.requires_grad_(False)
            self.lora_modules = inject_lora(self.gpt, cfg.lora_rank, cfg.lora_alpha, cfg.lora_dropout, cfg.shared_adapter)
        elif cfg.backbone == "mlp":
            self.d_model = cfg.d_model
            self.gpt = None
        else:
            raise ValueError(f"unknown backbone '{cfg.backbone}'")

    def set_stage(self, stage: str) -> None:
        set_lora_stage(self.lora_modules, stage)

    def config_dict(self) -> Optional[dict]:
        """Architecture of the sequence model (stored in checkpoints so they load without downloads)."""
        if self.gpt is None:
            return None
        d = self.gpt.config.to_dict()
        d.pop("_attn_implementation_internal", None)
        d.pop("_attn_implementation", None)
        d["attn_implementation"] = "eager"
        return d

    def forward(self, tokens: torch.Tensor, mask: torch.Tensor, stage: str) -> torch.Tensor:
        """``tokens``: (B, L, d); ``mask``: (B, L) bool (True = real token). Returns (B, L, d)."""
        if self.gpt is None:
            raise RuntimeError("the MLP backbone has no sequence model; use VNEGenModel's MLP path")
        self.set_stage(stage)
        position_ids = (mask.long().cumsum(dim=1) - 1).clamp(min=0)
        out = self.gpt(inputs_embeds=tokens, attention_mask=mask.long(), position_ids=position_ids)
        return out.last_hidden_state
