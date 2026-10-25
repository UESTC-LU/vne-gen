"""Model hyper-parameters (defaults follow the evaluated configuration of Sec. VI-A)."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field, fields
from typing import Any, Dict, Optional


@dataclass
class ModelConfig:
    # graph encoders
    gnn_hidden: int = 128  # 128-dimensional encoder outputs
    gnn_layers: int = 3  # three-layer GNNs
    message_passing: bool = True  # ablation: False -> node-wise MLP encoders only
    # causal backbone
    backbone: str = "gpt2"  # "gpt2" (causal Transformer) or "mlp" (GNN-Pointer behaviour-cloning baseline)
    pretrained: Optional[str] = "gpt2"  # Hugging Face name/path of the language-pretrained checkpoint; None = random init
    d_model: int = 768  # hidden size of the backbone (GPT-2 small)
    n_layer: int = 12  # used only for random initialisation
    n_head: int = 12
    context_blocks: int = 3  # K decision blocks in the causal context
    max_timesteps: int = 16  # decision-index embedding table size (>= max request size)
    # stage-specific LoRA adapters
    lora_rank: int = 64
    lora_alpha: float = 64.0
    lora_dropout: float = 0.05
    shared_adapter: bool = False  # ablation: one parameter-matched shared adapter instead of two stage adapters
    # pointer heads / conditioning
    pointer_dim: int = 128
    return_conditioning: bool = True  # ablation: False -> the return input is removed
    return_feat_dim: int = 32
    dropout: float = 0.1
    # feature scales (must match the data generation)
    cap_scale: float = 100.0
    demand_scale: float = 20.0
    max_request_nodes: int = 10

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "ModelConfig":
        names = {f.name for f in fields(cls)}
        return cls(**{k: v for k, v in d.items() if k in names})
