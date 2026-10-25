"""Training hyper-parameters (torch-free so that baseline-only scripts can import it)."""

from __future__ import annotations

from dataclasses import asdict, dataclass, fields
from typing import Any, Dict, Optional


@dataclass
class TrainConfig:
    batch_size: int = 64
    epochs: int = 10
    max_steps: Optional[int] = None  # overrides epochs when set
    lr: float = 1e-4
    weight_decay: float = 1e-4
    warmup_steps: int = 2000
    schedule: str = "constant"  # "constant" or "cosine" after warm-up
    grad_clip: float = 0.5
    num_workers: int = 4
    val_fraction: float = 0.02
    log_every: int = 50
    eval_every: int = 1000
    seed: int = 0
    per_trajectory_loss: bool = False  # weight transitions by 1/T_tau (expectation over trajectories)
    completed_only: bool = False  # ablation: drop terminated mapping attempts from the corpus
    amp: bool = False

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "TrainConfig":
        names = {f.name for f in fields(cls)}
        return cls(**{k: v for k, v in d.items() if k in names})
