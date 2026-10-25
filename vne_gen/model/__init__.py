"""VNE-Gen architecture (Sec. IV-B): dual GNN encoders, structured token blocks, a return-conditioned causal
Transformer backbone with stage-specific LoRA adapters, and two feasibility-masked pointer heads.

``VNEGenModel`` requires PyTorch and is imported lazily; ``ModelConfig`` is torch-free.
"""

from .config import ModelConfig

__all__ = ["ModelConfig", "VNEGenModel", "batch_to_tensors"]


def __getattr__(name):
    if name in ("VNEGenModel", "batch_to_tensors"):
        from . import vne_gen as _m

        return getattr(_m, name)
    raise AttributeError(name)
