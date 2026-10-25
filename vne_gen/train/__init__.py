"""Offline training of VNE-Gen (Sec. IV-C).

``Trainer`` and ``save_checkpoint`` require PyTorch and are imported lazily; ``TrainConfig`` and
``compute_return_table`` are torch-free.
"""

from .config import TrainConfig
from .return_table import compute_return_table

__all__ = ["compute_return_table", "TrainConfig", "Trainer", "save_checkpoint"]


def __getattr__(name):
    if name in ("Trainer", "save_checkpoint", "split_records"):
        from . import trainer

        return getattr(trainer, name)
    raise AttributeError(name)
