"""Offline training loop (Sec. IV-C).

The two pointer stages are trained by negative log-likelihood over the emitted action pairs of the corpus,
combined with learned uncertainty weights [30]:

.. math:: \\mathcal L_{total} = \\frac{\\mathcal L_V}{2e^{2\\alpha_V}} + \\frac{\\mathcal L_S}{2e^{2\\alpha_S}} + \\alpha_V + \\alpha_S .

The paper's configuration uses AdamW with learning rate 1e-4, weight decay 1e-4, 2,000 warm-up steps and
gradient clipping at 0.5; only the GNN encoders, token projections, pointer heads and LoRA factors are
trainable, the language-pretrained backbone stays frozen.  Once the corpus is collected, training requires
no interaction with the VNE environment.
"""

from __future__ import annotations

import json
import logging
import math
import os
import time
from dataclasses import asdict, dataclass, field, fields
from typing import Any, Dict, List, Optional, Sequence

import numpy as np
import torch
from torch.utils.data import DataLoader

from ..data.dataset import Example, TrajectoryDataset, collate_examples
from ..data.features import FeatureConfig
from ..model.config import ModelConfig
from ..model.vne_gen import VNEGenModel, batch_to_tensors
from .config import TrainConfig

logger = logging.getLogger(__name__)


class _Collate:
    """Picklable collate function (works with spawn-based DataLoader workers as well)."""

    def __init__(self, K: int) -> None:
        self.K = int(K)

    def __call__(self, examples: Sequence[Example]) -> Dict[str, np.ndarray]:
        batch = collate_examples(examples, self.K)
        batch["num_steps"] = np.asarray([ex.num_steps for ex in examples], dtype=np.int64)
        return batch


def _make_collate(K: int) -> _Collate:
    return _Collate(K)


def save_checkpoint(path: str, model: VNEGenModel, return_table: Dict[int, float], train_cfg: TrainConfig,
                    feature_cfg: FeatureConfig, extra: Optional[dict] = None) -> None:
    payload = {
        "model_state": model.state_dict(),
        "model_config": model.cfg.to_dict(),
        "backbone_config": model.backbone.config_dict(),
        "return_table": {int(k): float(v) for k, v in return_table.items()},
        "train_config": train_cfg.to_dict(),
        "feature_config": asdict(feature_cfg),
        "extra": extra or {},
    }
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    torch.save(payload, path)


class Trainer:
    def __init__(self, model: VNEGenModel, cfg: TrainConfig, device: Optional[torch.device] = None,
                 out_dir: str = "runs/vne_gen") -> None:
        self.model = model
        self.cfg = cfg
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
        self.out_dir = out_dir
        os.makedirs(out_dir, exist_ok=True)
        params = model.trainable_parameters()
        self.optimizer = torch.optim.AdamW(params, lr=cfg.lr, weight_decay=cfg.weight_decay)
        self.step = 0
        self.history: List[Dict[str, float]] = []

    def _lr_factor(self, total_steps: int) -> float:
        s = self.step
        if s < self.cfg.warmup_steps:
            return (s + 1) / float(self.cfg.warmup_steps)
        if self.cfg.schedule == "cosine" and total_steps > self.cfg.warmup_steps:
            progress = (s - self.cfg.warmup_steps) / float(max(1, total_steps - self.cfg.warmup_steps))
            return 0.5 * (1.0 + math.cos(math.pi * min(1.0, progress)))
        return 1.0

    def _run_batch(self, batch: Dict[str, np.ndarray], train: bool) -> Dict[str, float]:
        tb = batch_to_tensors(batch, self.device)
        with torch.autocast(device_type=self.device.type, enabled=self.cfg.amp and self.device.type == "cuda"):
            out = self.model(tb)
            losses = self.model.losses(out, tb, per_trajectory=self.cfg.per_trajectory_loss)
        if train:
            self.optimizer.zero_grad(set_to_none=True)
            losses["loss"].backward()
            if self.cfg.grad_clip and self.cfg.grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(self.model.trainable_parameters(), self.cfg.grad_clip)
            self.optimizer.step()
        return {k: float(v.detach()) for k, v in losses.items()}

    @torch.no_grad()
    def evaluate(self, loader: DataLoader, max_batches: Optional[int] = 50) -> Dict[str, float]:
        self.model.eval()
        agg: Dict[str, List[float]] = {}
        for i, batch in enumerate(loader):
            if max_batches is not None and i >= max_batches:
                break
            res = self._run_batch(batch, train=False)
            for k, v in res.items():
                agg.setdefault(k, []).append(v)
        self.model.train()
        return {k: float(np.mean(v)) for k, v in agg.items()}

    def fit(self, dataset: TrajectoryDataset, return_table: Dict[int, float], feature_cfg: FeatureConfig,
            val_dataset: Optional[TrajectoryDataset] = None, checkpoint_name: str = "vne_gen.pt") -> str:
        cfg = self.cfg
        torch.manual_seed(cfg.seed)
        collate = _make_collate(self.model.K)
        loader = DataLoader(dataset, batch_size=cfg.batch_size, shuffle=True, collate_fn=collate,
                            num_workers=cfg.num_workers, drop_last=True, persistent_workers=cfg.num_workers > 0)
        val_loader = None
        if val_dataset is not None and len(val_dataset) > 0:
            val_loader = DataLoader(val_dataset, batch_size=cfg.batch_size, shuffle=False, collate_fn=collate,
                                    num_workers=min(cfg.num_workers, 2))
        steps_per_epoch = max(1, len(loader))
        total_steps = cfg.max_steps if cfg.max_steps is not None else cfg.epochs * steps_per_epoch
        ckpt_path = os.path.join(self.out_dir, checkpoint_name)
        self.model.train()
        t0 = time.time()
        done = False
        epoch = 0
        while not done:
            for batch in loader:
                for g in self.optimizer.param_groups:
                    g["lr"] = cfg.lr * self._lr_factor(total_steps)
                res = self._run_batch(batch, train=True)
                self.step += 1
                if self.step % cfg.log_every == 0:
                    res.update({"step": self.step, "epoch": epoch, "lr": self.optimizer.param_groups[0]["lr"],
                                "elapsed_s": time.time() - t0})
                    self.history.append(res)
                    logger.info("step %d | loss %.4f | L_V %.4f (acc %.3f) | L_S %.4f (acc %.3f) | lr %.2e",
                                self.step, res["loss"], res["L_V"], res["acc_V"], res["L_S"], res["acc_S"], res["lr"])
                if val_loader is not None and self.step % cfg.eval_every == 0:
                    val = self.evaluate(val_loader)
                    val.update({"step": self.step, "split": "val"})
                    self.history.append(val)
                    logger.info("  val | L_V %.4f (acc %.3f) | L_S %.4f (acc %.3f)", val["L_V"], val["acc_V"], val["L_S"], val["acc_S"])
                    save_checkpoint(ckpt_path, self.model, return_table, cfg, feature_cfg,
                                    extra={"step": self.step, "val": val})
                if self.step >= total_steps:
                    done = True
                    break
            epoch += 1
        final_val = self.evaluate(val_loader) if val_loader is not None else {}
        save_checkpoint(ckpt_path, self.model, return_table, cfg, feature_cfg,
                        extra={"step": self.step, "val": final_val, "train_time_s": time.time() - t0})
        with open(os.path.join(self.out_dir, "history.json"), "w") as f:
            json.dump(self.history, f, indent=1)
        return ckpt_path


def split_records(records: Sequence, val_fraction: float, seed: int):
    rng = np.random.default_rng(seed)
    idx = rng.permutation(len(records))
    n_val = int(round(len(records) * val_fraction))
    val_idx = set(idx[:n_val].tolist())
    train = [r for i, r in enumerate(records) if i not in val_idx]
    val = [r for i, r in enumerate(records) if i in val_idx]
    return train, val
