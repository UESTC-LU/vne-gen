"""Shared utilities: logging, seeding, YAML configs and solver/source factories used by the scripts."""

from __future__ import annotations

import logging
import os
import random
import sys
from typing import Any, Dict, List, Optional

import numpy as np
import yaml

from .heuristics.mcts import MCTSSolver
from .heuristics.ranking import make_rank_solver
from .model.config import ModelConfig
from .selector.proposal import PortfolioSource
from .train.config import TrainConfig

HEURISTIC_NAMES = {"grc": "GRC-Rank", "nrm": "NRM-Rank", "nea": "NEA-Rank", "mcts": "MCTS"}


def setup_logging(level: str = "INFO", log_file: Optional[str] = None) -> None:
    handlers: List[logging.Handler] = [logging.StreamHandler(sys.stdout)]
    if log_file:
        os.makedirs(os.path.dirname(os.path.abspath(log_file)), exist_ok=True)
        handlers.append(logging.FileHandler(log_file))
    logging.basicConfig(level=getattr(logging, level.upper(), logging.INFO),
                        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s", handlers=handlers, force=True)


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except ImportError:
        pass


def load_yaml(path: str) -> Dict[str, Any]:
    with open(path, "r") as f:
        return yaml.safe_load(f) or {}


def configs_from_yaml(path: str, overrides: Optional[Dict[str, Any]] = None):
    """Return ``(ModelConfig, TrainConfig, raw dict)`` from a YAML file with ``model:``/``train:`` sections."""
    raw = load_yaml(path)
    overrides = overrides or {}
    model_d = dict(raw.get("model", {}))
    train_d = dict(raw.get("train", {}))
    for k, v in overrides.items():
        if k.startswith("model."):
            model_d[k[6:]] = v
        elif k.startswith("train."):
            train_d[k[6:]] = v
    return ModelConfig.from_dict(model_d), TrainConfig.from_dict(train_d), raw


def parse_overrides(items: Optional[List[str]]) -> Dict[str, Any]:
    """Parse ``key=value`` overrides (values are YAML-parsed: ``train.lr=1e-4``, ``model.pretrained=null``)."""
    out: Dict[str, Any] = {}
    for item in items or []:
        if "=" not in item:
            raise ValueError(f"override '{item}' must have the form key=value")
        k, v = item.split("=", 1)
        out[k.strip()] = yaml.safe_load(v)
    return out


def build_baselines(names: List[str], seed: int = 0, mcts_simulations: int = 64) -> Dict[str, object]:
    """Instantiate heuristic baselines by short name (``grc``, ``nrm``, ``nea``, ``mcts``, ``portfolio``)."""
    out: Dict[str, object] = {}
    for n in names:
        key = n.strip().lower()
        if not key:
            continue
        if key == "mcts":
            out["MCTS"] = MCTSSolver(simulations=mcts_simulations, seed=seed)
        elif key == "portfolio":
            out["Portfolio"] = PortfolioSource([make_rank_solver(x) for x in ("grc", "nrm", "nea")])
        else:
            solver = make_rank_solver(key)
            out[solver.name] = solver
    return out


def load_generator_if_given(checkpoint: Optional[str], device: Optional[str] = None, **kwargs):
    if not checkpoint:
        return None
    from .generator.generator import load_generator  # imported lazily so that torch is optional for baselines

    return load_generator(checkpoint, device=device, **kwargs)
