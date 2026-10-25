"""Offline VNE evaluation (Sec. VI-E): sequential commitment vs. joint selection, proposal sources,
candidate-pool expansion and selector-derived capacity profiles."""

from __future__ import annotations

import json
import logging
import os
from dataclasses import asdict, dataclass, field
from typing import Dict, List, Optional, Sequence

import numpy as np

from ..env.simulator import run_online
from ..problem.generators import WorkloadConfig, generate_request_set
from ..selector.offline import CandidatePoolSelector, SelectorConfig
from .online_eval import build_substrate
from .stats import summarize_runs

logger = logging.getLogger(__name__)


@dataclass
class OfflineEvalConfig:
    substrate: str = "geant"
    request_set_sizes: Sequence[int] = (50, 100, 200, 400)
    seeds: Sequence[int] = (0, 1, 2, 3, 4, 5, 6)
    rounds: int = 5
    schedule: Sequence = (1.0,)  # S_i = (1.0) in every round; use ((1.0, 1.0),) for the single-round control
    fixed_rounds: bool = False
    time_limit: Optional[float] = 300.0
    profile_mode: str = "selector"  # or "base"
    run_sequential: bool = True
    run_joint: bool = True
    workload: WorkloadConfig = field(default_factory=WorkloadConfig)
    reference: str = "NEA-Rank"


def request_set_for(seed: int, size: int, cfg: WorkloadConfig = WorkloadConfig()):
    rng = np.random.default_rng(30_000 + 1000 * seed + size)
    return generate_request_set(rng, size, cfg)


def evaluate_offline(sources: Dict[str, object], cfg: OfflineEvalConfig, out_path: Optional[str] = None) -> dict:
    """``sources`` maps a name to a proposal source (``propose``) that may also offer ``embed`` for sequential use."""
    runs: List[dict] = []
    for N in cfg.request_set_sizes:
        for seed in cfg.seeds:
            substrate = build_substrate(cfg.substrate, seed)
            requests = request_set_for(seed, N, cfg.workload)
            for name, source in sources.items():
                if cfg.run_sequential and hasattr(source, "embed"):
                    res = run_online(substrate, requests, source, with_departures=False)
                    entry = {"variant": "sequential", "source": name, "seed": seed, "N": N,
                             "revenue": res.total_revenue, "total_cost": res.total_cost, "r2c_lt": res.r2c_lt,
                             "mean_r2c": res.mean_r2c, "num_selected": len(res.admitted)}
                    runs.append(entry)
                    logger.info("sequential %s | N=%d seed=%d | revenue %.1f", name, N, seed, res.total_revenue)
                if cfg.run_joint:
                    scfg = SelectorConfig(setting="off", rounds=cfg.rounds, schedule=cfg.schedule,
                                          fixed_rounds=cfg.fixed_rounds, time_limit=cfg.time_limit,
                                          profile_mode=cfg.profile_mode, seed=seed)
                    out = CandidatePoolSelector(substrate, requests, source, scfg).run()
                    entry = {"variant": "joint", "source": name, "seed": seed, "N": N, **out.summary()}
                    runs.append(entry)
                    logger.info("joint %s | N=%d seed=%d | revenue %.1f (%d rounds)", name, N, seed, out.revenue, len(out.rounds))
    summary = {}
    for N in cfg.request_set_sizes:
        for variant in ("sequential", "joint"):
            per = {}
            for name in sources:
                vals = [r["revenue"] for r in runs if r["variant"] == variant and r["source"] == name and r["N"] == N]
                if vals:
                    per[name] = vals
            if per:
                summary[f"N={N}/{variant}"] = summarize_runs(per, cfg.reference if cfg.reference in per else None)
        # joint selection vs sequential commitment of the same source
        for name in sources:
            seq = [r["revenue"] for r in runs if r["variant"] == "sequential" and r["source"] == name and r["N"] == N]
            joint = [r["revenue"] for r in runs if r["variant"] == "joint" and r["source"] == name and r["N"] == N]
            if seq and joint and len(seq) == len(joint):
                summary[f"N={N}/joint_vs_sequential/{name}"] = summarize_runs({"joint": joint, "sequential": seq}, "sequential")["joint"]
    result = {"config": {**asdict(cfg), "workload": asdict(cfg.workload)}, "runs": runs, "summary": summary}
    if out_path:
        os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
        with open(out_path, "w") as f:
            json.dump(result, f, indent=1, default=float)
    return result
