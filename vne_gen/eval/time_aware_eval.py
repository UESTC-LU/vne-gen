"""Time-aware offline VNE evaluation (Sec. VI-F): 50-request episodes with known activity intervals."""

from __future__ import annotations

import json
import logging
import os
from dataclasses import asdict, dataclass, field
from typing import Dict, List, Optional, Sequence

import numpy as np

from ..env.simulator import run_online
from ..problem.generators import WorkloadConfig, generate_workload
from ..selector.offline import CandidatePoolSelector, SelectorConfig
from .online_eval import build_substrate
from .stats import summarize_runs

logger = logging.getLogger(__name__)


@dataclass
class TimeAwareEvalConfig:
    substrate: str = "geant"
    episode_size: int = 50
    arrival_rates: Sequence[float] = (0.06, 0.08, 0.10)
    seeds: Sequence[int] = (0, 1, 2, 3, 4, 5, 6)
    rounds: int = 5
    schedule: Sequence = (1.0,)
    fixed_rounds: bool = False
    time_limit: Optional[float] = 300.0
    run_sequential: bool = True
    run_joint: bool = True
    workload: WorkloadConfig = field(default_factory=WorkloadConfig)
    reference: str = "NEA-Rank"


def episode_for(seed: int, lam: float, size: int, cfg: WorkloadConfig = WorkloadConfig()):
    rng = np.random.default_rng(40_000 + 1000 * seed + int(round(lam * 1000)))
    return generate_workload(rng, size, lam, cfg)


def evaluate_time_aware(sources: Dict[str, object], cfg: TimeAwareEvalConfig, out_path: Optional[str] = None) -> dict:
    runs: List[dict] = []
    for lam in cfg.arrival_rates:
        for seed in cfg.seeds:
            substrate = build_substrate(cfg.substrate, seed)
            requests = episode_for(seed, lam, cfg.episode_size, cfg.workload)
            for name, source in sources.items():
                if cfg.run_sequential and hasattr(source, "embed"):
                    res = run_online(substrate, requests, source, with_departures=True)
                    runs.append({"variant": "sequential", "source": name, "seed": seed, "lambda": lam,
                                 "revenue": res.total_revenue, "total_cost": res.total_cost, "r2c_lt": res.r2c_lt,
                                 "mean_r2c": res.mean_r2c, "num_selected": len(res.admitted)})
                    logger.info("sequential %s | lambda=%.2f seed=%d | revenue %.1f", name, lam, seed, res.total_revenue)
                if cfg.run_joint:
                    scfg = SelectorConfig(setting="ta", rounds=cfg.rounds, schedule=cfg.schedule,
                                          fixed_rounds=cfg.fixed_rounds, time_limit=cfg.time_limit, seed=seed)
                    out = CandidatePoolSelector(substrate, requests, source, scfg).run()
                    runs.append({"variant": "joint", "source": name, "seed": seed, "lambda": lam, **out.summary()})
                    logger.info("time-aware %s | lambda=%.2f seed=%d | revenue %.1f", name, lam, seed, out.revenue)
    summary = {}
    for lam in cfg.arrival_rates:
        for variant in ("sequential", "joint"):
            per = {}
            for name in sources:
                vals = [r["revenue"] for r in runs if r["variant"] == variant and r["source"] == name and r["lambda"] == lam]
                if vals:
                    per[name] = vals
            if per:
                summary[f"lambda={lam}/{variant}"] = summarize_runs(per, cfg.reference if cfg.reference in per else None)
        for name in sources:
            seq = [r["revenue"] for r in runs if r["variant"] == "sequential" and r["source"] == name and r["lambda"] == lam]
            joint = [r["revenue"] for r in runs if r["variant"] == "joint" and r["source"] == name and r["lambda"] == lam]
            if seq and joint and len(seq) == len(joint):
                summary[f"lambda={lam}/joint_vs_sequential/{name}"] = summarize_runs({"joint": joint, "sequential": seq}, "sequential")["joint"]
    result = {"config": {**asdict(cfg), "workload": asdict(cfg.workload)}, "runs": runs, "summary": summary}
    if out_path:
        os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
        with open(out_path, "w") as f:
            json.dump(result, f, indent=1, default=float)
    return result
