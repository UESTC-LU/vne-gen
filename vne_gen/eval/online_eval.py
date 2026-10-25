"""Online VNE evaluation and generalization without retraining (Sec. VI-B, VI-D).

For every (seed, arrival rate) the same substrate and workload are supplied to all methods; admitted revenue
is measured over a fixed number of arrivals (1,000 by default).  Substrates: the fixed 40-node GEANT
topology (capacities sampled per seed), or one independently generated Waxman substrate per (size, seed).
"""

from __future__ import annotations

import json
import logging
import os
from dataclasses import asdict, dataclass, field
from typing import Callable, Dict, List, Optional, Sequence

import numpy as np

from ..env.simulator import Solver, run_online
from ..problem.generators import WorkloadConfig, generate_workload, load_geant, waxman_substrate
from ..problem.substrate import Substrate
from .stats import summarize_runs

logger = logging.getLogger(__name__)


@dataclass
class OnlineEvalConfig:
    substrate: str = "geant"  # "geant", "waxman100", "waxman500"
    arrival_rates: Sequence[float] = (0.04, 0.05, 0.06, 0.07, 0.08, 0.09)
    num_arrivals: int = 1000
    seeds: Sequence[int] = (0, 1, 2, 3, 4, 5, 6)
    workload: WorkloadConfig = field(default_factory=WorkloadConfig)
    reference: str = "NEA-Rank"


def build_substrate(kind: str, seed: int) -> Substrate:
    """Substrate for (kind, seed): capacities (and the Waxman topology) depend on the seed only."""
    rng = np.random.default_rng(10_000 + seed)
    if kind.lower() in ("geant", "geant40", "geant-40"):
        return load_geant(rng)
    if kind.lower().startswith("waxman"):
        n = int(kind.lower().replace("waxman", "").replace("-", "") or 100)
        return waxman_substrate(n, rng)
    raise ValueError(f"unknown substrate kind '{kind}'")


def workload_for(seed: int, lam: float, num_arrivals: int, cfg: WorkloadConfig = WorkloadConfig()):
    """Evaluation workload for (seed, lambda); evaluation seeds are disjoint from corpus-collection seeds."""
    rng = np.random.default_rng(20_000 + 1000 * seed + int(round(lam * 1000)))
    return generate_workload(rng, num_arrivals, lam, cfg)


def evaluate_online(
    solvers: Dict[str, Solver],
    cfg: OnlineEvalConfig,
    out_path: Optional[str] = None,
    solver_factory: Optional[Callable[[str, int], Solver]] = None,
) -> dict:
    """Run every solver on every (seed, lambda) and return per-run metrics plus seed-level summaries.

    ``solver_factory(name, seed)`` (optional) builds a fresh solver per seed (e.g. to reseed stochastic
    baselines); otherwise the solver objects in ``solvers`` are reused.
    """
    runs: List[dict] = []
    for lam in cfg.arrival_rates:
        for seed in cfg.seeds:
            substrate = build_substrate(cfg.substrate, seed)
            workload = workload_for(seed, lam, cfg.num_arrivals, cfg.workload)
            for name in solvers:
                solver = solver_factory(name, seed) if solver_factory else solvers[name]
                res = run_online(substrate, workload, solver, num_arrivals=cfg.num_arrivals)
                entry = res.summary()
                entry.update({"solver": name, "seed": seed, "lambda": lam, "substrate": cfg.substrate})
                runs.append(entry)
                logger.info("%s | lambda %.3f | seed %d | revenue %.1f | admitted %d | R2C-LT %.3f",
                            name, lam, seed, res.total_revenue, len(res.admitted), res.r2c_lt)
    summary = {}
    for lam in cfg.arrival_rates:
        per_seed = {name: [r["total_revenue"] for r in runs if r["solver"] == name and r["lambda"] == lam] for name in solvers}
        summary[str(lam)] = summarize_runs(per_seed, cfg.reference if cfg.reference in solvers else None)
    result = {"config": {**asdict(cfg), "workload": asdict(cfg.workload)}, "runs": runs, "summary": summary}
    if out_path:
        os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
        with open(out_path, "w") as f:
            json.dump(result, f, indent=1)
    return result
