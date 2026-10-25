"""Request-level runtime and memory measurements (Sec. VI-G).

For each substrate (GEANT-40, Waxman-100, Waxman-500) and request-size bucket (2-4, 5-7, 8-10 nodes) every
method is timed on the same request instances and residual-capacity snapshots.  Latency statistics cover
successful calls only; the success-conditioned sequential rate is the inverse of the mean latency.
Checkpoint loading is excluded.  For VNE-Gen the timing covers graph-state construction, decoding, routing
and residual-resource updates but excludes cross-request candidate selection.
"""

from __future__ import annotations

import json
import logging
import os
import time
from dataclasses import asdict, dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from ..env.simulator import run_online
from ..heuristics.ranking import make_rank_solver
from ..problem.generators import WorkloadConfig, generate_workload, sample_request
from ..problem.substrate import CapacityProfile, Substrate
from .online_eval import build_substrate

logger = logging.getLogger(__name__)


@dataclass
class RuntimeConfig:
    substrates: Sequence[str] = ("geant", "waxman100", "waxman500")
    size_buckets: Sequence[Tuple[int, int]] = ((2, 4), (5, 7), (8, 10))
    calls_per_config: int = 100
    seed: int = 0
    warmup_arrivals: int = 200  # arrivals of a NEA-Rank warm-up run used to collect capacity snapshots
    warmup_lambda: Dict[str, float] = field(default_factory=lambda: {"geant": 0.06, "waxman100": 1.0, "waxman500": 8.0})


def _snapshots(substrate: Substrate, lam: float, num: int, seed: int) -> List[CapacityProfile]:
    """Residual capacity snapshots taken at arrivals of a warm-up run of NEA-Rank."""
    rng = np.random.default_rng(50_000 + seed)
    workload = generate_workload(rng, num, lam)
    snaps: List[CapacityProfile] = []
    run_online(substrate, workload, make_rank_solver("nea"), profile_hook=lambda req, prof: snaps.append(prof))
    return snaps or [substrate.base_profile()]


def _sync():
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.synchronize()
    except Exception:  # torch not installed: nothing to synchronise
        pass


def _gpu_memory() -> dict:
    try:
        import torch

        if torch.cuda.is_available():
            return {"allocated_mb": torch.cuda.memory_allocated() / 2**20,
                    "reserved_mb": torch.cuda.memory_reserved() / 2**20}
    except Exception:
        pass
    return {}


def measure_runtime(methods: Dict[str, object], cfg: RuntimeConfig, out_path: Optional[str] = None) -> dict:
    """Time ``methods[name].embed`` for every substrate and size bucket; returns latency statistics."""
    rows: List[dict] = []
    base_mem = _gpu_memory()
    for kind in cfg.substrates:
        substrate = build_substrate(kind, cfg.seed)
        snaps = _snapshots(substrate, cfg.warmup_lambda.get(kind, 0.06), cfg.warmup_arrivals, cfg.seed)
        for lo, hi in cfg.size_buckets:
            rng = np.random.default_rng(60_000 + cfg.seed + 100 * lo)
            # shared instances: (request, snapshot) pairs; keep sampling until each method has enough successes
            instances = [(sample_request(rng, i, num_nodes=int(rng.integers(lo, hi + 1))), snaps[int(rng.integers(len(snaps)))])
                         for i in range(cfg.calls_per_config * 6)]
            for name, method in methods.items():
                latencies: List[float] = []
                peak_increments: List[float] = []
                attempts = 0
                for req, prof in instances:
                    if len(latencies) >= cfg.calls_per_config:
                        break
                    attempts += 1
                    try:
                        import torch

                        if torch.cuda.is_available():
                            torch.cuda.reset_peak_memory_stats()
                            start_alloc = torch.cuda.memory_allocated()
                    except Exception:
                        start_alloc = None
                    _sync()
                    t0 = time.perf_counter()
                    mapping = method.embed(req, substrate, prof)
                    _sync()
                    dt = time.perf_counter() - t0
                    if mapping is not None:
                        latencies.append(dt)
                        if start_alloc is not None:
                            import torch

                            peak_increments.append((torch.cuda.max_memory_allocated() - start_alloc) / 2**20)
                lat = np.asarray(latencies)
                row = {
                    "method": name, "substrate": kind, "bucket": f"{lo}-{hi}", "successful_calls": int(lat.size),
                    "attempted_calls": attempts, "success_rate": lat.size / max(1, attempts),
                    "mean_latency_s": float(lat.mean()) if lat.size else float("nan"),
                    "p50_latency_s": float(np.percentile(lat, 50)) if lat.size else float("nan"),
                    "p99_latency_s": float(np.percentile(lat, 99)) if lat.size else float("nan"),
                    "sequential_rate_rps": float(1.0 / lat.mean()) if lat.size else float("nan"),
                    "peak_incremental_mb": float(np.mean(peak_increments)) if peak_increments else float("nan"),
                }
                rows.append(row)
                logger.info("%s | %s | %s nodes | mean %.4fs p99 %.4fs (%d calls)", name, kind, f"{lo}-{hi}",
                            row["mean_latency_s"], row["p99_latency_s"], lat.size)
    result = {"config": asdict(cfg), "post_load_gpu_memory": base_mem, "rows": rows}
    if out_path:
        os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
        with open(out_path, "w") as f:
            json.dump(result, f, indent=1, default=float)
    return result
