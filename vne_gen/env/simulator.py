"""Event-driven online VNE simulator (Sec. III-B, VI-A).

Requests arrive sequentially.  At each arrival the solver observes the request and the current residual
capacity profile and returns either one validated mapping or ``None``.  A returned mapping is committed
immediately (without advancing simulation time for computation); resources are released at the request's
departure.  The same substrate, arrivals and departures are supplied to every method within a comparison.
"""

from __future__ import annotations

import heapq
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Protocol, Sequence, Tuple

import numpy as np

from ..problem.feasibility import ValidationError, validate_mapping
from ..problem.mapping import Allocation, Mapping, long_term_r2c, mapping_cost, r2c
from ..problem.request import Request, request_revenue
from ..problem.substrate import CapacityProfile, Substrate


class Solver(Protocol):
    """Interface implemented by every online method (VNE-Gen, ranking heuristics, external baselines)."""

    name: str

    def embed(self, request: Request, substrate: Substrate, profile: CapacityProfile) -> Optional[Mapping]:
        """Return a complete mapping feasible under ``profile`` or ``None``."""


class SubstrateState:
    """Residual substrate resources of an online run (the unallocated substrate resources)."""

    def __init__(self, substrate: Substrate) -> None:
        self.substrate = substrate
        self.residual = substrate.base_profile()
        self.active: Dict[int, Allocation] = {}

    def profile(self) -> CapacityProfile:
        return self.residual.copy()

    def commit(self, request: Request, mapping: Mapping) -> Allocation:
        alloc = mapping.allocation(request, self.substrate)
        self.residual.node -= alloc.node
        self.residual.link -= alloc.link
        self.active[request.id] = alloc
        return alloc

    def release(self, request_id: int) -> None:
        alloc = self.active.pop(request_id, None)
        if alloc is not None:
            self.residual.node += alloc.node
            self.residual.link += alloc.link


@dataclass
class OnlineResult:
    solver: str
    num_arrivals: int
    admitted: List[int] = field(default_factory=list)
    revenues: List[float] = field(default_factory=list)
    costs: List[float] = field(default_factory=list)
    r2cs: List[float] = field(default_factory=list)
    mappings: Dict[int, Mapping] = field(default_factory=dict)
    latencies: List[float] = field(default_factory=list)
    validation_failures: int = 0

    @property
    def total_revenue(self) -> float:
        return float(sum(self.revenues))

    @property
    def total_cost(self) -> float:
        return float(sum(self.costs))

    @property
    def acceptance_ratio(self) -> float:
        return len(self.admitted) / self.num_arrivals if self.num_arrivals else 0.0

    @property
    def mean_r2c(self) -> float:
        return float(np.mean(self.r2cs)) if self.r2cs else 0.0

    @property
    def r2c_lt(self) -> float:
        return long_term_r2c(self.revenues, self.costs)

    def summary(self) -> dict:
        return {
            "solver": self.solver,
            "num_arrivals": self.num_arrivals,
            "num_admitted": len(self.admitted),
            "acceptance_ratio": self.acceptance_ratio,
            "total_revenue": self.total_revenue,
            "total_cost": self.total_cost,
            "mean_r2c": self.mean_r2c,
            "r2c_lt": self.r2c_lt,
            "mean_latency_s": float(np.mean(self.latencies)) if self.latencies else 0.0,
            "validation_failures": self.validation_failures,
        }


def run_online(
    substrate: Substrate,
    requests: Sequence[Request],
    solver: Solver,
    num_arrivals: Optional[int] = None,
    with_departures: bool = True,
    keep_mappings: bool = False,
    profile_hook=None,
) -> OnlineResult:
    """Run an online simulation and return admitted revenue, cost and efficiency metrics.

    ``profile_hook(request, profile)`` (optional) is called at every arrival; it is used to collect residual
    capacity snapshots for the runtime study.  With ``with_departures=False`` accepted requests never leave
    (sequential commitment for the offline setting).
    """
    n = len(requests) if num_arrivals is None else min(int(num_arrivals), len(requests))
    state = SubstrateState(substrate)
    result = OnlineResult(solver=getattr(solver, "name", solver.__class__.__name__), num_arrivals=n)
    departures: List[Tuple[float, int, int]] = []  # (time, seq, request index)
    for idx in range(n):
        req = requests[idx]
        # release every request that departs no later than this arrival (departures are processed first)
        while departures and departures[0][0] <= req.arrival:
            _, _, ridx = heapq.heappop(departures)
            state.release(requests[ridx].id)
        profile = state.profile()
        if profile_hook is not None:
            profile_hook(req, profile)
        t0 = time.perf_counter()
        mapping = solver.embed(req, substrate, profile)
        result.latencies.append(time.perf_counter() - t0)
        if mapping is None:
            continue
        try:
            validate_mapping(req, mapping, substrate, profile)
        except ValidationError:
            result.validation_failures += 1
            continue
        state.commit(req, mapping)
        result.admitted.append(req.id)
        result.revenues.append(request_revenue(req))
        result.costs.append(mapping_cost(req, mapping))
        result.r2cs.append(r2c(req, mapping))
        if keep_mappings:
            result.mappings[req.id] = mapping
        if with_departures and np.isfinite(req.departure):
            heapq.heappush(departures, (float(req.departure), idx, idx))
    return result
