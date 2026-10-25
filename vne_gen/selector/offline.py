"""Offline and time-aware offline proposal selection (Algorithm 2, Sec. V-B/V-C).

Each generation round makes the scheduled proposal calls for every request under a request-specific
supplied capacity profile, deduplicates proposals by allocation signature, re-solves the exact
candidate-selection ILP (Eq. 3 or Eq. 4) over the accumulated pools and keeps the highest-revenue feasible
selection found so far.  From the second round on, the supplied profile of request ``r`` is the capacity
remaining after subtracting the selected allocations of all *other* requests (Eq. 10), or, in the time-aware
setting, the componentwise minimum of that residual over the activity interval of ``r`` (Eq. 11).  The
``base`` profile mode (control experiment) keeps supplying the physical capacity :math:`C_S` in every round.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from ..problem.feasibility import is_feasible
from ..problem.mapping import long_term_r2c
from ..problem.request import Request
from ..problem.substrate import CapacityProfile, Substrate
from .ilp import SelectionResult, check_selection, solve_selection
from .proposal import PortfolioSource, Proposal, ProposalPool, ProposalSource
from .timeslots import activity_matrix, elementary_slots

logger = logging.getLogger(__name__)


@dataclass
class SelectorConfig:
    setting: str = "off"  # "off" (Eq. 3 / Eq. 10) or "ta" (Eq. 4 / Eq. 11)
    rounds: int = 5  # round limit I_max
    schedule: Sequence[Sequence[float]] = ((1.0),)  # S_i: return scales per attempt, per round (cycled)
    fixed_rounds: bool = False  # complete every prescribed round (no early stopping)
    time_limit: Optional[float] = 300.0  # seconds per ILP solve
    profile_mode: str = "selector"  # "selector" (Eq. 10/11) or "base" (always C_S)
    compute_conflicts: bool = True
    max_conflict_candidates: int = 1500
    seed: int = 0

    def round_schedule(self, i: int) -> List[float]:
        sched = self.schedule
        if not sched:
            return [1.0]
        s = sched[(i - 1) % len(sched)]
        if isinstance(s, (int, float)):
            return [float(s)]
        return [float(x) for x in s]


@dataclass
class RoundDiagnostics:
    round: int
    revenue: float
    ilp_status: str
    num_calls: int
    num_valid: int
    num_new_signatures: int
    total_signatures: int
    conflict_rate: float
    generation_time_s: float
    selection_time_s: float
    r2c_lt: float


@dataclass
class OfflineRunResult:
    setting: str
    source: str
    best: List[Proposal] = field(default_factory=list)
    revenue: float = 0.0
    rounds: List[RoundDiagnostics] = field(default_factory=list)
    generation_time_s: float = 0.0
    selection_time_s: float = 0.0

    @property
    def total_cost(self) -> float:
        return float(sum(p.cost for p in self.best))

    @property
    def r2c_lt(self) -> float:
        return long_term_r2c([p.revenue for p in self.best], [p.cost for p in self.best])

    @property
    def mean_r2c(self) -> float:
        return float(np.mean([p.revenue / p.cost for p in self.best])) if self.best else 0.0

    def summary(self) -> dict:
        return {
            "setting": self.setting,
            "source": self.source,
            "num_selected": len(self.best),
            "revenue": self.revenue,
            "total_cost": self.total_cost,
            "r2c_lt": self.r2c_lt,
            "mean_r2c": self.mean_r2c,
            "rounds_completed": len(self.rounds),
            "generation_time_s": self.generation_time_s,
            "selection_time_s": self.selection_time_s,
            "round_revenues": [r.revenue for r in self.rounds],
            "proposal_valid_rate": (sum(r.num_valid for r in self.rounds) / max(1, sum(r.num_calls for r in self.rounds))),
            "distinct_signatures": self.rounds[-1].total_signatures if self.rounds else 0,
            "conflict_rate": self.rounds[-1].conflict_rate if self.rounds else 0.0,
        }


class CandidatePoolSelector:
    """Algorithm 2 for a proposal source (VNE-Gen generator, a ranking heuristic or a portfolio)."""

    def __init__(self, substrate: Substrate, requests: Sequence[Request], source, cfg: SelectorConfig) -> None:
        if cfg.setting not in ("off", "ta"):
            raise ValueError("setting must be 'off' or 'ta'")
        self.substrate = substrate
        self.requests = list(requests)
        self.source = source
        self.cfg = cfg
        self.rng = np.random.default_rng(cfg.seed)
        self.index = {r.id: i for i, r in enumerate(self.requests)}
        self.base = substrate.base_profile()
        if cfg.setting == "ta":
            self.slots = elementary_slots(self.requests)
            self.alpha = activity_matrix(self.requests, self.slots)
        else:
            self.slots = [(0.0, float("inf"))]
            self.alpha = np.ones((len(self.requests), 1), dtype=np.int8)

    # ------------------------------------------------------------------ capacity feedback (Eq. 10 / 11)
    def supplied_profiles(self, selection: Sequence[Proposal]) -> List[CapacityProfile]:
        if self.cfg.profile_mode == "base":
            return [self.base.copy() for _ in self.requests]
        T = self.alpha.shape[1]
        node_use = np.zeros((T,) + self.base.node.shape, dtype=np.int64)
        link_use = np.zeros((T, self.base.link.shape[0]), dtype=np.int64)
        selected_by_request: Dict[int, Proposal] = {}
        for p in selection:
            r_idx = self.index[p.request_id]
            selected_by_request[r_idx] = p
            for t in np.flatnonzero(self.alpha[r_idx]):
                node_use[t] += p.allocation.node
                link_use[t] += p.allocation.link
        profiles = []
        for r_idx, req in enumerate(self.requests):
            active = np.flatnonzero(self.alpha[r_idx])
            own = selected_by_request.get(r_idx)
            node_min = None
            link_min = None
            for t in active:  # the request's own allocation is excluded so that it may propose a replacement
                n_t = self.base.node - node_use[t]
                l_t = self.base.link - link_use[t]
                if own is not None:
                    n_t = n_t + own.allocation.node
                    l_t = l_t + own.allocation.link
                node_min = n_t if node_min is None else np.minimum(node_min, n_t)
                link_min = l_t if link_min is None else np.minimum(link_min, l_t)
            if node_min is None:  # request active in no slot (degenerate interval): supply the base profile
                node_min, link_min = self.base.node.copy(), self.base.link.copy()
            profiles.append(CapacityProfile(np.maximum(node_min, 0), np.maximum(link_min, 0)))
        return profiles

    # ------------------------------------------------------------------ diagnostics
    def _conflict_rate(self, pools: Sequence[ProposalPool]) -> float:
        props = [(r_idx, p) for r_idx, pool in enumerate(pools) for p in pool]
        P = len(props)
        if P < 2 or P > self.cfg.max_conflict_candidates:
            return float("nan")
        alloc = np.stack([np.concatenate([p.allocation.node.reshape(-1), p.allocation.link]) for _, p in props])
        cap = np.concatenate([self.base.node.reshape(-1), self.base.link])
        req = np.array([r for r, _ in props])
        overlap = (self.alpha[req] @ self.alpha[req].T) > 0  # pairs whose activity intervals share a slot
        pairs = 0
        conflicts = 0
        for i in range(P):
            others = np.arange(i + 1, P)
            if others.size == 0:
                continue
            diff_req = req[others] != req[i]
            cand = others[diff_req & overlap[i, others]]
            pairs += int(np.sum(diff_req))
            if cand.size:
                exceeded = np.any(alloc[cand] + alloc[i][None, :] > cap[None, :], axis=1)
                conflicts += int(exceeded.sum())
        return conflicts / pairs if pairs else float("nan")

    # ------------------------------------------------------------------ Algorithm 2
    def run(self) -> OfflineRunResult:
        cfg = self.cfg
        pools = [ProposalPool(r.id) for r in self.requests]
        best: List[Proposal] = []
        best_revenue = 0.0
        result = OfflineRunResult(setting=cfg.setting, source=getattr(self.source, "name", "source"))
        prev_selection: List[Proposal] = []
        for i in range(1, cfg.rounds + 1):
            profiles = [self.base.copy() for _ in self.requests] if i == 1 else self.supplied_profiles(prev_selection)
            t_gen = time.time()
            num_calls = num_valid = num_new = 0
            for r_idx, req in enumerate(self.requests):
                for rho in cfg.round_schedule(i):
                    profile = profiles[r_idx].copy()  # fresh copy per attempt
                    if isinstance(self.source, PortfolioSource):
                        mappings = self.source.propose_all(req, self.substrate, profile, rho, self.rng)
                        num_calls += len(self.source.sources)
                    else:
                        m = self.source.propose(req, self.substrate, profile, rho, self.rng)
                        mappings = [m] if m is not None else []
                        num_calls += 1
                    for m in mappings:
                        if not is_feasible(req, m, self.substrate, profile):
                            continue
                        num_valid += 1
                        if pools[r_idx].add(Proposal.build(req, m, self.substrate, getattr(self.source, "name", ""))):
                            num_new += 1
            gen_time = time.time() - t_gen
            # selection over all accumulated pools
            t_sel = time.time()
            sel = solve_selection(pools, self.base.node, self.base.link,
                                  None if cfg.setting == "off" else self.alpha, time_limit=cfg.time_limit)
            sel_time = time.time() - t_sel
            current = list(best)  # B^(i) <- B_best
            if sel.feasible and check_selection(sel.selected, self.base.node, self.base.link,
                                                None if cfg.setting == "off" else self.alpha, self.index):
                current = list(sel.selected)
            current_revenue = float(sum(p.revenue for p in current))
            improved = current_revenue > best_revenue
            if improved:
                best, best_revenue = list(current), current_revenue
            total_sigs = sum(len(p) for p in pools)
            diag = RoundDiagnostics(
                round=i, revenue=current_revenue, ilp_status=sel.status, num_calls=num_calls, num_valid=num_valid,
                num_new_signatures=num_new, total_signatures=total_sigs,
                conflict_rate=self._conflict_rate(pools) if cfg.compute_conflicts else float("nan"),
                generation_time_s=gen_time, selection_time_s=sel_time,
                r2c_lt=long_term_r2c([p.revenue for p in current], [p.cost for p in current]),
            )
            result.rounds.append(diag)
            result.generation_time_s += gen_time
            result.selection_time_s += sel_time
            logger.info("round %d | revenue %.1f (best %.1f) | %d/%d valid | %d signatures | ilp %s in %.1fs",
                        i, current_revenue, best_revenue, num_valid, num_calls, total_sigs, sel.status, sel_time)
            prev_selection = current
            if i > 1 and not cfg.fixed_rounds and not improved:
                break
        result.best = best
        result.revenue = best_revenue
        return result
