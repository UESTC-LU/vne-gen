"""Offline corpus collection (Sec. IV-C).

The corpus :math:`\\mathcal D_{map}` is collected by running GRC-Rank, NRM-Rank and NEA-Rank on independently
sampled GEANT workloads through the event-driven online simulator: every mapping attempt (completed or
terminated by placement/routing infeasibility) is recorded together with the residual capacity profile that
was supplied at its arrival.  The three heuristics contribute approximately 20%, 30% and 50% of the
attempts, respectively.  Evaluation workloads are generated from disjoint seeds.
"""

from __future__ import annotations

import heapq
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from ..env.mapping_env import AttemptRecord
from ..env.simulator import SubstrateState
from ..heuristics.ranking import RankSolver, make_rank_solver
from ..problem.generators import WorkloadConfig, generate_workload
from ..problem.substrate import Substrate
from .dataset import CorpusMeta


def _collect_with_solver(
    substrate: Substrate,
    solver: RankSolver,
    rng: np.random.Generator,
    num_attempts: int,
    arrival_rates: Sequence[float],
    arrivals_per_episode: int,
    workload_cfg: WorkloadConfig,
    source: str,
) -> List[AttemptRecord]:
    """Run online episodes with ``solver`` until ``num_attempts`` attempts have been recorded."""
    records: List[AttemptRecord] = []
    episode = 0
    while len(records) < num_attempts:
        lam = float(arrival_rates[int(rng.integers(len(arrival_rates)))])
        requests = generate_workload(rng, arrivals_per_episode, lam, workload_cfg, id_offset=episode * arrivals_per_episode)
        state = SubstrateState(substrate)
        departures: List[Tuple[float, int]] = []
        for idx, req in enumerate(requests):
            while departures and departures[0][0] <= req.arrival:
                _, ridx = heapq.heappop(departures)
                state.release(requests[ridx].id)
            profile = state.profile()
            attempt = solver.run_attempt(req, substrate, profile)
            records.append(attempt.record(source))
            if attempt.success:
                state.commit(req, attempt.mapping())
                heapq.heappush(departures, (float(req.departure), idx))
            if len(records) >= num_attempts:
                break
        episode += 1
    return records


def collect_corpus(
    substrate: Substrate,
    total_attempts: int = 75000,
    mix: Dict[str, float] = None,
    arrival_rates: Sequence[float] = (0.04, 0.05, 0.06, 0.07, 0.08, 0.09),
    arrivals_per_episode: int = 1000,
    seed: int = 12345,
    workload_cfg: WorkloadConfig = WorkloadConfig(),
) -> Tuple[List[AttemptRecord], CorpusMeta]:
    """Collect the offline corpus from the ranking heuristics (default mix GRC 20% / NRM 30% / NEA 50%)."""
    mix = mix or {"grc": 0.2, "nrm": 0.3, "nea": 0.5}
    rng = np.random.default_rng(seed)
    records: List[AttemptRecord] = []
    sources: Dict[str, int] = {}
    for name, frac in mix.items():
        n = int(round(total_attempts * frac))
        solver = make_rank_solver(name)
        recs = _collect_with_solver(substrate, solver, rng, n, arrival_rates, arrivals_per_episode, workload_cfg, solver.name)
        sources[solver.name] = len(recs)
        records.extend(recs)
    meta = corpus_statistics(records)
    meta.substrate_name = substrate.name
    meta.sources = sources
    meta.extra = {"seed": seed, "arrival_rates": list(arrival_rates), "arrivals_per_episode": arrivals_per_episode}
    return records, meta


def corpus_statistics(records: Sequence[AttemptRecord]) -> CorpusMeta:
    num_transitions = sum(r.num_steps for r in records)
    completed = sum(1 for r in records if r.success)
    return CorpusMeta(
        num_attempts=len(records),
        num_transitions=num_transitions,
        completed_fraction=completed / len(records) if records else 0.0,
    )
