from unittest.mock import Mock

import networkx as nx
import numpy as np
import pytest

from vne_gen.env.simulator import run_online
from vne_gen.heuristics import make_rank_solver
from vne_gen.problem.generators import WorkloadConfig, generate_workload
from vne_gen.problem.substrate import Substrate
from vne_gen.selector.timeslots import activity_matrix, elementary_slots


@pytest.fixture
def timed_requests():
    rng = Mock(wraps=np.random.default_rng(42))
    # Interleaved pairs of (arrival gap, holding time); all graph draws use the real RNG.
    rng.exponential.side_effect = [0.25, 0.125, 0.25, 0.0, 0.75, 1.875, 2.0, 4.0]
    cfg = WorkloadConfig(min_nodes=2, max_nodes=2, demand_low=1, demand_high=1)
    return generate_workload(rng, 4, lam=1, cfg=cfg)


def test_rounds_cumulative_arrivals_and_holding_times_separately(timed_requests):
    # Continuous arrival times are 0.25, 0.5, 1.25, 3.25. Rounding each gap would separate the first two.
    assert [(r.arrival, r.departure) for r in timed_requests] == [(1, 2), (1, 2), (2, 4), (4, 8)]


def test_online_releases_capacity_before_same_slot_arrival(timed_requests):
    substrate = Substrate.from_graph(nx.path_graph(2), np.ones((2, 3), dtype=int), link_cap=np.array([1]))
    result = run_online(substrate, timed_requests, make_rank_solver("nrm"))
    assert result.admitted == [0, 2, 3]


def test_compressed_activity_matches_all_integer_slots(timed_requests):
    slots = elementary_slots(timed_requests)
    assert slots == [(1, 2), (2, 4), (4, 8)]
    compressed = activity_matrix(timed_requests, slots)
    expanded = np.repeat(compressed, [int(end - start) for start, end in slots], axis=1)
    expected = [[r.arrival <= slot < r.departure for slot in range(1, 8)] for r in timed_requests]
    np.testing.assert_array_equal(expanded, expected)
