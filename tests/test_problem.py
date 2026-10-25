import networkx as nx
import numpy as np
import pytest

from vne_gen.problem import (
    Mapping,
    Request,
    Substrate,
    mapping_cost,
    min_hop_feasible_path,
    r2c,
    request_revenue,
    sample_request,
    validate_mapping,
    ValidationError,
    waxman_substrate,
)
from vne_gen.problem.substrate import NUM_RESOURCES, hop_distances


def test_geant_topology(geant):
    assert geant.num_nodes == 40 and geant.num_links == 61
    assert nx.is_connected(geant.graph)
    assert geant.node_cap.min() >= 50 and geant.node_cap.max() <= 100
    assert geant.link_cap.min() >= 50 and geant.link_cap.max() <= 100


def test_waxman_substrates():
    for n in (100, 500):
        sub = waxman_substrate(n, np.random.default_rng(3))
        assert sub.num_nodes == n and nx.is_connected(sub.graph)


def test_request_sampling_and_revenue():
    rng = np.random.default_rng(0)
    for k in range(50):
        req = sample_request(rng, k)
        assert 2 <= req.num_nodes <= 10
        g = nx.Graph()
        g.add_nodes_from(range(req.num_nodes))
        g.add_edges_from(req.edges)
        assert nx.is_connected(g)
        assert req.node_demand.min() >= 1 and req.node_demand.max() <= 20
        # Eq. (1)
        expected = req.node_demand.sum() / NUM_RESOURCES + req.link_demand.sum()
        assert request_revenue(req) == pytest.approx(expected)


def test_routing_min_hop_and_lexicographic_tie_break():
    # square with a diagonal shortcut: 0-1, 1-3, 0-2, 2-3, 0-3
    g = nx.Graph([(0, 1), (1, 3), (0, 2), (2, 3), (0, 3)])
    sub = Substrate.from_graph(g, node_cap=np.full((4, NUM_RESOURCES), 100), link_cap=np.full(5, 100))
    residual = sub.link_cap.copy()
    assert min_hop_feasible_path(sub, 0, 3, 10, residual) == [0, 3]
    residual[sub.edge_id(0, 3)] = 5  # direct edge no longer feasible for demand 10
    # two 2-hop paths: 0-1-3 and 0-2-3 -> lexicographically smallest node sequence wins
    assert min_hop_feasible_path(sub, 0, 3, 10, residual) == [0, 1, 3]
    residual[sub.edge_id(0, 1)] = 5
    assert min_hop_feasible_path(sub, 0, 3, 10, residual) == [0, 2, 3]
    residual[sub.edge_id(2, 3)] = 5
    assert min_hop_feasible_path(sub, 0, 3, 10, residual) is None


def test_cost_and_r2c():
    g = nx.path_graph(3)
    sub = Substrate.from_graph(g, node_cap=np.full((3, NUM_RESOURCES), 100), link_cap=np.full(2, 100))
    req = Request(id=0, node_demand=np.full((2, NUM_RESOURCES), 3), edges=[(0, 1)], link_demand=np.array([7]))
    m = Mapping(node_map=[0, 2], link_paths={0: [0, 1, 2]})
    # Eq. (2): cost = revenue + (hops - 1) * demand
    assert mapping_cost(req, m) == pytest.approx(request_revenue(req) + 7)
    assert r2c(req, m) == pytest.approx(request_revenue(req) / (request_revenue(req) + 7))
    validate_mapping(req, m, sub, sub.base_profile())
    bad = Mapping(node_map=[0, 0], link_paths={0: [0]})
    with pytest.raises(ValidationError):
        validate_mapping(req, bad, sub, sub.base_profile())


def test_hop_distances(geant):
    d = hop_distances(geant, [0])
    assert d[0] == 0 and d.min() == 0 and np.all(d >= 0)
    assert hop_distances(geant, []).max() == -1
