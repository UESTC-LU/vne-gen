"""Substrate topologies, capacity sampling, request sampling and workload generation (Sec. VI-A).

Experimental defaults follow the paper:

* substrate CPU/GPU/RAM and link-bandwidth capacities are integers sampled uniformly from ``[50, 100]``;
* each request has 2-10 virtual nodes, node-resource and link-bandwidth demands are integers sampled
  uniformly from ``[1, 20]``, each possible virtual link is included independently with probability 0.5 and
  the request graph is resampled until it is connected;
* requests arrive according to a Poisson process with rate ``lam`` requests per slot and hold resources for an
  exponentially distributed time with mean 500 slots.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import List, Optional, Tuple

import networkx as nx
import numpy as np

from .request import Request
from .substrate import NUM_RESOURCES, Substrate

_TOPOLOGY_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data", "topologies")


# --------------------------------------------------------------------------- substrates
def load_gml_substrate(path: str, rng: Optional[np.random.Generator] = None, cap_low: int = 50,
                       cap_high: int = 100, name: Optional[str] = None) -> Substrate:
    """Load a GML topology (e.g. from the Internet Topology Zoo) and sample capacities."""
    g = nx.read_gml(path, label="id")
    g = nx.Graph(g)  # drop multi-edges
    g.remove_edges_from(nx.selfloop_edges(g))
    if not nx.is_connected(g):
        largest = max(nx.connected_components(g), key=len)
        g = g.subgraph(largest).copy()
    rng = rng if rng is not None else np.random.default_rng(0)
    sub = Substrate.from_graph(g, node_cap=np.zeros((g.number_of_nodes(), NUM_RESOURCES), dtype=np.int64),
                               name=name or os.path.splitext(os.path.basename(path))[0])
    node_cap, link_cap = sample_capacities(rng, sub.num_nodes, sub.num_links, cap_low, cap_high)
    return sub.with_capacities(node_cap, link_cap)


def load_geant(rng: Optional[np.random.Generator] = None, cap_low: int = 50, cap_high: int = 100) -> Substrate:
    """The 40-node GEANT topology (Internet Topology Zoo, ``Geant2012``) with sampled capacities."""
    return load_gml_substrate(os.path.join(_TOPOLOGY_DIR, "Geant2012.gml"), rng, cap_low, cap_high, name="GEANT-40")


def waxman_substrate(
    num_nodes: int,
    rng: np.random.Generator,
    alpha: float = 0.5,
    beta: float = 0.2,
    cap_low: int = 50,
    cap_high: int = 100,
    name: Optional[str] = None,
) -> Substrate:
    """A connected Waxman random graph [34] with sampled capacities (used for the unseen 100/500-node substrates)."""
    seed = int(rng.integers(0, 2**31 - 1))
    for attempt in range(1000):
        g = nx.waxman_graph(num_nodes, beta=beta, alpha=alpha, seed=seed + attempt)
        if nx.is_connected(g):
            break
    else:  # pragma: no cover - extremely unlikely for the default parameters
        raise RuntimeError("could not sample a connected Waxman graph")
    sub = Substrate.from_graph(g, node_cap=np.zeros((num_nodes, NUM_RESOURCES), dtype=np.int64),
                               name=name or f"Waxman-{num_nodes}")
    node_cap, link_cap = sample_capacities(rng, sub.num_nodes, sub.num_links, cap_low, cap_high)
    return sub.with_capacities(node_cap, link_cap)


def random_connected_substrate(num_nodes: int, edge_prob: float, rng: np.random.Generator,
                               cap_low: int = 50, cap_high: int = 100, name: Optional[str] = None) -> Substrate:
    """A connected Erdos-Renyi substrate (used for the small-instance optimality study)."""
    while True:
        g = nx.gnp_random_graph(num_nodes, edge_prob, seed=int(rng.integers(0, 2**31 - 1)))
        if nx.is_connected(g):
            break
    sub = Substrate.from_graph(g, node_cap=np.zeros((num_nodes, NUM_RESOURCES), dtype=np.int64),
                               name=name or f"ER-{num_nodes}")
    node_cap, link_cap = sample_capacities(rng, sub.num_nodes, sub.num_links, cap_low, cap_high)
    return sub.with_capacities(node_cap, link_cap)


def sample_capacities(rng: np.random.Generator, num_nodes: int, num_links: int, low: int = 50,
                      high: int = 100) -> Tuple[np.ndarray, np.ndarray]:
    node_cap = rng.integers(low, high + 1, size=(num_nodes, NUM_RESOURCES), dtype=np.int64)
    link_cap = rng.integers(low, high + 1, size=(num_links,), dtype=np.int64)
    return node_cap, link_cap


# --------------------------------------------------------------------------- requests / workloads
@dataclass
class WorkloadConfig:
    min_nodes: int = 2
    max_nodes: int = 10
    demand_low: int = 1
    demand_high: int = 20
    link_prob: float = 0.5
    mean_holding: float = 500.0


def sample_request(rng: np.random.Generator, request_id: int, cfg: WorkloadConfig = WorkloadConfig(),
                   arrival: float = 0.0, departure: float = float("inf"), num_nodes: Optional[int] = None) -> Request:
    """Sample one connected request graph with uniform integer demands."""
    n = int(num_nodes) if num_nodes is not None else int(rng.integers(cfg.min_nodes, cfg.max_nodes + 1))
    while True:
        edges = [(i, j) for i in range(n) for j in range(i + 1, n) if rng.random() < cfg.link_prob]
        g = nx.Graph()
        g.add_nodes_from(range(n))
        g.add_edges_from(edges)
        if nx.is_connected(g):
            break
    node_demand = rng.integers(cfg.demand_low, cfg.demand_high + 1, size=(n, NUM_RESOURCES), dtype=np.int64)
    link_demand = rng.integers(cfg.demand_low, cfg.demand_high + 1, size=(len(edges),), dtype=np.int64)
    return Request(id=request_id, node_demand=node_demand, edges=edges, link_demand=link_demand,
                   arrival=arrival, departure=departure)


def generate_workload(rng: np.random.Generator, num_requests: int, lam: float,
                      cfg: WorkloadConfig = WorkloadConfig(), id_offset: int = 0) -> List[Request]:
    """Poisson arrivals (rate ``lam`` per slot) with exponentially distributed holding times."""
    t = 0.0
    requests = []
    for k in range(num_requests):
        t += float(rng.exponential(1.0 / lam))
        holding = float(rng.exponential(cfg.mean_holding))
        requests.append(sample_request(rng, id_offset + k, cfg, arrival=t, departure=t + holding))
    return requests


def generate_request_set(rng: np.random.Generator, num_requests: int, cfg: WorkloadConfig = WorkloadConfig(),
                         id_offset: int = 0) -> List[Request]:
    """A request set without timing information (offline VNE treats all requests as simultaneously active)."""
    return [sample_request(rng, id_offset + k, cfg) for k in range(num_requests)]
