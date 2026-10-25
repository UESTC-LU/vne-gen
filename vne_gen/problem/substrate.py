"""Substrate network and capacity profiles (Sec. III-A).

The substrate is an undirected graph :math:`G_S=(V_S,E_S)`.  Every substrate node ``u`` carries a
node-resource capacity vector :math:`C_S(u)=(C_S^q(u))_{q\\in Q}` over the resource dimensions
:math:`Q=\\{\\mathrm{CPU},\\mathrm{GPU},\\mathrm{RAM}\\}` and every link ``e`` a bandwidth capacity
:math:`C_S(e)`.  The scalar resources are indexed by :math:`X=(V_S\\times Q)\\cup E_S`.

A :class:`CapacityProfile` is a vector over :math:`X` (stored as a node matrix and a link vector) and is
used both for the *supplied* profile :math:`\\hat C_{S,r}` handed to the generator by a selector and for the
within-attempt residual profile :math:`\\tilde C_{S,r,k}`.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

import networkx as nx
import numpy as np

RESOURCES: Tuple[str, ...] = ("cpu", "gpu", "ram")
NUM_RESOURCES: int = len(RESOURCES)


@dataclass
class CapacityProfile:
    """A capacity (or residual-capacity) vector over all scalar substrate resources.

    ``node`` has shape ``(|V_S|, |Q|)`` and ``link`` has shape ``(|E_S|,)`` (indexed by edge id).
    """

    node: np.ndarray
    link: np.ndarray

    def copy(self) -> "CapacityProfile":
        return CapacityProfile(self.node.copy(), self.link.copy())

    def minimum(self, other: "CapacityProfile") -> "CapacityProfile":
        """Componentwise minimum (used by the time-aware profile, Eq. 11)."""
        return CapacityProfile(np.minimum(self.node, other.node), np.minimum(self.link, other.link))

    def subtract(self, node_alloc: np.ndarray, link_alloc: np.ndarray) -> "CapacityProfile":
        return CapacityProfile(self.node - node_alloc, self.link - link_alloc)

    def clip_nonnegative(self) -> "CapacityProfile":
        return CapacityProfile(np.maximum(self.node, 0), np.maximum(self.link, 0))

    def is_within(self, other: "CapacityProfile") -> bool:
        """``0 <= self <= other`` componentwise (requirement on supplied profiles, Sec. III-A)."""
        return bool(
            np.all(self.node >= 0)
            and np.all(self.link >= 0)
            and np.all(self.node <= other.node)
            and np.all(self.link <= other.link)
        )

    def as_vector(self) -> np.ndarray:
        return np.concatenate([self.node.reshape(-1), self.link.reshape(-1)])


@dataclass
class Substrate:
    """Undirected substrate graph with integer node ids ``0..n-1`` and edge ids ``0..m-1``."""

    graph: nx.Graph
    node_cap: np.ndarray  # (n, |Q|)
    link_cap: np.ndarray  # (m,)
    edges: List[Tuple[int, int]] = field(default_factory=list)  # edge id -> (u, v) with u < v
    name: str = "substrate"

    # derived structures (filled in __post_init__)
    edge_index: Dict[Tuple[int, int], int] = field(default_factory=dict, repr=False)
    adjacency: List[List[Tuple[int, int]]] = field(default_factory=list, repr=False)  # u -> [(v, e)]

    def __post_init__(self) -> None:
        n = self.graph.number_of_nodes()
        if sorted(self.graph.nodes()) != list(range(n)):
            raise ValueError("substrate nodes must be relabelled to 0..n-1 (use Substrate.from_graph)")
        if not self.edges:
            self.edges = [(min(u, v), max(u, v)) for u, v in sorted((min(a, b), max(a, b)) for a, b in self.graph.edges())]
        self.edge_index = {}
        for e, (u, v) in enumerate(self.edges):
            self.edge_index[(u, v)] = e
            self.edge_index[(v, u)] = e
        self.adjacency = [[] for _ in range(n)]
        for e, (u, v) in enumerate(self.edges):
            self.adjacency[u].append((v, e))
            self.adjacency[v].append((u, e))
        for u in range(n):
            self.adjacency[u].sort()  # deterministic neighbour iteration (ascending node index)
        self.node_cap = np.asarray(self.node_cap, dtype=np.int64)
        self.link_cap = np.asarray(self.link_cap, dtype=np.int64)
        if self.node_cap.shape != (n, NUM_RESOURCES):
            raise ValueError(f"node_cap must have shape ({n}, {NUM_RESOURCES}), got {self.node_cap.shape}")
        if self.link_cap.shape != (len(self.edges),):
            raise ValueError("link_cap must have one entry per edge")

    # ------------------------------------------------------------------ constructors
    @classmethod
    def from_graph(
        cls,
        graph: nx.Graph,
        node_cap: np.ndarray,
        link_cap: Optional[np.ndarray] = None,
        name: str = "substrate",
    ) -> "Substrate":
        """Relabel an arbitrary graph to ``0..n-1`` (sorted by original label) and build the substrate."""
        mapping = {old: i for i, old in enumerate(sorted(graph.nodes(), key=lambda x: (str(type(x)), x)))}
        g = nx.relabel_nodes(nx.Graph(graph), mapping, copy=True)
        edges = sorted((min(a, b), max(a, b)) for a, b in g.edges())
        if link_cap is None:
            link_cap = np.zeros(len(edges), dtype=np.int64)
        return cls(graph=g, node_cap=np.asarray(node_cap), link_cap=np.asarray(link_cap), edges=edges, name=name)

    # ------------------------------------------------------------------ properties
    @property
    def num_nodes(self) -> int:
        return self.graph.number_of_nodes()

    @property
    def num_links(self) -> int:
        return len(self.edges)

    def degree(self, u: int) -> int:
        return len(self.adjacency[u])

    def neighbors(self, u: int) -> List[int]:
        return [v for v, _ in self.adjacency[u]]

    def edge_id(self, u: int, v: int) -> int:
        return self.edge_index[(u, v)]

    def base_profile(self) -> CapacityProfile:
        """The physical capacity profile :math:`C_S` (first-round profile of the selectors)."""
        return CapacityProfile(self.node_cap.copy(), self.link_cap.copy())

    def edge_array(self) -> np.ndarray:
        """``(m, 2)`` integer array of edges (u < v)."""
        return np.asarray(self.edges, dtype=np.int64).reshape(-1, 2)

    def with_capacities(self, node_cap: np.ndarray, link_cap: np.ndarray, name: Optional[str] = None) -> "Substrate":
        return Substrate(
            graph=self.graph,
            node_cap=np.asarray(node_cap),
            link_cap=np.asarray(link_cap),
            edges=list(self.edges),
            name=name or self.name,
        )

    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "num_nodes": self.num_nodes,
            "edges": [list(e) for e in self.edges],
            "node_cap": self.node_cap.tolist(),
            "link_cap": self.link_cap.tolist(),
        }

    @classmethod
    def from_dict(cls, d: dict) -> "Substrate":
        g = nx.Graph()
        g.add_nodes_from(range(int(d["num_nodes"])))
        g.add_edges_from([tuple(e) for e in d["edges"]])
        return cls(
            graph=g,
            node_cap=np.asarray(d["node_cap"]),
            link_cap=np.asarray(d["link_cap"]),
            edges=[tuple(e) for e in d["edges"]],
            name=d.get("name", "substrate"),
        )


def hop_distances(substrate: Substrate, sources: Sequence[int]) -> np.ndarray:
    """Multi-source BFS hop distance from the nearest node in ``sources`` (``-1`` where unreachable/empty)."""
    n = substrate.num_nodes
    dist = np.full(n, -1, dtype=np.int64)
    if len(sources) == 0:
        return dist
    frontier = list(sources)
    for s in frontier:
        dist[s] = 0
    while frontier:
        nxt = []
        for u in frontier:
            du = dist[u] + 1
            for v, _ in substrate.adjacency[u]:
                if dist[v] < 0:
                    dist[v] = du
                    nxt.append(v)
        frontier = nxt
    return dist
