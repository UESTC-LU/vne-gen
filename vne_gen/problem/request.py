"""Virtual network requests and the mapping-independent revenue (Eq. 1)."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Tuple

import numpy as np

from .substrate import NUM_RESOURCES


@dataclass
class Request:
    """A virtual network request :math:`G_r=(V_r,E_r)` with demands and an activity interval.

    * ``node_demand`` has shape ``(|V_r|, |Q|)``.
    * ``edges`` lists virtual links ``(i, j)`` with ``i < j``; ``link_demand[l]`` is the bandwidth demand of
      ``edges[l]``.
    * ``arrival`` / ``departure`` are the activity-interval end points :math:`a_r` and :math:`e_r`.
    """

    id: int
    node_demand: np.ndarray
    edges: List[Tuple[int, int]]
    link_demand: np.ndarray
    arrival: float = 0.0
    departure: float = float("inf")

    link_index: Dict[Tuple[int, int], int] = field(default_factory=dict, repr=False)
    adjacency: List[List[Tuple[int, int]]] = field(default_factory=list, repr=False)  # i -> [(j, l)]

    def __post_init__(self) -> None:
        self.node_demand = np.asarray(self.node_demand, dtype=np.int64)
        self.link_demand = np.asarray(self.link_demand, dtype=np.int64)
        if self.node_demand.ndim != 2 or self.node_demand.shape[1] != NUM_RESOURCES:
            raise ValueError("node_demand must have shape (|V_r|, |Q|)")
        self.edges = [(min(i, j), max(i, j)) for i, j in self.edges]
        if self.link_demand.shape != (len(self.edges),):
            raise ValueError("link_demand must have one entry per virtual link")
        n = self.num_nodes
        self.link_index = {}
        self.adjacency = [[] for _ in range(n)]
        for l, (i, j) in enumerate(self.edges):
            if i == j or not (0 <= i < n and 0 <= j < n):
                raise ValueError(f"invalid virtual link {(i, j)}")
            self.link_index[(i, j)] = l
            self.link_index[(j, i)] = l
            self.adjacency[i].append((j, l))
            self.adjacency[j].append((i, l))
        for i in range(n):
            self.adjacency[i].sort()  # original neighbour-iteration order = ascending neighbour index

    @property
    def num_nodes(self) -> int:
        return int(self.node_demand.shape[0])

    @property
    def num_links(self) -> int:
        return len(self.edges)

    @property
    def holding_time(self) -> float:
        return self.departure - self.arrival

    def degree(self, i: int) -> int:
        return len(self.adjacency[i])

    def neighbors(self, i: int) -> List[int]:
        return [j for j, _ in self.adjacency[i]]

    def link_id(self, i: int, j: int) -> int:
        return self.link_index[(i, j)]

    @property
    def revenue(self) -> float:
        return request_revenue(self)

    def to_dict(self) -> dict:
        return {
            "id": int(self.id),
            "node_demand": self.node_demand.tolist(),
            "edges": [list(e) for e in self.edges],
            "link_demand": self.link_demand.tolist(),
            "arrival": float(self.arrival),
            "departure": float(self.departure),
        }

    @classmethod
    def from_dict(cls, d: dict) -> "Request":
        return cls(
            id=int(d["id"]),
            node_demand=np.asarray(d["node_demand"]),
            edges=[tuple(e) for e in d["edges"]],
            link_demand=np.asarray(d["link_demand"]),
            arrival=float(d.get("arrival", 0.0)),
            departure=float(d.get("departure", float("inf"))),
        )


def request_revenue(request: Request) -> float:
    r"""Mapping-independent revenue (Eq. 1).

    .. math:: b_r = \frac{1}{|Q|}\sum_{i\in V_r}\sum_{q\in Q} d_r^q(i) + \sum_{(i,j)\in E_r} d_r(i,j)
    """
    node_part = float(request.node_demand.sum()) / NUM_RESOURCES
    link_part = float(request.link_demand.sum())
    return node_part + link_part
