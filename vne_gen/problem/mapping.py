"""Mappings, induced allocations, routed-resource cost (Eq. 2) and revenue-to-cost ratios."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Iterable, List, Sequence, Tuple

import numpy as np

from .request import Request, request_revenue
from .substrate import NUM_RESOURCES, Substrate


@dataclass
class Allocation:
    """Aggregate node and link allocations :math:`A_r(m,u)` and :math:`A_r(m,e)` of a mapping."""

    node: np.ndarray  # (|V_S|, |Q|)
    link: np.ndarray  # (|E_S|,)

    def signature(self) -> bytes:
        """Allocation signature :math:`\\sigma_r(m)` used for proposal deduplication (Sec. V-A)."""
        return self.node.astype(np.int64).tobytes() + b"|" + self.link.astype(np.int64).tobytes()

    def copy(self) -> "Allocation":
        return Allocation(self.node.copy(), self.link.copy())


@dataclass
class Mapping:
    """A complete mapping :math:`m=(m^V_r, m^E_r)`.

    * ``node_map[i]`` is the substrate node hosting virtual node ``i``.
    * ``link_paths[l]`` is the substrate path (list of substrate node ids) of virtual link ``edges[l]``,
      oriented from ``node_map[i]`` to ``node_map[j]`` for ``(i, j) = edges[l]``.
    """

    node_map: List[int]
    link_paths: Dict[int, List[int]] = field(default_factory=dict)

    def path_hops(self, l: int) -> int:
        return len(self.link_paths[l]) - 1

    def allocation(self, request: Request, substrate: Substrate) -> Allocation:
        node = np.zeros((substrate.num_nodes, NUM_RESOURCES), dtype=np.int64)
        link = np.zeros(substrate.num_links, dtype=np.int64)
        for i, u in enumerate(self.node_map):
            node[u] += request.node_demand[i]
        for l, path in self.link_paths.items():
            d = int(request.link_demand[l])
            for a, b in zip(path[:-1], path[1:]):
                link[substrate.edge_id(a, b)] += d
        return Allocation(node, link)

    def used_substrate_nodes(self) -> List[int]:
        return sorted(set(self.node_map))

    def to_dict(self) -> dict:
        return {"node_map": [int(u) for u in self.node_map],
                "link_paths": {int(l): [int(x) for x in p] for l, p in self.link_paths.items()}}

    @classmethod
    def from_dict(cls, d: dict) -> "Mapping":
        return cls(node_map=[int(u) for u in d["node_map"]],
                   link_paths={int(l): [int(x) for x in p] for l, p in d["link_paths"].items()})


def mapping_cost(request: Request, mapping: Mapping) -> float:
    r"""Routed-resource cost (Eq. 2): revenue plus the excess bandwidth consumed by multi-hop routes.

    .. math:: c_{r,m} = b_r + \sum_{(i,j)\in E_r}\big(\ell(m^E_r(i,j)) - 1\big)\, d_r(i,j)
    """
    cost = request_revenue(request)
    for l in range(request.num_links):
        cost += (mapping.path_hops(l) - 1) * float(request.link_demand[l])
    return cost


def r2c(request: Request, mapping: Mapping) -> float:
    """Request-level revenue-to-cost ratio :math:`\\mathrm{R2C}_r(m)=b_r/c_{r,m}\\le 1`."""
    return request_revenue(request) / mapping_cost(request, mapping)


def long_term_r2c(revenues: Iterable[float], costs: Iterable[float]) -> float:
    """R2C-LT: total admitted revenue divided by total routed-resource cost."""
    rev = float(sum(revenues))
    cost = float(sum(costs))
    return rev / cost if cost > 0 else 0.0
