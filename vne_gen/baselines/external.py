"""Plugging external solvers (A3C-GCN, GAL-VNE, HRL-ACRA, FlagVNE, ...) into the evaluation protocols.

The learning-based baselines of the paper are evaluated with their official implementations (see
``docs/EXTERNAL_BASELINES.md``).  Any solver can take part in the online, offline and time-aware protocols by
implementing the :class:`~vne_gen.env.simulator.Solver` interface::

    mapping_or_none = solver.embed(request, substrate, profile)

where ``profile`` is the residual capacity profile at arrival and the returned
:class:`~vne_gen.problem.mapping.Mapping` must be feasible under it (every admitted mapping is re-checked by
the common validator).  :class:`FunctionSolver` wraps a plain Python callable; :class:`ExternalSolver` adds
the conversion helpers that most external code bases need (dense capacity matrices, edge lists, ...).
"""

from __future__ import annotations

from typing import Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np

from ..problem.mapping import Mapping
from ..problem.request import Request
from ..problem.substrate import CapacityProfile, Substrate


class FunctionSolver:
    def __init__(self, fn: Callable[[Request, Substrate, CapacityProfile], Optional[Mapping]], name: str) -> None:
        self.fn = fn
        self.name = name

    def embed(self, request: Request, substrate: Substrate, profile: CapacityProfile) -> Optional[Mapping]:
        return self.fn(request, substrate, profile)

    def propose(self, request, substrate, profile, rho: float = 1.0, rng=None) -> Optional[Mapping]:
        return self.embed(request, substrate, profile)


class ExternalSolver(FunctionSolver):
    """Base class with helpers to translate between this package's data model and external formats."""

    @staticmethod
    def substrate_arrays(substrate: Substrate, profile: CapacityProfile) -> Dict[str, np.ndarray]:
        """Node residual ``(n, |Q|)``, edge list ``(m, 2)`` and link residual ``(m,)``."""
        return {"node_residual": profile.node.copy(), "edges": substrate.edge_array(), "link_residual": profile.link.copy()}

    @staticmethod
    def request_arrays(request: Request) -> Dict[str, np.ndarray]:
        return {"node_demand": request.node_demand.copy(), "edges": np.asarray(request.edges, dtype=np.int64).reshape(-1, 2),
                "link_demand": request.link_demand.copy()}

    @staticmethod
    def mapping_from_paths(request: Request, node_map: Sequence[int], paths: Dict[Tuple[int, int], List[int]]) -> Mapping:
        """Build a :class:`Mapping` from a node map and ``{(i, j): [substrate nodes]}`` paths (either orientation)."""
        link_paths: Dict[int, List[int]] = {}
        for (i, j), path in paths.items():
            l = request.link_id(i, j)
            a, b = request.edges[l]
            p = list(path)
            if p[0] != node_map[a]:
                p = p[::-1]
            link_paths[l] = p
        return Mapping(node_map=[int(u) for u in node_map], link_paths=link_paths)
