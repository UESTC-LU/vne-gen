"""Feasibility masks :math:`\\mathcal F_V`, :math:`\\mathcal F_S` and the common mapping validator."""

from __future__ import annotations

from typing import Iterable, List, Optional, Sequence, Set

import numpy as np

from .mapping import Mapping
from .request import Request
from .substrate import CapacityProfile, NUM_RESOURCES, Substrate


class ValidationError(ValueError):
    """Raised by :func:`validate_mapping` when a mapping violates a constraint."""


def node_fits(demand: np.ndarray, residual: np.ndarray) -> bool:
    """``demand <= residual`` in every resource dimension."""
    return bool(np.all(residual >= demand))


def feasible_substrate_nodes(
    request: Request,
    v: int,
    used_substrate: Set[int],
    residual_node: np.ndarray,
) -> List[int]:
    """:math:`\\mathcal F_S(s_k, v_k)`: unused substrate nodes whose residual node resources fit ``v``."""
    demand = request.node_demand[v]
    fits = np.all(residual_node >= demand, axis=1)
    if used_substrate:
        fits[list(used_substrate)] = False
    return [int(u) for u in np.flatnonzero(fits)]


def feasible_substrate_mask(
    request: Request,
    v: int,
    used_substrate: Set[int],
    residual_node: np.ndarray,
) -> np.ndarray:
    demand = request.node_demand[v]
    fits = np.all(residual_node >= demand, axis=1)
    if used_substrate:
        fits[list(used_substrate)] = False
    return fits


def feasible_virtual_nodes(
    request: Request,
    placed: Set[int],
    used_substrate: Set[int],
    residual_node: np.ndarray,
) -> List[int]:
    """:math:`\\mathcal F_V(s_k)`: unplaced virtual nodes that fit on at least one unused substrate node."""
    avail = np.ones(residual_node.shape[0], dtype=bool)
    if used_substrate:
        avail[list(used_substrate)] = False
    res = residual_node[avail]
    out = []
    for i in range(request.num_nodes):
        if i in placed:
            continue
        if res.shape[0] and bool(np.any(np.all(res >= request.node_demand[i], axis=1))):
            out.append(i)
    return out


def validate_mapping(
    request: Request,
    mapping: Mapping,
    substrate: Substrate,
    profile: CapacityProfile,
) -> None:
    """Check injectivity, node capacity, path continuity/end points and aggregate link capacity.

    Raises :class:`ValidationError` on the first violation.  This validator is shared by all methods
    ("every admitted mapping produced by any method is checked by the same feasibility validator").
    """
    n_v = request.num_nodes
    if len(mapping.node_map) != n_v:
        raise ValidationError("node_map has the wrong length")
    for i, u in enumerate(mapping.node_map):
        if not (0 <= u < substrate.num_nodes):
            raise ValidationError(f"virtual node {i} mapped to invalid substrate node {u}")
    if len(set(mapping.node_map)) != n_v:
        raise ValidationError("node mapping is not injective")
    if set(mapping.link_paths.keys()) != set(range(request.num_links)):
        raise ValidationError("every virtual link must be routed exactly once")
    node_alloc = np.zeros((substrate.num_nodes, NUM_RESOURCES), dtype=np.int64)
    link_alloc = np.zeros(substrate.num_links, dtype=np.int64)
    for i, u in enumerate(mapping.node_map):
        node_alloc[u] += request.node_demand[i]
    for l, (i, j) in enumerate(request.edges):
        path = mapping.link_paths[l]
        if len(path) < 2:
            raise ValidationError(f"virtual link {l} has a degenerate path")
        if path[0] != mapping.node_map[i] or path[-1] != mapping.node_map[j]:
            raise ValidationError(f"virtual link {l} path does not connect its end points")
        if len(set(path)) != len(path):
            raise ValidationError(f"virtual link {l} path is not simple")
        for a, b in zip(path[:-1], path[1:]):
            if (a, b) not in substrate.edge_index:
                raise ValidationError(f"virtual link {l} path uses non-existent edge {(a, b)}")
            link_alloc[substrate.edge_index[(a, b)]] += int(request.link_demand[l])
    if np.any(node_alloc > profile.node):
        raise ValidationError("node capacity violated")
    if np.any(link_alloc > profile.link):
        raise ValidationError("link capacity violated")


def is_feasible(request: Request, mapping: Mapping, substrate: Substrate, profile: CapacityProfile) -> bool:
    try:
        validate_mapping(request, mapping, substrate, profile)
    except ValidationError:
        return False
    return True
