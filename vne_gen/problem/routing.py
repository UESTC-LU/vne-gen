"""Deterministic min-hop feasible routing (Eq. 5).

For a virtual link whose endpoints are placed on substrate nodes ``src`` and ``dst``, the routing procedure
selects, among all simple substrate paths whose every edge has residual bandwidth at least the link demand,
a path with the minimum number of hops; ties are broken by the lexicographically smallest sequence of
substrate-node indices.  The implementation runs a BFS from ``dst`` over the feasible edges and then walks
greedily from ``src`` towards ``dst``, always choosing the smallest-index neighbour that lies on a shortest
feasible path, which yields exactly the lexicographically smallest shortest path.
"""

from __future__ import annotations

from typing import List, Optional

import numpy as np

from .substrate import Substrate


def min_hop_feasible_path(
    substrate: Substrate,
    src: int,
    dst: int,
    demand: int,
    residual_link: np.ndarray,
) -> Optional[List[int]]:
    """Return the lexicographically smallest min-hop path from ``src`` to ``dst`` or ``None`` if unroutable.

    Parameters
    ----------
    substrate
        The substrate graph.
    src, dst
        Substrate end points (``src != dst``; the node mapping is injective within a request).
    demand
        Bandwidth demand of the virtual link.
    residual_link
        Residual bandwidth per substrate edge id at the time of routing.
    """
    if src == dst:
        return [src]
    n = substrate.num_nodes
    dist = np.full(n, -1, dtype=np.int64)
    dist[dst] = 0
    frontier = [dst]
    adjacency = substrate.adjacency
    while frontier and dist[src] < 0:
        nxt = []
        for u in frontier:
            du = dist[u] + 1
            for v, e in adjacency[u]:
                if dist[v] < 0 and residual_link[e] >= demand:
                    dist[v] = du
                    nxt.append(v)
        frontier = nxt
    if dist[src] < 0:
        return None
    path = [src]
    cur = src
    while cur != dst:
        target = dist[cur] - 1
        nxt_node = -1
        for v, e in adjacency[cur]:  # adjacency lists are sorted by neighbour index
            if dist[v] == target and residual_link[e] >= demand:
                nxt_node = v
                break
        if nxt_node < 0:  # cannot happen when dist is a valid BFS layering
            return None
        path.append(nxt_node)
        cur = nxt_node
    return path
