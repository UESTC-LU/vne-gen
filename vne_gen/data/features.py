"""Node, edge and scalar features of a mapping state :math:`s_k` (Sec. IV-B).

Substrate-node features include residual resources, occupancy and partial-mapping statistics; virtual-node
features include demands, degree, placement status and mapped-neighbour information.  All features are
normalised with scale constants that are independent of the substrate size, so that a model trained on
GEANT can be applied to unseen 100- and 500-node substrates without retraining.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional

import numpy as np

from ..env.mapping_env import MappingAttempt
from ..problem.substrate import NUM_RESOURCES, Substrate, hop_distances

NUM_SUB_FEATURES = 2 * NUM_RESOURCES + 6  # 12 for |Q| = 3
NUM_REQ_FEATURES = NUM_RESOURCES + 8  # 11 for |Q| = 3
NUM_EDGE_FEATURES = 2
NUM_SCALARS = 6


@dataclass
class FeatureConfig:
    cap_scale: float = 100.0  # upper bound of substrate capacities
    demand_scale: float = 20.0  # upper bound of request demands
    max_request_nodes: int = 10
    sub_size_scale: float = 500.0
    sub_links_scale: float = 1000.0


@dataclass
class StateFeatures:
    """Numpy feature bundle of one state (request graph + substrate graph + scalars)."""

    req_x: np.ndarray  # (|V_r|, NUM_REQ_FEATURES)
    req_edge_index: np.ndarray  # (2, 2|E_r|) both directions
    req_edge_attr: np.ndarray  # (2|E_r|, NUM_EDGE_FEATURES)
    req_placed: np.ndarray  # (|V_r|,) bool
    sub_x: np.ndarray  # (|V_S|, NUM_SUB_FEATURES)
    sub_edge_index: np.ndarray  # (2, 2|E_S|)
    sub_edge_attr: np.ndarray  # (2|E_S|, NUM_EDGE_FEATURES)
    sub_used: np.ndarray  # (|V_S|,) bool
    scalars: np.ndarray  # (NUM_SCALARS,)
    fv_mask: np.ndarray  # (|V_r|,) bool, feasible virtual nodes F_V(s_k)

    @property
    def num_req_nodes(self) -> int:
        return int(self.req_x.shape[0])

    @property
    def num_sub_nodes(self) -> int:
        return int(self.sub_x.shape[0])


class _SubstrateStatic:
    """Per-substrate static quantities, cached across states."""

    def __init__(self, substrate: Substrate) -> None:
        n, m = substrate.num_nodes, substrate.num_links
        self.degree = np.array([substrate.degree(u) for u in range(n)], dtype=np.float64)
        self.max_degree = float(self.degree.max()) if n else 1.0
        edges = substrate.edge_array()
        self.edge_index = np.concatenate([edges.T, edges[:, ::-1].T], axis=1).astype(np.int64)  # (2, 2m)
        self.base_link = substrate.link_cap.astype(np.float64)
        self.base_node = substrate.node_cap.astype(np.float64)
        # incidence (n, m) as sparse lists for adjacent-bandwidth sums
        self.inc_rows = np.concatenate([edges[:, 0], edges[:, 1]])
        self.inc_cols = np.concatenate([np.arange(m), np.arange(m)])


_STATIC_CACHE: Dict[int, _SubstrateStatic] = {}


def _static(substrate: Substrate) -> _SubstrateStatic:
    key = id(substrate)
    st = _STATIC_CACHE.get(key)
    if st is None or st.base_link.shape[0] != substrate.num_links:
        st = _SubstrateStatic(substrate)
        if len(_STATIC_CACHE) > 64:
            _STATIC_CACHE.clear()
        _STATIC_CACHE[key] = st
    return st


def encode_state(attempt: MappingAttempt, cfg: FeatureConfig = FeatureConfig()) -> StateFeatures:
    """Compute the features of the current state of ``attempt``."""
    sub = attempt.substrate
    req = attempt.request
    st = _static(sub)
    n, m = sub.num_nodes, sub.num_links
    nv = req.num_nodes
    # ------------------------------------------------------------------ substrate nodes
    res_node = attempt.residual_node.astype(np.float64)
    res_link = attempt.residual_link.astype(np.float64)
    with np.errstate(divide="ignore", invalid="ignore"):
        frac = np.where(st.base_node > 0, res_node / st.base_node, 0.0)
        link_frac = np.where(st.base_link > 0, res_link / st.base_link, 0.0)
    adj_bw = np.zeros(n, dtype=np.float64)
    np.add.at(adj_bw, st.inc_rows, res_link[st.inc_cols])
    adj_health = np.zeros(n, dtype=np.float64)
    np.add.at(adj_health, st.inc_rows, link_frac[st.inc_cols])
    deg_safe = np.maximum(st.degree, 1.0)
    used = np.zeros(n, dtype=bool)
    if attempt.used_substrate:
        used[list(attempt.used_substrate)] = True
    hops = hop_distances(sub, sorted(attempt.used_substrate))
    prox = np.zeros(n, dtype=np.float64)
    reachable = hops >= 0
    prox[reachable] = 1.0 / (1.0 + hops[reachable])
    sub_x = np.concatenate(
        [
            frac,  # residual / base per dimension
            res_node / cfg.cap_scale,  # residual / global scale
            (1.0 - frac.mean(axis=1))[:, None],  # occupancy
            (st.degree / st.max_degree)[:, None],
            (adj_bw / (deg_safe * cfg.cap_scale))[:, None],
            used.astype(np.float64)[:, None],
            prox[:, None],
            (adj_health / deg_safe)[:, None],
        ],
        axis=1,
    ).astype(np.float32)
    sub_edge_attr = np.stack([link_frac, res_link / cfg.cap_scale], axis=1)
    sub_edge_attr = np.concatenate([sub_edge_attr, sub_edge_attr], axis=0).astype(np.float32)  # both directions
    # ------------------------------------------------------------------ request nodes
    dem = req.node_demand.astype(np.float64)
    deg = np.array([req.degree(i) for i in range(nv)], dtype=np.float64)
    deg_safe_v = np.maximum(deg, 1.0)
    incident_bw = np.zeros(nv, dtype=np.float64)
    placed_nb = np.zeros(nv, dtype=np.float64)
    placed_bw = np.zeros(nv, dtype=np.float64)
    placed = np.zeros(nv, dtype=bool)
    order_pos = np.zeros(nv, dtype=np.float64)
    for pos, v in enumerate(attempt.order):
        placed[v] = True
        order_pos[v] = (pos + 1) / nv
    for l, (i, j) in enumerate(req.edges):
        d = float(req.link_demand[l])
        incident_bw[i] += d
        incident_bw[j] += d
        if placed[j]:
            placed_nb[i] += 1
            placed_bw[i] += d
        if placed[i]:
            placed_nb[j] += 1
            placed_bw[j] += d
    fv_mask = attempt.feasible_virtual_mask()
    req_x = np.concatenate(
        [
            dem / cfg.demand_scale,
            (incident_bw / (deg_safe_v * cfg.demand_scale))[:, None],
            (deg / max(1, nv - 1))[:, None],
            placed.astype(np.float64)[:, None],
            (placed_nb / deg_safe_v)[:, None],
            (placed_bw / (deg_safe_v * cfg.demand_scale))[:, None],
            order_pos[:, None],
            fv_mask.astype(np.float64)[:, None],
            np.full((nv, 1), nv / cfg.max_request_nodes),
        ],
        axis=1,
    ).astype(np.float32)
    if req.num_links:
        redges = np.asarray(req.edges, dtype=np.int64).reshape(-1, 2)
        req_edge_index = np.concatenate([redges.T, redges[:, ::-1].T], axis=1)
        routed = np.array([1.0 if l in attempt.link_paths else 0.0 for l in range(req.num_links)])
        rattr = np.stack([req.link_demand.astype(np.float64) / cfg.demand_scale, routed], axis=1)
        req_edge_attr = np.concatenate([rattr, rattr], axis=0).astype(np.float32)
    else:
        req_edge_index = np.zeros((2, 0), dtype=np.int64)
        req_edge_attr = np.zeros((0, NUM_EDGE_FEATURES), dtype=np.float32)
    # ------------------------------------------------------------------ scalars
    scalars = np.array(
        [
            attempt.step_index / nv,
            nv / cfg.max_request_nodes,
            (attempt.num_routed_links / req.num_links) if req.num_links else 0.0,
            attempt.excess_cost / attempt.revenue,
            n / cfg.sub_size_scale,
            m / cfg.sub_links_scale,
        ],
        dtype=np.float32,
    )
    return StateFeatures(
        req_x=req_x,
        req_edge_index=req_edge_index.astype(np.int64),
        req_edge_attr=req_edge_attr,
        req_placed=placed,
        sub_x=sub_x,
        sub_edge_index=st.edge_index,
        sub_edge_attr=sub_edge_attr,
        sub_used=used,
        scalars=scalars,
        fv_mask=fv_mask,
    )
