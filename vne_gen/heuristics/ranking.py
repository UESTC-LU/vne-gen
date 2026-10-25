"""Node-ranking heuristics used as online baselines and as sources of the offline corpus (Sec. IV-C).

All ranking solvers share the same procedure:

1. rank the virtual nodes of the request (descending score) - this fixes the mapping order;
2. rank the substrate nodes under the supplied capacity profile (descending score);
3. place each virtual node, in rank order, on the highest-ranked unused substrate node whose residual
   node resources fit it;
4. route the newly activated virtual links right after each placement through the common mapping process
   (:class:`~vne_gen.env.mapping_env.MappingAttempt`, min-hop feasible routing with immediate reservation).

Running the heuristics through the shared mapping process makes every heuristic trajectory an ordered
sequence of ``(v_k, u_k)`` pairs with the rewards of Eq. (6), i.e. directly usable as offline training data,
and guarantees that heuristic and generator mappings are validated identically.  Attempts whose routing
fails are *retained* as terminated trajectories.

The three scores follow the cited papers, generalised to the multi-dimensional node-resource model of
Sec. III-A (node resources are averaged over the dimensions in ``Q`` before being combined with bandwidth):

* **GRC-Rank** [4]: global resource capacity, the stationary vector of
  ``r = (1 - d) h + d M r`` with ``h`` the normalised node resource and ``M`` the bandwidth-weighted
  transition matrix (damping ``d = 0.85``).
* **NRM-Rank** [5]: node resource management score, the product of the node resource and the sum of the
  adjacent link bandwidth.
* **NEA-Rank** [6]: node essentiality assessment, the NRM-style resource score scaled by the resource
  essentiality of the neighbourhood (each adjacent link is weighted by the relative resource of the
  neighbour it leads to), which favours nodes whose surroundings can also host the remaining virtual nodes.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Sequence, Tuple, Type

import numpy as np

from ..env.mapping_env import MappingAttempt
from ..problem.mapping import Mapping
from ..problem.request import Request
from ..problem.substrate import CapacityProfile, Substrate


def _mean_resource(x: np.ndarray) -> np.ndarray:
    return np.asarray(x, dtype=np.float64).mean(axis=1)


def _adjacent_bandwidth(num_nodes: int, edges: Sequence[Tuple[int, int]], bw: np.ndarray) -> np.ndarray:
    out = np.zeros(num_nodes, dtype=np.float64)
    for l, (a, b) in enumerate(edges):
        out[a] += float(bw[l])
        out[b] += float(bw[l])
    return out


def grc_scores(num_nodes: int, edges: Sequence[Tuple[int, int]], node_res: np.ndarray, bw: np.ndarray,
               damping: float = 0.85, tol: float = 1e-6, max_iter: int = 1000) -> np.ndarray:
    """Global resource capacity (GRC) vector [4]."""
    res = _mean_resource(node_res)
    total = res.sum()
    h = res / total if total > 0 else np.full(num_nodes, 1.0 / num_nodes)
    M = np.zeros((num_nodes, num_nodes), dtype=np.float64)
    for l, (a, b) in enumerate(edges):
        M[a, b] += float(bw[l])
        M[b, a] += float(bw[l])
    col = M.sum(axis=0)
    with np.errstate(divide="ignore", invalid="ignore"):
        M = np.where(col > 0, M / col, 0.0)
    r = h.copy()
    for _ in range(max_iter):
        r_new = (1.0 - damping) * h + damping * (M @ r)
        if np.abs(r_new - r).max() < tol:
            r = r_new
            break
        r = r_new
    return r


def nrm_scores(num_nodes: int, edges: Sequence[Tuple[int, int]], node_res: np.ndarray, bw: np.ndarray) -> np.ndarray:
    """Node resource management (NRM) score [5]: node resource x adjacent bandwidth."""
    return _mean_resource(node_res) * _adjacent_bandwidth(num_nodes, edges, bw)


def nea_scores(num_nodes: int, edges: Sequence[Tuple[int, int]], node_res: np.ndarray, bw: np.ndarray) -> np.ndarray:
    """Node essentiality assessment (NEA) score [6]: resource score scaled by neighbourhood essentiality."""
    res = _mean_resource(node_res)
    max_res = res.max() if res.size and res.max() > 0 else 1.0
    score = np.zeros(num_nodes, dtype=np.float64)
    for l, (a, b) in enumerate(edges):
        score[a] += float(bw[l]) * (1.0 + res[b] / max_res)
        score[b] += float(bw[l]) * (1.0 + res[a] / max_res)
    return res * score


class RankSolver:
    """Base class of the ranking heuristics (see module docstring)."""

    name = "rank"

    def score(self, num_nodes: int, edges: Sequence[Tuple[int, int]], node_res: np.ndarray,
              bw: np.ndarray) -> np.ndarray:  # pragma: no cover - abstract
        raise NotImplementedError

    def virtual_order(self, request: Request) -> List[int]:
        s = self.score(request.num_nodes, request.edges, request.node_demand, request.link_demand)
        return [int(i) for i in np.lexsort((np.arange(request.num_nodes), -s))]

    def substrate_order(self, substrate: Substrate, profile: CapacityProfile) -> List[int]:
        s = self.score(substrate.num_nodes, substrate.edges, profile.node, profile.link)
        return [int(u) for u in np.lexsort((np.arange(substrate.num_nodes), -s))]

    def run_attempt(self, request: Request, substrate: Substrate, profile: CapacityProfile) -> MappingAttempt:
        """Execute the heuristic through the common mapping process and return the (possibly failed) attempt."""
        attempt = MappingAttempt(substrate, request, profile)
        v_order = self.virtual_order(request)
        u_order = self.substrate_order(substrate, profile)
        for v in v_order:
            feasible_v = attempt.feasible_virtual_nodes()
            if not feasible_v:
                attempt.terminate_empty()
                break
            if v not in feasible_v:
                # the ranked node cannot be placed anywhere: emit the pair on the best unused substrate node so
                # that the failure is recorded as a placement-infeasible decision (lower return, Sec. IV-A)
                u = next(u for u in u_order if u not in attempt.used_substrate)
                attempt.step(v, u)
                break
            fits = attempt.feasible_substrate_mask(v)
            u = next(u for u in u_order if fits[u])
            _, done = attempt.step(v, u)
            if done:
                break
        return attempt

    # Solver protocol ------------------------------------------------------------------
    def embed(self, request: Request, substrate: Substrate, profile: CapacityProfile) -> Optional[Mapping]:
        attempt = self.run_attempt(request, substrate, profile)
        return attempt.mapping() if attempt.success else None

    # ProposalSource protocol (offline / time-aware selectors) -----------------------------
    def propose(self, request: Request, substrate: Substrate, profile: CapacityProfile,
                rho: float = 1.0, rng: Optional[np.random.Generator] = None) -> Optional[Mapping]:
        return self.embed(request, substrate, profile)


class GRCRank(RankSolver):
    name = "GRC-Rank"

    def __init__(self, damping: float = 0.85) -> None:
        self.damping = damping

    def score(self, num_nodes, edges, node_res, bw):
        return grc_scores(num_nodes, edges, node_res, bw, damping=self.damping)


class NRMRank(RankSolver):
    name = "NRM-Rank"

    def score(self, num_nodes, edges, node_res, bw):
        return nrm_scores(num_nodes, edges, node_res, bw)


class NEARank(RankSolver):
    name = "NEA-Rank"

    def score(self, num_nodes, edges, node_res, bw):
        return nea_scores(num_nodes, edges, node_res, bw)


RANK_SOLVERS: Dict[str, Type[RankSolver]] = {"grc": GRCRank, "nrm": NRMRank, "nea": NEARank}


def make_rank_solver(name: str) -> RankSolver:
    key = name.lower().replace("-rank", "").replace("_rank", "")
    if key not in RANK_SOLVERS:
        raise KeyError(f"unknown ranking heuristic '{name}' (choose from {sorted(RANK_SOLVERS)})")
    return RANK_SOLVERS[key]()
