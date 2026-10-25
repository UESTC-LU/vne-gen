"""The request-level mapping process of Sec. IV-A.

A :class:`MappingAttempt` is the state machine behind Algorithm 1.  At decision ``k`` the learned (or
heuristic) policy selects an action pair ``(v_k, u_k)``: the next virtual node in the mapping order and its
substrate location.  After the placement the newly activated virtual links

.. math:: E_k^+ = \\{(v_k, j)\\in E_r : j\\in U_k\\}

are routed one by one (sorted by non-increasing bandwidth demand, stable with respect to the original
neighbour-iteration order), each route reserving bandwidth before the next link is routed (Eq. 5).  If the
placement or any route is infeasible, all tentative reservations of the decision are rolled back and the
attempt terminates with failure.  Rewards follow Eq. (6):

* ``+1/|V_r|`` for a feasible non-terminal decision,
* ``R2C_r(m) + 1/|V_r|`` for the decision that completes the mapping,
* ``-1/|V_r|`` for the decision whose placement or routing fails,

and an attempt that terminates because :math:`\\mathcal F_V(s_k)=\\emptyset` receives the terminal reward
``-1/|V_r|`` without emitting a new action pair (Eq. 6, terminal term).  The return-to-go follows Eq. (7).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Set, Tuple

import numpy as np

from ..problem.feasibility import feasible_substrate_mask, feasible_virtual_nodes
from ..problem.mapping import Mapping, mapping_cost, r2c
from ..problem.request import Request, request_revenue
from ..problem.routing import min_hop_feasible_path
from ..problem.substrate import CapacityProfile, Substrate


@dataclass
class AttemptRecord:
    """A recorded mapping attempt :math:`\\tau` (one trajectory of the offline corpus)."""

    request: Request
    profile: CapacityProfile  # supplied capacity profile at the start of the attempt
    actions: List[Tuple[int, int]]  # emitted action pairs (v_k, u_k)
    rewards: List[float]  # g_k for every emitted action pair
    terminal_reward: float  # g^tau_term
    success: bool
    source: str = ""

    @property
    def num_steps(self) -> int:
        return len(self.actions)

    @property
    def returns_to_go(self) -> List[float]:
        """``R_k`` for ``k = 0..T_tau`` (Eq. 7); ``R_T = g_term``."""
        rtg = [0.0] * (len(self.rewards) + 1)
        rtg[-1] = float(self.terminal_reward)
        for k in range(len(self.rewards) - 1, -1, -1):
            rtg[k] = rtg[k + 1] + float(self.rewards[k])
        return rtg

    @property
    def initial_return(self) -> float:
        return self.returns_to_go[0]

    def to_dict(self) -> dict:
        return {
            "request": self.request.to_dict(),
            "profile_node": self.profile.node.astype(np.int16).tolist(),
            "profile_link": self.profile.link.astype(np.int16).tolist(),
            "actions": [list(a) for a in self.actions],
            "rewards": [float(g) for g in self.rewards],
            "terminal_reward": float(self.terminal_reward),
            "success": bool(self.success),
            "source": self.source,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "AttemptRecord":
        return cls(
            request=Request.from_dict(d["request"]),
            profile=CapacityProfile(np.asarray(d["profile_node"], dtype=np.int64),
                                    np.asarray(d["profile_link"], dtype=np.int64)),
            actions=[tuple(a) for a in d["actions"]],
            rewards=[float(g) for g in d["rewards"]],
            terminal_reward=float(d["terminal_reward"]),
            success=bool(d["success"]),
            source=d.get("source", ""),
        )


class MappingAttempt:
    """State machine of one request-level mapping attempt (Algorithm 1 without the policy)."""

    def __init__(self, substrate: Substrate, request: Request, profile: CapacityProfile) -> None:
        self.substrate = substrate
        self.request = request
        self.supplied = profile
        self.residual_node = profile.node.astype(np.int64).copy()
        self.residual_link = profile.link.astype(np.int64).copy()
        self.node_map: List[int] = [-1] * request.num_nodes
        self.order: List[int] = []  # placed virtual nodes in mapping order (U_k)
        self.placed: Set[int] = set()
        self.used_substrate: Set[int] = set()
        self.link_paths: Dict[int, List[int]] = {}
        self.actions: List[Tuple[int, int]] = []
        self.rewards: List[float] = []
        self.terminal_reward: float = 0.0
        self.done: bool = False
        self.success: bool = False
        self.failure_reason: Optional[str] = None
        self.revenue = request_revenue(request)
        self.excess_cost = 0.0  # sum over routed links of (hops - 1) * demand
        self._unit = 1.0 / request.num_nodes

    # ------------------------------------------------------------------ queries
    @property
    def step_index(self) -> int:
        """Decision index ``k`` (number of placed virtual nodes)."""
        return len(self.order)

    @property
    def num_routed_links(self) -> int:
        return len(self.link_paths)

    def residual_profile(self) -> CapacityProfile:
        return CapacityProfile(self.residual_node.copy(), self.residual_link.copy())

    def feasible_virtual_nodes(self) -> List[int]:
        return feasible_virtual_nodes(self.request, self.placed, self.used_substrate, self.residual_node)

    def feasible_virtual_mask(self) -> np.ndarray:
        mask = np.zeros(self.request.num_nodes, dtype=bool)
        mask[self.feasible_virtual_nodes()] = True
        return mask

    def feasible_substrate_mask(self, v: int) -> np.ndarray:
        return feasible_substrate_mask(self.request, v, self.used_substrate, self.residual_node)

    def feasible_substrate_nodes(self, v: int) -> List[int]:
        return [int(u) for u in np.flatnonzero(self.feasible_substrate_mask(v))]

    def newly_activated_links(self, v: int) -> List[int]:
        """Link ids of :math:`E_k^+` in routing order (non-increasing demand, stable)."""
        links = [l for j, l in self.request.adjacency[v] if j in self.placed]
        links.sort(key=lambda l: -int(self.request.link_demand[l]))  # stable sort keeps neighbour order for ties
        return links

    # ------------------------------------------------------------------ transitions
    def terminate_empty(self) -> float:
        """Terminate because :math:`\\mathcal F_V(s_k)=\\emptyset` (no action pair is emitted)."""
        assert not self.done
        self.done = True
        self.success = False
        self.failure_reason = "empty_feasible_set"
        self.terminal_reward = -self._unit
        return self.terminal_reward

    def step(self, v: int, u: int) -> Tuple[float, bool]:
        """Place ``v`` on ``u``, route the newly activated links and return ``(reward, done)``."""
        assert not self.done, "attempt already terminated"
        req = self.request
        if v in self.placed:
            raise ValueError(f"virtual node {v} already placed")
        self.actions.append((int(v), int(u)))
        # placement checks (injectivity and residual node capacity)
        if u in self.used_substrate or not np.all(self.residual_node[u] >= req.node_demand[v]):
            return self._fail("placement_infeasible")
        tentative_link = self.residual_link.copy()
        new_paths: Dict[int, List[int]] = {}
        excess = 0.0
        for l in self.newly_activated_links(v):
            i, j = req.edges[l]
            other = j if i == v else i
            demand = int(req.link_demand[l])
            path = min_hop_feasible_path(self.substrate, u, self.node_map[other], demand, tentative_link)
            if path is None:
                return self._fail("routing_infeasible")
            for a, b in zip(path[:-1], path[1:]):
                tentative_link[self.substrate.edge_id(a, b)] -= demand
            # store paths oriented from node_map[i] to node_map[j] for edges[l] = (i, j)
            new_paths[l] = path if v == i else path[::-1]
            excess += (len(path) - 2) * demand
        # commit placement and routes
        self.residual_node[u] -= req.node_demand[v]
        self.residual_link = tentative_link
        self.node_map[v] = int(u)
        self.order.append(int(v))
        self.placed.add(int(v))
        self.used_substrate.add(int(u))
        self.link_paths.update(new_paths)
        self.excess_cost += excess
        if len(self.placed) == req.num_nodes:
            self.done = True
            self.success = True
            reward = self.r2c + self._unit
        else:
            reward = self._unit
        self.rewards.append(float(reward))
        return float(reward), self.done

    def _fail(self, reason: str) -> Tuple[float, bool]:
        # tentative reservations were never committed, so the residual state is unchanged (rollback)
        self.done = True
        self.success = False
        self.failure_reason = reason
        reward = -self._unit
        self.rewards.append(float(reward))
        return float(reward), True

    # ------------------------------------------------------------------ results
    @property
    def cost_so_far(self) -> float:
        return self.revenue + self.excess_cost

    @property
    def r2c(self) -> float:
        return self.revenue / self.cost_so_far

    def mapping(self) -> Mapping:
        assert self.success, "mapping() is only available for completed attempts"
        return Mapping(node_map=list(self.node_map), link_paths=dict(self.link_paths))

    def returns_to_go(self) -> List[float]:
        rtg = [0.0] * (len(self.rewards) + 1)
        rtg[-1] = float(self.terminal_reward)
        for k in range(len(self.rewards) - 1, -1, -1):
            rtg[k] = rtg[k + 1] + float(self.rewards[k])
        return rtg

    def record(self, source: str = "") -> AttemptRecord:
        return AttemptRecord(
            request=self.request,
            profile=self.supplied.copy(),
            actions=list(self.actions),
            rewards=list(self.rewards),
            terminal_reward=float(self.terminal_reward),
            success=bool(self.success),
            source=source,
        )


def replay_attempt(substrate: Substrate, record: AttemptRecord, upto: Optional[int] = None) -> MappingAttempt:
    """Re-execute the first ``upto`` action pairs of a recorded attempt and return the resulting state."""
    attempt = MappingAttempt(substrate, record.request, record.profile)
    n = len(record.actions) if upto is None else int(upto)
    for k in range(n):
        v, u = record.actions[k]
        _, done = attempt.step(v, u)
        if done and k < n - 1:
            raise RuntimeError("recorded attempt terminated earlier than expected during replay")
    return attempt
