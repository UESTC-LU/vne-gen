"""A compact Monte Carlo tree search baseline for VNE in the spirit of Haeri and Trajkovic [7].

The search explores alternative placement sequences under a bounded simulation budget.  Tree nodes are
partial mapping attempts; an action is a feasible ``(v, u)`` pair (the virtual node is the next one of a
fixed degree-descending order, the substrate node any feasible one).  Rollouts complete the attempt with
uniformly random feasible actions through the common mapping process and return the completed mapping's
revenue-to-cost ratio (``0`` for failed rollouts).  Selection uses UCT.

The official implementation used in the paper is more elaborate (see ``docs/EXTERNAL_BASELINES.md``);
this version is provided so that the full evaluation pipeline runs without external dependencies.
"""

from __future__ import annotations

import math
from typing import Dict, List, Optional, Tuple

import numpy as np

from ..env.mapping_env import MappingAttempt
from ..problem.mapping import Mapping
from ..problem.request import Request
from ..problem.substrate import CapacityProfile, Substrate


class _Node:
    __slots__ = ("actions", "children", "visits", "value", "untried")

    def __init__(self, actions: List[Tuple[int, int]], untried: List[Tuple[int, int]]) -> None:
        self.actions = actions
        self.children: Dict[Tuple[int, int], "_Node"] = {}
        self.visits = 0
        self.value = 0.0
        self.untried = untried


class MCTSSolver:
    name = "MCTS"

    def __init__(self, simulations: int = 64, exploration: float = 1.0, seed: int = 0) -> None:
        self.simulations = int(simulations)
        self.exploration = float(exploration)
        self.rng = np.random.default_rng(seed)

    # ------------------------------------------------------------------ helpers
    @staticmethod
    def _order(request: Request) -> List[int]:
        deg = np.array([request.degree(i) for i in range(request.num_nodes)])
        return [int(i) for i in np.lexsort((np.arange(request.num_nodes), -deg))]

    def _replay(self, request: Request, substrate: Substrate, profile: CapacityProfile,
                actions: List[Tuple[int, int]]) -> MappingAttempt:
        attempt = MappingAttempt(substrate, request, profile)
        for v, u in actions:
            attempt.step(v, u)
        return attempt

    def _candidates(self, attempt: MappingAttempt, order: List[int]) -> List[Tuple[int, int]]:
        if attempt.done:
            return []
        feasible_v = set(attempt.feasible_virtual_nodes())
        v = next((x for x in order if x in feasible_v), None)
        if v is None:
            return []
        return [(v, u) for u in attempt.feasible_substrate_nodes(v)]

    def _rollout(self, attempt: MappingAttempt, order: List[int]) -> float:
        while not attempt.done:
            cands = self._candidates(attempt, order)
            if not cands:
                attempt.terminate_empty()
                break
            v, u = cands[int(self.rng.integers(len(cands)))]
            attempt.step(v, u)
        return attempt.r2c if attempt.success else 0.0

    # ------------------------------------------------------------------ search
    def embed(self, request: Request, substrate: Substrate, profile: CapacityProfile) -> Optional[Mapping]:
        order = self._order(request)
        root_attempt = MappingAttempt(substrate, request, profile)
        root = _Node([], self._candidates(root_attempt, order))
        for _ in range(self.simulations):
            node = root
            attempt = MappingAttempt(substrate, request, profile)
            # selection
            while not node.untried and node.children and not attempt.done:
                best, best_score = None, -1.0
                for a, child in node.children.items():
                    score = child.value / child.visits + self.exploration * math.sqrt(math.log(node.visits + 1) / child.visits)
                    if score > best_score:
                        best, best_score = a, score
                node = node.children[best]
                attempt.step(*best)
            # expansion
            if node.untried and not attempt.done:
                a = node.untried.pop(int(self.rng.integers(len(node.untried))))
                attempt.step(*a)
                child = _Node(node.actions + [a], self._candidates(attempt, order))
                node.children[a] = child
                node = child
            value = attempt.r2c if attempt.success else (self._rollout(attempt, order) if not attempt.done else 0.0)
            # backup along the path of actions
            path = root
            path.visits += 1
            path.value += value
            for a in node.actions:
                path = path.children[a]
                path.visits += 1
                path.value += value
        # choose the most visited successful path: greedy descent by visit count
        node = root
        actions: List[Tuple[int, int]] = []
        while node.children:
            a, child = max(node.children.items(), key=lambda kv: kv[1].visits)
            actions.append(a)
            node = child
        attempt = self._replay(request, substrate, profile, actions)
        if not attempt.done:
            self._rollout(attempt, order)
        return attempt.mapping() if attempt.success else None

    def propose(self, request, substrate, profile, rho: float = 1.0, rng=None):
        return self.embed(request, substrate, profile)
