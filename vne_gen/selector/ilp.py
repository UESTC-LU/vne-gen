"""Exact candidate-selection ILPs (Eq. 3 and Eq. 4) solved with HiGHS through ``scipy.optimize.milp``.

.. math::
   \\max_f \\sum_{r}\\sum_{m\\in\\mathcal M_r} b_r f_{r,m}
   \\quad\\text{s.t.}\\quad \\sum_{m} f_{r,m}\\le 1\\ \\forall r,\\qquad
   \\sum_{r}\\sum_{m} \\alpha_{r,t} A_r(m,x) f_{r,m} \\le C_S(x)\\ \\forall x\\in X,\\ t\\in T.

For offline VNE there is a single slot with :math:`\\alpha_{r,t}=1` for all requests.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
from scipy.optimize import Bounds, LinearConstraint, milp
from scipy.sparse import coo_matrix, csr_matrix

from .proposal import Proposal, ProposalPool


@dataclass
class SelectionResult:
    selected: List[Proposal] = field(default_factory=list)
    revenue: float = 0.0
    status: str = "no_solution"  # "optimal", "time_limit", "infeasible", "no_solution"
    solve_time_s: float = 0.0
    num_variables: int = 0
    num_constraints: int = 0

    @property
    def feasible(self) -> bool:
        return self.status in ("optimal", "time_limit") and (self.selected or self.revenue == 0.0)


DEFAULT_BACKEND = "highs"


def set_default_backend(name: str) -> None:
    """Choose the MILP backend globally: ``"highs"`` (scipy, default) or ``"gurobi"`` (requires gurobipy)."""
    global DEFAULT_BACKEND
    if name not in ("highs", "gurobi"):
        raise ValueError("backend must be 'highs' or 'gurobi'")
    DEFAULT_BACKEND = name


def solve_selection(
    pools: Sequence[ProposalPool],
    capacity_node: np.ndarray,
    capacity_link: np.ndarray,
    alpha: Optional[np.ndarray] = None,
    time_limit: Optional[float] = 300.0,
    mip_rel_gap: float = 0.0,
    backend: Optional[str] = None,
) -> SelectionResult:
    """Select at most one proposal per request maximising admitted revenue under (time-indexed) capacities.

    ``alpha`` is the ``(num_requests, num_slots)`` activity matrix (``None`` = offline VNE, one slot).
    """
    import time

    backend = backend or DEFAULT_BACKEND
    t0 = time.time()
    props: List[Tuple[int, Proposal]] = []
    for r_idx, pool in enumerate(pools):
        for p in pool:
            props.append((r_idx, p))
    nvar = len(props)
    if nvar == 0:
        return SelectionResult(status="optimal", solve_time_s=time.time() - t0)
    R = len(pools)
    if alpha is None:
        alpha = np.ones((R, 1), dtype=np.int8)
    T = alpha.shape[1]
    n_nodes, n_q = capacity_node.shape
    n_links = capacity_link.shape[0]
    n_x = n_nodes * n_q + n_links
    cap = np.concatenate([capacity_node.reshape(-1), capacity_link]).astype(np.float64)
    # --- objective (maximise revenue -> minimise -revenue)
    c = np.array([-p.revenue for _, p in props], dtype=np.float64)
    # --- per-request constraints
    rows, cols, vals = [], [], []
    for j, (r_idx, _) in enumerate(props):
        rows.append(r_idx)
        cols.append(j)
        vals.append(1.0)
    A_req = csr_matrix(coo_matrix((vals, (rows, cols)), shape=(R, nvar)))
    # --- capacity constraints: row index = t * n_x + x
    rows, cols, vals = [], [], []
    for j, (r_idx, p) in enumerate(props):
        a = np.concatenate([p.allocation.node.reshape(-1), p.allocation.link]).astype(np.float64)
        nz = np.flatnonzero(a)
        if nz.size == 0:
            continue
        active = np.flatnonzero(alpha[r_idx])
        for t in active:
            rows.extend((t * n_x + nz).tolist())
            cols.extend([j] * nz.size)
            vals.extend(a[nz].tolist())
    A_cap = csr_matrix(coo_matrix((vals, (rows, cols)), shape=(T * n_x, nvar)))
    # drop empty capacity rows
    used_rows = np.flatnonzero(np.diff(A_cap.indptr))
    A_cap = A_cap[used_rows]
    ub_cap = np.tile(cap, T)[used_rows]
    result = SelectionResult(num_variables=nvar, num_constraints=R + int(A_cap.shape[0]))
    if backend == "gurobi":
        x, status = _solve_gurobi(c, A_req, A_cap, ub_cap, time_limit, mip_rel_gap)
    else:
        constraints = [LinearConstraint(A_req, -np.inf, 1.0)]
        if A_cap.shape[0] > 0:
            constraints.append(LinearConstraint(A_cap, -np.inf, ub_cap))
        options = {"disp": False, "mip_rel_gap": mip_rel_gap}
        if time_limit is not None:
            options["time_limit"] = float(time_limit)
        res = milp(c, constraints=constraints, integrality=np.ones(nvar), bounds=Bounds(0, 1), options=options)
        x = None if res.x is None else np.asarray(res.x)
        status = "optimal" if res.status == 0 else ("infeasible" if res.status == 2 else "time_limit")
    result.solve_time_s = time.time() - t0
    if x is None:
        result.status = status if status in ("infeasible",) else "no_solution"
        return result
    chosen = [props[j][1] for j in np.flatnonzero(x > 0.5)]
    # a request may only appear once (guard against numerical noise)
    seen = set()
    selected = []
    for p in chosen:
        if p.request_id in seen:
            continue
        seen.add(p.request_id)
        selected.append(p)
    result.selected = selected
    result.revenue = float(sum(p.revenue for p in selected))
    result.status = status
    return result


def _solve_gurobi(c, A_req, A_cap, ub_cap, time_limit, mip_rel_gap):
    """Optional Gurobi backend (``pip install gurobipy``; a licence is required for large models)."""
    import gurobipy as gp  # noqa: WPS433 (optional dependency)
    from gurobipy import GRB

    n = c.shape[0]
    model = gp.Model("vne_selection")
    model.Params.OutputFlag = 0
    if time_limit is not None:
        model.Params.TimeLimit = float(time_limit)
    model.Params.MIPGap = float(mip_rel_gap)
    x = model.addMVar(n, vtype=GRB.BINARY)
    model.setObjective(c @ x, GRB.MINIMIZE)
    model.addConstr(A_req @ x <= np.ones(A_req.shape[0]))
    if A_cap.shape[0] > 0:
        model.addConstr(A_cap @ x <= ub_cap)
    model.optimize()
    if model.SolCount == 0:
        return None, "infeasible" if model.Status == GRB.INFEASIBLE else "no_solution"
    status = "optimal" if model.Status == GRB.OPTIMAL else "time_limit"
    return np.asarray(x.X), status


def check_selection(selected: Sequence[Proposal], capacity_node: np.ndarray, capacity_link: np.ndarray,
                    alpha: Optional[np.ndarray] = None, request_index: Optional[Dict[int, int]] = None) -> bool:
    """Capacity validation of a selection (used after every ILP solve)."""
    T = 1 if alpha is None else alpha.shape[1]
    node = np.zeros((T,) + capacity_node.shape, dtype=np.int64)
    link = np.zeros((T, capacity_link.shape[0]), dtype=np.int64)
    ids = set()
    for p in selected:
        if p.request_id in ids:
            return False
        ids.add(p.request_id)
        if alpha is None:
            slots = [0]
        else:
            slots = np.flatnonzero(alpha[request_index[p.request_id]])
        for t in slots:
            node[t] += p.allocation.node
            link[t] += p.allocation.link
    return bool(np.all(node <= capacity_node[None]) and np.all(link <= capacity_link[None]))
