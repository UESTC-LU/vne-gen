"""Small-instance optimality analysis (Sec. VI-H).

The full-space reference is a joint integer program that maximises admitted revenue over all feasible request
admissions, injective node placements and unsplittable link routes under the CPU, GPU, RAM and bandwidth
constraints of Sec. III.  For every instance the exact gap of a method with revenue ``J`` is

.. math:: \\mathrm{Gap} = \\frac{J^\\star_{full} - J}{J^\\star_{full}}\\times 100\\%.
"""

from __future__ import annotations

import json
import logging
import os
import time
from dataclasses import asdict, dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
from scipy.optimize import Bounds, LinearConstraint, milp
from scipy.sparse import coo_matrix, csr_matrix

from ..problem.feasibility import validate_mapping
from ..problem.generators import WorkloadConfig, generate_request_set, random_connected_substrate
from ..problem.mapping import Mapping
from ..problem.request import Request, request_revenue
from ..problem.substrate import NUM_RESOURCES, Substrate
from ..selector.offline import CandidatePoolSelector, SelectorConfig

logger = logging.getLogger(__name__)


@dataclass
class OptimalityConfig:
    seeds: Sequence[int] = (2014, 2015, 2016, 2017, 2018, 2019, 2020)
    substrate_nodes: int = 8
    substrate_edge_prob: float = 0.45
    num_requests: int = 8
    min_nodes: int = 2
    max_nodes: int = 4
    rounds: int = 5
    schedule: Sequence = (1.0,)
    time_limit: Optional[float] = 600.0  # full-space ILP time limit
    selector_time_limit: Optional[float] = 300.0


@dataclass
class FullSpaceResult:
    revenue: float
    status: str
    admitted: List[int]
    mappings: Dict[int, Mapping]
    solve_time_s: float


def instance_for(seed: int, cfg: OptimalityConfig) -> Tuple[Substrate, List[Request]]:
    rng = np.random.default_rng(seed)
    substrate = random_connected_substrate(cfg.substrate_nodes, cfg.substrate_edge_prob, rng, name=f"ER-{cfg.substrate_nodes}")
    wcfg = WorkloadConfig(min_nodes=cfg.min_nodes, max_nodes=cfg.max_nodes)
    requests = generate_request_set(rng, cfg.num_requests, wcfg)
    return substrate, requests


def full_space_optimum(substrate: Substrate, requests: Sequence[Request], time_limit: Optional[float] = 600.0) -> FullSpaceResult:
    """Solve the joint admission + placement + unsplittable-routing ILP with HiGHS."""
    t0 = time.time()
    n, m = substrate.num_nodes, substrate.num_links
    arcs = []  # arc index -> (tail, head, edge id)
    for e, (u, v) in enumerate(substrate.edges):
        arcs.append((u, v, e))
        arcs.append((v, u, e))
    A = len(arcs)
    out_arcs = [[] for _ in range(n)]
    in_arcs = [[] for _ in range(n)]
    for a, (u, v, _) in enumerate(arcs):
        out_arcs[u].append(a)
        in_arcs[v].append(a)
    # ---- variable layout
    z_idx = {}
    x_idx = {}
    y_idx = {}
    nvar = 0
    for r, req in enumerate(requests):
        z_idx[r] = nvar
        nvar += 1
    for r, req in enumerate(requests):
        for i in range(req.num_nodes):
            for u in range(n):
                x_idx[(r, i, u)] = nvar
                nvar += 1
    for r, req in enumerate(requests):
        for l in range(req.num_links):
            for a in range(A):
                y_idx[(r, l, a)] = nvar
                nvar += 1
    c = np.zeros(nvar)
    for r, req in enumerate(requests):
        c[z_idx[r]] = -request_revenue(req)
    rows, cols, vals, lbs, ubs = [], [], [], [], []
    row = 0

    def add_row(entries, lb, ub):
        nonlocal row
        for j, v in entries:
            rows.append(row)
            cols.append(j)
            vals.append(float(v))
        lbs.append(lb)
        ubs.append(ub)
        row += 1

    for r, req in enumerate(requests):
        # each virtual node placed exactly once iff admitted
        for i in range(req.num_nodes):
            add_row([(x_idx[(r, i, u)], 1.0) for u in range(n)] + [(z_idx[r], -1.0)], 0.0, 0.0)
        # injectivity within the request
        for u in range(n):
            add_row([(x_idx[(r, i, u)], 1.0) for i in range(req.num_nodes)] + [(z_idx[r], -1.0)], -np.inf, 0.0)
        # flow conservation for every virtual link and substrate node
        for l, (i, j) in enumerate(req.edges):
            for u in range(n):
                entries = [(y_idx[(r, l, a)], 1.0) for a in out_arcs[u]] + [(y_idx[(r, l, a)], -1.0) for a in in_arcs[u]]
                entries += [(x_idx[(r, i, u)], -1.0), (x_idx[(r, j, u)], 1.0)]
                add_row(entries, 0.0, 0.0)
            # a link uses at most one outgoing arc per node (simple paths) and only when admitted
            for u in range(n):
                add_row([(y_idx[(r, l, a)], 1.0) for a in out_arcs[u]] + [(z_idx[r], -1.0)], -np.inf, 0.0)
    # node capacities
    for u in range(n):
        for q in range(NUM_RESOURCES):
            entries = [(x_idx[(r, i, u)], float(req.node_demand[i, q])) for r, req in enumerate(requests) for i in range(req.num_nodes)]
            add_row(entries, -np.inf, float(substrate.node_cap[u, q]))
    # link capacities
    for e in range(m):
        entries = []
        for r, req in enumerate(requests):
            for l in range(req.num_links):
                d = float(req.link_demand[l])
                entries.append((y_idx[(r, l, 2 * e)], d))
                entries.append((y_idx[(r, l, 2 * e + 1)], d))
        add_row(entries, -np.inf, float(substrate.link_cap[e]))
    Acon = csr_matrix(coo_matrix((vals, (rows, cols)), shape=(row, nvar)))
    options = {"disp": False, "mip_rel_gap": 0.0}
    if time_limit is not None:
        options["time_limit"] = float(time_limit)
    res = milp(c, constraints=[LinearConstraint(Acon, np.asarray(lbs), np.asarray(ubs))], integrality=np.ones(nvar),
               bounds=Bounds(0, 1), options=options)
    if res.x is None:
        return FullSpaceResult(0.0, "no_solution", [], {}, time.time() - t0)
    x = np.asarray(res.x)
    admitted = [r for r in range(len(requests)) if x[z_idx[r]] > 0.5]
    mappings: Dict[int, Mapping] = {}
    for r in admitted:
        req = requests[r]
        node_map = [int(np.argmax([x[x_idx[(r, i, u)]] for u in range(n)])) for i in range(req.num_nodes)]
        paths = {}
        for l, (i, j) in enumerate(req.edges):
            nxt = {}
            for a, (u, v, _) in enumerate(arcs):
                if x[y_idx[(r, l, a)]] > 0.5:
                    nxt[u] = v
            path = [node_map[i]]
            seen = {node_map[i]}
            while path[-1] != node_map[j]:
                v = nxt[path[-1]]
                if v in seen:  # should not happen with the simple-path constraints
                    break
                seen.add(v)
                path.append(v)
            paths[l] = path
        mapping = Mapping(node_map=node_map, link_paths=paths)
        validate_mapping(req, mapping, substrate, substrate.base_profile())
        mappings[req.id] = mapping
    revenue = float(sum(request_revenue(requests[r]) for r in admitted))
    status = "optimal" if res.status == 0 else "time_limit"
    return FullSpaceResult(revenue, status, [requests[r].id for r in admitted], mappings, time.time() - t0)


def evaluate_optimality(sources: Dict[str, object], cfg: OptimalityConfig, out_path: Optional[str] = None) -> dict:
    """Compare every proposal source (through the offline selector) with the full-space optimum."""
    rows: List[dict] = []
    for seed in cfg.seeds:
        substrate, requests = instance_for(seed, cfg)
        ref = full_space_optimum(substrate, requests, cfg.time_limit)
        logger.info("seed %d | full-space optimum %.3f (%s, %.1fs)", seed, ref.revenue, ref.status, ref.solve_time_s)
        for name, source in sources.items():
            scfg = SelectorConfig(setting="off", rounds=cfg.rounds, schedule=cfg.schedule, fixed_rounds=False,
                                  time_limit=cfg.selector_time_limit, seed=seed)
            out = CandidatePoolSelector(substrate, requests, source, scfg).run()
            gap = 100.0 * (ref.revenue - out.revenue) / ref.revenue if ref.revenue > 0 else 0.0
            rows.append({"seed": seed, "source": name, "revenue": out.revenue, "full_space_optimum": ref.revenue,
                         "reference_status": ref.status, "gap_pct": gap})
            logger.info("  %s | revenue %.3f | gap %.3f%%", name, out.revenue, gap)
    summary = {}
    for name in sources:
        rs = [r for r in rows if r["source"] == name]
        tot = sum(r["revenue"] for r in rs)
        tot_ref = sum(r["full_space_optimum"] for r in rs)
        summary[name] = {
            "mean_revenue": float(np.mean([r["revenue"] for r in rs])),
            "mean_optimum": float(np.mean([r["full_space_optimum"] for r in rs])),
            "mean_gap_pct": float(np.mean([r["gap_pct"] for r in rs])),
            "max_gap_pct": float(np.max([r["gap_pct"] for r in rs])),
            "num_optimal": int(sum(1 for r in rs if r["gap_pct"] <= 1e-9)),
            "aggregate_gap_pct": 100.0 * (tot_ref - tot) / tot_ref if tot_ref > 0 else 0.0,
        }
    result = {"config": asdict(cfg), "rows": rows, "summary": summary}
    if out_path:
        os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
        with open(out_path, "w") as f:
            json.dump(result, f, indent=1, default=float)
    return result
