import numpy as np
import pytest

from vne_gen.env.simulator import run_online
from vne_gen.eval.optimality import OptimalityConfig, full_space_optimum, instance_for
from vne_gen.heuristics import make_rank_solver
from vne_gen.problem.generators import generate_request_set, generate_workload
from vne_gen.selector import CandidatePoolSelector, PortfolioSource, Proposal, ProposalPool, SelectorConfig
from vne_gen.selector.ilp import check_selection, solve_selection
from vne_gen.selector.timeslots import activity_matrix, elementary_slots


def _proposals(geant, requests, solver):
    pools = []
    for r in requests:
        pool = ProposalPool(r.id)
        m = solver.embed(r, geant, geant.base_profile())
        if m is not None:
            pool.add(Proposal.build(r, m, geant))
        pools.append(pool)
    return pools


def test_dedup_by_signature(geant):
    reqs = generate_request_set(np.random.default_rng(0), 3)
    solver = make_rank_solver("nea")
    pool = ProposalPool(reqs[0].id)
    m = solver.embed(reqs[0], geant, geant.base_profile())
    assert pool.add(Proposal.build(reqs[0], m, geant))
    assert not pool.add(Proposal.build(reqs[0], m, geant))  # identical allocation signature
    assert len(pool) == 1


def test_offline_ilp_respects_capacity(geant):
    reqs = generate_request_set(np.random.default_rng(2), 40)
    pools = _proposals(geant, reqs, make_rank_solver("nea"))
    res = solve_selection(pools, geant.node_cap, geant.link_cap, time_limit=60)
    assert res.status == "optimal"
    assert check_selection(res.selected, geant.node_cap, geant.link_cap)
    assert res.revenue == pytest.approx(sum(p.revenue for p in res.selected))
    assert len({p.request_id for p in res.selected}) == len(res.selected)


def test_time_aware_ilp_reuses_resources_over_time(geant):
    reqs = generate_workload(np.random.default_rng(3), 30, 0.05)
    slots = elementary_slots(reqs)
    alpha = activity_matrix(reqs, slots)
    assert alpha.shape == (30, len(slots)) and alpha.sum(axis=1).min() >= 1
    pools = _proposals(geant, reqs, make_rank_solver("nea"))
    off = solve_selection(pools, geant.node_cap, geant.link_cap, time_limit=60)
    ta = solve_selection(pools, geant.node_cap, geant.link_cap, alpha=alpha, time_limit=60)
    index = {r.id: i for i, r in enumerate(reqs)}
    assert check_selection(ta.selected, geant.node_cap, geant.link_cap, alpha, index)
    assert ta.revenue >= off.revenue - 1e-6  # non-overlapping requests may share resources


def test_algorithm2_improves_over_sequential(geant):
    reqs = generate_request_set(np.random.default_rng(4), 30)
    nea = make_rank_solver("nea")
    seq = run_online(geant, reqs, nea, with_departures=False).total_revenue
    cfg = SelectorConfig(setting="off", rounds=3, fixed_rounds=True, time_limit=60)
    out = CandidatePoolSelector(geant, reqs, nea, cfg).run()
    assert len(out.rounds) == 3
    assert out.revenue >= min(seq, out.rounds[0].revenue) - 1e-6
    assert out.rounds[-1].revenue >= out.rounds[0].revenue - 1e-6
    # base profile control: the deterministic heuristic cannot produce new signatures
    cfg_base = SelectorConfig(setting="off", rounds=2, fixed_rounds=True, time_limit=60, profile_mode="base")
    base = CandidatePoolSelector(geant, reqs, nea, cfg_base).run()
    assert base.rounds[1].num_new_signatures == 0
    # early stopping stops once a round does not improve
    cfg_es = SelectorConfig(setting="off", rounds=5, fixed_rounds=False, time_limit=60, profile_mode="base")
    es = CandidatePoolSelector(geant, reqs, nea, cfg_es).run()
    assert len(es.rounds) == 2


def test_portfolio_source(geant):
    reqs = generate_request_set(np.random.default_rng(5), 10)
    port = PortfolioSource([make_rank_solver(n) for n in ("grc", "nrm", "nea")])
    cfg = SelectorConfig(setting="off", rounds=1, time_limit=30)
    out = CandidatePoolSelector(geant, reqs, port, cfg).run()
    assert out.rounds[0].num_calls == 30


def test_full_space_optimum_dominates_selector():
    cfg = OptimalityConfig(time_limit=120)
    sub, reqs = instance_for(2014, cfg)
    ref = full_space_optimum(sub, reqs, time_limit=120)
    assert ref.status == "optimal" and ref.revenue > 0
    scfg = SelectorConfig(setting="off", rounds=3, time_limit=60)
    out = CandidatePoolSelector(sub, reqs, make_rank_solver("nea"), scfg).run()
    assert out.revenue <= ref.revenue + 1e-6
