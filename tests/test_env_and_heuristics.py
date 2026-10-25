import numpy as np
import pytest

from vne_gen.data.collect import collect_corpus
from vne_gen.data.dataset import TrajectoryDataset, collate_examples, load_corpus, save_corpus
from vne_gen.data.features import NUM_REQ_FEATURES, NUM_SUB_FEATURES, encode_state
from vne_gen.env.mapping_env import MappingAttempt, replay_attempt
from vne_gen.env.simulator import run_online
from vne_gen.heuristics import MCTSSolver, make_rank_solver
from vne_gen.problem.feasibility import validate_mapping
from vne_gen.train.return_table import compute_return_table


def test_rewards_and_returns(geant, workload):
    solver = make_rank_solver("nea")
    req = workload[0]
    attempt = solver.run_attempt(req, geant, geant.base_profile())
    assert attempt.success
    n = req.num_nodes
    assert len(attempt.actions) == n
    # Eq. (6): 1/|V_r| for non-terminal feasible steps, R2C + 1/|V_r| at completion
    assert all(g == pytest.approx(1.0 / n) for g in attempt.rewards[:-1])
    assert attempt.rewards[-1] == pytest.approx(attempt.r2c + 1.0 / n)
    rtg = attempt.returns_to_go()
    assert rtg[0] == pytest.approx(1.0 + attempt.r2c)  # completed mappings have return 1 + R2C in (1, 2]
    assert 1.0 < rtg[0] <= 2.0
    assert rtg[-1] == 0.0
    for k in range(n):
        assert rtg[k + 1] == pytest.approx(rtg[k] - attempt.rewards[k])  # Eq. (7)
    validate_mapping(req, attempt.mapping(), geant, geant.base_profile())


def test_failed_attempts_have_return_below_one(geant, workload):
    solver = make_rank_solver("grc")
    profile = geant.base_profile()
    profile.link[:] = 1  # bandwidth too scarce to route anything
    failures = 0
    for req in workload[:30]:
        attempt = solver.run_attempt(req, geant, profile)
        if not attempt.success:
            failures += 1
            assert attempt.returns_to_go()[0] < 1.0
            if attempt.actions:
                assert attempt.rewards[-1] == pytest.approx(-1.0 / req.num_nodes)
            else:
                assert attempt.terminal_reward == pytest.approx(-1.0 / req.num_nodes)
    assert failures > 0


def test_empty_feasible_set_termination(geant, workload):
    req = workload[0]
    profile = geant.base_profile()
    profile.node[:] = 0  # no node can host anything
    attempt = MappingAttempt(geant, req, profile)
    assert attempt.feasible_virtual_nodes() == []
    attempt.terminate_empty()
    assert attempt.done and not attempt.success and attempt.actions == []
    assert attempt.terminal_reward == pytest.approx(-1.0 / req.num_nodes)


def test_replay_matches(geant, workload):
    solver = make_rank_solver("nrm")
    for req in workload[:20]:
        attempt = solver.run_attempt(req, geant, geant.base_profile())
        rec = attempt.record("nrm")
        rep = replay_attempt(geant, rec)
        assert rep.success == attempt.success and rep.node_map == attempt.node_map
        assert rep.link_paths == attempt.link_paths and rep.rewards == pytest.approx(attempt.rewards)


def test_online_simulation_and_validation(geant, workload):
    for name in ("grc", "nrm", "nea"):
        res = run_online(geant, workload, make_rank_solver(name), num_arrivals=200)
        assert res.validation_failures == 0
        assert 0 < len(res.admitted) <= 200
        assert res.total_revenue > 0 and 0 < res.r2c_lt <= 1.0


def test_mcts_runs(geant, workload):
    res = run_online(geant, workload[:20], MCTSSolver(simulations=10, seed=0))
    assert res.validation_failures == 0


def test_features_shapes(geant, workload):
    attempt = MappingAttempt(geant, workload[0], geant.base_profile())
    f = encode_state(attempt)
    assert f.sub_x.shape == (40, NUM_SUB_FEATURES) and f.req_x.shape == (workload[0].num_nodes, NUM_REQ_FEATURES)
    assert f.sub_edge_index.shape == (2, 2 * 61)
    assert np.isfinite(f.sub_x).all() and np.isfinite(f.req_x).all()


def test_corpus_dataset_collate(geant, tmp_path):
    records, meta = collect_corpus(geant, total_attempts=120, seed=5, arrivals_per_episode=100)
    assert meta.num_attempts == 120 and 0 < meta.completed_fraction < 1
    assert set(meta.sources) == {"GRC-Rank", "NRM-Rank", "NEA-Rank"}
    path = str(tmp_path / "corpus.pkl.gz")
    save_corpus(path, geant, records, meta)
    sub2, recs2, meta2 = load_corpus(path)
    assert len(recs2) == 120 and meta2.num_transitions == meta.num_transitions
    table = compute_return_table(recs2)
    assert all(v <= 2.0 for v in table.values())
    ds = TrajectoryDataset(sub2, recs2, context_blocks=3)
    assert len(ds) == meta.num_transitions
    exs = [ds[i] for i in range(0, len(ds), max(1, len(ds) // 16))][:16]
    batch = collate_examples(exs, 3)
    B = len(exs)
    assert batch["win_valid"][:, -1].all()  # current block always present
    assert batch["fv_mask"][np.arange(B), batch["target_v"]].all()
    assert batch["fs_mask"][np.arange(B), batch["target_u"]].all()
    assert batch["req_x"].shape[1] == NUM_REQ_FEATURES and batch["sub_x"].shape[1] == NUM_SUB_FEATURES
    # window returns are consistent with Eq. (7): consecutive valid slots differ by 1/|V_r|
    for b in range(B):
        valid = batch["win_valid"][b]
        r = batch["win_rtg"][b][valid]
        n = batch["request_size"][b]
        for a, c in zip(r[:-1], r[1:]):
            assert a - c == pytest.approx(1.0 / n, abs=1e-5)
