"""Model, generator and training tests (skipped when PyTorch / transformers are not installed)."""

import numpy as np
import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("transformers")

from vne_gen.data.collect import collect_corpus  # noqa: E402
from vne_gen.data.dataset import TrajectoryDataset, collate_examples  # noqa: E402
from vne_gen.data.features import FeatureConfig  # noqa: E402
from vne_gen.env.simulator import run_online  # noqa: E402
from vne_gen.generator.generator import VNEGenGenerator, load_generator  # noqa: E402
from vne_gen.model.config import ModelConfig  # noqa: E402
from vne_gen.model.vne_gen import VNEGenModel, batch_to_tensors  # noqa: E402
from vne_gen.problem.generators import waxman_substrate  # noqa: E402
from vne_gen.selector import CandidatePoolSelector, SelectorConfig  # noqa: E402
from vne_gen.train.return_table import compute_return_table  # noqa: E402
from vne_gen.train.trainer import TrainConfig, Trainer, save_checkpoint  # noqa: E402

TINY = dict(gnn_hidden=16, gnn_layers=2, backbone="gpt2", pretrained=None, d_model=32, n_layer=2, n_head=2,
            context_blocks=3, lora_rank=4, lora_alpha=4.0, pointer_dim=16, return_feat_dim=4, dropout=0.0)


@pytest.fixture(scope="module")
def corpus(geant):
    records, meta = collect_corpus(geant, total_attempts=150, seed=11, arrivals_per_episode=100)
    return records


@pytest.fixture(scope="module")
def batch(geant, corpus):
    ds = TrajectoryDataset(geant, corpus, context_blocks=3)
    exs = [ds[i] for i in range(0, len(ds), max(1, len(ds) // 12))][:12]
    return collate_examples(exs, 3)


def _model(**overrides):
    cfg = ModelConfig.from_dict({**TINY, **overrides})
    return VNEGenModel(cfg)


@pytest.mark.parametrize("overrides", [{}, {"shared_adapter": True}, {"message_passing": False},
                                       {"backbone": "mlp"}, {"return_conditioning": False}])
def test_forward_and_losses(batch, overrides):
    model = _model(**overrides)
    tb = batch_to_tensors(batch, torch.device("cpu"))
    out = model(tb)
    B = batch["target_v"].shape[0]
    assert out["logits_V"].shape == (B, batch["fv_mask"].shape[1])
    assert out["logits_S"].shape == (B, batch["fs_mask"].shape[1])
    # infeasible candidates receive (numerically) zero probability
    probs_v = torch.softmax(out["logits_V"], -1)
    assert torch.all(probs_v[~tb["fv_mask"].bool()] < 1e-6)
    losses = model.losses(out, tb)
    assert torch.isfinite(losses["loss"])
    losses["loss"].backward()
    grads = [p.grad for p in model.trainable_parameters() if p.grad is not None]
    assert grads, "no gradients reached the trainable parameters"


def test_backbone_frozen_and_lora_trainable():
    model = _model()
    for name, p in model.backbone.gpt.named_parameters():
        if ".A." in name or ".B." in name:
            assert p.requires_grad, name
        else:
            assert not p.requires_grad, name
    total, trainable = model.parameter_counts()
    assert 0 < trainable < total


def test_generator_online_offline_and_checkpoint(geant, corpus, workload, tmp_path):
    model = _model()
    table = compute_return_table(corpus)
    gen = VNEGenGenerator(model, table, torch.device("cpu"))
    res = run_online(geant, workload[:30], gen)
    assert res.validation_failures == 0
    # stochastic proposals under a tighter supplied profile stay feasible
    profile = geant.base_profile()
    profile.node //= 2
    for req in workload[:5]:
        m = gen.propose(req, geant, profile, rho=1.0, rng=np.random.default_rng(0))
        if m is not None:
            from vne_gen.problem.feasibility import validate_mapping

            validate_mapping(req, m, geant, profile)
    # fixed orderings (ablation) run without retraining
    for ordering in ("ascending_id", "degree_desc", "fixed_random"):
        g2 = VNEGenGenerator(model, table, torch.device("cpu"), ordering=ordering)
        run_online(geant, workload[:10], g2)
    # generalization to an unseen substrate size
    wax = waxman_substrate(100, np.random.default_rng(0))
    run_online(wax, workload[:10], gen)
    # save / load round trip
    path = str(tmp_path / "ckpt.pt")
    save_checkpoint(path, model, table, TrainConfig(), FeatureConfig())
    gen2 = load_generator(path, device="cpu")
    assert gen2.return_table == {int(k): float(v) for k, v in table.items()}
    out1 = run_online(geant, workload[:15], gen).total_revenue
    out2 = run_online(geant, workload[:15], gen2).total_revenue
    assert out1 == pytest.approx(out2)
    # joint selection with the generator as proposal source
    from vne_gen.problem.generators import generate_request_set

    reqs = generate_request_set(np.random.default_rng(1), 8)
    out = CandidatePoolSelector(geant, reqs, gen2, SelectorConfig(setting="off", rounds=2, time_limit=30)).run()
    assert out.revenue >= 0


def test_training_steps(geant, corpus, tmp_path):
    model = _model()
    ds = TrajectoryDataset(geant, corpus, context_blocks=3)
    cfg = TrainConfig(batch_size=8, max_steps=5, warmup_steps=2, num_workers=0, log_every=1, eval_every=100)
    trainer = Trainer(model, cfg, device=torch.device("cpu"), out_dir=str(tmp_path))
    ckpt = trainer.fit(ds, compute_return_table(corpus), FeatureConfig(), val_dataset=None)
    assert trainer.step == 5
    gen = load_generator(ckpt, device="cpu")
    assert gen.model.K == 3
