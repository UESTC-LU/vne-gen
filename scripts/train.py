#!/usr/bin/env python3
"""Offline training of VNE-Gen (Sec. IV-C).

Examples:
    python scripts/train.py --config configs/paper.yaml --corpus data/corpus_geant.pkl.gz --out runs/paper
    python scripts/train.py --config configs/tiny.yaml  --corpus data/corpus_small.pkl.gz --out runs/tiny
    # ablations (Sec. VI-C)
    python scripts/train.py --config configs/paper.yaml --corpus ... --out runs/shared_adapter --set model.shared_adapter=true
    python scripts/train.py --config configs/paper.yaml --corpus ... --out runs/no_mp         --set model.message_passing=false
    python scripts/train.py --config configs/paper.yaml --corpus ... --out runs/bc            --set model.backbone=mlp
    python scripts/train.py --config configs/paper.yaml --corpus ... --out runs/no_return     --set model.return_conditioning=false
    python scripts/train.py --config configs/paper.yaml --corpus ... --out runs/completed    --set train.completed_only=true
"""

from __future__ import annotations

import argparse
import logging
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from vne_gen.data.dataset import TrajectoryDataset, load_corpus  # noqa: E402
from vne_gen.data.features import FeatureConfig  # noqa: E402
from vne_gen.train.return_table import compute_return_table  # noqa: E402
from vne_gen.utils import configs_from_yaml, parse_overrides, seed_everything, setup_logging  # noqa: E402


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--config", default="configs/paper.yaml")
    p.add_argument("--corpus", required=True)
    p.add_argument("--out", default="runs/vne_gen")
    p.add_argument("--device", default=None)
    p.add_argument("--set", nargs="*", default=[], help="overrides, e.g. model.lora_rank=32 train.epochs=5")
    args = p.parse_args()
    setup_logging(log_file=os.path.join(args.out, "train.log"))
    log = logging.getLogger("train")
    model_cfg, train_cfg, _ = configs_from_yaml(args.config, parse_overrides(args.set))
    seed_everything(train_cfg.seed)

    import torch  # noqa: WPS433

    from vne_gen.model.vne_gen import VNEGenModel
    from vne_gen.train.trainer import Trainer, split_records

    substrate, records, meta = load_corpus(args.corpus)
    log.info("corpus: %d attempts, %d transitions, %.1f%% completed", meta.num_attempts, meta.num_transitions,
             100 * meta.completed_fraction)
    return_table = compute_return_table(records)
    log.info("R*(n) table: %s", {k: round(v, 4) for k, v in return_table.items()})
    feature_cfg = FeatureConfig(cap_scale=model_cfg.cap_scale, demand_scale=model_cfg.demand_scale,
                                max_request_nodes=model_cfg.max_request_nodes)
    train_recs, val_recs = split_records(records, train_cfg.val_fraction, train_cfg.seed)
    train_ds = TrajectoryDataset(substrate, train_recs, model_cfg.context_blocks, feature_cfg, train_cfg.completed_only)
    val_ds = TrajectoryDataset(substrate, val_recs, model_cfg.context_blocks, feature_cfg, train_cfg.completed_only)
    log.info("train transitions %d | val transitions %d", len(train_ds), len(val_ds))
    model = VNEGenModel(model_cfg)
    total, trainable = model.parameter_counts()
    log.info("parameters: %.2fM total, %.2fM trainable", total / 1e6, trainable / 1e6)
    device = torch.device(args.device) if args.device else None
    trainer = Trainer(model, train_cfg, device=device, out_dir=args.out)
    ckpt = trainer.fit(train_ds, return_table, feature_cfg, val_dataset=val_ds)
    log.info("saved checkpoint to %s", ckpt)


if __name__ == "__main__":
    main()
