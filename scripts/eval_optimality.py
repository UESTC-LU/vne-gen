#!/usr/bin/env python3
"""Small-instance optimality analysis (Sec. VI-H): seven 8-node instances with eight 2-4-node requests.

Example:
    python scripts/eval_optimality.py --checkpoint runs/paper/vne_gen.pt --sources vne_gen,nea --out results/optimality.json
"""

from __future__ import annotations

import argparse
import logging
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from vne_gen.eval.optimality import OptimalityConfig, evaluate_optimality  # noqa: E402
from vne_gen.utils import build_baselines, load_generator_if_given, setup_logging  # noqa: E402


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--checkpoint", default=None)
    p.add_argument("--sources", default="vne_gen,nea")
    p.add_argument("--seeds", default="2014,2015,2016,2017,2018,2019,2020")
    p.add_argument("--substrate-nodes", type=int, default=8)
    p.add_argument("--edge-prob", type=float, default=0.45)
    p.add_argument("--requests", type=int, default=8)
    p.add_argument("--rounds", type=int, default=5)
    p.add_argument("--time-limit", type=float, default=600.0)
    p.add_argument("--device", default=None)
    p.add_argument("--out", default="results/optimality.json")
    args = p.parse_args()
    setup_logging()
    names = [s.strip().lower() for s in args.sources.split(",") if s.strip()]
    sources = build_baselines([n for n in names if n != "vne_gen"])
    if "vne_gen" in names:
        gen = load_generator_if_given(args.checkpoint, device=args.device)
        if gen is None:
            raise SystemExit("--checkpoint is required when vne_gen is among the sources")
        sources = {"VNE-Gen": gen, **sources}
    cfg = OptimalityConfig(seeds=[int(x) for x in args.seeds.split(",")], substrate_nodes=args.substrate_nodes,
                           substrate_edge_prob=args.edge_prob, num_requests=args.requests, rounds=args.rounds,
                           time_limit=args.time_limit)
    result = evaluate_optimality(sources, cfg, out_path=args.out)
    log = logging.getLogger("eval_optimality")
    for name, s in result["summary"].items():
        log.info("%s: mean gap %.3f%% | max gap %.3f%% | optimal on %d/%d | aggregate gap %.3f%%", name,
                 s["mean_gap_pct"], s["max_gap_pct"], s["num_optimal"], len(cfg.seeds), s["aggregate_gap_pct"])
    log.info("results written to %s", args.out)


if __name__ == "__main__":
    main()
