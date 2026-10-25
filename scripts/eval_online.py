#!/usr/bin/env python3
"""Online VNE evaluation (Sec. VI-B) and generalization without retraining (Sec. VI-D).

Examples:
    # GEANT, paper loads, seven matched seeds
    python scripts/eval_online.py --checkpoint runs/paper/vne_gen.pt --substrate geant \
        --rates 0.04,0.05,0.06,0.07,0.08,0.09 --baselines nea,mcts --out results/online_geant.json
    # unseen Waxman-100 / Waxman-500 substrates and shifted GEANT loads
    python scripts/eval_online.py --checkpoint runs/paper/vne_gen.pt --substrate waxman100 \
        --rates 0.5,0.625,0.75,0.875,1.0,1.125,1.25,1.375,1.5 --baselines nea,mcts --out results/gen_waxman100.json
    python scripts/eval_online.py --checkpoint runs/paper/vne_gen.pt --substrate waxman500 \
        --rates 4,6,8,10,12,14 --baselines nea,mcts --out results/gen_waxman500.json
    python scripts/eval_online.py --checkpoint runs/paper/vne_gen.pt --substrate geant \
        --rates 0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9 --baselines nea,mcts --out results/gen_geant_shift.json
    # virtual-node ordering ablation on one frozen checkpoint
    python scripts/eval_online.py --checkpoint ... --ordering degree_desc --rates 0.06,0.08 --out results/order_degree.json
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from vne_gen.eval.online_eval import OnlineEvalConfig, evaluate_online  # noqa: E402
from vne_gen.utils import build_baselines, load_generator_if_given, setup_logging  # noqa: E402


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--checkpoint", default=None, help="VNE-Gen checkpoint (omit to evaluate baselines only)")
    p.add_argument("--substrate", default="geant", help="geant | waxman100 | waxman500")
    p.add_argument("--rates", default="0.04,0.05,0.06,0.07,0.08,0.09")
    p.add_argument("--seeds", default="0,1,2,3,4,5,6")
    p.add_argument("--arrivals", type=int, default=1000)
    p.add_argument("--baselines", default="nea", help="comma-separated: grc,nrm,nea,mcts")
    p.add_argument("--mcts-simulations", type=int, default=64)
    p.add_argument("--ordering", default="learned", help="learned | ascending_id | degree_desc | fixed_random")
    p.add_argument("--device", default=None)
    p.add_argument("--out", default="results/online.json")
    args = p.parse_args()
    setup_logging()
    rates = [float(x) for x in args.rates.split(",") if x]
    seeds = [int(x) for x in args.seeds.split(",") if x]
    solvers = build_baselines([b for b in args.baselines.split(",") if b], mcts_simulations=args.mcts_simulations)
    gen = load_generator_if_given(args.checkpoint, device=args.device, ordering=args.ordering)
    if gen is not None:
        name = "VNE-Gen" if args.ordering == "learned" else f"VNE-Gen[{args.ordering}]"
        solvers[name] = gen
    cfg = OnlineEvalConfig(substrate=args.substrate, arrival_rates=rates, num_arrivals=args.arrivals, seeds=seeds)
    result = evaluate_online(solvers, cfg, out_path=args.out)
    log = logging.getLogger("eval_online")
    for lam, summ in result["summary"].items():
        log.info("lambda=%s: %s", lam, json.dumps({k: round(v["mean"], 1) for k, v in summ.items()}))
    log.info("results written to %s", args.out)


if __name__ == "__main__":
    main()
