#!/usr/bin/env python3
"""Offline VNE evaluation (Sec. VI-E).

Examples:
    # end-to-end: sequential commitment vs. joint selection, N in {50,100,200,400}, up to five rounds
    python scripts/eval_offline.py --checkpoint runs/paper/vne_gen.pt --sources vne_gen,nea,grc,nrm \
        --sizes 50,100,200,400 --rounds 5 --out results/offline_end2end.json
    # proposal-source comparison (Table I): two fixed rounds at N in {50,100}, five at N = 50
    python scripts/eval_offline.py --checkpoint ... --sources vne_gen,grc,nrm,nea,portfolio --sizes 50,100 \
        --rounds 2 --fixed-rounds --no-sequential --out results/offline_sources_2r.json
    # capacity-profile comparison: selector-derived (Eq. 10) vs. base profile
    python scripts/eval_offline.py --checkpoint ... --sources vne_gen --sizes 50 --rounds 5 --fixed-rounds \
        --profile-mode base --no-sequential --out results/offline_profile_base.json
    # candidate-pool expansion control: one round with two calls per request
    python scripts/eval_offline.py --checkpoint ... --sources vne_gen --sizes 50,100 --rounds 1 --schedule 1.0,1.0 \
        --fixed-rounds --no-sequential --out results/offline_single_round.json
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from vne_gen.eval.offline_eval import OfflineEvalConfig, evaluate_offline  # noqa: E402
from vne_gen.utils import build_baselines, load_generator_if_given, setup_logging  # noqa: E402


def parse_schedule(text: str):
    """``"1.0"`` -> one call per round; ``"1.0,1.0"`` -> two calls per round; ``"1.0;0.9,1.0"`` -> per-round lists."""
    rounds = []
    for part in text.split(";"):
        vals = [float(x) for x in part.split(",") if x.strip()]
        rounds.append(vals if len(vals) > 1 else vals[0])
    return tuple(rounds)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--checkpoint", default=None)
    p.add_argument("--sources", default="vne_gen,nea", help="comma-separated: vne_gen,grc,nrm,nea,portfolio")
    p.add_argument("--substrate", default="geant")
    p.add_argument("--sizes", default="50,100,200,400")
    p.add_argument("--seeds", default="0,1,2,3,4,5,6")
    p.add_argument("--rounds", type=int, default=5)
    p.add_argument("--schedule", default="1.0", help="return scales per round, see parse_schedule")
    p.add_argument("--fixed-rounds", action="store_true", help="complete every prescribed round (no early stopping)")
    p.add_argument("--profile-mode", default="selector", help="selector (Eq. 10) | base")
    p.add_argument("--time-limit", type=float, default=300.0)
    p.add_argument("--no-sequential", action="store_true")
    p.add_argument("--no-joint", action="store_true")
    p.add_argument("--temperature", type=float, default=1.0)
    p.add_argument("--device", default=None)
    p.add_argument("--out", default="results/offline.json")
    args = p.parse_args()
    setup_logging()
    names = [s.strip().lower() for s in args.sources.split(",") if s.strip()]
    sources = build_baselines([n for n in names if n != "vne_gen"])
    if "vne_gen" in names:
        gen = load_generator_if_given(args.checkpoint, device=args.device, temperature=args.temperature)
        if gen is None:
            raise SystemExit("--checkpoint is required when vne_gen is among the sources")
        sources = {"VNE-Gen": gen, **sources}
    cfg = OfflineEvalConfig(substrate=args.substrate, request_set_sizes=[int(x) for x in args.sizes.split(",")],
                            seeds=[int(x) for x in args.seeds.split(",")], rounds=args.rounds,
                            schedule=parse_schedule(args.schedule), fixed_rounds=args.fixed_rounds,
                            time_limit=args.time_limit, profile_mode=args.profile_mode,
                            run_sequential=not args.no_sequential, run_joint=not args.no_joint)
    result = evaluate_offline(sources, cfg, out_path=args.out)
    log = logging.getLogger("eval_offline")
    for key, summ in result["summary"].items():
        if "joint_vs_sequential" in key:
            log.info("%s: +%.1f (paired 95%% CI [%.1f, %.1f])", key, summ["paired_diff_mean"], summ["paired_ci_low"], summ["paired_ci_high"])
        else:
            log.info("%s: %s", key, json.dumps({k: round(v["mean"], 1) for k, v in summ.items()}))
    log.info("results written to %s", args.out)


if __name__ == "__main__":
    main()
