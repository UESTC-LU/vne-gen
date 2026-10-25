#!/usr/bin/env python3
"""Time-aware offline VNE evaluation (Sec. VI-F): 50-request episodes at lambda in {0.06, 0.08, 0.10}.

Example:
    python scripts/eval_time_aware.py --checkpoint runs/paper/vne_gen.pt --sources vne_gen,grc,nrm,nea \
        --rates 0.06,0.08,0.10 --rounds 5 --out results/time_aware.json
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from vne_gen.eval.time_aware_eval import TimeAwareEvalConfig, evaluate_time_aware  # noqa: E402
from vne_gen.utils import build_baselines, load_generator_if_given, setup_logging  # noqa: E402


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--checkpoint", default=None)
    p.add_argument("--sources", default="vne_gen,nea")
    p.add_argument("--substrate", default="geant")
    p.add_argument("--rates", default="0.06,0.08,0.10")
    p.add_argument("--seeds", default="0,1,2,3,4,5,6")
    p.add_argument("--episode-size", type=int, default=50)
    p.add_argument("--rounds", type=int, default=5)
    p.add_argument("--fixed-rounds", action="store_true")
    p.add_argument("--time-limit", type=float, default=300.0)
    p.add_argument("--no-sequential", action="store_true")
    p.add_argument("--temperature", type=float, default=1.0)
    p.add_argument("--device", default=None)
    p.add_argument("--out", default="results/time_aware.json")
    args = p.parse_args()
    setup_logging()
    names = [s.strip().lower() for s in args.sources.split(",") if s.strip()]
    sources = build_baselines([n for n in names if n != "vne_gen"])
    if "vne_gen" in names:
        gen = load_generator_if_given(args.checkpoint, device=args.device, temperature=args.temperature)
        if gen is None:
            raise SystemExit("--checkpoint is required when vne_gen is among the sources")
        sources = {"VNE-Gen": gen, **sources}
    cfg = TimeAwareEvalConfig(substrate=args.substrate, episode_size=args.episode_size,
                              arrival_rates=[float(x) for x in args.rates.split(",")],
                              seeds=[int(x) for x in args.seeds.split(",")], rounds=args.rounds,
                              fixed_rounds=args.fixed_rounds, time_limit=args.time_limit,
                              run_sequential=not args.no_sequential)
    result = evaluate_time_aware(sources, cfg, out_path=args.out)
    log = logging.getLogger("eval_time_aware")
    for key, summ in result["summary"].items():
        if "joint_vs_sequential" in key:
            log.info("%s: +%.1f (paired 95%% CI [%.1f, %.1f])", key, summ["paired_diff_mean"], summ["paired_ci_low"], summ["paired_ci_high"])
        else:
            log.info("%s: %s", key, json.dumps({k: round(v["mean"], 1) for k, v in summ.items()}))
    log.info("results written to %s", args.out)


if __name__ == "__main__":
    main()
