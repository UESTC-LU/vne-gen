#!/usr/bin/env python3
"""Request-level runtime and memory measurements (Sec. VI-G).

Example:
    python scripts/eval_runtime.py --checkpoint runs/paper/vne_gen.pt --methods vne_gen,grc,nrm,nea \
        --substrates geant,waxman100,waxman500 --calls 100 --out results/runtime.json
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from vne_gen.eval.runtime import RuntimeConfig, measure_runtime  # noqa: E402
from vne_gen.utils import build_baselines, load_generator_if_given, setup_logging  # noqa: E402


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--checkpoint", default=None)
    p.add_argument("--methods", default="vne_gen,grc,nrm,nea")
    p.add_argument("--substrates", default="geant,waxman100,waxman500")
    p.add_argument("--calls", type=int, default=100)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", default=None)
    p.add_argument("--out", default="results/runtime.json")
    args = p.parse_args()
    setup_logging()
    log = logging.getLogger("eval_runtime")
    names = [s.strip().lower() for s in args.methods.split(",") if s.strip()]
    methods = build_baselines([n for n in names if n != "vne_gen"])
    load_time = None
    if "vne_gen" in names:
        t0 = time.time()
        gen = load_generator_if_given(args.checkpoint, device=args.device)
        load_time = time.time() - t0
        if gen is None:
            raise SystemExit("--checkpoint is required when vne_gen is among the methods")
        methods = {"VNE-Gen": gen, **methods}
        log.info("checkpoint loading took %.1fs (excluded from per-call latency)", load_time)
    cfg = RuntimeConfig(substrates=[s for s in args.substrates.split(",")], calls_per_config=args.calls, seed=args.seed)
    result = measure_runtime(methods, cfg, out_path=args.out)
    if load_time is not None:
        result["checkpoint_load_time_s"] = load_time
    for row in result["rows"]:
        log.info("%-10s %-10s %-6s mean %.4fs  p99 %.4fs  rate %.2f req/s  (%d ok / %d calls)", row["method"],
                 row["substrate"], row["bucket"], row["mean_latency_s"], row["p99_latency_s"],
                 row["sequential_rate_rps"], row["successful_calls"], row["attempted_calls"])
    log.info("results written to %s", args.out)


if __name__ == "__main__":
    main()
