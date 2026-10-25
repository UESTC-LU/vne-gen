#!/usr/bin/env python3
"""Collect the offline mapping corpus from the ranking heuristics on GEANT workloads (Sec. IV-C).

Example (paper scale, ~75k attempts / ~370k transitions):
    python scripts/collect_data.py --out data/corpus_geant.pkl.gz --attempts 75000 --seed 12345
"""

from __future__ import annotations

import argparse
import logging
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from vne_gen.data.collect import collect_corpus  # noqa: E402
from vne_gen.data.dataset import save_corpus  # noqa: E402
from vne_gen.problem.generators import load_geant  # noqa: E402
from vne_gen.utils import setup_logging  # noqa: E402


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--out", default="data/corpus_geant.pkl.gz")
    p.add_argument("--attempts", type=int, default=75000, help="total number of mapping attempts to record")
    p.add_argument("--mix", default="grc:0.2,nrm:0.3,nea:0.5", help="heuristic mixture by attempt share")
    p.add_argument("--rates", default="0.04,0.05,0.06,0.07,0.08,0.09", help="arrival rates sampled per episode")
    p.add_argument("--arrivals-per-episode", type=int, default=1000)
    p.add_argument("--seed", type=int, default=12345, help="corpus seed (evaluation seeds are disjoint)")
    p.add_argument("--substrate-seed", type=int, default=12345, help="seed of the GEANT capacity sample")
    args = p.parse_args()
    setup_logging()
    log = logging.getLogger("collect")
    mix = {k: float(v) for k, v in (item.split(":") for item in args.mix.split(","))}
    rates = [float(x) for x in args.rates.split(",")]
    substrate = load_geant(np.random.default_rng(args.substrate_seed))
    log.info("substrate %s: %d nodes, %d links", substrate.name, substrate.num_nodes, substrate.num_links)
    records, meta = collect_corpus(substrate, total_attempts=args.attempts, mix=mix, arrival_rates=rates,
                                   arrivals_per_episode=args.arrivals_per_episode, seed=args.seed)
    log.info("collected %d attempts, %d transitions, %.1f%% completed; sources %s",
             meta.num_attempts, meta.num_transitions, 100 * meta.completed_fraction, meta.sources)
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    save_corpus(args.out, substrate, records, meta)
    log.info("saved corpus to %s", args.out)


if __name__ == "__main__":
    main()
