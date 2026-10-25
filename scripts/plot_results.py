#!/usr/bin/env python3
"""Reproduce the figures of the paper from the JSON results written by the evaluation scripts.

Examples:
    python scripts/plot_results.py online --input results/online_geant.json --out figures/fig3_online_geant.pdf
    python scripts/plot_results.py gain --input results/gen_waxman100.json results/gen_waxman500.json \
        results/gen_geant_shift.json --out figures/fig4_generalization.pdf
    python scripts/plot_results.py offline --input results/offline_end2end.json --out figures/fig5a_offline.pdf
    python scripts/plot_results.py rounds --input results/offline_profile_base.json results/offline_profile_selector.json \
        --labels "Base profile" "Selector-derived profile" --out figures/fig5b_rounds.pdf
"""

from __future__ import annotations

import argparse
import json
import os
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from vne_gen.eval.stats import mean_ci  # noqa: E402


def _load(path):
    with open(path) as f:
        return json.load(f)


def plot_online(paths, out, metric="total_revenue"):
    data = _load(paths[0])
    runs = data["runs"]
    rates = sorted({r["lambda"] for r in runs})
    methods = list(dict.fromkeys(r["solver"] for r in runs))
    fig, ax = plt.subplots(figsize=(5.2, 3.6))
    for m in methods:
        means, lows, highs = [], [], []
        for lam in rates:
            vals = [r[metric] for r in runs if r["solver"] == m and r["lambda"] == lam]
            mu, lo, hi = mean_ci(vals)
            means.append(mu / 1e3)
            lows.append(lo / 1e3)
            highs.append(hi / 1e3)
        ax.plot(rates, means, marker="o", label=m)
        ax.fill_between(rates, lows, highs, alpha=0.15)
    ax.set_xlabel(r"Arrival rate $\lambda$")
    ax.set_ylabel(r"Admitted revenue ($\times 10^3$)")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8, ncol=2)
    fig.tight_layout()
    fig.savefig(out)
    print("wrote", out)


def plot_gain(paths, out, reference="NEA-Rank"):
    fig, axes = plt.subplots(1, len(paths), figsize=(4.0 * len(paths), 3.4), squeeze=False)
    for ax, path in zip(axes[0], paths):
        data = _load(path)
        runs = data["runs"]
        rates = sorted({r["lambda"] for r in runs})
        methods = [m for m in dict.fromkeys(r["solver"] for r in runs)]
        ref_means = {lam: np.mean([r["total_revenue"] for r in runs if r["solver"] == reference and r["lambda"] == lam]) for lam in rates}
        for m in methods:
            gains = []
            for lam in rates:
                mu = np.mean([r["total_revenue"] for r in runs if r["solver"] == m and r["lambda"] == lam])
                gains.append(100.0 * (mu - ref_means[lam]) / ref_means[lam])
            ax.plot(rates, gains, marker="o", label=m, linestyle="--" if m == reference else "-")
        ax.axhline(0, color="k", lw=0.5)
        ax.set_title(data["config"]["substrate"])
        ax.set_xlabel(r"Arrival rate $\lambda$")
        ax.grid(alpha=0.3)
    axes[0][0].set_ylabel(f"Revenue gain over {reference} (%)")
    axes[0][-1].legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(out)
    print("wrote", out)


def plot_offline(paths, out):
    data = _load(paths[0])
    runs = data["runs"]
    sizes = sorted({r["N"] for r in runs})
    fig, ax = plt.subplots(figsize=(5.2, 3.6))
    for variant, style in (("sequential", "--"), ("joint", "-")):
        for src in dict.fromkeys(r["source"] for r in runs):
            vals = [[r["revenue"] for r in runs if r["variant"] == variant and r["source"] == src and r["N"] == N] for N in sizes]
            if not all(vals):
                continue
            stats = [mean_ci(v) for v in vals]
            ax.plot(sizes, [s[0] / 1e3 for s in stats], style, marker="o", label=f"{src} ({variant})")
            ax.fill_between(sizes, [s[1] / 1e3 for s in stats], [s[2] / 1e3 for s in stats], alpha=0.15)
    ax.set_xlabel("Request-set size N")
    ax.set_ylabel(r"Admitted revenue ($\times 10^3$)")
    ax.set_xticks(sizes)
    ax.grid(alpha=0.3)
    ax.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(out)
    print("wrote", out)


def plot_rounds(paths, labels, out):
    fig, ax = plt.subplots(figsize=(4.6, 3.4))
    for path, label in zip(paths, labels):
        data = _load(path)
        runs = [r for r in data["runs"] if r["variant"] == "joint"]
        max_rounds = max(len(r["round_revenues"]) for r in runs)
        per_round = []
        for i in range(max_rounds):
            # a run that stopped early keeps its best revenue for the remaining rounds
            per_round.append(np.mean([max(r["round_revenues"][: i + 1]) for r in runs]))
        ax.plot(range(1, max_rounds + 1), np.asarray(per_round) / 1e3, marker="o", label=label)
    ax.set_xlabel("Generation round")
    ax.set_ylabel(r"Admitted revenue ($\times 10^3$)")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(out)
    print("wrote", out)


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("kind", choices=["online", "gain", "offline", "rounds"])
    p.add_argument("--input", nargs="+", required=True)
    p.add_argument("--labels", nargs="*", default=None)
    p.add_argument("--reference", default="NEA-Rank")
    p.add_argument("--out", required=True)
    args = p.parse_args()
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    if args.kind == "online":
        plot_online(args.input, args.out)
    elif args.kind == "gain":
        plot_gain(args.input, args.out, args.reference)
    elif args.kind == "offline":
        plot_offline(args.input, args.out)
    else:
        labels = args.labels or [os.path.basename(x) for x in args.input]
        plot_rounds(args.input, labels, args.out)


if __name__ == "__main__":
    main()
