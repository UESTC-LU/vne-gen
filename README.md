# VNE-Gen: Virtual Network Embedding via Generative Sequence Modeling

Reference implementation of **VNE-Gen**, a generative architecture that constructs complete request-level
mappings for variable-size virtual network requests and substrate networks, and of the
**Generator–Selector framework** that reuses one frozen generator for online, offline and time-aware
offline Virtual Network Embedding (VNE).

> Paper: *Virtual Network Embedding via Generative Sequence Modeling*
> Code: `https://github.com/UESTC-LU/vne-gen` (replace the placeholder after uploading)

<p align="center">
<b>Dual GNN encoders → structured [return, state, action] tokens → shared causal Transformer (GPT-2 small, frozen) with stage-specific LoRA adapters → two feasibility-masked pointer heads → interleaved min-hop routing and residual-resource updates</b>
</p>

---

## What is in the repository

| Paper section | Module | Contents |
|---|---|---|
| III-A Networks, resources, mappings | `vne_gen/problem/` | substrate / request / mapping model, revenue (Eq. 1), routed-resource cost (Eq. 2), R2C and R2C-LT, min-hop feasible routing with lexicographic tie-break (Eq. 5), feasibility masks, the shared mapping validator, GEANT / Waxman substrates, Poisson–exponential workloads |
| III-B Operating settings, VI-A Simulator | `vne_gen/env/` | request-level mapping process (state machine of Algorithm 1), rewards (Eq. 6), return-to-go (Eq. 7), event-driven online simulator |
| IV-B Architecture | `vne_gen/model/` | `GraphEncoder` (three-layer message passing), `CausalBackbone` (GPT-2 small + `LoRAConv1D` stage adapters), `VNEGenModel` (structured token blocks, K-block causal context, two-pass decoding, masked pointer heads) |
| IV-C Offline training | `vne_gen/data/`, `vne_gen/train/` | corpus collection from GRC/NRM/NEA-Rank, state features, trajectory dataset with K-block windows, AdamW trainer with uncertainty-weighted losses, size-conditioned return table `R*(n)` |
| Algorithm 1 | `vne_gen/generator/` | `VNEGenGenerator`: greedy (online) and stochastic (offline) decoding, fixed-ordering ablation |
| V Generator–Selector, Algorithm 2 | `vne_gen/selector/` | proposals with allocation-signature deduplication, exact selection ILPs (Eq. 3 / Eq. 4, HiGHS or Gurobi), selector-derived capacity profiles (Eq. 10 / Eq. 11), multi-round candidate-pool expansion, early stopping / fixed rounds |
| II, VI Baselines | `vne_gen/heuristics/`, `vne_gen/baselines/` | GRC-Rank, NRM-Rank, NEA-Rank, a compact MCTS, adapter for external learned baselines |
| VI Evaluation | `vne_gen/eval/`, `scripts/` | online, generalization, offline, time-aware, runtime/memory, small-instance full-space optimum (Eq. 12), Student-t and paired confidence intervals, figure scripts |

## Installation

```bash
git clone https://github.com/UESTC-LU/vne-gen.git
cd vne-gen
python -m venv .venv && source .venv/bin/activate      # optional
pip install -r requirements.txt
pip install -e .                                        # optional, makes `vne_gen` importable everywhere
pytest                                                  # unit tests (model tests are skipped without torch)
```

Requirements: Python ≥ 3.9, PyTorch ≥ 2.0, `transformers` ≥ 4.30 (for the GPT-2 backbone), NumPy, SciPy ≥ 1.9
(HiGHS MILP solver), NetworkX, PyYAML, matplotlib. `gurobipy` is optional
(`vne_gen.selector.ilp.set_default_backend("gurobi")`).

The 40-node GEANT topology (`vne_gen/data/topologies/Geant2012.gml`) comes from the Internet Topology Zoo
(Knight et al., 2011) and is redistributed under its CC BY 3.0 licence.

## Quick start

```bash
# 1) offline corpus from the three ranking heuristics on GEANT workloads (Sec. IV-C)
python scripts/collect_data.py --out data/corpus_geant.pkl.gz --attempts 75000 --seed 12345

# 2) train the generator (GPT-2-small backbone, K = 3, stage-specific LoRA; Sec. VI-A)
python scripts/train.py --config configs/paper.yaml --corpus data/corpus_geant.pkl.gz --out runs/paper

# 3) online GEANT evaluation over seven matched seeds (Sec. VI-B)
python scripts/eval_online.py --checkpoint runs/paper/vne_gen.pt --substrate geant \
    --rates 0.04,0.05,0.06,0.07,0.08,0.09 --baselines nea,mcts --out results/online_geant.json

# 4) the same frozen checkpoint in the offline and time-aware offline settings (Sec. VI-E, VI-F)
python scripts/eval_offline.py    --checkpoint runs/paper/vne_gen.pt --sources vne_gen,nea,grc,nrm --sizes 50,100,200,400
python scripts/eval_time_aware.py --checkpoint runs/paper/vne_gen.pt --sources vne_gen,grc,nrm,nea --rates 0.06,0.08,0.10
```

`bash scripts/smoke_test.sh` runs the entire pipeline with a tiny randomly initialised model in a few
minutes on a CPU (it checks that everything executes; the numbers it produces are meaningless).
`bash scripts/run_all.sh` reproduces every experiment of Sec. VI and the figures; see
[`docs/REPRODUCING.md`](docs/REPRODUCING.md) for the mapping between scripts, paper tables and figures.

## Using the generator from Python

```python
import numpy as np
from vne_gen.generator import load_generator
from vne_gen.problem import load_geant, generate_workload
from vne_gen.env import run_online
from vne_gen.selector import CandidatePoolSelector, SelectorConfig

gen = load_generator("runs/paper/vne_gen.pt")                       # frozen generator + R*(n) table
substrate = load_geant(np.random.default_rng(0))
requests = generate_workload(np.random.default_rng(1), 1000, lam=0.06)

# online VNE: one greedy call per arrival, immediate commitment
print(run_online(substrate, requests, gen).summary())

# offline VNE: Algorithm 2 with selector-derived capacity profiles (Eq. 10) and the exact ILP (Eq. 3)
offline = CandidatePoolSelector(substrate, requests[:100], gen, SelectorConfig(setting="off", rounds=5)).run()
print(offline.summary())

# time-aware offline VNE: activity-indexed capacities (Eq. 4) and profiles (Eq. 11)
ta = CandidatePoolSelector(substrate, requests[:50], gen, SelectorConfig(setting="ta", rounds=5)).run()
print(ta.summary())
```

Any solver implementing `embed(request, substrate, profile) -> Mapping | None` can take part in every
protocol (see `vne_gen/baselines/external.py` and `docs/EXTERNAL_BASELINES.md`).

## Design notes (how the code follows the paper)

* **Mapping process.** `MappingAttempt.step(v, u)` places a virtual node, routes the newly activated links
  `E_k^+` in non-increasing-demand order (stable w.r.t. the neighbour-iteration order) with the min-hop /
  lexicographic rule of Eq. (5), reserves bandwidth immediately, and rolls back on failure. Rewards and
  returns follow Eq. (6)–(7): completed mappings have initial return `1 + R2C ∈ (1, 2]`, terminated
  attempts a return below one. Terminated attempts are retained in the corpus.
* **Variable-size decoder.** Node embeddings of the current graphs are the pointer keys, so the output
  spaces adapt to `|V_r|` and `|V_S|` without substrate-size-dependent layers. `F_V` and `F_S` masks are
  applied inside the pointer heads; because the residual state does not change between the two stages,
  `v ∈ F_V` guarantees a non-empty `F_S`.
* **Causal context.** Each decision block is `[R_k, S_k, A_k]` (return-to-go, state token from pooled GNN
  embeddings and partial-mapping statistics, selected pair); the window holds the current block and the
  `K − 1` preceding completed blocks. Stage V reads `h^V_k` at the state token, stage S appends a
  half-action token for `v_k` and reads `h^S_k` there. Both passes share the backbone and use separate LoRA
  factors `W_z = W + B_z A_z` (original parameters frozen).
* **Offline training.** NLL of both stages over emitted action pairs only, combined with learned
  uncertainty weights; `R*(n)` is the maximum initial return per request size in the corpus and is stored in
  the checkpoint. Online decoding is greedy with `R*(|V_r|)`; offline decoding is stochastic with `ρ R*`.
* **Selectors.** Proposals are deduplicated by allocation signature. Algorithm 2 re-solves the ILP over the
  accumulated pools after every round, keeps `B_best`, and derives next-round profiles from the selected
  allocations of the *other* requests (componentwise minimum over the request's activity interval in the
  time-aware setting). Time slots are the elementary intervals between consecutive arrival/departure times.

## Baselines

GRC-Rank, NRM-Rank and NEA-Rank are re-implemented in `vne_gen/heuristics/ranking.py` following the
cited papers, generalised to the three-dimensional node-resource model and executed through the common
mapping process (so that their trajectories form the offline corpus). MCTS is a compact re-implementation.
The learning-based baselines of the paper (A3C-GCN, GAL-VNE, HRL-ACRA, FlagVNE) and the official heuristic
code bases are *not* vendored here; `docs/EXTERNAL_BASELINES.md` explains how to run them through the same
simulator and validator.

## Configuration

`configs/paper.yaml` is the evaluated configuration (12-layer GPT-2-small backbone with hidden size 768,
`K = 3`, two three-layer GNN encoders with 128-dimensional outputs, LoRA rank 64, AdamW with learning rate
1e-4, weight decay 1e-4, 2,000 warm-up steps, gradient clipping 0.5). Every field can be overridden on the
command line, e.g. `--set model.shared_adapter=true train.epochs=5`. The ablations of Sec. VI-C are
`scripts/run_ablations.sh`.

## Repository layout

```
vne-gen/
├── configs/            paper.yaml (evaluated configuration), tiny.yaml (smoke test)
├── docs/               REPRODUCING.md, EXTERNAL_BASELINES.md, paper_code_link.tex
├── scripts/            collect_data.py, train.py, eval_*.py, plot_results.py, run_all.sh, run_ablations.sh, smoke_test.sh
├── tests/              pytest suite
└── vne_gen/
    ├── problem/        substrate, request, mapping, routing, feasibility, generators, data/topologies/Geant2012.gml
    ├── env/            mapping_env (Algorithm 1 state machine), simulator
    ├── heuristics/     ranking (GRC/NRM/NEA), mcts
    ├── data/           features, dataset, collect
    ├── model/          config, gnn, lora, backbone, vne_gen
    ├── train/          config, trainer, return_table
    ├── generator/      generator (Algorithm 1)
    ├── selector/       proposal, ilp, timeslots, online, offline (Algorithm 2)
    ├── baselines/      external-solver adapter
    └── eval/           online_eval, offline_eval, time_aware_eval, runtime, optimality, stats
```

## Citation

If you use this code, please cite the paper (see `CITATION.cff`):

```bibtex
@article{vnegen2026,
  title   = {Virtual Network Embedding via Generative Sequence Modeling},
  author  = {<authors>},
  journal = {<journal / venue>},
  year    = {2026}
}
```

## License

MIT (see `LICENSE`). The GEANT topology file is CC BY 3.0 (Internet Topology Zoo).
