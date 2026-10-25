"""VNE-Gen: Virtual Network Embedding via Generative Sequence Modeling.

Reference implementation of the generator (dual GNN encoders, return-conditioned
causal backbone with stage-specific LoRA adapters, feasibility-masked pointer
heads), the request-level mapping process with interleaved routing, offline
training, and the Generator-Selector framework for online, offline and
time-aware offline VNE.

The package is organised as follows (section numbers refer to the paper):

* :mod:`vne_gen.problem`   - substrate / request / mapping model, revenue and cost (Sec. III-A),
                             min-hop feasible routing (Eq. 5), feasibility masks, workload generators.
* :mod:`vne_gen.env`       - request-level mapping process, rewards (Eq. 6) and return-to-go (Eq. 7),
                             event-driven online simulator (Sec. IV-A, VI-A).
* :mod:`vne_gen.heuristics`- GRC-Rank, NRM-Rank, NEA-Rank and MCTS baselines / corpus sources.
* :mod:`vne_gen.data`      - offline corpus collection, state features, trajectory dataset (Sec. IV-C).
* :mod:`vne_gen.model`     - VNE-Gen architecture (Sec. IV-B).
* :mod:`vne_gen.train`     - offline training with uncertainty-weighted losses and the R*(n) table.
* :mod:`vne_gen.generator` - Algorithm 1 (request-level mapping with VNE-Gen).
* :mod:`vne_gen.selector`  - proposals, online / offline / time-aware selectors, Algorithm 2 (Sec. V).
* :mod:`vne_gen.eval`      - evaluation protocols, statistics, runtime and optimality studies (Sec. VI).
"""

__version__ = "0.1.0"
