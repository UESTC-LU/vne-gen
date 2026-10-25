# Reproducing the experiments of Sec. VI

All commands run from the repository root. `bash scripts/run_all.sh` chains every step below.

## 0. Corpus and training (Sec. IV-C, VI-A)

| Step | Command | Notes |
|---|---|---|
| Offline corpus | `python scripts/collect_data.py --out data/corpus_geant.pkl.gz --attempts 75000 --seed 12345` | GRC 20 % / NRM 30 % / NEA 50 % of attempts; arrival rates sampled from {0.04,…,0.09}; completed and terminated attempts are both kept. The paper's corpus had 74,909 attempts / 372,671 transitions (53.3 % completed). |
| Training | `python scripts/train.py --config configs/paper.yaml --corpus data/corpus_geant.pkl.gz --out runs/paper` | GPT-2-small backbone initialised from the language-pretrained checkpoint (downloaded through `transformers`; set `model.pretrained` to a local path on an offline machine), K = 3, LoRA rank 64, AdamW 1e-4 / wd 1e-4 / 2,000 warm-up steps / clip 0.5. The checkpoint stores the frozen weights, the trained parameters and the `R*(n)` table. |

## 1. Online GEANT (Sec. VI-B, Fig. 3)

```bash
python scripts/eval_online.py --checkpoint runs/paper/vne_gen.pt --substrate geant \
    --rates 0.04,0.05,0.06,0.07,0.08,0.09 --seeds 0,1,2,3,4,5,6 --arrivals 1000 \
    --baselines nea,mcts --out results/online_geant.json
python scripts/plot_results.py online --input results/online_geant.json --out figures/fig3_online_geant.pdf
```

`results/*.json` contain every (solver, seed, λ) run plus Student-t 95 % CIs across seeds and paired CIs of
the seed-wise differences to the reference method (`summary`).  A3C-GCN, GAL-VNE, HRL-ACRA and FlagVNE are
added through their official implementations (`docs/EXTERNAL_BASELINES.md`).

## 2. Ablations (Sec. VI-C)

`bash scripts/run_ablations.sh data/corpus_geant.pkl.gz runs/paper/vne_gen.pt` evaluates, at λ ∈ {0.06, 0.08}:

* virtual-node ordering (inference-time replacement of the learned pointer on the frozen checkpoint):
  `--ordering ascending_id | degree_desc | fixed_random`;
* stage-specific vs. one parameter-matched shared adapter: `model.shared_adapter=true`;
* graph message passing removed: `model.message_passing=false`;
* GNN–Pointer behaviour-cloning baseline (no causal sequence model): `model.backbone=mlp`;
* no return conditioning: `model.return_conditioning=false`;
* completed-mappings-only corpus: `train.completed_only=true` (same number of updates and batch size).

## 3. Generalization without retraining (Sec. VI-D, Fig. 4)

```bash
python scripts/eval_online.py --checkpoint runs/paper/vne_gen.pt --substrate waxman100 \
    --rates 0.5,0.625,0.75,0.875,1.0,1.125,1.25,1.375,1.5 --baselines nea,mcts --out results/gen_waxman100.json
python scripts/eval_online.py --checkpoint runs/paper/vne_gen.pt --substrate waxman500 \
    --rates 4,6,8,10,12,14 --baselines nea,mcts --out results/gen_waxman500.json
python scripts/eval_online.py --checkpoint runs/paper/vne_gen.pt --substrate geant \
    --rates 0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9 --baselines nea,mcts --out results/gen_geant_shift.json
python scripts/plot_results.py gain --input results/gen_waxman100.json results/gen_waxman500.json results/gen_geant_shift.json \
    --out figures/fig4_generalization.pdf
```

One independently generated Waxman substrate per (size, seed); request-size and demand distributions are
unchanged from training; the GEANT-trained checkpoint is used without retraining or retuning.

## 4. Offline VNE (Sec. VI-E, Table I, Fig. 5)

| Experiment | Command |
|---|---|
| End-to-end: sequential commitment vs. joint selection, N ∈ {50,100,200,400}, ≤ 5 rounds, 300 s ILP limit | `python scripts/eval_offline.py --checkpoint runs/paper/vne_gen.pt --sources vne_gen,nea,grc,nrm --sizes 50,100,200,400 --rounds 5 --out results/offline_end2end.json` |
| Proposal source (Table I, two rounds) | `python scripts/eval_offline.py --checkpoint ... --sources vne_gen,grc,nrm,nea,portfolio --sizes 50,100 --rounds 2 --fixed-rounds --no-sequential --out results/offline_sources_2r.json` |
| Proposal source (Table I, five rounds, N = 50) | `... --sizes 50 --rounds 5 --fixed-rounds --no-sequential --out results/offline_sources_5r.json` |
| Selector-derived (Eq. 10) vs. base capacity profile (Fig. 5b) | `... --sources vne_gen --sizes 50 --rounds 5 --fixed-rounds --no-sequential --profile-mode base --out results/offline_profile_base.json` and `--profile-mode selector --out results/offline_profile_selector.json` |
| Candidate-pool expansion control (one round, two calls per request) | `... --sources vne_gen --sizes 50,100 --rounds 1 --schedule 1.0,1.0 --fixed-rounds --no-sequential --out results/offline_single_round.json` |

Each JSON row of a joint run carries per-round diagnostics: revenue, ILP status, number of proposal calls
and valid proposals (proposal-valid rate), distinct allocation signatures, cross-request candidate-pair
conflict rate, generation and selection time, R2C-LT.

```bash
python scripts/plot_results.py offline --input results/offline_end2end.json --out figures/fig5a_offline.pdf
python scripts/plot_results.py rounds  --input results/offline_profile_base.json results/offline_profile_selector.json \
    --labels "Base profile" "Selector-derived profile" --out figures/fig5b_rounds.pdf
```

## 5. Time-aware offline VNE (Sec. VI-F, Table II)

```bash
python scripts/eval_time_aware.py --checkpoint runs/paper/vne_gen.pt --sources vne_gen,grc,nrm,nea \
    --rates 0.06,0.08,0.10 --episode-size 50 --rounds 5 --out results/time_aware.json
```

## 6. Runtime and memory (Sec. VI-G)

```bash
python scripts/eval_runtime.py --checkpoint runs/paper/vne_gen.pt --methods vne_gen,grc,nrm,nea \
    --substrates geant,waxman100,waxman500 --calls 100 --out results/runtime.json
```

100 CUDA-synchronised successful calls per method / substrate / size bucket on shared request instances and
capacity snapshots; mean, p50, p99 latency, success-conditioned sequential rate, post-load GPU memory and the
incremental per-call peak; checkpoint loading is reported separately.

## 7. Small-instance optimality (Sec. VI-H, Table III)

```bash
python scripts/eval_optimality.py --checkpoint runs/paper/vne_gen.pt --sources vne_gen,nea --out results/optimality.json
```

The full-space reference solves the joint admission + injective placement + unsplittable routing ILP with
HiGHS (`vne_gen/eval/optimality.py`); the exact gap follows Eq. (12).  Instance parameters (8-node substrate,
eight 2–4-node requests, seeds 2014–2020) are command-line options.

## Notes on exact numbers

Absolute revenues depend on the sampled capacities, workloads, the baseline implementations and the MILP
backend/time limit; use matched seeds and the paired confidence intervals in the JSON summaries when
comparing methods.  Set `vne_gen.selector.ilp.set_default_backend("gurobi")` to use Gurobi for the
selection ILPs.
