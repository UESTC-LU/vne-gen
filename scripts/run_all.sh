#!/usr/bin/env bash
# Full reproduction pipeline (Sec. VI). Expect several GPU-hours for training and several CPU-hours for the
# selector studies; every step writes JSON results under results/ and figures under figures/.
set -euo pipefail
CORPUS=data/corpus_geant.pkl.gz
CKPT=runs/paper/vne_gen.pt

# 1. offline corpus (Sec. IV-C)
python scripts/collect_data.py --out "$CORPUS" --attempts 75000 --seed 12345

# 2. training (Sec. IV-C, VI-A)
python scripts/train.py --config configs/paper.yaml --corpus "$CORPUS" --out runs/paper

# 3. online GEANT (Sec. VI-B)
python scripts/eval_online.py --checkpoint "$CKPT" --substrate geant --rates 0.04,0.05,0.06,0.07,0.08,0.09 \
    --baselines nea,mcts --out results/online_geant.json

# 4. generalization without retraining (Sec. VI-D)
python scripts/eval_online.py --checkpoint "$CKPT" --substrate waxman100 \
    --rates 0.5,0.625,0.75,0.875,1.0,1.125,1.25,1.375,1.5 --baselines nea,mcts --out results/gen_waxman100.json
python scripts/eval_online.py --checkpoint "$CKPT" --substrate waxman500 --rates 4,6,8,10,12,14 \
    --baselines nea,mcts --out results/gen_waxman500.json
python scripts/eval_online.py --checkpoint "$CKPT" --substrate geant --rates 0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9 \
    --baselines nea,mcts --out results/gen_geant_shift.json

# 5. offline VNE (Sec. VI-E)
python scripts/eval_offline.py --checkpoint "$CKPT" --sources vne_gen,nea,grc,nrm --sizes 50,100,200,400 --rounds 5 \
    --out results/offline_end2end.json
python scripts/eval_offline.py --checkpoint "$CKPT" --sources vne_gen,grc,nrm,nea,portfolio --sizes 50,100 --rounds 2 \
    --fixed-rounds --no-sequential --out results/offline_sources_2r.json
python scripts/eval_offline.py --checkpoint "$CKPT" --sources vne_gen,grc,nrm,nea,portfolio --sizes 50 --rounds 5 \
    --fixed-rounds --no-sequential --out results/offline_sources_5r.json
python scripts/eval_offline.py --checkpoint "$CKPT" --sources vne_gen --sizes 50 --rounds 5 --fixed-rounds \
    --profile-mode base --no-sequential --out results/offline_profile_base.json
python scripts/eval_offline.py --checkpoint "$CKPT" --sources vne_gen --sizes 50 --rounds 5 --fixed-rounds \
    --profile-mode selector --no-sequential --out results/offline_profile_selector.json
python scripts/eval_offline.py --checkpoint "$CKPT" --sources vne_gen --sizes 50,100 --rounds 1 --schedule 1.0,1.0 \
    --fixed-rounds --no-sequential --out results/offline_single_round.json

# 6. time-aware offline VNE (Sec. VI-F)
python scripts/eval_time_aware.py --checkpoint "$CKPT" --sources vne_gen,grc,nrm,nea --rates 0.06,0.08,0.10 \
    --rounds 5 --out results/time_aware.json

# 7. runtime (Sec. VI-G) and small-instance optimality (Sec. VI-H)
python scripts/eval_runtime.py --checkpoint "$CKPT" --methods vne_gen,grc,nrm,nea --out results/runtime.json
python scripts/eval_optimality.py --checkpoint "$CKPT" --sources vne_gen,nea --out results/optimality.json

# 8. ablations (Sec. VI-C)
bash scripts/run_ablations.sh "$CORPUS" "$CKPT"

# 9. figures
python scripts/plot_results.py online  --input results/online_geant.json --out figures/fig3_online_geant.pdf
python scripts/plot_results.py gain    --input results/gen_waxman100.json results/gen_waxman500.json results/gen_geant_shift.json \
    --out figures/fig4_generalization.pdf
python scripts/plot_results.py offline --input results/offline_end2end.json --out figures/fig5a_offline.pdf
python scripts/plot_results.py rounds  --input results/offline_profile_base.json results/offline_profile_selector.json \
    --labels "Base profile" "Selector-derived profile" --out figures/fig5b_rounds.pdf
