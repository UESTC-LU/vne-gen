#!/usr/bin/env bash
# Architecture and training ablations of Sec. VI-C on GEANT at lambda in {0.06, 0.08}.
# Usage: bash scripts/run_ablations.sh data/corpus_geant.pkl.gz runs/paper/vne_gen.pt
set -euo pipefail
CORPUS=${1:-data/corpus_geant.pkl.gz}
FULL_CKPT=${2:-runs/paper/vne_gen.pt}
RATES="0.06,0.08"
CFG=configs/paper.yaml
mkdir -p results/ablations

# 1) Virtual-node ordering: inference-time change on the frozen full checkpoint (no retraining)
for ORD in learned ascending_id degree_desc fixed_random; do
  python scripts/eval_online.py --checkpoint "$FULL_CKPT" --ordering "$ORD" --rates "$RATES" --baselines "" \
      --out "results/ablations/ordering_${ORD}.json"
done

# 2) Variants trained separately with the same corpus and training budget
declare -A VARIANTS=(
  [shared_adapter]="model.shared_adapter=true"
  [no_message_passing]="model.message_passing=false"
  [gnn_pointer_bc]="model.backbone=mlp"
  [no_return_conditioning]="model.return_conditioning=false"
  [completed_only]="train.completed_only=true"
)
for NAME in "${!VARIANTS[@]}"; do
  python scripts/train.py --config "$CFG" --corpus "$CORPUS" --out "runs/ablation_${NAME}" --set ${VARIANTS[$NAME]}
  python scripts/eval_online.py --checkpoint "runs/ablation_${NAME}/vne_gen.pt" --rates "$RATES" --baselines "" \
      --out "results/ablations/${NAME}.json"
done
echo "ablation results in results/ablations/"
