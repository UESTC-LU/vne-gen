#!/usr/bin/env bash
# End-to-end smoke test on a CPU in a few minutes: tiny corpus -> tiny model -> every evaluation protocol.
# The numbers it produces are meaningless; it only checks that the pipeline runs.
set -euo pipefail
OUT=${1:-runs/smoke}
mkdir -p "$OUT"
python scripts/collect_data.py --out "$OUT/corpus.pkl.gz" --attempts 1500 --arrivals-per-episode 300 --seed 777
python scripts/train.py --config configs/tiny.yaml --corpus "$OUT/corpus.pkl.gz" --out "$OUT/model" --set train.max_steps=100
python scripts/eval_online.py --checkpoint "$OUT/model/vne_gen.pt" --rates 0.06 --seeds 0 --arrivals 100 --baselines nea \
    --out "$OUT/online.json"
python scripts/eval_online.py --checkpoint "$OUT/model/vne_gen.pt" --substrate waxman100 --rates 1.0 --seeds 0 --arrivals 50 \
    --baselines nea --out "$OUT/gen_waxman100.json"
python scripts/eval_offline.py --checkpoint "$OUT/model/vne_gen.pt" --sources vne_gen,nea --sizes 20 --seeds 0 --rounds 2 \
    --time-limit 30 --out "$OUT/offline.json"
python scripts/eval_time_aware.py --checkpoint "$OUT/model/vne_gen.pt" --sources vne_gen,nea --rates 0.08 --seeds 0 \
    --episode-size 20 --rounds 2 --time-limit 30 --out "$OUT/time_aware.json"
python scripts/eval_optimality.py --checkpoint "$OUT/model/vne_gen.pt" --sources vne_gen,nea --seeds 2014 --rounds 2 \
    --out "$OUT/optimality.json"
python scripts/eval_runtime.py --checkpoint "$OUT/model/vne_gen.pt" --methods vne_gen,nea --substrates geant --calls 5 \
    --out "$OUT/runtime.json"
python scripts/plot_results.py online --input "$OUT/online.json" --out "$OUT/online.pdf"
echo "smoke test passed; outputs in $OUT"
