#!/usr/bin/env bash
# Full pipeline: train every experiment in configs/default.yaml, evaluate, explain, render demos, build report figures.
# Usage: bash scripts/run_all.sh            (about 45-60 min on a 2-core CPU)
#        QUICK=1 bash scripts/run_all.sh    (smoke test, a few minutes)
set -euo pipefail
cd "$(dirname "$0")/.."
if [[ "${QUICK:-0}" == "1" ]]; then
  # writes to runs_quick/ and results_quick/ so the real results are never overwritten
  python scripts/train.py --algo ppo --policy mlp --timesteps 20000 --out runs_quick/ppo_mlp_s0
  python scripts/train.py --algo ppo --policy attention --timesteps 20000 --out runs_quick/ppo_attention_s0
  python scripts/evaluate.py --episodes 10 --runs runs_quick/ppo_mlp_s0 runs_quick/ppo_attention_s0 --out results_quick/evaluation
  python scripts/explain.py --model runs_quick/ppo_mlp_s0/model.zip --n-episodes 3 --n-states 50 --out results_quick/explain_ppo_mlp_s0
  python scripts/demo.py --models runs_quick/ppo_mlp_s0/model.zip --scenarios corridor --out results_quick/demo
  exit 0
fi
for ex in ppo_mlp ppo_attention dqn_mlp ppo_mlp_cont; do
  python scripts/train.py --experiment "$ex"
done
python scripts/evaluate.py
for m in runs/ppo_mlp_s0 runs/ppo_attention_s0 runs/dqn_mlp_s0 runs/ppo_mlp_cont_s0; do
  python scripts/explain.py --model "$m/model.zip"
done
python scripts/demo.py --models runs/ppo_mlp_s0/model.zip runs/ppo_attention_s0/model.zip
python scripts/make_report_figs.py
