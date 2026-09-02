#!/usr/bin/env bash
set -euo pipefail

# Permanent campaign semantics from this fork onward:
# discount 1.0 for shop actions and 0.99 only at battle boundaries.
checkpoint_dir=/workspace/runs/turtle-league-v13-perturn99-from2300-to11500
base_checkpoint=/workspace/runs/turtle-league-v13-selected-pa99-diverse-to2000/update-002300.pt
resume_checkpoint="${base_checkpoint}"
run_seed=52

if [[ -d "${checkpoint_dir}" ]]; then
  latest_checkpoint="$(
    find "${checkpoint_dir}" -maxdepth 1 -type f -name 'update-*.pt' \
      -print | sort | tail -n 1
  )"
  if [[ -n "${latest_checkpoint}" ]]; then
    resume_checkpoint="${latest_checkpoint}"
    update_name="${latest_checkpoint##*/update-}"
    update_number="${update_name%.pt}"
    run_seed=$((52 + 10#${update_number}))
  fi
fi

if [[ ! -f "${resume_checkpoint}" ]]; then
  echo "Missing resume checkpoint: ${resume_checkpoint}" >&2
  exit 1
fi

exec python scripts/train_entity_selfplay.py \
  --distributed \
  --updates 11500 \
  --num-envs 32 \
  --games-per-actor 3 \
  --segments-per-update 64 \
  --transitions-per-update 8192 \
  --trajectory-queue-size 1024 \
  --inference-batch-delay-ms 5 \
  --max-behavior-policy-lag 1 \
  --checkpoint-every 100 \
  --checkpoint-dir "${checkpoint_dir}" \
  --resume-checkpoint "${resume_checkpoint}" \
  --resume-learning-rate 2e-6 \
  --allow-simulator-mismatch \
  --discount-mode per_turn \
  --discount 0.99 \
  --expert-anchor-checkpoint /workspace/checkpoints/imitation-guide-v13-guide-v3.pt \
  --resident-checkpoint /workspace/runs/turtle-league-v13-guide-v3-onefood-fresh/update-000375.pt \
  --resident-checkpoint /workspace/runs/turtle-league-v13-guide-v3-onefood-fresh/update-000400.pt \
  --resident-checkpoint /workspace/runs/turtle-league-v13-guide-v3-discount-pt94-to1200/update-001200.pt \
  --resident-checkpoint /workspace/runs/turtle-league-v12-guide-v3-discount-pa99-to1600/update-001600.pt \
  --resident-checkpoint /workspace/runs/turtle-league-v13-selected-pa99-diverse-to2000/update-002100.pt \
  --resident-checkpoint /workspace/runs/turtle-league-v13-selected-pa99-diverse-to2000/update-002200.pt \
  --max-historical-residents 2 \
  --seed "${run_seed}"
