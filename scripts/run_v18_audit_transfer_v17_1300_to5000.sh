#!/usr/bin/env bash
set -euo pipefail

checkpoint_dir=/workspace/runs/turtle-league-v18-audit-transfer-v17-1300-to5000
transfer_checkpoint=/workspace/runs/turtle-league-v17-shop-rabbit-v8-transfer-v16-8300-to5000/update-001300.pt
run_seed=83
initialization_args=(
  --init-checkpoint "${transfer_checkpoint}"
  --learning-rate 2e-6
)

if [[ -d "${checkpoint_dir}" ]]; then
  latest_checkpoint="$(
    find "${checkpoint_dir}" -maxdepth 1 -type f -name 'update-*.pt' \
      -print | sort | tail -n 1
  )"
  if [[ -n "${latest_checkpoint}" ]]; then
    update_name="${latest_checkpoint##*/update-}"
    update_number="${update_name%.pt}"
    run_seed=$((83 + 10#${update_number}))
    initialization_args=(
      --resume-checkpoint "${latest_checkpoint}"
      --resume-learning-rate 2e-6
    )
  fi
fi

if [[ ! -f "${initialization_args[1]}" ]]; then
  echo "Missing initialization checkpoint: ${initialization_args[1]}" >&2
  exit 1
fi

exec python scripts/train_entity_selfplay.py \
  --distributed \
  --updates 8300 \
  --num-envs 32 \
  --games-per-actor 3 \
  --segments-per-update 64 \
  --transitions-per-update 8192 \
  --trajectory-queue-size 1024 \
  --inference-batch-delay-ms 5 \
  --max-behavior-policy-lag 1 \
  --checkpoint-every 100 \
  --checkpoint-dir "${checkpoint_dir}" \
  "${initialization_args[@]}" \
  --allow-simulator-mismatch \
  --discount-mode per_turn \
  --discount 0.99 \
  --battle-aux-cost 0.05 \
  --max-historical-residents 2 \
  --seed "${run_seed}"
