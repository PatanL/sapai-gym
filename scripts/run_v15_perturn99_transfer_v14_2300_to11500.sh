#!/usr/bin/env bash
set -euo pipefail

# Corrected v15 campaign semantics:
# - stable snapshot resolution for simultaneous Self-sell and friend-sold
#   abilities; the sold pet is removed exactly once;
# - all v14 discover-freeze and shop-capacity corrections;
# - discount 1.0 within shops and 0.99 only across battle boundaries.
checkpoint_dir=/workspace/runs/turtle-league-v15-perturn99-transfer-v14-2300-to11500
transfer_checkpoint=/workspace/runs/turtle-league-v14-perturn99-transfer2300-to11500/update-002300.pt
run_seed=54
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
    run_seed=$((54 + 10#${update_number}))
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
  "${initialization_args[@]}" \
  --allow-simulator-mismatch \
  --discount-mode per_turn \
  --discount 0.99 \
  --resident-checkpoint /workspace/runs/turtle-league-v14-perturn99-transfer2300-to11500/update-002100.pt \
  --resident-checkpoint /workspace/runs/turtle-league-v14-perturn99-transfer2300-to11500/update-001000.pt \
  --max-historical-residents 2 \
  --seed "${run_seed}"
