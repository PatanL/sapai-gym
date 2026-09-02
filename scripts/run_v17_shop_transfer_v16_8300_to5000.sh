#!/usr/bin/env bash
set -euo pipefail

# Corrected v17 shop campaign:
# - Rabbit and Seal trigger once on any food or genuine perk gain;
# - built-in Bus Chili is not a perk-gain event;
# - shop summon capacity and Honey/Spider ordering are regression-locked;
# - stable identity-based shop Hurt/Faint/Pill resolution;
# - authoritative shared RNG for random shop effects;
# - exact incomplete-shop serialization/restoration;
# - all v16 battle, discover, overflow, and per-turn-discount corrections.
checkpoint_dir=/workspace/runs/turtle-league-v17-shop-rabbit-v8-transfer-v16-8300-to5000
transfer_checkpoint=/workspace/runs/turtle-league-v16-perturn99-transfer-v14-2300-to11500/update-008300.pt
run_seed=71
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
    run_seed=$((71 + 10#${update_number}))
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
  --updates 6200 \
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
  --max-historical-residents 2 \
  --seed "${run_seed}"
