#!/usr/bin/env bash
set -euo pipefail

# Corrected v14 campaign semantics:
# - independent freezing of linked discover choices;
# - frozen discovers consume normal pet-shop capacity;
# - linked purchase removal persists when both choices are retained;
# - discount 1.0 within shops and 0.99 only across battle boundaries.
checkpoint_dir=/workspace/runs/turtle-league-v14-perturn99-transfer2300-to11500
transfer_checkpoint=/workspace/runs/turtle-league-v13-selected-pa99-diverse-to2000/update-002300.pt
run_seed=53
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
    run_seed=$((53 + 10#${update_number}))
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
  --expert-anchor-checkpoint /workspace/checkpoints/imitation-guide-v13-guide-v3.pt \
  --resident-checkpoint /workspace/runs/turtle-league-v13-guide-v3-onefood-fresh/update-000375.pt \
  --resident-checkpoint /workspace/runs/turtle-league-v13-guide-v3-onefood-fresh/update-000400.pt \
  --resident-checkpoint /workspace/runs/turtle-league-v13-guide-v3-discount-pt94-to1200/update-001200.pt \
  --resident-checkpoint /workspace/runs/turtle-league-v12-guide-v3-discount-pa99-to1600/update-001600.pt \
  --resident-checkpoint /workspace/runs/turtle-league-v13-selected-pa99-diverse-to2000/update-002100.pt \
  --resident-checkpoint /workspace/runs/turtle-league-v13-selected-pa99-diverse-to2000/update-002200.pt \
  --max-historical-residents 2 \
  --seed "${run_seed}"
