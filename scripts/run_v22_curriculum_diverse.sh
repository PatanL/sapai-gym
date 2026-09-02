#!/usr/bin/env bash
set -euo pipefail

checkpoint_dir=/workspace/runs/turtle-league-v22-curriculum-diverse-v3
control_dir=/workspace/runs/turtle-league-v21-live-sync-transfer-v19-600-to5000
curriculum=/workspace/runs/curriculum/turtle-p1-scenarios-v3-20260902.jsonl
target_updates="${V22_TARGET_UPDATES:-5000}"
run_seed=2202

transfer_checkpoint="${V22_TRANSFER_CHECKPOINT:-}"
if [[ -z "${transfer_checkpoint}" ]]; then
  transfer_checkpoint="$({
    find "${control_dir}" -maxdepth 1 -type f -name 'update-*.pt' -print
  } | sort | tail -n 1)"
fi

initialization_args=(
  --init-checkpoint "${transfer_checkpoint}"
  --learning-rate 2e-6
)

if [[ -d "${checkpoint_dir}" ]]; then
  latest_checkpoint="$({
    find "${checkpoint_dir}" -maxdepth 1 -type f -name 'update-*.pt' -print
  } | sort | tail -n 1)"
  if [[ -n "${latest_checkpoint}" ]]; then
    update_name="${latest_checkpoint##*/update-}"
    update_number="${update_name%.pt}"
    run_seed=$((2202 + 10#${update_number}))
    initialization_args=(
      --resume-checkpoint "${latest_checkpoint}"
      --resume-learning-rate 2e-6
    )
  fi
fi

for required in "${initialization_args[1]}" "${curriculum}" \
  "${control_dir}/update-002000.pt" \
  "${control_dir}/update-005000.pt" \
  "${control_dir}/update-006500.pt" \
  "${control_dir}/update-010400.pt"; do
  if [[ ! -f "${required}" ]]; then
    echo "Missing required input: ${required}" >&2
    exit 1
  fi
done

exec python scripts/train_entity_selfplay.py \
  --distributed \
  --updates "${target_updates}" \
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
  --discount-mode per_turn \
  --discount 0.99 \
  --battle-aux-cost 0.05 \
  --opponent-aux-cost 0.1 \
  --opponent-module-learning-rate 1e-4 \
  --opponent-adapter-learning-rate 1e-5 \
  --demo-data "${curriculum}" \
  --demo-batch-size 64 \
  --demo-until-update 1000 \
  --demo-every 2 \
  --demo-learning-rate-scale 0.2 \
  --expert-anchor-checkpoint "${control_dir}/update-002000.pt" \
  --resident-checkpoint "${control_dir}/update-005000.pt" \
  --resident-checkpoint "${control_dir}/update-006500.pt" \
  --resident-checkpoint "${control_dir}/update-010400.pt" \
  --max-historical-residents 6 \
  --seed "${run_seed}"
