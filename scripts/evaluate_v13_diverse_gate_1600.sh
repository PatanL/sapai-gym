#!/usr/bin/env bash
set -euo pipefail

checkpoint=/workspace/runs/turtle-league-v13-selected-pa99-diverse-to2000/update-001600.pt
output_dir=/workspace/runs/turtle-league-v13-selected-pa99-diverse-to2000

while [[ ! -f "${checkpoint}" ]]; do
  sleep 30
done

python scripts/evaluate_paired_policy.py \
  "${checkpoint}" --pairs 250 --opponent guide --seed 20260828 \
  --output "${output_dir}/paired-1600-guide.json" &
guide_pid=$!

python scripts/evaluate_paired_policy.py \
  "${checkpoint}" --pairs 250 --opponent checkpoint \
  --opponent-checkpoint /workspace/runs/turtle-league-v13-guide-v3-discount-pa99-to1200/update-001200.pt \
  --seed 20261105 --output "${output_dir}/paired-1600-vs-1200.json" &
baseline_pid=$!

python scripts/evaluate_paired_policy.py \
  "${checkpoint}" --pairs 250 --opponent checkpoint \
  --opponent-checkpoint /workspace/checkpoints/imitation-guide-v13-guide-v3.pt \
  --seed 20260930 --output "${output_dir}/paired-1600-vs-anchor.json" &
anchor_pid=$!

python scripts/evaluate_paired_policy.py \
  "${checkpoint}" --pairs 250 --opponent checkpoint \
  --opponent-checkpoint /workspace/runs/turtle-league-v13-guide-v3-onefood-fresh/update-000375.pt \
  --seed 20261001 --output "${output_dir}/paired-1600-vs-375.json" &
historical_pid=$!

python scripts/evaluate_entity_policy.py \
  "${checkpoint}" --games 200 --opponent guide --seed 20261102 \
  --output "${output_dir}/eval-1600-vs-guide-behavior-200.json" &
behavior_pid=$!

wait "${guide_pid}" "${baseline_pid}" "${anchor_pid}" "${historical_pid}" "${behavior_pid}"
