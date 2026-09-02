#!/usr/bin/env bash
set -euo pipefail

checkpoint=/workspace/runs/turtle-league-v13-selected-pa99-diverse-to2000/update-002000.pt
output_dir=/workspace/runs/turtle-league-v13-selected-pa99-diverse-to2000

while [[ ! -f "${checkpoint}" ]]; do
  sleep 30
done

# Serialize live-gate jobs so uninterrupted training retains most of the GPU.
python scripts/evaluate_paired_policy.py \
  "${checkpoint}" --pairs 250 --opponent checkpoint \
  --opponent-checkpoint "${output_dir}/update-001600.pt" \
  --seed 20261106 --output "${output_dir}/paired-2000-vs-1600.json"

python scripts/evaluate_paired_policy.py \
  "${checkpoint}" --pairs 250 --opponent guide --seed 20260828 \
  --output "${output_dir}/paired-2000-guide.json"

python scripts/evaluate_entity_policy.py \
  "${checkpoint}" --games 200 --opponent guide --seed 20261102 \
  --output "${output_dir}/eval-2000-vs-guide-behavior-200.json"
