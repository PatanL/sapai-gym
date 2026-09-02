#!/usr/bin/env bash
set -euo pipefail

v13_checkpoint=/workspace/runs/turtle-league-v13-selected-pa99-diverse-to2000/update-001600.pt

while [[ ! -f "${v13_checkpoint}" ]]; do
  sleep 30
done

exec python scripts/evaluate_paired_policy.py \
  "${v13_checkpoint}" \
  --pairs 250 \
  --opponent checkpoint \
  --opponent-checkpoint /workspace/runs/turtle-league-v12-guide-v3-discount-pa99-to1600/update-001600.pt \
  --seed 20261104 \
  --output /workspace/runs/turtle-league-v13-selected-pa99-diverse-to2000/paired-1600-vs-v12-1600-diagnostic.json
