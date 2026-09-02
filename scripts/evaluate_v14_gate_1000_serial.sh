#!/usr/bin/env bash
set -euo pipefail

output_dir=/workspace/runs/turtle-league-v14-perturn99-transfer2300-to11500
checkpoint="${output_dir}/update-001000.pt"

while [[ ! -f "${checkpoint}" ]]; do
  sleep 30
done

python - <<'PY'
from sapai.turtle_pack import SIMULATOR_REVISION
expected = "turtle-current-audit-v14-20260828"
if SIMULATOR_REVISION != expected:
    raise SystemExit(
        f"Refusing mixed-revision gate: {SIMULATOR_REVISION!r} != {expected!r}"
    )
PY

# Sparse diagnostic gate only. Results never revert per-turn semantics.
python scripts/evaluate_paired_policy.py \
  "${checkpoint}" --pairs 200 --opponent checkpoint \
  --opponent-checkpoint /workspace/runs/turtle-league-v13-selected-pa99-diverse-to2000/update-002300.pt \
  --seed 20261201 --output "${output_dir}/paired-1000-vs-transfer-2300.json"

python scripts/evaluate_paired_policy.py \
  "${checkpoint}" --pairs 200 --opponent checkpoint \
  --opponent-checkpoint /workspace/runs/turtle-league-v13-selected-pa99-diverse-to2000/update-002000.pt \
  --seed 20261202 --output "${output_dir}/paired-1000-vs-v13-2000.json"

python scripts/evaluate_paired_policy.py \
  "${checkpoint}" --pairs 200 --opponent checkpoint \
  --opponent-checkpoint /workspace/checkpoints/imitation-guide-v13-guide-v3.pt \
  --seed 20261203 --output "${output_dir}/paired-1000-vs-imitation-anchor.json"

python scripts/evaluate_paired_policy.py \
  "${checkpoint}" --pairs 200 --opponent guide --seed 20261204 \
  --output "${output_dir}/paired-1000-vs-guide-v4.json"

python scripts/evaluate_entity_policy.py \
  "${checkpoint}" --games 200 --opponent guide --seed 20261205 \
  --output "${output_dir}/eval-1000-vs-guide-v4-behavior-200.json"
