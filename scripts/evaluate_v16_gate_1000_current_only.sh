#!/usr/bin/env bash
set -euo pipefail

# Evaluation runs beside the long-lived GPU learner. Keep it on a bounded CPU
# budget so loading two policies cannot evict the training model from unified
# memory on DGX Spark.
export OMP_NUM_THREADS=4
export MKL_NUM_THREADS=4
export PYTHONUNBUFFERED=1

output_dir=/workspace/runs/turtle-league-v16-perturn99-transfer-v14-2300-to11500
checkpoint="${output_dir}/update-001000.pt"

while [[ ! -f "${checkpoint}" ]]; do
  sleep 30
done

python - <<'PY'
from sapai.turtle_pack import SIMULATOR_REVISION
expected = "turtle-current-audit-v16-20260828"
if SIMULATOR_REVISION != expected:
    raise SystemExit(
        f"Refusing mixed-revision gate: {SIMULATOR_REVISION!r} != {expected!r}"
    )
PY

python scripts/evaluate_paired_policy.py \
  "${checkpoint}" --pairs 200 --opponent checkpoint \
  --opponent-checkpoint "${output_dir}/update-000500.pt" \
  --device cpu \
  --seed 20261601 --output "${output_dir}/paired-1000-vs-500.json"

python scripts/evaluate_paired_policy.py \
  "${checkpoint}" --pairs 200 --opponent checkpoint \
  --opponent-checkpoint /workspace/runs/turtle-league-v14-perturn99-transfer2300-to11500/update-002300.pt \
  --device cpu \
  --seed 20261602 --output "${output_dir}/paired-1000-vs-transfer-v14-2300.json"

python scripts/evaluate_entity_policy.py \
  "${checkpoint}" --games 200 --opponent checkpoint \
  --opponent-checkpoint "${output_dir}/update-000500.pt" \
  --device cpu \
  --seed 20261603 \
  --output "${output_dir}/eval-1000-vs-500-behavior-200.json"
