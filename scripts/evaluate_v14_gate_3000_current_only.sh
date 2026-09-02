#!/usr/bin/env bash
set -euo pipefail

output_dir=/workspace/runs/turtle-league-v14-perturn99-transfer2300-to11500
checkpoint="${output_dir}/update-003000.pt"

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

# Guide and imitation-anchor evaluation were retired after saturating at the
# update-1000 gate. Spend evaluation compute only on informative v14 policies.
python scripts/evaluate_paired_policy.py \
  "${checkpoint}" --pairs 200 --opponent checkpoint \
  --opponent-checkpoint "${output_dir}/update-002100.pt" \
  --seed 20261301 --output "${output_dir}/paired-3000-vs-2100.json"

python scripts/evaluate_paired_policy.py \
  "${checkpoint}" --pairs 200 --opponent checkpoint \
  --opponent-checkpoint "${output_dir}/update-001000.pt" \
  --seed 20261302 --output "${output_dir}/paired-3000-vs-1000.json"

python scripts/evaluate_entity_policy.py \
  "${checkpoint}" --games 200 --opponent checkpoint \
  --opponent-checkpoint "${output_dir}/update-002100.pt" \
  --seed 20261303 \
  --output "${output_dir}/eval-3000-vs-2100-behavior-200.json"
