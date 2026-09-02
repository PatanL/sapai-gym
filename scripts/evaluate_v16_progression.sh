#!/usr/bin/env bash
set -euo pipefail

# Unattended longitudinal evaluation for the v16 campaign. This deliberately
# uses a bounded CPU allocation in a separate container so the learner retains
# exclusive access to the GPU.
export OMP_NUM_THREADS=8
export MKL_NUM_THREADS=8
export PYTHONUNBUFFERED=1

run_dir=/workspace/runs/turtle-league-v16-perturn99-transfer-v14-2300-to11500
evaluation_dir="${run_dir}/progression"
# Early checkpoints established update 1000 as an absolute baseline. From
# update 8000 onward, compare against the policy from 2000 updates earlier so
# the matchup remains discriminating instead of saturating against a weak
# anchor.
first_update=8000
last_update=11000
step=1000
pairs=50
behavior_games=50
reference_lag=2000
# Reuse update 2000's exact seed sets at every later milestone. This turns the
# fixed-anchor and behavior series into true longitudinal comparisons instead
# of introducing a fresh seed-distribution sample each time.
fixed_anchor_seed=20264000
behavior_seed=20284000

mkdir -p "${evaluation_dir}"

python - <<'PY'
from sapai.turtle_pack import SIMULATOR_REVISION
expected = "turtle-current-audit-v16-20260828"
if SIMULATOR_REVISION != expected:
    raise SystemExit(
        f"Refusing mixed-revision progression: {SIMULATOR_REVISION!r} != {expected!r}"
    )
PY

for ((update=first_update; update<=last_update; update+=step)); do
  candidate="${run_dir}/update-$(printf '%06d' "${update}").pt"
  while [[ ! -f "${candidate}" ]]; do
    sleep 30
  done

  reference_update=$((update - reference_lag))
  fixed="${run_dir}/update-$(printf '%06d' "${reference_update}").pt"
  fixed_output="${evaluation_dir}/paired-$(printf '%06d' "${update}")-vs-$(printf '%06d' "${reference_update}").json"
  if [[ ! -f "${fixed_output}" ]]; then
    echo "evaluating update=${update} opponent=${reference_update} pairs=${pairs}"
    python scripts/evaluate_paired_policy.py \
      "${candidate}" --pairs "${pairs}" --opponent checkpoint \
      --opponent-checkpoint "${fixed}" --device cpu \
      --batch-size 32 --progress-every 25 \
      --seed "${fixed_anchor_seed}" --output "${fixed_output}"
  fi

  fresh_output="${evaluation_dir}/paired-fresh-$(printf '%06d' "${update}")-vs-$(printf '%06d' "${reference_update}").json"
  if [[ ! -f "${fresh_output}" ]]; then
    # A distinct deterministic panel per milestone estimates generalization;
    # the fixed panel above remains the low-variance longitudinal canary.
    fresh_seed=$((20300000 + update))
    echo "evaluating-fresh update=${update} opponent=${reference_update} pairs=${pairs} seed=${fresh_seed}"
    python scripts/evaluate_paired_policy.py \
      "${candidate}" --pairs "${pairs}" --opponent checkpoint \
      --opponent-checkpoint "${fixed}" --device cpu \
      --batch-size 32 --progress-every 25 \
      --seed "${fresh_seed}" --output "${fresh_output}"
  fi

  behavior_output="${evaluation_dir}/behavior-v2-$(printf '%06d' "${update}")-vs-$(printf '%06d' "${reference_update}").json"
  if [[ ! -f "${behavior_output}" ]]; then
    echo "behavior update=${update} opponent=${reference_update} games=${behavior_games}"
    python scripts/evaluate_entity_policy.py \
      "${candidate}" --games "${behavior_games}" --opponent checkpoint \
      --opponent-checkpoint "${fixed}" --device cpu \
      --seed "${behavior_seed}" --output "${behavior_output}"
  fi

  python scripts/summarize_v16_progression.py "${evaluation_dir}"
done

echo "finished progression through update=${last_update}"
