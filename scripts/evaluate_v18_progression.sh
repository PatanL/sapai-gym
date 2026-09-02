#!/usr/bin/env bash
set -euo pipefail

export OMP_NUM_THREADS=8
export MKL_NUM_THREADS=8
export PYTHONUNBUFFERED=1

run_dir=/workspace/runs/turtle-league-v18-audit-transfer-v17-1300-to5000
evaluation_dir="${run_dir}/progression"
parent=/workspace/runs/turtle-league-v17-shop-rabbit-v8-transfer-v16-8300-to5000/update-001300.pt
fixed_seed=2026181000
behavior_seed=2026191000
mkdir -p "${evaluation_dir}"

check_revision() {
  python - <<'PY'
from sapai.turtle_pack import SIMULATOR_REVISION
expected = "turtle-current-audit-v18-20260830"
if SIMULATOR_REVISION != expected:
    raise SystemExit(
        f"Refusing mixed-revision v18 evaluation: "
        f"{SIMULATOR_REVISION!r} != {expected!r}"
    )
PY
}

wait_for_checkpoint() {
  local checkpoint="$1"
  while [[ ! -f "${checkpoint}" ]]; do
    sleep 30
  done
}

run_paired() {
  local candidate="$1" opponent="$2" pairs="$3" seed="$4" output="$5"
  if [[ -f "${output}" ]]; then
    return
  fi
  check_revision
  python scripts/evaluate_paired_policy.py \
    "${candidate}" --pairs "${pairs}" --opponent checkpoint \
    --opponent-checkpoint "${opponent}" --device cpu \
    --batch-size 32 --progress-every 50 --seed "${seed}" \
    --output "${output}"
}

run_behavior() {
  local candidate="$1" opponent="$2" seed="$3" output="$4"
  if [[ -f "${output}" ]]; then
    return
  fi
  check_revision
  python scripts/evaluate_entity_policy.py \
    "${candidate}" --games 50 --opponent checkpoint \
    --opponent-checkpoint "${opponent}" --device cpu \
    --seed "${seed}" --output "${output}"
}

# First milestone: establish both adaptation from the v17 parent and progress
# within v18 using a stronger 200-game paired panel for each comparison.
candidate_1000="${run_dir}/update-001000.pt"
checkpoint_500="${run_dir}/update-000500.pt"
wait_for_checkpoint "${candidate_1000}"
run_paired "${candidate_1000}" "${parent}" 100 "${fixed_seed}" \
  "${evaluation_dir}/paired-001000-vs-v17-001300.json"
run_paired "${candidate_1000}" "${checkpoint_500}" 100 "${fixed_seed}" \
  "${evaluation_dir}/paired-001000-vs-000500.json"
run_behavior "${candidate_1000}" "${checkpoint_500}" "${behavior_seed}" \
  "${evaluation_dir}/behavior-001000-vs-000500.json"

# Later milestones use a rolling 1,000-update reference and 100 paired games,
# preserving most compute for training while maintaining a fixed seed panel.
for update in 2000 3000 4000 5000 6000 7000 8000; do
  reference=$((update - 1000))
  candidate="${run_dir}/update-$(printf '%06d' "${update}").pt"
  opponent="${run_dir}/update-$(printf '%06d' "${reference}").pt"
  wait_for_checkpoint "${candidate}"
  run_paired "${candidate}" "${opponent}" 50 "${fixed_seed}" \
    "${evaluation_dir}/paired-$(printf '%06d' "${update}")-vs-$(printf '%06d' "${reference}").json"
  run_behavior "${candidate}" "${opponent}" "${behavior_seed}" \
    "${evaluation_dir}/behavior-$(printf '%06d' "${update}")-vs-$(printf '%06d' "${reference}").json"
done

echo "finished v18 progression through update 8000"
