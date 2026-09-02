#!/usr/bin/env bash
set -euo pipefail

export OMP_NUM_THREADS=8
export MKL_NUM_THREADS=8
export PYTHONUNBUFFERED=1

v19_run=/workspace/runs/turtle-league-v19-opponent-transfer-v18-5000
v18_run=/workspace/runs/turtle-league-v18-audit-transfer-v17-1300-to5000
gate_update="${V19_GATE_UPDATE:-1000}"
gate_pairs="${V19_GATE_PAIRS:-100}"
gate_seed="${V19_GATE_SEED:-2026195000}"
candidate="${v19_run}/update-$(printf '%06d' "${gate_update}").pt"
baseline="${v18_run}/update-005600.pt"
output_dir="${v19_run}/gate"
output="${output_dir}/paired-$(printf '%06d' "${gate_update}")-vs-v18-005600.json"

python - <<'PY'
from sapai.turtle_pack import SIMULATOR_REVISION
expected = "turtle-current-opponent-v19-20260831"
if SIMULATOR_REVISION != expected:
    raise SystemExit(
        f"Refusing mixed-revision v19 gate: "
        f"{SIMULATOR_REVISION!r} != {expected!r}"
    )
PY

if [[ ! -f "${candidate}" || ! -f "${baseline}" ]]; then
  echo "Missing candidate or baseline: ${candidate} ${baseline}" >&2
  exit 1
fi
mkdir -p "${output_dir}"

if [[ ! -f "${output}" ]]; then
  python scripts/evaluate_paired_policy.py \
    "${candidate}" \
    --pairs "${gate_pairs}" \
    --opponent checkpoint \
    --opponent-checkpoint "${baseline}" \
    --device cpu \
    --batch-size 32 \
    --progress-every 50 \
    --seed "${gate_seed}" \
    --output "${output}"
fi

python - "${output}" <<'PY'
import json
import sys

payload = json.load(open(sys.argv[1]))
low, high = payload["pairedScore95CI"]
print(
    f"gate_update={payload['checkpointUpdate']} "
    f"score={payload['pairedMeanScore']:.4f} "
    f"ci95={low:.4f},{high:.4f}"
)
PY
