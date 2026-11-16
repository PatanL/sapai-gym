#!/bin/bash
# Run a script from this directory inside the frozen SAP image: project read-only, only this run dir writable.
# usage: ./run.sh NAME CPUS MEM script.py [args...]
set -euo pipefail
HERE=$(cd "$(dirname "$0")" && pwd); ROOT=$(cd "$HERE/../.." && pwd); REL=runs/$(basename "$HERE")
NAME=$1; CPUS=$2; MEM=$3; shift 3
exec docker run --rm --name "claude-sap-$NAME" --gpus all --cpus "$CPUS" --memory "$MEM" --memory-swap "$MEM" --shm-size 2g \
  --network none --user "$(id -u):$(id -g)" -e PYTHONDONTWRITEBYTECODE=1 -e PYTHONUNBUFFERED=1 -e OMP_NUM_THREADS=1 \
  -e OPENBLAS_NUM_THREADS=1 -e TORCH_ALLOW_TF32_CUBLAS_OVERRIDE=0 \
  -v "$ROOT:/workspace:ro" -v "$HERE:/workspace/$REL" -w "/workspace/$REL" \
  --entrypoint python sha256:3ed8782e02a5d46e400bc39a5a77654c26e8a86b1d210c160e5965b765955ce1 "$@"
