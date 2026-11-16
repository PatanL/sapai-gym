#!/bin/bash
# Reference + four cheaper public positioning settings, run one after another on the same seeds as diag-t3
# (namespace claude-diag-sim-v1, 6 pairs = 12 games, ~117 decisions). Each waits for the start gate (GPU < 20%, load < 12).
set -u
cd "$(dirname "$0")"
COMMON="--mode planner --pairs 6 --namespace claude-diag-sim-v1 --workers 5 --position-only --sim-opponent --diag-oracle --position-margin 0.0"
run() { name=$1; shift
  while :; do g=$(nvidia-smi --query-gpu=utilization.gpu --format=csv,noheader,nounits | head -1); l=$(cut -d' ' -f1 /proc/loadavg)
    if [ "${g:-100}" -lt 20 ] && awk "BEGIN{exit !($l < 12)}"; then break; fi; echo "gate wait gpu=$g load=$l"; sleep 60; done
  echo "== $name start $(date +%T) gpu=$g load=$l"
  ./run.sh "cheap-$name" 6 10g match.py $COMMON "$@" --out "$name.jsonl" > "out/$name.log" 2>&1
  echo "== $name end $(date +%T) rc=$?"; }
run ref    --sim-samples 8 --screen-seeds 1 --exact-top 6 --sim-seeds 4
run half   --sim-samples 4 --screen-seeds 1 --exact-top 3 --sim-seeds 2
run small  --sim-samples 2 --screen-seeds 1 --exact-top 2 --sim-seeds 1
run one    --sim-samples 1 --screen-seeds 1 --exact-top 1 --sim-seeds 1
run prev   --prev-as-sim   --screen-seeds 1 --exact-top 2 --sim-seeds 2
echo "== panel done $(date +%T)"
