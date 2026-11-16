#!/bin/bash
# Learned END positioner vs the deployed search, on the same 6 seed pairs as the cost panel and diag-t3
# (namespace claude-diag-sim-v1). Diagnostic scoring (privileged, every order vs the opponent's real board) is
# logged only and excluded from the timing. usage: ./panel.sh CKPT [NAME]
set -u
cd "$(dirname "$0")"
CKPT=$1; NAME=${2:-$(basename "$CKPT" .pt)}
COMMON="--mode planner --pairs 6 --namespace claude-diag-sim-v1 --workers 5 --position-only --diag-oracle"
while :; do l=$(cut -d' ' -f1 /proc/loadavg); awk "BEGIN{exit !($l < 12)}" && break; echo "gate wait load=$l"; sleep 60; done
echo "== $NAME start $(date +%T) load=$l"
./run.sh "posnet-eval-$NAME" 6 10g match.py $COMMON --posnet "$CKPT" --out "eval-$NAME.jsonl" > "out/eval-$NAME.log" 2>&1
echo "== $NAME end $(date +%T) rc=$?"
