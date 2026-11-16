#!/bin/bash
# Public order labels for the learned END positioner: 300 games (150 seed pairs, both seats), in two resumable
# jobs on namespace claude-posnet-labels-v1 (disjoint from the evaluation panel, claude-diag-sim-v1).
cd "$(dirname "$0")"
COMMON="--mode planner --namespace claude-posnet-labels-v1 --workers 5 --position-only --both-seats --label-out --sim-samples 8 --label-seeds 4"
./run.sh posnet-labels-a 6 10g match.py $COMMON --first 0 --pairs 75 --out labels-a.jsonl > out/labels-a.log 2>&1 &
./run.sh posnet-labels-b 6 10g match.py $COMMON --first 75 --pairs 75 --out labels-b.jsonl > out/labels-b.log 2>&1 &
wait
echo "labels done $(date +%T)"
