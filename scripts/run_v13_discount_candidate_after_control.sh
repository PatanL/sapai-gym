#!/usr/bin/env bash
set -euo pipefail

control_checkpoint=/workspace/runs/turtle-league-v13-guide-v3-discount-pa99-to1200/update-001200.pt

while [[ ! -f "${control_checkpoint}" ]]; do
  sleep 30
done

# The checkpoint is written at the end of the final update. Give the control
# process time to flush league/config state and exit before allocating CUDA.
sleep 30

exec python scripts/train_entity_selfplay.py \
  --distributed \
  --updates 1200 \
  --num-envs 32 \
  --games-per-actor 3 \
  --segments-per-update 64 \
  --transitions-per-update 8192 \
  --trajectory-queue-size 1024 \
  --inference-batch-delay-ms 5 \
  --max-behavior-policy-lag 1 \
  --checkpoint-every 25 \
  --checkpoint-dir /workspace/runs/turtle-league-v13-guide-v3-discount-pt94-to1200 \
  --resume-checkpoint /workspace/runs/turtle-league-v13-guide-v3-onefood-fresh/update-000400.pt \
  --resume-learning-rate 5e-6 \
  --discount-mode per_turn \
  --discount 0.94 \
  --expert-anchor-checkpoint /workspace/checkpoints/imitation-guide-v13-guide-v3.pt \
  --resident-checkpoint /workspace/runs/turtle-league-v13-guide-v3-onefood-fresh/update-000375.pt \
  --resident-checkpoint /workspace/runs/turtle-league-v13-guide-v3-onefood-fresh/update-000400.pt \
  --max-historical-residents 0 \
  --seed 49
