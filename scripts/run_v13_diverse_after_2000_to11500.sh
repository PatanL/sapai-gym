#!/usr/bin/env bash
set -euo pipefail

resume_checkpoint=/workspace/runs/turtle-league-v13-selected-pa99-diverse-to2000/update-002000.pt

while [[ ! -f "${resume_checkpoint}" ]]; do
  sleep 30
done

# Allow the update-2000 process to flush league state and exit before CUDA is
# allocated by the continuation.
sleep 30

exec python scripts/train_entity_selfplay.py \
  --distributed \
  --updates 11500 \
  --num-envs 32 \
  --games-per-actor 3 \
  --segments-per-update 64 \
  --transitions-per-update 8192 \
  --trajectory-queue-size 1024 \
  --inference-batch-delay-ms 5 \
  --max-behavior-policy-lag 1 \
  --checkpoint-every 100 \
  --checkpoint-dir /workspace/runs/turtle-league-v13-selected-pa99-diverse-to2000 \
  --resume-checkpoint "${resume_checkpoint}" \
  --resume-learning-rate 2e-6 \
  --allow-simulator-mismatch \
  --discount-mode per_action \
  --discount 0.99 \
  --expert-anchor-checkpoint /workspace/checkpoints/imitation-guide-v13-guide-v3.pt \
  --resident-checkpoint /workspace/runs/turtle-league-v13-guide-v3-onefood-fresh/update-000375.pt \
  --resident-checkpoint /workspace/runs/turtle-league-v13-guide-v3-onefood-fresh/update-000400.pt \
  --resident-checkpoint /workspace/runs/turtle-league-v13-guide-v3-discount-pt94-to1200/update-001200.pt \
  --resident-checkpoint /workspace/runs/turtle-league-v12-guide-v3-discount-pa99-to1600/update-001600.pt \
  --max-historical-residents 2 \
  --seed 51
