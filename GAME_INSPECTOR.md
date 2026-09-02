# Entity-policy game inspector

`scripts/visualize_entity_game.py` runs one deterministic match with CPU
inference and writes a self-contained HTML replay. It is intended for mechanic
audits and policy review, not strength evaluation.

```bash
cd /home/dzliu/pzliu/sap-rl/sapai-gym

python scripts/visualize_entity_game.py \
  ../runs/turtle-league-v16-perturn99-transfer-v14-2300-to11500/update-000700.pt \
  --opponent-checkpoint \
  ../runs/turtle-league-v16-perturn99-transfer-v14-2300-to11500/update-000600.pt \
  --seed 20260829 \
  --perspective p0 \
  --threads 4 \
  --output ../runs/game-inspector.html \
  --trace-json ../runs/game-inspector.json
```

## What it exposes

- Every visible shop action, organized by turn.
- Before/after toggling with exact team, shop, stats, equipment, experience,
  sell value, ability counters, freezes, physical slot indices, and discover
  group IDs.
- Exact state-change chips for gold, stats, team slots, shop slots, and future
  shop buffs.
- Live action-cap/reorder progress and the physical shop sources already
  freeze-toggled in the current shop state.
- Selected source and target highlighting.
- The last opponent board actually observable by the policy; current private
  opponent shops remain hidden.
- Submitted battle boards, outcome, net hearts, wins, and dense battle reward.
- The policy value, full chosen-action probability, entropy, and all legal
  action-type probabilities at each decision.
- Automatic warnings for non-atomic or duplicate discovers, shop overflow,
  unexpected gold accounting (including Pig/Cake sell math), nonempty sold
  slots, failed freeze/unfreeze mutations, and ending with unspent gold outside
  the 50-action escape hatch.
- Local pet, food, and equipment artwork from the checked-out
  `sap-replay-bot/Sprite` tree. The `runs/sap-sprites` symlink keeps generated
  HTML small and offline.

Use the Previous/Next buttons or left/right arrow keys to step. Press `B` to
toggle the current action between its before and after state. The optional JSON
trace is the machine-readable evidence for a suspected bug.

## Interpretation limits

- A deterministic sample demonstrates behavior but does not estimate win rate.
- Shop entries retain their physical list positions; after a purchase, later
  entries may shift left. The exact-change panel intentionally reports those
  shifts.
- The battle-outcome auxiliary head is included in the JSON but is not treated
  as a calibrated prediction during RL because the current RL loop does not
  train that auxiliary loss.
- The viewer uses the runtime simulator. Always compare the checkpoint revision
  displayed in the header with the runtime revision before treating a replay as
  native to that checkpoint.
- The replay-bot repository declares ISC package metadata, but the bundled game
  artwork's redistribution rights are not explicit. Keep sprite-backed viewers
  private/local unless Team Wood's asset license is confirmed.
