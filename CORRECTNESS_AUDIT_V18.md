# Turtle P1 correctness and CPU-work audit — v18

Audit date: 2026-08-30

Simulator revision: `turtle-current-audit-v18-20260830`

Guide revision: `turtle-p1-guide-v8-20260829`

Pinned battle authority: SAP-Calculator
`d165eb0a02f8aa0b54d72ed1d5490a44390d07f4`

## Confirmed v18 corrections

### Badger nearest-living targeting

SAP-Calculator resolves Badger against the nearest living pet on each side.
The old physical-neighbor implementation could miss through an empty shop
team slot or target an already-dead queued pet. V18 skips gaps/deaths and still
reaches the opposing front pet when no friendly pet remains ahead.

### Complete legality state for the entity policy

The global entity now exposes shop-action progress, reorder progress,
freeze-toggle count, visited-order count, and turn-ended state. Shop tokens mark
which concrete items were toggled in the current shop state. The complete
type/source/target legal masks are also projected into the transformer global
token, allowing the policy and critic to distinguish exact reorder and toggle
histories rather than using masks only after logits are produced.

The legality projection is zero-initialized. `EntityTransformerPolicy` migrates
pre-v18 checkpoints by injecting that neutral tensor, so transfer loading does
not fail or immediately add a random representation.

### RL battle auxiliary target

Every turn segment now carries the acting side's exact win/loss/draw outcome.
RL trains the existing battle auxiliary head with configurable
`--battle-aux-cost` (default `0.05`). Frozen-resident trajectories remain
excluded, so no resident GRU representation enters the main learner.

### CPU environment throughput

- Exact legal-action dictionaries are cached behind a legality fingerprint.
- The RL environment disables full `Player.action_history` serialization;
  legacy/debug Player users retain recording by default.
- A reproducible profiler is available at
  `scripts/profile_cpu_environment.py`.

On the same 20,000-decision, 515-battle CPU profile:

- before: 1,042 actions/second
- after: 1,494 actions/second
- improvement: 43.3%

### Replay reconstruction

- Untargeted random/team food actions no longer require a nonexistent Aim.
- Combine requests propagate a known retained-pet identity to an anonymous
  source copy through the exact same-pet constraint.
- Unknown items remain quarantined; no name is guessed from stats or hash.

The cached 14-turn private replay retains 28 hash-validated end states and 24
genuinely unresolved purchase/food identities. Exact post-roll shop recovery
remains blocked on Team Wood's unpublished seed-to-shop algorithm. Public code
search found simulators and live UI mods, but no implementation that converts
the replay response Seed/Hash into the full hidden shop.

The supplied expert-player page-3 panel now has a reproducible cached report:

- 10 Turtle-vs-Turtle replays and exactly 1,004 shop decisions
- 449/539 object actions resolved exactly (83.3%)
- Cow bought 10 times and sold 10 times
- all eight Cake purchases occurred on turns 5–6; seven were held into the
  turn-9 shop and three remained on an end-turn board at turn 9 or later
- Parrot–Deer appeared on six end-turn boards, Elephant–Blowfish on 16,
  Elephant–Camel on three, and Mammoth–Tiger on five

Report: `replay-data/player-analysis/player-03978ba3-c677-49a8-adb9-d8d99d9bd144-page-3.json`.

### Viewer/debugging

The HTML inspector now shows action-cap progress, reorder progress, and current
freeze-toggle sources. It validates SELL gold including Pig and accumulated
Cake value, verifies emptied sell slots, and verifies freeze/unfreeze mutations.

## Cross-engine evidence

`sapai/tools/differential_calculator.py` and `calculator_batch.js` run the
pinned SAP-Calculator bundle and sapai on identical randomized Turtle boards.
The v18 run covered all 60 Turtle pets across 40 scenarios and 10,000 battles
per engine:

- statistically flagged scenarios: 0
- mean absolute player-score delta: 0.0002
- maximum absolute delta: 0.0060

Report: `runs/differential-calculator-v18-seed20260830.json`.

## Checkpoint evidence

- v17 update 1,000 vs v16 update 8,300 parent: 121–79 (60.5%, 95% CI
  54.1–66.9%).
- v17 update 1,000 vs v17 update 500: 104–96 (52.0%, CI 46.1–57.9%).
- v17 update 1,300 vs v17 update 1,000 under the v18 Badger runtime:
  100–100 (50.0%, CI 43.9–56.1%). This last result is explicitly a
  cross-revision diagnostic, not a native-v17 promotion gate.

The evidence supports strong early adaptation by update 1,000 and no measured
gain from update 1,000 to 1,300.

## Verification

- Focused v18 gate: 321 passing tests.
- Production-size v17-update-1300 to v18 CPU transfer smoke: passed one learner
  update and checkpoint save.
- Small-model CPU RL smoke with battle auxiliary loss: passed two updates.
- Production-size distributed CPU transfer smoke: passed two updates, shared
  trajectory-ring transport, resident routing, checkpoint rotation, and save.
- Observation transfer panel: 0/256 deterministic actions changed from the
  newly nonzero history fields; mean type-distribution L1 was 0.0357. The value
  head requires recalibration (mean absolute delta 0.0839), which is expected
  and is why v18 starts a fresh optimizer branch.

## Explicit unresolved live-rule questions

These remain documented rather than guessed:

1. Whether Rabbit/Seal respond to genuine perk gains during battle or only in
   the shop.
2. Whether reapplying an already-equipped identical perk counts as a new gain.
3. Cow replacement behavior when extra Pigeon/Worm foods and frozen entries
   coexist at the ten-slot cap.
4. Whether accumulated Cake sell bonuses add or take the maximum when two
   equipped copies combine. Current code takes the maximum.
5. Whether a shop-pilled Rat creates any persistent opponent-side effect.
6. Exact insertion positions for Worm/Pigeon stocked food.
7. Exact live behavior when an atomic discover pair has one free capacity slot
   but no removable unfrozen entry. Current code discards the pair.
8. Equal-priority ordering for non-start/end shop observers. Start/end turn is
   confirmed attack-priority with seeded random ties.

## Restart/launch gate

- The distributed CUDA smoke passed two updates without OOM, actor failure, or
  policy lag above one. Full v18 training launched on 2026-08-31 without
  stopping the idle VLLM services.
- Do not resume the v17 optimizer or league state.
- Initialize v18 weights from v17 update 1,300 with a fresh optimizer, update
  counter, transition counter, and v18 league.
- Old imitation files carry pre-v18 observations. Do not use them for gradients
  unless regenerated with v18 revision metadata. The prepared campaign uses no
  demonstrations.
- The 321-test focused gate and two-update distributed CUDA smoke both passed
  before launch.
