# Cheap END positioning: cost vs share of the oracle gain (2026-11-12)

Offline measurement only. Nothing is trained, and the checkpoint and deployed planner are unchanged.

**Question.** The deployed public positioner (`claude-planner-20261013-v1`, `--position-only --sim-opponent`)
captures **0.40** of the oracle's positioning gain, but it costs 2–4 s per end of turn. Does any setting near
**50 ms per END decision** keep most of that gain?

**Method.**
- The code is a copy of `claude-planner-20261013-v1`, with two changes:
  - per-decision wall-clock timing by part, logged in each diagnostic row (`timing`);
  - one new public opponent model, `--prev-as-sim`: the opponent's last revealed board, with no simulated turn.
- The **diagnostic scoring is excluded from the cost**: `--diag-oracle` plays 32 privileged battles per order against the opponent's real board, logging only.
- Every setting plays the same 6 seed pairs as `diag-t3` (namespace `claude-diag-sim-v1`, 12 games, about 117
  decisions), with 5 battle workers and 6 CPUs (`run.sh`).
- Runs are sequential, each behind the start gate (GPU < 20%, 1-min load < 12).
- `panel.sh` runs the panel and `analyze_panel.py` summarises it into `out/panel-summary.json`.

Parts of one END decision:

| Part | What it does |
|---|---|
| `paths` | enumerate reachable pet orders (pop-insert moves, reorder cap) |
| `sim` | simulate the opponent's turn from its last revealed board with the policy (GPU, lockstep) |
| `screen` | one battle per order per simulated board on reordered boards (only when there are more than 7 orders) |
| `apply` | clone the env and apply moves + END for the kept orders (END effects can depend on order) |
| `exact` | battles of the kept orders against the simulated boards |

## Results (`out/panel-summary.json`)

| Setting | Sim turns / exact top / seeds | Decisions | Captured | Real gain per decision | Gain kept vs ref | ms median (p90) | Battles |
|---|---|---|---|---|---|---|---|
| `ref` (deployed search, margin 0) | 8 / 6 / 4 | 117 | **0.40** | +0.112 ±0.036 | 100% | 1732 (1986) | 992 |
| `half` | 4 / 3 / 2 | 112 | 0.37 | +0.079 ±0.031 | 71% | 816 (946) | 418 |
| `small` | 2 / 2 / 1 | 124 | 0.25 | +0.040 ±0.031 | 36% | 454 (576) | 201 |
| `one` | 1 / 1 / 1 | 111 | 0.28 | +0.038 ±0.025 | 34% | 261 (351) | 100 |
| `prev` (no simulated turn) | last board / 2 / 2 | 118 | 0.21 | +0.027 ±0.016 | 24% | 171 (195) | 104 |

- **Compare by gain, not by the captured share:** the oracle's best order (`realBest`) is the best among the orders each setting kept, not among all reachable orders. Settings that keep fewer orders have a smaller denominator, which inflates their share. Real gain per decision is comparable across settings, and the conclusion uses it.
- **Reproduction:** `ref` reproduced `diag-t3` exactly: 0.40 captured, +0.112, the same 117 decisions. The
  seeded harness is deterministic.
- **Margin:** `ref` and the whole panel run with `position_margin 0` (the earlier "No margin" row). The deployed
  margin of 0.02 also captured 0.40, with a gain of +0.104 ±0.036.
- **Mean cost of a `ref` decision, by part:**
  - screen: 797 ms
  - sim: 397 ms
  - exact: 318 ms
  - apply: 81 ms
  - paths: under 1 ms
- **Cost per battle:** about 1 ms wall clock with 5 workers, so roughly 5 ms of CPU each.

**Conclusion.** No tested setting comes near 50 ms. Not one decision in any setting took 50 ms or less.
- **The cheapest public model (`prev`)** still costs 158 ms mean. It keeps only about a quarter of the real gain.
- **Where the floor comes from:** with up to 120 reachable orders, even one battle per order against one board is about 100 ms of screening. Cloning the env for END effects adds about 25–40 ms.
- **Halving everything (`half`)** keeps about 70% of the gain at about half the cost (0.8 s). Below that, the
  gain falls faster than the cost.
- **Noise:** the standard errors are about ±0.03, so `half` vs `ref` is within noise. `small`, `one` and `prev` are about 2 standard errors below `ref`.

**Not tested.** Two things could lower the search floor:
- screening fewer orders, such as a capped or learned candidate list;
- separating `pool.map` task overhead from battle time: the 1 ms per battle includes pickling the boards for each task.

**Pointer: the END-time module.** None of the tested search settings reaches 50 ms. Neither can the existing battle predictor: in `claude-planner-20261013-v1`, predictor-only positioning captured 0.23 with a gain of +0.029. The path is the separate END-time positioning module that the earlier README proposes:
- **Input:** our pre-END board, the opponent's revealed boards and the turn.
- **Output:** the order. One forward pass should be a few ms on GPU.
- **Training data must be public too:** the training positioner may only use what the deployed one sees. Train it on public-planner labels (`ref`, or `half` for twice the label throughput) or on real battle outcomes. Never use oracle-board choices as labels.
- **Labels already on disk:** 265 recorded seat-games in `../claude-planner-20261013-v1/out/distill-v1`.
- **Acceptance bar:** the same `--diag-oracle` panel as here, at 50 ms or less per decision. Gain kept against `ref` (+0.112) should be at least that of `half` (+0.079).
