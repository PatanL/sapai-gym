# Turn planner on top of the plateaued RL agent (Claude, 2026-10-13)

Goal (user): beat, and ideally drastically improve, the current RL setup. The current best checkpoint,
`runs/ordinary-rl-shared-20260924-v2/outputs/chunk-20/update-025772.pt` (221M decisions), scores about 64% against
the 10050 anchor and about 50% against every parent since update 20806.

## Approach

A planner wraps the unchanged checkpoint at inference. It chooses each shop turn by the **next battle's equity**
against the opponent's board distribution, instead of noisy whole-game rollouts or the value net. Earlier work
showed the value net doesn't rank actions (correlation −0.02) and that rollout search collapses under selection
optimism.

- **Opponent boards.** `build_library.py` plays 600 greedy self-play games of the same checkpoint and keeps every
  battle board, about 14k. Opponents for a turn are sampled from the same turn. With `--condition`, the planner
  prefers library boards whose previous board shares pets with the opponent's last revealed board.
- **No peeking.** Plans run on clones whose shop tape is replaced (`enable_paired_tape`, fresh seed), so no real
  future roll or ability outcome is visible. The clone's opponent is never read.
- **Plan sources:**
  - `--enumerate`: every distinct end-of-turn board reachable without rolling (buy, food, sell, combine), found by
    DFS across a worker pool.
  - Default: the policy's own sampled plans. This turned out to be neutral.
- **The baseline it must beat** is the policy's own expected turn *with its rolls*: the policy plays the rest of the
  turn on 8 fresh-shop clones, and their boards' equity is averaged. A no-roll greedy baseline is a strawman (it
  ends turns with gold unused) and led to losses.
- **Choice.** Override only if the best enumerated board beats that expectation by `--enum-margin` (0.06). Then
  commit and block the policy's rolls for that turn. Otherwise follow the policy for the whole turn.
- **`--position`.** Before END, search the reachable pet orders (pop-insert moves, reorder cap, no revisits). The
  battle predictor screens them, and the top 7 are scored exactly, including order-dependent END effects.
- **Final choices are always made with exact sapai battles** (CRN: same opponents and seeds for every candidate).
- **Battle predictor** (`bp_data.py`, `bp_train.py`, `out/battle_net.pt`): a 4-layer transformer over 10 pet slots.
  It reaches correlation 0.84 with 64-battle equities and is used only to screen.

## Harness

`match.py` mirrors `public_match_evaluation.play`: paired-tape worlds, both seats, greedy FP32 actor, the same
winner function. The raw-vs-raw control comes out at exactly 0.5 (every pair splits).
`./run.sh NAME CPUS MEM script.py ...` runs inside the frozen image with the project read-only.
`summarize.py out/eval-v3-*.jsonl` reports the paired score, bootstrap 95% interval, pair outcomes and battle share.

## Results: paired evaluations against raw greedy 25772, both seats per seed

The raw-vs-raw control comes out at exactly 0.500.

| Planner variant | Pairs | Match score [95% bootstrap] | Battle share |
|---|---|---|---|
| Calibration: raw 25772 vs raw 10050 (no planner) | 93 | **0.645** [0.58, 0.70] | 0.560 |
| Policy-sampled plans + positioning (library opponents) | 12 | 0.50 | 0.47 |
| Enumerated plans, exact-spend only (v3) | 7 | 0.50 | 0.51 |
| Positioning only, library opponents | 100 | 0.52 [0.46, 0.58] | 0.51 |
| Positioning only, opponent's previous revealed board | 11 | 0.55 | 0.52 |
| **Positioning only, simulated opponent turns** (public) | 99 | **0.657** [0.59, 0.72] (41 won / 48 split / 10 lost) | 0.554 |
| same, fresh-seed confirmation panel (`claude-confirm-v1`) | 100 | **0.660** [0.60, 0.72] (8 pairs lost) | 0.550 |
| same + `--mix-prev --screen-seeds 4 --exact-top 8 --sim-seeds 8`, fresh panel (`claude-confirm-v2`) | 96 | **0.662** [0.61, 0.71] (33 won / 61 split / 2 lost) | 0.553 |
| Opponent-aware turn planning (enumerated plans scored against simulated opponents) | 5 | 0.20 (stopped) | 0.35 |
| *Oracle (privileged diagnostic): positioning vs the opponent's actual board* | 8 | *0.94* [0.81, 1.0] | *0.67* |

**Key finding.** Next-battle tactics are worth a lot, but only against the *actual* opponent. Positioning alone
with perfect knowledge of the opponent's board wins about 94% of matches. Library averages capture almost nothing.
So the lever is opponent modelling:

- `--sim-opponent` rebuilds the opponent at its last **revealed** board.
- It unfreezes the shop and installs a fresh shop tape.
- It starts a new turn (start-of-turn effects, 10 gold, tier-correct shop).
- The known opponent policy (zero memory) plays the turn, 8 samples per decision.
- Positioning then maximises equity against those boards. Nothing hidden is read; the clone's opponent state is
  overwritten before use.

Myopic plan changes (buy/sell for this battle) cost development and lost matches; positioning has no such cost.

**How much of the oracle's positioning gain the simulated opponent captures** (`--diag-oracle`, logging only;
about 117 decisions each):

| Opponent model | Share of available gain captured | Real gain per positioned battle |
|---|---|---|
| Blank memory, 8 samples (deployed) | 0.40 | +0.104 ±0.036 |
| Own memory | 0.30 | +0.077 ±0.032 |
| *True opponent memory (privileged)* | 0.28 | +0.070 ±0.034 |
| *Actual frozen items (privileged)* | 0.31 | +0.080 ±0.037 |
| 24 samples (partial) | 0.24 | +0.055 ±0.043 |

None beats the deployed model, so the remaining gap is mostly the opponent's unpredictable shop luck.
Other tuning, by share of the available gain captured:

| Variant | Captured | Real gain per decision |
|---|---|---|
| `--mix-prev` (also battle the opponent's previous board) | 0.46 | +0.125 |
| Heavier screen | 0.40 | +0.134 |
| No margin | 0.40 | +0.112 |
| Predictor-only positioning | 0.23 | **+0.029** |

Predictor-only positioning is too coarse for pet order, so labels need exact battles.

## How to run the winning agent

```
./run.sh NAME 6 10g match.py --mode planner --pairs 100 --workers 5 --namespace <fresh> \
  --position-only --sim-opponent [--mix-prev --screen-seeds 4 --exact-top 8 --sim-seeds 8] --out <file>.jsonl
python3 summarize.py 'out/<file>*.jsonl'
```

It costs about 30 s per game (about 2–4 s per end of turn) with 4–5 battle workers.

## Distillation: naive fine-tuning failed

The preliminary run used 223 seat-games: 4,762 forced moves, head-only training with KL anchoring, learning rate
3e-6, 2,000 steps.

- Held-out agreement with the planner's moves rose from 0.03 to 0.15.
- Agreement with the policy's own other decisions fell to 0.52.
- The plain fine-tuned network scored **0.325** [0.24, 0.43] over 40 pairs against 25772 (3 won, 20 split,
  17 lost).

Changing the shared decision heads to imitate reorders broke the rest of the policy.

**Next design:** a separate positioning module that takes over at END (input: our pre-END board, the opponent's
revealed boards and the turn; output: the order). It would be trained on the planner's labels or on real battle
outcomes, and it leaves buying and rolling untouched. 265 recorded seat-games (537 MB) are in `out/distill-v1`. To collect
more, rerun the same `match.py --both-seats --record` command; it resumes from its JSONL files.

### Earlier notes

- `match.py --both-seats --record out/distill-v1` stores every decision: observation, memory, executed action,
  the policy's own greedy action, and whether the planner forced it.
- `finetune.py` trains the policy on the planner's moves at end-of-turn states. A KL penalty to a frozen copy keeps
  everything else in place; by default only the decision heads are trained.
- Smoke-test caution: many decisions are near-ties. Even tiny updates flip the greedy choice, so agreement alone
  is not a quality measure; strength is checked with `match.py --mode raw --candidate out/<ckpt>.pt:<sha>`.

## Findings so far

- **Turn level** (against the opponent's actual board, diagnostics only): enumerated plans beat the
  policy-continuation baseline by about +0.1 to +0.2 battle equity on overridden turns.
- **Offline positioning headroom:** the best ordering of the policy's own boards scores +0.045 on fresh opponents.
- **Battle hangs.** Rare sapai battle loops hang workers; every simulation now goes through `safe_battle`
  (SIGALRM, 2 s).
- **Memory.** Library boards carry per-pet MT19937 states (39 KB per board). Strip them (`strip_rng`) or forked
  workers run out of memory.
