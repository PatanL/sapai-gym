# Entity-transformer self-play architecture

## Status

This document describes the implemented architecture checkpoint. It is no
longer only a proposal: the action schema, entity encoder, recurrent model,
explicit two-player environment, batched self-play collector, V-trace loss,
league state, historical-opponent path, checkpointing, and CUDA update path
are present and covered by smoke tests.

The v19 production model has **14,346,156 parameters** with the current Turtle
vocabulary. V19 is initialized from v18 with a zero-gated opponent-history
adapter, so its initial policy and value behavior are unchanged.

## Data flow

```text
independent randomized self-play games
             │
             ├── player 0 information set
             └── player 1 information set
             │
             ▼
batched entity-transformer inference (BF16 on GB10)
             │
             ▼
explicit type -> source -> target actions
             │
             ▼
zero-sum turn segments + recurrent states
             │
             ▼
V-trace learner update
             │
             ├── current-v-current self-play
             └── current-v-frozen-historical self-play
```

`ParallelSelfPlayEnv` exposes both players. It does not hide a random opponent
inside the learning player's `step()`. Both policies receive their own legal
information set, choose independently, and contribute trajectories.

## Entity observation

Every observation contains 21 fixed positional tokens:

- 1 global game token
- 5 physical team slots
- 10 physical shop slots
- 5 previously revealed opponent slots

V19 additionally carries a two-board public opponent-history tensor (two by
five slots). It records identity, ordering, total visible stats, level,
experience, perk, and the public booleans indicating temporary attack or
health. It deliberately does not reveal the hidden amount of a temporary stat
buff. A small 128-wide history transformer summarizes these boards, and a
zero-initialized adapter injects that belief into the existing 384-wide policy
context. This sidecar leaves transferred v18 behavior bit-for-bit neutral until
the adapter learns a nonzero weight.

Each token contains categorical embeddings for item, entity kind, perk/status,
and position, plus normalized numeric features including attack, health, tier,
level, experience, cost, frozen/occupied flags, gold, turn, lives, wins, sell
value, action-cap progress, reorder progress, and freeze/order history.

The observation also carries the complete conditional legality tensors:

- `type_mask[9]`
- `source_mask[9, 11]`
- `target_mask[9, 11, 6]`

These masks are derived from the simulator's current legal flat actions, so the
autoregressive representation is a lossless view of the environment contract.
V18 also projects the flattened masks into the global entity before transformer
attention. This makes exact toggle/reorder history visible to the critic and
GRU instead of using legality only to suppress output logits. The projection is
zero-initialized for behavior-preserving pre-v18 checkpoint transfer.

## Action contract

| Type | Source | Target |
|---|---|---|
| End turn | none | none |
| Roll | none | none |
| Buy pet | shop slot | empty slot or matching pet |
| Buy food | shop slot | team pet or none for random/team food |
| Sell | team slot | none |
| Combine | source team pet | matching destination pet |
| Freeze | shop slot | none |
| Unfreeze | shop slot | none |
| Move | source team slot | destination team slot |

`FlatActionCodec` supplies a 197-action compatibility bridge for Gymnasium.
The transformer never learns a single 197-class output: it predicts the nine
types and conditional source/target pointers.

## Model

Default production configuration:

- model width: 384
- transformer layers: 6
- attention heads: 8
- feed-forward width: 1,536
- recurrent memory: 512
- dropout: 0 during the initial RL baseline
- BF16 CUDA inference/training

The transformer contextualizes all entities. Its encoder blocks already use
residual connections. A GRU carries information across
shop decisions and turns. The decoder first samples the action type, then uses
entity pointer logits for the source and target. The model also emits a scalar
value, three-way battle-outcome logits, and a board-strength auxiliary output.
V19 adds auxiliary heads that predict the opponent's next five-slot board:
pet identity, perk, level, visible attack/health, and temporary-stat flags.
These targets come for free from the next submitted battle board and train the
history representation before its policy contribution becomes large.
For the transfer trial, inherited v18 parameters remain at `2e-6`, the new
history encoder/prediction heads use `1e-4`, and the zero-gated adapter uses
`1e-5`. Checkpoints preserve the three optimizer groups and their rates across
resume.

The GRU is optional at inference. Passing no memory is exactly equivalent to a
zero GRU state, which supports chess-engine-style queries from a fully supplied
position. Carrying memory across recommendations may improve multi-action plan
continuity, but it is not required for a valid action. A complete arbitrary
mid-shop query must include the current public counters and exact legal-action
masks as well as turn/team/shop/history; those determine freeze, reorder, and
action-cap legality.

## Learning and self-play

The implemented learner uses IMPALA-style V-trace targets. Behavior log
probabilities travel with each turn segment, allowing the main policy to learn
from main-policy behavior with at most one update of lag. Frozen-resident sides
are excluded because their GRU states are incompatible with the main network.
Variable-length shop turns are evaluated as a padded transformer batch, while
each V-trace recursion uses its correct immediate bootstrap state. Discount is
1.0 between shop actions, 0.99 only across a battle/turn boundary, and 0.0 at
terminal matches. Each segment's exact win/loss/draw also trains the existing
battle auxiliary head.

The single-device runtime supports:

- batched latest-v-latest symmetric self-play;
- a frozen historical policy on player 1;
- recurrent state per environment and player;
- independent random streams per game/player;
- checkpoint and league metadata persistence;
- PFSP-style opponent sampling in `LeagueManager`.
- 32 spawned CPU actor processes with three games per actor in the prepared
  v18 campaign;
- shared-memory observation/action/recurrent-state buffers;
- a background dynamic GPU inference batcher;
- a bounded preallocated shared-memory trajectory ring (queues carry only slot IDs);
- a separately versioned actor-inference replica, allowing simulation and
  inference to continue while the learner updates;
- a persistent expert anchor plus up to four resident historical/exploiter
  policies with a per-game shared sampling router;
- length-bucketed recurrent learner batches and fixed transition-count updates.
- optional two-games-per-actor operation for future simulator backends.
- dominated freeze/unfreeze cycles pruned per item between rolls;
- recurrent guide minibatches during the early self-play phase;
- simulator-revision gates on imitation, initialization, and resume.

The training entrypoint allocates half of games to current-v-current play. The
other half samples an expert anchor or one of the four newest frozen
checkpoints for player 1 at the start of each game. Actor matchup results are returned to `LeagueManager`; empirical
win rates drive squared PFSP priorities with a nonzero sampling floor. Resume
restores the serialized league and reloads its newest resident policies.
Only completed-match outcomes update PFSP; individual round results remain RL
rewards but no longer distort opponent win-rate estimates.

## Run

From the DGX PyTorch container:

```bash
cd /workspace/sapai-gym
bash scripts/run_v19_opponent_transfer_v18_5000.sh
```

Fast functional test:

```bash
python scripts/train_entity_selfplay.py \
  --smoke \
  --updates 2 \
  --checkpoint-dir /tmp/entity-smoke
```

## Verification gates

```bash
python tests/test_current_env.py
python tests/test_rl_architecture.py
```

Coverage includes action-codec round trips, lossless masks, 100-step entity
rollouts, model sampling and teacher evaluation, backward propagation,
production parameter bounds, analytic V-trace returns, league persistence and
sampling, explicit zero-sum self-play, random parallel self-play, a batched
self-play collection, and a real optimizer update.

## Measured DGX Spark performance

Short production-model profiles on this DGX Spark:

| Runtime | Throughput | Notes |
|---|---:|---|
| Original single-process collector | ~438 transitions/s | serialized collection/learning |
| Shared actors, 256 turn segments | up to ~1,156 transitions/s | variable transition count |
| Fixed 8,192-transition updates | ~825 overall, ~1,130 steady-state transitions/s | includes startup/checkpoint overhead |
| Two games per actor, fixed updates | ~719 overall, ~863 steady-state transitions/s | failed the +20% gate; not the default |

The dynamic batcher averages roughly 15-28 active decisions per inference
launch. Observed GB10 SM utilization peaks at 92% during learner batches and
power peaks around 40 W; utilization is intentionally less than 100% between
bursts because the workload alternates irregular symbolic simulation and dense
learning. The inference replica and shared trajectory ring now overlap these
phases.
The v17 campaign sustained roughly 1,065 transitions/s. V18 additionally
caches exact legal actions and disables unused full action-history
serialization in RL environments. A controlled 20,000-decision CPU profile
improved from 1,042 to 1,494 environment actions/s (+43.3%); its end-to-end GPU
training impact remains to be measured once the accelerator is available.
V19 adds about 303k parameters (2.2%) and ten small history tokens. Deployment
still uses one policy process and the same autoregressive action loop; callers
only retain one additional opponent board and two public temporary-stat flags
per pet.

The shared ring is sized for 64 actions per turn and has a regression covering
the environment's maximum valid 51-action turn (50 shop actions plus forced
end turn).

## Current training handoff

The frozen v18 campaign runs through update 5000. A host-side handoff service
then starts a frozen v19 transfer branch for 1000 updates. Its paired-seat gate
compares v19 directly with v18 update 5000 under the same v19 runtime. Scores at
or above 0.52 promote v19; scores at or below 0.48 return to frozen v18; the
uncertain interval receives one additional 1000-update v19 trial before a final
head-to-head decision. This avoids treating evaluation noise as architecture
evidence while keeping most compute in training.

Replay ingestion, conservative reconstruction, recurrent behavior cloning,
empirical PFSP feedback, and imitation-initialized profiling are implemented.
The first serious league run is documented in `runs/turtle-league-v1/RUN.md`.
Human demonstrations can replace the scripted bootstrap without changing the
action or entity contracts established here. Value-guided shop-sequence search
is a possible later addition once the learned value is calibrated.
