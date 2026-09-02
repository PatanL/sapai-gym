# Expert imitation and replay ingestion

## Why use it

Expert behavior cloning should primarily reduce the expensive early phase in
which self-play learns gold economy, reasonable purchases, food targets,
combining, positioning, and when to stop rolling. It does not replace league
self-play: the latter is still required for counters, robustness, and exceeding
the demonstrator.

A reasonable planning range, to be validated experimentally, is 3-10x fewer
steps to basic competence and 1.5-4x fewer steps to strong play. The eventual
self-play ceiling may move little; imitation changes initialization more than
the final objective.

## Replay source

For a provided replay participation ID:

```text
GET https://sap-library.vercel.app/api/search?pid=<PID>
GET https://sap-library.vercel.app/api/replays/<LIBRARY_ID>/perspectives
GET https://sap-library.vercel.app/api/replays/<LIBRARY_ID>/turns
```

`perspectives` supplies both players' raw action logs. `turns` supplies a
normalized view of team state, economy, rolls, lives, wins, battle outcomes,
and both perspectives.

Raw actions map naturally to the implemented autoregressive contract:

| Raw type | Training action |
|---:|---|
| 5 | Roll |
| 6 | Buy pet, optional destination/combine target |
| 7 | Combine |
| 8 | Buy food and target |
| 9 | Sell |
| 11 | Move/freeze state followed by end turn |

## Required reconstruction gate

The replay does not necessarily contain a complete shop snapshot immediately
before every action. Training examples must therefore be reconstructed by
replaying the raw requests/responses from the preceding validated state.

Only accept a trajectory when:

1. Item identities and unique object IDs resolve.
2. Every requested action is legal in the reconstructed state.
3. The reconstructed end-of-turn build matches the normalized replay.
4. Recorded board hashes match when available.
5. The resulting battle snapshot and outcome agree.

Ambiguous or mismatched actions are retained in a quarantine table for engine
debugging, never silently guessed into the imitation dataset.

The implemented client and reconstructor were exercised against a current
14-turn Turtle-vs-Turtle replay. Both perspectives produced 28 hash-validated
end-turn examples; 24 ambiguous identity references entered quarantine. The
dataset hydrator also rejected a pack-incompatible final-shop item instead of
crashing or allowing it into training.

## Dataset schema

Store compact semantic records, not rendered images or precomputed embeddings:

```json
{
  "schema_version": 1,
  "replay_id": "...",
  "perspective": "player",
  "turn": 5,
  "decision": 12,
  "state": {},
  "type_mask": [],
  "source_mask": [],
  "target_mask": [],
  "action": [2, 3, 1],
  "battle_outcome": 1,
  "final_result": 1,
  "validation": {
    "board_hash_match": true,
    "final_build_match": true
  }
}
```

The loader converts `state` through the same `EntityObservationEncoder` used
by self-play. This prevents a frozen training-data representation from drifting
away from simulator changes.

Split train/validation/test by complete replay ID, never by individual action.
Otherwise adjacent decisions from one game leak nearly identical states across
the split.

## Pretraining objective

Use teacher-forced sequence training with recurrent memory:

```text
L = CE(action type)
  + CE(source | type)
  + CE(target | type, source)
  + 0.25 * battle-outcome CE
  + 0.5 * return/value regression
```

All categorical losses retain the recorded legality masks. Report type,
source, target, full-action, and top-k accuracy separately. Full-action
accuracy is the useful headline; aggregate accuracy can hide a broken target
head.

The teacher-forced recurrent objective is implemented in
`sapai_gym1/rl/imitation.py` as `behavior_cloning_loss`. It accepts masked
`[time, batch]` entity sequences, expert action triples, optional battle labels,
and optional return targets. Public ingestion, conservative reconstruction,
quarantine output, semantic hydration, and replay-level train/validation splits
are implemented. Public replays currently validate complete end-turn states;
intermediate policy labels remain quarantined when the shop cannot be proven.

Exact action demonstrations can be recorded directly with:

```bash
python scripts/record_expert.py \
  --episodes 20 \
  --output replay-data/expert/human-expert.jsonl
```

`--auto --teacher random` retains the original weighted-random teacher only for
pipeline tests. `--auto --teacher guide` implements the supplied Turtle P1
guide as a legal-action policy prior: turn-band roll notes, pair/triple equity,
odd-turn discovers, board commitment, sell-pet economy, food targeting, and
positioning all contribute to action scores.

Guide demonstrations are grouped into complete shop-turn sequences. Recurrent
behavior cloning carries GRU memory through their true action order and pads
only between sequences; the earlier one-step loader did not actually train
decision memory. Dataset and checkpoints include a simulator revision, and
cross-revision direct observations are rejected.

The first guide baseline recorded 13,000 exact actions from 200 games with zero
mask violations and reached 46.1% held-out full hierarchical action accuracy,
versus 19.1% for the old weighted-random bootstrap. The fully audited v3
dataset is regenerated after all mechanics fixes rather than reusing those
now-stale labels. Human sessions remain the highest-quality source for
positions where the guide does not define a unique action.

## RL handoff

1. Load the behavior-cloning checkpoint into both main and inference replicas.
2. Place the frozen expert model in the initial league.
3. Begin current-v-current and current-v-expert V-trace self-play.
4. Run a low-learning-rate recurrent demonstration minibatch every fourth
   learner update through the early RL phase.
5. Decay the imitation coefficient based on validation/league performance,
   not only wall-clock steps.
6. Retain a low-probability expert anchor if self-play begins forgetting sound
   economy or positioning.

Include losses and recovery games, not only ten-win runs. Successful-only data
teaches a biased state distribution and omits precisely the pivots an expert
agent needs to learn.
