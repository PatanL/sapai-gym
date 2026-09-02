# Turtle P1 shop interaction audit — v17

Historical v17 audit. Current follow-up work is recorded in
`CORRECTNESS_AUDIT_V18.md`; do not use this file's training-impact section as a
live process status.

Audit date: 2026-08-29

Simulator revision: `turtle-current-shop-v17-20260829`

Guide revision: `turtle-p1-guide-v8-20260829`

## Authority

- Expert-confirmed live rules are authoritative for shop mechanics.
- SAP-Calculator is used for battle ordering and as supporting metadata only.
- Replay-bot current metadata is supporting evidence, but can lag live shop
  text (Rabbit was a confirmed example).

## Confirmed corrections

### Rabbit and Seal food/perk triggers

Rabbit triggers when a friendly eats any shop food or genuinely gains a
positive perk. It works three times per turn and gives the triggering pet +1
health per Rabbit level. Seal observes the same two event classes for itself,
then gives up to three distinct other friends +1 attack per Seal level.

Verified shop cases:

- Buying Scorpion grants Peanut, then Rabbit buffs Scorpion.
- Apple and every other stat food trigger Rabbit/Seal when eaten.
- Perk food triggers Rabbit/Seal exactly once for eating; assigning the perk
  carried by that food is not a second trigger.
- Pilling a friend ahead of Ox makes Ox gain Melon, then Rabbit buffs Ox.
- Pilling Turtle grants Melon before Ox's faint observer resolves. Rabbit sees
  both distinct Melon gains.
- Turtle skips friends that already have Melon and targets the next eligible
  friend behind.
- Deer's Bus enters with built-in Chili and therefore does not notify Rabbit.
- Scorpion genuinely gains Peanut when summoned, so buying or respawning one
  does notify Rabbit/Seal as applicable.
- Cat remains stat-food-only and does not consume either use on perk-only food.

### Stable shop Hurt/Faint resolution

Sleeping Pill and cascading shop damage now reuse the stable battle event
resolver on the actual shop team. Physical slot movement can no longer replace
or skip queued pets.

Verified cases:

- Cricket, Spider, Sheep, Deer, and Rooster summons persist after Pill.
- Shark observes the original faint exactly once for each friend.
- Ox observes its original friend ahead even when summons move slots.
- Lethal Camel Hurt resolves before Camel faints.
- `Camel → 6/3 Badger → Peacock → Shark`: Pill Badger leaves Peacock 6/4
  and level-one Shark 6/6 from two friend faints.
- Honey and Mushroom post-removal summons retain ownership and trigger summon
  listeners.
- A Sheep pilled from a full five-pet team summons the one Ram that fits and
  discards the overflow Ram.
- When a Honey-equipped Spider is pilled with room for both summons, the Bee
  is physically ahead of Spider's tier-three summon.

### Authoritative shop-action RNG

All random shop effects now consume the owning Shop RNG rather than advancing
private per-object copies. This covers random food targets and random buy,
sell, level-up, and food-trigger abilities. Serialization preserves the
advanced stream and reproduces future rolls.

### Exact Shop restoration

Restoring a non-fixed shop no longer derives base capacity from the currently
surviving slots or automatically rolls purchased-away slots. Base pet/food
capacity comes from the turn rules, while the serialized incomplete shop and
RNG remain exact until the next real roll/start-turn action.

## Previously verified mechanics

- One authoritative sequential RNG for rolls and discovers.
- Frozen slots consume no roll draw and cannot rewind RNG.
- Tier-two shops contain three pets and one food.
- Ten physical shop slots and atomic level-up reward overflow.
- Two distinct linked next-tier discover choices; purchase removes the group.
- Discover choices freeze independently and retain linkage across rolls.
- Start/end-turn abilities use attack priority with seeded random ties.
- Buy-combine uses the resulting level for Otter and Cow Buy abilities.
- Correct stack experience/stat merging; direct level-two merge gives no
  discover.
- Stable sell listener snapshots for Duck, Beaver, Pig, and Pigeon.
- Worm/Pigeon generated-food capacity and rightmost-unfrozen eviction.
- Cow produces two free level-scaled Milks.
- Canned Food buffs current and future shop pets and preserves ownership.
- Cat uses only stat food and works twice per turn.
- Seal targets up to three distinct other friends.
- Dragon works four times per turn and excludes the bought tier-one pet.
- Cake accumulates +1 sell value each end turn; accumulated value survives
  replacing Cake. Pig receives normal sell value, Cake value, and Pig bonus.
- Bread grants temporary +7 health at end turn and expires at start turn.
- END_TURN normally requires zero gold; the 50-action escape hatch remains.
- Turn-two heart restoration preserves the loss flag and uses net life reward.

## Explicit unresolved live-rule questions

These are not silently guessed:

1. Does Rabbit/Seal trigger from perk gains during battle, or only during shops?
2. Does reapplying the same perk (for example Ox gaining Melon while already
   Meloned) count as another Rabbit trigger? v17 currently counts it.
3. When Cow is bought with extra Pigeon/Worm-stocked foods present, does it
   replace only the two base food slots or every food entry? Current code
   replaces every food entry.
4. When two Cake-equipped copies with accumulated sell bonuses combine, are
   bonuses added or is only the larger bonus retained? Current code retains
   the larger bonus.
5. Does pilling Rat in the shop create any persistent opponent Dirty Rat
   effect? Current shop resolution discards the temporary enemy-shop team.
6. Exact physical insertion position for Worm/Pigeon food remains unverified;
   current code appends right while preserving capacity/order of survivors.
7. When level-up overflow has some free capacity but not enough unfrozen
   entries for both choices, current code conservatively discards the atomic
   pair.
8. Trigger priority for simultaneous non-start/end shop observers has not been
   confirmed independently. v17 uses attack priority for Rabbit and stable
   battle priorities for Hurt/Faint; legacy additive buy/summon listeners
   remain team-order where their effects commute.

## Training impact

- V16 and v17 training are stopped. V17 update 1,300 is the selected transfer
  weight, not a revision-current promotion checkpoint.
- V18 must use a fresh directory and reset optimizer/update/transition/league
  state. See `CORRECTNESS_AUDIT_V18.md`.
