# Super Auto Pets Turtle Pack P1 Guide

Guide version: `turtle-p1-guide-v12-20260904`

This is the canonical working strategy guide for Turtle Pack P1 versus P1. It
is a decision document, not a dump of source data. The executable
`TurtleGuideTeacher` is an approximation of this guide and must never be
treated as more authoritative than this prose.

Battle authority: SAP-Calculator commit
`d165eb0a02f8aa0b54d72ed1d5490a44390d07f4` for battle ability semantics and
event ordering. SAP-Calculator is not authoritative for the shop phase, which
it does not model completely. Shop authority is verified current-game rules
and the expert-confirmed P1 mechanics recorded in this guide and regression
tests. Strategy authority is the expert knowledge in this document. First
evaluation mode may use arena-style board building; long-term decisions should
use the last observed 1v1 opponent board.

## Core decision loop

On every shop:

1. Update the persistent intended line; do not independently relabel the board
   after every action.
2. Record line confidence, committed pets, flexible slots, the intended sell
   slot, required hits, and known matchup counters.
3. Spend gold toward the highest-confidence line.
4. Preserve board space and a sell slot unless the board is solved.
5. Re-evaluate tempo, greed, and outs after every meaningful purchase, sell,
   combine, or roll.
6. Spend all gold before ending the turn.

Persistent line state should contain:

- intended line and confidence;
- pets that are committed because of stats, level, equipment, or composition;
- flexible pets and the preferred sell slot;
- pets, foods, equipment, levels, and positioning still required;
- favorable and unfavorable opponent interactions;
- whether the current action is tempo, scaling, conversion, or out-fishing.

The line may change, but it should require stronger evidence to abandon a
high-confidence plan than to choose a plan from an uncommitted board.

## P1 match rules

### Lives and the turn-2 restore

- Each player starts with 6 hearts.
- Each battle loss removes 1 heart.
- After the turn-2 battle, both players regain 1 heart, capped at 6.
- The battle result flag is not erased by that restore.
- A turn-2 loss at 6 hearts therefore ends at 6 hearts and activates loss
  effects such as turn-3 Snail.
- At 6 hearts on turn 2, deliberately losing can be correct when the future
  Snail/scaling value exceeds the small tempo value of winning.
- RL dense reward should follow net life change. A restored turn-2 loss should
  not receive an automatic negative life reward.

### Gold

- Gold does not carry over.
- A normal pet or food costs 3 unless explicitly discounted or generated.
- Roll costs 1.
- Sleeping Pill costs 1 and is always stocked.
- A pet sells for gold equal to its level, plus sell abilities and Cake value.
- Pig sells for 2/4/6 total at levels 1/2/3 before Cake additions.
- Always spend all gold before END_TURN.
- With 1 or 2 gold, rolling is weakly better than ending because it can reveal
  something worth freezing for the next turn. If it reveals nothing, the gold
  would have expired anyway.
- A hard action cap may force END_TURN as a safety valve, not as strategy.

### Combining and discovers

- Three total copies make level 2; six total copies make level 3.
- Combining merges permanent stats and all experience represented by both
  stacks.
- A normal level-up creates two distinct linked, normal-cost pets from one
  tier above the current shop tier.
- Buying either discover choice removes both choices.
- Freeze state is independent: freezing one discover choice does not freeze
  its sibling. On a roll, the unfrozen sibling disappears while the retained
  choice remains available.
- Each frozen discover occupies one normal pet-shop slot on later rolls, so
  freezing both choices exposes one fewer fresh pet than freezing only one.
- If both choices are frozen, their purchase linkage persists across rolls and
  turn refreshes; buying either one still removes its sibling.
- Combining two level-2 pets directly makes level 3 but gives no discover.
- When a purchased copy combines and levels a pet, its Buy ability resolves at
  the resulting combined level. A third Otter therefore uses level 2 and a
  third Cow stocks Better Milk.
- Hold combines for odd-turn tier access (3, 5, 7, 9) when practical, but
  combine earlier for space, survival, or a line-defining discover.

### Trigger and generated-item ordering

- Start-turn and end-turn shop abilities resolve from highest attack to lowest
  attack. Equal-attack ties are random and seeded.
- At the ten-slot cap, Worm and Pigeon stock by replacing the rightmost
  unfrozen entries first. If no eligible entry or free slot exists, the item is
  not generated.
- Cow replaces the two food-shop entries with its two free scaled Milks,
  including frozen food entries.
- Battle event priority follows SAP-Calculator: Hurt and Friend Hurt, own
  pre-removal Faint, faint observers such as Ox, Knock Out, post-removal
  summons/equipment, then Shark/Fly.
- Positive lethal damage still triggers Hurt before the pet faints. Camel,
  Blowfish, Peacock, and Gorilla therefore activate on a lethal hit.
- Verified live-game exception: for a post-removal summon Faint, a
  simultaneously dead intervening pet does not block Tiger.
  Deer-Wolverine-Tiger therefore repeats Deer when Deer and Wolverine die
  together. With Deer-Mammoth-Tiger, Mammoth's own Faint is Tiger-repeated and
  Deer also reaches Tiger after synchronized removal, so both abilities
  repeat.

## Global heuristics

### Roll note

Keep an explicit, narrow list of real hits:

- early: about 2-4 hits;
- midgame: about 4-6 hits;
- late: the best 4-7 cards, fewer when desperate.

Do not roll for everything and do not roll without knowing what improves the
board.

### Commitment and sell slots

- Turns 1-4: normally no more than three committed pets.
- Turns 5-6: normally no more than four.
- Turn 7+: usually four or five.
- Turn 2 should usually contain two unique tier-1 pets; three is acceptable;
  four distinct tier-1s sacrifices too much pair/triple equity.
- Keep two flexible slots early where possible.
- The first planned sell should not receive permanent stats or key equipment.

### Tempo and greed

Use the expert rule: lose by as much as possible; win by as little as
possible. If a round is already lost, do not ruin the future for fake tempo.
If it is already won, do not overpay to win harder. More lives permit greed;
low lives demand tempo. When the opponent is low, convert faster.
The executable teacher treats 2 opposing hearts as tempo pressure and 1 heart
as stronger pressure: immediate stats, equipment, Deer/Turtle/Skunk-style
tempo, and selling obsolete low-tier bodies rise in value, while Canned Food
and slow Cat/Dragon/Monkey/Penguin/Giraffe investments fall in value. This is
a soft preference, not a ban on a clearly winning scaling purchase.

### Freeze discipline

Freeze rolling is usually bad. Freeze when the card is one of the best hits,
the tier timing matters, or the remaining pool is weak. Do not freeze merely
decent cards while several better outs remain. A frozen shop should not block
triples, Turtle, Skunk, Deer, or core scaling pieces.

After spending the last gold, explicitly inspect the shop before ending. If a
visible pet would complete three or six total copies next turn, freezing it is
normally mandatory; FREEZE costs no gold and the exact frozen pet survives the
turn refresh. Skip the freeze only for a deliberate reason such as abandoning
that pet, preserving more valuable fresh-shop capacity, or avoiding a badly
timed combine. With 1-2 gold, freeze the hit first and then roll so the retained
copy is not lost.

When at least 3 gold remains and a desired pet can simply be bought, freezing
it first is normally a wasted action. Before turn 7, do not freeze-roll an
affordable pet merely to preserve it. Important exceptions are:

- delaying a real even-turn level-up so its discover occurs on the following
  odd turn;
- turns 5-6 Salad isolation, where keeping the team narrow while preserving a
  desired pet can protect the random buff targets;
- turn 7+ full-board plans where no acceptable sell slot exists yet, such as
  freezing Cow while rolling for a Turtle/Pill sequence that creates space.

These are strategy defaults, not mechanics restrictions. Exhaustive search
should continue auditing excluded affordable freezes so a real exception can
be discovered.

### Reposition timing

Ordinary swaps usually do not matter until the shop plan is complete. Defer
them until zero gold and set the final battle order then. Moving Snail toward
the back is reasonable, but it need not precede purchases and rolls.

Move earlier when board geometry affects an immediate Sleeping Pill action,
especially Turtle recipients, Flamingo buff recipients, or placing a fainting
pet directly ahead of Ox. Other Pill-sensitive Faint interactions such as
Badger and Hedgehog should likewise be positioned before the Pill is used.

### Separate copies versus half-combines

Buying a matching shop pet may either place it in an empty team position or
combine it onto an existing copy. On turn 2, separate copies are often better
when the extra body materially improves the battle and the combine neither
levels the pet nor solves board space. This is especially relevant after a
turn-1 win, when contesting turn 2 may be worth more than the small stats from
a half-combine. Hold separate copies for an odd-turn level-up when possible;
combine early only for space, concentrated buffs, or a real tempo breakpoint.

### Sleeping Pill discipline

Sleeping Pill is highly conditional. It is excellent when one gold converts a
useful Faint ability into permanent value—especially Flamingo stats or Turtle
Melon—and may also enable a deliberate summon or faint engine. It is severely
negative on an ordinary pet with no valuable Faint payoff. Score the target,
the recipients, available board space, and the resulting team together; do not
learn a generic preference either for buying or avoiding Pill.

## Persistent line catalog

### Sell-pet / triple economy

Best early pieces are Duck, Beaver, Pig, Pigeon, Fish, and Ant. Prefer pairs,
sell-pet triples, odd-turn discovers, and flexible space. Pigeon generates free
Bread Crumbs; Cake can turn a planned sell into additional economy. Convert
the line once its triples, shop buffs, or gold have been extracted.

### Big tier 1

Build around a naturally paired/tripled Ant, Cricket, Fish, Swan, or another
statted tier-1. Apples and Flamingo support concentrated stats. Preserve one
or two tech slots rather than filling the board with mediocre bodies. A level-3
friend also enables Bison.

### Horse-Cricket summon tempo

Horse plus Cricket is real early tempo, not a permanent identity by itself.
Continue through Spider, Sheep, Deer, Turkey, Shark, Rooster, or Fly when the
shops support it. Because this line prioritizes summons over permanent
scaling, do not apply the turns 3-4 scaling-Apple rule merely because a target
is available.

### Generic stat-tech

Concentrate stats on two or three keepers and preserve slots for Turtle,
Skunk, Deer, Scorpion, Crocodile, equipment, and matchup answers. Pear is the
clean default on turns 7-8. Equipment and composition often matter more than
another small late stat purchase.

### Elephant hurt engine

Elephant immediately ahead of Blowfish or Camel is a strong ordered engine.
Score the composition, not the pets independently:

- Elephant must be directly ahead of the hurt payoff.
- Blowfish converts each hurt into enemy damage.
- Camel converts hurt into stats for the next friend behind.
- Wolverine is a premium continuation after Elephant/hurt infrastructure
  exists.
- Deer may serve as a tempo/faint bridge without becoming the core identity.
- Protect the back payoff from Crocodile, Leopard, and opposing repeated shots.

### Shark / faint

Shark wants Cricket, Spider, Sheep, Turtle, Deer, Rooster, Fly, and other real
faint volume. Shark makes Turtle and Deer much more valuable and becomes a
game-winning level-2 line. Avoid consuming slots with summons that do not
produce enough combat value.

### Turkey summon

Turkey is a conversion point for Sheep, Deer, Whale, Rooster, Fly, and related
chains. Decide whether the current shop is buying immediate summon tempo or
committing to the full late engine. Turkey, Shark, and Fly can coexist when the
ordered summons and board slots justify them.

### Rabbit / Seal food and perk scaling; Cat stat food

Rabbit triggers when a friendly eats any shop food or genuinely gains a perk,
gives that pet +1 health per Rabbit level, and works three times per turn.
Apple, Milk, other stat foods, and perk foods all count. Eating a perk food is
one event, not both a food event and a second perk-gain event. Separate non-food
perk gains such as Scorpion gaining Peanut or Ox/Turtle granting Melon also
count. A generated unit entering with built-in equipment, such as Deer's Chili
Bus, did not gain that equipment and does not trigger Rabbit.

Seal follows the same event distinction for itself: it activates when Seal
eats any shop food or genuinely gains a perk, then gives up to three distinct
other friends +1 attack per Seal level. Cat is narrower: it multiplies only
the attack/health portion of stat food, works twice per turn, and does not use
a trigger on perk-only food. Once Cat is online, prioritize staying alive while
converting food density into permanent stats.

### Cow scaling conversion

When the goal is scaling or food scaling, Cow is near-mandatory if a sell slot
can be created.

- On a full board, first plan and execute the lowest-cost sell.
- Buy Cow and consume both free Milks.
- Put Milk on real keepers, not the planned sell slot.
- After the Milks are consumed, decide whether Cow is a keeper, pair/level
  candidate, Cat enabler, or cycle unit.
- Do not reject Cow solely because the board was full before planning the sell.

### Dragon cycling

Dragon with lives and open space can cycle tier-1 purchases for team-wide
scaling. Preserve a real cycle slot. When low, take immediate value and convert
to endgame tech instead of greed-looping forever.

### Tiger-Mammoth-Mushroom

Mammoth with Mushroom immediately ahead of Tiger is a premium completed line.
Mammoth faints, its ability fires and Tiger repeats it; Mushroom revives
Mammoth as 1/1, enabling another Mammoth ability and Tiger repeat when it
faints again.

Matchups:

- Stronger into Skunk: Skunk does not disable the faint engine and may
  accelerate Mammoth's first faint.
- Weaker into Rhino: Rhino can chain knockouts through the revived 1/1 Mammoth
  and small supports.
- Weaker into Elephant-Blowfish: repeated shots may remove Tiger/support before
  the chain completes.
- Crocodile punishes exposed backline Tiger.
- Strong Leopard can disrupt Tiger or support before full resolution,
  especially with weak Melon coverage.

Mammoth should be immediately ahead of Tiger. Mushroom belongs on Mammoth,
not on an unrelated body when this line is available.

### Scam / out-fishing

When normal scaling cannot catch up, narrow toward Skunk, Scorpion, Mushroom,
Steak, Chili, Crocodile, Leopard, Snake, Tiger, Fly, and matchup-specific outs.
These are real recovery tools, not filler. Buy only actions that materially
increase the chance to win the matchup.

## Opponent-conditioned rules

Use the last observed opponent board; never use private current-shop state.
When no opponent is known, use the favorable arena default.

- **Rhino versus Deer/summons:** Rhino makes extra Deer and small summon chains
  substantially worse because Bus/support knockouts can chain. Without known
  Rhino, multiple Deer remain a favorable default.
- **Skunk versus tall stats:** diversify or protect tall units; Skunk removes
  the value of pure-health concentration. Tiger-Mammoth-Mushroom is an
  exception because accelerated Mammoth faint can be favorable.
- **Elephant-Blowfish versus fragile engines:** protect or replace exposed
  Tiger, Turkey, Shark, and small supports that repeated shots can remove.
- **Crocodile versus backline Tiger:** move/protect Tiger or use a different
  back unit.
- **Leopard versus weak Melon coverage:** prioritize Melon/protection and avoid
  multiple fragile premium supports.
- **Scorpion versus tall fronts:** do not expose the only tall carry without a
  countermeasure; use summons, equipment, or sacrificial ordering.

## Deer rules

- Deer is one of the best purchases on turns 7 and 8.
- Additional Deer copies remain strong for tempo, levels, and repeated Buses.
- Default to multiple Deer when no Rhino is known.
- Substantially reduce extra-Deer and summon-heavy Deer value against known
  Rhino.
- Deer generally belongs front or mid-front so it faints on time.
- Deer is also a strong bridge into Shark, Turkey, Fly, and faint engines.

## Turns 1-2

Objective: build pair/triple equity without overcommitting.

Priority plans:

1. Sell-pet pairs/triples.
2. Big tier-1 preparation.
3. Horse-Cricket tempo.
4. Natural Otter/Fish health scaling when a real target already exists.

Freeze Apple aggressively when an early discover line exists, but freeze a
premium pair first if both would choke the shop. Usually hold a turn-2 combine
for a turn-3 tier-3 discover. On turn 2 at 6 hearts, consider spending gold on
future value and deliberately fielding a losing board so turn-3 Snail becomes
active at no net heart cost.

## Turns 3-4

Objective: identify a primary buff target and a secondary target while
retaining a flexible slot. Peacock, Swan, Flamingo, Spider, Crab, and Kangaroo
are common hits. Kangaroo is normally secondary, not the initial sole carry.

### Scaling Apple rule

On turns 3 and 4, when following a scaling line:

- do not apply this rule to Horse-Cricket summon tempo;
- do not apply it when Dog is the active summon-scaling plan;
- buy Apple for an ahead-of-curve tier-3 pet;
- otherwise buy Apple for Peacock or Kangaroo when the team has two copies;
- otherwise buy Apple for the committed big tier-1 pet;
- prefer these permanent targets over random temporary bodies.

Flamingo plus immediate Pill is often correct when it opens future food space.
Swan is a serious on-curve economy hit. Active Snail is premium on boards that
use attack well.

## Turns 5-6

Objective: convert the early shell into a real midgame board while preserving
isolated stats and tech slots.

Common plans:

- isolated board plus Salad;
- Giraffe scaling on clear targets;
- Rabbit food with genuine food/economy support;
- Elephant-Blowfish/Camel hurt engine;
- Turtle/Deer/Skunk universal value;
- Ox behind a naturally fainting unit.

Garlic is not automatic; buy it only when it changes fights. Parrot is valuable
when the copied ability has a real current and future ceiling.

## Turns 7-8

Objective: maintain two or three statted keepers plus flexible tech/conversion
slots.

Priority cards include Pear, Deer, Skunk, Turtle, Chocolate, Monkey, Shark,
Turkey, Cow, and important triples. Multiple Deer are favorable by default.
Against known Rhino, reduce Deer/summon commitment sharply. Cow is a priority
scaling conversion even on a full board when a sell slot can be created.

Buy Canned Food only when replacing several future slots, tempo is safe, and
the board is not already committed. Pear is the default clean stat purchase.

## Turns 9+

Stop buying filler. Narrow to Chocolate, equipment, Cow/Cat/Dragon conversion,
Skunk/Scorpion/Crocodile tech, and complete engines such as
Tiger-Mammoth-Mushroom or late summon chains.

Common equipment homes:

- Melon: main carries and essential protected engines;
- Steak: small tech units that need one impactful attack;
- Mushroom: Mammoth, Deer, Scorpion, and composition-specific faint pieces;
- Chili: high-health non-Melon front units.

Sell obsolete scalers sooner than inexperienced players expect. When behind,
fish for actual outs rather than incremental stats that cannot close the gap.

## Positioning defaults

- Biggest generic tempo unit in front.
- Kangaroo directly behind an attacker that will attack repeatedly.
- Ox directly behind a faintable front slot.
- Elephant immediately ahead of Blowfish or Camel.
- Snake behind friends that will actually attack several times.
- Turkey behind the summon chain.
- Shark in the back half, not front.
- Badger front or back, not arbitrary middle.
- Deer front or mid-front.
- Turtle directly ahead of its Melon target.
- Mammoth with Mushroom immediately ahead of Tiger.
- Protect backline Tiger from Crocodile and Leopard where possible.

Extra copies can have positional diminishing returns; for example, a second
Snake may not fire enough when too few friends remain ahead.

## Turtle Pack reference

All generated pets are gameplay tier 1, even though they are hidden and can
never be rolled or discovered directly. This includes Bee, Zombie Cricket,
Dirty Rat, Ram, Bus, Chick, and Zombie Fly. Tier-sensitive battle effects
therefore treat them as tier 1; in particular, Rhino deals doubled Knock Out
damage to them.

### Tier 1

- Ant 2/2: faint, give one random friend +1/+1 per level.
- Beaver 3/2: sell, give two random friends +1 attack per level.
- Cricket 1/3: faint, summon a 1/1, 2/2, or 3/3 Zombie Cricket.
- Duck 2/2: sell, give shop pets +1 health per level.
- Fish 2/3: level up, give two friends +1/+1 per level; level 3 has no ability.
- Horse 2/1: friend summoned, give +1 attack per level until next turn.
- Mosquito 2/2: start battle, deal 1 to 1/2/3 random enemies.
- Otter 1/4: buy, give 1/2/3 random friends +1 health.
- Pig 4/1: sell, gain +1/+2/+3 gold.
- Pigeon 3/2: sell, stock 1/2/3 free Bread Crumbs.

### Tier 2

- Crab 4/1: start battle, gain 25/50/75% of healthiest friend's health.
- Flamingo 3/2: faint, give two nearest friends behind +1/+1 per level.
- Hedgehog 4/2: faint, deal 2/4/6 to all pets.
- Kangaroo 2/2: friend ahead attacks, gain +1/+1 per level.
- Peacock 2/5: hurt, gain +3 attack per level.
- Rat 3/6: faint, summon 1/2/3 Dirty Rats for the opponent.
- Snail 2/3: end turn after a loss, give three nearest friends ahead +1 attack
  per level.
- Spider 2/2: faint, summon a tier-3 pet as 2/2, 4/4, or 6/6.
- Swan 1/2: start turn, gain +1/+2/+3 gold.
- Worm 1/4: start turn, stock a 2-gold Apple/Better Apple/Best Apple.

### Tier 3

- Badger 6/3: faint, deal 50/100/150% attack to adjacent pets.
- Camel 3/3: hurt, give nearest friend behind +1/+2 per level.
- Dodo 4/2: start battle, give 50/100/150% attack to friend ahead.
- Dog 3/2: friend summoned, gain +2/+1 per level until next turn.
- Dolphin 4/3: start battle, deal 4 to lowest-health enemy 1/2/3 times.
- Elephant 3/7: after attack, deal 1 to nearest friend behind 1/2/3 times.
- Giraffe 1/2: start turn, give nearest 1/2/3 friends ahead +1/+1.
- Ox 1/3: friend ahead faints, gain Melon and +1 attack 1/2/3 times.
- Rabbit 1/2: friendly eats any shop food or gains a perk, give that pet +1
  health per level, three times/turn; perk food is one event.
- Sheep 2/2: faint, summon two 2/2, 4/4, or 6/6 Rams.

### Tier 4

- Bison 4/4: end turn with level-3 friend, gain +2/+2 per level; one Bison.
- Blowfish 3/6: hurt, deal 3 damage per level to a random enemy.
- Deer 2/2: faint, summon 5/3, 10/6, or 15/9 Bus with Chili.
- Hippo 4/6: knockout, gain +3/+3 per level, up to three times.
- Parrot 4/2: copy nearest pet ahead at Parrot level until next turn.
- Penguin 2/3: start turn, buff two level-2+ friends +1/+1 per level.
- Skunk 3/5: start battle, remove 33/66/99% health from healthiest enemy.
- Squirrel 3/5: start turn, discount all food by 1/2/3.
- Turtle 2/5: faint, give Melon to nearest 1/2/3 friends behind.
- Whale 3/7: swallow nearest friend ahead and release at Whale level on faint.

### Tier 5

- Armadillo 4/8: start battle, give all pets +8 health per level.
- Cow 4/6: buy, replace food shop with two free scaled Milks.
- Crocodile 8/4: start battle, deal 8 to last enemy 1/2/3 times.
- Monkey 1/2: end turn, give front friend +2/+2 per level.
- Rhino 6/7: knockout, deal 4 per level to first enemy, doubled vs tier 1.
- Rooster 6/5: faint, summon 1/2/3 Chicks at half Rooster attack.
- Scorpion 1/3: summoned, gain Peanut.
- Seal 3/8: eats any shop food or gains a perk, give up to three distinct other
  friends +1 attack per level.
- Shark 2/2: friend faints, gain +2/+2 per level.
- Turkey 3/4: friend summoned, give +3/+1 per level.

### Tier 6

- Boar 10/6: before attack, gain +4/+2 per level.
- Cat 4/5: stat food gives 2x/3x/4x stats, twice per turn.
- Dragon 3/8: tier-1 friend bought, give other friends +1/+1 per level,
  four times per turn.
- Fly 4/4: friend faints, summon 4/4, 8/8, or 12/12 Zombie Fly, three times.
- Gorilla 7/10: hurt, gain Coconut 1/2/3 times.
- Leopard 10/4: start battle, deal 50% attack to 1/2/3 random enemies.
- Mammoth 4/12: faint, give friends +2/+2 per level.
- Snake 8/3: friend ahead attacks, deal 5 per level, five times.
- Tiger 6/4: friend ahead repeats battle ability at Tiger level.
- Wolverine 5/7: after four friends are hurt, remove 3 health per level from all
  enemies.

### Foods and equipment

- Tier 1: Apple +1/+1; Honey summons Bee; generated Bread Crumbs +1 attack;
  generated Better/Best Apple and Milk scale by source level.
- Tier 2: Sleeping Pill; Cupcake +3/+3 until next turn; Meat Bone +3 attack
  damage.
- Tier 3: Cake increases sell value at end turn; Salad buffs two random pets;
  Garlic reduces damage by 2 with minimum 2.
- Tier 4: Pear +2/+2; Canned Food +1/+1 to current/future shop pets; Bread
  gives temporary +7 health.
- Tier 5: Chili splashes second enemy for 5; Chocolate gives +1 experience to
  a pet below level 3 and cannot target a level-3 pet; Sushi
  buffs three random pets; Scorpion generates Peanut.
- Tier 6: Melon blocks 20; Mushroom revives as 1/1; Steak adds 20 attack damage
  once; Pizza buffs two random pets +2/+2; Coconut blocks one hit.

## Candidate-action tie breakers

Buy a pet when it is a required hit, materially improves the next fight,
preserves/upgrades the intended line, creates gold/space efficiency, or is a
premium matchup tool. Skip it when it is merely fine, consumes required
flexibility, or delays a stronger line.

Combine early for immediate survival, space, or a line-defining discover. Hold
for odd-turn tier access when greed is affordable. Freeze only true hits. Sell
the planned flexible pet before rejecting a near-mandatory conversion such as
Cow.

The executable teacher should expose its line, confidence, commitments,
flexible slots, required hits, and counters in every recorded decision so
changes can be audited against this guide.
