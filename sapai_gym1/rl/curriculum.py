"""Versioned, targeted Turtle P1 scenario demonstrations.

These examples shorten exploration paths to mechanics and compositions that
ordinary self-play reaches too rarely. They do not alter simulator rewards or
make actions illegal: every label is checked against the exact environment
action mask before it is emitted.
"""

from __future__ import annotations

from collections import Counter
import json

import numpy as np

from sapai import Food, Pet, Player, Shop, Team
from sapai.turtle_pack import SIMULATOR_REVISION, TURTLE_PET_STATS
from sapai_gym1 import SuperAutoPetsEnv

from .action_schema import (
    ActionType,
    AutoregressiveAction,
    FlatActionCodec,
    NO_SOURCE,
    NO_TARGET,
)
from .entity_observation import EntityObservationEncoder
from .guide_teacher import GUIDE_SHA256, GUIDE_VERSION, TEACHER_SHA256


CURRICULUM_VERSION = "turtle-p1-scenarios-v3-20260902"
SCENARIO_NAMES = (
    "triple-freeze",
    "avoid-freeze-level-three",
    "avoid-freeze-obsolete-pair",
    "turn-two-separate-pair",
    "pill-flamingo",
    "pill-turtle",
    "avoid-pill-no-faint",
    "shark-fly",
    "mushroom-deer-shark",
    "elephant-hurt-wolverine",
    "scorpion-vs-tall",
)

FILLER_PETS = (
    "pet-ant",
    "pet-beaver",
    "pet-cricket",
    "pet-fish",
    "pet-flamingo",
    "pet-snail",
    "pet-giraffe",
    "pet-penguin",
    "pet-skunk",
    "pet-turtle",
)


def _pet(name, rng, *, copies=1, tall=False):
    pet = Pet(name)
    pet.set_total_experience(int(copies) - 1)
    if tall:
        pet._attack = int(rng.integers(35, 51))
        pet._health = int(rng.integers(35, 51))
    else:
        pet._attack += int(rng.integers(0, 5))
        pet._health += int(rng.integers(0, 5))
    return pet


def _team(specs, rng):
    team = Team()
    for index, spec in enumerate(specs[:5]):
        if spec is None:
            continue
        if isinstance(spec, str):
            team[index] = _pet(spec, rng)
        else:
            name, options = spec
            team[index] = _pet(name, rng, **options)
    return team


def _make_env(*, turn, gold, team, shop, opponent, rng):
    shop_objects = [
        Food(name) if name.startswith("food-") else Pet(name)
        for name in shop
    ]
    player_shop = Shop(shop_objects, turn=turn, fixed_rules=True)
    player = Player(
        shop=player_shop,
        team=_team(team, rng),
        lives=6,
        gold=gold,
        turn=turn,
        wins=min(9, max(0, turn - 3)),
        record_actions=False,
    )
    opponent_team = _team(opponent, rng)
    opponent_player = Player(
        team=Team.from_state(opponent_team.state),
        lives=6,
        gold=0,
        turn=turn,
        wins=min(9, max(0, turn - 3)),
        record_actions=False,
    )
    env = SuperAutoPetsEnv(
        opponent_generator=None,
        valid_actions_only=True,
        manual_battles=True,
    )
    env.agent = player
    env.shop_agent = player_shop
    env.team_agent = player.team
    env.opponent = opponent_player
    env.turn = turn
    env.previous_opponent_state = Team.from_state(opponent_team.state)
    env.previous_previous_opponent_state = Team.from_state(opponent_team.state)
    env.shop_action_count_agent = 0
    env.reorder_count_agent = 0
    env._freeze_toggled_agent.clear()
    env._seen_team_orders_agent = {env._team_order_key(player)}
    env._agent_actions_cache = None
    return env


def _serialize_observation(observation):
    return {key: value.tolist() for key, value in observation.items()}


def _emit_sequence(env, actions, *, scenario, sequence_id, encoder):
    examples = []
    for decision, action in enumerate(actions):
        observation = encoder.encode(env, 0)
        encoded = FlatActionCodec.encode(action)
        if encoded not in env._avail_actions(0):
            raise RuntimeError(
                f"Curriculum emitted illegal action {action} for {scenario}"
            )
        examples.append(
            {
                "schema_version": 1,
                "replay_id": f"curriculum-{scenario}-{sequence_id:06d}",
                "perspective": "curriculum",
                "turn": int(env.agent.turn),
                "decision": decision,
                "observation": _serialize_observation(observation),
                "action": list(action.as_tuple()),
                "battle_outcome": None,
                "final_result": None,
                "validation": {
                    "exactSimulatorState": True,
                    "simulatorRevision": SIMULATOR_REVISION,
                    "curriculumVersion": CURRICULUM_VERSION,
                    "guideVersion": GUIDE_VERSION,
                    "guideSha256": GUIDE_SHA256,
                    "executableTeacherSha256": TEACHER_SHA256,
                },
                "teacher": {
                    "name": "targeted-scenario-curriculum",
                    "version": CURRICULUM_VERSION,
                    "scenario": scenario,
                },
            }
        )
        env.resolve_action(0, encoded)
    return examples


def _triple_freeze(rng):
    turn = int(rng.integers(2, 11))
    eligible = [
        name
        for name, (_, _, tier) in TURTLE_PET_STATS.items()
        if tier <= min(6, (turn + 1) // 2)
    ]
    target = str(rng.choice(eligible))
    copies = int(rng.choice((2, 5)))
    gold = int(rng.integers(0, 3))
    fillers = list(rng.choice(FILLER_PETS, size=3, replace=False))
    team = [(target, {"copies": copies}), *fillers, None]
    rng.shuffle(team)
    distractors = list(rng.choice(FILLER_PETS, size=3, replace=False))
    shop = [target, *distractors, "food-apple"]
    rng.shuffle(shop)
    env = _make_env(
        turn=turn,
        gold=gold,
        team=team,
        shop=shop,
        opponent=["pet-fish", "pet-peacock", None, None, None],
        rng=rng,
    )
    source = next(
        index for index, slot in enumerate(env.agent.shop)
        if slot.obj.name == target
    )
    actions = [AutoregressiveAction(ActionType.FREEZE, source, NO_TARGET)]
    actions.extend(
        AutoregressiveAction(ActionType.ROLL, NO_SOURCE, NO_TARGET)
        for _ in range(gold)
    )
    actions.append(
        AutoregressiveAction(ActionType.END_TURN, NO_SOURCE, NO_TARGET)
    )
    return env, actions


def _avoid_freeze_level_three(rng):
    target = str(rng.choice(("pet-ant", "pet-fish", "pet-beaver")))
    env = _make_env(
        turn=int(rng.integers(7, 12)),
        gold=0,
        team=[(target, {"copies": 6}), "pet-deer", "pet-skunk", None, None],
        shop=[target, "pet-penguin", "pet-parrot", "food-pear"],
        opponent=["pet-deer", "pet-shark", "pet-turkey", None, None],
        rng=rng,
    )
    return env, [
        AutoregressiveAction(ActionType.END_TURN, NO_SOURCE, NO_TARGET)
    ]


def _avoid_freeze_obsolete_pair(rng):
    target = str(rng.choice(("pet-ant", "pet-beaver", "pet-horse")))
    env = _make_env(
        turn=int(rng.integers(9, 14)),
        gold=0,
        team=[target, "pet-deer", "pet-skunk", "pet-monkey", "pet-crocodile"],
        shop=[target, "pet-dragon", "pet-snake", "pet-tiger", "food-melon"],
        opponent=[("pet-hippo", {"tall": True}), "pet-skunk", "pet-monkey", None, None],
        rng=rng,
    )
    return env, [
        AutoregressiveAction(ActionType.END_TURN, NO_SOURCE, NO_TARGET)
    ]


def _turn_two_separate_pair(rng):
    """Preserve a second battle body instead of reflexively half-combining."""

    pair = str(rng.choice(("pet-ant", "pet-beaver", "pet-cricket")))
    env = _make_env(
        turn=2,
        gold=3,
        team=[pair, None, "pet-fish", None, None],
        shop=[pair, "pet-pig", "pet-mosquito", "food-apple"],
        opponent=["pet-ant", "pet-fish", "pet-horse", None, None],
        rng=rng,
    )
    env.agent.lf_winner = True
    env.agent.last_battle_outcome = 1
    return env, [
        AutoregressiveAction(ActionType.BUY_PET, 0, 1),
        AutoregressiveAction(ActionType.END_TURN, NO_SOURCE, NO_TARGET),
    ]


def _pill_flamingo(rng):
    env = _make_env(
        turn=int(rng.integers(3, 7)),
        gold=1,
        team=["pet-flamingo", "pet-peacock", "pet-kangaroo", None, None],
        shop=[
            "pet-swan",
            "pet-crab",
            "pet-spider",
            "food-sleeping-pill",
        ],
        opponent=["pet-fish", "pet-ant", "pet-beaver", None, None],
        rng=rng,
    )
    return env, [
        AutoregressiveAction(ActionType.BUY_FOOD, 3, 0),
        AutoregressiveAction(ActionType.END_TURN, NO_SOURCE, NO_TARGET),
    ]


def _pill_turtle(rng):
    env = _make_env(
        turn=int(rng.integers(7, 11)),
        gold=1,
        team=["pet-turtle", "pet-peacock", "pet-skunk", None, None],
        shop=[
            "pet-deer",
            "pet-penguin",
            "pet-parrot",
            "food-sleeping-pill",
        ],
        opponent=[("pet-hippo", {"tall": True}), "pet-monkey", None, None, None],
        rng=rng,
    )
    return env, [
        AutoregressiveAction(ActionType.BUY_FOOD, 3, 0),
        AutoregressiveAction(ActionType.END_TURN, NO_SOURCE, NO_TARGET),
    ]


def _avoid_pill_no_faint(rng):
    """Retain the dominant negative case: do not Pill an ordinary pet."""

    env = _make_env(
        turn=int(rng.integers(3, 9)),
        gold=1,
        team=["pet-fish", "pet-peacock", "pet-swan", None, None],
        shop=[
            "pet-rat",
            "pet-kangaroo",
            "pet-crab",
            "food-sleeping-pill",
        ],
        opponent=["pet-fish", "pet-ant", "pet-beaver", None, None],
        rng=rng,
    )
    return env, [
        AutoregressiveAction(ActionType.ROLL, NO_SOURCE, NO_TARGET),
        AutoregressiveAction(ActionType.END_TURN, NO_SOURCE, NO_TARGET),
    ]


def _shark_fly(rng):
    env = _make_env(
        turn=int(rng.integers(11, 14)),
        gold=3,
        team=["pet-deer", "pet-turtle", "pet-rooster", "pet-shark", None],
        shop=["pet-fly", "pet-cat", "pet-leopard", "food-pizza"],
        opponent=[("pet-hippo", {"tall": True}), "pet-skunk", None, None, None],
        rng=rng,
    )
    return env, [
        AutoregressiveAction(ActionType.BUY_PET, 0, 4),
        AutoregressiveAction(ActionType.END_TURN, NO_SOURCE, NO_TARGET),
    ]


def _mushroom_deer_shark(rng):
    env = _make_env(
        turn=int(rng.integers(11, 14)),
        gold=3,
        team=["pet-deer", "pet-turtle", "pet-rooster", None, "pet-shark"],
        shop=["pet-monkey", "pet-skunk", "food-mushroom", "food-steak"],
        opponent=[("pet-hippo", {"tall": True}), "pet-crocodile", None, None, None],
        rng=rng,
    )
    return env, [
        AutoregressiveAction(ActionType.BUY_FOOD, 2, 0),
        AutoregressiveAction(ActionType.END_TURN, NO_SOURCE, NO_TARGET),
    ]


def _elephant_hurt_wolverine(rng):
    hurt_target = str(rng.choice(("pet-blowfish", "pet-camel")))
    env = _make_env(
        turn=int(rng.integers(11, 14)),
        gold=3,
        team=["pet-elephant", hurt_target, "pet-deer", "pet-skunk", None],
        shop=["pet-wolverine", "pet-dragon", "pet-leopard", "food-melon"],
        opponent=["pet-deer", "pet-rooster", "pet-turkey", None, None],
        rng=rng,
    )
    return env, [
        AutoregressiveAction(ActionType.BUY_PET, 0, 4),
        AutoregressiveAction(ActionType.END_TURN, NO_SOURCE, NO_TARGET),
    ]


def _scorpion_vs_tall(rng):
    env = _make_env(
        turn=int(rng.integers(9, 14)),
        gold=2,
        team=["pet-ant", "pet-snail", "pet-giraffe", "pet-penguin", "pet-skunk"],
        shop=["pet-scorpion", "pet-monkey", "pet-crocodile", "food-pear"],
        opponent=[
            ("pet-hippo", {"tall": True}),
            ("pet-bison", {"tall": True}),
            "pet-monkey",
            None,
            None,
        ],
        rng=rng,
    )
    return env, [
        AutoregressiveAction(ActionType.SELL, 0, NO_TARGET),
        AutoregressiveAction(ActionType.BUY_PET, 0, 0),
        AutoregressiveAction(ActionType.END_TURN, NO_SOURCE, NO_TARGET),
    ]


BUILDERS = {
    "triple-freeze": _triple_freeze,
    "avoid-freeze-level-three": _avoid_freeze_level_three,
    "avoid-freeze-obsolete-pair": _avoid_freeze_obsolete_pair,
    "turn-two-separate-pair": _turn_two_separate_pair,
    "pill-flamingo": _pill_flamingo,
    "pill-turtle": _pill_turtle,
    "avoid-pill-no-faint": _avoid_pill_no_faint,
    "shark-fly": _shark_fly,
    "mushroom-deer-shark": _mushroom_deer_shark,
    "elephant-hurt-wolverine": _elephant_hurt_wolverine,
    "scorpion-vs-tall": _scorpion_vs_tall,
}


def generate_curriculum_examples(*, examples_per_scenario=128, seed=20260901):
    if examples_per_scenario <= 0:
        raise ValueError("examples_per_scenario must be positive")
    rng = np.random.default_rng(seed)
    encoder = EntityObservationEncoder()
    examples = []
    sequence_id = 0
    for scenario in SCENARIO_NAMES:
        for _ in range(examples_per_scenario):
            env, actions = BUILDERS[scenario](rng)
            examples.extend(
                _emit_sequence(
                    env,
                    actions,
                    scenario=scenario,
                    sequence_id=sequence_id,
                    encoder=encoder,
                )
            )
            sequence_id += 1
    return examples


def scenario_counts(examples):
    return Counter(example["teacher"]["scenario"] for example in examples)


def write_curriculum_jsonl(path, examples):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as output:
        for example in examples:
            output.write(json.dumps(example, separators=(",", ":")) + "\n")
