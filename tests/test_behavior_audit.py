from pathlib import Path
import sys

import numpy as np

from sapai import Team, Shop, Pet, Food
from sapai_gym1.rl import ParallelSelfPlayEnv, AutoregressiveAction, ActionType, FlatActionCodec
from sapai_gym1.rl.end_freeze import step_with_end_freeze
from test_end_freeze import assert_same_environment

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / 'scripts'))
from behavior_audit import ObservedSelfPlayEnv, BehaviorTracker, merge_reports


def setup(env, pets, shop, turn=1):
    player = env.core.agent
    player.team = Team(pets)
    player.shop = Shop(shop, turn=turn, fixed_rules=True, seed_state=np.random.RandomState(219).get_state())
    player._connect_objects()
    player.turn = env.core.turn = turn
    env.core.team_agent, env.core.shop_agent = player.team, player.shop
    player.gold = 20
    return player


def resolve(env, kind, source=10, target=5):
    env.core.resolve_action(0, FlatActionCodec.encode(AutoregressiveAction(kind, source, target)))


def test_passive_observer_preserves_seeded_gameplay_and_rng():
    plain, observed = ParallelSelfPlayEnv(), ObservedSelfPlayEnv()
    plain.reset(seed=310); observed.reset(seed=310)
    rng = np.random.default_rng(77)
    for _ in range(160):
        actions = []
        for side, active in enumerate(plain.active_players()):
            legal = list(plain.core._avail_actions(side)) if active else []
            actions.append(FlatActionCodec.decode(int(rng.choice(legal))) if active else None)
        first, second = plain.step(*actions), observed.step(*actions)
        assert (first.rewards, first.battle_result, first.terminated) == (second.rewards, second.battle_result, second.terminated)
        assert_same_environment(plain, observed)
        if first.terminated:
            break


def test_joint_freezes_are_counted_as_clicks_but_end_coverage_uses_final_flags():
    env = ObservedSelfPlayEnv(); env.reset(seed=23)
    setup(env, ['fish', 'fish'], [Pet('fish'), Pet('cricket'), Food('chocolate')])
    env.core.agent.gold = env.core.opponent.gold = 0
    end = AutoregressiveAction(ActionType.END_TURN)
    step_with_end_freeze(env, end, end, mask_ids=(5, 0))
    counts = env.tracker.counts[0]
    assert counts['action:FREEZE'] == 2 and counts['action:END_TURN'] == 1
    assert counts['level1_matching_shop_slots'] == counts['frozen_level1_matching_shop_slots'] == 1
    assert counts['level2_completing_copy_slots'] == counts['frozen_level2_completing_copy_slots'] == 1
    assert counts['usable_chocolate_slots'] == counts['frozen_usable_chocolate_slots'] == 1
    assert counts['end_pet_slots'] == 2 and counts['end_frozen_pet_slots'] == 1
    assert counts['end_unowned_pet_slots'] == 1 and counts['end_frozen_unowned_pet_slots'] == 0
    assert counts['end_food_slots'] == counts['end_frozen_food_slots'] == 1


def test_buy_modes_levelups_and_pill_target_describe_actual_actions():
    env = ObservedSelfPlayEnv(); env.reset(seed=73)
    player = setup(env, ['fish', 'none', 'flamingo'], [Pet('fish'), Pet('fish'), Food('sleeping-pill')], turn=2)
    resolve(env, ActionType.BUY_PET, 0, 1)
    resolve(env, ActionType.BUY_PET, 0, 0)
    resolve(env, ActionType.COMBINE, 1, 0)
    pill = next(i for i, slot in enumerate(player.shop) if slot.obj.name == 'food-sleeping-pill')
    resolve(env, ActionType.BUY_FOOD, pill, 2)
    counts = env.tracker.turn_counts[0][2]
    assert counts['matching_pet_buy_separate'] == counts['matching_pet_buy_stacked'] == 1
    assert counts['levelup:pet-fish:L2'] == 1
    assert counts['pill_target:pet-flamingo'] == 1


def test_composition_geometry_is_ordered_not_just_cooccurrence():
    env = ParallelSelfPlayEnv(); env.reset(seed=9)
    tracker = BehaviorTracker()
    env.core.turn = 8
    env.core.previous_agent_state = Team(['elephant', 'blowfish', 'wolverine'])
    env.core.previous_opponent_state = Team(['blowfish', 'elephant', 'wolverine'])
    tracker.battle(env.core)
    assert tracker.counts[0]['line:elephant_hurt_wolverine'] == 1
    assert tracker.counts[1]['line:elephant_hurt_wolverine'] == 0
    mammoth = Pet('mammoth'); mammoth.status = 'status-extra-life'
    env.core.previous_agent_state = Team([mammoth, 'tiger'])
    env.core.previous_opponent_state = Team(['deer', 'parrot'])
    tracker.battle(env.core)
    assert tracker.counts[0]['line:mushroom_mammoth_tiger'] == 1
    assert tracker.counts[1]['line:parrot_deer'] == 1
    merged = merge_reports([tracker.side_report(0), tracker.side_report(1)])
    assert merged['counts']['battle_boards'] == 4


def test_low_gold_sale_sequence_counts_are_not_causal_reward_labels():
    for buys_later in (False, True):
        env = ObservedSelfPlayEnv(); env.reset(seed=91)
        player = setup(env, ['fish', 'pig'], [Pet('otter')])
        player.gold = 2 if buys_later else 0
        resolve(env, ActionType.SELL, 0)
        if buys_later:
            resolve(env, ActionType.BUY_PET, 0, 0)
        else:
            resolve(env, ActionType.ROLL)
        resolve(env, ActionType.END_TURN)
        counts = env.tracker.counts[0]
        assert counts['low_gold_sells'] == 1
        assert counts['low_gold_sells_followed_by_purchase'] == int(buys_later)
        assert counts['low_gold_sells_without_later_purchase'] == int(not buys_later)


def test_early_cake_retention_and_sale_follow_the_same_pet_only():
    env = ObservedSelfPlayEnv(); env.reset(seed=67)
    player = setup(env, ['fish'], [Food('cake'), Food('cake')], turn=5)
    resolve(env, ActionType.BUY_FOOD, 0, 0)
    resolve(env, ActionType.BUY_FOOD, 0, 0)
    assert env.tracker.counts[0]['early_cake_purchases'] == 2
    assert env.tracker.counts[0]['early_cake_distinct_targets'] == 1
    env.core.turn = 10  # record the completed turn-9 battle fixture
    env.core.previous_agent_state = Team.from_state(player.team.state)
    env.core.previous_opponent_state = Team([])
    env.tracker.battle(env.core)
    assert env.tracker.counts[0]['early_cake_same_pet_present_turn9'] == 1
    assert env.tracker.counts[0]['early_cake_equipped_turn9'] == 1
    env.core.turn = 10
    gold = player.gold
    resolve(env, ActionType.SELL, 0)
    assert env.tracker.counts[0]['early_cake_pet_sold_turn9_plus'] == 1
    assert env.tracker.counts[0]['early_cake_pet_sale_gold_turn9_plus'] == player.gold - gold


def test_offer_counts_deduplicate_no_cost_clicks_and_distinguish_full_board():
    env = ObservedSelfPlayEnv(); env.reset(seed=53)
    setup(env, ['flamingo', 'fish', 'pig', 'ant', 'beaver'], [Pet('dragon'), Food('sleeping-pill')], turn=11)
    resolve(env, ActionType.FREEZE, 0)
    resolve(env, ActionType.MOVE, 0, 1)
    counts = env.tracker.counts[0]
    assert counts['affordable_offer_turn:pet-dragon'] == 1
    assert counts['directly_buyable_offer_turn:pet-dragon'] == 0
    assert counts['directly_buyable_pill_with_faint_pet_offer_turns'] == 1
    resolve(env, ActionType.SELL, 4)
    resolve(env, ActionType.BUY_PET, 0, 4)
    assert counts['affordable_offer_turn:pet-dragon'] == 1
    assert counts['directly_buyable_offer_turn:pet-dragon'] == 1
