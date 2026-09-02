#!/usr/bin/env python3
"""Evaluate an entity policy on complete corrected Turtle matches."""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import json
from pathlib import Path
import time

import numpy as np
import torch

from sapai.turtle_pack import SIMULATOR_REVISION, TURTLE_PET_STATS

from sapai_gym1.rl import (
    ActionType,
    AutoregressiveAction,
    EntityTransformerConfig,
    EntityTransformerPolicy,
    ParallelSelfPlayEnv,
    TurtleGuideTeacher,
)
from sapai_gym1.rl.distributed import _completed_match_result
from sapai_gym1.rl.guide_teacher import legal_actions
from sapai_gym1.rl.synergy_metrics import (
    SYNERGY_LABELS,
    detect_team_synergies,
)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("--games", type=int, default=100)
    parser.add_argument(
        "--opponent", choices=("guide", "random", "checkpoint"), default="guide"
    )
    parser.add_argument("--opponent-checkpoint", type=Path)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seed", type=int, default=20260828)
    parser.add_argument("--stochastic", action="store_true")
    parser.add_argument("--output", type=Path)
    return parser.parse_args()


def load_policy(path, device):
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    policy = EntityTransformerPolicy(
        EntityTransformerConfig(**checkpoint["config"])
    ).to(device)
    policy.load_state_dict(checkpoint["model"])
    return policy.eval(), checkpoint


@torch.no_grad()
def policy_action(policy, observation, memory, device, deterministic):
    batch = {
        key: torch.as_tensor(value, device=device).unsqueeze(0)
        for key, value in observation.items()
    }
    autocast = (
        torch.autocast("cuda", dtype=torch.bfloat16)
        if device.type == "cuda"
        else torch.autocast("cpu", enabled=False)
    )
    with autocast:
        output = policy(batch, memory=memory, deterministic=deterministic)
    action = AutoregressiveAction(*map(int, output.actions[0].cpu()))
    return action, output.memory


def main():
    args = parse_args()
    if args.games <= 0:
        raise SystemExit("--games must be positive")
    if args.opponent == "checkpoint" and args.opponent_checkpoint is None:
        raise SystemExit("--opponent-checkpoint is required for checkpoint opponents")
    device = torch.device(
        args.device if torch.cuda.is_available() or args.device == "cpu" else "cpu"
    )
    policy, checkpoint = load_policy(args.checkpoint, device)
    opponent_policy = None
    if args.opponent == "checkpoint":
        opponent_policy, _ = load_policy(args.opponent_checkpoint, device)

    rng = np.random.default_rng(args.seed)
    guide = TurtleGuideTeacher(rng)
    outcomes = Counter()
    battles = Counter()
    action_types = Counter()
    turns = []
    final_stats = []
    unspent_gold = []
    unspent_gold_by_turn = defaultdict(list)
    discover_decisions = 0
    discover_buys = 0
    discover_groups_seen = 0
    discover_freeze_actions = 0
    discover_first_choice_freezes = 0
    discover_second_choice_freezes = 0
    discover_unfreeze_actions = 0
    discover_group_roll_exposures = Counter()
    discover_unique_groups_rolled = Counter()
    discover_group_refresh_exposures = Counter()
    sell_stats_sacrificed = []
    sell_then_buy_turns = 0
    full_board_end_turns = 0
    late_low_tier_pets = []
    statted_pet_counts = []
    premium_full_board_skips = Counter()
    pivot_match_outcomes = Counter()
    nonpivot_match_outcomes = Counter()
    matches_with_pivot = 0
    move_actions_by_turn = Counter()
    moves_per_completed_turn = []
    move_position_deltas = []
    move_revisited_arrangements = 0
    immediate_move_reversals = 0
    bought_pets = Counter()
    bought_foods = Counter()
    sold_pets = Counter()
    final_pets = Counter()
    final_statuses = Counter()
    final_compositions = Counter()
    synergy_end_turns = Counter()
    synergy_matches = Counter()
    synergy_final_teams = Counter()
    synergy_first_turns = defaultdict(list)
    synergy_match_outcomes = defaultdict(Counter)
    level_two_targets = {"pet-dragon", "pet-monkey"}
    level_two_eligibility_turn = {"pet-dragon": 11, "pet-monkey": 9}
    level_two_eligible_matches = Counter()
    level_two_end_turns = Counter()
    level_two_matches = Counter()
    level_two_final_teams = Counter()
    level_two_first_turns = defaultdict(list)
    level_two_achieved_actions = defaultdict(Counter)
    early_cake_purchases = 0
    early_cake_target_pets = Counter()
    early_cake_retained_records = set()
    early_cake_realized_records = set()
    early_cake_matches = 0
    early_cake_retained_matches = 0
    early_cake_realized_matches = 0
    early_cake_late_sale_events = 0
    early_cake_early_sale_events = 0
    early_cake_late_sale_turns = []
    early_cake_late_sale_gold = 0
    early_cake_late_sale_bonus = 0
    early_cake_cashout_gold_spent = 0
    early_cake_cashout_turns = 0
    early_cake_cashout_actions = Counter()
    next_cake_record = 0
    actions_by_opponent_lives = defaultdict(Counter)
    premium_pivot_pets = {
        "pet-cow", "pet-deer", "pet-shark", "pet-turkey", "pet-monkey",
        "pet-skunk", "pet-turtle", "pet-cat", "pet-dragon", "pet-fly",
        "pet-tiger", "pet-mammoth", "pet-snake", "pet-scorpion",
        "pet-crocodile", "pet-leopard", "pet-wolverine",
    }
    started = time.perf_counter()

    for _ in range(args.games):
        env = ParallelSelfPlayEnv()
        observations = env.reset(seed=int(rng.integers(0, 2**31 - 1)))
        player_memory = policy.initial_memory(1, device=device)
        opponent_memory = (
            opponent_policy.initial_memory(1, device=device)
            if opponent_policy is not None
            else None
        )
        match_had_pivot = False
        turn_had_sell = False
        turn_pivot_recorded = False
        premium_opportunities = set()
        moves_this_turn = 0
        last_move_before = None
        seen_discover_groups = set()
        seen_discover_roll_states = set()
        match_synergies = set()
        match_level_two = set()
        match_early_cake_records = set()
        match_retained_cake_records = set()
        cake_lineages = defaultdict(set)
        match_realized_cake_records = set()
        cake_cashout_budget = 0
        cake_cashout_spent_this_turn = False

        def team_order():
            return tuple(
                None if slot.empty else slot.pet.name
                for slot in env.core.agent.team
            )

        def default_position_score():
            return sum(
                TurtleGuideTeacher._position_value(slot.pet.name, index)
                for index, slot in enumerate(env.core.agent.team)
                if not slot.empty
            )

        seen_arrangements = {team_order()}
        while True:
            move_before = None
            move_position_before = None
            level_target_before = None
            level_target_index = None
            early_cake_target_index = None
            combine_cake_records = set()
            combine_target_index = None
            early_cake_sale = None
            paid_action_cost = 0
            paid_action_label = None
            active = env.active_players()
            player_action = None
            opponent_action = None
            if active[0]:
                if all(not slot.empty for slot in env.core.agent.team):
                    premium_opportunities.update(
                        slot.obj.name
                        for slot in env.core.agent.shop
                        if slot.slot_type in {"pet", "levelup"}
                        and slot.obj.name in premium_pivot_pets
                        and slot.cost <= env.core.agent.gold
                    )
                has_discover = any(
                    slot.slot_type == "levelup" for slot in env.core.agent.shop
                )
                discover_decisions += int(has_discover)
                discover_groups = defaultdict(list)
                for slot in env.core.agent.shop:
                    if (
                        slot.slot_type == "levelup"
                        and slot.reward_group is not None
                    ):
                        discover_groups[int(slot.reward_group)].append(slot)
                for group in discover_groups:
                    if group not in seen_discover_groups:
                        seen_discover_groups.add(group)
                        discover_groups_seen += 1
                player_action, player_memory = policy_action(
                    policy,
                    observations[0],
                    player_memory,
                    device,
                    deterministic=not args.stochastic,
                )
                action_types[ActionType(player_action.action_type).name] += 1
                action_name = ActionType(player_action.action_type).name
                if player_action.action_type in {
                    ActionType.BUY_PET,
                    ActionType.BUY_FOOD,
                } and player_action.source < len(env.core.agent.shop):
                    paid_slot = env.core.agent.shop[player_action.source]
                    paid_action_cost = int(paid_slot.cost)
                    paid_action_label = f"{action_name}:{paid_slot.obj.name}"
                elif player_action.action_type == ActionType.ROLL:
                    paid_action_cost = 1
                    paid_action_label = "ROLL"
                if (
                    player_action.action_type
                    in {
                        ActionType.BUY_PET,
                        ActionType.BUY_FOOD,
                        ActionType.COMBINE,
                    }
                    and 0 <= player_action.target < len(env.core.agent.team)
                ):
                    target_slot = env.core.agent.team[player_action.target]
                    if not target_slot.empty:
                        level_target_before = (
                            target_slot.pet.name,
                            int(target_slot.pet.level),
                        )
                        level_target_index = int(player_action.target)
                if (
                    player_action.action_type == ActionType.BUY_FOOD
                    and player_action.source < len(env.core.agent.shop)
                    and env.core.agent.shop[player_action.source].obj.name
                    == "food-cake"
                    and int(env.core.turn) in {5, 6}
                ):
                    early_cake_target_index = int(player_action.target)
                if player_action.action_type == ActionType.COMBINE:
                    source_slot = env.core.agent.team[player_action.source]
                    target_slot = env.core.agent.team[player_action.target]
                    if not source_slot.empty:
                        combine_cake_records.update(
                            cake_lineages.get(id(source_slot.pet), set())
                        )
                    if not target_slot.empty:
                        combine_cake_records.update(
                            cake_lineages.get(id(target_slot.pet), set())
                        )
                    combine_target_index = int(player_action.target)
                actions_by_opponent_lives[
                    int(env.core.opponent.lives)
                ][action_name] += 1
                if (
                    player_action.action_type
                    in {ActionType.FREEZE, ActionType.UNFREEZE}
                    and player_action.source < len(env.core.agent.shop)
                ):
                    selected_shop_slot = env.core.agent.shop[
                        player_action.source
                    ]
                    if (
                        selected_shop_slot.slot_type == "levelup"
                        and selected_shop_slot.reward_group is not None
                    ):
                        group_slots = discover_groups.get(
                            int(selected_shop_slot.reward_group), []
                        )
                        if player_action.action_type == ActionType.FREEZE:
                            discover_freeze_actions += 1
                            if any(
                                slot is not selected_shop_slot and slot.frozen
                                for slot in group_slots
                            ):
                                discover_second_choice_freezes += 1
                            else:
                                discover_first_choice_freezes += 1
                        else:
                            discover_unfreeze_actions += 1
                if player_action.action_type in {
                    ActionType.ROLL,
                    ActionType.END_TURN,
                }:
                    for group, slots in discover_groups.items():
                        frozen_count = min(
                            2, sum(bool(slot.frozen) for slot in slots)
                        )
                        discover_group_refresh_exposures[frozen_count] += 1
                        if player_action.action_type == ActionType.ROLL:
                            discover_group_roll_exposures[frozen_count] += 1
                            state_key = (group, frozen_count)
                            if state_key not in seen_discover_roll_states:
                                seen_discover_roll_states.add(state_key)
                                discover_unique_groups_rolled[frozen_count] += 1
                if player_action.action_type == ActionType.MOVE:
                    move_before = team_order()
                    move_position_before = default_position_score()
                    moves_this_turn += 1
                    move_actions_by_turn[int(env.core.turn)] += 1
                if player_action.action_type == ActionType.SELL:
                    sold = env.core.agent.team[player_action.source].pet
                    sold_pets[sold.name] += 1
                    sell_stats_sacrificed.append(
                        float(sold.attack + sold.health)
                    )
                    turn_had_sell = True
                    cake_records = set(
                        cake_lineages.get(id(sold), set())
                    ) & match_early_cake_records
                    if cake_records:
                        early_cake_sale = {
                            "records": cake_records,
                            "turn": int(env.core.turn),
                            "goldBefore": int(env.core.agent.gold),
                            "sellValue": int(sold.sell_value),
                            "level": int(sold.level),
                        }
                elif player_action.action_type == ActionType.BUY_PET:
                    if player_action.source < len(env.core.agent.shop):
                        bought_pets[
                            env.core.agent.shop[player_action.source].obj.name
                        ] += 1
                        premium_opportunities.discard(
                            env.core.agent.shop[player_action.source].obj.name
                        )
                    if turn_had_sell and not turn_pivot_recorded:
                        sell_then_buy_turns += 1
                        turn_pivot_recorded = True
                        match_had_pivot = True
                elif player_action.action_type == ActionType.BUY_FOOD:
                    if player_action.source < len(env.core.agent.shop):
                        bought_foods[
                            env.core.agent.shop[player_action.source].obj.name
                        ] += 1
                if (
                    has_discover
                    and player_action.action_type == ActionType.BUY_PET
                    and player_action.source < len(env.core.agent.shop)
                    and env.core.agent.shop[player_action.source].slot_type == "levelup"
                ):
                    discover_buys += 1
                if player_action.action_type == ActionType.END_TURN:
                    remaining_gold = int(env.core.agent.gold)
                    unspent_gold.append(remaining_gold)
                    unspent_gold_by_turn[int(env.core.turn)].append(
                        remaining_gold
                    )
                    pets = [
                        slot.pet
                        for slot in env.core.agent.team
                        if not slot.empty
                    ]
                    full_board_end_turns += int(len(pets) == 5)
                    if env.core.turn >= 9:
                        late_low_tier_pets.append(
                            sum(
                                isinstance(pet.tier, int) and pet.tier <= 2
                                for pet in pets
                            )
                        )
                    statted_pet_counts.append(
                        sum(
                            pet.name in TURTLE_PET_STATS
                            and (
                                pet._attack + pet._health
                                > sum(TURTLE_PET_STATS[pet.name][:2])
                            )
                            for pet in pets
                        )
                    )
                    premium_full_board_skips.update(premium_opportunities)
                    current_synergies = detect_team_synergies(
                        env.core.agent.team
                    )
                    synergy_end_turns.update(current_synergies)
                    for synergy in current_synergies - match_synergies:
                        synergy_first_turns[synergy].append(
                            int(env.core.turn)
                        )
                    match_synergies.update(current_synergies)
                    for pet in pets:
                        if pet.name not in level_two_targets or pet.level < 2:
                            continue
                        level_two_end_turns[pet.name] += 1
                        if pet.name not in match_level_two:
                            level_two_first_turns[pet.name].append(
                                int(env.core.turn)
                            )
                            match_level_two.add(pet.name)
                    if int(env.core.turn) >= 9:
                        for pet in pets:
                            if pet.status != "status-cake":
                                continue
                            records = cake_lineages.get(id(pet), set())
                            match_retained_cake_records.update(records)
                            early_cake_retained_records.update(records)
            if active[1]:
                if args.opponent == "guide":
                    opponent_action = guide.choose(
                        env.core,
                        1,
                        observations[1],
                        env.core.shop_action_count_opponent,
                    ).action
                elif args.opponent == "random":
                    candidates = legal_actions(observations[1])
                    opponent_action = candidates[int(rng.integers(len(candidates)))]
                else:
                    opponent_action, opponent_memory = policy_action(
                        opponent_policy,
                        observations[1],
                        opponent_memory,
                        device,
                        deterministic=not args.stochastic,
                    )
            step = env.step(player_action, opponent_action)
            observations = step.observations
            if active[0] and paid_action_cost and cake_cashout_budget > 0:
                spent = min(cake_cashout_budget, paid_action_cost)
                cake_cashout_budget -= spent
                early_cake_cashout_gold_spent += spent
                early_cake_cashout_actions[paid_action_label] += 1
                if not cake_cashout_spent_this_turn:
                    early_cake_cashout_turns += 1
                    cake_cashout_spent_this_turn = True
            if active[0] and early_cake_sale is not None:
                records = early_cake_sale["records"]
                if early_cake_sale["turn"] >= 9:
                    gold_received = (
                        int(env.core.agent.gold) - early_cake_sale["goldBefore"]
                    )
                    early_cake_late_sale_events += 1
                    early_cake_late_sale_turns.append(
                        early_cake_sale["turn"]
                    )
                    early_cake_late_sale_gold += gold_received
                    early_cake_late_sale_bonus += max(
                        0,
                        early_cake_sale["sellValue"]
                        - early_cake_sale["level"],
                    )
                    early_cake_realized_records.update(records)
                    match_realized_cake_records.update(records)
                    cake_cashout_budget += max(0, gold_received)
                else:
                    early_cake_early_sale_events += 1
            if active[0] and early_cake_target_index is not None:
                target_slot = env.core.agent.team[early_cake_target_index]
                if not target_slot.empty:
                    record = next_cake_record
                    next_cake_record += 1
                    early_cake_purchases += 1
                    early_cake_target_pets[target_slot.pet.name] += 1
                    match_early_cake_records.add(record)
                    cake_lineages[id(target_slot.pet)].add(record)
            if active[0] and combine_target_index is not None and combine_cake_records:
                target_slot = env.core.agent.team[combine_target_index]
                if not target_slot.empty:
                    cake_lineages[id(target_slot.pet)].update(
                        combine_cake_records
                    )
            if active[0] and level_target_index is not None:
                target_slot = env.core.agent.team[level_target_index]
                if (
                    not target_slot.empty
                    and target_slot.pet.name in level_two_targets
                    and target_slot.pet.level >= 2
                    and (
                        level_target_before is None
                        or level_target_before[0] != target_slot.pet.name
                        or level_target_before[1] < 2
                    )
                ):
                    level_two_achieved_actions[target_slot.pet.name][
                        ActionType(player_action.action_type).name
                    ] += 1
            if move_before is not None:
                after_move = team_order()
                move_revisited_arrangements += int(
                    after_move in seen_arrangements
                )
                immediate_move_reversals += int(
                    last_move_before is not None
                    and after_move == last_move_before
                )
                seen_arrangements.add(after_move)
                move_position_deltas.append(
                    default_position_score() - move_position_before
                )
                last_move_before = move_before
            if step.battle_resolved:
                battles[{0: "win", 1: "loss", 2: "draw"}[step.battle_result]] += 1
                turn_had_sell = False
                turn_pivot_recorded = False
                premium_opportunities.clear()
                moves_per_completed_turn.append(moves_this_turn)
                moves_this_turn = 0
                last_move_before = None
                seen_arrangements = {team_order()}
                cake_cashout_budget = 0
                cake_cashout_spent_this_turn = False
            if not step.terminated:
                continue
            result = _completed_match_result(env.core)
            final_team = [
                slot.pet for slot in env.core.agent.team if not slot.empty
            ]
            final_pets.update(pet.name for pet in final_team)
            final_statuses.update(pet.status for pet in final_team)
            final_compositions.update(
                [" > ".join(pet.name for pet in final_team)]
            )
            result_name = {0: "win", 1: "loss", 2: "draw"}[result]
            outcomes[result_name] += 1
            final_synergies = detect_team_synergies(env.core.agent.team)
            synergy_final_teams.update(final_synergies)
            synergy_matches.update(match_synergies)
            for synergy in match_synergies:
                synergy_match_outcomes[synergy][result_name] += 1
            level_two_matches.update(match_level_two)
            for pet, eligible_turn in level_two_eligibility_turn.items():
                if int(env.core.turn) >= eligible_turn:
                    level_two_eligible_matches[pet] += 1
            for pet in final_team:
                if pet.name in level_two_targets and pet.level >= 2:
                    level_two_final_teams[pet.name] += 1
            early_cake_matches += int(bool(match_early_cake_records))
            early_cake_retained_matches += int(
                bool(match_retained_cake_records)
            )
            early_cake_realized_matches += int(
                bool(match_realized_cake_records)
            )
            (pivot_match_outcomes if match_had_pivot else nonpivot_match_outcomes)[
                result_name
            ] += 1
            matches_with_pivot += int(match_had_pivot)
            turns.append(env.core.turn)
            final_stats.append(
                sum(
                    slot.pet.attack + slot.pet.health
                    for slot in env.core.agent.team
                    if not slot.empty
                )
            )
            break

    elapsed = time.perf_counter() - started
    payload = {
        "checkpoint": str(args.checkpoint),
        "checkpointUpdate": checkpoint.get("update"),
        "checkpointTransitions": checkpoint.get("transitions"),
        "checkpointSimulatorRevision": checkpoint.get("simulator_revision"),
        "runtimeSimulatorRevision": SIMULATOR_REVISION,
        "guideVersion": guide.version,
        "guideSha256": guide.guide_sha256,
        "opponent": args.opponent,
        "opponentCheckpoint": (
            str(args.opponent_checkpoint)
            if args.opponent_checkpoint is not None
            else None
        ),
        "games": args.games,
        "matchOutcomes": dict(outcomes),
        "matchWinRate": outcomes["win"] / args.games,
        "battleOutcomes": dict(battles),
        "meanTurns": float(np.mean(turns)),
        "meanFinalTeamStats": float(np.mean(final_stats)),
        "meanUnspentGold": float(np.mean(unspent_gold)) if unspent_gold else None,
        "unspentGoldHistogram": {
            str(gold): count
            for gold, count in sorted(Counter(unspent_gold).items())
        },
        "endTurnsWithThreePlusGold": sum(gold >= 3 for gold in unspent_gold),
        "fractionEndTurnsWithThreePlusGold": (
            sum(gold >= 3 for gold in unspent_gold) / len(unspent_gold)
            if unspent_gold
            else None
        ),
        "meanUnspentGoldByTurn": {
            str(turn): float(np.mean(values))
            for turn, values in sorted(unspent_gold_by_turn.items())
        },
        "discoverDecisionStates": discover_decisions,
        "discoverPurchases": discover_buys,
        "discoverGroupsSeen": discover_groups_seen,
        "discoverFreezeActions": discover_freeze_actions,
        "discoverFirstChoiceFreezeActions": discover_first_choice_freezes,
        "discoverSecondChoiceFreezeActions": discover_second_choice_freezes,
        "discoverUnfreezeActions": discover_unfreeze_actions,
        "discoverGroupRollExposuresByFrozenChoices": {
            str(count): discover_group_roll_exposures[count]
            for count in range(3)
        },
        "discoverUniqueGroupsRolledByFrozenChoices": {
            str(count): discover_unique_groups_rolled[count]
            for count in range(3)
        },
        "discoverGroupRefreshExposuresByFrozenChoices": {
            str(count): discover_group_refresh_exposures[count]
            for count in range(3)
        },
        "meanSellsPerMatch": action_types["SELL"] / args.games,
        "sellThenBuyTurns": sell_then_buy_turns,
        "matchesWithPivot": matches_with_pivot,
        "pivotMatchOutcomes": dict(pivot_match_outcomes),
        "nonpivotMatchOutcomes": dict(nonpivot_match_outcomes),
        "pivotMatchWinRate": (
            pivot_match_outcomes["win"] / sum(pivot_match_outcomes.values())
            if pivot_match_outcomes
            else None
        ),
        "nonpivotMatchWinRate": (
            nonpivot_match_outcomes["win"] / sum(nonpivot_match_outcomes.values())
            if nonpivot_match_outcomes
            else None
        ),
        "meanStatsSacrificedOnSell": (
            float(np.mean(sell_stats_sacrificed))
            if sell_stats_sacrificed
            else None
        ),
        "fractionEndTurnsWithFullBoard": (
            full_board_end_turns / len(unspent_gold)
            if unspent_gold
            else None
        ),
        "meanTierOneTwoPetsAfterTurnEight": (
            float(np.mean(late_low_tier_pets))
            if late_low_tier_pets
            else None
        ),
        "premiumFullBoardSkips": dict(premium_full_board_skips),
        "meanPermanentlyStattedPets": (
            float(np.mean(statted_pet_counts))
            if statted_pet_counts
            else None
        ),
        "moveActionsByTurn": {
            str(turn): count
            for turn, count in sorted(move_actions_by_turn.items())
        },
        "movesPerTurnHistogram": {
            str(count): frequency
            for count, frequency in sorted(
                Counter(moves_per_completed_turn).items()
            )
        },
        "meanMovesPerTurn": (
            float(np.mean(moves_per_completed_turn))
            if moves_per_completed_turn
            else None
        ),
        "moveRevisitedArrangements": move_revisited_arrangements,
        "immediateMoveReversals": immediate_move_reversals,
        "boughtPetCounts": dict(bought_pets.most_common()),
        "boughtFoodCounts": dict(bought_foods.most_common()),
        "soldPetCounts": dict(sold_pets.most_common()),
        "finalPetCounts": dict(final_pets.most_common()),
        "finalStatusCounts": dict(final_statuses.most_common()),
        "finalCompositionCounts": dict(
            final_compositions.most_common(30)
        ),
        "synergyLabels": SYNERGY_LABELS,
        "synergyEndTurnCounts": {
            synergy: synergy_end_turns[synergy]
            for synergy in SYNERGY_LABELS
        },
        "synergyMatchCounts": {
            synergy: synergy_matches[synergy]
            for synergy in SYNERGY_LABELS
        },
        "synergyFinalTeamCounts": {
            synergy: synergy_final_teams[synergy]
            for synergy in SYNERGY_LABELS
        },
        "synergyMeanFirstTurn": {
            synergy: (
                float(np.mean(synergy_first_turns[synergy]))
                if synergy_first_turns[synergy]
                else None
            )
            for synergy in SYNERGY_LABELS
        },
        "synergyMatchOutcomes": {
            synergy: dict(synergy_match_outcomes[synergy])
            for synergy in SYNERGY_LABELS
        },
        "levelTwoTargetMetrics": {
            pet: {
                "endTurnCount": level_two_end_turns[pet],
                "eligibleMatchCount": level_two_eligible_matches[pet],
                "matchCount": level_two_matches[pet],
                "prevalenceAmongEligibleMatches": (
                    level_two_matches[pet] / level_two_eligible_matches[pet]
                    if level_two_eligible_matches[pet]
                    else None
                ),
                "finalTeamCount": level_two_final_teams[pet],
                "meanFirstTurn": (
                    float(np.mean(level_two_first_turns[pet]))
                    if level_two_first_turns[pet]
                    else None
                ),
                "achievedByAction": dict(level_two_achieved_actions[pet]),
            }
            for pet in sorted(level_two_targets)
        },
        "earlyCakeMetrics": {
            "purchaseTurns": [5, 6],
            "retentionTurn": 9,
            "purchases": early_cake_purchases,
            "matchesWithPurchase": early_cake_matches,
            "retainedPurchases": len(early_cake_retained_records),
            "matchesWithRetention": early_cake_retained_matches,
            "retentionRate": (
                len(early_cake_retained_records) / early_cake_purchases
                if early_cake_purchases
                else None
            ),
            "targetPetCounts": dict(early_cake_target_pets),
            "soldBeforeTurn9Events": early_cake_early_sale_events,
            "soldTurn9PlusEvents": early_cake_late_sale_events,
            "realizedPurchases": len(early_cake_realized_records),
            "matchesWithRealization": early_cake_realized_matches,
            "meanRealizationTurn": (
                float(np.mean(early_cake_late_sale_turns))
                if early_cake_late_sale_turns
                else None
            ),
            "lateSaleGoldReceived": early_cake_late_sale_gold,
            "lateSaleCakeBonus": early_cake_late_sale_bonus,
            "cashoutGoldSpentSameTurn": early_cake_cashout_gold_spent,
            "cashoutSpendTurns": early_cake_cashout_turns,
            "cashoutSpendActions": dict(early_cake_cashout_actions),
        },
        "actionTypesByOpponentLives": {
            str(lives): dict(counts)
            for lives, counts in sorted(actions_by_opponent_lives.items())
        },
        "meanDefaultPositionScoreDeltaPerMove": (
            float(np.mean(move_position_deltas))
            if move_position_deltas
            else None
        ),
        "actionTypes": {
            action_type.name: action_types[action_type.name]
            for action_type in ActionType
        },
        "elapsedSeconds": elapsed,
    }
    rendered = json.dumps(payload, indent=2, sort_keys=True)
    print(rendered)
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered + "\n")


if __name__ == "__main__":
    main()
