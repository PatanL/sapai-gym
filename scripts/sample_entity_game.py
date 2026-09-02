#!/usr/bin/env python3
"""Render one deterministic entity-policy match as a readable Markdown trace."""

from __future__ import annotations

import argparse
from collections import Counter
from pathlib import Path

import torch

from sapai.turtle_pack import SIMULATOR_REVISION

from sapai_gym1.rl import (
    ActionType,
    AutoregressiveAction,
    EntityTransformerConfig,
    EntityTransformerPolicy,
    ParallelSelfPlayEnv,
)
from sapai_gym1.rl.action_schema import NO_TARGET
from sapai_gym1.rl.distributed import _completed_match_result


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("--opponent-checkpoint", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=20260828)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--device", choices=("cpu",), default="cpu")
    parser.add_argument(
        "--perspective",
        choices=("both", "p0", "p1"),
        default="both",
        help="Render both private shops or one player's information perspective.",
    )
    return parser.parse_args()


def short_name(name: str) -> str:
    for prefix in ("pet-", "food-", "status-"):
        if name.startswith(prefix):
            name = name[len(prefix) :]
            break
    return name.replace("-", " ").title()


def format_pet(pet) -> str:
    status = short_name(pet.status)
    status_text = "" if status in {"None", ""} else f" {status}"
    return (
        f"{short_name(pet.name)} {pet.attack}/{pet.health} "
        f"L{pet.level}+{pet.experience}{status_text}"
    )


def format_team(team) -> str:
    entries = []
    for index, slot in enumerate(team):
        if slot.empty:
            entries.append(f"{index}:{'_' if index else '_ (front)'}")
            continue
        front = " front" if index == 0 else ""
        entries.append(f"{index}{front}:{format_pet(slot.pet)}")
    return " | ".join(entries)


def format_shop(shop) -> str:
    entries = []
    for index, slot in enumerate(shop):
        if slot.empty:
            entries.append(f"{index}:_")
            continue
        flags = []
        if slot.frozen:
            flags.append("frozen")
        if slot.reward_group is not None:
            flags.append(f"discover#{slot.reward_group}")
        suffix = f" [{', '.join(flags)}]" if flags else ""
        if slot.slot_type in {"pet", "levelup"}:
            item = format_pet(slot.obj)
        else:
            item = short_name(slot.obj.name)
        entries.append(f"{index}:{item} ${slot.cost}{suffix}")
    return " | ".join(entries)


def load_policy(path: Path):
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    policy = EntityTransformerPolicy(
        EntityTransformerConfig(**checkpoint["config"])
    ).cpu()
    policy.load_state_dict(checkpoint["model"])
    return policy.eval(), checkpoint


@torch.inference_mode()
def policy_action(policy, observation, memory):
    batch = {
        key: torch.as_tensor(value, device="cpu").unsqueeze(0)
        for key, value in observation.items()
    }
    output = policy(batch, memory=memory, deterministic=True)
    action = AutoregressiveAction(*map(int, output.actions[0].cpu()))
    return action, output.memory


def describe_action(player, action: AutoregressiveAction) -> str:
    action_type = ActionType(action.action_type)
    if action_type is ActionType.END_TURN:
        return "END TURN"
    if action_type is ActionType.ROLL:
        return "ROLL"
    if action_type in {ActionType.FREEZE, ActionType.UNFREEZE}:
        item = short_name(player.shop[action.source].obj.name)
        return f"{action_type.name} shop {action.source} {item}"
    if action_type is ActionType.BUY_PET:
        item = format_pet(player.shop[action.source].obj)
        target = player.team[action.target]
        if target.empty:
            return f"BUY {item} from shop {action.source} -> team {action.target}"
        return (
            f"BUY-COMBINE {item} from shop {action.source} -> "
            f"team {action.target} {format_pet(target.pet)}"
        )
    if action_type is ActionType.BUY_FOOD:
        food = short_name(player.shop[action.source].obj.name)
        if action.target == NO_TARGET:
            return f"BUY {food} from shop {action.source} -> team"
        return (
            f"BUY {food} from shop {action.source} -> "
            f"team {action.target} {format_pet(player.team[action.target].pet)}"
        )
    if action_type is ActionType.SELL:
        return f"SELL team {action.source} {format_pet(player.team[action.source].pet)}"
    if action_type is ActionType.COMBINE:
        return (
            f"COMBINE team {action.source} {format_pet(player.team[action.source].pet)} "
            f"-> team {action.target} {format_pet(player.team[action.target].pet)}"
        )
    if action_type is ActionType.MOVE:
        source = (
            "empty"
            if player.team[action.source].empty
            else format_pet(player.team[action.source].pet)
        )
        target = (
            "empty"
            if player.team[action.target].empty
            else format_pet(player.team[action.target].pet)
        )
        return (
            f"MOVE/SWAP team {action.source} {source} <-> "
            f"team {action.target} {target}"
        )
    raise AssertionError(action_type)


def discover_groups(player) -> dict[int, tuple[str, ...]]:
    groups: dict[int, list[str]] = {}
    for slot in player.shop:
        if slot.slot_type != "levelup" or slot.reward_group is None:
            continue
        groups.setdefault(int(slot.reward_group), []).append(
            short_name(slot.obj.name)
        )
    return {group: tuple(names) for group, names in groups.items()}


def main():
    args = parse_args()
    torch.set_num_threads(max(1, args.threads))
    torch.set_num_interop_threads(1)
    candidate, candidate_checkpoint = load_policy(args.checkpoint)
    opponent, opponent_checkpoint = load_policy(args.opponent_checkpoint)

    env = ParallelSelfPlayEnv()
    observations = env.reset(seed=args.seed)
    memories = [
        candidate.initial_memory(1, device="cpu"),
        opponent.initial_memory(1, device="cpu"),
    ]
    policies = [candidate, opponent]
    labels = [
        f"P0 update {candidate_checkpoint.get('update')}",
        f"P1 update {opponent_checkpoint.get('update')}",
    ]
    actions_by_player = [Counter(), Counter()]
    lines = [
        f"# Sample Turtle P1 match: update {candidate_checkpoint.get('update')} vs update {opponent_checkpoint.get('update')}",
        "",
        f"- Seed: `{args.seed}`",
        "- Inference: deterministic CPU",
        f"- Checkpoint revision: `{candidate_checkpoint.get('simulator_revision')}`",
        f"- Runtime simulator revision: `{SIMULATOR_REVISION}`",
        f"- Rendered perspective: `{args.perspective}`",
        "- Team slot 0 is the front.",
        "",
    ]
    visible_players = (
        (0, 1)
        if args.perspective == "both"
        else (0,) if args.perspective == "p0" else (1,)
    )
    displayed_turn = None

    while True:
        current_turn = int(env.core.turn)
        if current_turn != displayed_turn:
            displayed_turn = current_turn
            lines.extend([f"## Turn {current_turn}", ""])
            for player_idx in visible_players:
                player = (
                    env.core.agent if player_idx == 0 else env.core.opponent
                )
                lines.extend(
                    [
                        (
                            f"- {labels[player_idx]}: {player.lives} hearts, "
                            f"{player.wins} wins, {player.gold} gold"
                        ),
                        f"  - Team: {format_team(player.team)}",
                        f"  - Shop: {format_shop(player.shop)}",
                    ]
                )
                if args.perspective != "both":
                    previous_opponent = (
                        env.core.previous_opponent_state
                        if player_idx == 0
                        else env.core.previous_agent_state
                    )
                    if previous_opponent is not None:
                        lines.append(
                            "  - Last observed opponent board: "
                            + format_team(previous_opponent)
                        )
            lines.extend(["", "Actions:", ""])

        active = env.active_players()
        selected = [None, None]
        descriptions = [None, None]
        before_gold = [env.core.agent.gold, env.core.opponent.gold]
        before_discover = [
            discover_groups(env.core.agent),
            discover_groups(env.core.opponent),
        ]
        for player_idx in (0, 1):
            if not active[player_idx]:
                continue
            selected[player_idx], memories[player_idx] = policy_action(
                policies[player_idx], observations[player_idx], memories[player_idx]
            )
            player = env.core.agent if player_idx == 0 else env.core.opponent
            descriptions[player_idx] = describe_action(player, selected[player_idx])
            actions_by_player[player_idx][
                ActionType(selected[player_idx].action_type).name
            ] += 1

        step = env.step(selected[0], selected[1])
        observations = step.observations
        for player_idx in (0, 1):
            if (
                descriptions[player_idx] is None
                or player_idx not in visible_players
            ):
                continue
            lines.append(
                f"- {labels[player_idx]} [{before_gold[player_idx]}g]: "
                f"{descriptions[player_idx]}"
            )
            player = env.core.agent if player_idx == 0 else env.core.opponent
            action_type = ActionType(selected[player_idx].action_type)
            if action_type is ActionType.ROLL and not step.battle_resolved:
                lines.append(f"  - New shop: {format_shop(player.shop)}")
            if not step.battle_resolved:
                after_discover = discover_groups(player)
                new_groups = {
                    group: names
                    for group, names in after_discover.items()
                    if group not in before_discover[player_idx]
                }
                for group, names in new_groups.items():
                    lines.append(
                        f"  - Linked discover #{group}: {' / '.join(names)}"
                    )

        if not step.battle_resolved:
            continue

        outcome = {0: labels[0] + " wins", 1: labels[1] + " wins", 2: "draw"}[
            step.battle_result
        ]
        lines.extend(
            [
                "",
                "Battle boards:",
                "",
                f"- {labels[0]}: {format_team(env.core.previous_agent_state)}",
                f"- {labels[1]}: {format_team(env.core.previous_opponent_state)}",
                f"- Result: **{outcome}**",
                (
                    f"- Score after battle: {labels[0]} {env.core.agent.lives} hearts/"
                    f"{env.core.agent.wins} wins; {labels[1]} "
                    f"{env.core.opponent.lives} hearts/{env.core.opponent.wins} wins"
                ),
                "",
            ]
        )
        if not step.terminated:
            continue

        result = _completed_match_result(env.core)
        winner = {0: labels[0], 1: labels[1], 2: "Draw"}[result]
        lines.extend(
            [
                "## Match result",
                "",
                f"**Winner: {winner}.**",
                "",
                f"- Final {labels[0]}: {env.core.agent.lives} hearts, {env.core.agent.wins} wins",
                f"- Final {labels[1]}: {env.core.opponent.lives} hearts, {env.core.opponent.wins} wins",
            ]
        )
        for player_idx in visible_players:
            lines.append(
                f"- {labels[player_idx]} action counts: "
                f"{dict(actions_by_player[player_idx])}"
            )
        lines.append("")
        break

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text("\n".join(lines))
    print(args.output)


if __name__ == "__main__":
    main()
