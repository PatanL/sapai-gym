#!/usr/bin/env python3
"""Record exact expert state/action demonstrations in the repaired simulator."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from sapai.turtle_pack import SIMULATOR_REVISION

from sapai_gym1.rl import (
    ActionType,
    AutoregressiveAction,
    FlatActionCodec,
    EntityTransformerConfig,
    EntityTransformerPolicy,
    NO_SOURCE,
    NO_TARGET,
    ParallelSelfPlayEnv,
    TurtleGuideTeacher,
)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=Path("replay-data/expert/expert.jsonl"))
    parser.add_argument("--episodes", type=int, default=1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--auto",
        action="store_true",
        help="Use an automatic teacher instead of interactive recording",
    )
    parser.add_argument(
        "--teacher",
        choices=("guide", "random"),
        default="guide",
        help="Automatic teacher; random is retained only for pipeline smoke tests",
    )
    parser.add_argument(
        "--record-opponent",
        action="store_true",
        help="Also store exact automatic-opponent decisions",
    )
    parser.add_argument(
        "--rollout-checkpoint",
        type=Path,
        help=(
            "Execute this policy while recording guide labels on the policy's "
            "visited states (DAgger-style correction data)."
        ),
    )
    parser.add_argument("--rollout-device", default="cuda")
    parser.add_argument(
        "--stochastic-rollout",
        action="store_true",
        help="Sample the rollout policy instead of using deterministic actions.",
    )
    parser.add_argument("--replay-prefix", default="recorded")
    return parser.parse_args()


def legal_actions(observation):
    actions = []
    for action_type in np.flatnonzero(observation["type_mask"]):
        for source in np.flatnonzero(observation["source_mask"][action_type]):
            for target in np.flatnonzero(
                observation["target_mask"][action_type, source]
            ):
                actions.append(
                    AutoregressiveAction(
                        int(action_type), int(source), int(target)
                    )
                )
    return actions


def describe(action):
    action_type = ActionType(action.action_type).name
    source = "-" if action.source == NO_SOURCE else str(action.source)
    target = "-" if action.target == NO_TARGET else str(action.target)
    return f"{action_type:<10} source={source} target={target}"


def choose_scripted(observation, action_count, rng):
    actions = legal_actions(observation)
    non_end = [action for action in actions if action.action_type != ActionType.END_TURN]
    if not non_end:
        return actions[0]
    if action_count >= 3:
        end_probability = min(0.15 + 0.08 * (action_count - 3), 0.75)
        if rng.random() < end_probability:
            return next(
                action for action in actions if action.action_type == ActionType.END_TURN
            )
    weights_by_type = {
        ActionType.BUY_PET: 4.0,
        ActionType.BUY_FOOD: 3.0,
        ActionType.COMBINE: 4.0,
        ActionType.ROLL: 1.5,
        ActionType.SELL: 0.7,
        ActionType.FREEZE: 0.25,
        ActionType.UNFREEZE: 0.2,
        ActionType.MOVE: 0.15,
    }
    weights = np.array(
        [weights_by_type.get(ActionType(action.action_type), 1.0) for action in non_end],
        dtype=np.float64,
    )
    weights /= weights.sum()
    return non_end[int(rng.choice(len(non_end), p=weights))]


def show_state(env):
    player = env.core.agent
    print(f"\nturn={player.turn} gold={player.gold} lives={player.lives} wins={player.wins}")
    print("team:")
    for index, slot in enumerate(player.team):
        print(
            f"  {index}: empty"
            if slot.empty
            else f"  {index}: {slot.pet.name} {slot.pet.attack}/{slot.pet.health} "
                 f"L{slot.pet.level} {slot.pet.status}"
        )
    print("shop:")
    for index, slot in enumerate(player.shop):
        print(
            f"  {index}: {slot.obj.name} cost={slot.cost} frozen={slot.frozen}"
        )


def choose_interactive(env, observation):
    show_state(env)
    actions = legal_actions(observation)
    for index, action in enumerate(actions):
        print(f"  [{index:>2}] {describe(action)}")
    while True:
        try:
            choice = int(input("action> "))
            return actions[choice]
        except (ValueError, IndexError):
            print("Choose a listed action index.")


def serializable_observation(observation):
    return {key: value.tolist() for key, value in observation.items()}


def guide_metadata(teacher, decision):
    return {
        "name": teacher.name,
        "version": teacher.version,
        "sha256": teacher.guide_sha256,
        "executableSha256": teacher.executable_sha256,
        "boardClass": decision.board_class,
        "line": decision.line_name,
        "lineConfidence": decision.line_confidence,
        "committedSlots": list(decision.committed_slots),
        "flexibleSlots": list(decision.flexible_slots),
        "sellSlot": decision.sell_slot,
        "requiredHits": list(decision.required_hits),
        "matchupCounters": list(decision.matchup_counters),
        "opponentLives": decision.opponent_lives,
        "opponentWins": decision.opponent_wins,
        "tempoPressure": decision.tempo_pressure,
        "score": decision.score,
    }


def load_rollout_policy(path, device):
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    checkpoint_revision = checkpoint.get("simulator_revision")
    if checkpoint_revision != SIMULATOR_REVISION:
        raise SystemExit(
            "Rollout checkpoint simulator mismatch: "
            f"checkpoint={checkpoint_revision!r} current={SIMULATOR_REVISION!r}"
        )
    policy = EntityTransformerPolicy(
        EntityTransformerConfig(**checkpoint["config"])
    ).to(device)
    policy.load_state_dict(checkpoint["model"])
    return policy.eval()


@torch.no_grad()
def choose_rollout(policy, observation, memory, device, deterministic):
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
        result = policy(
            batch, memory=memory, deterministic=deterministic
        )
    action = AutoregressiveAction(*map(int, result.actions[0].cpu()))
    return action, result.memory


def main():
    args = parse_args()
    if args.rollout_checkpoint is not None and (
        not args.auto or args.teacher != "guide"
    ):
        raise SystemExit(
            "--rollout-checkpoint requires --auto --teacher guide"
        )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(args.seed)
    guide_teacher = TurtleGuideTeacher(rng)
    rollout_device = torch.device(
        args.rollout_device
        if torch.cuda.is_available() or args.rollout_device == "cpu"
        else "cpu"
    )
    rollout_policy = (
        load_rollout_policy(args.rollout_checkpoint, rollout_device)
        if args.rollout_checkpoint is not None
        else None
    )
    written = 0
    with args.output.open("w") as output:
        for episode in range(args.episodes):
            env = ParallelSelfPlayEnv()
            observations = env.reset(seed=int(rng.integers(0, 2**31 - 1)))
            pending = [[], []]
            rollout_memories = (
                [rollout_policy.initial_memory(1, device=rollout_device) for _ in range(2)]
                if rollout_policy is not None
                else [None, None]
            )
            while True:
                user_active, opponent_active = env.active_players()
                user_action = None
                user_teacher_metadata = None
                if user_active:
                    if args.auto and args.teacher == "guide":
                        decision = guide_teacher.choose(
                            env.core,
                            0,
                            observations[0],
                            env.core.shop_action_count_agent,
                        )
                        user_action = decision.action
                        user_teacher_metadata = guide_metadata(
                            guide_teacher, decision
                        )
                    elif args.auto:
                        user_action = choose_scripted(
                            observations[0], env.core.shop_action_count_agent, rng
                        )
                        user_teacher_metadata = {"name": "weighted-random"}
                    else:
                        user_action = choose_interactive(env, observations[0])
                        user_teacher_metadata = {"name": "human"}
                    pending[0].append(
                        {
                            "schema_version": 1,
                            "replay_id": f"{args.replay_prefix}-{args.seed}-{episode}",
                            "perspective": "expert",
                            "turn": env.core.turn,
                            "decision": len(pending[0]),
                            "observation": serializable_observation(observations[0]),
                            "action": list(user_action.as_tuple()),
                            "battle_outcome": None,
                            "final_result": None,
                            "validation": {
                                "exactSimulatorState": True,
                                "simulatorRevision": SIMULATOR_REVISION,
                                "rolloutCheckpoint": (
                                    str(args.rollout_checkpoint)
                                    if args.rollout_checkpoint is not None
                                    else None
                                ),
                            },
                            "teacher": user_teacher_metadata,
                        }
                    )
                    if rollout_policy is not None:
                        user_action, rollout_memories[0] = choose_rollout(
                            rollout_policy,
                            observations[0],
                            rollout_memories[0],
                            rollout_device,
                            deterministic=not args.stochastic_rollout,
                        )
                opponent_action = None
                opponent_teacher_metadata = None
                if opponent_active:
                    if args.teacher == "guide":
                        opponent_decision = guide_teacher.choose(
                            env.core,
                            1,
                            observations[1],
                            env.core.shop_action_count_opponent,
                        )
                        opponent_action = opponent_decision.action
                        opponent_teacher_metadata = guide_metadata(
                            guide_teacher, opponent_decision
                        )
                    else:
                        opponent_action = choose_scripted(
                            observations[1],
                            env.core.shop_action_count_opponent,
                            rng,
                        )
                        opponent_teacher_metadata = {
                            "name": "weighted-random"
                        }
                    if args.record_opponent:
                        pending[1].append(
                            {
                                "schema_version": 1,
                                "replay_id": f"{args.replay_prefix}-{args.seed}-{episode}",
                                "perspective": "automatic-opponent",
                                "turn": env.core.turn,
                                "decision": len(pending[1]),
                                "observation": serializable_observation(
                                    observations[1]
                                ),
                                "action": list(opponent_action.as_tuple()),
                                "battle_outcome": None,
                                "final_result": None,
                                "validation": {
                                    "exactSimulatorState": True,
                                    "simulatorRevision": SIMULATOR_REVISION,
                                    "rolloutCheckpoint": (
                                        str(args.rollout_checkpoint)
                                        if args.rollout_checkpoint is not None
                                        else None
                                    ),
                                },
                                "teacher": opponent_teacher_metadata,
                            }
                        )
                    if rollout_policy is not None:
                        opponent_action, rollout_memories[1] = choose_rollout(
                            rollout_policy,
                            observations[1],
                            rollout_memories[1],
                            rollout_device,
                            deterministic=not args.stochastic_rollout,
                        )
                step = env.step(user_action, opponent_action)
                observations = step.observations
                if step.battle_resolved:
                    outcomes = (
                        step.battle_result,
                        2
                        if step.battle_result == 2
                        else 1 - step.battle_result,
                    )
                    for side in range(2):
                        for record in pending[side]:
                            record["battle_outcome"] = outcomes[side]
                            output.write(
                                json.dumps(record, separators=(",", ":"))
                                + "\n"
                            )
                            written += 1
                        pending[side] = []
                    output.flush()
                if step.terminated:
                    break
    print(f"recorded_examples={written} output={args.output}")


if __name__ == "__main__":
    main()
