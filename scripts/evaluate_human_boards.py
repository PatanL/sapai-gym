#!/usr/bin/env python3
"""Turn-matched battle benchmark against validated human replay boards."""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import json
from pathlib import Path

import numpy as np
import torch

from sapai import Battle, Pet, Team
from sapai.turtle_pack import SIMULATOR_REVISION
from sapai_gym1.rl import (
    ActionType,
    AutoregressiveAction,
    EntityTransformerConfig,
    EntityTransformerPolicy,
    ParallelSelfPlayEnv,
    TurtleGuideTeacher,
)


REPLAY_PERK_TO_STATUS = {
    "1": "status-extra-life",
    "2": "status-poison-attack",
    "12": "status-steak-attack",
    "13": "status-melon-armor",
    # Game-version-48 replay enum. The +7 recorded health and persistence
    # across end turns identify current Turtle Bread.
    "41": "status-bread",
}


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("human_report", type=Path)
    parser.add_argument("--games", type=int, default=100)
    parser.add_argument("--battle-seeds", type=int, default=5)
    parser.add_argument("--seed", type=int, default=20261110)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def load_policy(path, device):
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    if checkpoint.get("simulator_revision") != SIMULATOR_REVISION:
        raise SystemExit(
            "Checkpoint simulator mismatch: "
            f"{checkpoint.get('simulator_revision')!r} != "
            f"{SIMULATOR_REVISION!r}"
        )
    policy = EntityTransformerPolicy(
        EntityTransformerConfig(**checkpoint["config"])
    ).to(device)
    policy.load_state_dict(checkpoint["model"])
    return policy.eval(), checkpoint


@torch.no_grad()
def policy_action(policy, observation, memory, device):
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
        output = policy(batch, memory=memory, deterministic=True)
    return (
        AutoregressiveAction(*map(int, output.actions[0].cpu())),
        output.memory,
    )


def human_team(state):
    team = Team()
    for index, entry in enumerate(state["team"]):
        if entry is None:
            continue
        pet = Pet(entry["sapaiId"])
        pet._attack = int(entry["attack"])
        pet._health = int(entry["health"])
        pet.level = int(entry["level"])
        pet.experience = int(entry.get("experience", 0))
        perk = entry.get("perkEnum")
        pet.status = REPLAY_PERK_TO_STATUS.get(str(perk), "none")
        team[index] = pet
    return team


def result_name(result):
    return {0: "win", 1: "loss", 2: "draw"}[int(result)]


def main():
    args = parse_args()
    if args.games <= 0 or args.battle_seeds <= 0:
        raise SystemExit("--games and --battle-seeds must be positive")
    device = torch.device(
        args.device if torch.cuda.is_available() or args.device == "cpu" else "cpu"
    )
    policy, checkpoint = load_policy(args.checkpoint, device)
    report = json.loads(args.human_report.read_text())
    human = defaultdict(dict)
    recorded_outcomes = defaultdict(dict)
    for row in report["examples"]:
        perspective = row["perspective"]
        turn = int(row["turn"])
        human[turn][perspective] = human_team(row["state"])
        recorded_outcomes[turn][perspective] = int(row["battle_outcome"])
    complete_turns = sorted(
        turn
        for turn, perspectives in human.items()
        if {"player", "opponent"} <= set(perspectives)
    )

    rng = np.random.default_rng(args.seed)
    calibration = {}
    for turn in complete_turns:
        outcomes = Counter(
            Battle(
                human[turn]["player"],
                human[turn]["opponent"],
                seed=int(rng.integers(0, 2**32 - 1)),
            ).battle()
            for _ in range(max(21, args.battle_seeds))
        )
        calibration[str(turn)] = {
            "recordedPlayerOutcome": recorded_outcomes[turn]["player"],
            "simulated": {
                result_name(result): count for result, count in outcomes.items()
            },
            "majority": result_name(outcomes.most_common(1)[0][0]),
        }

    totals = defaultdict(Counter)
    by_turn = defaultdict(lambda: defaultdict(Counter))
    captured = defaultdict(Counter)
    guide = TurtleGuideTeacher(rng)
    for _ in range(args.games):
        env = ParallelSelfPlayEnv()
        observations = env.reset(seed=int(rng.integers(0, 2**31 - 1)))
        memories = [
            policy.initial_memory(1, device=device),
            None,
        ]
        pending = [None, None]
        while True:
            active = env.active_players()
            actions = [None, None]
            turn = int(env.core.turn)
            if active[0]:
                actions[0], memories[0] = policy_action(
                    policy, observations[0], memories[0], device
                )
                if actions[0].action_type == ActionType.END_TURN:
                    pending[0] = Team.from_state(env.core.agent.team.state)
            if active[1]:
                actions[1] = guide.choose(
                    env.core,
                    1,
                    observations[1],
                    env.core.shop_action_count_opponent,
                ).action
                if actions[1].action_type == ActionType.END_TURN:
                    pending[1] = Team.from_state(env.core.opponent.team.state)
            step = env.step(actions[0], actions[1])
            observations = step.observations
            if step.battle_resolved and turn in human:
                for builder_index, builder_name in ((0, "policy"), (1, "guide")):
                    board = pending[builder_index]
                    if board is None:
                        continue
                    captured[builder_name][turn] += 1
                    for perspective, target in human[turn].items():
                        for _ in range(args.battle_seeds):
                            outcome = Battle(
                                board,
                                target,
                                seed=int(rng.integers(0, 2**32 - 1)),
                            ).battle()
                            name = result_name(outcome)
                            key = f"{builder_name}|{perspective}"
                            totals[key][name] += 1
                            by_turn[key][turn][name] += 1
                pending = [None, None]
            if step.terminated:
                break

    payload = {
        "checkpoint": str(args.checkpoint),
        "checkpointUpdate": checkpoint.get("update"),
        "checkpointTransitions": checkpoint.get("transitions"),
        "simulatorRevision": SIMULATOR_REVISION,
        "humanReplayId": report.get("replayId"),
        "humanReplayCount": 1,
        "turns": complete_turns,
        "games": args.games,
        "battleSeedsPerPair": args.battle_seeds,
        "calibration": calibration,
        "capturedBoards": {
            builder: {str(turn): count for turn, count in sorted(counts.items())}
            for builder, counts in captured.items()
        },
        "totals": {key: dict(counts) for key, counts in totals.items()},
        "byTurn": {
            key: {
                str(turn): dict(counts)
                for turn, counts in sorted(turns.items())
            }
            for key, turns in by_turn.items()
        },
        "limitations": [
            "Single ranked Turtle 1v1 replay; not a population estimate.",
            "Measures turn-matched battle-board strength, not shop decision quality.",
            "Replay-era perk 41 is mapped to Bread from recorded +7 health behavior.",
        ],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, sort_keys=True))
    print(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
