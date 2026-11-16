#!/usr/bin/env python3
"""Generate a compact recurrent behavior anchor from a frozen checkpoint."""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path

import numpy as np
import torch

from generate_search_demonstrations import load_policy, policy_action
from sapai.turtle_pack import SIMULATOR_REVISION
from sapai_gym1.rl import ParallelSelfPlayEnv
from sapai_gym1.rl.distributed import _completed_match_result


ANCHOR_VERSION = "checkpoint-policy-anchor-v1-20260903"


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--summary-output", type=Path, required=True)
    parser.add_argument("--games", type=int, default=32)
    parser.add_argument("--max-sequences", type=int, default=512)
    parser.add_argument("--teacher-weight", type=float, default=0.25)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seed", type=int, default=2792001)
    parser.add_argument("--deterministic-play", action="store_true")
    return parser.parse_args()


def serialize_observation(observation):
    return {name: value.tolist() for name, value in observation.items()}


def turn_band(turn):
    if turn <= 2:
        return "early"
    if turn <= 6:
        return "mid"
    if turn <= 10:
        return "late"
    return "endgame"


def main():
    args = parse_args()
    if args.games <= 0 or args.max_sequences <= 0:
        raise SystemExit("--games and --max-sequences must be positive")
    if not 0 < args.teacher_weight <= 1:
        raise SystemExit("--teacher-weight must be in (0, 1]")
    device = torch.device(
        args.device
        if torch.cuda.is_available() or args.device == "cpu"
        else "cpu"
    )
    torch.manual_seed(args.seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(args.seed)
    policy, checkpoint = load_policy(args.checkpoint, device, False)
    checkpoint_hash = hashlib.sha256(args.checkpoint.read_bytes()).hexdigest()
    rng = np.random.default_rng(args.seed)
    sequences = defaultdict(list)
    outcome_counts = Counter()
    action_counts = Counter()

    for game in range(args.games):
        env = ParallelSelfPlayEnv()
        observations = env.reset(seed=int(rng.integers(0, 2**31 - 1)))
        memories = [
            policy.initial_memory(1, device="cpu")[0] for _ in range(2)
        ]
        decisions = Counter()
        while True:
            active = env.active_players()
            actions = [None, None]
            for side in range(2):
                if not active[side]:
                    continue
                turn = int(env.core.turn)
                observation = observations[side]
                memory = memories[side].clone()
                action, memories[side] = policy_action(
                    policy,
                    observation,
                    memories[side],
                    device,
                    args.deterministic_play,
                )
                actions[side] = action
                key = (game, side, turn)
                sequences[key].append(
                    {
                        "schema_version": 1,
                        "replay_id": (
                            f"policy-anchor-game-{game:05d}-p{side}-turn-{turn:02d}"
                        ),
                        "perspective": f"p{side}",
                        "turn": turn,
                        "decision": decisions[(side, turn)],
                        "observation": serialize_observation(observation),
                        "initialMemory": memory.float().cpu().tolist(),
                        "action": list(action.as_tuple()),
                        "teacherWeight": args.teacher_weight,
                        "battle_outcome": None,
                        "final_result": None,
                        "validation": {
                            "exactSimulatorState": True,
                            "simulatorRevision": SIMULATOR_REVISION,
                            "publicInformationOnly": True,
                        },
                        "teacher": {
                            "name": "checkpoint-policy-anchor",
                            "version": ANCHOR_VERSION,
                            "checkpoint": str(args.checkpoint),
                            "checkpointSha256": checkpoint_hash,
                            "sampledPolicy": not args.deterministic_play,
                        },
                    }
                )
                decisions[(side, turn)] += 1
                action_counts[action.action_type] += 1
            step = env.step(actions[0], actions[1])
            observations = step.observations
            if step.terminated:
                outcome_counts[_completed_match_result(env.core)] += 1
                break

    grouped = defaultdict(list)
    for key, sequence in sequences.items():
        grouped[turn_band(key[2])].append((key, sequence))
    selected = []
    bands = tuple(sorted(grouped))
    quota = max(1, args.max_sequences // max(1, len(bands)))
    for band in bands:
        candidates = grouped[band]
        rng.shuffle(candidates)
        selected.extend(candidates[:quota])
    if len(selected) < min(args.max_sequences, len(sequences)):
        selected_keys = {key for key, _ in selected}
        remainder = [
            entry for entry in sequences.items() if entry[0] not in selected_keys
        ]
        rng.shuffle(remainder)
        selected.extend(remainder[: args.max_sequences - len(selected)])
    selected = selected[: args.max_sequences]
    selected.sort(key=lambda item: item[0])

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w") as stream:
        for _, sequence in selected:
            for example in sequence:
                stream.write(json.dumps(example, separators=(",", ":")) + "\n")
    selected_action_counts = Counter(
        example["action"][0]
        for _, sequence in selected
        for example in sequence
    )
    summary = {
        "version": ANCHOR_VERSION,
        "simulatorRevision": SIMULATOR_REVISION,
        "checkpoint": str(args.checkpoint),
        "checkpointSha256": checkpoint_hash,
        "checkpointUpdate": checkpoint.get("update"),
        "games": args.games,
        "generatedSequences": len(sequences),
        "selectedSequences": len(selected),
        "selectedActions": sum(map(len, (sequence for _, sequence in selected))),
        "teacherWeight": args.teacher_weight,
        "sampledPolicy": not args.deterministic_play,
        "outcomes": dict(sorted(outcome_counts.items())),
        "actionTypeCounts": dict(sorted(selected_action_counts.items())),
        "turnBandCounts": dict(
            sorted(Counter(turn_band(key[2]) for key, _ in selected).items())
        ),
        "sha256": hashlib.sha256(args.output.read_bytes()).hexdigest(),
    }
    args.summary_output.write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n"
    )
    print(json.dumps(summary, sort_keys=True))


if __name__ == "__main__":
    main()
