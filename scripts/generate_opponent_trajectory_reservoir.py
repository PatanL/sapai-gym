#!/usr/bin/env python3
"""Collect coherent turn-by-turn board trajectories from a frozen policy."""

from __future__ import annotations

import argparse
from contextlib import nullcontext
import hashlib
import json
from pathlib import Path

import numpy as np
import torch

from sapai.turtle_pack import SIMULATOR_REVISION
from sapai_gym1.rl import (
    AutoregressiveAction,
    EntityTransformerConfig,
    EntityTransformerPolicy,
    ParallelSelfPlayEnv,
)
from sapai_gym1.rl.opponent_reservoir import RESERVOIR_SCHEMA_VERSION
from sapai_gym1.rl.joint_action import JointAction, step_decisions, JOINT_SCHEMA, PRIMITIVE_SCHEMA


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--games", type=int, default=256)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seed", type=int, default=20260902)
    parser.add_argument("--deterministic", action="store_true")
    parser.add_argument(
        "--allow-checkpoint-training-revision-mismatch",
        action="store_true",
        help=(
            "Run an older frozen policy inside the current simulator solely "
            "as a behavior source. Its original revision remains recorded."
        ),
    )
    return parser.parse_args()


def main():
    args = parse_args()
    if args.games <= 0 or args.batch_size <= 0:
        raise SystemExit("--games and --batch-size must be positive")
    if args.output.exists():
        raise SystemExit('Reservoirs are immutable; choose a new output')
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = torch.device(
        args.device
        if torch.cuda.is_available() or args.device == "cpu"
        else "cpu"
    )
    checkpoint = torch.load(
        args.checkpoint, map_location="cpu", weights_only=False
    )
    if (
        checkpoint.get("simulator_revision") != SIMULATOR_REVISION
        and not args.allow_checkpoint_training_revision_mismatch
    ):
        raise SystemExit(
            "Checkpoint simulator revision mismatch: "
            f"checkpoint={checkpoint.get('simulator_revision')!r}, "
            f"runtime={SIMULATOR_REVISION!r}"
        )
    policy = EntityTransformerPolicy(
        EntityTransformerConfig(**checkpoint["config"])
    ).to(device)
    policy.load_state_dict(checkpoint["model"])
    policy.eval()
    rng = np.random.default_rng(args.seed)

    env_count = args.games
    envs = [ParallelSelfPlayEnv() for _ in range(env_count)]
    observations = [
        env.reset(seed=int(rng.integers(0, 2**31 - 1))) for env in envs
    ]
    memories = [
        [policy.initial_memory(1, device="cpu")[0] for _ in range(2)]
        for _ in envs
    ]
    boards = [[[], []] for _ in envs]
    decision_history = [[[], []] for _ in envs]
    completed = [False] * env_count

    while not all(completed):
        requests = []
        for game_index, env in enumerate(envs):
            if completed[game_index]:
                continue
            for side, active in enumerate(env.active_players()):
                if active:
                    requests.append((game_index, side))
        if not requests:
            break
        actions_by_game = [[None, None] for _ in envs]
        for start in range(0, len(requests), args.batch_size):
            request_chunk = requests[start : start + args.batch_size]
            batch = {
                key: torch.as_tensor(
                    np.stack(
                        [
                            observations[game_index][side][key]
                            for game_index, side in request_chunk
                        ]
                    ),
                    device=device,
                )
                for key in observations[request_chunk[0][0]][request_chunk[0][1]]
            }
            memory_batch = torch.stack(
                [
                    memories[game_index][side]
                    for game_index, side in request_chunk
                ]
            ).to(device)
            autocast = (
                torch.autocast("cuda", dtype=torch.bfloat16)
                if device.type == "cuda"
                else nullcontext()
            )
            with torch.inference_mode(), autocast:
                output = policy.act(
                    batch,
                    memory=memory_batch,
                    deterministic=args.deterministic,
                )
            for index, (game_index, side) in enumerate(request_chunk):
                mask = None if output.end_freeze_mask is None else int(output.end_freeze_mask[index].cpu())
                decision = JointAction(AutoregressiveAction(*map(int, output.actions[index].cpu())), mask)
                actions_by_game[game_index][side] = decision
                decision_history[game_index][side].append({
                    'turn': int(envs[game_index].core.turn), 'decision': decision.to_dict(),
                    'behaviorLogProbability': float(output.log_prob[index].cpu()),
                })
                memories[game_index][side] = (
                    output.memory[index].float().cpu()
                )

        for game_index, env in enumerate(envs):
            if completed[game_index]:
                continue
            step = step_decisions(env, actions_by_game[game_index]).result
            observations[game_index] = step.observations
            if not step.battle_resolved:
                continue
            fought_turn = env.core.turn - 1
            boards[game_index][0].append(
                {
                    "turn": fought_turn,
                    "team": env.core.previous_agent_state.state,
                }
            )
            boards[game_index][1].append(
                {
                    "turn": fought_turn,
                    "team": env.core.previous_opponent_state.state,
                }
            )
            completed[game_index] = step.terminated

    if not all(completed):
        raise RuntimeError('Incomplete matches must not be saved as complete trajectories')
    trajectories = []
    for game_index, sides in enumerate(boards):
        for side, side_boards in enumerate(sides):
            opposing_by_turn = {
                int(board["turn"]): board["team"]
                for board in sides[1 - side]
            }
            trajectories.append(
                {
                    "id": f"game-{game_index:06d}-p{side}",
                    "decisionHistory": decision_history[game_index][side],
                    "boards": [
                        {
                            **board,
                            "opponentTeam": opposing_by_turn.get(
                                int(board["turn"])
                            ),
                        }
                        for board in side_boards
                    ],
                }
            )
    payload = {
        "schemaVersion": RESERVOIR_SCHEMA_VERSION,
        "metadata": {
            "simulatorRevision": SIMULATOR_REVISION,
            "checkpointTrainingRevision": checkpoint.get(
                "simulator_revision"
            ),
            "allowedCheckpointTrainingRevisionMismatch": bool(
                args.allow_checkpoint_training_revision_mismatch
            ),
            "checkpoint": str(args.checkpoint),
            "checkpointSha256": hashlib.sha256(
                args.checkpoint.read_bytes()
            ).hexdigest(),
            "games": env_count,
            "seed": args.seed,
            "deterministic": args.deterministic,
            "actionSchema": JOINT_SCHEMA if policy.config.end_freeze else PRIMITIVE_SCHEMA,
            "decisionHistorySchema": 'typed-reservoir-decisions-v1-20260906',
            "recurrentOwnership": 'one checkpoint per match; one advance per JointAction',
        },
        "trajectories": trajectories,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, separators=(",", ":")) + "\n")
    lengths = [len(trajectory["boards"]) for trajectory in trajectories]
    print(f"simulator_revision={SIMULATOR_REVISION}")
    print(f"games={env_count} trajectories={len(trajectories)}")
    print(
        f"turns=min:{min(lengths)} mean:{np.mean(lengths):.2f} "
        f"max:{max(lengths)}"
    )
    print(
        "sha256="
        + hashlib.sha256(args.output.read_bytes()).hexdigest()
    )


if __name__ == "__main__":
    main()
