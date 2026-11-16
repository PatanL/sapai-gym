#!/usr/bin/env python3
"""Paired-seat fixed-seed evaluation for policy promotion decisions."""

from __future__ import annotations

import argparse
from collections import Counter
from contextlib import nullcontext
import json
from pathlib import Path
import time

import numpy as np
import torch

from sapai.turtle_pack import SIMULATOR_REVISION

from sapai_gym1.rl import (
    AutoregressiveAction,
    EntityTransformerConfig,
    EntityTransformerPolicy,
    ParallelSelfPlayEnv,
    TurtleGuideTeacher,
)
from sapai_gym1.rl.distributed import _completed_match_result
from sapai_gym1.rl.guide_teacher import legal_actions
from sapai_gym1.rl.end_freeze import step_with_end_freeze
from sapai_gym1.rl.end_freeze_checkpoint import END_FREEZE_SCHEMA, PRIMITIVE_SCHEMA


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("--pairs", type=int, default=250)
    parser.add_argument(
        "--opponent", choices=("guide", "random", "checkpoint"), default="guide"
    )
    parser.add_argument("--opponent-checkpoint", type=Path)
    parser.add_argument("--device", default="cuda")
    parser.add_argument(
        "--batch-size",
        type=int,
        default=1,
        help=(
            "Concurrent checkpoint-vs-checkpoint games. Values above one "
            "batch policy inference; guide/random opponents remain serial."
        ),
    )
    parser.add_argument(
        "--progress-every",
        type=int,
        default=50,
        help="Print progress after this many completed games; zero disables.",
    )
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
        else nullcontext()
    )
    with autocast:
        options = {'allow_end_freeze': True} if getattr(policy.config, 'end_freeze', False) else {}
        output = policy(batch, memory=memory, deterministic=deterministic, **options)
    return (
        AutoregressiveAction(*map(int, output.actions[0].cpu())),
        output.memory,
        None if output.end_freeze_mask is None else int(output.end_freeze_mask[0].cpu()),
    )


def candidate_result(player_zero_result, candidate_side):
    if candidate_side == 0 or player_zero_result == 2:
        return player_zero_result
    return 1 - player_zero_result


def evaluation_game_record(core, pair_index, seed, candidate_side, result_name):
    """Expose simulated termination rules without changing scoring or play."""
    if core.agent.lives <= 0 or core.opponent.lives <= 0:
        reason = 'lives'
    elif core.turn >= core.MAX_TURN:
        reason = 'turn_limit'
    else:
        reason = 'other'
    return {
        'pairIndex': int(pair_index), 'seed': int(seed), 'candidateSide': int(candidate_side),
        'candidateResult': result_name, 'terminationReason': reason,
        'turnCounter': int(core.turn), 'lastBattleTurn': int(core.turn - 1),
        'livesBySeat': [int(core.agent.lives), int(core.opponent.lives)],
        'winsBySeat': [int(core.agent.wins), int(core.opponent.wins)],
        'turnLimit': int(core.MAX_TURN), 'rearrangementLimit': int(core.MAX_REORDERS_PER_TURN),
    }


def write_evaluation_report(path, payload):
    """A complete report appears only after serialization succeeds."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + '\n')
    temporary.replace(path)


@torch.no_grad()
def batched_policy_actions(policy, requests, device, deterministic):
    observations = {
        key: torch.as_tensor(
            np.stack([request["observation"][key] for request in requests]),
            device=device,
        )
        for key in requests[0]["observation"]
    }
    memories = torch.stack([request["memory"] for request in requests]).to(
        device
    )
    autocast = (
        torch.autocast("cuda", dtype=torch.bfloat16)
        if device.type == "cuda"
        else nullcontext()
    )
    with autocast:
        output = policy(
            observations,
            memory=memories,
            deterministic=deterministic,
            **({'allow_end_freeze': True} if getattr(policy.config, 'end_freeze', False) else {}),
        )
    actions = [
        AutoregressiveAction(*map(int, row.cpu())) for row in output.actions
    ]
    next_memories = [row.detach().cpu() for row in output.memory]
    masks = getattr(output, 'end_freeze_mask', None)
    return actions, next_memories, ([None] * len(actions) if masks is None else masks.cpu().tolist())


def step_policy_actions(env, actions, masks):
    if any(mask is not None for mask in masks):
        return step_with_end_freeze(env, *actions, mask_ids=tuple(0 if mask is None else mask for mask in masks)).result
    return env.step(*actions)


def run_batched_checkpoint_evaluation(
    candidate,
    opponent,
    pair_seeds,
    device,
    *,
    batch_size,
    progress_every,
    game_records=None,
):
    """Play deterministic paired seats with batched policy forwards."""

    tasks = [
        (pair_index, int(match_seed), candidate_side)
        for pair_index, match_seed in enumerate(pair_seeds)
        for candidate_side in (0, 1)
    ]
    outcomes = Counter()
    seat_outcomes = {0: Counter(), 1: Counter()}
    pair_points = np.zeros(len(pair_seeds), dtype=np.float64)
    completed_games = 0
    next_progress = max(1, int(progress_every)) if progress_every else None
    started = time.perf_counter()

    for chunk_start in range(0, len(tasks), batch_size):
        chunk = tasks[chunk_start : chunk_start + batch_size]
        games = []
        for pair_index, match_seed, candidate_side in chunk:
            env = ParallelSelfPlayEnv()
            observations = env.reset(seed=match_seed)
            games.append(
                {
                    "pairIndex": pair_index,
                    "seed": int(match_seed),
                    "candidateSide": candidate_side,
                    "env": env,
                    "observations": observations,
                    "memories": [
                        (
                            candidate.initial_memory(1, device="cpu")[0]
                            if side == candidate_side
                            else opponent.initial_memory(1, device="cpu")[0]
                        )
                        for side in (0, 1)
                    ],
                    "done": False,
                }
            )

        while not all(game["done"] for game in games):
            actions = [[None, None] for _ in games]
            freeze_masks = [[None, None] for _ in games]
            candidate_requests = []
            opponent_requests = []
            for game_index, game in enumerate(games):
                if game["done"]:
                    continue
                active = game["env"].active_players()
                for side in (0, 1):
                    if not active[side]:
                        continue
                    request = {
                        "gameIndex": game_index,
                        "side": side,
                        "observation": game["observations"][side],
                        "memory": game["memories"][side],
                    }
                    (
                        candidate_requests
                        if side == game["candidateSide"]
                        else opponent_requests
                    ).append(request)

            for policy, requests in (
                (candidate, candidate_requests),
                (opponent, opponent_requests),
            ):
                if not requests:
                    continue
                selected, next_memories, masks = batched_policy_actions(
                    policy, requests, device, deterministic=True
                )
                for request, action, memory, mask in zip(
                    requests, selected, next_memories, masks
                ):
                    game_index = request["gameIndex"]
                    side = request["side"]
                    actions[game_index][side] = action
                    freeze_masks[game_index][side] = mask
                    games[game_index]["memories"][side] = memory

            for game_index, game in enumerate(games):
                if game["done"]:
                    continue
                step = step_policy_actions(game['env'], actions[game_index], freeze_masks[game_index])
                game["observations"] = step.observations
                if not step.terminated:
                    continue
                game["done"] = True
                result = candidate_result(
                    _completed_match_result(game["env"].core),
                    game["candidateSide"],
                )
                result_name = {0: "win", 1: "loss", 2: "draw"}[result]
                if game_records is not None:
                    game_records.append(evaluation_game_record(
                        game['env'].core, game['pairIndex'], game['seed'],
                        game['candidateSide'], result_name,
                    ))
                outcomes[result_name] += 1
                seat_outcomes[game["candidateSide"]][result_name] += 1
                pair_points[game["pairIndex"]] += (
                    1.0 if result == 0 else 0.5 if result == 2 else 0.0
                )
                completed_games += 1

            if next_progress is not None and completed_games >= next_progress:
                elapsed = time.perf_counter() - started
                print(
                    f"progress games={completed_games}/{len(tasks)} "
                    f"games_per_second={completed_games / max(elapsed, 1e-9):.2f}",
                    flush=True,
                )
                while next_progress <= completed_games:
                    next_progress += progress_every

    return outcomes, seat_outcomes, (pair_points / 2.0).tolist()


def confidence_interval(pair_scores):
    """Normal CI over independent seed-pair mean scores in [0, 1]."""

    values = np.asarray(pair_scores, dtype=np.float64)
    mean = float(values.mean())
    if len(values) < 2:
        return mean, mean
    margin = 1.96 * float(values.std(ddof=1)) / np.sqrt(len(values))
    return max(0.0, mean - margin), min(1.0, mean + margin)


def main():
    args = parse_args()
    if args.pairs <= 0:
        raise SystemExit("--pairs must be positive")
    if args.batch_size <= 0:
        raise SystemExit("--batch-size must be positive")
    if args.progress_every < 0:
        raise SystemExit("--progress-every must be non-negative")
    if args.opponent == "checkpoint" and args.opponent_checkpoint is None:
        raise SystemExit("--opponent-checkpoint is required")

    device = torch.device(
        args.device if torch.cuda.is_available() or args.device == "cpu" else "cpu"
    )
    candidate, checkpoint = load_policy(args.checkpoint, device)
    opponent_policy = None
    opponent_checkpoint = None
    if args.opponent == "checkpoint":
        opponent_policy, opponent_checkpoint = load_policy(
            args.opponent_checkpoint, device
        )

    master_rng = np.random.default_rng(args.seed)
    pair_seeds = master_rng.integers(0, 2**31 - 1, size=args.pairs)
    started = time.perf_counter()

    game_records = []
    if args.batch_size > 1:
        if args.opponent != "checkpoint":
            raise SystemExit(
                "--batch-size > 1 currently requires --opponent checkpoint"
            )
        if args.stochastic:
            raise SystemExit(
                "Batched paired evaluation is deterministic; omit --stochastic"
            )
        outcomes, seat_outcomes, pair_scores = (
            run_batched_checkpoint_evaluation(
                candidate,
                opponent_policy,
                pair_seeds,
                device,
                batch_size=args.batch_size,
                progress_every=args.progress_every,
                game_records=game_records,
            )
        )
    else:
        outcomes = Counter()
        seat_outcomes = {0: Counter(), 1: Counter()}
        pair_scores = []
        completed_games = 0
        for pair_index, match_seed in enumerate(pair_seeds):
            pair_points = 0.0
            for candidate_side in (0, 1):
                opponent_side = 1 - candidate_side
                env = ParallelSelfPlayEnv()
                observations = env.reset(seed=int(match_seed))
                candidate_memory = candidate.initial_memory(1, device=device)
                opponent_memory = (
                    opponent_policy.initial_memory(1, device=device)
                    if opponent_policy is not None
                    else None
                )
                action_rng = np.random.default_rng(
                    int(match_seed) ^ (0x45D9F3B + candidate_side)
                )
                guide = TurtleGuideTeacher(
                    np.random.default_rng(int(match_seed) ^ 0x9E3779B)
                )

                while True:
                    active = env.active_players()
                    actions = [None, None]
                    freeze_masks = [None, None]
                    if active[candidate_side]:
                        actions[candidate_side], candidate_memory, freeze_masks[candidate_side] = policy_action(
                        candidate,
                        observations[candidate_side],
                        candidate_memory,
                        device,
                        deterministic=not args.stochastic,
                    )
                    if active[opponent_side]:
                        if args.opponent == "guide":
                            action_count = (
                                env.core.shop_action_count_agent
                                if opponent_side == 0
                                else env.core.shop_action_count_opponent
                            )
                            actions[opponent_side] = guide.choose(
                                env.core,
                                opponent_side,
                                observations[opponent_side],
                                action_count,
                            ).action
                        elif args.opponent == "random":
                            candidates = legal_actions(
                                observations[opponent_side]
                            )
                            actions[opponent_side] = candidates[
                                int(action_rng.integers(len(candidates)))
                            ]
                        else:
                            actions[opponent_side], opponent_memory, freeze_masks[opponent_side] = policy_action(
                            opponent_policy,
                            observations[opponent_side],
                            opponent_memory,
                            device,
                            deterministic=not args.stochastic,
                        )
                    step = step_policy_actions(env, actions, freeze_masks)
                    observations = step.observations
                    if not step.terminated:
                        continue
                    result = candidate_result(
                        _completed_match_result(env.core), candidate_side
                    )
                    result_name = {0: "win", 1: "loss", 2: "draw"}[result]
                    game_records.append(evaluation_game_record(
                        env.core, pair_index, match_seed, candidate_side, result_name,
                    ))
                    outcomes[result_name] += 1
                    seat_outcomes[candidate_side][result_name] += 1
                    pair_points += (
                        1.0 if result == 0 else 0.5 if result == 2 else 0.0
                    )
                    completed_games += 1
                    if (
                        args.progress_every
                        and completed_games % args.progress_every == 0
                    ):
                        elapsed = time.perf_counter() - started
                        print(
                            f"progress games={completed_games}/{args.pairs * 2} "
                            f"games_per_second={completed_games / max(elapsed, 1e-9):.2f}",
                            flush=True,
                        )
                    break
            pair_scores.append(pair_points / 2.0)

    score = float(np.mean(pair_scores))
    ci_low, ci_high = confidence_interval(pair_scores)
    payload = {
        "evaluationRevision": "paired-seat-with-termination-v1-20260905",
        "terminationCounts": dict(Counter(record['terminationReason'] for record in game_records)),
        "gameResults": game_records,
        "checkpoint": str(args.checkpoint),
        "checkpointUpdate": checkpoint.get("update"),
        "checkpointTransitions": checkpoint.get("transitions"),
        "checkpointSimulatorRevision": checkpoint.get("simulator_revision"),
        "candidateActionSchema": END_FREEZE_SCHEMA if candidate.config.end_freeze else PRIMITIVE_SCHEMA,
        "opponentActionSchema": (END_FREEZE_SCHEMA if opponent_policy.config.end_freeze else PRIMITIVE_SCHEMA)
        if opponent_policy is not None else PRIMITIVE_SCHEMA,
        "runtimeSimulatorRevision": SIMULATOR_REVISION,
        "opponent": args.opponent,
        "opponentCheckpoint": (
            str(args.opponent_checkpoint)
            if args.opponent_checkpoint is not None
            else None
        ),
        "opponentCheckpointUpdate": (
            opponent_checkpoint.get("update")
            if opponent_checkpoint is not None
            else None
        ),
        "opponentCheckpointSimulatorRevision": (
            opponent_checkpoint.get("simulator_revision")
            if opponent_checkpoint is not None
            else None
        ),
        "guideVersion": guide.version if args.opponent == "guide" else None,
        "guideSha256": (
            guide.guide_sha256 if args.opponent == "guide" else None
        ),
        "pairs": args.pairs,
        "games": args.pairs * 2,
        "seed": args.seed,
        "pairSeeds": list(map(int, pair_seeds)),
        "outcomes": dict(outcomes),
        "seatOutcomes": {
            str(seat): dict(counts) for seat, counts in seat_outcomes.items()
        },
        "pairedMeanScore": score,
        "pairedScore95CI": [ci_low, ci_high],
        "pairScores": list(map(float, pair_scores)),
        "elapsedSeconds": time.perf_counter() - started,
        "batchSize": args.batch_size,
    }
    rendered = json.dumps(payload, indent=2, sort_keys=True)
    print(rendered)
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        write_evaluation_report(args.output, payload)


if __name__ == "__main__":
    main()
