#!/usr/bin/env python3
"""Compare critic predictions with repeated long public-information rollouts."""

from __future__ import annotations

import argparse
from contextlib import nullcontext
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
from sapai_gym1.rl.opponent_reservoir import OpponentTrajectoryReservoir
from sapai_gym1.rl.rollout_search import (
    CounterfactualRolloutSearch,
    NeuralSearchEvaluator,
    PublicSearchState,
    RolloutSearchConfig,
)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("--opponent-reservoir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--states", type=int, default=128)
    parser.add_argument("--rollouts", type=int, default=32)
    parser.add_argument("--horizon-battles", type=int, default=10)
    parser.add_argument("--max-continuation-steps", type=int, default=500)
    parser.add_argument("--opponent-candidates", type=int, default=32)
    parser.add_argument("--state-probability", type=float, default=0.15)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--seed", type=int, default=20260902)
    return parser.parse_args()


def tensor_observation(observation, device):
    return {
        key: torch.as_tensor(value, device=device).unsqueeze(0)
        for key, value in observation.items()
    }


@torch.inference_mode()
def act(policy, observation, memory, device):
    autocast = (
        torch.autocast("cuda", dtype=torch.bfloat16)
        if device.type == "cuda"
        else nullcontext()
    )
    with autocast:
        output = policy(
            tensor_observation(observation, device),
            memory=memory.unsqueeze(0).to(device),
            deterministic=False,
        )
    return (
        AutoregressiveAction(*map(int, output.actions[0].cpu())),
        output.memory[0].float().cpu(),
    )


def metrics(rows):
    predicted = np.asarray([row["predicted"] for row in rows])
    observed = np.asarray([row["monteCarlo"] for row in rows])
    errors = predicted - observed
    correlation = (
        0.0
        if len(rows) < 2
        or predicted.std() == 0
        or observed.std() == 0
        else float(np.corrcoef(predicted, observed)[0, 1])
    )
    return {
        "states": len(rows),
        "correlation": correlation,
        "mae": float(np.mean(np.abs(errors))),
        "rmse": float(np.sqrt(np.mean(errors**2))),
        "bias": float(np.mean(errors)),
        "meanPrediction": float(predicted.mean()),
        "meanMonteCarlo": float(observed.mean()),
    }


def recommended_leaf_weight(summary, slices=()):
    candidates = [summary]
    candidates.extend(
        value for value in slices if value.get("states", 0) >= 20
    )
    return min(_recommended_leaf_weight(value) for value in candidates)


def _recommended_leaf_weight(summary):
    if (
        summary["correlation"] >= 0.50
        and summary["mae"] <= 0.75
        and abs(summary["bias"]) <= 0.25
    ):
        return 1.0
    if (
        summary["correlation"] >= 0.25
        and summary["mae"] <= 1.10
        and abs(summary["bias"]) <= 0.50
    ):
        return 0.5
    return 0.25


def calibration_rollout_config(args):
    """Build the actual Monte Carlo rollout config from the CLI request."""

    return RolloutSearchConfig(
        horizon_battles=args.horizon_battles,
        max_continuation_steps=args.max_continuation_steps,
        leaf_value_weight=0.0,
    )


def main():
    args = parse_args()
    if args.states <= 0 or args.rollouts < 2:
        raise SystemExit("--states must be positive and --rollouts at least two")
    if not 0.0 < args.state_probability <= 1.0:
        raise SystemExit("--state-probability must be in (0, 1]")
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    rng = np.random.default_rng(args.seed)
    device = torch.device(
        args.device
        if torch.cuda.is_available() or args.device == "cpu"
        else "cpu"
    )
    checkpoint = torch.load(
        args.checkpoint, map_location="cpu", weights_only=False
    )
    if checkpoint.get("simulator_revision") != SIMULATOR_REVISION:
        raise SystemExit("Checkpoint simulator revision mismatch")
    reservoir = OpponentTrajectoryReservoir.load(args.opponent_reservoir)
    if reservoir.metadata.get("simulatorRevision") != SIMULATOR_REVISION:
        raise SystemExit("Opponent reservoir simulator revision mismatch")
    policy = EntityTransformerPolicy(
        EntityTransformerConfig(**checkpoint["config"])
    ).to(device)
    policy.load_state_dict(checkpoint["model"])
    policy.eval()
    evaluator = NeuralSearchEvaluator(
        policy, device=device, chunk_size=args.batch_size
    )
    rollout = CounterfactualRolloutSearch(
        evaluator,
        calibration_rollout_config(args),
    )

    rows = []
    game_index = 0
    while len(rows) < args.states:
        env = ParallelSelfPlayEnv()
        observations = env.reset(seed=int(rng.integers(0, 2**31 - 1)))
        memories = [policy.initial_memory(1, device="cpu")[0] for _ in range(2)]
        while True:
            active = env.active_players()
            actions = [None, None]
            for side in range(2):
                if not active[side]:
                    continue
                if (
                    len(rows) < args.states
                    and env.core.turn >= 2
                    and rng.random() < args.state_probability
                ):
                    state = PublicSearchState.from_core(env.core, side)
                    latest = (
                        env.core.previous_opponent_state
                        if side == 0
                        else env.core.previous_agent_state
                    )
                    observed_by_opponent = (
                        env.core.previous_agent_state
                        if side == 0
                        else env.core.previous_opponent_state
                    )
                    older_latest = (
                        env.core.previous_previous_opponent_state
                        if side == 0
                        else env.core.previous_previous_agent_state
                    )
                    older_observed_by_opponent = (
                        env.core.previous_previous_agent_state
                        if side == 0
                        else env.core.previous_previous_opponent_state
                    )
                    matches = reservoir.match(
                        latest,
                        observed_enemy=observed_by_opponent,
                        older_observed=older_latest,
                        older_observed_enemy=older_observed_by_opponent,
                        current_turn=env.core.turn,
                        horizon=args.horizon_battles,
                        limit=args.opponent_candidates,
                    )
                    if matches:
                        state.set_opponent_trajectories(
                            [match.teams for match in matches],
                            [match.trajectory_id for match in matches],
                        )
                        state.set_opponent_transition_model(
                            reservoir,
                            candidate_limit=args.opponent_candidates,
                        )
                        prediction = evaluator.evaluate(
                            state, memories[side]
                        ).value
                        estimate = rollout.rollout_state_value(
                            state,
                            memories[side],
                            rollouts=args.rollouts,
                            horizon_battles=args.horizon_battles,
                            seed=int(rng.integers(0, 2**63 - 1)),
                            deterministic=False,
                        )
                        own = state.core.agent
                        opponent = state.core.opponent
                        score = (own.lives, own.wins)
                        opponent_score = (opponent.lives, opponent.wins)
                        rows.append(
                            {
                                "game": game_index,
                                "side": side,
                                "turn": env.core.turn,
                                "position": (
                                    "ahead"
                                    if score > opponent_score
                                    else "behind"
                                    if score < opponent_score
                                    else "even"
                                ),
                                "predicted": prediction,
                                "monteCarlo": estimate.mean,
                                "monteCarloStandardError": (
                                    estimate.standard_error
                                ),
                                "opponentMatches": len(matches),
                            }
                        )
                        print(
                            f"state={len(rows)}/{args.states} "
                            f"turn={env.core.turn} predicted={prediction:.3f} "
                            f"mc={estimate.mean:.3f}",
                            flush=True,
                        )
                actions[side], memories[side] = act(
                    policy,
                    observations[side],
                    memories[side],
                    device,
                )
            step = env.step(actions[0], actions[1])
            observations = step.observations
            if step.terminated:
                break
        game_index += 1

    overall = metrics(rows)
    by_turn = {
        str(turn): metrics([row for row in rows if row["turn"] == turn])
        for turn in sorted({row["turn"] for row in rows})
    }
    by_position = {
        position: metrics(
            [row for row in rows if row["position"] == position]
        )
        for position in ("ahead", "even", "behind")
        if any(row["position"] == position for row in rows)
    }
    bands = {
        "early": (2, 4),
        "mid": (5, 8),
        "late": (9, 25),
    }
    by_band = {
        name: metrics(
            [row for row in rows if lower <= row["turn"] <= upper]
        )
        for name, (lower, upper) in bands.items()
        if any(lower <= row["turn"] <= upper for row in rows)
    }
    slice_recommendations = {
        f"band:{name}": _recommended_leaf_weight(value)
        for name, value in by_band.items()
        if value["states"] >= 20
    }
    slice_recommendations.update(
        {
            f"position:{name}": _recommended_leaf_weight(value)
            for name, value in by_position.items()
            if value["states"] >= 20
        }
    )
    leaf_weight = recommended_leaf_weight(
        overall, [*by_band.values(), *by_position.values()]
    )
    payload = {
        "checkpoint": str(args.checkpoint),
        "simulatorRevision": SIMULATOR_REVISION,
        "states": args.states,
        "rolloutsPerState": args.rollouts,
        "horizonBattles": args.horizon_battles,
        "overall": overall,
        "byTurn": by_turn,
        "byBand": by_band,
        "byPosition": by_position,
        "sliceRecommendations": slice_recommendations,
        "recommendedLeafValueWeight": leaf_weight,
        "rows": rows,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print(json.dumps({
        "overall": overall,
        "recommendedLeafValueWeight": payload["recommendedLeafValueWeight"],
    }, sort_keys=True))


if __name__ == "__main__":
    main()
