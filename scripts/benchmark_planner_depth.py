#!/usr/bin/env python3
"""Measure whether selective nested planning materially beats root search."""

from __future__ import annotations

import argparse
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
)
from sapai_gym1.rl.opponent_reservoir import OpponentTrajectoryReservoir
from sapai_gym1.rl.game_trace import action_snapshot, view_snapshot
from sapai_gym1.rl.rollout_search import (
    SEARCH_VERSION,
    CounterfactualRolloutSearch,
    NeuralSearchEvaluator,
    PublicSearchState,
    RolloutSearchConfig,
    SelectiveNestedRolloutSearch,
)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("--opponent-reservoir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--states", type=int, default=16)
    parser.add_argument("--horizon-battles", type=int, default=5)
    parser.add_argument("--rollouts-per-action", type=int, default=4)
    parser.add_argument("--confirmation-rollouts", type=int, default=8)
    parser.add_argument("--validation-rollouts", type=int, default=16)
    parser.add_argument("--leaf-value-weight", type=float, default=0.25)
    parser.add_argument("--opponent-candidates", type=int, default=32)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--seed", type=int, default=20260902)
    return parser.parse_args()


def as_batch(observation, device):
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
            as_batch(observation, device),
            memory=memory.unsqueeze(0).to(device),
            deterministic=False,
        )
    return (
        AutoregressiveAction(*map(int, output.actions[0].cpu())),
        output.memory[0].float().cpu(),
    )


def collect_state(
    policy, reservoir, rng, device, target_turn, side, horizon, match_limit
):
    env = ParallelSelfPlayEnv()
    observations = env.reset(seed=int(rng.integers(0, 2**31 - 1)))
    memories = [policy.initial_memory(1, device="cpu")[0] for _ in range(2)]
    while True:
        active = env.active_players()
        if env.core.turn == target_turn and active[side]:
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
                current_turn=target_turn,
                horizon=horizon,
                limit=match_limit,
            )
            if matches:
                state.set_opponent_trajectories(
                    [match.teams for match in matches],
                    [match.trajectory_id for match in matches],
                )
                state.set_opponent_transition_model(
                    reservoir, candidate_limit=match_limit
                )
                return state, memories[side], len(matches)
        actions = [None, None]
        for acting_side in range(2):
            if not active[acting_side]:
                continue
            actions[acting_side], memories[acting_side] = act(
                policy,
                observations[acting_side],
                memories[acting_side],
                device,
            )
        step = env.step(actions[0], actions[1])
        observations = step.observations
        if step.terminated:
            return None


def planner_config(args, nested):
    return RolloutSearchConfig(
        rollouts_per_action=args.rollouts_per_action,
        confirmation_rollouts=args.confirmation_rollouts,
        confirmation_top_k=4,
        validation_rollouts=args.validation_rollouts,
        audit_rollouts=0,
        horizon_battles=args.horizon_battles,
        max_continuation_steps=180,
        minimum_improvement=0.0,
        confidence_z=1.96,
        exhaustive_root_probability=0.0,
        leaf_value_weight=args.leaf_value_weight,
        max_nested_searches=nested,
        nested_rollouts_per_action=2,
        nested_confirmation_rollouts=4,
        nested_confirmation_top_k=3,
        nested_lookahead_battles=2,
    )


def aggregate(rows, planner):
    selected = [row["planners"][planner] for row in rows]
    return {
        "states": len(selected),
        "meanOuterReturnAdvantageVsRaw": float(
            np.mean([item["outerReturnAdvantageVsRaw"] for item in selected])
        ),
        "meanOuterFirstBattleScoreAdvantageVsRaw": float(
            np.mean(
                [item["outerFirstBattleScoreAdvantageVsRaw"] for item in selected]
            )
        ),
        "meanOuterHorizonBattleScoreAdvantageVsRaw": float(
            np.mean(
                [
                    item["outerHorizonBattleScoreAdvantageVsRaw"]
                    for item in selected
                ]
            )
        ),
        "meanWallSeconds": float(
            np.mean([item["wallSeconds"] for item in selected])
        ),
        "meanContinuationStates": float(
            np.mean([item["compute"]["continuationStates"] for item in selected])
        ),
    }


def main():
    args = parse_args()
    if args.states <= 0:
        raise SystemExit("--states must be positive")
    torch.manual_seed(args.seed)
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

    sampled = []
    attempts = 0
    while len(sampled) < args.states:
        attempts += 1
        target_turn = int(rng.integers(2, 12))
        side = int(rng.integers(0, 2))
        state = collect_state(
            policy,
            reservoir,
            rng,
            device,
            target_turn,
            side,
            args.horizon_battles,
            args.opponent_candidates,
        )
        if state is not None:
            sampled.append((target_turn, side, *state))

    rows = []
    planner_names = ("root", "nested1", "nested2")
    for state_index, (turn, side, state, memory, match_count) in enumerate(sampled):
        row = {
            "state": state_index,
            "turn": turn,
            "side": side,
            "opponentMatches": match_count,
            "publicState": view_snapshot(state.core, 0),
            "planners": {},
        }
        common_seed = int(rng.integers(0, 2**63 - 1))
        for planner_name, nested in zip(planner_names, (0, 1, 2)):
            evaluator = NeuralSearchEvaluator(
                policy, device=device, chunk_size=args.batch_size
            )
            planner_class = (
                CounterfactualRolloutSearch
                if nested == 0
                else SelectiveNestedRolloutSearch
            )
            planner = planner_class(
                evaluator, planner_config(args, nested)
            )
            started = time.perf_counter()
            result = planner.search(state, memory, seed=common_seed)
            selected_action = (
                result.best_action
                if result.accepted
                else result.policy_action
            )
            deployed_return_advantage = (
                result.validation_return_improvement
                if result.accepted
                else 0.0
            )
            deployed_first_battle_advantage = (
                result.validation_first_battle_score_improvement
                if result.accepted
                else 0.0
            )
            deployed_horizon_advantage = (
                result.validation_horizon_battle_score_improvement
                if result.accepted
                else 0.0
            )
            row["planners"][planner_name] = {
                "action": list(selected_action.as_tuple()),
                "rawAction": list(result.policy_action.as_tuple()),
                "actionDetail": action_snapshot(
                    state.core.agent, selected_action
                ),
                "rawActionDetail": action_snapshot(
                    state.core.agent, result.policy_action
                ),
                "accepted": result.accepted,
                "proposedAction": list(result.best_action.as_tuple()),
                "proposedReturnAdvantageVsRaw": (
                    result.validation_return_improvement
                ),
                "proposedReturnLowerBound": (
                    result.validation_return_lower_bound
                ),
                "outerReturnAdvantageVsRaw": deployed_return_advantage,
                "outerReturnLowerBound": (
                    result.validation_return_lower_bound
                    if result.accepted
                    else 0.0
                ),
                "outerFirstBattleScoreAdvantageVsRaw": (
                    deployed_first_battle_advantage
                ),
                "outerHorizonBattleScoreAdvantageVsRaw": (
                    deployed_horizon_advantage
                ),
                "wallSeconds": time.perf_counter() - started,
                "compute": evaluator.metrics(),
                "meanNestedSearches": float(
                    np.mean(
                        [
                            trace.nested_searches
                            for traces in result.rollouts.values()
                            for trace in traces
                        ]
                    )
                ),
            }
        rows.append(row)
        print(json.dumps(row, sort_keys=True), flush=True)

    summaries = {
        planner: aggregate(rows, planner) for planner in planner_names
    }
    nested2_gain = (
        summaries["nested2"]["meanOuterHorizonBattleScoreAdvantageVsRaw"]
        - summaries["root"]["meanOuterHorizonBattleScoreAdvantageVsRaw"]
    )
    nested2_compute_ratio = (
        summaries["nested2"]["meanContinuationStates"]
        / max(1.0, summaries["root"]["meanContinuationStates"])
    )
    if nested2_gain >= 0.03 and nested2_compute_ratio <= 5.0:
        recommendation = "prototype-puct"
    elif nested2_gain <= 0.01:
        recommendation = "keep-root-search"
    else:
        recommendation = "expand-planner-ablation"
    payload = {
        "checkpoint": str(args.checkpoint),
        "searchVersion": SEARCH_VERSION,
        "simulatorRevision": SIMULATOR_REVISION,
        "plannerConfig": {
            "rolloutsPerAction": args.rollouts_per_action,
            "confirmationRollouts": args.confirmation_rollouts,
            "validationRollouts": args.validation_rollouts,
            "horizonBattles": args.horizon_battles,
            "leafValueWeight": args.leaf_value_weight,
            "opponentCandidates": args.opponent_candidates,
            "canonicalizeEarlyMoves": True,
            "pruneAffordableFreezes": False,
        },
        "states": args.states,
        "samplingAttempts": attempts,
        "summaries": summaries,
        "nested2MinusRootHorizonBattleScore": nested2_gain,
        "nested2ComputeRatio": nested2_compute_ratio,
        "recommendation": recommendation,
        "rows": rows,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print(json.dumps({
        "summaries": summaries,
        "nested2MinusRootHorizonBattleScore": nested2_gain,
        "nested2ComputeRatio": nested2_compute_ratio,
        "recommendation": recommendation,
    }, sort_keys=True))


if __name__ == "__main__":
    main()
