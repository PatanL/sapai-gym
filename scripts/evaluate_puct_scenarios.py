#!/usr/bin/env python3
"""Equal-budget root/PUCT evaluation on targeted SAP strategy scenarios."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import time

import numpy as np
import torch

from sapai import Team
from sapai.turtle_pack import SIMULATOR_REVISION
from sapai_gym1.rl import EntityTransformerConfig, EntityTransformerPolicy
from sapai_gym1.rl.curriculum import BUILDERS, SCENARIO_NAMES
from sapai_gym1.rl.game_trace import action_snapshot, view_snapshot
from sapai_gym1.rl.opponent_reservoir import OpponentTrajectoryReservoir
from sapai_gym1.rl.puct_search import (
    PUCT_VERSION,
    PUCTConfig,
    StochasticPUCTSearch,
)
from sapai_gym1.rl.rollout_search import (
    SEARCH_VERSION,
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
    parser.add_argument("--scenario", action="append", choices=SCENARIO_NAMES)
    parser.add_argument("--examples", type=int, default=1)
    parser.add_argument("--validation-rollouts", type=int, default=32)
    parser.add_argument("--horizon-battles", type=int, default=5)
    parser.add_argument("--leaf-value-weight", type=float, default=0.5)
    parser.add_argument("--opponent-candidates", type=int, default=32)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--cpuct", type=float, default=1.5)
    parser.add_argument("--prior-floor", type=float, default=0.02)
    parser.add_argument("--root-noise-fraction", type=float, default=0.0)
    parser.add_argument("--min-root-visits", type=int, default=1)
    parser.add_argument("--seed", type=int, default=2720001)
    return parser.parse_args()


def attach_opponent_model(state, reservoir, horizon, limit):
    matches = reservoir.match(
        state.core.previous_opponent_state,
        observed_enemy=state.core.previous_agent_state,
        older_observed=state.core.previous_previous_opponent_state,
        older_observed_enemy=state.core.previous_previous_agent_state,
        current_turn=state.core.turn,
        horizon=horizon,
        limit=limit,
    )
    if matches:
        state.set_opponent_trajectories(
            [match.teams for match in matches],
            [match.trajectory_id for match in matches],
        )
        state.set_opponent_transition_model(
            reservoir, candidate_limit=limit
        )
    return len(matches)


def main():
    args = parse_args()
    if args.examples <= 0:
        raise SystemExit("--examples must be positive")
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
    policy = EntityTransformerPolicy(
        EntityTransformerConfig(**checkpoint["config"])
    ).to(device)
    policy.load_state_dict(checkpoint["model"])
    policy.eval()
    reservoir = OpponentTrajectoryReservoir.load(args.opponent_reservoir)
    scenarios = args.scenario or list(SCENARIO_NAMES)
    rows = []

    for scenario in scenarios:
        for example in range(args.examples):
            env, teacher_actions = BUILDERS[scenario](rng)
            # Scenario builders define the current submitted team. Use it as
            # the opponent's best available estimate of our previous board.
            if env.previous_agent_state is None:
                env.previous_agent_state = Team.from_state(
                    env.agent.team.state
                )
            state = PublicSearchState.from_core(env, 0)
            match_count = attach_opponent_model(
                state,
                reservoir,
                args.horizon_battles,
                args.opponent_candidates,
            )
            memory = policy.initial_memory(1, device="cpu")[0]
            search_seed = int(rng.integers(0, 2**63 - 1))
            validation_seed = int(rng.integers(0, 2**63 - 1))

            root_evaluator = NeuralSearchEvaluator(
                policy, device=device, chunk_size=args.batch_size
            )
            root_planner = CounterfactualRolloutSearch(
                root_evaluator,
                RolloutSearchConfig(
                    rollouts_per_action=4,
                    confirmation_rollouts=16,
                    confirmation_top_k=4,
                    validation_rollouts=args.validation_rollouts,
                    audit_rollouts=0,
                    horizon_battles=args.horizon_battles,
                    max_continuation_steps=180,
                    minimum_improvement=0.0,
                    confidence_z=1.96,
                    exhaustive_root_probability=0.0,
                    leaf_value_weight=args.leaf_value_weight,
                    max_nested_searches=0,
                    prune_affordable_freezes=False,
                ),
            )
            started = time.perf_counter()
            root = root_planner.search(
                state,
                memory,
                seed=search_seed,
                validation_seed=validation_seed,
            )
            root_wall = time.perf_counter() - started
            budget = root.selection_simulator_transitions

            puct_evaluator = NeuralSearchEvaluator(
                policy, device=device, chunk_size=args.batch_size
            )
            puct_planner = StochasticPUCTSearch(
                puct_evaluator,
                PUCTConfig(
                    max_simulations=4096,
                    max_simulator_transitions=budget,
                    max_rollout_steps=180,
                    horizon_battles=args.horizon_battles,
                    world_pool_size=32,
                    min_root_visits=args.min_root_visits,
                    cpuct=args.cpuct,
                    prior_floor=args.prior_floor,
                    root_noise_fraction=args.root_noise_fraction,
                    leaf_value_weight=args.leaf_value_weight,
                    validation_rollouts=args.validation_rollouts,
                    confidence_z=1.96,
                    minimum_improvement=0.0,
                    prune_affordable_freezes=False,
                ),
            )
            started = time.perf_counter()
            puct = puct_planner.search(
                state,
                memory,
                seed=search_seed,
                validation_seed=validation_seed,
            )
            puct_wall = time.perf_counter() - started
            expected = teacher_actions[0]
            root_selected = (
                root.best_action if root.accepted else root.policy_action
            )
            root_target = next(
                (
                    target
                    for target in puct.targets
                    if expected in target.actions
                    and target.visits
                    == sum(puct.visit_counts.values())
                ),
                None,
            )
            row = {
                "scenario": scenario,
                "example": example,
                "turn": int(state.core.turn),
                "opponentMatches": match_count,
                "publicState": view_snapshot(state.core, 0),
                "expectedAction": list(expected.as_tuple()),
                "expectedActionDetail": action_snapshot(
                    state.core.agent, expected
                ),
                "rawAction": list(root.policy_action.as_tuple()),
                "rawMatchesExpected": root.policy_action == expected,
                "selectionTransitionBudget": budget,
                "root": {
                    "accepted": root.accepted,
                    "proposedAction": list(root.best_action.as_tuple()),
                    "selectedAction": list(root_selected.as_tuple()),
                    "selectedMatchesExpected": root_selected == expected,
                    "validationReturnImprovement": (
                        root.validation_return_improvement
                    ),
                    "validationReturnLowerBound": (
                        root.validation_return_lower_bound
                    ),
                    "wallSeconds": root_wall,
                },
                "puct": {
                    "cpuct": args.cpuct,
                    "priorFloor": args.prior_floor,
                    "rootNoiseFraction": args.root_noise_fraction,
                    "minRootVisits": args.min_root_visits,
                    "accepted": puct.accepted,
                    "proposedAction": list(puct.proposed_action.as_tuple()),
                    "selectedAction": list(puct.selected_action.as_tuple()),
                    "selectedMatchesExpected": (
                        puct.selected_action == expected
                    ),
                    "validationReturnImprovement": (
                        puct.validation_return_improvement
                    ),
                    "validationReturnLowerBound": (
                        puct.validation_return_lower_bound
                    ),
                    "validationRolloutsUsed": (
                        puct.validation_rollouts_used
                    ),
                    "outerAuditReturnImprovement": (
                        puct.audit_return_improvement
                    ),
                    "outerAuditReturnLowerBound": (
                        puct.audit_return_lower_bound
                    ),
                    "validationHorizonBattleScoreImprovement": (
                        puct.validation_horizon_battle_score_improvement
                    ),
                    "simulations": puct.simulations,
                    "simulatorTransitions": puct.simulator_transitions,
                    "treeNodes": puct.tree_nodes,
                    "trainingTargets": len(puct.targets),
                    "expectedRootTargetProbability": (
                        None
                        if root_target is None
                        else root_target.probabilities[
                            root_target.actions.index(expected)
                        ]
                    ),
                    "wallSeconds": puct_wall,
                },
            }
            rows.append(row)
            print(json.dumps(row, sort_keys=True), flush=True)

    summary = {
        "examples": len(rows),
        "rawMatchesExpected": sum(row["rawMatchesExpected"] for row in rows),
        "rootMatchesExpected": sum(
            row["root"]["selectedMatchesExpected"] for row in rows
        ),
        "puctMatchesExpected": sum(
            row["puct"]["selectedMatchesExpected"] for row in rows
        ),
        "rootAccepted": sum(row["root"]["accepted"] for row in rows),
        "puctAccepted": sum(row["puct"]["accepted"] for row in rows),
        "meanRootWallSeconds": float(
            np.mean([row["root"]["wallSeconds"] for row in rows])
        ),
        "meanPuctWallSeconds": float(
            np.mean([row["puct"]["wallSeconds"] for row in rows])
        ),
    }
    payload = {
        "puctVersion": PUCT_VERSION,
        "searchVersion": SEARCH_VERSION,
        "simulatorRevision": SIMULATOR_REVISION,
        "checkpoint": str(args.checkpoint),
        "summary": summary,
        "rows": rows,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print(json.dumps(summary, sort_keys=True))


if __name__ == "__main__":
    main()
