#!/usr/bin/env python3
"""Run a bounded stochastic-PUCT smoke/evaluation on real SAP states."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import time

import numpy as np
import torch

from benchmark_planner_depth import collect_state
from sapai.turtle_pack import SIMULATOR_REVISION
from sapai_gym1.rl import EntityTransformerConfig, EntityTransformerPolicy
from sapai_gym1.rl.game_trace import action_snapshot, view_snapshot
from sapai_gym1.rl.opponent_reservoir import OpponentTrajectoryReservoir
from sapai_gym1.rl.puct_search import (
    PUCT_VERSION,
    PUCTConfig,
    StochasticPUCTSearch,
)
from sapai_gym1.rl.rollout_search import NeuralSearchEvaluator


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("--opponent-reservoir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--states", type=int, default=1)
    parser.add_argument("--target-turn", type=int, default=3)
    parser.add_argument("--transition-budget", type=int, default=2_000)
    parser.add_argument("--max-simulations", type=int, default=256)
    parser.add_argument("--validation-rollouts", type=int, default=8)
    parser.add_argument("--horizon-battles", type=int, default=5)
    parser.add_argument("--leaf-value-weight", type=float, default=0.5)
    parser.add_argument("--opponent-candidates", type=int, default=32)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--seed", type=int, default=2701001)
    return parser.parse_args()


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
    policy = EntityTransformerPolicy(
        EntityTransformerConfig(**checkpoint["config"])
    ).to(device)
    policy.load_state_dict(checkpoint["model"])
    policy.eval()
    reservoir = OpponentTrajectoryReservoir.load(args.opponent_reservoir)
    if reservoir.metadata.get("simulatorRevision") != SIMULATOR_REVISION:
        raise SystemExit("Opponent reservoir simulator revision mismatch")

    rows = []
    attempts = 0
    while len(rows) < args.states:
        attempts += 1
        side = int(rng.integers(0, 2))
        collected = collect_state(
            policy,
            reservoir,
            rng,
            device,
            args.target_turn,
            side,
            args.horizon_battles,
            args.opponent_candidates,
        )
        if collected is None:
            continue
        state, memory, match_count = collected
        evaluator = NeuralSearchEvaluator(
            policy, device=device, chunk_size=args.batch_size
        )
        planner = StochasticPUCTSearch(
            evaluator,
            PUCTConfig(
                max_simulations=args.max_simulations,
                max_simulator_transitions=args.transition_budget,
                horizon_battles=args.horizon_battles,
                validation_rollouts=args.validation_rollouts,
                leaf_value_weight=args.leaf_value_weight,
            ),
        )
        search_seed = int(rng.integers(0, 2**63 - 1))
        validation_seed = int(rng.integers(0, 2**63 - 1))
        started = time.perf_counter()
        result = planner.search(
            state,
            memory,
            seed=search_seed,
            validation_seed=validation_seed,
        )
        root_actions = sorted(
            result.visit_counts,
            key=lambda action: (
                result.q_lower_bounds[action],
                result.q_values[action],
            ),
            reverse=True,
        )
        row = {
            "state": len(rows),
            "turn": args.target_turn,
            "side": side,
            "opponentMatches": match_count,
            "publicState": view_snapshot(state.core, 0),
            "policyAction": list(result.policy_action.as_tuple()),
            "policyActionDetail": action_snapshot(
                state.core.agent, result.policy_action
            ),
            "proposedAction": list(result.proposed_action.as_tuple()),
            "proposedActionDetail": action_snapshot(
                state.core.agent, result.proposed_action
            ),
            "selectedAction": list(result.selected_action.as_tuple()),
            "accepted": result.accepted,
            "simulations": result.simulations,
            "simulatorTransitions": result.simulator_transitions,
            "treeNodes": result.tree_nodes,
            "trainingTargets": len(result.targets),
            "validationReturnImprovement": (
                result.validation_return_improvement
            ),
            "validationReturnLowerBound": (
                result.validation_return_lower_bound
            ),
            "validationRolloutsUsed": result.validation_rollouts_used,
            "outerAuditReturnImprovement": (
                result.audit_return_improvement
            ),
            "outerAuditReturnLowerBound": result.audit_return_lower_bound,
            "validationFirstBattleScoreImprovement": (
                result.validation_first_battle_score_improvement
            ),
            "validationHorizonBattleScoreImprovement": (
                result.validation_horizon_battle_score_improvement
            ),
            "wallSeconds": time.perf_counter() - started,
            "evaluator": evaluator.metrics(),
            "root": [
                {
                    "action": list(action.as_tuple()),
                    "description": action_snapshot(
                        state.core.agent, action
                    )["description"],
                    "prior": result.priors[action],
                    "visits": result.visit_counts[action],
                    "q": result.q_values[action],
                    "qStandardError": result.q_standard_errors[action],
                    "qLowerBound": result.q_lower_bounds[action],
                }
                for action in root_actions[:10]
            ],
        }
        rows.append(row)
        print(json.dumps(row, sort_keys=True), flush=True)
    payload = {
        "puctVersion": PUCT_VERSION,
        "simulatorRevision": SIMULATOR_REVISION,
        "checkpoint": str(args.checkpoint),
        "states": len(rows),
        "samplingAttempts": attempts,
        "config": vars(args) | {
            "checkpoint": str(args.checkpoint),
            "opponent_reservoir": str(args.opponent_reservoir),
            "output": str(args.output),
        },
        "rows": rows,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")


if __name__ == "__main__":
    main()
