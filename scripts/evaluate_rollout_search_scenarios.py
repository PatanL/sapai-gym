#!/usr/bin/env python3
"""Probe counterfactual search on versioned targeted scenario states."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import time

import numpy as np
import torch

from sapai.turtle_pack import SIMULATOR_REVISION
from sapai_gym1.rl import EntityTransformerConfig, EntityTransformerPolicy
from sapai_gym1.rl.curriculum import BUILDERS, SCENARIO_NAMES
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
    parser.add_argument("--scenario", action="append", choices=SCENARIO_NAMES)
    parser.add_argument("--examples", type=int, default=4)
    parser.add_argument("--rollouts-per-action", type=int, default=4)
    parser.add_argument("--confirmation-rollouts", type=int, default=16)
    parser.add_argument("--confirmation-top-k", type=int, default=4)
    parser.add_argument("--validation-rollouts", type=int, default=32)
    parser.add_argument("--horizon-battles", type=int, default=5)
    parser.add_argument("--max-continuation-steps", type=int, default=60)
    parser.add_argument("--minimum-improvement", type=float, default=0.02)
    parser.add_argument("--confidence-z", type=float, default=1.96)
    parser.add_argument("--exhaustive-root-probability", type=float, default=0.10)
    parser.add_argument("--leaf-value-weight", type=float, default=1.0)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seed", type=int, default=20260902)
    parser.add_argument("--allow-simulator-mismatch", action="store_true")
    parser.add_argument("--output", type=Path)
    return parser.parse_args()


def action_tuple(action):
    return list(action.as_tuple())


def main():
    args = parse_args()
    if args.examples <= 0:
        raise SystemExit("--examples must be positive")
    device = torch.device(
        args.device
        if torch.cuda.is_available() or args.device == "cpu"
        else "cpu"
    )
    checkpoint = torch.load(
        args.checkpoint, map_location="cpu", weights_only=False
    )
    checkpoint_revision = checkpoint.get("simulator_revision")
    if (
        checkpoint_revision != SIMULATOR_REVISION
        and not args.allow_simulator_mismatch
    ):
        raise SystemExit(
            "Checkpoint simulator revision mismatch: "
            f"checkpoint={checkpoint_revision!r}, runtime={SIMULATOR_REVISION!r}"
        )
    policy = EntityTransformerPolicy(
        EntityTransformerConfig(**checkpoint["config"])
    ).to(device)
    policy.load_state_dict(checkpoint["model"])
    policy.eval()
    evaluator = NeuralSearchEvaluator(
        policy, device=device, chunk_size=args.batch_size
    )
    search = CounterfactualRolloutSearch(
        evaluator,
        RolloutSearchConfig(
            rollouts_per_action=args.rollouts_per_action,
            confirmation_rollouts=args.confirmation_rollouts,
            confirmation_top_k=args.confirmation_top_k,
            validation_rollouts=args.validation_rollouts,
            horizon_battles=args.horizon_battles,
            max_continuation_steps=args.max_continuation_steps,
            minimum_improvement=args.minimum_improvement,
            confidence_z=args.confidence_z,
            exhaustive_root_probability=args.exhaustive_root_probability,
            leaf_value_weight=args.leaf_value_weight,
        ),
    )
    rng = np.random.default_rng(args.seed)
    scenarios = args.scenario or list(SCENARIO_NAMES)
    rows = []
    started = time.perf_counter()
    for scenario in scenarios:
        for example_index in range(args.examples):
            env, teacher_actions = BUILDERS[scenario](rng)
            state = PublicSearchState.from_core(env, 0)
            result = search.search(
                state,
                policy.initial_memory(1, device="cpu")[0],
                seed=int(rng.integers(0, 2**63 - 1)),
            )
            expected = teacher_actions[0]
            row = {
                "scenario": scenario,
                "example": example_index,
                "expected": action_tuple(expected),
                "policy": action_tuple(result.policy_action),
                "best": action_tuple(result.best_action),
                "matchesExpected": result.best_action == expected,
                "changedPolicy": result.best_action != result.policy_action,
                "accepted": result.accepted,
                "improvement": result.improvement,
                "improvementStandardError": result.improvement_standard_error,
                "improvementLowerBound": result.improvement_lower_bound,
                "validationReturnImprovement": (
                    result.validation_return_improvement
                ),
                "validationReturnLowerBound": (
                    result.validation_return_lower_bound
                ),
                "validationHorizonBattleScoreImprovement": (
                    result.validation_horizon_battle_score_improvement
                ),
                "candidateCount": len(result.scores),
                "screeningCandidateCount": len(result.screening_scores),
                "exhaustiveRoot": result.exhaustive_root,
                "screeningBestExcludedByPruning": (
                    result.screening_best_excluded_by_pruning
                ),
                "topCandidates": [
                    {
                        "action": action_tuple(action),
                        "score": score,
                        "prior": result.priors[action],
                    }
                    for action, score in sorted(
                        result.scores.items(),
                        key=lambda item: item[1],
                        reverse=True,
                    )[:5]
                ],
                "bestRollouts": [
                    {
                        "actions": [
                            action_tuple(action) for action in trace.actions
                        ],
                        "rewards": list(trace.rewards),
                        "battleResults": list(trace.battle_results),
                        "leafValue": trace.leaf_value,
                        "totalReturn": trace.total_return,
                    }
                    for trace in result.rollouts[result.best_action]
                ],
            }
            rows.append(row)
            print(json.dumps(row, sort_keys=True), flush=True)
    payload = {
        "searchVersion": SEARCH_VERSION,
        "simulatorRevision": SIMULATOR_REVISION,
        "checkpoint": str(args.checkpoint),
        "checkpointRevision": checkpoint_revision,
        "rows": rows,
        "summary": {
            "examples": len(rows),
            "matchesExpected": sum(row["matchesExpected"] for row in rows),
            "changedPolicy": sum(row["changedPolicy"] for row in rows),
            "accepted": sum(row["accepted"] for row in rows),
            "elapsedSeconds": time.perf_counter() - started,
        },
    }
    print(json.dumps(payload["summary"], sort_keys=True))
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")


if __name__ == "__main__":
    main()
