#!/usr/bin/env python3
"""Generate audited PUCT targets for reproducible raw-policy scenario failures."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import time

import numpy as np
import torch

from generate_puct_demonstrations import (
    audit_record,
    filtered_target_actions,
    independently_confirmed,
    mean_ci,
    puct_example,
)
from generate_search_demonstrations import load_policy
from sapai import Team
from sapai.turtle_pack import SIMULATOR_REVISION
from sapai_gym1.rl.curriculum import BUILDERS, PUCT_SCENARIO_VERSION
from sapai_gym1.rl.opponent_reservoir import OpponentTrajectoryReservoir
from sapai_gym1.rl.puct_search import (
    PUCT_VERSION,
    PUCTConfig,
    StochasticPUCTSearch,
)
from sapai_gym1.rl.rollout_search import NeuralSearchEvaluator, PublicSearchState


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("failures", type=Path)
    parser.add_argument("--opponent-reservoir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--holdout-output", type=Path, required=True)
    parser.add_argument("--audit-output", type=Path, required=True)
    parser.add_argument("--summary-output", type=Path, required=True)
    parser.add_argument("--holdout-fraction", type=float, default=0.20)
    parser.add_argument("--transition-budget", type=int, default=7_500)
    parser.add_argument("--max-simulations", type=int, default=4_096)
    parser.add_argument("--min-root-visits", type=int, default=4)
    parser.add_argument("--horizon-battles", type=int, default=5)
    parser.add_argument("--validation-rollouts", type=int, default=32)
    parser.add_argument("--max-validation-rollouts", type=int, default=512)
    parser.add_argument("--audit-rollouts", type=int, default=256)
    parser.add_argument("--leaf-value-weight", type=float, default=0.5)
    parser.add_argument(
        "--exhaustive-root-probability", type=float, default=1.0
    )
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--seed", type=int, default=2760001)
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument("--num-shards", type=int, default=1)
    return parser.parse_args()


def main():
    args = parse_args()
    if args.num_shards <= 0 or not 0 <= args.shard_index < args.num_shards:
        raise SystemExit("invalid shard index/count")
    device = torch.device(
        args.device
        if torch.cuda.is_available() or args.device == "cpu"
        else "cpu"
    )
    policy, checkpoint_payload = load_policy(args.checkpoint, device, False)
    checkpoint_hash = hashlib.sha256(args.checkpoint.read_bytes()).hexdigest()
    evaluator = NeuralSearchEvaluator(
        policy, device=device, chunk_size=args.batch_size
    )
    planner = StochasticPUCTSearch(
        evaluator,
        PUCTConfig(
            max_simulations=args.max_simulations,
            max_simulator_transitions=args.transition_budget,
            max_rollout_steps=180,
            horizon_battles=args.horizon_battles,
            world_pool_size=32,
            min_root_visits=args.min_root_visits,
            cpuct=2.0,
            prior_floor=0.10,
            leaf_value_weight=args.leaf_value_weight,
            validation_rollouts=args.validation_rollouts,
            max_validation_rollouts=args.max_validation_rollouts,
            audit_rollouts=args.audit_rollouts,
            confidence_z=1.96,
            minimum_improvement=0.0,
            prune_affordable_freezes=False,
            exhaustive_root_probability=args.exhaustive_root_probability,
        ),
    )
    reservoir = OpponentTrajectoryReservoir.load(args.opponent_reservoir)
    failure_payload = json.loads(args.failures.read_text())
    scenario_version = failure_payload.get(
        "scenarioVersion", PUCT_SCENARIO_VERSION
    )
    all_failure_rows = failure_payload["rows"]
    ordered_failures = sorted(
        all_failure_rows,
        key=lambda row: hashlib.sha256(
            f"{row['scenario']}:{row['seed']}".encode()
        ).hexdigest(),
    )
    failure_rows = ordered_failures[
        args.shard_index :: args.num_shards
    ]
    rng = np.random.default_rng(args.seed)
    for path in (
        args.output,
        args.holdout_output,
        args.audit_output,
        args.summary_output,
    ):
        path.parent.mkdir(parents=True, exist_ok=True)
    train_stream = args.output.open("w")
    holdout_stream = args.holdout_output.open("w")
    audit_stream = args.audit_output.open("w")
    accepted = 0
    deployed_accepted = 0
    candidates = 0
    train_examples = 0
    holdout_examples = 0
    root_target_nodes = 0
    internal_target_nodes = 0
    validation_values = []
    audit_values = []
    scenario_results = []

    for index, failure in enumerate(failure_rows):
        scenario = failure["scenario"]
        scenario_seed = int(failure["seed"])
        scenario_rng = np.random.default_rng(scenario_seed)
        env, expected_actions = BUILDERS[scenario](scenario_rng)
        if env.previous_agent_state is None:
            env.previous_agent_state = Team.from_state(env.agent.team.state)
        state = PublicSearchState.from_core(env, 0)
        matches = reservoir.match(
            state.core.previous_opponent_state,
            observed_enemy=state.core.previous_agent_state,
            older_observed=state.core.previous_previous_opponent_state,
            older_observed_enemy=state.core.previous_previous_agent_state,
            current_turn=state.core.turn,
            horizon=args.horizon_battles,
            limit=32,
        )
        state.set_opponent_trajectories(
            [match.teams for match in matches],
            [match.trajectory_id for match in matches],
        )
        state.set_opponent_transition_model(reservoir, candidate_limit=32)
        memory = policy.initial_memory(1, device="cpu")[0]
        started = time.perf_counter()
        result = planner.search(
            state,
            memory,
            seed=int(rng.integers(0, 2**63 - 1)),
            validation_seed=int(rng.integers(0, 2**63 - 1)),
        )
        elapsed = time.perf_counter() - started
        split = (
            "holdout"
            if rng.random() < args.holdout_fraction
            else "train"
        )
        audit = audit_record(
            result,
            state,
            game=index,
            side=0,
            turn=state.core.turn,
            reasons=[f"scenario:{scenario}", "raw-policy-failure"],
            split=split,
            elapsed=elapsed,
        )
        audit.update(
            {
                "scenario": scenario,
                "scenarioSeed": scenario_seed,
                "expectedAction": list(expected_actions[0].as_tuple()),
                "searchMatchesExpected": (
                    result.proposed_action == expected_actions[0]
                ),
            }
        )
        audit_stream.write(
            json.dumps(audit, separators=(",", ":")) + "\n"
        )
        audit_stream.flush()
        if result.candidate_eligible:
            candidates += 1
            accepted += int(result.accepted)
            deployed_accepted += int(independently_confirmed(result))
            validation_values.append(result.validation_return_improvement)
            audit_values.append(result.audit_return_improvement)
            destination = (
                holdout_stream if split == "holdout" else train_stream
            )
            for target_index, target in enumerate(result.targets):
                root_target_nodes += int(target.is_root)
                internal_target_nodes += int(not target.is_root)
                for action_index, (
                    action,
                    probability,
                ) in enumerate(
                    filtered_target_actions(
                        target,
                        0.02,
                        preferred_action=(
                            result.proposed_action
                            if target.is_root
                            else None
                        ),
                    )
                ):
                    example = puct_example(
                        target,
                        action,
                        probability,
                        target_index,
                        action_index,
                        result,
                        checkpoint=args.checkpoint,
                        checkpoint_hash=checkpoint_hash,
                        game=index,
                        side=0,
                        turn=state.core.turn,
                        reasons=[
                            f"scenario:{scenario}",
                            "raw-policy-failure",
                        ],
                        split=split,
                        internal_weight=0.25,
                        puct_config={
                            "scenarioVersion": scenario_version,
                            "transitionBudget": args.transition_budget,
                            "minRootVisits": args.min_root_visits,
                            "cpuct": 2.0,
                            "priorFloor": 0.10,
                            "horizonBattles": args.horizon_battles,
                            "validationRollouts": args.validation_rollouts,
                            "maxValidationRollouts": (
                                args.max_validation_rollouts
                            ),
                            "auditRollouts": args.audit_rollouts,
                            "leafValueWeight": args.leaf_value_weight,
                            "exhaustiveRootProbability": (
                                args.exhaustive_root_probability
                            ),
                        },
                        root_node_weight=min(
                            1.0,
                            max(
                                0.1,
                                result.validation_return_improvement / 0.10,
                            ),
                        ),
                    )
                    destination.write(
                        json.dumps(example, separators=(",", ":")) + "\n"
                    )
                    if split == "holdout":
                        holdout_examples += 1
                    else:
                        train_examples += 1
            destination.flush()
        scenario_results.append(
            {
                "scenario": scenario,
                "seed": scenario_seed,
                "accepted": result.accepted,
                "independentlyConfirmed": independently_confirmed(result),
                "candidateEligible": result.candidate_eligible,
                "matchesExpected": result.proposed_action
                == expected_actions[0],
                "validation": result.validation_return_improvement,
                "validationLower": result.validation_return_lower_bound,
                "validationRollouts": result.validation_rollouts_used,
                "outerAudit": result.audit_return_improvement,
                "outerAuditLower": result.audit_return_lower_bound,
                "targetNodes": len(result.targets),
                "wallSeconds": elapsed,
            }
        )
        print(json.dumps(scenario_results[-1], sort_keys=True), flush=True)

    train_stream.close()
    holdout_stream.close()
    audit_stream.close()
    summary = {
        "puctVersion": PUCT_VERSION,
        "scenarioVersion": scenario_version,
        "simulatorRevision": SIMULATOR_REVISION,
        "checkpointRevision": checkpoint_payload.get("simulator_revision"),
        "attempted": len(failure_rows),
        "minRootVisits": args.min_root_visits,
        "accepted": accepted,
        "candidateEligible": candidates,
        "deployedAccepted": deployed_accepted,
        "independentConfirmationRate": (
            deployed_accepted / max(1, len(failure_rows))
        ),
        "trainExamples": train_examples,
        "holdoutExamples": holdout_examples,
        "rootTargetNodes": root_target_nodes,
        "internalTargetNodes": internal_target_nodes,
        "validation": mean_ci(validation_values),
        "outerAudit": mean_ci(audit_values),
        "scenarios": scenario_results,
        "meanWallSeconds": (
            None
            if not scenario_results
            else float(
                np.mean([row["wallSeconds"] for row in scenario_results])
            )
        ),
        "reasonCounts": {
            "raw-policy-failure": len(failure_rows),
        },
        "actionChanges": {},
        "trainSha256": hashlib.sha256(args.output.read_bytes()).hexdigest(),
        "holdoutSha256": hashlib.sha256(
            args.holdout_output.read_bytes()
        ).hexdigest(),
    }
    args.summary_output.write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n"
    )
    print(json.dumps(summary, sort_keys=True))


if __name__ == "__main__":
    main()
