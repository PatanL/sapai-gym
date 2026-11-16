#!/usr/bin/env python3
"""Validate proposed actions on matched futures and emit safe policy targets."""

from __future__ import annotations

import argparse
from contextlib import nullcontext
import hashlib
import json
import math
from pathlib import Path
import time

import numpy as np
import torch

from sapai import Team
from sapai.turtle_pack import SIMULATOR_REVISION
from sapai_gym1.rl import (
    AutoregressiveAction,
    EntityTransformerConfig,
    EntityTransformerPolicy,
)
from sapai_gym1.rl.curriculum import BUILDERS, PUCT_SCENARIO_VERSION
from sapai_gym1.rl.game_trace import action_snapshot, view_snapshot
from sapai_gym1.rl.opponent_reservoir import OpponentTrajectoryReservoir
from sapai_gym1.rl.rollout_search import (
    CounterfactualRolloutSearch,
    NeuralSearchEvaluator,
    PublicSearchState,
    RolloutSearchConfig,
)


VALIDATOR_VERSION = "matched-candidate-validator-v1-20260903"


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("candidates", type=Path)
    parser.add_argument("--opponent-reservoir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--holdout-output", type=Path, required=True)
    parser.add_argument("--audit-output", type=Path, required=True)
    parser.add_argument("--summary-output", type=Path, required=True)
    parser.add_argument("--initial-rollouts", type=int, default=128)
    parser.add_argument("--max-rollouts", type=int, default=512)
    parser.add_argument("--audit-rollouts", type=int, default=256)
    parser.add_argument("--horizon-battles", type=int, default=5)
    parser.add_argument("--opponent-candidates", type=int, default=32)
    parser.add_argument("--confidence-z", type=float, default=1.96)
    parser.add_argument("--minimum-improvement", type=float, default=0.0)
    parser.add_argument("--holdout-fraction", type=float, default=0.20)
    parser.add_argument("--candidate-source", default="expert-diagnostic")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--seed", type=int, default=2795001)
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument("--num-shards", type=int, default=1)
    return parser.parse_args()


def as_batch(observation, device):
    return {
        key: torch.as_tensor(value, device=device).unsqueeze(0)
        for key, value in observation.items()
    }


@torch.inference_mode()
def policy_action(policy, state, memory, device):
    autocast = (
        torch.autocast("cuda", dtype=torch.bfloat16)
        if device.type == "cuda"
        else nullcontext()
    )
    with autocast:
        output = policy(
            as_batch(state.observation(), device),
            memory=memory.unsqueeze(0).to(device),
            deterministic=True,
        )
    return AutoregressiveAction(*map(int, output.actions[0].cpu()))


def statistics(differences, confidence_z):
    values = np.asarray(differences, dtype=np.float64)
    mean = float(values.mean())
    standard_error = float(values.std(ddof=1) / math.sqrt(len(values)))
    return {
        "count": len(values),
        "mean": mean,
        "standardError": standard_error,
        "lowerBound": mean - confidence_z * standard_error,
        "upperBound": mean + confidence_z * standard_error,
    }


def aggregate(values, confidence_z):
    if not values:
        return {
            "count": 0,
            "mean": None,
            "standardError": None,
            "lowerBound": None,
            "upperBound": None,
        }
    if len(values) == 1:
        value = float(values[0])
        return {
            "count": 1,
            "mean": value,
            "standardError": None,
            "lowerBound": None,
            "upperBound": None,
        }
    return statistics(values, confidence_z)


def evaluate_actions(
    search,
    state,
    next_memory,
    candidate,
    policy,
    seeds,
):
    records = search._rollouts(
        state,
        [candidate, policy],
        next_memory,
        seeds,
        allow_nested=False,
        retain_trace=False,
    )
    by_action = {candidate: [], policy: []}
    for record in records:
        by_action[record["root_action"]].append(record["total"])
    return np.asarray(by_action[candidate]) - np.asarray(by_action[policy])


def serialize_observation(observation):
    return {name: value.tolist() for name, value in observation.items()}


def main():
    args = parse_args()
    if args.initial_rollouts < 2:
        raise SystemExit("--initial-rollouts must be at least two")
    if args.max_rollouts < args.initial_rollouts:
        raise SystemExit("--max-rollouts must cover --initial-rollouts")
    if args.audit_rollouts < 2:
        raise SystemExit("--audit-rollouts must be at least two")
    if not 0 <= args.shard_index < args.num_shards:
        raise SystemExit("invalid shard index/count")
    if not 0 <= args.holdout_fraction < 1:
        raise SystemExit("--holdout-fraction must be in [0, 1)")

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
    checkpoint_hash = hashlib.sha256(args.checkpoint.read_bytes()).hexdigest()
    evaluator = NeuralSearchEvaluator(
        policy, device=device, chunk_size=args.batch_size
    )
    search = CounterfactualRolloutSearch(
        evaluator,
        RolloutSearchConfig(
            horizon_battles=args.horizon_battles,
            max_continuation_steps=180,
            leaf_value_weight=0.5,
            audit_rollouts=0,
        ),
    )
    search.simulator_transitions = 0
    reservoir = OpponentTrajectoryReservoir.load(args.opponent_reservoir)
    if reservoir.metadata.get("simulatorRevision") != SIMULATOR_REVISION:
        raise SystemExit("Opponent reservoir simulator revision mismatch")
    payload = json.loads(args.candidates.read_text())
    rows = sorted(
        payload["rows"],
        key=lambda row: hashlib.sha256(
            f"{row['scenario']}:{row['seed']}".encode()
        ).hexdigest(),
    )[args.shard_index :: args.num_shards]
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
    train_examples = 0
    holdout_examples = 0
    validation_means = []
    audit_means = []
    elapsed_values = []

    for index, row in enumerate(rows):
        started = time.perf_counter()
        scenario = row["scenario"]
        scenario_seed = int(row["seed"])
        env, proposed_actions = BUILDERS[scenario](
            np.random.default_rng(scenario_seed)
        )
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
            limit=args.opponent_candidates,
        )
        state.set_opponent_trajectories(
            [match.teams for match in matches],
            [match.trajectory_id for match in matches],
        )
        state.set_opponent_transition_model(
            reservoir, candidate_limit=args.opponent_candidates
        )
        memory = policy.initial_memory(1, device="cpu")[0]
        root = evaluator.evaluate(state, memory)
        raw = policy_action(policy, state, memory, device)
        candidate = proposed_actions[0]
        differences = []
        while True:
            remaining = args.max_rollouts - len(differences)
            batch = min(args.initial_rollouts, remaining)
            seeds = rng.integers(0, 2**63 - 1, size=batch)
            differences.extend(
                map(
                    float,
                    evaluate_actions(
                        search,
                        state,
                        root.next_memory,
                        candidate,
                        raw,
                        seeds,
                    ),
                )
            )
            validation = statistics(differences, args.confidence_z)
            if (
                validation["lowerBound"] > args.minimum_improvement
                or validation["upperBound"] <= args.minimum_improvement
                or len(differences) >= args.max_rollouts
            ):
                break
        candidate_passed = (
            candidate != raw
            and validation["lowerBound"] > args.minimum_improvement
        )
        outer = None
        if candidate_passed:
            outer = statistics(
                evaluate_actions(
                    search,
                    state,
                    root.next_memory,
                    candidate,
                    raw,
                    rng.integers(
                        0, 2**63 - 1, size=args.audit_rollouts
                    ),
                ),
                args.confidence_z,
            )
        accepted_here = bool(
            candidate_passed
            and outer is not None
            and outer["lowerBound"] > args.minimum_improvement
        )
        split = (
            "holdout"
            if rng.random() < args.holdout_fraction
            else "train"
        )
        elapsed = time.perf_counter() - started
        elapsed_values.append(elapsed)
        audit_row = {
            "validatorVersion": VALIDATOR_VERSION,
            "scenarioVersion": payload.get(
                "scenarioVersion", PUCT_SCENARIO_VERSION
            ),
            "scenario": scenario,
            "scenarioSeed": scenario_seed,
            "candidateSource": args.candidate_source,
            "split": split,
            "publicState": view_snapshot(state.core, 0),
            "candidateAction": list(candidate.as_tuple()),
            "candidateActionDetail": action_snapshot(
                state.core.agent, candidate
            ),
            "policyAction": list(raw.as_tuple()),
            "policyActionDetail": action_snapshot(state.core.agent, raw),
            "validation": validation,
            "outerAudit": outer,
            "accepted": accepted_here,
            "wallSeconds": elapsed,
        }
        audit_stream.write(
            json.dumps(audit_row, separators=(",", ":")) + "\n"
        )
        audit_stream.flush()
        if accepted_here:
            accepted += 1
            validation_means.append(validation["mean"])
            audit_means.append(outer["mean"])
            weight = min(1.0, max(0.1, outer["mean"] / 0.10))
            example = {
                "schema_version": 1,
                "replay_id": (
                    f"validated-candidate-{scenario}-{scenario_seed}"
                ),
                "perspective": "p0",
                "turn": int(state.core.turn),
                "decision": 0,
                "observation": serialize_observation(state.observation()),
                "initialMemory": memory.float().cpu().tolist(),
                "action": list(candidate.as_tuple()),
                "teacherWeight": weight,
                "battle_outcome": None,
                "final_result": None,
                "validation": {
                    "exactSimulatorState": True,
                    "simulatorRevision": SIMULATOR_REVISION,
                    "publicInformationOnly": True,
                    "validatorVersion": VALIDATOR_VERSION,
                },
                "teacher": {
                    "name": "independently-validated-candidate",
                    "version": VALIDATOR_VERSION,
                    "checkpoint": str(args.checkpoint),
                    "checkpointSha256": checkpoint_hash,
                    "candidateSource": args.candidate_source,
                    "scenario": scenario,
                    "scenarioSeed": scenario_seed,
                    "validation": validation,
                    "outerAudit": outer,
                },
            }
            destination = (
                holdout_stream if split == "holdout" else train_stream
            )
            destination.write(json.dumps(example, separators=(",", ":")) + "\n")
            destination.flush()
            if split == "holdout":
                holdout_examples += 1
            else:
                train_examples += 1
        print(
            json.dumps(
                {
                    "scenario": scenario,
                    "seed": scenario_seed,
                    "accepted": accepted_here,
                    "validation": validation,
                    "outerAudit": outer,
                    "wallSeconds": elapsed,
                },
                sort_keys=True,
            ),
            flush=True,
        )

    train_stream.close()
    holdout_stream.close()
    audit_stream.close()
    summary = {
        "validatorVersion": VALIDATOR_VERSION,
        "scenarioVersion": payload.get(
            "scenarioVersion", PUCT_SCENARIO_VERSION
        ),
        "simulatorRevision": SIMULATOR_REVISION,
        "checkpoint": str(args.checkpoint),
        "checkpointSha256": checkpoint_hash,
        "candidateSource": args.candidate_source,
        "attempted": len(rows),
        "accepted": accepted,
        "trainExamples": train_examples,
        "holdoutExamples": holdout_examples,
        "validation": aggregate(validation_means, args.confidence_z),
        "outerAudit": aggregate(audit_means, args.confidence_z),
        "meanWallSeconds": (
            float(np.mean(elapsed_values)) if elapsed_values else None
        ),
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
