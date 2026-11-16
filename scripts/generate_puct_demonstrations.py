#!/usr/bin/env python3
"""Generate weighted policy targets from globally budgeted stochastic PUCT."""

from __future__ import annotations

import argparse
from dataclasses import asdict
from collections import Counter
import hashlib
import json
import math
from pathlib import Path
import time

import numpy as np
import torch

from generate_search_demonstrations import (
    finite_or_none,
    force_action,
    load_policy,
    opportunity_reasons,
    policy_action,
    serialize_observation,
    step_policy_decisions,
)
from sapai.turtle_pack import SIMULATOR_REVISION
from sapai_gym1.rl.joint_demonstrations import serialize_action, make_joint_example, load_quality_gate, action_key
from sapai_gym1.rl import ActionType, ParallelSelfPlayEnv
from sapai_gym1.rl.distributed import _completed_match_result
from sapai_gym1.rl.game_trace import action_snapshot, view_snapshot
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
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--holdout-output", type=Path, required=True)
    parser.add_argument("--audit-output", type=Path, required=True)
    parser.add_argument("--summary-output", type=Path, required=True)
    parser.add_argument("--opponent-reservoir", type=Path, required=True)
    parser.add_argument("--games", type=int, default=128)
    parser.add_argument("--max-search-attempts", type=int, default=100)
    parser.add_argument("--max-searches-per-player-turn", type=int, default=1)
    parser.add_argument(
        "--reserve-zero-gold-search-slot",
        action="store_true",
        help=(
            "When at least two searches are allowed per player-turn, reserve "
            "the final slot while gold remains so an end-of-turn freeze "
            "decision cannot be crowded out by an earlier opportunity."
        ),
    )
    parser.add_argument("--opportunity-probability", type=float, default=0.5)
    parser.add_argument("--general-probability", type=float, default=0.02)
    parser.add_argument("--minimum-turn", type=int, default=2)
    parser.add_argument("--maximum-turn", type=int, default=25)
    parser.add_argument("--holdout-fraction", type=float, default=0.20)
    parser.add_argument("--transition-budget", type=int, default=7_500)
    parser.add_argument("--max-simulations", type=int, default=4_096)
    parser.add_argument("--min-root-visits", type=int, default=4)
    parser.add_argument("--horizon-battles", type=int, default=5)
    parser.add_argument("--validation-rollouts", type=int, default=32)
    parser.add_argument("--max-validation-rollouts", type=int, default=128)
    parser.add_argument("--audit-rollouts", type=int, default=64)
    parser.add_argument("--root-screening-top-k", type=int, default=8)
    parser.add_argument("--root-screening-rollouts", type=int, default=8)
    parser.add_argument("--root-screening-rounds", type=int, default=3)
    parser.add_argument(
        "--root-screening-min-survivors", type=int, default=4
    )
    parser.add_argument("--cpuct", type=float, default=2.0)
    parser.add_argument("--prior-floor", type=float, default=0.10)
    parser.add_argument(
        "--exhaustive-root-probability",
        type=float,
        default=0.05,
        help=(
            "Fraction of full searches that bypass action quotas and expose "
            "every legal root action, used to measure pruning false negatives."
        ),
    )
    parser.add_argument("--leaf-value-weight", type=float, default=0.5)
    parser.add_argument(
        "--screening-transition-budget",
        type=int,
        default=0,
        help=(
            "Optional cheap PUCT budget used before the full search. Zero "
            "disables multi-fidelity screening."
        ),
    )
    parser.add_argument("--screening-max-simulations", type=int, default=1024)
    parser.add_argument("--screening-validation-rollouts", type=int, default=8)
    parser.add_argument(
        "--screening-max-validation-rollouts", type=int, default=16
    )
    parser.add_argument(
        "--screening-bypass-probability",
        type=float,
        default=0.05,
        help=(
            "Fraction of screen rejects sent to full search to audit false "
            "negatives."
        ),
    )
    parser.add_argument(
        "--opportunity-screening-bypass-probability",
        type=float,
        help=(
            "Optional bypass rate for recognized opportunities; defaults to "
            "--screening-bypass-probability."
        ),
    )
    parser.add_argument(
        "--general-screening-bypass-probability",
        type=float,
        help=(
            "Optional bypass rate for general sampled states; defaults to "
            "--screening-bypass-probability."
        ),
    )
    parser.add_argument(
        "--screening-minimum-improvement",
        type=float,
        default=0.0,
        help="Minimum cheap-screen mean advantage required for escalation.",
    )
    parser.add_argument(
        "--screening-lower-bound-floor",
        type=float,
        default=float("-inf"),
        help=(
            "Require the cheap-screen confidence lower bound to exceed this "
            "floor before full-search escalation."
        ),
    )
    parser.add_argument("--internal-target-weight", type=float, default=0.25)
    parser.add_argument("--minimum-target-probability", type=float, default=0.02)
    parser.add_argument("--opponent-candidates", type=int, default=32)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--seed", type=int, default=2750001)
    parser.add_argument("--deterministic-play", action="store_true")
    parser.add_argument('--diagnostic-only', action='store_true')
    parser.add_argument('--search-quality-gate', type=Path)
    return parser.parse_args()


def filtered_target_actions(
    target,
    minimum_probability,
    *,
    preferred_action=None,
    preferred_min_probability=0.5,
):
    pairs = list(zip(target.actions, target.probabilities))
    if preferred_action is not None:
        probabilities = dict(pairs)
        if preferred_action in probabilities:
            current = probabilities[preferred_action]
            if current < preferred_min_probability:
                remaining = max(1e-12, 1.0 - current)
                scale = (1.0 - preferred_min_probability) / remaining
                pairs = [
                    (
                        action,
                        preferred_min_probability
                        if action == preferred_action
                        else probability * scale,
                    )
                    for action, probability in pairs
                ]
    kept = [pair for pair in pairs if pair[1] >= minimum_probability]
    if not kept:
        kept = [max(pairs, key=lambda pair: pair[1])]
    total = sum(probability for _, probability in kept)
    return [
        (action, probability / total) for action, probability in kept
    ]


def per_turn_search_limit(maximum, gold, reserve_zero_gold_slot=False):
    """Return the opportunity-search limit available at the current gold."""

    if reserve_zero_gold_slot and maximum >= 2 and gold > 0:
        return maximum - 1
    return maximum


def independently_confirmed(result):
    """Require both adaptive validation and untouched outer confirmation."""

    return bool(
        result.accepted
        and result.audit_return_lower_bound is not None
        and result.audit_return_lower_bound > 0.0
    )


def puct_example(
    target,
    action,
    probability,
    target_index,
    action_index,
    root_result,
    *,
    checkpoint,
    checkpoint_hash,
    game,
    side,
    turn,
    reasons,
    split,
    internal_weight,
    puct_config,
    root_node_weight,
    policy=None,
    observation_prefix=None,
    quality_gate=None,
):
    node_weight = root_node_weight * (
        1.0 if target.is_root else internal_weight
    )
    example = {
        "schema_version": 1,
        "replay_id": (
            f"puct-game-{game:06d}-p{side}-turn-{turn:02d}-"
            f"node-{target_index:03d}-action-{action_index:02d}"
        ),
        "perspective": f"p{side}",
        "turn": int(target.turn or turn),
        "decision": 0,
        "observation": serialize_observation(target.observation),
        "initialMemory": target.memory.float().cpu().tolist(),
        "action": serialize_action(action),
        "teacherWeight": float(probability * node_weight),
        "battle_outcome": None,
        "final_result": None,
        "validation": {
            "exactSimulatorState": True,
            "simulatorRevision": SIMULATOR_REVISION,
            "puctVersion": PUCT_VERSION,
            "publicInformationOnly": True,
        },
        "teacher": {
            "name": "stochastic-puct",
            "version": PUCT_VERSION,
            "checkpoint": str(checkpoint),
            "checkpointSha256": checkpoint_hash,
            "split": split,
            "rootAccepted": root_result.accepted,
            "rootCandidateEligible": root_result.candidate_eligible,
            "rootProposedAction": serialize_action(root_result.proposed_action),
            "rootTreeProposedAction": serialize_action(root_result.tree_proposed_action),
            "rootScreeningScores": {
                action_key(action): score
                for action, score in root_result.root_screening_scores.items()
            },
            "rootScreeningCounts": {
                action_key(action): count
                for action, count in root_result.root_screening_counts.items()
            },
            "rootPolicyAction": serialize_action(root_result.policy_action),
            "validationReturnImprovement": (
                root_result.validation_return_improvement
            ),
            "validationReturnLowerBound": (
                root_result.validation_return_lower_bound
            ),
            "validationRolloutsUsed": (
                root_result.validation_rollouts_used
            ),
            "outerAuditReturnImprovement": (
                root_result.audit_return_improvement
            ),
            "outerAuditReturnLowerBound": (
                root_result.audit_return_lower_bound
            ),
            "opportunityReasons": reasons,
            "targetNodeIndex": target_index,
            "targetIsRoot": target.is_root,
            "targetMethod": (
                target.method
                + ("+confirmed-root-anchor" if target.is_root else "")
            ),
            "targetNodeVisits": target.visits,
            "targetActionProbability": probability,
            "targetNodeWeight": node_weight,
            "puctConfig": puct_config,
        },
    }
    return make_joint_example(example, action, target.observation, target.memory,
        policy=policy, checkpoint_sha256=checkpoint_hash,
        omitted_mass=getattr(target, 'omitted_end_mask_probability_mass', 0.),
        observation_prefix=observation_prefix if target.is_root else None,
        is_root=target.is_root, quality_gate=quality_gate)


def audit_record(result, public_state, *, game, side, turn, reasons, split, elapsed):
    root_actions = sorted(
        result.visit_counts,
        key=lambda action: (
            result.q_lower_bounds[action],
            result.q_values[action],
        ),
        reverse=True,
    )
    return {
        "game": game,
        "side": side,
        "turn": turn,
        "split": split,
        "reasons": reasons,
        "publicState": view_snapshot(public_state.core, 0),
        "policyAction": serialize_action(result.policy_action),
        "policyActionDetail": action_snapshot(
            public_state.core.agent, result.policy_action
        ),
        "treeProposedAction": serialize_action(result.tree_proposed_action),
        "treeProposedActionDetail": action_snapshot(
            public_state.core.agent, result.tree_proposed_action
        ),
        "proposedAction": serialize_action(result.proposed_action),
        "proposedActionDetail": action_snapshot(
            public_state.core.agent, result.proposed_action
        ),
        "selectedAction": serialize_action(result.selected_action),
        "omittedEndMaskProbabilityMass": result.omitted_end_mask_probability_mass,
        "accepted": result.accepted,
        "independentlyConfirmed": independently_confirmed(result),
        "candidateEligible": result.candidate_eligible,
        "exhaustiveRoot": result.exhaustive_root,
        "fullCandidateCount": result.full_candidate_count,
        "prunedCandidateCount": result.pruned_candidate_count,
        "treeProposedExcludedByPruning": (
            result.tree_proposed_excluded_by_pruning
        ),
        "proposedExcludedByPruning": result.proposed_excluded_by_pruning,
        "prunedActionCounts": result.pruned_action_counts,
        "validationReturnImprovement": result.validation_return_improvement,
        "validationReturnStandardError": (
            result.validation_return_standard_error
        ),
        "validationReturnLowerBound": result.validation_return_lower_bound,
        "validationRolloutsUsed": result.validation_rollouts_used,
        "validationFirstBattleScoreImprovement": (
            result.validation_first_battle_score_improvement
        ),
        "validationHorizonBattleScoreImprovement": (
            result.validation_horizon_battle_score_improvement
        ),
        "outerAuditReturnImprovement": result.audit_return_improvement,
        "outerAuditReturnStandardError": result.audit_return_standard_error,
        "outerAuditReturnLowerBound": result.audit_return_lower_bound,
        "outerAuditFirstBattleScoreImprovement": (
            result.audit_first_battle_score_improvement
        ),
        "outerAuditHorizonBattleScoreImprovement": (
            result.audit_horizon_battle_score_improvement
        ),
        "simulations": result.simulations,
        "simulatorTransitions": result.simulator_transitions,
        "treeNodes": result.tree_nodes,
        "meanTreeDepth": result.mean_tree_depth,
        "maxTreeDepth": result.max_tree_depth,
        "validationTreeActionsMean": result.validation_tree_actions_mean,
        "outerAuditTreeActionsMean": result.audit_tree_actions_mean,
        "validationPlannedOpponentSourceCounts": (
            result.validation_planned_opponent_source_counts
        ),
        "validationPolicyOpponentSourceCounts": (
            result.validation_policy_opponent_source_counts
        ),
        "outerAuditPlannedOpponentSourceCounts": (
            result.audit_planned_opponent_source_counts
        ),
        "outerAuditPolicyOpponentSourceCounts": (
            result.audit_policy_opponent_source_counts
        ),
        "trainingTargetNodes": len(result.targets),
        "wallSeconds": elapsed,
        "root": [
            {
                "action": serialize_action(action),
                "description": action_snapshot(
                    public_state.core.agent, action
                )["description"],
                "prior": result.priors[action],
                "visits": result.visit_counts[action],
                "q": result.q_values[action],
                "qStandardError": finite_or_none(
                    result.q_standard_errors[action]
                ),
                "qLowerBound": finite_or_none(
                    result.q_lower_bounds[action]
                ),
            }
            for action in root_actions[:12]
        ],
        "rootScreening": [
            {
                "action": serialize_action(action),
                "description": action_snapshot(
                    public_state.core.agent, action
                )["description"],
                "meanImprovement": score,
                "rollouts": result.root_screening_counts[action],
            }
            for action, score in sorted(
                result.root_screening_scores.items(),
                key=lambda item: item[1],
                reverse=True,
            )
        ],
    }


def mean_ci(values):
    if not values:
        return {
            "count": 0,
            "mean": None,
            "standardError": None,
            "lowerBound95": None,
        }
    values = np.asarray(values, dtype=np.float64)
    mean = float(values.mean())
    if len(values) < 2:
        return {
            "count": len(values),
            "mean": mean,
            "standardError": None,
            "lowerBound95": None,
        }
    standard_error = float(values.std(ddof=1) / math.sqrt(len(values)))
    return {
        "count": len(values),
        "mean": mean,
        "standardError": standard_error,
        "lowerBound95": mean - 1.96 * standard_error,
    }


def main():
    args = parse_args()
    if not 0.0 <= args.holdout_fraction < 1.0:
        raise SystemExit("--holdout-fraction must be in [0, 1)")
    if args.minimum_turn < 1 or args.maximum_turn < args.minimum_turn:
        raise SystemExit("invalid turn range")
    if not 0.0 < args.internal_target_weight <= 1.0:
        raise SystemExit("--internal-target-weight must be in (0, 1]")
    if not 0.0 <= args.exhaustive_root_probability <= 1.0:
        raise SystemExit("--exhaustive-root-probability must be in [0, 1]")
    if args.screening_transition_budget < 0:
        raise SystemExit("--screening-transition-budget must be non-negative")
    if not 0.0 <= args.screening_bypass_probability <= 1.0:
        raise SystemExit(
            "--screening-bypass-probability must be in [0, 1]"
        )
    for value in (
        args.opportunity_screening_bypass_probability,
        args.general_screening_bypass_probability,
    ):
        if value is not None and not 0.0 <= value <= 1.0:
            raise SystemExit("conditional screening bypass rates must be in [0, 1]")
    device = torch.device(
        args.device
        if torch.cuda.is_available() or args.device == "cpu"
        else "cpu"
    )
    torch.manual_seed(args.seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(args.seed)
    policy, checkpoint_payload = load_policy(args.checkpoint, device, False)
    checkpoint_hash = hashlib.sha256(args.checkpoint.read_bytes()).hexdigest()
    if policy.config.end_freeze:
        if args.search_quality_gate:
            load_quality_gate(args.search_quality_gate, checkpoint_sha256=checkpoint_hash)
        elif not args.diagnostic_only:
            raise SystemExit('Joint corpus requires --diagnostic-only or an independent --search-quality-gate')
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
            cpuct=args.cpuct,
            prior_floor=args.prior_floor,
            root_noise_fraction=0.0,
            leaf_value_weight=args.leaf_value_weight,
            validation_rollouts=args.validation_rollouts,
            max_validation_rollouts=args.max_validation_rollouts,
            audit_rollouts=args.audit_rollouts,
            root_screening_top_k=args.root_screening_top_k,
            root_screening_rollouts=args.root_screening_rollouts,
            root_screening_rounds=args.root_screening_rounds,
            root_screening_min_survivors=(
                args.root_screening_min_survivors
            ),
            confidence_z=1.96,
            minimum_improvement=0.0,
            prune_affordable_freezes=False,
            exhaustive_root_probability=args.exhaustive_root_probability,
        ),
    )
    screening_planner = None
    if args.screening_transition_budget:
        screening_planner = StochasticPUCTSearch(
            evaluator,
            PUCTConfig(
                max_simulations=args.screening_max_simulations,
                max_simulator_transitions=(
                    args.screening_transition_budget
                ),
                max_rollout_steps=180,
                horizon_battles=args.horizon_battles,
                world_pool_size=16,
                min_root_visits=1,
                cpuct=args.cpuct,
                prior_floor=args.prior_floor,
                root_noise_fraction=0.0,
                leaf_value_weight=args.leaf_value_weight,
                validation_rollouts=args.screening_validation_rollouts,
                max_validation_rollouts=(
                    args.screening_max_validation_rollouts
                ),
                audit_rollouts=0,
                confidence_z=1.96,
                minimum_improvement=args.screening_minimum_improvement,
                root_screening_top_k=4,
                root_screening_rollouts=2,
                root_screening_rounds=1,
                root_screening_min_survivors=2,
                prune_affordable_freezes=False,
                exhaustive_root_probability=0.0,
            ),
        )
    reservoir = OpponentTrajectoryReservoir.load(args.opponent_reservoir)
    if reservoir.metadata.get("simulatorRevision") != SIMULATOR_REVISION:
        raise SystemExit("Opponent reservoir simulator revision mismatch")

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
    rng = np.random.default_rng(args.seed)
    attempted = 0
    accepted = 0
    deployed_accepted = 0
    candidates = 0
    root_targets = 0
    internal_targets = 0
    train_examples = 0
    holdout_examples = 0
    reason_counts = Counter()
    action_changes = Counter()
    validation_values = []
    audit_values = []
    elapsed_values = []
    screening_attempts = 0
    screening_passes = 0
    screening_bypasses = 0
    full_searches = 0
    bypass_candidates = 0
    screening_pass_candidates = 0
    exhaustive_root_searches = 0
    exhaustive_proposed_excluded = 0
    exhaustive_tree_proposed_excluded = 0
    confirmed_pruned_winners = 0
    pruned_winners_by_action = Counter()
    removed_actions_by_type = Counter()
    validation_planned_sources = Counter()
    validation_policy_sources = Counter()
    audit_planned_sources = Counter()
    audit_policy_sources = Counter()

    for game in range(args.games):
        if attempted >= args.max_search_attempts:
            break
        env = ParallelSelfPlayEnv()
        observations = env.reset(seed=int(rng.integers(0, 2**31 - 1)))
        memories = [policy.initial_memory(1, device="cpu")[0] for _ in range(2)]
        observation_prefixes = [[], []]
        per_turn_searches = Counter()
        while True:
            active = env.active_players()
            actions = [None, None]
            for side in range(2):
                if not active[side]:
                    continue
                observation = observations[side]
                reasons = opportunity_reasons(env.core, side)
                probability = (
                    args.opportunity_probability
                    if reasons
                    else args.general_probability
                )
                key = (side, env.core.turn)
                player = env.core.agent if side == 0 else env.core.opponent
                current_search_limit = per_turn_search_limit(
                    args.max_searches_per_player_turn,
                    player.gold,
                    args.reserve_zero_gold_search_slot,
                )
                run_search = (
                    attempted < args.max_search_attempts
                    and args.minimum_turn <= env.core.turn <= args.maximum_turn
                    and per_turn_searches[key]
                    < current_search_limit
                    and rng.random() < probability
                )
                result = None
                public_state = None
                if run_search:
                    try:
                        public_state = PublicSearchState.from_core(
                            env.core, side
                        )
                    except ValueError:
                        public_state = None
                if public_state is not None:
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
                        public_state.set_opponent_trajectories(
                            [match.teams for match in matches],
                            [match.trajectory_id for match in matches],
                        )
                        public_state.set_opponent_transition_model(
                            reservoir,
                            candidate_limit=args.opponent_candidates,
                        )
                        started = time.perf_counter()
                        screening_result = None
                        bypassed = False
                        screen_passed = False
                        bypass_probability = None
                        if screening_planner is not None:
                            screening_attempts += 1
                            screening_result = screening_planner.search(
                                public_state,
                                memories[side],
                                seed=int(rng.integers(0, 2**63 - 1)),
                                validation_seed=int(
                                    rng.integers(0, 2**63 - 1)
                                ),
                            )
                            screen_passed = bool(
                                screening_result.candidate_eligible
                                and screening_result.validation_return_lower_bound
                                > args.screening_lower_bound_floor
                            )
                            screening_passes += int(screen_passed)
                            bypass_probability = (
                                args.opportunity_screening_bypass_probability
                                if reasons
                                and args.opportunity_screening_bypass_probability
                                is not None
                                else args.general_screening_bypass_probability
                                if not reasons
                                and args.general_screening_bypass_probability
                                is not None
                                else args.screening_bypass_probability
                            )
                            bypassed = (
                                not screen_passed
                                and rng.random() < bypass_probability
                            )
                            screening_bypasses += int(bypassed)
                        run_full = (
                            screening_result is None
                            or screen_passed
                            or bypassed
                        )
                        if run_full:
                            full_searches += 1
                            result = planner.search(
                                public_state,
                                memories[side],
                                seed=int(rng.integers(0, 2**63 - 1)),
                                validation_seed=int(
                                    rng.integers(0, 2**63 - 1)
                                ),
                            )
                            bypass_candidates += int(
                                bypassed and result.candidate_eligible
                            )
                            screening_pass_candidates += int(
                                screening_result is not None
                                and screen_passed
                                and result.candidate_eligible
                            )
                            if result.exhaustive_root:
                                exhaustive_root_searches += 1
                                exhaustive_proposed_excluded += int(
                                    bool(result.proposed_excluded_by_pruning)
                                )
                                exhaustive_tree_proposed_excluded += int(
                                    bool(
                                        result.tree_proposed_excluded_by_pruning
                                    )
                                )
                                removed_actions_by_type.update(
                                    result.pruned_action_counts
                                )
                                confirmed_pruned = bool(
                                    result.proposed_excluded_by_pruning
                                    and result.accepted
                                    and result.audit_return_lower_bound
                                    is not None
                                    and result.audit_return_lower_bound > 0
                                )
                                confirmed_pruned_winners += int(
                                    confirmed_pruned
                                )
                                if confirmed_pruned:
                                    pruned_winners_by_action.update(
                                        [
                                            ActionType(
                                                result.proposed_action.action_type
                                            ).name
                                        ]
                                    )
                        else:
                            result = screening_result
                        elapsed = time.perf_counter() - started
                        validation_planned_sources.update(
                            result.validation_planned_opponent_source_counts
                        )
                        validation_policy_sources.update(
                            result.validation_policy_opponent_source_counts
                        )
                        audit_planned_sources.update(
                            result.audit_planned_opponent_source_counts
                        )
                        audit_policy_sources.update(
                            result.audit_policy_opponent_source_counts
                        )
                        elapsed_values.append(elapsed)
                        attempted += 1
                        per_turn_searches[key] += 1
                        reason_counts.update(reasons or ["general"])
                        split = (
                            "holdout"
                            if rng.random() < args.holdout_fraction
                            else "train"
                        )
                        audit = audit_record(
                            result,
                            public_state,
                            game=game,
                            side=side,
                            turn=env.core.turn,
                            reasons=reasons,
                            split=split,
                            elapsed=elapsed,
                        )
                        audit["multiFidelity"] = {
                            "enabled": screening_result is not None,
                            "fullSearch": run_full,
                            "screenPass": (
                                screening_result is not None
                                and screen_passed
                            ),
                            "screenBypass": bypassed,
                            "screenBypassProbability": bypass_probability,
                            "screenProposedAction": (
                                None
                                if screening_result is None
                                else serialize_action(screening_result.proposed_action)
                            ),
                            "screenValidationMean": (
                                None
                                if screening_result is None
                                else screening_result.validation_return_improvement
                            ),
                            "screenValidationLowerBound": (
                                None
                                if screening_result is None
                                else screening_result.validation_return_lower_bound
                            ),
                        }
                        audit_stream.write(
                            json.dumps(audit, separators=(",", ":")) + "\n"
                        )
                        audit_stream.flush()
                        if result.candidate_eligible:
                            candidates += 1
                            accepted += int(result.accepted)
                            deployed_accepted += int(
                                independently_confirmed(result)
                            )
                            validation_values.append(
                                result.validation_return_improvement
                            )
                            if result.audit_return_improvement is not None:
                                audit_values.append(
                                    result.audit_return_improvement
                                )
                            action_changes.update(
                                [
                                    f"{ActionType(result.policy_action.action_type).name}"
                                    f"->{ActionType(result.proposed_action.action_type).name}"
                                ]
                            )
                            destination = (
                                holdout_stream
                                if split == "holdout"
                                else train_stream
                            )
                            for target_index, target in enumerate(
                                result.targets
                            ):
                                root_targets += int(target.is_root)
                                internal_targets += int(not target.is_root)
                                for action_index, (
                                    target_action,
                                    target_probability,
                                ) in enumerate(
                                    filtered_target_actions(
                                        target,
                                        args.minimum_target_probability,
                                        preferred_action=(
                                            result.proposed_action
                                            if target.is_root
                                            else None
                                        ),
                                    )
                                ):
                                    example = puct_example(
                                        target,
                                        target_action,
                                        target_probability,
                                        target_index,
                                        action_index,
                                        result,
                                        checkpoint=args.checkpoint,
                                        checkpoint_hash=checkpoint_hash,
                                        policy=policy,
                                        observation_prefix=observation_prefixes[side],
                                        quality_gate=args.search_quality_gate,
                                        game=game,
                                        side=side,
                                        turn=env.core.turn,
                                        reasons=reasons,
                                        split=split,
                                        internal_weight=(
                                            args.internal_target_weight
                                        ),
                                        puct_config=asdict(planner.config),
                                        root_node_weight=min(
                                            1.0,
                                            max(
                                                0.1,
                                                result.validation_return_improvement
                                                / 0.10,
                                            ),
                                        ),
                                    )
                                    if policy.config.end_freeze and args.search_quality_gate and not example['productionTrainingAllowed']:
                                        continue
                                    destination.write(
                                        json.dumps(
                                            example,
                                            separators=(",", ":"),
                                        )
                                        + "\n"
                                    )
                                    if split == "holdout":
                                        holdout_examples += 1
                                    else:
                                        train_examples += 1
                            destination.flush()
                        print(
                            f"puct={attempted}/{args.max_search_attempts} "
                            f"game={game + 1}/{args.games} turn={env.core.turn} "
                            f"side={side} candidates={candidates} "
                            f"accepted={accepted} "
                            f"full={int(run_full)} "
                            f"screen_pass={int(screening_result is not None and screen_passed)} "
                            f"screen_bypass={int(bypassed)} "
                            f"validation={result.validation_return_improvement:.4f} "
                            f"validation_n={result.validation_rollouts_used} "
                            f"outer={finite_or_none(result.audit_return_improvement)} "
                            f"targets={len(result.targets)} wall={elapsed:.1f}s",
                            flush=True,
                        )
                if result is not None and independently_confirmed(result):
                    actions[side] = result.selected_action
                    memories[side] = force_action(
                        policy,
                        observation,
                        memories[side],
                        actions[side],
                        device,
                    )
                else:
                    actions[side], memories[side] = policy_action(
                        policy,
                        observation,
                        memories[side],
                        device,
                        args.deterministic_play,
                    )
                observation_prefixes[side].append({k: np.asarray(v).copy() for k, v in observation.items()})
            step = step_policy_decisions(env, actions)
            observations = step.observations
            if step.terminated:
                _completed_match_result(env.core)
                break

    train_stream.close()
    holdout_stream.close()
    audit_stream.close()
    summary = {
        "puctVersion": PUCT_VERSION,
        "simulatorRevision": SIMULATOR_REVISION,
        "checkpointRevision": checkpoint_payload.get("simulator_revision"),
        "attempted": attempted,
        "accepted": accepted,
        "candidateEligible": candidates,
        "candidateRate": candidates / max(1, attempted),
        "deployedAccepted": deployed_accepted,
        "independentConfirmationRate": (
            deployed_accepted / max(1, attempted)
        ),
        "acceptanceRate": accepted / max(1, attempted),
        "trainExamples": train_examples,
        "holdoutExamples": holdout_examples,
        "rootTargetNodes": root_targets,
        "internalTargetNodes": internal_targets,
        "validation": mean_ci(validation_values),
        "outerAudit": mean_ci(audit_values),
        "meanWallSeconds": (
            None if not elapsed_values else float(np.mean(elapsed_values))
        ),
        "reasonCounts": dict(sorted(reason_counts.items())),
        "actionChanges": dict(sorted(action_changes.items())),
        "multiFidelity": {
            "enabled": screening_planner is not None,
            "screeningAttempts": screening_attempts,
            "screeningPasses": screening_passes,
            "screeningBypasses": screening_bypasses,
            "fullSearches": full_searches,
            "bypassCandidates": bypass_candidates,
            "screeningPassCandidates": screening_pass_candidates,
            "screeningTransitionBudget": args.screening_transition_budget,
            "screeningMaxSimulations": args.screening_max_simulations,
            "screeningValidationRollouts": (
                args.screening_validation_rollouts
            ),
            "screeningMaxValidationRollouts": (
                args.screening_max_validation_rollouts
            ),
            "screeningBypassProbability": (
                args.screening_bypass_probability
            ),
            "opportunityScreeningBypassProbability": (
                args.opportunity_screening_bypass_probability
            ),
            "generalScreeningBypassProbability": (
                args.general_screening_bypass_probability
            ),
            "screeningMinimumImprovement": (
                args.screening_minimum_improvement
            ),
            "screeningLowerBoundFloor": (
                args.screening_lower_bound_floor
            ),
        },
        "rootPruningAudit": {
            "configuredExhaustiveProbability": (
                args.exhaustive_root_probability
            ),
            "exhaustiveFullSearches": exhaustive_root_searches,
            "proposedExcludedByPruning": exhaustive_proposed_excluded,
            "treeProposedExcludedByPruning": (
                exhaustive_tree_proposed_excluded
            ),
            "independentlyConfirmedPrunedWinners": (
                confirmed_pruned_winners
            ),
            "confirmedFalseNegativeRate": (
                confirmed_pruned_winners / max(1, exhaustive_root_searches)
            ),
            "confirmedWinnersByAction": dict(
                sorted(pruned_winners_by_action.items())
            ),
            "removedActionsByType": dict(
                sorted(removed_actions_by_type.items())
            ),
        },
        "opponentSourceTelemetry": {
            "validationPlanned": dict(
                sorted(validation_planned_sources.items())
            ),
            "validationPolicy": dict(
                sorted(validation_policy_sources.items())
            ),
            "outerAuditPlanned": dict(
                sorted(audit_planned_sources.items())
            ),
            "outerAuditPolicy": dict(
                sorted(audit_policy_sources.items())
            ),
        },
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
