#!/usr/bin/env python3
"""Compare rollout root search and PUCT at equal selection-transition cost."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import time

import numpy as np
import torch

from benchmark_planner_depth import collect_state
from search_state_mining import GOLD_BANDS, MINING_VERSION, mine_match_opportunity
from sapai.turtle_pack import SIMULATOR_REVISION
from sapai_gym1.rl import (
    AdditiveCriticCalibrator,
    EntityTransformerConfig,
    EntityTransformerPolicy,
    OPPONENT_BELIEF_VERSION,
    OpponentBeliefMix,
    ParallelSelfPlayEnv,
)
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
    parser.add_argument("--states", type=int, default=4)
    parser.add_argument(
        "--state-mode", choices=("uniform-turn", "opportunity"),
        default="uniform-turn",
    )
    parser.add_argument("--minimum-turn", type=int, default=2)
    parser.add_argument("--maximum-turn", type=int, default=11)
    parser.add_argument("--validation-rollouts", type=int, default=32)
    parser.add_argument("--comparison-rollouts", type=int, default=64)
    parser.add_argument("--horizon-battles", type=int, default=5)
    parser.add_argument("--leaf-value-weight", type=float, default=0.5)
    parser.add_argument("--critic-bias-calibration", type=Path)
    parser.add_argument("--puct-cpuct", type=float, default=1.5)
    parser.add_argument("--puct-prior-floor", type=float, default=0.02)
    parser.add_argument("--puct-min-root-visits", type=int, default=1)
    parser.add_argument(
        "--puct-transition-budget-scale",
        type=float,
        default=1.0,
        help=(
            "Scale PUCT's tree-selection transition budget relative to root "
            "search; values below one support equal-wall-clock ablations."
        ),
    )
    parser.add_argument(
        "--puct-selection-only",
        action="store_true",
        help=(
            "Use the PUCT tree proposal directly and leave all confirmation "
            "to the common independent comparison panel."
        ),
    )
    parser.add_argument(
        "--puct-external-confirmation-only",
        action="store_true",
        help=(
            "Use PUCT's matched-seed shortlist tournament but omit its "
            "internal validation/audit; the benchmark's common panel is the "
            "only confirmation."
        ),
    )
    parser.add_argument("--puct-root-screening-top-k", type=int, default=8)
    parser.add_argument("--puct-root-screening-rollouts", type=int, default=8)
    parser.add_argument("--puct-root-screening-rounds", type=int, default=3)
    parser.add_argument(
        "--puct-root-screening-min-survivors", type=int, default=4
    )
    parser.add_argument("--opponent-candidates", type=int, default=32)
    parser.add_argument("--learned-opponent-belief", action="store_true")
    parser.add_argument("--opponent-synthetic-weight", type=float, default=0.5)
    parser.add_argument("--opponent-persistence-weight", type=float, default=0.25)
    parser.add_argument("--opponent-reservoir-weight", type=float, default=0.25)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--seed", type=int, default=2710001)
    return parser.parse_args()


def equal_compute_puct_config(args, selection_budget):
    """Build the selection planner whose efficiency is being measured."""

    return PUCTConfig(
        max_simulations=4096,
        max_simulator_transitions=max(
            1,
            int(round(selection_budget * args.puct_transition_budget_scale)),
        ),
        max_rollout_steps=180,
        horizon_battles=args.horizon_battles,
        world_pool_size=32,
        min_root_visits=args.puct_min_root_visits,
        cpuct=args.puct_cpuct,
        prior_floor=args.puct_prior_floor,
        root_noise_fraction=0.0,
        leaf_value_weight=args.leaf_value_weight,
        validation_rollouts=args.validation_rollouts,
        confidence_z=1.96,
        minimum_improvement=0.0,
        prune_affordable_freezes=False,
        exhaustive_root_probability=0.0,
        selection_only=args.puct_selection_only,
        external_confirmation_only=args.puct_external_confirmation_only,
        root_screening_top_k=args.puct_root_screening_top_k,
        root_screening_rollouts=args.puct_root_screening_rollouts,
        root_screening_rounds=args.puct_root_screening_rounds,
        root_screening_min_survivors=(
            args.puct_root_screening_min_survivors
        ),
    )


def collect_opportunity_state(
    policy,
    reservoir,
    rng,
    device,
    horizon,
    match_limit,
    minimum_turn,
    maximum_turn,
    gold_band="any",
):
    """Sample across the shop phase, including end-of-shop decisions."""
    selected = mine_match_opportunity(
        policy, rng, device, minimum_turn, maximum_turn, gold_band
    )
    if selected is None:
        return None
    env, memories, side, reasons = selected
    state = PublicSearchState.from_core(env.core, side)
    matches = reservoir.match(
        state.core.previous_opponent_state,
        observed_enemy=state.core.previous_agent_state,
        older_observed=state.core.previous_previous_opponent_state,
        older_observed_enemy=state.core.previous_previous_agent_state,
        current_turn=state.core.turn,
        horizon=horizon,
        limit=match_limit,
    )
    if not matches:
        return None
    state.set_opponent_trajectories(
        [match.teams for match in matches],
        [match.trajectory_id for match in matches],
    )
    state.set_opponent_transition_model(reservoir, candidate_limit=match_limit)
    return state, memories[side], len(matches), side, reasons


def mean_ci(values):
    values = np.asarray(values, dtype=np.float64)
    mean = float(values.mean())
    if len(values) < 2:
        return {
            "mean": mean,
            "standardError": None,
            "lowerBound95": None,
            "upperBound95": None,
        }
    standard_error = float(values.std(ddof=1) / math.sqrt(len(values)))
    return {
        "mean": mean,
        "standardError": standard_error,
        "lowerBound95": mean - 1.96 * standard_error,
        "upperBound95": mean + 1.96 * standard_error,
    }


def main():
    args = parse_args()
    if args.states <= 0:
        raise SystemExit("--states must be positive")
    if args.minimum_turn < 1 or args.maximum_turn < args.minimum_turn:
        raise SystemExit("invalid turn window")
    if args.comparison_rollouts < 2:
        raise SystemExit("--comparison-rollouts must be at least two")
    if args.puct_transition_budget_scale <= 0:
        raise SystemExit("--puct-transition-budget-scale must be positive")
    if args.puct_selection_only and args.puct_external_confirmation_only:
        raise SystemExit("PUCT proposal modes are mutually exclusive")
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
    value_calibrator = None
    calibration_metadata = None
    if args.critic_bias_calibration is not None:
        calibration_report = json.loads(
            args.critic_bias_calibration.read_text()
        )
        value_calibrator = AdditiveCriticCalibrator.from_report(
            calibration_report, expected_revision=SIMULATOR_REVISION
        )
        calibration_metadata = {
            "path": str(args.critic_bias_calibration),
            "sha256": hashlib.sha256(
                args.critic_bias_calibration.read_bytes()
            ).hexdigest(),
            "method": calibration_report.get("method"),
        }

    rows = []
    attempts = 0
    while len(rows) < args.states:
        attempts += 1
        if attempts > args.states * 1000:
            raise RuntimeError("Could not fill the requested decision strata")
        reasons = ()
        if args.state_mode == "opportunity":
            collected = collect_opportunity_state(
                policy,
                reservoir,
                rng,
                device,
                args.horizon_battles,
                args.opponent_candidates,
                    args.minimum_turn,
                    args.maximum_turn,
                    gold_band=GOLD_BANDS[len(rows) % len(GOLD_BANDS)],
            )
            if collected is not None:
                state, memory, match_count, side, reasons = collected
                target_turn = int(state.core.turn)
                collected = state, memory, match_count
        else:
            target_turn = int(
                rng.integers(args.minimum_turn, args.maximum_turn + 1)
            )
            side = int(rng.integers(0, 2))
            collected = collect_state(
                policy,
                reservoir,
                rng,
                device,
                target_turn,
                side,
                args.horizon_battles,
                args.opponent_candidates,
            )
        if collected is None:
            continue
        state, memory, match_count = collected
        if args.learned_opponent_belief:
            state.enable_learned_opponent_belief(
                OpponentBeliefMix(
                    synthetic_weight=args.opponent_synthetic_weight,
                    persistence_weight=args.opponent_persistence_weight,
                    reservoir_weight=args.opponent_reservoir_weight,
                )
            )
        search_seed = int(rng.integers(0, 2**63 - 1))
        validation_seed = int(rng.integers(0, 2**63 - 1))

        root_evaluator = NeuralSearchEvaluator(
            policy,
            device=device,
            chunk_size=args.batch_size,
            value_calibrator=value_calibrator,
        )
        root_search = CounterfactualRolloutSearch(
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
        root = root_search.search(
            state,
            memory,
            seed=search_seed,
            validation_seed=validation_seed,
        )
        root_wall = time.perf_counter() - started
        selection_budget = root.selection_simulator_transitions

        puct_evaluator = NeuralSearchEvaluator(
            policy,
            device=device,
            chunk_size=args.batch_size,
            value_calibrator=value_calibrator,
        )
        puct_search = StochasticPUCTSearch(
            puct_evaluator,
            equal_compute_puct_config(args, selection_budget),
        )
        started = time.perf_counter()
        puct = puct_search.search(
            state,
            memory,
            seed=search_seed,
            validation_seed=validation_seed,
        )
        puct_wall = time.perf_counter() - started

        # Each planner consumes its selection/validation RNG differently, so
        # subtracting their internal validation arrays is not a paired
        # comparison. Re-evaluate all relevant actions on fresh common worlds
        # that neither planner used for selection.
        comparison_seeds = rng.integers(
            0, 2**63 - 1, size=args.comparison_rollouts
        )
        rollout_cache = {}
        rollout_tree_actions = {}

        def independent_returns(action, *, use_tree):
            key = (action, bool(use_tree))
            if key not in rollout_cache:
                records = puct_search._confirmation_rollouts(
                    state,
                    memory,
                    (action,),
                    comparison_seeds,
                    use_tree=use_tree,
                )
                rollout_cache[key] = np.asarray(
                    [record["total"] for record in records],
                    dtype=np.float64,
                )
                rollout_tree_actions[key] = float(
                    np.mean([record.get("tree_actions", 0) for record in records])
                )
            return rollout_cache[key]

        raw_action = puct.policy_action
        raw_returns = independent_returns(raw_action, use_tree=False)
        root_proposed_returns = independent_returns(
            root.best_action, use_tree=False
        )
        puct_proposed_returns = independent_returns(
            puct.proposed_action, use_tree=True
        )
        puct_proposed_policy_continuation_returns = independent_returns(
            puct.proposed_action, use_tree=False
        )
        root_selected = root.best_action if root.accepted else root.policy_action
        root_selected_returns = independent_returns(
            root_selected, use_tree=False
        )
        puct_confirmed = bool(
            puct.accepted
            and puct.audit_return_lower_bound is not None
            and puct.audit_return_lower_bound > 0.0
        )
        puct_selected = puct.proposed_action if puct_confirmed else raw_action
        puct_selected_returns = independent_returns(
            puct_selected, use_tree=puct_confirmed
        )
        root_differences = root_selected_returns - raw_returns
        puct_differences = puct_selected_returns - raw_returns
        puct_minus_root = puct_differences - root_differences
        root_proposed_differences = root_proposed_returns - raw_returns
        puct_proposed_differences = puct_proposed_returns - raw_returns
        puct_proposed_policy_continuation_differences = (
            puct_proposed_policy_continuation_returns - raw_returns
        )
        puct_planning_continuation_lift = (
            puct_proposed_returns
            - puct_proposed_policy_continuation_returns
        )
        puct_minus_root_proposed = (
            puct_proposed_returns - root_proposed_returns
        )
        row = {
            "state": len(rows),
            "turn": target_turn,
            "side": side,
            "stateMode": args.state_mode,
            "opportunityReasons": list(reasons),
            "samplingGoldBand": (
                GOLD_BANDS[len(rows) % len(GOLD_BANDS)]
                if args.state_mode == "opportunity" else None
            ),
            "opponentMatches": match_count,
            "publicState": view_snapshot(state.core, 0),
            "selectionTransitionBudget": selection_budget,
            "root": {
                "accepted": root.accepted,
                "policyAction": list(root.policy_action.as_tuple()),
                "proposedAction": list(root.best_action.as_tuple()),
                "selectedAction": list(root_selected.as_tuple()),
                "policyActionDetail": action_snapshot(
                    state.core.agent, root.policy_action
                ),
                "proposedActionDetail": action_snapshot(
                    state.core.agent, root.best_action
                ),
                "selectedActionDetail": action_snapshot(
                    state.core.agent, root_selected
                ),
                "returnAdvantageVsRaw": float(root_differences.mean()),
                "independentProposedReturnAdvantageVsRaw": float(
                    root_proposed_differences.mean()
                ),
                "independentProposedDifferencesVsRaw": list(
                    map(float, root_proposed_differences)
                ),
                "proposedReturnImprovement": (
                    root.validation_return_improvement
                ),
                "proposedReturnLowerBound": (
                    root.validation_return_lower_bound
                ),
                "horizonBattleAdvantageVsRaw": (
                    root.validation_horizon_battle_score_improvement
                    if root.accepted
                    else 0.0
                ),
                "selectionSimulatorTransitions": (
                    root.selection_simulator_transitions
                ),
                "phaseWallSeconds": {
                    "selection": root.selection_wall_seconds,
                    "validation": root.validation_wall_seconds,
                    "audit": root.audit_wall_seconds,
                },
                "wallSeconds": root_wall,
                "evaluator": root_evaluator.metrics(),
            },
            "puct": {
                "accepted": puct.accepted,
                "independentlyConfirmed": puct_confirmed,
                "candidateEligible": puct.candidate_eligible,
                "selectionOnly": args.puct_selection_only,
                "externalConfirmationOnly": (
                    args.puct_external_confirmation_only
                ),
                "policyAction": list(puct.policy_action.as_tuple()),
                "treeProposedAction": list(
                    puct.tree_proposed_action.as_tuple()
                ),
                "proposedAction": list(puct.proposed_action.as_tuple()),
                "selectedAction": list(puct_selected.as_tuple()),
                "policyActionDetail": action_snapshot(
                    state.core.agent, puct.policy_action
                ),
                "treeProposedActionDetail": action_snapshot(
                    state.core.agent, puct.tree_proposed_action
                ),
                "proposedActionDetail": action_snapshot(
                    state.core.agent, puct.proposed_action
                ),
                "selectedActionDetail": action_snapshot(
                    state.core.agent, puct_selected
                ),
                "returnAdvantageVsRaw": float(puct_differences.mean()),
                "independentProposedReturnAdvantageVsRaw": float(
                    puct_proposed_differences.mean()
                ),
                "independentProposedDifferencesVsRaw": list(
                    map(float, puct_proposed_differences)
                ),
                "independentProposedRootOnlyReturnAdvantageVsRaw": float(
                    puct_proposed_policy_continuation_differences.mean()
                ),
                "independentProposedRootOnlyDifferencesVsRaw": list(
                    map(float, puct_proposed_policy_continuation_differences)
                ),
                "independentPlanningContinuationLift": float(
                    puct_planning_continuation_lift.mean()
                ),
                "independentPlanningContinuationLiftDifferences": list(
                    map(float, puct_planning_continuation_lift)
                ),
                "independentTreeActionsMean": rollout_tree_actions[
                    (puct.proposed_action, True)
                ],
                "proposedReturnImprovement": (
                    puct.validation_return_improvement
                ),
                "proposedReturnLowerBound": (
                    puct.validation_return_lower_bound
                ),
                "validationRolloutsUsed": (
                    puct.validation_rollouts_used
                ),
                "validationTreeActionsMean": (
                    puct.validation_tree_actions_mean
                ),
                "outerAuditReturnImprovement": (
                    puct.audit_return_improvement
                ),
                "outerAuditReturnLowerBound": (
                    puct.audit_return_lower_bound
                ),
                "outerAuditTreeActionsMean": (
                    puct.audit_tree_actions_mean
                ),
                "horizonBattleAdvantageVsRaw": (
                    puct.validation_horizon_battle_score_improvement
                    if puct.accepted
                    else 0.0
                ),
                "simulations": puct.simulations,
                "simulatorTransitions": puct.simulator_transitions,
                "treeNodes": puct.tree_nodes,
                "meanTreeDepth": puct.mean_tree_depth,
                "maxTreeDepth": puct.max_tree_depth,
                "phaseWallSeconds": {
                    "tree": puct.tree_wall_seconds,
                    "screening": puct.screening_wall_seconds,
                    "validation": puct.validation_wall_seconds,
                    "audit": puct.audit_wall_seconds,
                },
                "phaseSimulatorTransitions": {
                    "tree": puct.simulator_transitions,
                    "screening": puct.screening_simulator_transitions,
                    "validation": puct.validation_simulator_transitions,
                    "audit": puct.audit_simulator_transitions,
                },
                "fullCandidateCount": puct.full_candidate_count,
                "prunedCandidateCount": puct.pruned_candidate_count,
                "exhaustiveRoot": puct.exhaustive_root,
                "proposedExcludedByPruning": (
                    puct.proposed_excluded_by_pruning
                ),
                "trainingTargets": len(puct.targets),
                "wallSeconds": puct_wall,
                "evaluator": puct_evaluator.metrics(),
            },
            "puctMinusRootReturn": float(puct_minus_root.mean()),
            "puctMinusRootValidationDifferences": list(
                map(float, puct_minus_root)
            ),
            "puctMinusRootProposedReturn": float(
                puct_minus_root_proposed.mean()
            ),
            "puctMinusRootProposedDifferences": list(
                map(float, puct_minus_root_proposed)
            ),
            "comparisonRollouts": args.comparison_rollouts,
        }
        rows.append(row)
        print(json.dumps(row, sort_keys=True), flush=True)

    root_advantages = [row["root"]["returnAdvantageVsRaw"] for row in rows]
    puct_advantages = [row["puct"]["returnAdvantageVsRaw"] for row in rows]
    differences = [row["puctMinusRootReturn"] for row in rows]
    proposal_differences = [
        row["puctMinusRootProposedReturn"] for row in rows
    ]
    summary = {
        "states": len(rows),
        "rootAccepted": sum(row["root"]["accepted"] for row in rows),
        "puctAccepted": sum(row["puct"]["accepted"] for row in rows),
        "rootVsRaw": mean_ci(root_advantages),
        "puctVsRaw": mean_ci(puct_advantages),
        "puctMinusRoot": mean_ci(differences),
        "puctMinusRootProposed": mean_ci(proposal_differences),
        "meanRootWallSeconds": float(
            np.mean([row["root"]["wallSeconds"] for row in rows])
        ),
        "meanPuctWallSeconds": float(
            np.mean([row["puct"]["wallSeconds"] for row in rows])
        ),
        "meanBudgetUtilization": float(
            np.mean(
                [
                    row["puct"]["simulatorTransitions"]
                    / row["selectionTransitionBudget"]
                    for row in rows
                ]
            )
        ),
    }
    lower = summary["puctMinusRootProposed"]["lowerBound95"]
    upper = summary["puctMinusRootProposed"]["upperBound95"]
    if lower is not None and lower > 0.0:
        recommendation = (
            "puct-more-transition-efficient"
            if args.puct_selection_only
            else "puct-higher-return-at-unequal-total-work"
        )
    elif upper is not None and upper <= 0.0:
        recommendation = (
            "root-more-transition-efficient"
            if args.puct_selection_only
            else "root-higher-return-at-unequal-total-work"
        )
    else:
        recommendation = (
            "expand-equal-transition-ablation"
            if args.puct_selection_only
            else "run-selection-only-equal-transition-ablation"
        )
    payload = {
        "puctVersion": PUCT_VERSION,
        "searchVersion": SEARCH_VERSION,
        "simulatorRevision": SIMULATOR_REVISION,
        "checkpoint": str(args.checkpoint),
        "stateMode": args.state_mode,
        "miningVersion": MINING_VERSION,
        "turnWindow": [args.minimum_turn, args.maximum_turn],
        "plannerConfig": {
            "cpuct": args.puct_cpuct,
            "priorFloor": args.puct_prior_floor,
            "minRootVisits": args.puct_min_root_visits,
            "transitionBudgetScale": args.puct_transition_budget_scale,
            "selectionOnly": args.puct_selection_only,
            "externalConfirmationOnly": args.puct_external_confirmation_only,
            "rootScreeningTopK": args.puct_root_screening_top_k,
            "rootScreeningRollouts": args.puct_root_screening_rollouts,
            "rootScreeningRounds": args.puct_root_screening_rounds,
            "rootScreeningMinSurvivors": (
                args.puct_root_screening_min_survivors
            ),
            "horizonBattles": args.horizon_battles,
            "leafValueWeight": args.leaf_value_weight,
            "criticBiasCalibration": calibration_metadata,
            "learnedOpponentBelief": args.learned_opponent_belief,
            "opponentBeliefVersion": (
                OPPONENT_BELIEF_VERSION
                if args.learned_opponent_belief
                else None
            ),
            "opponentBeliefWeights": {
                "synthetic": args.opponent_synthetic_weight,
                "persistence": args.opponent_persistence_weight,
                "reservoir": args.opponent_reservoir_weight,
            },
        },
        "states": len(rows),
        "samplingAttempts": attempts,
        "summary": summary,
        "recommendation": recommendation,
        "rows": rows,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"summary": summary, "recommendation": recommendation}, sort_keys=True))


if __name__ == "__main__":
    main()
