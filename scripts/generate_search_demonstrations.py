#!/usr/bin/env python3
"""Generate policy-improvement labels with public-information rollout search."""

from __future__ import annotations

import argparse
from dataclasses import asdict
from collections import Counter
from contextlib import nullcontext
import hashlib
import json
from pathlib import Path

import numpy as np
import torch

from sapai.turtle_pack import SIMULATOR_REVISION
from sapai_gym1.rl.joint_demonstrations import serialize_action, make_joint_example, load_quality_gate
from sapai_gym1.rl import (
    ActionType,
    AutoregressiveAction,
    EntityTransformerConfig,
    EntityTransformerPolicy,
    FlatActionCodec,
    ParallelSelfPlayEnv,
)
from sapai_gym1.rl.distributed import _completed_match_result
from sapai_gym1.rl.game_trace import action_snapshot, view_snapshot
from sapai_gym1.rl.opponent_reservoir import OpponentTrajectoryReservoir
from sapai_gym1.rl.rollout_search import (
    SEARCH_VERSION,
    CounterfactualRolloutSearch,
    NeuralSearchEvaluator,
    PublicSearchState,
    RolloutSearchConfig,
    soft_search_targets,
)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("--opponent-checkpoint", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--holdout-output", type=Path)
    parser.add_argument("--holdout-fraction", type=float, default=0.0)
    parser.add_argument("--summary-output", type=Path)
    parser.add_argument("--games", type=int, default=16)
    parser.add_argument("--max-search-attempts", type=int, default=64)
    parser.add_argument("--max-searches-per-player-turn", type=int, default=2)
    parser.add_argument("--rollouts-per-action", type=int, default=4)
    parser.add_argument("--confirmation-rollouts", type=int, default=16)
    parser.add_argument("--confirmation-top-k", type=int, default=4)
    parser.add_argument("--validation-rollouts", type=int, default=32)
    parser.add_argument("--audit-rollouts", type=int, default=32)
    parser.add_argument("--horizon-battles", type=int, default=5)
    parser.add_argument("--max-continuation-steps", type=int, default=60)
    parser.add_argument("--minimum-improvement", type=float, default=0.02)
    parser.add_argument("--confidence-z", type=float, default=1.96)
    parser.add_argument("--exhaustive-root-probability", type=float, default=0.10)
    parser.add_argument("--leaf-value-weight", type=float, default=1.0)
    parser.add_argument("--target-temperature", type=float, default=0.1)
    parser.add_argument("--target-top-k", type=int, default=4)
    parser.add_argument("--minimum-target-probability", type=float, default=0.02)
    parser.add_argument("--opportunity-probability", type=float, default=1.0)
    parser.add_argument("--general-probability", type=float, default=0.03)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--search-batch-size", type=int, default=64)
    parser.add_argument("--opponent-reservoir", type=Path)
    parser.add_argument("--opponent-candidates", type=int, default=32)
    parser.add_argument("--require-opponent-reservoir", action="store_true")
    parser.add_argument("--prune-affordable-freezes", action="store_true")
    parser.add_argument("--seed", type=int, default=20260902)
    parser.add_argument("--deterministic-play", action="store_true")
    parser.add_argument("--include-unaccepted", action="store_true")
    parser.add_argument(
        "--compact",
        action="store_true",
        help="Omit per-action scores and rollout traces from training JSONL.",
    )
    parser.add_argument(
        "--audit-output",
        type=Path,
        help="Optional one-record-per-search audit JSONL with full traces.",
    )
    parser.add_argument("--allow-simulator-mismatch", action="store_true")
    parser.add_argument('--diagnostic-only', action='store_true')
    parser.add_argument('--search-quality-gate', type=Path)
    return parser.parse_args()


def load_policy(path, device, allow_mismatch):
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    revision = checkpoint.get("simulator_revision")
    if revision != SIMULATOR_REVISION and not allow_mismatch:
        raise SystemExit(
            "Checkpoint simulator revision mismatch: "
            f"checkpoint={revision!r}, runtime={SIMULATOR_REVISION!r}"
        )
    policy = EntityTransformerPolicy(
        EntityTransformerConfig(**checkpoint["config"])
    ).to(device)
    policy.load_state_dict(checkpoint["model"])
    policy.eval()
    return policy, checkpoint


def as_batch(observation, device):
    return {
        key: torch.as_tensor(value, device=device).unsqueeze(0)
        for key, value in observation.items()
    }


def autocast_for(device):
    return (
        torch.autocast("cuda", dtype=torch.bfloat16)
        if device.type == "cuda"
        else nullcontext()
    )


@torch.inference_mode()
def policy_action(policy, observation, memory, device, deterministic):
    with autocast_for(device):
        output = policy.act(
            as_batch(observation, device),
            memory=memory.unsqueeze(0).to(device),
            deterministic=deterministic,
        )
    action = AutoregressiveAction(*map(int, output.actions[0].cpu()))
    if output.end_freeze_mask is not None:
        from sapai_gym1.rl.joint_action import JointAction
        action = JointAction(action, int(output.end_freeze_mask[0].cpu()))
    return action, output.memory[0].float().cpu()


@torch.inference_mode()
def force_action(policy, observation, memory, action, device):
    from sapai_gym1.rl.joint_action import JointAction
    from sapai_gym1.rl.decision_api import decode_decisions
    if isinstance(action, JointAction):
        with autocast_for(device):
            features = policy.encode_decision(as_batch(observation, device), memory.unsqueeze(0).to(device), forecast=False)
            _, output = decode_decisions(policy, features, decisions=(action,))
        return output.memory[0].float().cpu()
    with autocast_for(device):
        output = policy(
            as_batch(observation, device),
            memory=memory.unsqueeze(0).to(device),
            actions=torch.tensor(
                [action.as_tuple()], dtype=torch.long, device=device
            ),
        )
    return output.memory[0].float().cpu()


def step_policy_decisions(env, actions):
    """Shared generator execution; toggles never become extra policy decisions."""
    from sapai_gym1.rl.joint_action import JointAction, step_decisions
    if any(isinstance(a, JointAction) for a in actions):
        return step_decisions(env, [None if a is None else a if isinstance(a, JointAction) else JointAction(a)
                                    for a in actions]).result
    return env.step(*actions)


def opportunity_reasons(core, side):
    player = core.agent if side == 0 else core.opponent
    actions = [
        FlatActionCodec.decode(flat) for flat in core._avail_actions(side)
    ]
    reasons = []
    if player.gold == 0 and any(
        ActionType(action.action_type) is ActionType.FREEZE
        for action in actions
    ):
        reasons.append("zero-gold-freeze-decision")
    shop_names = [slot.obj.name for slot in player.shop if not slot.empty]
    shop_name_set = set(shop_names)
    if "food-sleeping-pill" in shop_names:
        reasons.append("sleeping-pill")
    if "food-mushroom" in shop_names:
        reasons.append("mushroom")
    team_names = {
        slot.pet.name for slot in player.team if not slot.empty
    }

    # These labels only decide which states deserve expensive counterfactual
    # search. They deliberately do not prescribe the action: PUCT must still
    # establish that completing (or declining) the composition is valuable in
    # the exact board, shop, and opponent context.
    if (
        "pet-shark" in team_names and "pet-fly" in shop_name_set
    ) or (
        "pet-fly" in team_names and "pet-shark" in shop_name_set
    ):
        reasons.append("synergy:shark-fly")

    hurt_pets = {"pet-blowfish", "pet-camel"}
    has_hurt_engine = (
        "pet-elephant" in team_names
        and bool(team_names & hurt_pets)
    )
    extends_hurt_engine = (
        has_hurt_engine and "pet-wolverine" in shop_name_set
    )
    starts_hurt_pair = (
        "pet-elephant" in team_names
        and bool(shop_name_set & hurt_pets)
    ) or (
        "pet-elephant" in shop_name_set
        and bool(team_names & hurt_pets)
    )
    if extends_hurt_engine or starts_hurt_pair:
        reasons.append("synergy:elephant-hurt-wolverine")

    if (
        "pet-parrot" in team_names and "pet-deer" in shop_name_set
    ) or (
        "pet-deer" in team_names and "pet-parrot" in shop_name_set
    ):
        reasons.append("synergy:parrot-deer")

    mammoth_tiger_pair = (
        "pet-mammoth" in team_names and "pet-tiger" in shop_name_set
    ) or (
        "pet-tiger" in team_names and "pet-mammoth" in shop_name_set
    )
    mammoth_tiger_owned = {
        "pet-mammoth",
        "pet-tiger",
    }.issubset(team_names)
    if mammoth_tiger_pair or (
        mammoth_tiger_owned and "food-mushroom" in shop_name_set
    ):
        reasons.append("synergy:tiger-mammoth-mushroom")

    faint_mushroom_targets = {
        "pet-deer",
        "pet-mammoth",
        "pet-rooster",
        "pet-scorpion",
    }
    if (
        "food-mushroom" in shop_name_set
        and bool(team_names & faint_mushroom_targets)
    ):
        reasons.append("synergy:mushroom-faint")

    previous_opponent = (
        core.previous_opponent_state
        if side == 0
        else core.previous_agent_state
    )
    opponent_pets = (
        []
        if previous_opponent is None
        else [
            slot.pet
            for slot in previous_opponent
            if not slot.empty
        ]
    )
    opponent_names = {pet.name for pet in opponent_pets}
    summon_infrastructure = {
        "pet-cricket",
        "pet-spider",
        "pet-sheep",
        "pet-deer",
        "pet-rooster",
        "pet-fly",
        "pet-turkey",
    }
    if (
        "pet-scorpion" in shop_name_set
        and any(
            max(int(pet.attack), int(pet.health)) >= 35
            for pet in opponent_pets
        )
    ):
        reasons.append("counter:scorpion-tall")
    if (
        "pet-skunk" in shop_name_set
        and any(int(pet.health) >= 35 for pet in opponent_pets)
    ):
        reasons.append("counter:skunk-tall")
    if (
        "pet-rhino" in shop_name_set
        and bool(opponent_names & summon_infrastructure)
    ):
        reasons.append("counter:rhino-summons")
    if (
        "pet-rhino" in opponent_names
        and bool((team_names | shop_name_set) & summon_infrastructure)
    ):
        reasons.append("counter:avoid-summons-rhino")
    back_opponent = opponent_pets[-1] if opponent_pets else None
    if (
        "pet-crocodile" in shop_name_set
        and back_opponent is not None
        and back_opponent.name == "pet-tiger"
    ):
        reasons.append("counter:crocodile-exposed-tiger")
    if (
        "pet-leopard" in shop_name_set
        and len(opponent_pets) >= 3
        and sum(pet.status == "status-melon-armor" for pet in opponent_pets)
        <= 1
    ):
        reasons.append("counter:leopard-weak-melon")
    fragile_engines = {"pet-tiger", "pet-turkey", "pet-fly", "pet-snake"}
    if (
        {"pet-elephant", "pet-blowfish"}.issubset(opponent_names)
        and bool(team_names & fragile_engines)
    ):
        reasons.append("counter:elephant-blowfish-fragile-engine")
    if any(
        slot.slot_type in {"pet", "levelup"}
        and slot.obj.name in team_names
        for slot in player.shop
    ):
        reasons.append("matching-pet")
    if len(player.team) == 5 and any(
        slot.cost <= player.gold for slot in player.shop
    ):
        reasons.append("full-board-pivot")
    team_pets = [slot.pet for slot in player.team if not slot.empty]
    team_name_counts = Counter(pet.name for pet in team_pets)
    if core.turn == 2 and len(player.team) < 5:
        for name, count in team_name_counts.items():
            same = [pet for pet in team_pets if pet.name == name]
            if count >= 2 and any(
                pet.level == 1 and pet.experience >= 1 for pet in same
            ):
                reasons.append("turn-two-on-board-triple")
                break
    if any(
        not slot.empty
        and slot.obj.name == "food-chocolate"
        and slot.cost > player.gold
        and not slot.frozen
        for slot in player.shop
    ):
        reasons.append("subcost-premium-freeze")
    if (
        core.turn >= 3
        and "pet-ant" not in team_name_counts
        and any(
            not slot.empty and slot.obj.name == "pet-ant"
            for slot in player.shop
        )
        and any(
            not slot.empty
            and slot.slot_type in {"pet", "levelup"}
            and slot.obj.tier >= 2
            for slot in player.shop
        )
    ):
        reasons.append("post-window-tier-one")
    if (
        team_name_counts.get("pet-snail", 0) >= 2
        and len(player.team) < 5
        and not player.lf_winner
    ):
        reasons.append("active-snail-consolidation")
    return reasons


def serialize_observation(observation):
    return {key: value.tolist() for key, value in observation.items()}


def action_key(action):
    from sapai_gym1.rl.joint_demonstrations import action_key as canonical_key
    return canonical_key(action)


def finite_or_none(value):
    return (
        float(value)
        if value is not None and np.isfinite(value)
        else None
    )


def serialize_rollout(trace):
    return {
        "actions": [serialize_action(action) for action in trace.actions],
        "rewards": list(trace.rewards),
        "battleResults": list(trace.battle_results),
        "leafValue": trace.leaf_value,
        "totalReturn": trace.total_return,
        "terminated": trace.terminated,
        "battles": trace.battles,
        "opponentTrajectoryIndex": trace.opponent_trajectory_index,
        "opponentTrajectoryIds": list(trace.opponent_trajectory_ids),
        "nestedSearches": trace.nested_searches,
    }


def search_example(
    observation,
    result,
    target_action,
    target_probability,
    target_rank,
    initial_memory,
    opponent_matches,
    include_details,
    split,
    *,
    game_index,
    side,
    turn,
    decision,
    reasons,
    checkpoint_path,
    checkpoint_sha256,
    policy=None,
    observation_prefix=None,
    quality_gate=None,
    planner_config=None,
):
    example = {
        "schema_version": 1,
        "replay_id": (
            f"search-game-{game_index:06d}-p{side}-{decision:04d}"
            f"-target-{target_rank:02d}"
        ),
        "perspective": f"p{side}",
        "turn": int(turn),
        "decision": 0,
        "observation": serialize_observation(observation),
        "initialMemory": initial_memory.float().cpu().tolist(),
        "action": serialize_action(target_action),
        "teacherWeight": float(target_probability),
        "battle_outcome": None,
        "final_result": None,
        "validation": {
            "exactSimulatorState": True,
            "simulatorRevision": SIMULATOR_REVISION,
            "searchVersion": SEARCH_VERSION,
            "publicInformationOnly": True,
        },
        "teacher": {
            "plannerConfig": planner_config or {},
            "name": "counterfactual-rollout-search",
            "version": SEARCH_VERSION,
            "checkpoint": str(checkpoint_path),
            "checkpointSha256": checkpoint_sha256,
            "accepted": bool(result.accepted),
            "opportunityReasons": reasons,
            "policyAction": serialize_action(result.policy_action),
            "improvement": result.improvement,
            "improvementStandardError": finite_or_none(
                result.improvement_standard_error
            ),
            "improvementLowerBound": finite_or_none(
                result.improvement_lower_bound
            ),
            "firstBattleScoreImprovement": (
                result.first_battle_score_improvement
            ),
            "horizonBattleScoreImprovement": (
                result.horizon_battle_score_improvement
            ),
            "validationReturnImprovement": (
                result.validation_return_improvement
            ),
            "validationReturnLowerBound": (
                result.validation_return_lower_bound
            ),
            "validationFirstBattleScoreImprovement": (
                result.validation_first_battle_score_improvement
            ),
            "validationHorizonBattleScoreImprovement": (
                result.validation_horizon_battle_score_improvement
            ),
            "outerAuditReturnImprovement": (
                result.audit_return_improvement
            ),
            "outerAuditReturnLowerBound": (
                result.audit_return_lower_bound
            ),
            "outerAuditFirstBattleScoreImprovement": (
                result.audit_first_battle_score_improvement
            ),
            "outerAuditHorizonBattleScoreImprovement": (
                result.audit_horizon_battle_score_improvement
            ),
            "searchBestAction": serialize_action(result.best_action),
            "searchTargetRank": int(target_rank),
            "searchTargetProbability": float(target_probability),
            "split": split,
            "opponentTrajectoryMatches": opponent_matches,
            "exhaustiveRoot": result.exhaustive_root,
            "fullCandidateCount": result.full_candidate_count,
            "prunedCandidateCount": result.pruned_candidate_count,
            "screeningBestExcludedByPruning": (
                result.screening_best_excluded_by_pruning
            ),
            "prunedActionCounts": result.pruned_action_counts,
            "confirmedBestExcludedByPruning": (
                result.confirmed_best_excluded_by_pruning
            ),
            "validatedPrunedWinner": result.validated_pruned_winner,
            "selectionSimulatorTransitions": (
                result.selection_simulator_transitions
            ),
            "totalSimulatorTransitions": result.total_simulator_transitions,
        },
    }
    if include_details:
        example["teacher"].update(
            {
                "scores": {
                    action_key(action): score
                    for action, score in result.scores.items()
                },
                "standardErrors": {
                    action_key(action): finite_or_none(error)
                    for action, error in result.standard_errors.items()
                },
                "priors": {
                    action_key(action): prior
                    for action, prior in result.priors.items()
                },
                "bestRollouts": [
                    serialize_rollout(trace)
                    for trace in result.rollouts[result.best_action]
                ],
                "policyRollouts": [
                    serialize_rollout(trace)
                    for trace in result.rollouts[result.policy_action]
                ],
            }
        )
    return make_joint_example(example, target_action, observation, initial_memory,
        policy=policy, checkpoint_sha256=checkpoint_sha256,
        omitted_mass=getattr(result, 'omitted_end_mask_probability_mass', 0.),
        observation_prefix=observation_prefix, is_root=True, quality_gate=quality_gate)


def audit_record(
    result,
    *,
    public_state,
    game_index,
    side,
    turn,
    decision,
    reasons,
    matches,
    split,
):
    return {
        "game": game_index,
        "side": side,
        "turn": turn,
        "decision": decision,
        "split": split,
        "reasons": reasons,
        "publicState": view_snapshot(public_state.core, 0),
        "policyActionDetail": action_snapshot(
            public_state.core.agent, result.policy_action
        ),
        "bestActionDetail": action_snapshot(
            public_state.core.agent, result.best_action
        ),
        "opponentTrajectoryMatches": matches,
        "policyAction": serialize_action(result.policy_action),
        "bestAction": serialize_action(result.best_action),
        "screeningBestAction": serialize_action(result.screening_best_action),
        "accepted": result.accepted,
        "improvement": result.improvement,
        "improvementStandardError": finite_or_none(
            result.improvement_standard_error
        ),
        "improvementLowerBound": finite_or_none(
            result.improvement_lower_bound
        ),
        "firstBattleScoreImprovement": (
            result.first_battle_score_improvement
        ),
        "horizonBattleScoreImprovement": (
            result.horizon_battle_score_improvement
        ),
        "policyFirstBattleScore": result.policy_first_battle_score,
        "bestFirstBattleScore": result.best_first_battle_score,
        "policyHorizonBattleScore": result.policy_horizon_battle_score,
        "bestHorizonBattleScore": result.best_horizon_battle_score,
        "validationReturnImprovement": result.validation_return_improvement,
        "validationReturnStandardError": (
            result.validation_return_standard_error
        ),
        "validationReturnLowerBound": result.validation_return_lower_bound,
        "validationFirstBattleScoreImprovement": (
            result.validation_first_battle_score_improvement
        ),
        "validationHorizonBattleScoreImprovement": (
            result.validation_horizon_battle_score_improvement
        ),
        "outerAuditReturnImprovement": result.audit_return_improvement,
        "outerAuditReturnStandardError": (
            result.audit_return_standard_error
        ),
        "outerAuditReturnLowerBound": result.audit_return_lower_bound,
        "outerAuditFirstBattleScoreImprovement": (
            result.audit_first_battle_score_improvement
        ),
        "outerAuditHorizonBattleScoreImprovement": (
            result.audit_horizon_battle_score_improvement
        ),
        "scores": {
            action_key(action): score for action, score in result.scores.items()
        },
        "screeningScores": {
            action_key(action): score
            for action, score in result.screening_scores.items()
        },
        "exhaustiveRoot": result.exhaustive_root,
        "fullCandidateCount": result.full_candidate_count,
        "prunedCandidateCount": result.pruned_candidate_count,
        "screeningBestExcludedByPruning": (
            result.screening_best_excluded_by_pruning
        ),
        "prunedActionCounts": result.pruned_action_counts,
        "confirmedBestExcludedByPruning": (
            result.confirmed_best_excluded_by_pruning
        ),
        "validatedPrunedWinner": result.validated_pruned_winner,
        "selectionSimulatorTransitions": (
            result.selection_simulator_transitions
        ),
        "totalSimulatorTransitions": result.total_simulator_transitions,
        "priors": {
            action_key(action): prior for action, prior in result.priors.items()
        },
        "bestRollouts": [
            serialize_rollout(trace)
            for trace in result.rollouts[result.best_action]
        ],
        "policyRollouts": [
            serialize_rollout(trace)
            for trace in result.rollouts[result.policy_action]
        ],
    }


def main():
    args = parse_args()
    if args.games <= 0 or args.max_search_attempts <= 0:
        raise SystemExit("--games and --max-search-attempts must be positive")
    if args.target_temperature < 0:
        raise SystemExit("--target-temperature must be non-negative")
    if args.target_top_k <= 0:
        raise SystemExit("--target-top-k must be positive")
    if args.opponent_candidates <= 0:
        raise SystemExit("--opponent-candidates must be positive")
    if not 0.0 <= args.minimum_target_probability < 1.0:
        raise SystemExit("--minimum-target-probability must be in [0, 1)")
    if not 0.0 <= args.holdout_fraction < 1.0:
        raise SystemExit("--holdout-fraction must be in [0, 1)")
    if args.holdout_fraction > 0 and args.holdout_output is None:
        raise SystemExit(
            "--holdout-fraction requires --holdout-output"
        )
    for probability in (
        args.opportunity_probability,
        args.general_probability,
    ):
        if not 0.0 <= probability <= 1.0:
            raise SystemExit("search probabilities must be in [0, 1]")

    torch.set_num_threads(1)
    device = torch.device(
        args.device
        if torch.cuda.is_available() or args.device == "cpu"
        else "cpu"
    )
    candidate, candidate_checkpoint = load_policy(
        args.checkpoint, device, args.allow_simulator_mismatch
    )
    opponent_path = args.opponent_checkpoint or args.checkpoint
    if opponent_path == args.checkpoint:
        opponent = candidate
        opponent_checkpoint = candidate_checkpoint
    else:
        opponent, opponent_checkpoint = load_policy(
            opponent_path, device, args.allow_simulator_mismatch
        )
    policies = (candidate, opponent)
    if any(p.config.end_freeze for p in policies):
        if args.search_quality_gate:
            if opponent_path != args.checkpoint:
                raise SystemExit('Production joint corpus requires one validated teacher checkpoint per invocation')
            with args.checkpoint.open('rb') as stream:
                load_quality_gate(args.search_quality_gate, checkpoint_sha256=hashlib.file_digest(stream, 'sha256').hexdigest())
        elif not args.diagnostic_only:
            raise SystemExit('Joint corpus requires --diagnostic-only or an independent --search-quality-gate')
    evaluators = tuple(
        NeuralSearchEvaluator(
            policy, device=device, chunk_size=args.search_batch_size
        )
        for policy in policies
    )
    searches = tuple(
        CounterfactualRolloutSearch(
            evaluator,
            RolloutSearchConfig(
                rollouts_per_action=args.rollouts_per_action,
                confirmation_rollouts=args.confirmation_rollouts,
                confirmation_top_k=args.confirmation_top_k,
                validation_rollouts=args.validation_rollouts,
                audit_rollouts=args.audit_rollouts,
                horizon_battles=args.horizon_battles,
                max_continuation_steps=args.max_continuation_steps,
                minimum_improvement=args.minimum_improvement,
                confidence_z=args.confidence_z,
                exhaustive_root_probability=(
                    args.exhaustive_root_probability
                ),
                leaf_value_weight=args.leaf_value_weight,
                prune_affordable_freezes=args.prune_affordable_freezes,
            ),
        )
        for evaluator in evaluators
    )
    checkpoint_hashes = (
        hashlib.sha256(args.checkpoint.read_bytes()).hexdigest(),
        hashlib.sha256(opponent_path.read_bytes()).hexdigest(),
    )
    checkpoint_paths = (args.checkpoint, opponent_path)
    opponent_reservoir = (
        None
        if args.opponent_reservoir is None
        else OpponentTrajectoryReservoir.load(args.opponent_reservoir)
    )
    if args.require_opponent_reservoir and opponent_reservoir is None:
        raise SystemExit(
            "--require-opponent-reservoir needs --opponent-reservoir"
        )
    if opponent_reservoir is not None:
        reservoir_revision = opponent_reservoir.metadata.get(
            "simulatorRevision"
        )
        if reservoir_revision != SIMULATOR_REVISION:
            raise SystemExit(
                "Opponent reservoir simulator revision mismatch: "
                f"reservoir={reservoir_revision!r}, "
                f"runtime={SIMULATOR_REVISION!r}"
            )

    rng = np.random.default_rng(args.seed)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    output_stream = args.output.open("w")
    holdout_stream = None
    if args.holdout_output is not None:
        args.holdout_output.parent.mkdir(parents=True, exist_ok=True)
        holdout_stream = args.holdout_output.open("w")
    audit_stream = None
    if args.audit_output is not None:
        args.audit_output.parent.mkdir(parents=True, exist_ok=True)
        audit_stream = args.audit_output.open("w")
    train_targets = 0
    holdout_targets = 0
    attempted = 0
    accepted = 0
    game_outcomes = Counter()
    reason_counts = Counter()
    action_changes = Counter()
    exhaustive_searches = 0
    excluded_by_pruning = 0
    pruned_actions_audited = Counter()
    confirmed_pruned_winners = Counter()
    validated_pruned_winners = Counter()
    split_attempts = Counter()
    split_accepts = Counter()
    split_improvements = {"train": [], "holdout": []}

    for game_index in range(args.games):
        if attempted >= args.max_search_attempts:
            break
        env = ParallelSelfPlayEnv()
        observations = env.reset(seed=int(rng.integers(0, 2**31 - 1)))
        memories = [
            policies[side].initial_memory(1, device="cpu")[0]
            for side in range(2)
        ]
        observation_prefixes = [[], []]
        per_turn_searches = Counter()
        decision = 0
        while True:
            active = env.active_players()
            actions = [None, None]
            for side in range(2):
                if not active[side]:
                    continue
                observation = observations[side]
                reasons = opportunity_reasons(env.core, side)
                chance = (
                    args.opportunity_probability
                    if reasons
                    else args.general_probability
                )
                key = (side, env.core.turn)
                run_search = (
                    attempted < args.max_search_attempts
                    and per_turn_searches[key]
                    < args.max_searches_per_player_turn
                    and rng.random() < chance
                )
                result = None
                if run_search:
                    try:
                        public_state = PublicSearchState.from_core(
                            env.core, side
                        )
                    except ValueError:
                        public_state = None
                    if public_state is not None:
                        opponent_matches = []
                        if opponent_reservoir is not None:
                            latest_opponent = (
                                env.core.previous_opponent_state
                                if side == 0
                                else env.core.previous_agent_state
                            )
                            observed_by_opponent = (
                                env.core.previous_agent_state
                                if side == 0
                                else env.core.previous_opponent_state
                            )
                            older_latest_opponent = (
                                env.core.previous_previous_opponent_state
                                if side == 0
                                else env.core.previous_previous_agent_state
                            )
                            older_observed_by_opponent = (
                                env.core.previous_previous_agent_state
                                if side == 0
                                else env.core.previous_previous_opponent_state
                            )
                            matches = opponent_reservoir.match(
                                latest_opponent,
                                observed_enemy=observed_by_opponent,
                                older_observed=older_latest_opponent,
                                older_observed_enemy=(
                                    older_observed_by_opponent
                                ),
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
                                    opponent_reservoir,
                                    candidate_limit=args.opponent_candidates,
                                )
                                opponent_matches = [
                                    {
                                        "id": match.trajectory_id,
                                        "distance": match.distance,
                                    }
                                    for match in matches
                                ]
                            elif args.require_opponent_reservoir:
                                public_state = None
                    if public_state is not None:
                        result = searches[side].search(
                            public_state,
                            memories[side],
                            seed=int(rng.integers(0, 2**63 - 1)),
                        )
                        attempted += 1
                        split = (
                            "holdout"
                            if rng.random() < args.holdout_fraction
                            else "train"
                        )
                        split_attempts[split] += 1
                        exhaustive_searches += int(result.exhaustive_root)
                        excluded_by_pruning += int(
                            result.screening_best_excluded_by_pruning is True
                        )
                        pruned_actions_audited.update(
                            result.pruned_action_counts
                        )
                        if result.confirmed_best_excluded_by_pruning:
                            confirmed_pruned_winners.update(
                                [ActionType(result.best_action.action_type).name]
                            )
                        if result.validated_pruned_winner:
                            validated_pruned_winners.update(
                                [ActionType(result.best_action.action_type).name]
                            )
                        per_turn_searches[key] += 1
                        reason_counts.update(reasons or ["general"])
                        if result.accepted:
                            accepted += 1
                            split_accepts[split] += 1
                            split_improvements[split].append(
                                {
                                    "return": result.improvement,
                                    "firstBattleScore": (
                                        result.first_battle_score_improvement
                                    ),
                                    "horizonBattleScore": (
                                        result.horizon_battle_score_improvement
                                    ),
                                    "validationReturn": (
                                        result.validation_return_improvement
                                    ),
                                    "validationFirstBattleScore": (
                                        result.validation_first_battle_score_improvement
                                    ),
                                    "validationHorizonBattleScore": (
                                        result.validation_horizon_battle_score_improvement
                                    ),
                                    "outerAuditReturn": (
                                        result.audit_return_improvement
                                    ),
                                    "outerAuditFirstBattleScore": (
                                        result.audit_first_battle_score_improvement
                                    ),
                                    "outerAuditHorizonBattleScore": (
                                        result.audit_horizon_battle_score_improvement
                                    ),
                                }
                            )
                            action_changes[
                                f"{ActionType(result.policy_action.action_type).name}"
                                f"->{ActionType(result.best_action.action_type).name}"
                            ] += 1
                        if audit_stream is not None:
                            audit_stream.write(
                                json.dumps(
                                    audit_record(
                                        result,
                                        public_state=public_state,
                                        game_index=game_index,
                                        side=side,
                                        turn=env.core.turn,
                                        decision=decision,
                                        reasons=reasons,
                                        matches=opponent_matches,
                                        split=split,
                                    ),
                                    separators=(",", ":"),
                                )
                                + "\n"
                            )
                            audit_stream.flush()
                        if (result.accepted or args.include_unaccepted) and (not args.search_quality_gate or
                                (result.accepted and result.audit_return_lower_bound is not None and result.audit_return_lower_bound > 0)):
                            targets = soft_search_targets(
                                result,
                                temperature=args.target_temperature,
                                top_k=args.target_top_k,
                                minimum_probability=(
                                    args.minimum_target_probability
                                ),
                            )
                            for target_rank, (
                                target_action,
                                target_probability,
                            ) in enumerate(targets):
                                destination = (
                                    holdout_stream
                                    if split == "holdout"
                                    else output_stream
                                )
                                destination.write(
                                    json.dumps(
                                        search_example(
                                        observation,
                                        result,
                                        target_action,
                                        target_probability,
                                        target_rank,
                                        memories[side],
                                        opponent_matches,
                                        not args.compact,
                                        split,
                                        game_index=game_index,
                                        side=side,
                                        turn=env.core.turn,
                                        decision=decision,
                                        reasons=reasons,
                                        checkpoint_path=checkpoint_paths[side],
                                        checkpoint_sha256=(
                                            checkpoint_hashes[side]
                                        ),
                                        policy=policies[side],
                                        observation_prefix=observation_prefixes[side],
                                        quality_gate=args.search_quality_gate,
                                        planner_config=asdict(searches[side].config),
                                        ),
                                        separators=(",", ":"),
                                    )
                                    + "\n"
                                )
                                if split == "holdout":
                                    holdout_targets += 1
                                else:
                                    train_targets += 1
                            output_stream.flush()
                            if holdout_stream is not None:
                                holdout_stream.flush()
                        print(
                            f"search={attempted}/{args.max_search_attempts} "
                            f"game={game_index + 1}/{args.games} "
                            f"turn={env.core.turn} side={side} "
                            f"accepted={accepted} split={split} "
                            f"improvement={result.improvement:.4f} "
                            f"validation={result.validation_return_improvement:.4f} "
                            f"outer_audit={finite_or_none(result.audit_return_improvement)}",
                            flush=True,
                        )
                if result is not None and result.accepted:
                    actions[side] = result.best_action
                    memories[side] = force_action(
                        policies[side],
                        observation,
                        memories[side],
                        actions[side],
                        device,
                    )
                else:
                    actions[side], memories[side] = policy_action(
                        policies[side],
                        observation,
                        memories[side],
                        device,
                        args.deterministic_play,
                    )
                decision += 1
                observation_prefixes[side].append({k: np.asarray(v).copy() for k, v in observation.items()})

            step = step_policy_decisions(env, actions)
            observations = step.observations
            if not step.terminated:
                continue
            outcome = _completed_match_result(env.core)
            game_outcomes.update([{0: "p0", 1: "p1", 2: "draw"}[outcome]])
            break

    output_stream.close()
    if holdout_stream is not None:
        holdout_stream.close()
    if audit_stream is not None:
        audit_stream.close()
    output_hash = hashlib.sha256(args.output.read_bytes()).hexdigest()
    def split_summary(name):
        values = split_improvements[name]
        return {
            "attempted": int(split_attempts[name]),
            "accepted": int(split_accepts[name]),
            "weightedTargets": (
                holdout_targets if name == "holdout" else train_targets
            ),
            "meanConfirmedReturnImprovement": (
                None
                if not values
                else float(np.mean([value["return"] for value in values]))
            ),
            "meanFirstBattleScoreImprovement": (
                None
                if not values
                else float(
                    np.mean([value["firstBattleScore"] for value in values])
                )
            ),
            "meanHorizonBattleScoreImprovement": (
                None
                if not values
                else float(
                    np.mean([value["horizonBattleScore"] for value in values])
                )
            ),
            "meanValidationReturnImprovement": (
                None
                if not values
                else float(
                    np.mean([value["validationReturn"] for value in values])
                )
            ),
            "meanValidationFirstBattleScoreImprovement": (
                None
                if not values
                else float(
                    np.mean(
                        [
                            value["validationFirstBattleScore"]
                            for value in values
                        ]
                    )
                )
            ),
            "meanValidationHorizonBattleScoreImprovement": (
                None
                if not values
                else float(
                    np.mean(
                        [
                            value["validationHorizonBattleScore"]
                            for value in values
                        ]
                    )
                )
            ),
            "meanOuterAuditReturnImprovement": (
                None
                if not values
                else float(
                    np.mean([value["outerAuditReturn"] for value in values])
                )
            ),
            "meanOuterAuditFirstBattleScoreImprovement": (
                None
                if not values
                else float(
                    np.mean(
                        [
                            value["outerAuditFirstBattleScore"]
                            for value in values
                        ]
                    )
                )
            ),
            "meanOuterAuditHorizonBattleScoreImprovement": (
                None
                if not values
                else float(
                    np.mean(
                        [
                            value["outerAuditHorizonBattleScore"]
                            for value in values
                        ]
                    )
                )
            ),
        }
    summary = {
        "searchVersion": SEARCH_VERSION,
        "simulatorRevision": SIMULATOR_REVISION,
        "attempted": attempted,
        "accepted": accepted,
        "acceptanceRate": accepted / max(1, attempted),
        "rootRecallAudits": exhaustive_searches,
        "screeningBestExcludedByPruning": excluded_by_pruning,
        "pruningAudit": {
            "prunedActionsAudited": dict(
                sorted(pruned_actions_audited.items())
            ),
            "confirmedPrunedWinners": dict(
                sorted(confirmed_pruned_winners.items())
            ),
            "validatedPrunedWinners": dict(
                sorted(validated_pruned_winners.items())
            ),
        },
        "splits": {
            "train": split_summary("train"),
            "holdout": split_summary("holdout"),
        },
    }
    if args.summary_output is not None:
        args.summary_output.parent.mkdir(parents=True, exist_ok=True)
        args.summary_output.write_text(
            json.dumps(summary, indent=2, sort_keys=True) + "\n"
        )
    print(f"search_version={SEARCH_VERSION}")
    print(f"simulator_revision={SIMULATOR_REVISION}")
    print(
        f"attempted={attempted} accepted={accepted} "
        f"train_targets={train_targets} "
        f"holdout_targets={holdout_targets}"
    )
    print(f"acceptance_rate={accepted / max(1, attempted):.4f}")
    print(f"reasons={dict(sorted(reason_counts.items()))}")
    print(f"action_changes={dict(sorted(action_changes.items()))}")
    print(
        f"root_recall_audits={exhaustive_searches} "
        f"best_excluded={excluded_by_pruning}"
    )
    print(
        "pruning_audit="
        + json.dumps(summary["pruningAudit"], sort_keys=True)
    )
    print(f"game_outcomes={dict(sorted(game_outcomes.items()))}")
    print(f"sha256={output_hash}")
    print("search_summary=" + json.dumps(summary, sort_keys=True))
    print(
        "opponent_checkpoint_revision="
        f"{opponent_checkpoint.get('simulator_revision')}"
    )


if __name__ == "__main__":
    main()
