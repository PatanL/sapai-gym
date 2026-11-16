#!/usr/bin/env python3
"""Judge public search proposals in untouched interactive self-play clones."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import statistics
import time

import numpy as np
import torch

from search_state_mining import GOLD_BANDS, MINING_VERSION, mine_match_opportunity
from sapai.turtle_pack import SIMULATOR_REVISION
from sapai_gym1.rl import (
    EntityTransformerConfig,
    EntityTransformerPolicy,
    InteractiveSelfPlayEvaluator,
    OpponentBeliefMix,
    ParallelSelfPlayEnv,
)
from sapai_gym1.rl.game_trace import action_snapshot, view_snapshot
from sapai_gym1.rl.decision_manifest import restore_decision
from sapai_gym1.rl.opponent_reservoir import OpponentTrajectoryReservoir
from sapai_gym1.rl.puct_search import PUCT_VERSION, PUCTConfig, StochasticPUCTSearch
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
    parser.add_argument("--states", type=int, default=8)
    parser.add_argument("--manifest-dir", type=Path)
    parser.add_argument("--state-start", type=int, default=0)
    parser.add_argument("--minimum-turn", type=int, default=2)
    parser.add_argument("--maximum-turn", type=int, default=13)
    parser.add_argument("--truth-rollouts", type=int, default=32)
    parser.add_argument("--truth-horizon-battles", type=int, default=30)
    parser.add_argument("--search-horizon-battles", type=int, default=5)
    parser.add_argument("--opponent-candidates", type=int, default=32)
    parser.add_argument("--leaf-value-weight", type=float, default=0.5)
    parser.add_argument("--puct-transition-budget-scale", type=float, default=0.8)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--seed", type=int, default=3805401)
    return parser.parse_args()


def mean_ci(values):
    values = list(map(float, values))
    mean = statistics.mean(values)
    if len(values) < 2:
        return {
            "mean": mean,
            "standardError": None,
            "lowerBound95": None,
            "upperBound95": None,
        }
    standard_error = statistics.stdev(values) / math.sqrt(len(values))
    return {
        "mean": mean,
        "standardError": standard_error,
        "lowerBound95": mean - 1.96 * standard_error,
        "upperBound95": mean + 1.96 * standard_error,
    }


def attach_opponent_model(state, core, side, reservoir, horizon, limit):
    latest = (
        core.previous_opponent_state
        if side == 0
        else core.previous_agent_state
    )
    observed_enemy = (
        core.previous_agent_state
        if side == 0
        else core.previous_opponent_state
    )
    older = (
        core.previous_previous_opponent_state
        if side == 0
        else core.previous_previous_agent_state
    )
    older_enemy = (
        core.previous_previous_agent_state
        if side == 0
        else core.previous_previous_opponent_state
    )
    matches = reservoir.match(
        latest,
        observed_enemy=observed_enemy,
        older_observed=older,
        older_observed_enemy=older_enemy,
        current_turn=core.turn,
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


def collect_state(policy, reservoir, rng, device, args, gold_band="any"):
    for _ in range(1000):
        selected = mine_match_opportunity(
            policy, rng, device, args.minimum_turn, args.maximum_turn, gold_band
        )
        if selected is None:
            continue
        environment, memories, side, reasons = selected
        public = PublicSearchState.from_core(environment.core, side)
        matches = attach_opponent_model(
            public, environment.core, side, reservoir,
            args.search_horizon_battles, args.opponent_candidates,
        )
        if matches:
            return environment, public, memories, side, reasons, matches
    raise RuntimeError(f"No eligible {gold_band} decision found in 1000 games")


def root_planner(policy, device, batch_size, horizon, leaf_weight):
    evaluator = NeuralSearchEvaluator(
        policy, device=device, chunk_size=batch_size
    )
    return CounterfactualRolloutSearch(
        evaluator,
        RolloutSearchConfig(
            rollouts_per_action=4,
            confirmation_rollouts=16,
            confirmation_top_k=4,
            validation_rollouts=32,
            audit_rollouts=0,
            horizon_battles=horizon,
            max_continuation_steps=180,
            minimum_improvement=0.0,
            confidence_z=1.96,
            exhaustive_root_probability=0.0,
            leaf_value_weight=leaf_weight,
            max_nested_searches=0,
            prune_affordable_freezes=False,
        ),
    )


def hybrid_planner(
    policy, device, batch_size, args, transition_budget
):
    evaluator = NeuralSearchEvaluator(
        policy, device=device, chunk_size=batch_size
    )
    planner = StochasticPUCTSearch(
        evaluator,
        PUCTConfig(
            max_simulations=4096,
            max_simulator_transitions=max(
                1,
                int(round(transition_budget * args.puct_transition_budget_scale)),
            ),
            max_rollout_steps=180,
            horizon_battles=args.search_horizon_battles,
            world_pool_size=32,
            min_root_visits=2,
            cpuct=2.0,
            prior_floor=0.10,
            leaf_value_weight=args.leaf_value_weight,
            external_confirmation_only=True,
            root_screening_top_k=4,
            root_screening_rollouts=2,
            root_screening_rounds=2,
            root_screening_min_survivors=2,
            prune_affordable_freezes=False,
            exhaustive_root_probability=0.0,
        ),
    )
    return planner


def grouped_returns(results):
    grouped = {}
    for result in results:
        grouped.setdefault(result.root_action, []).append(result.total_return)
    return {
        action: np.asarray(values, dtype=np.float64)
        for action, values in grouped.items()
    }


def main():
    args = parse_args()
    if args.states <= 0 or args.truth_rollouts < 2:
        raise SystemExit("states must be positive and truth-rollouts at least two")
    torch.manual_seed(args.seed)
    rng = np.random.default_rng(args.seed)
    device = torch.device(
        args.device
        if torch.cuda.is_available() or args.device == "cpu"
        else "cpu"
    )
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    checkpoint_hash = hashlib.sha256(args.checkpoint.read_bytes()).hexdigest()
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
    truth = InteractiveSelfPlayEvaluator(
        policy,
        device=device,
        batch_size=args.batch_size,
        horizon_battles=args.truth_horizon_battles,
        max_steps=2500,
    )

    manifest = None
    if args.manifest_dir:
        manifest = json.loads((args.manifest_dir / 'manifest.json').read_text())
        if manifest['checkpointSha256'] != checkpoint_hash:
            raise SystemExit('Manifest recurrent states belong to another checkpoint')
        if args.truth_rollouts > manifest['truthRollouts']:
            raise SystemExit('Not enough untouched seeds in the manifest')
    if args.state_start < 0:
        raise SystemExit('state-start must be nonnegative')
    metadata = {
        'arguments': {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
        'checkpointSha256': checkpoint_hash,
        'reservoirSha256': hashlib.sha256(args.opponent_reservoir.read_bytes()).hexdigest(),
        'puctVersion': PUCT_VERSION, 'searchVersion': SEARCH_VERSION,
        'miningVersion': MINING_VERSION, 'simulatorRevision': SIMULATOR_REVISION,
        'scriptSha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
    }
    parts_dir = args.output.with_suffix('.parts')
    parts_dir.mkdir(parents=True, exist_ok=True)
    metadata_path = parts_dir / 'config.json'
    if metadata_path.exists() and json.loads(metadata_path.read_text()) != metadata:
        raise SystemExit('Refusing to mix incompatible interactive benchmark runs')
    metadata_path.write_text(json.dumps(metadata, indent=2) + '\n')

    rows = []
    for state_index in range(args.state_start, args.state_start + args.states):
        part_path = parts_dir / f'state-{state_index:04d}.json'
        if part_path.exists() and manifest is not None:
            rows.append(json.loads(part_path.read_text()))
            print(f'restored state={state_index}', flush=True)
            continue
        if manifest is None:
            environment, public, memories, side, reasons, matches = collect_state(
                policy, reservoir, rng, device, args,
                gold_band=GOLD_BANDS[state_index % len(GOLD_BANDS)],
            )
            search_seed = int(rng.integers(0, 2**63 - 1))
            validation_seed = int(rng.integers(0, 2**63 - 1))
            truth_seeds = rng.integers(0, 2**63 - 1, size=args.truth_rollouts)
        else:
            record = json.loads((args.manifest_dir / f'state-{state_index:04d}.json').read_text())
            if record['checkpointSha256'] != checkpoint_hash:
                raise SystemExit('State checkpoint mismatch')
            environment, memories, side, reasons = restore_decision(record)
            public = PublicSearchState.from_core(environment.core, side)
            matches = attach_opponent_model(
                public, environment.core, side, reservoir,
                args.search_horizon_battles, args.opponent_candidates,
            )
            if not matches:
                raise RuntimeError(f'No reservoir candidates for frozen state {state_index}')
            search_seed = record['searchSeed']
            validation_seed = record['validationSeed']
            truth_seeds = record['truthSeeds'][:args.truth_rollouts]
        memory = memories[side]
        root = root_planner(
            policy,
            device,
            args.batch_size,
            args.search_horizon_battles,
            args.leaf_value_weight,
        ).search(
            public, memory, seed=search_seed, validation_seed=validation_seed
        )
        transition_budget = root.selection_simulator_transitions

        baseline = hybrid_planner(
            policy, device, args.batch_size, args, transition_budget
        ).search(
            public.clone(),
            memory,
            seed=search_seed,
            validation_seed=validation_seed,
        )
        learned_public = public.clone()
        learned_public.enable_learned_opponent_belief(OpponentBeliefMix())
        learned = hybrid_planner(
            policy, device, args.batch_size, args, transition_budget
        ).search(
            learned_public,
            memory,
            seed=search_seed,
            validation_seed=validation_seed,
        )
        raw_action = root.policy_action
        actions = tuple(
            dict.fromkeys(
                [
                    raw_action,
                    root.best_action,
                    baseline.proposed_action,
                    learned.proposed_action,
                ]
            )
        )
        started = time.perf_counter()
        truth_results = truth.evaluate(
            environment, memories, side, actions, truth_seeds
        )
        truth_wall = time.perf_counter() - started
        returns = grouped_returns(truth_results)
        raw_returns = returns[raw_action]
        complete = all(result.terminated for result in truth_results)
        scores = {}
        if complete:
            for result in truth_results:
                scores.setdefault(result.root_action, []).append(result.match_score)
            scores = {action: np.asarray(values) for action, values in scores.items()}

        def proposal(action):
            differences = returns[action] - raw_returns
            return {
                "action": list(action.as_tuple()),
                "actionDetail": action_snapshot(
                    public.core.agent, action
                ),
                "truthAdvantageVsRaw": float(differences.mean()),
                "truthDifferencesVsRaw": list(map(float, differences)),
                "matchScore": float(scores[action].mean()) if complete else None,
                "matchScoreGainVsRaw": (
                    float((scores[action] - scores[raw_action]).mean()) if complete else None
                ),
                "matchScoreDifferencesVsRaw": (
                    list(map(float, scores[action] - scores[raw_action])) if complete else None
                ),
            }

        row = {
            "state": state_index,
            "turn": int(public.core.turn),
            "side": side,
            "reasons": list(reasons),
            "samplingGoldBand": GOLD_BANDS[state_index % len(GOLD_BANDS)],
            "opponentMatches": matches,
            "publicState": view_snapshot(public.core, 0),
            "rawAction": list(raw_action.as_tuple()),
            "rawActionDetail": action_snapshot(public.core.agent, raw_action),
            "root": proposal(root.best_action),
            "baselineHybrid": proposal(baseline.proposed_action),
            "learnedBeliefHybrid": proposal(learned.proposed_action),
            "selectionTransitions": {
                "root": transition_budget,
                "baselineHybrid": (
                    baseline.simulator_transitions
                    + baseline.screening_simulator_transitions
                ),
                "learnedBeliefHybrid": (
                    learned.simulator_transitions
                    + learned.screening_simulator_transitions
                ),
            },
            "truthRollouts": args.truth_rollouts,
            "truthSeeds": list(map(int, truth_seeds)),
            "searchSeed": search_seed,
            "validationSeed": validation_seed,
            "completeMatches": complete,
            "truthWallSeconds": truth_wall,
            "truthTerminatedFraction": float(
                np.mean([result.terminated for result in truth_results])
            ),
        }
        learned_minus_baseline = (
            returns[learned.proposed_action]
            - returns[baseline.proposed_action]
        )
        row["learnedBeliefMinusBaselineHybrid"] = float(
            learned_minus_baseline.mean()
        )
        row["learnedBeliefMinusBaselineDifferences"] = list(
            map(float, learned_minus_baseline)
        )
        rows.append(row)
        temporary = part_path.with_suffix('.json.tmp')
        temporary.write_text(json.dumps(row, indent=2) + '\n')
        temporary.replace(part_path)
        print(json.dumps(row, sort_keys=True), flush=True)

    summary = {
        "states": len(rows),
        "rootVsRaw": mean_ci(
            row["root"]["truthAdvantageVsRaw"] for row in rows
        ),
        "baselineHybridVsRaw": mean_ci(
            row["baselineHybrid"]["truthAdvantageVsRaw"] for row in rows
        ),
        "learnedBeliefHybridVsRaw": mean_ci(
            row["learnedBeliefHybrid"]["truthAdvantageVsRaw"] for row in rows
        ),
        "learnedBeliefMinusBaselineHybrid": mean_ci(
            row["learnedBeliefMinusBaselineHybrid"] for row in rows
        ),
        "meanTruthTerminatedFraction": statistics.mean(
            row["truthTerminatedFraction"] for row in rows
        ),
        "completeDecisionCount": sum(row["completeMatches"] for row in rows),
        "matchScoreGains": {
            planner: mean_ci([row[planner]['matchScoreGainVsRaw'] for row in rows])
            if all(row['completeMatches'] for row in rows) else None
            for planner in ('root', 'baselineHybrid', 'learnedBeliefHybrid')
        },
    }
    payload = {
        "schemaVersion": 1,
        "miningVersion": MINING_VERSION,
        "searchVersion": SEARCH_VERSION,
        "puctVersion": PUCT_VERSION,
        "seed": args.seed,
        "simulatorRevision": SIMULATOR_REVISION,
        "checkpoint": str(args.checkpoint),
        "states": len(rows),
        "truthRollouts": args.truth_rollouts,
        "truthHorizonBattles": args.truth_horizon_battles,
        "summary": summary,
        "experiment": metadata,
        "rows": rows,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print(json.dumps(summary, sort_keys=True))


if __name__ == "__main__":
    main()
