#!/usr/bin/env python3
"""Validate public root/PUCT proposals against a real historical policy."""
import argparse
import hashlib
import json
from pathlib import Path
import time

import numpy as np
import torch

from benchmark_search_interactive_truth import attach_opponent_model, root_planner, hybrid_planner
from generate_historical_search_manifest import load_policy
from summarize_interactive_panel import planner_estimate
from sapai.turtle_pack import SIMULATOR_REVISION
from sapai_gym1.rl.game_trace import action_snapshot, view_snapshot
from sapai_gym1.rl.historical_evaluation import (
    HistoricalOpponentEvaluator, POLICY_OWNERSHIP_VERSION, restore_historical_decision,
)
from sapai_gym1.rl.opponent_reservoir import OpponentTrajectoryReservoir
from sapai_gym1.rl.public_opponent_model import public_state_from_reservoir
from sapai_gym1.rl.puct_search import PUCT_VERSION
from sapai_gym1.rl.rollout_search import SEARCH_VERSION, PublicSearchState


def atomic_json(path, payload):
    temporary = path.with_suffix('.json.tmp')
    temporary.write_text(json.dumps(payload, indent=2) + '\n')
    temporary.replace(path)


def summarize(rows):
    complete = all(row['completeMatches'] for row in rows)
    return {
        'decisionStates': len(rows),
        'completeDecisionCount': sum(row['completeMatches'] for row in rows),
        'returnGainsVsRaw': {
            name: planner_estimate(rows, name, 'truthAdvantageVsRaw', 'truthDifferencesVsRaw')
            for name in ('root', 'hybrid')
        },
        'matchScoreGainsVsRaw': {
            name: planner_estimate(rows, name, 'matchScoreGainVsRaw', 'matchScoreDifferencesVsRaw')
            for name in ('root', 'hybrid')
        } if complete else None,
        'interpretation': (
            'Fixed-decision conditional action improvement against an interactive '
            'historical policy, not a full-policy win rate. Selection transition '
            'budgets are matched approximately; all extra root validation work '
            'and measured wall time are reported separately, not hidden.'
        ),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('checkpoint', type=Path)
    parser.add_argument('--opponent-checkpoint', type=Path, required=True)
    parser.add_argument('--allow-older-opponent-revision', action='store_true')
    parser.add_argument('--opponent-reservoir', type=Path, required=True)
    parser.add_argument('--manifest-dir', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--states', type=int, default=16)
    parser.add_argument('--state-start', type=int, default=0)
    parser.add_argument('--truth-rollouts', type=int, default=32)
    parser.add_argument('--search-horizon-battles', type=int, default=5)
    parser.add_argument('--truth-horizon-battles', type=int, default=30)
    parser.add_argument('--leaf-value-weight', type=float, default=0.5)
    parser.add_argument('--puct-transition-budget-scale', type=float, default=0.8)
    parser.add_argument('--opponent-candidates', type=int, default=32)
    parser.add_argument('--batch-size', type=int, default=64)
    parser.add_argument('--threads', type=int, default=1)
    args = parser.parse_args()
    if args.states < 1 or args.state_start < 0 or args.truth_rollouts < 2 or args.threads < 1:
        raise SystemExit('Invalid count')
    torch.set_num_threads(args.threads)
    device = torch.device('cpu')
    policy, main_hash, _ = load_policy(args.checkpoint, device)
    opponent, opponent_hash, opponent_revision = load_policy(
        args.opponent_checkpoint, device, allow_older_revision=args.allow_older_opponent_revision
    )
    manifest = json.loads((args.manifest_dir / 'manifest.json').read_text())
    if (manifest['checkpointSha256'] != main_hash
            or manifest['opponentCheckpointSha256'] != opponent_hash
            or manifest['policyOwnershipVersion'] != POLICY_OWNERSHIP_VERSION
            or manifest['simulatorRevision'] != SIMULATOR_REVISION):
        raise ValueError('Manifest policies or revision do not match the evaluator')
    if args.truth_rollouts > manifest['truthRollouts']:
        raise ValueError('Not enough preselected untouched futures')
    if args.state_start + args.states > manifest['states']:
        raise ValueError('Requested states exceed the frozen manifest')
    reservoir = OpponentTrajectoryReservoir.load(args.opponent_reservoir)
    if reservoir.metadata.get('simulatorRevision') != SIMULATOR_REVISION:
        raise ValueError('Reservoir revision mismatch')
    metadata = {
        'arguments': {key: str(value) if isinstance(value, Path) else value
                      for key, value in vars(args).items()},
        'checkpointSha256': main_hash, 'opponentCheckpointSha256': opponent_hash,
        'opponentTrainingSimulatorRevision': opponent_revision,
        'manifestSha256': hashlib.sha256((args.manifest_dir / 'manifest.json').read_bytes()).hexdigest(),
        'reservoirSha256': hashlib.sha256(args.opponent_reservoir.read_bytes()).hexdigest(),
        'scriptSha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        'puctVersion': PUCT_VERSION, 'searchVersion': SEARCH_VERSION,
        'policyOwnershipVersion': POLICY_OWNERSHIP_VERSION,
        'simulatorRevision': SIMULATOR_REVISION,
    }
    parts = args.output.with_suffix('.parts')
    parts.mkdir(parents=True, exist_ok=True)
    config_path = parts / 'config.json'
    if config_path.exists() and json.loads(config_path.read_text()) != metadata:
        raise ValueError('Refusing to mix incompatible historical benchmarks')
    atomic_json(config_path, metadata)
    rows = []
    for index in range(args.state_start, args.state_start + args.states):
        part = parts / f'state-{index:04d}.json'
        if part.exists():
            rows.append(json.loads(part.read_text()))
            continue
        record = json.loads((args.manifest_dir / f'state-{index:04d}.json').read_text())
        env, memories, side, reasons = restore_historical_decision(record, main_hash, opponent_hash)
        public, opponent_model = public_state_from_reservoir(
            env.core, side, reservoir, horizon=args.search_horizon_battles,
            limit=args.opponent_candidates, opening_seed=record.get('openingPriorSeed'),
        )
        matches = len(opponent_model['candidateIds'])
        started = time.perf_counter()
        root = root_planner(policy, device, args.batch_size,
                            args.search_horizon_battles, args.leaf_value_weight).search(
            public, memories[side], seed=record['searchSeed'], validation_seed=record['validationSeed']
        )
        root_wall = time.perf_counter() - started
        started = time.perf_counter()
        hybrid = hybrid_planner(policy, device, args.batch_size, args,
                                root.selection_simulator_transitions).search(
            public.clone(), memories[side], seed=record['searchSeed'], validation_seed=record['validationSeed']
        )
        hybrid_wall = time.perf_counter() - started
        raw = root.policy_action
        proposals = {'root': root.best_action, 'hybrid': hybrid.proposed_action}
        actions = tuple(dict.fromkeys([raw, *proposals.values()]))
        truth = HistoricalOpponentEvaluator(
            policy, opponent, acting_side=side, device=device, batch_size=args.batch_size,
            horizon_battles=args.truth_horizon_battles, max_steps=2500,
        )
        started = time.perf_counter()
        results = truth.evaluate(env, memories, side, actions, record['truthSeeds'][:args.truth_rollouts])
        truth_wall = time.perf_counter() - started
        complete = all(result.terminated for result in results)
        returns = {action: np.array([r.total_return for r in results if r.root_action == action])
                   for action in actions}
        scores = ({action: np.array([r.match_score for r in results if r.root_action == action])
                   for action in actions} if complete else {})
        row = {
            'state': index, 'side': side, 'turn': record['turn'], 'gold': record['gold'],
            'reasons': list(reasons), 'goldBand': record['goldBand'],
            'publicState': view_snapshot(public.core, 0), 'opponentMatches': matches,
            'opponentModel': opponent_model,
            'rawAction': action_snapshot(public.core.agent, raw),
            'completeMatches': complete, 'truthRollouts': args.truth_rollouts,
            'truthSeeds': record['truthSeeds'][:args.truth_rollouts],
            'work': {
                'rootSelectionTransitions': root.selection_simulator_transitions,
                'rootTotalTransitions': root.total_simulator_transitions,
                'hybridTotalTransitions': hybrid.simulator_transitions + hybrid.screening_simulator_transitions,
                'rootTotalSeconds': root_wall, 'rootSelectionSeconds': root.selection_wall_seconds,
                'hybridSeconds': hybrid_wall, 'outerTruthSeconds': truth_wall,
            },
        }
        for name, action in proposals.items():
            differences = returns[action] - returns[raw]
            score_differences = scores[action] - scores[raw] if complete else None
            row[name] = {
                'actionDetail': action_snapshot(public.core.agent, action),
                'truthAdvantageVsRaw': float(differences.mean()),
                'truthDifferencesVsRaw': differences.tolist(),
                'matchScore': float(scores[action].mean()) if complete else None,
                'matchScoreGainVsRaw': float(score_differences.mean()) if complete else None,
                'matchScoreDifferencesVsRaw': score_differences.tolist() if complete else None,
            }
        rows.append(row)
        atomic_json(part, row)
        print(json.dumps({
            'state': index, 'side': side, 'turn': record['turn'], 'gold': record['gold'],
            'completeMatches': complete,
            **{name: {'action': row[name]['actionDetail']['description'],
                      'matchScoreGain': row[name]['matchScoreGainVsRaw']}
               for name in proposals},
        }), flush=True)
    summary = summarize(rows)
    atomic_json(args.output, {'experiment': metadata, 'summary': summary, 'rows': rows})
    print(json.dumps(summary), flush=True)


if __name__ == '__main__':
    main()
