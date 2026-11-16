#!/usr/bin/env python3
"""Distinguish first-action label quality from executing a cached PUCT plan."""
import argparse
import hashlib
import json
from pathlib import Path
import time

import numpy as np
import torch

from benchmark_search_interactive_truth import attach_opponent_model, hybrid_planner
from generate_historical_search_manifest import load_policy
from summarize_interactive_panel import paired_future_interval
from sapai_gym1.rl.cached_tree_evaluation import CachedTreeOpponentEvaluator
from sapai_gym1.rl.historical_evaluation import HistoricalOpponentEvaluator
from sapai_gym1.rl.decision_manifest import restore_decision
from sapai_gym1.rl.action_schema import AutoregressiveAction
from sapai_gym1.rl.game_trace import action_snapshot
from sapai_gym1.rl.opponent_reservoir import OpponentTrajectoryReservoir
from sapai_gym1.rl.puct_search import PUCT_VERSION
from sapai_gym1.rl.rollout_search import PublicSearchState


def grouped(results, attribute):
    actions = tuple(dict.fromkeys(result.root_action for result in results))
    return {action: np.array([getattr(result, attribute) for result in results if result.root_action == action])
            for action in actions}


def effect(differences):
    return {'mean': float(differences.mean()), 'pairedDifferences': differences.tolist(),
            'fixedRootPairedFutureBootstrap95CI': paired_future_interval([differences])}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('checkpoint', type=Path)
    parser.add_argument('--record', type=Path, required=True)
    parser.add_argument('--comparison-row', type=Path, required=True)
    parser.add_argument('--opponent-reservoir', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--search-horizon-battles', type=int, default=5)
    parser.add_argument('--opponent-candidates', type=int, default=32)
    parser.add_argument('--leaf-value-weight', type=float, default=0.5)
    parser.add_argument('--puct-transition-budget-scale', type=float, default=0.8)
    parser.add_argument('--batch-size', type=int, default=64)
    parser.add_argument('--truth-rollouts', type=int, default=32)
    parser.add_argument('--truth-seed', type=int, default=4005801)
    parser.add_argument('--numerical-memory-tolerance', type=float, default=0.)
    args = parser.parse_args()
    if args.truth_rollouts < 2:
        raise ValueError('At least two independent paired futures are required')
    torch.set_num_threads(1)
    device = torch.device('cpu')
    policy, policy_hash, _ = load_policy(args.checkpoint, device)
    record = json.loads(args.record.read_text())
    comparison = json.loads(args.comparison_row.read_text())
    if record['checkpointSha256'] != policy_hash or record['state'] != comparison['state']:
        raise ValueError('Checkpoint or root mismatch')
    if record['searchSeed'] != comparison['searchSeed']:
        raise ValueError('Search seeds differ')
    truth_seeds = list(map(int, np.random.default_rng(args.truth_seed).integers(0, 2**63 - 1, args.truth_rollouts)))
    if set(truth_seeds) & set(record['truthSeeds']):
        raise ValueError('New diagnostic futures overlap the earlier outer panel')
    args.output.parent.mkdir(parents=True, exist_ok=True)
    design = {
        'state': record['state'], 'checkpointSha256': policy_hash, 'puctVersion': PUCT_VERSION,
        'recordSha256': hashlib.sha256(args.record.read_bytes()).hexdigest(),
        'truthSeeds': truth_seeds, 'scriptSha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        'purpose': 'selected-failure diagnosis; not a generalization test or training labels',
        'numericalMemoryTolerance': args.numerical_memory_tolerance,
    }
    design_path = args.output.with_suffix('.design.json')
    if design_path.exists() and json.loads(design_path.read_text()) != design:
        raise ValueError('Refusing to overwrite a different experiment design')
    design_path.write_text(json.dumps(design, indent=2) + '\n')
    env, memories, side, _ = restore_decision(record)
    public = PublicSearchState.from_core(env.core, side)
    reservoir = OpponentTrajectoryReservoir.load(args.opponent_reservoir)
    attach_opponent_model(public, env.core, side, reservoir,
                          args.search_horizon_battles, args.opponent_candidates)
    tree = hybrid_planner(policy, device, args.batch_size, args,
                          comparison['selectionTransitions']['root'])
    started = time.perf_counter()
    result = tree.search(public, memories[side], seed=record['searchSeed'],
                         validation_seed=record['validationSeed'])
    search_seconds = time.perf_counter() - started
    raw = AutoregressiveAction(*comparison['rawAction'])
    proposals = {'raw': raw, 'tree': result.tree_proposed_action, 'screened': result.proposed_action}
    actions = tuple(dict.fromkeys(proposals.values()))
    options = dict(acting_side=side, device=device, batch_size=args.batch_size,
                   horizon_battles=30, max_steps=2500)
    plain = HistoricalOpponentEvaluator(policy, policy, **options)
    planned = CachedTreeOpponentEvaluator(policy, policy, tree=tree,
                                          numerical_memory_tolerance=args.numerical_memory_tolerance,
                                          **options)
    started = time.perf_counter()
    plain_results = plain.evaluate(env, memories, side, actions, truth_seeds)
    planned_results = planned.evaluate(env, memories, side, actions, truth_seeds)
    complete = all(result.terminated for result in (*plain_results, *planned_results))
    plain_returns, planned_returns = grouped(plain_results, 'total_return'), grouped(planned_results, 'total_return')
    plain_scores = grouped(plain_results, 'match_score') if complete else None
    planned_scores = grouped(planned_results, 'match_score') if complete else None
    rows = {}
    for name, action in proposals.items():
        rows[name] = {
            'action': action_snapshot(public.core.agent, action),
            'rootOnlyReturnGainVsRaw': effect(plain_returns[action] - plain_returns[raw]),
            'cachedReturnGainVsRaw': effect(planned_returns[action] - plain_returns[raw]),
            'cachedIncrementalReturn': effect(planned_returns[action] - plain_returns[action]),
            'rootOnlyMatchGainVsRaw': effect(plain_scores[action] - plain_scores[raw]) if complete else None,
            'cachedMatchGainVsRaw': effect(planned_scores[action] - plain_scores[raw]) if complete else None,
            'cachedIncrementalMatchGain': effect(planned_scores[action] - plain_scores[action]) if complete else None,
        }
    payload = {
        'design': design, 'rows': rows, 'completeMatches': complete,
        'cacheTelemetry': dict(planned.counters), 'uniqueRootActions': len(actions),
        'searchSeconds': search_seconds, 'outerSeconds': time.perf_counter() - started,
        'meanTreeDepth': result.mean_tree_depth, 'maxTreeDepth': result.max_tree_depth,
        'rootTurn': record['turn'], 'gold': record['gold'],
    }
    temporary = args.output.with_suffix('.json.tmp')
    temporary.write_text(json.dumps(payload, indent=2) + '\n')
    temporary.replace(args.output)
    print(json.dumps({key: value for key, value in payload.items() if key not in ('rows', 'design')}), flush=True)


if __name__ == '__main__':
    main()
