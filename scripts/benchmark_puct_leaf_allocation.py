#!/usr/bin/env python3
"""Compare pure-tree allocation limits with identical transition caps."""
import argparse
from collections import Counter
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import time

import numpy as np
import torch

from benchmark_search_interactive_truth import attach_opponent_model
from benchmark_cached_tree_execution import effect, grouped
from generate_historical_search_manifest import load_policy
from sapai_gym1.rl.cached_tree_evaluation import CachedTreeOpponentEvaluator
from sapai_gym1.rl.decision_manifest import restore_decision
from sapai_gym1.rl.historical_evaluation import HistoricalOpponentEvaluator
from sapai_gym1.rl.game_trace import action_snapshot
from sapai_gym1.rl.opponent_reservoir import OpponentTrajectoryReservoir
from sapai_gym1.rl.puct_search import PUCT_VERSION, PUCTConfig, StochasticPUCTSearch
from sapai_gym1.rl.rollout_search import SEARCH_VERSION, NeuralSearchEvaluator, PublicSearchState
from sapai.turtle_pack import SIMULATOR_REVISION


def atomic_json(path, data):
    temporary = path.with_suffix('.json.tmp')
    temporary.write_text(json.dumps(data, indent=2) + '\n')
    temporary.replace(path)


def make_config(limit, budget, unvisited_value_weight=None):
    return PUCTConfig(
        max_simulations=10000, max_simulator_transitions=budget,
        max_rollout_steps=180, tree_leaf_rollout_steps=limit,
        horizon_battles=5, world_pool_size=32, min_root_visits=2,
        cpuct=2., prior_floor=0.1, leaf_value_weight=0.5,
        unvisited_value_weight=unvisited_value_weight,
        selection_only=True, prune_affordable_freezes=False,
        exhaustive_root_probability=0.,
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('checkpoint', type=Path)
    parser.add_argument('--manifest-dir', type=Path, required=True)
    parser.add_argument('--opponent-reservoir', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--states', nargs='+', type=int, default=[0, 3, 7, 9])
    parser.add_argument('--leaf-limits', nargs='+', type=int, default=[0, 16, 180])
    parser.add_argument('--transition-budget', type=int, default=6000)
    parser.add_argument('--truth-rollouts', type=int, default=32)
    parser.add_argument('--truth-seed', type=int, default=4105001)
    parser.add_argument('--batch-size', type=int, default=64)
    parser.add_argument('--compare-first-play', action='store_true',
                        help='Compare legacy versus leaf-scaled unvisited-action values for each leaf limit.')
    args = parser.parse_args()
    if (args.transition_budget <= 0 or args.truth_rollouts < 2
            or len(args.states) != len(set(args.states))
            or len(args.leaf_limits) != len(set(args.leaf_limits))):
        raise ValueError('Invalid or duplicate experiment settings')
    for limit in args.leaf_limits:
        StochasticPUCTSearch(None, make_config(limit, args.transition_budget))
    designs = ([(f'{limit}-parent', limit, None) for limit in args.leaf_limits]
               + [(f'{limit}-scaled', limit, 0.5) for limit in args.leaf_limits]
               if args.compare_first_play else [(str(limit), limit, None) for limit in args.leaf_limits])
    torch.set_num_threads(1)
    device = torch.device('cpu')
    policy, policy_hash, _ = load_policy(args.checkpoint, device)
    manifest = json.loads((args.manifest_dir / 'manifest.json').read_text())
    if manifest['checkpointSha256'] != policy_hash:
        raise ValueError('Manifest recurrent states belong to another policy')
    reservoir = OpponentTrajectoryReservoir.load(args.opponent_reservoir)
    if reservoir.metadata.get('simulatorRevision') != SIMULATOR_REVISION:
        raise ValueError('Reservoir revision mismatch')
    parts = args.output.with_suffix('.parts')
    parts.mkdir(parents=True, exist_ok=True)
    metadata = {
        'arguments': {key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()},
        'puctVersion': PUCT_VERSION, 'checkpointSha256': policy_hash,
        'searchVersion': SEARCH_VERSION,
        'reservoirSha256': hashlib.sha256(args.opponent_reservoir.read_bytes()).hexdigest(),
        'scriptSha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        'purpose': 'developmental allocation pilot, not independent generalization or training labels',
        'controls': 'Equal tree transition caps, no final screening, identical public inputs; report neural and wall costs separately.',
        'numericalMemoryTolerance': 2e-6,
    }
    config_path = parts / 'config.json'
    if config_path.exists() and json.loads(config_path.read_text()) != metadata:
        raise ValueError('Refusing to mix allocation experiments')
    atomic_json(config_path, metadata)
    rows = []
    for index in args.states:
        path = parts / f'state-{index:04d}.json'
        if path.exists():
            rows.append(json.loads(path.read_text()))
            continue
        record_path = args.manifest_dir / f'state-{index:04d}.json'
        record = json.loads(record_path.read_text())
        if record['checkpointSha256'] != policy_hash:
            raise ValueError('State policy mismatch')
        env, memories, side, _ = restore_decision(record)
        public = PublicSearchState.from_core(env.core, side)
        matches = attach_opponent_model(public, env.core, side, reservoir, 5, 32)
        if not matches:
            raise ValueError('Frozen state lacks public opponent candidates')
        seeds = list(map(int, np.random.default_rng(args.truth_seed + index * 1009).integers(0, 2**63 - 1, args.truth_rollouts)))
        if set(seeds) & set(record['truthSeeds']):
            raise ValueError('Pilot futures overlap the earlier panel')
        atomic_json(parts / f'state-{index:04d}.design.json', {
            'recordSha256': hashlib.sha256(record_path.read_bytes()).hexdigest(), 'truthSeeds': seeds,
        })
        trees, results, telemetry = {}, {}, {}
        raw = None
        for variant_id, limit, unvisited_weight in designs:
            evaluator = NeuralSearchEvaluator(policy, device=device, chunk_size=args.batch_size)
            tree = StochasticPUCTSearch(evaluator, make_config(limit, args.transition_budget, unvisited_weight))
            result = tree.search(public.clone(), memories[side], seed=record['searchSeed'],
                                 validation_seed=record['validationSeed'])
            if raw is not None and raw != result.policy_action:
                raise RuntimeError('Allocation changed the unsearched baseline action')
            raw = result.policy_action
            if result.simulator_transitions > args.transition_budget:
                raise RuntimeError('Planner exceeded the shared tree budget')
            trees[variant_id], results[variant_id] = tree, result
            visits = sum(node.visits for node in tree.nodes.values())
            telemetry[variant_id] = {
                'config': asdict(tree.config), 'simulations': result.simulations,
                'rootValue': float(tree.root_node.value),
                'transitions': result.simulator_transitions, 'treeSelectedTransitions': visits,
                'policyRolloutTransitions': result.simulator_transitions - visits,
                'meanDepth': result.mean_tree_depth, 'maxDepth': result.max_tree_depth,
                'treeNodes': result.tree_nodes, 'treeSeconds': result.tree_wall_seconds,
                'visitedNodesByTurn': dict(sorted(Counter(node.turn for node in tree.nodes.values() if node.visits).items())),
                'unvisitedRootActions': sum(count == 0 for count in result.visit_counts.values()),
                'neuralWork': evaluator.metrics(),
                'action': action_snapshot(public.core.agent, result.tree_proposed_action),
            }
            print(json.dumps({'state': index, 'variant': variant_id, 'leafLimit': limit, 'meanDepth': result.mean_tree_depth,
                              'maxDepth': result.max_tree_depth, 'seconds': result.tree_wall_seconds}), flush=True)
        options = dict(acting_side=side, device=device, batch_size=args.batch_size,
                       horizon_battles=30, max_steps=2500)
        plain = HistoricalOpponentEvaluator(policy, policy, **options)
        actions = tuple(dict.fromkeys([raw, *(result.tree_proposed_action for result in results.values())]))
        started = time.perf_counter()
        plain_results = plain.evaluate(env, memories, side, actions, seeds)
        complete = all(result.terminated for result in plain_results)
        plain_returns = grouped(plain_results, 'total_return')
        plain_scores = grouped(plain_results, 'match_score') if complete else None
        for variant_id, limit, unvisited_weight in designs:
            planned = CachedTreeOpponentEvaluator(policy, policy, tree=trees[variant_id],
                                                  numerical_memory_tolerance=2e-6, **options)
            action = results[variant_id].tree_proposed_action
            planned_results = planned.evaluate(env, memories, side, [action], seeds)
            this_complete = complete and all(result.terminated for result in planned_results)
            planned_returns = grouped(planned_results, 'total_return')[action]
            planned_scores = grouped(planned_results, 'match_score')[action] if this_complete else None
            telemetry[variant_id].update({
                'completeMatches': this_complete, 'cacheTelemetry': dict(planned.counters),
                'rootOnlyReturnGain': effect(plain_returns[action] - plain_returns[raw]),
                'cachedReturnGain': effect(planned_returns - plain_returns[raw]),
                'rootOnlyMatchGain': effect(plain_scores[action] - plain_scores[raw]) if this_complete else None,
                'cachedMatchGain': effect(planned_scores - plain_scores[raw]) if this_complete else None,
                'cachedIncrementalMatchGain': effect(planned_scores - plain_scores[action]) if this_complete else None,
            })
        row = {'state': index, 'turn': record['turn'], 'gold': record['gold'], 'truthSeeds': seeds,
               'rawAction': action_snapshot(public.core.agent, raw), 'variants': telemetry,
               'outerSeconds': time.perf_counter() - started}
        rows.append(row)
        atomic_json(path, row)
        print(f'completed state={index}', flush=True)
    atomic_json(args.output, {'experiment': metadata, 'rows': rows})


if __name__ == '__main__':
    main()
