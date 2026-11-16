#!/usr/bin/env python3
"""Bounded CPU profile of tree construction, excluding checkpoint/data loading."""
import argparse
import cProfile
import hashlib
import json
from pathlib import Path
import pstats
import time

import torch

from benchmark_puct_leaf_allocation import atomic_json, make_config
from benchmark_search_interactive_truth import attach_opponent_model
from generate_historical_search_manifest import load_policy
from sapai_gym1.rl.decision_manifest import restore_decision
from sapai_gym1.rl.opponent_reservoir import OpponentTrajectoryReservoir
from sapai_gym1.rl.puct_search import PUCT_VERSION, StochasticPUCTSearch
from sapai_gym1.rl.rollout_search import SEARCH_VERSION, NeuralSearchEvaluator, PublicSearchState


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('checkpoint', type=Path)
    parser.add_argument('--record', type=Path, required=True)
    parser.add_argument('--opponent-reservoir', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--transition-budget', type=int, default=300)
    parser.add_argument('--leaf-limit', type=int, default=0)
    args = parser.parse_args()
    torch.set_num_threads(1)
    policy, policy_hash, _ = load_policy(args.checkpoint, torch.device('cpu'))
    record = json.loads(args.record.read_text())
    if record['checkpointSha256'] != policy_hash:
        raise ValueError('Profile policy differs from the frozen root')
    env, memories, side, _ = restore_decision(record)
    public = PublicSearchState.from_core(env.core, side)
    reservoir = OpponentTrajectoryReservoir.load(args.opponent_reservoir)
    attach_opponent_model(public, env.core, side, reservoir, 5, 32)
    evaluator = NeuralSearchEvaluator(policy, device='cpu', chunk_size=64)
    tree = StochasticPUCTSearch(evaluator, make_config(args.leaf_limit, args.transition_budget))
    profiler = cProfile.Profile()
    started = time.perf_counter()
    profiler.enable()
    result = tree.search(public, memories[side], seed=record['searchSeed'],
                         validation_seed=record['validationSeed'])
    profiler.disable()
    wall = time.perf_counter() - started
    stats = pstats.Stats(profiler)
    entries = []
    for (filename, line, name), (primitive, calls, exclusive, cumulative, _) in stats.stats.items():
        entries.append({'file': filename, 'line': line, 'function': name,
                        'calls': calls, 'primitiveCalls': primitive,
                        'exclusiveSeconds': exclusive, 'cumulativeSeconds': cumulative})
    hot_names = {'_expand', '_policy_rollout', 'clone', 'evaluate', 'evaluate_all_actions',
                 'encode_entities', '_state_key', '_avail_actions', 'from_state', '_copy_team',
                 '_connect_objects', 'observation', 'step', '_simulate'}
    report = {
        'puctVersion': PUCT_VERSION, 'checkpointSha256': policy_hash,
        'searchVersion': SEARCH_VERSION,
        'treeFingerprint': hashlib.sha256(json.dumps([
            [list(action.as_tuple()), result.visit_counts[action], result.q_values[action], result.priors[action]]
            for action in sorted(result.visit_counts, key=lambda action: action.as_tuple())
        ], separators=(',', ':')).encode()).hexdigest(),
        'proposedAction': list(result.tree_proposed_action.as_tuple()),
        'state': record['state'], 'leafLimit': args.leaf_limit,
        'transitionBudget': args.transition_budget, 'transitions': result.simulator_transitions,
        'simulations': result.simulations, 'meanDepth': result.mean_tree_depth,
        'wallSeconds': wall, 'profilerSeconds': stats.total_tt,
        'neuralWork': evaluator.metrics(),
        'exclusiveTop': sorted(entries, key=lambda row: row['exclusiveSeconds'], reverse=True)[:25],
        'cumulativeHotspots': sorted((row for row in entries if row['function'] in hot_names),
                                    key=lambda row: row['cumulativeSeconds'], reverse=True),
        'note': 'Small-root CPU profile with cProfile overhead; cumulative rows overlap and must not be summed.',
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    profiler.dump_stats(str(args.output.with_suffix('.pstats')))
    atomic_json(args.output, report)
    print(json.dumps({'wallSeconds': wall, 'transitions': result.simulator_transitions,
                      'exclusiveTop': report['exclusiveTop'][:8]}), flush=True)


if __name__ == '__main__':
    main()
