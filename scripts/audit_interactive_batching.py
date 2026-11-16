#!/usr/bin/env python3
"""Replay original futures to separate sampling variance from batching drift."""
import argparse
import hashlib
import json
from pathlib import Path
import time

import numpy as np
import torch

from generate_historical_search_manifest import load_policy
from benchmark_cached_tree_execution import grouped
from sapai_gym1.rl.action_schema import AutoregressiveAction
from sapai_gym1.rl.decision_manifest import restore_decision
from sapai_gym1.rl.historical_evaluation import HistoricalOpponentEvaluator
from sapai_gym1.rl.interactive_evaluation import InteractiveSelfPlayEvaluator


def paired_agreement(left, right):
    left, right = np.asarray(left), np.asarray(right)
    if left.shape != right.shape or left.ndim != 1 or not len(left):
        raise ValueError('Aligned nonempty future vectors are required')
    return {'count': len(left), 'differentCount': int(np.count_nonzero(left != right)),
            'maxAbsoluteDifference': float(np.max(np.abs(left - right))),
            'meanDifference': float(np.mean(left - right))}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('checkpoint', type=Path)
    parser.add_argument('--record', type=Path, required=True)
    parser.add_argument('--comparison-row', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--batch-size', type=int, default=64)
    args = parser.parse_args()
    torch.set_num_threads(1)
    device = torch.device('cpu')
    policy, checkpoint_hash, _ = load_policy(args.checkpoint, device)
    record = json.loads(args.record.read_text())
    row = json.loads(args.comparison_row.read_text())
    if record['checkpointSha256'] != checkpoint_hash or record['state'] != row['state']:
        raise ValueError('Policy or frozen root mismatch')
    if row['truthSeeds'] != record['truthSeeds'][:row['truthRollouts']]:
        raise ValueError('Original future seeds do not match the frozen manifest')
    env, memories, side, _ = restore_decision(record)
    raw = AutoregressiveAction(*row['rawAction'])
    proposals = {name: AutoregressiveAction(*row[name]['action'])
                 for name in ('root', 'baselineHybrid', 'learnedBeliefHybrid')}
    actions = tuple(dict.fromkeys([raw, *proposals.values()]))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    metadata = {
        'checkpointSha256': checkpoint_hash, 'state': record['state'],
        'recordSha256': hashlib.sha256(args.record.read_bytes()).hexdigest(),
        'comparisonRowSha256': hashlib.sha256(args.comparison_row.read_bytes()).hexdigest(),
        'truthSeeds': row['truthSeeds'], 'batchSize': args.batch_size,
        'purpose': 'reproduction and batching audit on already-used futures, not fresh strength evidence',
    }
    args.output.with_suffix('.design.json').write_text(json.dumps(metadata, indent=2) + '\n')
    options = dict(device=device, batch_size=args.batch_size, horizon_battles=30, max_steps=2500)
    evaluators = {
        'mixedSeats': InteractiveSelfPlayEvaluator(policy, **options),
        'separateSeats': HistoricalOpponentEvaluator(policy, policy, acting_side=side, **options),
    }
    outputs, timings = {}, {}
    for name, evaluator in evaluators.items():
        started = time.perf_counter()
        results = evaluator.evaluate(env, memories, side, actions, row['truthSeeds'])
        if not all(result.terminated for result in results):
            raise RuntimeError(f'{name}: unfinished matches cannot establish outcome reproduction')
        outputs[name] = {'returns': grouped(results, 'total_return'), 'scores': grouped(results, 'match_score')}
        timings[name] = time.perf_counter() - started
        print(f'completed mode={name} seconds={timings[name]:.2f}', flush=True)
    replay = outputs['mixedSeats']
    comparisons = {}
    for name, action in proposals.items():
        comparisons[name] = {
            'oldScoreDifferencesReproduced': paired_agreement(
                replay['scores'][action] - replay['scores'][raw], row[name]['matchScoreDifferencesVsRaw']),
            'oldReturnDifferencesReproduced': paired_agreement(
                replay['returns'][action] - replay['returns'][raw], row[name]['truthDifferencesVsRaw']),
            'oldMeanScore': row[name]['matchScore'], 'replayedMeanScore': float(replay['scores'][action].mean()),
        }
    layouts = {}
    for label, action in {'raw': raw, **proposals}.items():
        layouts[label] = {
            'matchOutcomes': paired_agreement(outputs['separateSeats']['scores'][action], replay['scores'][action]),
            'returns': paired_agreement(outputs['separateSeats']['returns'][action], replay['returns'][action]),
            'mixedMeanScore': float(replay['scores'][action].mean()),
            'separateMeanScore': float(outputs['separateSeats']['scores'][action].mean()),
        }
    report = {'design': metadata, 'originalReproduction': comparisons,
              'batchingComparison': layouts, 'wallSeconds': timings}
    temporary = args.output.with_suffix('.json.tmp')
    temporary.write_text(json.dumps(report, indent=2) + '\n')
    temporary.replace(args.output)
    print(json.dumps(report), flush=True)


if __name__ == '__main__':
    main()
