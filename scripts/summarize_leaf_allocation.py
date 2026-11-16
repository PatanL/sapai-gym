#!/usr/bin/env python3
"""Report allocation quality separately from depth, cost and cache coverage."""
import argparse
import json
from pathlib import Path

import numpy as np

from summarize_interactive_panel import estimate, paired_future_interval


def paired_metric(rows, limit, field, baseline=None):
    vectors = []
    for row in rows:
        metric = row['variants'][limit][field]
        vector = np.asarray(metric['pairedDifferences'], dtype=float)
        if len(vector) != len(row['truthSeeds']) or not np.isclose(vector.mean(), metric['mean']):
            raise ValueError('Unaligned or inconsistent outcome vectors')
        if baseline is not None:
            other = row['variants'][baseline][field]
            reference = np.asarray(other['pairedDifferences'], dtype=float)
            if reference.shape != vector.shape or not np.isclose(reference.mean(), other['mean']):
                raise ValueError('Unaligned baseline outcome vectors')
            vector = vector - reference
        vectors.append(vector)
    metric = estimate([float(vector.mean()) for vector in vectors])
    metric['fixedRootsPairedFutureBootstrap95CI'] = paired_future_interval(vectors)
    return metric


def summarize(payload, reference_variant='180'):
    rows = payload['rows']
    if not rows or len({row['state'] for row in rows}) != len(rows):
        raise ValueError('Nonempty distinct decision states are required')
    limits = tuple(rows[0]['variants'])
    if any(set(row['variants']) != set(limits) for row in rows):
        raise ValueError('Every state must contain the same variants')
    result = {}
    for limit in limits:
        variants = [row['variants'][limit] for row in rows]
        complete = all(variant['completeMatches'] for variant in variants)
        result[limit] = {
            'completedDecisionStates': sum(variant['completeMatches'] for variant in variants),
            'rootOnlyGainVsRaw': paired_metric(rows, limit, 'rootOnlyMatchGain') if complete else None,
            'cachedGainVsRaw': paired_metric(rows, limit, 'cachedMatchGain') if complete else None,
            'cachedIncrementalGain': paired_metric(rows, limit, 'cachedIncrementalMatchGain') if complete else None,
            'meanDepth': float(np.mean([variant['meanDepth'] for variant in variants])),
            'maxDepth': max(variant['maxDepth'] for variant in variants),
            'meanTreeSeconds': float(np.mean([variant['treeSeconds'] for variant in variants])),
            'totalCachedActionChanges': sum(variant['cacheTelemetry'].get('changedActions', 0) for variant in variants),
            'statesWithChangedFollowups': sum(variant['cacheTelemetry'].get('changedActions', 0) > 0 for variant in variants),
            'visitedNodesBeyondRootTurn': sum(
                sum(count for turn, count in variant['visitedNodesByTurn'].items() if int(turn) > row['turn'])
                for row, variant in zip(rows, variants)
            ),
        }
        if reference_variant in limits and complete and all(row['variants'][reference_variant]['completeMatches'] for row in rows):
            result[limit]['rootOnlyGainVsReference'] = paired_metric(rows, limit, 'rootOnlyMatchGain', reference_variant)
            result[limit]['cachedGainVsReference'] = paired_metric(rows, limit, 'cachedMatchGain', reference_variant)
    return {
        'decisionStates': len(rows), 'variants': result,
        'referenceVariant': reference_variant if reference_variant in limits else None,
        'searchVersion': payload.get('experiment', {}).get('searchVersion', 'not recorded'),
        'interpretation': 'Developmental fixed-state pilot, not policy win rates or an independent generalization test. Timings apply to the implementation recorded in the source experiment, not other revisions.',
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('input', type=Path)
    parser.add_argument('--output', required=True, type=Path)
    parser.add_argument('--reference-variant', default='180')
    args = parser.parse_args()
    payload = json.loads(args.input.read_text())
    report = {'experiment': payload['experiment'], **summarize(payload, args.reference_variant)}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_suffix('.json.tmp')
    temporary.write_text(json.dumps(report, indent=2) + '\n')
    temporary.replace(args.output)
    print(json.dumps({key: value for key, value in report.items() if key != 'experiment'}))


if __name__ == '__main__':
    main()
