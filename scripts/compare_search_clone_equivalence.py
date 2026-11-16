#!/usr/bin/env python3
"""Check semantic equality of allocation runs, not merely equal mean win rates."""
import argparse
import json
from pathlib import Path


IGNORED_PRESENTATION = {'description', 'treeSeconds', 'outerSeconds'}


def differences(left, right, path='root'):
    result = []
    if isinstance(left, dict) and isinstance(right, dict):
        keys = (set(left) | set(right)) - IGNORED_PRESENTATION
        for key in sorted(keys):
            child = f'{path}.{key}'
            if key not in left or key not in right:
                result.append(child)
            else:
                result.extend(differences(left[key], right[key], child))
    elif isinstance(left, list) and isinstance(right, list):
        if len(left) != len(right):
            result.append(f'{path}.length')
        else:
            for index, (a, b) in enumerate(zip(left, right)):
                result.extend(differences(a, b, f'{path}[{index}]'))
    elif type(left) is not type(right) or left != right:
        result.append(path)
    return result


def compare_rows(old, new):
    if not all(variant['completeMatches'] for row in (old, new) for variant in row['variants'].values()):
        raise ValueError('Both runs must finish all matched continuations')
    mismatch = differences(old, new)
    timings = {limit: {
        'oldTreeSeconds': old['variants'][limit]['treeSeconds'],
        'newTreeSeconds': new['variants'][limit]['treeSeconds'],
        'speedup': old['variants'][limit]['treeSeconds'] / new['variants'][limit]['treeSeconds'],
    } for limit in old['variants'] if limit in new['variants']}
    return {'equivalent': not mismatch, 'mismatches': mismatch, 'timings': timings,
            'scope': 'All recorded action fields, tree/NN metrics, cache telemetry, seeds, and paired outcomes; only descriptions and elapsed times ignored.'}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('original_state', type=Path)
    parser.add_argument('optimized_run', type=Path)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    old = json.loads(args.original_state.read_text())
    optimized = json.loads(args.optimized_run.read_text())
    new_rows = [row for row in optimized['rows'] if row['state'] == old['state']]
    if len(new_rows) != 1:
        raise ValueError('Optimized run must contain exactly one matching root')
    old_config = json.loads((args.original_state.parent / 'config.json').read_text())
    for key in ('checkpointSha256', 'reservoirSha256', 'puctVersion'):
        if old_config[key] != optimized['experiment'][key]:
            raise ValueError(f'Incompatible comparison: {key}')
    report = {'state': old['state'], **compare_rows(old, new_rows[0])}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_suffix('.json.tmp')
    temporary.write_text(json.dumps(report, indent=2) + '\n')
    temporary.replace(args.output)
    print(json.dumps(report))
    if not report['equivalent']:
        raise SystemExit(1)


if __name__ == '__main__':
    main()
