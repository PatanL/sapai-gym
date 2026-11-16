#!/usr/bin/env python3
"""Merge compatible historical-policy diagnostic shards without double counting."""
import argparse
import json
from pathlib import Path
import re

from benchmark_historical_search import summarize, atomic_json
from summarize_interactive_panel import compatible_config


def merge(paths):
    rows, configs = [], []
    for path in map(Path, paths):
        if path.is_dir():
            configs.append(json.loads((path / 'config.json').read_text()))
            rows.extend(json.loads(part.read_text()) for part in sorted(path.glob('state-*.json'))
                        if re.fullmatch(r'state-\d+\.json', part.name))
        else:
            payload = json.loads(path.read_text())
            configs.append(payload['experiment'])
            rows.extend(payload['rows'])
    canonical = [compatible_config(config) for config in configs]
    if not canonical or any(config != canonical[0] for config in canonical):
        raise ValueError('Historical experiments do not match')
    ids = [row['state'] for row in rows]
    if len(ids) != len(set(ids)):
        raise ValueError('Duplicate historical decision state')
    rows.sort(key=lambda row: row['state'])
    return {'experiment': canonical[0], 'summary': summarize(rows), 'rows': rows}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('inputs', nargs='+', type=Path)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    report = merge(args.inputs)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    atomic_json(args.output, report)
    print(json.dumps(report['summary']))


if __name__ == '__main__':
    main()
