#!/usr/bin/env python3
"""Freeze versioned public one-step splits; no model inference or GPU access."""
import argparse
import hashlib
import json
from pathlib import Path

from sapai_gym1.rl.opponent_forecast_data import freeze_match_splits


def digest(path):
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('inputs', nargs='+', type=Path)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--heldout-population', action='append', default=[])
    parser.add_argument('--salt', default='forecast-quality-v1')
    parser.add_argument('--validation-fraction', type=float, default=.15)
    parser.add_argument('--test-fraction', type=float, default=.15)
    parser.add_argument('--diagnostic', action='store_true', help='Allow empty splits; never label this a quality corpus')
    args = parser.parse_args()
    if args.output.exists():
        raise ValueError('Frozen datasets are immutable; choose a new output')
    rows, inputs = [], []
    for path in args.inputs:
        shard = json.loads(path.read_text())
        # Input collector splits were only a plumbing pilot, not this frozen
        # train/validation/test experiment. Refuse to reshuffle a frozen corpus.
        if 'assignments' in shard.get('metadata', {}):
            raise ValueError('Do not re-split a previously frozen quality dataset')
        rows.extend(shard['train'])
        rows.extend(shard['heldout'])
        inputs.append({'path': str(path), 'sha256': digest(path), 'metadata': shard['metadata']})
    result = freeze_match_splits(rows, heldout_populations=args.heldout_population,
                                salt=args.salt, validation_fraction=args.validation_fraction,
                                test_fraction=args.test_fraction, require_nonempty=not args.diagnostic)
    result['metadata'].update(inputs=inputs, builderSHA256=digest(Path(__file__)),
                              diagnosticOnly=args.diagnostic, policyTrainingActivated=False)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open('x') as output:
        json.dump(result, output, indent=2)
        output.write('\n')
    print(json.dumps(result['metadata'], sort_keys=True))


if __name__ == '__main__':
    main()
