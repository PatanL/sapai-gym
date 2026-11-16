#!/usr/bin/env python3
"""Summarize paired planner gains across distinct frozen decision states."""
import argparse
import json
from pathlib import Path

import numpy as np


PLANNERS = ('root', 'baselineHybrid', 'learnedBeliefHybrid')
MIN_STATES_FOR_BOOTSTRAP = 8


def compatible_config(config):
    result = dict(config)
    arguments = dict(result['arguments'])
    for key in ('states', 'state_start', 'output'):
        arguments.pop(key, None)
    result['arguments'] = arguments
    return result


def load_reports(paths):
    rows = []
    configs = []
    for path in paths:
        path = Path(path)
        if path.is_dir():
            configs.append(json.loads((path / 'config.json').read_text()))
            rows.extend(json.loads(part.read_text()) for part in sorted(path.glob('state-*.json')))
        else:
            payload = json.loads(path.read_text())
            configs.append(payload['experiment'])
            rows.extend(payload['rows'])
    canonical = [compatible_config(config) for config in configs]
    if any(config != canonical[0] for config in canonical):
        raise ValueError('Panel inputs have incompatible experiment configurations')
    indices = [row['state'] for row in rows]
    if len(indices) != len(set(indices)):
        raise ValueError('Duplicate decision states would overcount the evidence')
    for row in rows:
        for planner in PLANNERS:
            if len(row[planner]['truthDifferencesVsRaw']) != row['truthRollouts']:
                raise ValueError('Return vector does not match the declared rollout count')
    return sorted(rows, key=lambda row: row['state']), canonical[0]


def estimate(values):
    values = np.asarray(values, dtype=np.float64)
    if not np.isfinite(values).all():
        raise ValueError('Nonfinite planner measurement')
    if not len(values):
        return {'decisionStates': 0, 'mean': None, 'stateBootstrap95CI': None}
    interval = None
    if len(values) >= MIN_STATES_FOR_BOOTSTRAP:
        # Resample independent decisions, not all rollouts as independent
        # observations of strategy. Within-state paired seeds remain paired.
        rng = np.random.default_rng(4005901)
        means = values[rng.integers(len(values), size=(10000, len(values)))].mean(axis=1)
        interval = list(map(float, np.quantile(means, [0.025, 0.975])))
    return {
        'decisionStates': len(values), 'mean': float(values.mean()),
        'stateBootstrap95CI': interval,
        'smallStateSample': len(values) < MIN_STATES_FOR_BOOTSTRAP,
    }


def paired_future_interval(vectors):
    """Monte Carlo uncertainty for these fixed roots, not population strength.

    Each vector is already planner-minus-raw on the same future seeds. Resample
    those paired differences within roots; never separate the two actions or
    pretend that 32 futures are 32 independent strategic decision states.
    """
    if not vectors:
        return None
    rng = np.random.default_rng(4005902)
    means = np.zeros(10000, dtype=np.float64)
    for vector in vectors:
        values = np.asarray(vector, dtype=np.float64)
        if values.ndim != 1 or len(values) < 2 or not np.isfinite(values).all():
            raise ValueError('Each root needs at least two finite paired futures')
        means += values[rng.integers(len(values), size=(10000, len(values)))].mean(axis=1)
    return list(map(float, np.quantile(means / len(vectors), [0.025, 0.975])))


def planner_estimate(rows, planner, mean_key, vector_key):
    vectors = [row[planner][vector_key] for row in rows]
    for row, vector in zip(rows, vectors):
        if len(vector) != row['truthRollouts']:
            raise ValueError('Paired future count does not match the declared count')
        if not np.isclose(np.mean(vector), row[planner][mean_key], rtol=1e-6, atol=1e-8):
            raise ValueError('Stored mean does not match paired future measurements')
    result = estimate([row[planner][mean_key] for row in rows])
    result['fixedRootsPairedFutureBootstrap95CI'] = paired_future_interval(vectors)
    return result


def summarize(rows):
    complete = [row for row in rows if row['completeMatches']]
    return {
        'decisionStates': len(rows),
        'completedMatchDecisionStates': len(complete),
        'interpretation': (
            'Conditional action gains on the stratified fixed manifest, not a '
            'full-policy win rate. Future intervals measure simulation noise '
            'for these roots; state intervals measure between-root variability '
            'and are suppressed below eight roots. Neither is a promotion gate.'
        ),
        'returnGainsVsRaw': {
            planner: planner_estimate(rows, planner, 'truthAdvantageVsRaw', 'truthDifferencesVsRaw')
            for planner in PLANNERS
        },
        # Do not silently select easier-to-finish states for match scoring.
        'matchScoreGainsVsRaw': {
            planner: planner_estimate(rows, planner, 'matchScoreGainVsRaw', 'matchScoreDifferencesVsRaw')
            for planner in PLANNERS
        } if len(complete) == len(rows) else None,
        'hybridMinusRootReturn': estimate([
            row['baselineHybrid']['truthAdvantageVsRaw'] - row['root']['truthAdvantageVsRaw']
            for row in rows
        ]),
        'learnedBeliefMinusBaselineReturn': estimate([
            row['learnedBeliefMinusBaselineHybrid'] for row in rows
        ]),
        'decisions': [{
            'state': row['state'], 'turn': row['turn'],
            'gold': row['publicState']['player']['gold'],
            'goldBand': row['samplingGoldBand'], 'reasons': row['reasons'],
            'rawAction': row['rawActionDetail']['description'],
            'proposals': {planner: {
                'action': row[planner]['actionDetail']['description'],
                'returnGain': row[planner]['truthAdvantageVsRaw'],
                'matchScoreGain': row[planner]['matchScoreGainVsRaw'],
            } for planner in PLANNERS},
        } for row in rows],
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('inputs', nargs='+', type=Path)
    parser.add_argument('--output', required=True, type=Path)
    args = parser.parse_args()
    rows, config = load_reports(args.inputs)
    report = {'experiment': config, **summarize(rows)}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_suffix('.json.tmp')
    temporary.write_text(json.dumps(report, indent=2) + '\n')
    temporary.replace(args.output)
    print(json.dumps({key: value for key, value in report.items() if key not in ('decisions', 'experiment')}))


if __name__ == '__main__':
    main()
