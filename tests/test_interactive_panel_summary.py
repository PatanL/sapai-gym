import copy
import json
from pathlib import Path
import sys

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
from summarize_interactive_panel import PLANNERS, load_reports, summarize


def row(index, gain):
    return {
        'state': index, 'turn': 4, 'samplingGoldBand': 'zero',
        'publicState': {'player': {'gold': 0}}, 'reasons': ['freeze'],
        'rawActionDetail': {'description': 'End'}, 'completeMatches': True,
        'truthRollouts': 32, 'learnedBeliefMinusBaselineHybrid': 0.,
        **{planner: {
            'actionDetail': {'description': 'Freeze'},
            'truthAdvantageVsRaw': gain, 'truthDifferencesVsRaw': [gain] * 32,
            'matchScoreGainVsRaw': gain,
            'matchScoreDifferencesVsRaw': [gain] * 32,
        } for planner in PLANNERS},
    }


def test_panel_uncertainty_counts_states_not_rollouts():
    result = summarize([row(0, 1.), row(1, -1.)])
    metric = result['matchScoreGainsVsRaw']['root']
    assert metric['decisionStates'] == 2
    assert metric['mean'] == 0.
    assert metric['stateBootstrap95CI'] is None
    assert metric['smallStateSample']
    assert metric['fixedRootsPairedFutureBootstrap95CI'] == [0., 0.]


def test_future_uncertainty_does_not_disappear_with_identical_state_means():
    rows = [row(index, 0.) for index in range(2)]
    for entry in rows:
        for planner in PLANNERS:
            entry[planner]['matchScoreDifferencesVsRaw'] = [-1., 1.] * 16
    interval = summarize(rows)['matchScoreGainsVsRaw']['root']['fixedRootsPairedFutureBootstrap95CI']
    assert interval[0] < 0 < interval[1]


def test_stored_means_are_verified_against_paired_outcomes():
    entry = row(0, 0.)
    entry['root']['matchScoreGainVsRaw'] = 1.
    with pytest.raises(ValueError, match='Stored mean'):
        summarize([entry])


def test_truncation_does_not_silently_filter_the_match_score_panel():
    incomplete = row(1, 1.)
    incomplete['completeMatches'] = False
    result = summarize([row(0, 1.), incomplete])
    assert result['completedMatchDecisionStates'] == 1
    assert result['matchScoreGainsVsRaw'] is None


def test_duplicate_and_mixed_planner_inputs_are_rejected(tmp_path):
    payload = {'experiment': {'arguments': {'states': 1}, 'puctVersion': 'v11'}, 'rows': [row(0, 1.)]}
    first = tmp_path / 'first.json'
    second = tmp_path / 'second.json'
    first.write_text(json.dumps(payload))
    second.write_text(json.dumps(payload))
    with pytest.raises(ValueError, match='Duplicate'):
        load_reports([first, second])
    changed = copy.deepcopy(payload)
    changed['experiment']['puctVersion'] = 'v10'
    changed['rows'][0]['state'] = 1
    second.write_text(json.dumps(changed))
    with pytest.raises(ValueError, match='incompatible'):
        load_reports([first, second])
