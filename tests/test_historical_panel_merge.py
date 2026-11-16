import json
from pathlib import Path
import sys

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
from summarize_historical_panel import merge


def payload(index, opponent='v31'):
    metric = {'truthAdvantageVsRaw': 0., 'truthDifferencesVsRaw': [0., 0.],
              'matchScoreGainVsRaw': 0., 'matchScoreDifferencesVsRaw': [0., 0.]}
    return {'experiment': {'arguments': {'states': 1, 'state_start': index}, 'opponent': opponent},
            'rows': [{'state': index, 'completeMatches': True, 'truthRollouts': 2,
                      'root': metric, 'hybrid': metric}]}


def test_merge_keeps_opponents_separate_and_rejects_duplicates(tmp_path):
    first, second = tmp_path / 'first.json', tmp_path / 'second.json'
    first.write_text(json.dumps(payload(0)))
    second.write_text(json.dumps(payload(1)))
    assert merge([first, second])['summary']['decisionStates'] == 2
    with pytest.raises(ValueError, match='Duplicate'):
        merge([first, first])
    second.write_text(json.dumps(payload(1, 'different-opponent')))
    with pytest.raises(ValueError, match='do not match'):
        merge([first, second])
