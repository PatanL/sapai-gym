from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
from benchmark_historical_search import summarize


def test_historical_summary_preserves_paired_outcome_uncertainty():
    values = [-1., 1.] * 16
    metrics = {
        'truthAdvantageVsRaw': 0., 'truthDifferencesVsRaw': values,
        'matchScoreGainVsRaw': 0., 'matchScoreDifferencesVsRaw': values,
    }
    row = {'completeMatches': True, 'truthRollouts': 32, 'root': metrics, 'hybrid': metrics}
    report = summarize([row])
    assert report['decisionStates'] == 1
    assert report['completeDecisionCount'] == 1
    score = report['matchScoreGainsVsRaw']['root']
    assert score['mean'] == 0.
    assert score['stateBootstrap95CI'] is None
    assert score['fixedRootsPairedFutureBootstrap95CI'][0] < 0
    assert score['fixedRootsPairedFutureBootstrap95CI'][1] > 0
    row['completeMatches'] = False
    assert summarize([row])['matchScoreGainsVsRaw'] is None
