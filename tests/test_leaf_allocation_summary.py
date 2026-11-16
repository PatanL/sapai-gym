from pathlib import Path
import copy
import sys

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
from summarize_leaf_allocation import summarize


def payload():
    metric = {'mean': 0., 'pairedDifferences': [-1., 1.]}
    variant = {'completeMatches': True, 'rootOnlyMatchGain': metric,
               'cachedMatchGain': metric, 'cachedIncrementalMatchGain': metric,
               'meanDepth': 4., 'maxDepth': 12, 'treeSeconds': 20.,
               'cacheTelemetry': {}, 'visitedNodesByTurn': {'2': 10, '3': 5}}
    return {'rows': [{'state': 0, 'turn': 2, 'truthSeeds': [1,2], 'variants': {'0': variant}}]}


def test_summary_preserves_small_sample_uncertainty_and_depth_scope():
    report = summarize(payload())
    metric = report['variants']['0']['rootOnlyGainVsRaw']
    assert metric['decisionStates'] == 1 and metric['stateBootstrap95CI'] is None
    assert metric['fixedRootsPairedFutureBootstrap95CI'] == [-1., 1.]
    assert report['variants']['0']['visitedNodesBeyondRootTurn'] == 5


def test_unfinished_and_unaligned_outcomes_are_not_certified():
    value = payload()
    value['rows'][0]['variants']['0']['completeMatches'] = False
    assert summarize(value)['variants']['0']['rootOnlyGainVsRaw'] is None
    value = payload()
    value['rows'][0]['truthSeeds'].append(3)
    with pytest.raises(ValueError, match='Unaligned'):
        summarize(value)


def test_reference_variant_uses_matched_futures():
    value = payload()
    baseline = value['rows'][0]['variants'].pop('0')
    changed = copy.deepcopy(baseline)
    for field in ('rootOnlyMatchGain', 'cachedMatchGain'):
        changed[field] = {'mean': 1., 'pairedDifferences': [1., 1.]}
    value['rows'][0]['variants'] = {'parent': baseline, 'scaled': changed}
    report = summarize(value, 'parent')
    assert report['referenceVariant'] == 'parent'
    assert report['variants']['scaled']['rootOnlyGainVsReference']['mean'] == 1.
