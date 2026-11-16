import copy
from pathlib import Path
import sys

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
from compare_search_clone_equivalence import compare_rows


def row():
    return {'state': 0, 'variants': {'0': {'completeMatches': True, 'treeSeconds': 10.,
                                           'description': 'old caption', 'paired': [-1., 1.], 'visits': [1, 2]}}}


def test_presentation_and_timing_can_change_but_outcomes_cannot():
    old, new = row(), row()
    new['variants']['0'].update(treeSeconds=2., description='new caption')
    result = compare_rows(old, new)
    assert result['equivalent'] and result['timings']['0']['speedup'] == 5.
    new['variants']['0']['paired'] = [1., -1.]
    assert not compare_rows(old, new)['equivalent']


def test_incomplete_or_changed_tree_is_not_certified():
    old, new = row(), row()
    new['variants']['0']['visits'] = [2, 1]
    assert not compare_rows(old, new)['equivalent']
    new = copy.deepcopy(old)
    new['variants']['0']['completeMatches'] = False
    with pytest.raises(ValueError, match='finish'):
        compare_rows(old, new)
