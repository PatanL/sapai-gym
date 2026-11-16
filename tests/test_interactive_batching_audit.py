from pathlib import Path
import sys

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
from audit_interactive_batching import paired_agreement


def test_equal_means_do_not_hide_changed_seed_outcomes():
    report = paired_agreement([1., 0., 0., 1.], [0., 1., 1., 0.])
    assert report['meanDifference'] == 0.
    assert report['differentCount'] == 4
    assert report['maxAbsoluteDifference'] == 1.


def test_agreement_and_unaligned_input():
    assert paired_agreement([1., 0.], [1., 0.])['differentCount'] == 0
    with pytest.raises(ValueError, match='Aligned'):
        paired_agreement([1.], [1., 0.])
