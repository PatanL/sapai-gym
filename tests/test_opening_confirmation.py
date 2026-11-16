from pathlib import Path
import sys

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
from confirm_opening_proposal import exact_paired_pvalue


def test_exact_paired_gate_counts_discordances_not_total_games():
    assert exact_paired_pvalue([1.] * 9 + [-1.]) == pytest.approx(11/1024)
    assert exact_paired_pvalue([1.] * 9 + [-1.] + [0.] * 100) == pytest.approx(11/1024)
    assert exact_paired_pvalue([1.] * 10) == pytest.approx(1/1024)
    assert exact_paired_pvalue([0.] * 20) == 1.


def test_nonbinary_draw_differences_are_not_misclassified():
    assert exact_paired_pvalue([0.5, -0.5]) is None
