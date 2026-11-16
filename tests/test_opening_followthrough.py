from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
from audit_opening_followthrough import summarize


def test_followthrough_separates_preservation_purchase_and_level_timing():
    rows = [
        dict(name='raw', originalPurchased=False, originalUnfrozen=False,
             targetPurchases=1, targetSales=0, firstLevelTwoTurn=3),
        dict(name='freeze', originalPurchased=True, originalUnfrozen=False,
             targetPurchases=2, targetSales=0, firstLevelTwoTurn=2),
        dict(name='freeze', originalPurchased=False, originalUnfrozen=True,
             targetPurchases=0, targetSales=1, firstLevelTwoTurn=None),
    ]
    report = summarize(rows)
    assert report['freeze']['futures'] == 2
    assert report['freeze']['originalShopCopyPurchased'] == 1
    assert report['freeze']['originalShopCopyUnfrozen'] == 1
    assert report['freeze']['levelTwoReachedByTurn'] == {'2': 1, 'not_by_prefix_end': 1}
    assert report['raw']['anyTargetCopyPurchased'] == 1
