from pathlib import Path
import sys

import pytest


sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from compare_puct_calibration import compare  # noqa: E402


def payload(puct_gain, root_gain=0.0, *, action=1, state=None):
    state = {"player": {"gold": 3}} if state is None else state
    return {
        "checkpoint": "model.pt",
        "simulatorRevision": "current",
        "stateMode": "opportunity",
        "states": 1,
        "rows": [
            {
                "turn": 5,
                "side": 0,
                "stateMode": "opportunity",
                "opportunityReasons": ["matching-pet"],
                "publicState": state,
                "puctMinusRootProposedReturn": puct_gain - root_gain,
                "root": {
                    "accepted": False,
                    "proposedAction": [1, 10, 5],
                    "independentProposedReturnAdvantageVsRaw": root_gain,
                    "wallSeconds": 1.0,
                },
                "puct": {
                    "accepted": True,
                    "independentlyConfirmed": True,
                    "proposedAction": [action, 10, 5],
                    "independentProposedReturnAdvantageVsRaw": puct_gain,
                    "independentProposedRootOnlyReturnAdvantageVsRaw": (
                        puct_gain - 0.1
                    ),
                    "independentPlanningContinuationLift": 0.1,
                    "validationTreeActionsMean": 2.0,
                    "wallSeconds": 3.0,
                },
            }
        ],
    }


def test_compare_reports_paired_calibration_effects():
    result = compare(payload(0.2), payload(0.5, action=2))
    assert result["raw"]["puctProposedVsRaw"]["mean"] == pytest.approx(0.2)
    assert result["calibrated"]["puctProposedVsRaw"]["mean"] == pytest.approx(
        0.5
    )
    assert result["calibratedMinusRaw"]["puctProposedVsRaw"][
        "mean"
    ] == pytest.approx(0.3)
    assert result["proposalActionChanges"] == {"puct": 1, "root": 0}


def test_compare_refuses_unmatched_public_states():
    with pytest.raises(ValueError, match="publicState"):
        compare(payload(0.2), payload(0.2, state={"different": True}))
