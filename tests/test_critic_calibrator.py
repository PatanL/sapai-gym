from types import SimpleNamespace

import pytest

from sapai_gym1.rl import AdditiveCriticCalibrator


def state(turn, own=(6, 2), opponent=(4, 1)):
    return SimpleNamespace(
        core=SimpleNamespace(
            turn=turn,
            agent=SimpleNamespace(lives=own[0], wins=own[1]),
            opponent=SimpleNamespace(lives=opponent[0], wins=opponent[1]),
        )
    )


def test_additive_calibrator_uses_turn_and_public_score_position():
    calibrator = AdditiveCriticCalibrator((1.0, 0.2, 0.4, 0.1, 0.3))
    assert calibrator.correct(2.0, state(3)) == pytest.approx(1.0)
    assert calibrator.correct(2.0, state(6)) == pytest.approx(0.8)
    assert calibrator.correct(2.0, state(10)) == pytest.approx(0.6)
    assert calibrator.correct(
        2.0, state(3, own=(4, 1), opponent=(4, 1))
    ) == pytest.approx(0.9)
    assert calibrator.correct(
        2.0, state(3, own=(3, 5), opponent=(4, 0))
    ) == pytest.approx(0.7)


def test_calibrator_refuses_unvalidated_or_mismatched_reports():
    report = {
        "recommended": "use-additive-bias",
        "featureNames": ["intercept", "mid", "late", "even", "behind"],
        "finalCoefficients": [0, 0, 0, 0, 0],
        "simulatorRevision": "current",
    }
    AdditiveCriticCalibrator.from_report(report, expected_revision="current")
    with pytest.raises(ValueError, match="revision"):
        AdditiveCriticCalibrator.from_report(
            report, expected_revision="different"
        )
    with pytest.raises(ValueError, match="does not recommend"):
        AdditiveCriticCalibrator.from_report(
            {**report, "recommended": "raw-critic"}
        )
