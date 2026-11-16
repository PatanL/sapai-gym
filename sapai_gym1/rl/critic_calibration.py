"""Held-out validated value calibration for simulator search leaves."""

from __future__ import annotations

from dataclasses import dataclass


CALIBRATION_FEATURES = ("intercept", "mid", "late", "even", "behind")


@dataclass(frozen=True)
class AdditiveCriticCalibrator:
    coefficients: tuple[float, ...]
    simulator_revision: str | None = None

    def __post_init__(self):
        if len(self.coefficients) != len(CALIBRATION_FEATURES):
            raise ValueError("Critic calibration coefficient shape mismatch")

    @classmethod
    def from_report(cls, report, expected_revision=None):
        if report.get("recommended") != "use-additive-bias":
            raise ValueError("Critic report does not recommend calibration")
        if tuple(report.get("featureNames", ())) != CALIBRATION_FEATURES:
            raise ValueError("Unsupported critic calibration features")
        revision = report.get("simulatorRevision")
        if expected_revision is not None and revision != expected_revision:
            raise ValueError("Critic calibration simulator revision mismatch")
        return cls(
            tuple(map(float, report["finalCoefficients"])),
            simulator_revision=revision,
        )

    @staticmethod
    def features(state):
        turn = int(state.core.turn)
        own = state.core.agent
        opponent = state.core.opponent
        own_score = (int(own.lives), int(own.wins))
        opponent_score = (int(opponent.lives), int(opponent.wins))
        position = (
            "ahead"
            if own_score > opponent_score
            else "behind"
            if own_score < opponent_score
            else "even"
        )
        return (
            1.0,
            float(5 <= turn <= 8),
            float(turn >= 9),
            float(position == "even"),
            float(position == "behind"),
        )

    def correct(self, value, state):
        bias = sum(
            coefficient * feature
            for coefficient, feature in zip(
                self.coefficients, self.features(state)
            )
        )
        return float(value) - bias
