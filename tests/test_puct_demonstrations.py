from pathlib import Path
from types import SimpleNamespace
import sys

from sapai_gym1.rl import ActionType, AutoregressiveAction


sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from filter_puct_demonstrations import keep_row  # noqa: E402
from generate_puct_demonstrations import (  # noqa: E402
    filtered_target_actions,
    independently_confirmed,
    per_turn_search_limit,
)


def test_confirmed_root_action_receives_minimum_target_mass():
    preferred = AutoregressiveAction(ActionType.FREEZE, 2)
    alternatives = [
        preferred,
        AutoregressiveAction(ActionType.ROLL),
        AutoregressiveAction(ActionType.END_TURN),
    ]
    target = SimpleNamespace(
        actions=alternatives,
        probabilities=(0.05, 0.75, 0.20),
    )
    result = dict(
        filtered_target_actions(
            target,
            0.0,
            preferred_action=preferred,
            preferred_min_probability=0.5,
        )
    )
    assert result[preferred] == 0.5
    assert abs(sum(result.values()) - 1.0) < 1e-9


def test_strict_filter_can_exclude_internal_targets():
    root = {
        "teacher": {
            "name": "stochastic-puct",
            "rootAccepted": True,
            "outerAuditReturnImprovement": 0.1,
            "targetIsRoot": True,
        }
    }
    internal = {
        "teacher": {
            **root["teacher"],
            "targetIsRoot": False,
        }
    }

    assert keep_row(root)
    assert keep_row(internal)
    assert keep_row(root, root_only=True)
    assert not keep_row(internal, root_only=True)


def test_strict_filter_rejects_unconfirmed_or_nonpositive_targets():
    base = {
        "name": "stochastic-puct",
        "rootAccepted": True,
        "outerAuditReturnImprovement": 0.1,
        "targetIsRoot": True,
    }
    assert not keep_row({"teacher": {**base, "rootAccepted": False}})
    assert not keep_row(
        {"teacher": {**base, "outerAuditReturnImprovement": 0.0}}
    )
    assert not keep_row({"teacher": {**base, "name": "guide"}})


def test_strict_filter_can_require_positive_outer_confidence_bound():
    base = {
        "name": "stochastic-puct",
        "rootAccepted": True,
        "outerAuditReturnImprovement": 0.1,
        "targetIsRoot": True,
    }
    assert not keep_row(
        {
            "teacher": {
                **base,
                "outerAuditReturnLowerBound": -0.001,
            }
        },
        minimum_outer_lower_bound=0.0,
    )
    assert keep_row(
        {
            "teacher": {
                **base,
                "outerAuditReturnLowerBound": 0.001,
            }
        },
        minimum_outer_lower_bound=0.0,
    )


def test_zero_gold_slot_reservation_preserves_the_last_turn_search():
    assert per_turn_search_limit(2, gold=10, reserve_zero_gold_slot=True) == 1
    assert per_turn_search_limit(2, gold=1, reserve_zero_gold_slot=True) == 1
    assert per_turn_search_limit(2, gold=0, reserve_zero_gold_slot=True) == 2
    assert per_turn_search_limit(1, gold=10, reserve_zero_gold_slot=True) == 1
    assert per_turn_search_limit(2, gold=10, reserve_zero_gold_slot=False) == 2


def test_deployment_requires_untouched_outer_confirmation():
    accepted = SimpleNamespace(
        accepted=True, audit_return_lower_bound=0.01
    )
    outer_rejected = SimpleNamespace(
        accepted=True, audit_return_lower_bound=-0.01
    )
    missing_outer = SimpleNamespace(
        accepted=True, audit_return_lower_bound=None
    )
    assert independently_confirmed(accepted)
    assert not independently_confirmed(outer_rejected)
    assert not independently_confirmed(missing_outer)
