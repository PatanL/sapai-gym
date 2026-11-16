from pathlib import Path
import sys


sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from analyze_puct_exploration import summarize, turn_band  # noqa: E402


def row(
    turn,
    *,
    mode,
    candidate=False,
    accepted=False,
    reasons=(),
    split="train",
    **extra,
):
    multi = {
        "enabled": True,
        "fullSearch": mode != "screen-reject",
        "screenPass": mode == "screen-pass",
        "screenBypass": mode == "bypass",
    }
    return {
        "game": 0,
        "side": 0,
        "turn": turn,
        "split": split,
        "reasons": list(reasons),
        "candidateEligible": candidate,
        "accepted": accepted,
        "wallSeconds": 10.0,
        "multiFidelity": multi,
        "policyActionDetail": {"description": "Roll shop"},
        "proposedActionDetail": {
            "description": "Freeze Chocolate",
            "type": "FREEZE",
        },
        "validationReturnImprovement": 0.1,
        "validationReturnLowerBound": 0.01,
        "outerAuditReturnImprovement": 0.08,
        "outerAuditReturnLowerBound": 0.02,
        **extra,
    }


def test_summary_exposes_turn_coverage_and_screen_precision():
    report = summarize(
        [
            row(2, mode="screen-reject", reasons=("matching-pet",)),
            row(
                10,
                mode="bypass",
                candidate=True,
                accepted=True,
                reasons=("subcost-premium-freeze",),
            ),
            row(16, mode="screen-pass", candidate=True),
        ]
    )
    assert report["turnBandCounts"] == {
        "early": 1,
        "mid": 0,
        "late": 1,
        "endgame": 1,
    }
    assert report["escalationModes"]["bypass"]["acceptanceRate"] == 1.0
    assert report["escalationModes"]["screen-pass"]["acceptanceRate"] == 0.0
    assert report["accepted"][0]["puctAction"] == "Freeze Chocolate"
    assert report["strictAcceptedCount"] == 1


def test_turn_band_boundaries_match_real_six_heart_match_length():
    assert turn_band(4) == "early"
    assert turn_band(5) == "mid"
    assert turn_band(10) == "late"
    assert turn_band(11) == "endgame"


def test_summary_reports_independently_confirmed_pruning_false_negative():
    report = summarize(
        [
            row(
                7,
                mode="bypass",
                candidate=True,
                accepted=True,
                exhaustiveRoot=True,
                proposedExcludedByPruning=True,
                treeProposedExcludedByPruning=True,
                prunedActionCounts={"BUY_PET": 4, "BUY_FOOD": 2},
            ),
            row(
                8,
                mode="bypass",
                exhaustiveRoot=True,
                proposedExcludedByPruning=False,
                treeProposedExcludedByPruning=False,
                prunedActionCounts={"BUY_PET": 3},
            ),
        ]
    )
    audit = report["rootPruningAudit"]
    assert audit["exhaustiveFullSearches"] == 2
    assert audit["independentlyConfirmedPrunedWinners"] == 1
    assert audit["confirmedFalseNegativeRate"] == 0.5
    assert audit["confirmedWinnersByAction"] == {"FREEZE": 1}
    assert audit["removedActionsByType"] == {
        "BUY_FOOD": 2,
        "BUY_PET": 7,
    }


def test_readiness_can_require_exhaustive_root_audit_coverage():
    rows = [
        row(
            7,
            mode="bypass",
            exhaustiveRoot=index < 2,
            proposedExcludedByPruning=False,
        )
        for index in range(3)
    ]
    insufficient = summarize(rows, minimum_exhaustive_root_audits=3)
    sufficient = summarize(rows, minimum_exhaustive_root_audits=2)
    assert not insufficient["readiness"]["readyForDistillation"]
    assert sufficient["readiness"]["readyForDistillation"]


def test_readiness_requires_actual_opponent_source_diversity():
    base = row(
        7,
        mode="bypass",
        validationPlannedOpponentSourceCounts={
            "source-00": 10,
            "source-01": 5,
            "source-02": 3,
        },
    )
    insufficient = summarize([base], minimum_opponent_sources=4)
    assert not insufficient["readiness"]["readyForDistillation"]
    base["validationPlannedOpponentSourceCounts"]["source-03"] = 2
    sufficient = summarize([base], minimum_opponent_sources=4)
    assert sufficient["readiness"]["readyForDistillation"]


def test_distillation_readiness_requires_every_band_and_strict_roots():
    incomplete = summarize(
        [row(2, mode="bypass", candidate=True, accepted=True)],
        minimum_states_per_band=1,
        minimum_full_searches_per_band=1,
        minimum_strict_roots=1,
        minimum_strict_bands=2,
    )
    assert not incomplete["readiness"]["readyForDistillation"]

    complete = summarize(
        [
            row(turn, mode="bypass", candidate=True, accepted=True)
            for turn in (2, 5, 9, 13)
        ],
        minimum_states_per_band=1,
        minimum_full_searches_per_band=1,
        minimum_strict_roots=4,
        minimum_strict_bands=4,
    )
    assert complete["readiness"]["readyForDistillation"]
    assert complete["strictAcceptedByBand"] == {
        "early": 1,
        "mid": 1,
        "late": 1,
        "endgame": 1,
    }
    assert complete["fullSearchesByBand"] == {
        "early": 1,
        "mid": 1,
        "late": 1,
        "endgame": 1,
    }


def test_distillation_readiness_requires_disjoint_strict_holdout():
    rows = [
        row(2, mode="bypass", candidate=True, accepted=True),
        row(5, mode="bypass", candidate=True, accepted=True),
        row(9, mode="bypass", candidate=True, accepted=True),
        row(11, mode="bypass", candidate=True, accepted=True),
    ]
    no_holdout = summarize(
        rows,
        minimum_strict_train_roots=3,
        minimum_strict_holdout_roots=1,
    )
    assert not no_holdout["readiness"]["readyForDistillation"]
    rows[-1]["split"] = "holdout"
    with_holdout = summarize(
        rows,
        minimum_strict_train_roots=3,
        minimum_strict_holdout_roots=1,
    )
    assert with_holdout["readiness"]["readyForDistillation"]
    assert with_holdout["strictAcceptedBySplit"] == {
        "holdout": 1,
        "train": 3,
    }
