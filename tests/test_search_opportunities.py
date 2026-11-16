from pathlib import Path
import sys

import numpy as np

from sapai_gym1.rl.curriculum import BUILDERS, _make_env


sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from generate_search_demonstrations import opportunity_reasons  # noqa: E402


def reasons(scenario, seed=123):
    env, _ = BUILDERS[scenario](np.random.default_rng(seed))
    return set(opportunity_reasons(env, 0))


def test_conditional_search_opportunities_cover_observed_failures():
    assert "turn-two-on-board-triple" in reasons(
        "search-turn-two-delay-triple"
    )
    assert "post-window-tier-one" in reasons("search-avoid-lone-tier-one")
    assert "active-snail-consolidation" in reasons(
        "search-preserve-active-snails"
    )
    assert "subcost-premium-freeze" in reasons(
        "search-premium-subcost-freeze"
    )


def test_conditional_search_opportunities_do_not_overfire_counterexamples():
    assert "turn-two-on-board-triple" not in reasons(
        "search-turn-two-triple-for-space"
    )
    assert "active-snail-consolidation" not in reasons(
        "search-combine-snails-for-space"
    )
    assert "subcost-premium-freeze" not in reasons(
        "search-buy-premium-affordable"
    )


def test_search_opportunities_cover_multi_pet_synergies_and_counters():
    assert "synergy:shark-fly" in reasons("shark-fly")
    assert "synergy:elephant-hurt-wolverine" in reasons(
        "elephant-hurt-wolverine"
    )
    assert "synergy:mushroom-faint" in reasons("mushroom-deer-shark")
    assert "counter:scorpion-tall" in reasons("scorpion-vs-tall")


def test_search_opportunities_cover_parrot_deer_and_tiger_mammoth():
    rng = np.random.default_rng(321)
    parrot_deer = _make_env(
        turn=8,
        gold=10,
        team=["pet-deer", "pet-fish", None, None, None],
        shop=["pet-parrot", "pet-penguin", "food-pear"],
        opponent=["pet-hippo", None, None, None, None],
        rng=rng,
    )
    assert "synergy:parrot-deer" in opportunity_reasons(parrot_deer, 0)

    tiger_mammoth = _make_env(
        turn=13,
        gold=10,
        team=["pet-mammoth", "pet-tiger", "pet-fish", None, None],
        shop=["food-mushroom", "pet-snake", "food-pizza"],
        opponent=["pet-rhino", None, None, None, None],
        rng=rng,
    )
    assert "synergy:tiger-mammoth-mushroom" in opportunity_reasons(
        tiger_mammoth, 0
    )


def test_search_opportunities_cover_public_matchup_counters():
    rng = np.random.default_rng(654)
    cases = (
        (
            "counter:rhino-summons",
            ["pet-fish", None, None, None, None],
            ["pet-rhino", "pet-penguin", "food-pear"],
            ["pet-deer", "pet-turkey", None, None, None],
        ),
        (
            "counter:avoid-summons-rhino",
            ["pet-deer", "pet-shark", None, None, None],
            ["pet-deer", "pet-penguin", "food-pear"],
            ["pet-rhino", None, None, None, None],
        ),
        (
            "counter:crocodile-exposed-tiger",
            ["pet-fish", None, None, None, None],
            ["pet-crocodile", "pet-monkey", "food-pear"],
            ["pet-hippo", "pet-tiger", None, None, None],
        ),
        (
            "counter:leopard-weak-melon",
            ["pet-fish", None, None, None, None],
            ["pet-leopard", "pet-monkey", "food-pizza"],
            ["pet-hippo", "pet-snake", "pet-tiger", None, None],
        ),
        (
            "counter:elephant-blowfish-fragile-engine",
            ["pet-mammoth", "pet-tiger", None, None, None],
            ["pet-snake", "pet-monkey", "food-melon"],
            ["pet-elephant", "pet-blowfish", None, None, None],
        ),
    )
    for reason, team, shop, opponent in cases:
        env = _make_env(
            turn=11,
            gold=10,
            team=team,
            shop=shop,
            opponent=opponent,
            rng=rng,
        )
        assert reason in opportunity_reasons(env, 0)

    skunk = _make_env(
        turn=9,
        gold=10,
        team=["pet-fish", None, None, None, None],
        shop=["pet-skunk", "pet-deer", "food-pear"],
        opponent=["pet-hippo", None, None, None, None],
        rng=rng,
    )
    skunk.previous_opponent_state[0].pet._health = 40
    assert "counter:skunk-tall" in opportunity_reasons(skunk, 0)
