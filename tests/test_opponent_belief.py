import numpy as np
import pytest

from sapai import Team
from sapai_gym1.rl import (
    EntityObservationEncoder,
    OpponentBeliefMix,
    OpponentBoardBelief,
)


def belief_for(names):
    encoder = EntityObservationEncoder()
    pet_logits = np.full((5, encoder.num_items), -20.0)
    status_logits = np.full((5, encoder.num_statuses), -20.0)
    level_logits = np.full((5, 4), -20.0)
    stats = np.full((5, 2), 0.02)
    temporary_logits = np.full((5, 2), -20.0)
    for position, name in enumerate(names):
        item_id = 0 if name is None else encoder.item_to_id[f"pet-{name}"]
        pet_logits[position, item_id] = 20.0
        status_logits[position, encoder.status_to_id["none"]] = 20.0
        level_logits[position, 1] = 20.0
        if name is not None:
            stats[position] = (0.20 + position * 0.02, 0.30)
    return OpponentBoardBelief.from_logits(
        pet_logits,
        status_logits,
        level_logits,
        stats,
        temporary_logits,
        encoder,
    )


def test_modal_board_is_a_battle_legal_team():
    belief = belief_for(["ant", "fish", None, None, None])
    team = belief.modal_team()
    assert [slot.pet.name for slot in team if not slot.empty] == [
        "pet-ant",
        "pet-fish",
    ]
    assert (team[0].attack, team[0].health, team[0].level) == (10, 15, 1)
    assert (team[1].attack, team[1].health, team[1].level) == (11, 15, 1)


def test_neural_belief_reranks_coherent_reservoir_boards():
    belief = belief_for(["ant", "fish", None, None, None])
    matching = Team(["ant", "fish"])
    different = Team(["dragon", "cat"])
    probabilities = belief.reservoir_probabilities(
        [matching, different], temperature=0.25, uniform_mix=0.1
    )
    assert probabilities.sum() == pytest.approx(1.0)
    assert probabilities[0] > 0.9
    assert belief.team_score(matching) > belief.team_score(different)


def test_opponent_mixture_configuration_is_validated():
    assert OpponentBeliefMix().normalized_weights.sum() == pytest.approx(1.0)
    with pytest.raises(ValueError, match="weights"):
        OpponentBeliefMix(-1, 0, 0)
    with pytest.raises(ValueError, match="temperature"):
        OpponentBeliefMix(reservoir_temperature=0)
