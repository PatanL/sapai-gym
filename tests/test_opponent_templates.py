import copy
from dataclasses import FrozenInstanceError
from unittest.mock import patch

import numpy as np
import pytest

from sapai import Pet, Team
from sapai_gym1.rl import ParallelSelfPlayEnv
from sapai_gym1.rl.opponent_templates import FrozenOpponentTrajectories
from sapai_gym1.rl.rollout_search import PublicSearchState
from sapai_gym1.rl.decision_manifest import observation_hash


def seeded_team():
    pet = Pet('fish', seed_state=np.random.RandomState(981).get_state())
    pet._attack = 10
    pet._until_end_of_battle_attack_buff = 2
    return Team([pet], seed_state=np.random.RandomState(982).get_state())


def test_templates_preserve_state_and_isolate_both_input_and_materialized_edits():
    team = seeded_team()
    expected = Team.from_state(team.state).state
    frozen = FrozenOpponentTrajectories.freeze([[team]])
    assert frozen[0][0].state == expected
    team[0].pet._attack = 49
    first = frozen[0][0]
    first[0].pet._attack = 1
    first[0].pet.override_ability_dict = {'new': [1, 2, 3]}
    assert frozen[0][0].state == expected
    assert copy.deepcopy(frozen) is frozen
    assert copy.deepcopy(frozen[0]) is frozen[0]
    with pytest.raises(FrozenInstanceError):
        frozen.trajectories = ()


def test_search_clone_shares_only_immutable_templates_without_rebuilding_population():
    env = ParallelSelfPlayEnv()
    env.reset(seed=981)
    env.core.previous_opponent_state = seeded_team()
    state = PublicSearchState.from_core(env.core, 0)
    state.set_opponent_trajectories([[seeded_team() for _ in range(5)] for _ in range(32)])
    before = observation_hash(state.observation())
    with patch.object(Pet, 'from_state', wraps=Pet.from_state) as from_state:
        cloned = state.clone()
        # The old clone rebuilt 32 * 5 * 5 pets, plus ordinary player copies.
        assert from_state.call_count < 100
    assert cloned.opponent_trajectories is state.opponent_trajectories
    assert observation_hash(cloned.observation()) == before
    materialized = cloned.opponent_trajectories[0][0]
    materialized[0].pet._attack = 49
    assert state.opponent_trajectories[0][0][0].pet.attack == 12
    assert cloned.opponent_trajectories[0][0][0].pet.attack == 12
    with pytest.raises(ValueError, match='trajectory_ids'):
        cloned.set_opponent_trajectories([[seeded_team()]], ['one', 'extra'])
    assert cloned.opponent_trajectories is state.opponent_trajectories


def test_empty_trajectories_are_rejected_or_filtered_consistently():
    with pytest.raises(ValueError, match='At least one'):
        FrozenOpponentTrajectories.freeze([[], iter(())])
    frozen = FrozenOpponentTrajectories.freeze([[], [seeded_team()]])
    assert len(frozen) == 1 and len(frozen[0]) == 1
