import numpy as np
import pytest

from sapai import Team, Shop, Pet
from sapai_gym1.rl import ParallelSelfPlayEnv, ActionType, AutoregressiveAction
from sapai_gym1.rl.decision_manifest import observation_hash
from sapai_gym1.rl.opponent_reservoir import OpponentTrajectoryReservoir
from sapai_gym1.rl.rollout_search import PublicSearchState
from sapai_gym1.rl.public_opponent_model import public_state_from_reservoir


def reservoir():
    trajectories = []
    for source, pet in enumerate(['ant', 'beaver']):
        for index in range(5):
            trajectories.append({'id': f'{source}-{index}', 'sourceIndex': source,
                                 'boards': [{'turn': turn, 'team': Team([pet]).state} for turn in range(1,6)]})
    return OpponentTrajectoryReservoir(trajectories)


def test_opening_prior_is_reproducible_source_balanced_and_coherent():
    pool = reservoir()
    first = pool.opening_prior(seed=17, limit=4)
    second = pool.opening_prior(seed=17, limit=4)
    assert [entry.trajectory_id for entry in first] == [entry.trajectory_id for entry in second]
    assert len(set(entry.trajectory_id for entry in first)) == 4
    assert {entry.source_index for entry in first} == {0,1}
    for entry in first:
        assert len(entry.teams) == 5
        assert len({team[0].pet.name for team in entry.teams}) == 1
    with pytest.raises(ValueError, match='No complete opening'):
        pool.opening_prior(seed=17, horizon=6)


@pytest.mark.parametrize('side', [0,1])
def test_opening_search_preserves_unknown_history_and_never_copies_hidden_opponent(side):
    env = ParallelSelfPlayEnv()
    env.reset(seed=918)
    matches = reservoir().opening_prior(seed=11, limit=4)
    prior = [entry.teams for entry in matches]
    with pytest.raises(ValueError, match='observed opponent board'):
        PublicSearchState.from_core(env.core, side)
    with pytest.raises(ValueError, match='populated'):
        PublicSearchState.from_core(env.core, side, opening_prior=[[Team([])]])
    first = PublicSearchState.from_core(env.core, side, opening_prior=prior)
    actual = env.observations()[side]
    assert observation_hash(first.observation()) == observation_hash(actual)
    hidden = env.core.opponent if side == 0 else env.core.agent
    hidden.team = Team(['dragon', 'tiger', 'fly'])
    hidden.shop = Shop([Pet('leopard')], turn=1, fixed_rules=True)
    hidden.gold = 3
    hidden._connect_objects()
    second = PublicSearchState.from_core(env.core, side, opening_prior=prior)
    assert observation_hash(second.observation()) == observation_hash(first.observation())
    clone = second.clone()
    assert clone.core.previous_opponent_state is None
    assert clone.opponent_trajectories is second.opponent_trajectories
    assert observation_hash(clone.observation()) == observation_hash(first.observation())
    # A real prior board must fight; the empty placeholder cannot award a win.
    clone.core.agent.gold = 0
    clone.core._agent_actions_cache = None
    result = clone.step(AutoregressiveAction(ActionType.END_TURN), np.random.default_rng(44))
    assert result.battle_result == 1
    assert clone.core.agent.lives == 5
    assert clone.core.previous_opponent_state is not None
    assert any(not slot.empty for slot in clone.core.previous_opponent_state)
    assert observation_hash(clone.clone().observation()) == observation_hash(clone.observation())


def test_opening_prior_cannot_hide_missing_later_history():
    env = ParallelSelfPlayEnv()
    env.reset(seed=93)
    env.core.agent.turn = 2
    with pytest.raises(ValueError, match='only on turn one'):
        PublicSearchState.from_core(env.core, 0, opening_prior=[[Team(['ant'])]])


def test_public_builder_installs_opening_prior_then_reactive_progression():
    env = ParallelSelfPlayEnv()
    env.reset(seed=934)
    pool = reservoir()
    with pytest.raises(ValueError, match='declared opening'):
        public_state_from_reservoir(env.core, 0, pool)
    state, info = public_state_from_reservoir(env.core, 0, pool, opening_seed=11, limit=4)
    assert info['kind'] == 'opening_population'
    assert info['sourceCounts'] == {'0': 2, '1': 2}
    assert state.opponent_transition_model is pool
    assert state.core.previous_opponent_state is None
    state.core.agent.gold = 0
    state.core._agent_actions_cache = None
    state.step(AutoregressiveAction(ActionType.END_TURN), np.random.default_rng(45))
    rebuilt, info = public_state_from_reservoir(state.core, 0, pool, horizon=2, limit=4)
    assert info['kind'] == 'matched_public_history'
    assert rebuilt.core.previous_opponent_state is not None
    with pytest.raises(ValueError, match='replace observed'):
        public_state_from_reservoir(state.core, 0, pool, opening_seed=11)


def test_long_opening_priors_cannot_select_only_survivors():
    with pytest.raises(ValueError, match='survival'):
        reservoir().opening_prior(seed=1, horizon=7)
