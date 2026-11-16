import copy
import json

import pytest
from sapai import Team
from sapai_gym1.rl.selfplay_env import ParallelSelfPlayEnv
from sapai_gym1.rl.opponent_context_data import PublicOpponentContext, OneStepRevealCollector, split_by_match, validate_row


def environment():
    env=ParallelSelfPlayEnv();env.reset(seed=501)
    env.core.previous_opponent_state=Team(['pet-fish'])
    env.core.previous_previous_opponent_state=Team(['pet-ant'])
    env.core.previous_agent_state=Team(['pet-pig'])
    return env


def test_context_is_independent_of_current_private_shops_teams_and_gold():
    env=environment()
    before=PublicOpponentContext.capture(env.core,0).features()
    env.core.agent.team=Team(['pet-dragon'])
    env.core.opponent.team=Team(['pet-cat'])
    env.core.agent.gold=99;env.core.opponent.gold=88
    env.core.agent.shop.roll();env.core.opponent.shop.roll()
    assert PublicOpponentContext.capture(env.core,0).features()==before
    env.core.previous_opponent_state[0].pet._attack+=1
    assert PublicOpponentContext.capture(env.core,0).features()!=before


def test_only_temporary_flags_not_hidden_amounts_are_captured():
    env=environment();pet=env.core.previous_opponent_state[0].pet
    pet._attack=9;pet._until_end_of_battle_attack_buff=3
    first=PublicOpponentContext.capture(env.core,0).features()
    pet._attack=10;pet._until_end_of_battle_attack_buff=2
    assert PublicOpponentContext.capture(env.core,0).features()==first
    assert first['opponent_last'][0]['attack']==12
    assert first['opponent_last'][0]['temporaryAttack'] is True
    assert 'sellValue' not in first['opponent_last'][0]


def test_perspective_and_missing_flags():
    env=environment()
    p0=PublicOpponentContext.capture(env.core,0).features()
    p1=PublicOpponentContext.capture(env.core,1).features()
    assert p0['opponent_last'][0]['species']=='pet-fish'
    assert p0['our_last'][0]['species']=='pet-pig'
    assert p1['opponent_last'][0]['species']=='pet-pig'
    assert p1['our_last'][0]['species']=='pet-fish'
    fresh=ParallelSelfPlayEnv();fresh.reset(seed=4)
    assert PublicOpponentContext.capture(fresh.core,0).features()['missing_history']==[True,True,True]


def test_terminal_reveal_is_kept_without_any_future_survival_requirement():
    env=environment()
    collector=OneStepRevealCollector(simulator_revision='test',collection_id='collection')
    collector.begin_turn(env.core,'last-turn',populations=('new','old'))
    collector.revealed(match_id='last-turn',turn=1,submitted_boards=(Team(['pet-ant']),Team(['pet-fish'])),terminal=True)
    assert len(collector.examples)==2 and not collector.pending
    for row in collector.examples:
        assert row['metadata']['terminal_after_reveal']
        restored=json.loads(json.dumps(row))
        assert validate_row(restored) is restored
    with pytest.raises(ValueError,match='No matching'):
        collector.revealed(match_id='last-turn',turn=1,submitted_boards=(Team([]),Team([])))


def test_all_sides_and_turns_of_a_match_stay_together_and_population_holdout_is_conservative():
    env=environment()
    collector=OneStepRevealCollector(simulator_revision='test',collection_id='collection')
    for match in ('known','heldout'):
        populations=('new','old') if match=='known' else ('new','unseen')
        collector.begin_turn(env.core,match,populations=populations)
        collector.revealed(match_id=match,turn=1,submitted_boards=(Team([]),Team(['pet-fish'])))
    train,test=split_by_match(collector.examples,heldout_populations={'unseen'},fraction=0)
    assert len(train)==len(test)==2
    assert {r['metadata']['match_id'] for r in train}=={'known'}
    assert {r['metadata']['match_id'] for r in test}=={'heldout'}


def test_private_extra_features_and_wrong_reveal_timing_are_rejected():
    env=environment();collector=OneStepRevealCollector(simulator_revision='test',collection_id='c')
    env.core.shop_action_count_opponent=1
    with pytest.raises(ValueError,match='before either'):
        collector.begin_turn(env.core,'m',populations=('a','b'))
    env.core.shop_action_count_opponent=0
    collector.begin_turn(env.core,'m',populations=('a','b'))
    with pytest.raises(ValueError,match='No matching'):
        collector.revealed(match_id='m',turn=2,submitted_boards=(Team([]),Team([])))
    collector.revealed(match_id='m',turn=1,submitted_boards=(Team(['pet-fish']),Team(['pet-ant'])))
    row=copy.deepcopy(collector.examples[0]);row['context']['opponent_gold']=10
    with pytest.raises(ValueError,match='private features'):
        validate_row(row)
    row=copy.deepcopy(collector.examples[0]);row['target'][0]['temporaryAmount']=7
    with pytest.raises(ValueError,match='private pet'):
        validate_row(row)
