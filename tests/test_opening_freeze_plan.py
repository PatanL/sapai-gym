from types import SimpleNamespace

import numpy as np
import pytest
import torch

from sapai_gym1.rl import ActionType, AutoregressiveAction
from sapai_gym1.rl import ParallelSelfPlayEnv
from sapai import Team, Shop, Pet
from sapai_gym1.rl.opening_freeze_plan import next_freeze_plan_action, OpeningFreezePlanEvaluator


def observation():
    result = {'item_ids': np.zeros(21, dtype=int), 'numeric': np.zeros((21,16), dtype=np.float32),
              'type_mask': np.ones(9, dtype=bool), 'source_mask': np.ones((9,11), dtype=bool),
              'target_mask': np.ones((9,11,6), dtype=bool)}
    result['item_ids'][6:9] = [7,8,7]
    result['numeric'][0,12] = 1/25
    return result


def test_canonical_plan_freezes_both_then_ends_without_mutating_observation():
    obs = observation()
    assert next_freeze_plan_action(obs, [7,8,7], [True,False,True]) == AutoregressiveAction(ActionType.FREEZE,0)
    assert not obs['numeric'][6,6]
    obs['numeric'][6,6] = 1
    assert next_freeze_plan_action(obs, [7,8,7], [True,False,True]) == AutoregressiveAction(ActionType.FREEZE,2)
    obs['numeric'][8,6] = 1
    assert next_freeze_plan_action(obs, [7,8,7], [True,False,True]) == AutoregressiveAction(ActionType.END_TURN)
    obs['item_ids'][6] = 9
    with pytest.raises(ValueError, match='inventory'):
        next_freeze_plan_action(obs, [7,8,7], [True,False,True])


def test_constraint_respects_legality_and_only_own_opening_turn():
    obs = observation()
    obs['target_mask'][6,0,5] = False
    with pytest.raises(ValueError, match='legally reachable'):
        next_freeze_plan_action(obs, [7,8,7], [True,False,True])
    obs = observation()
    policy = SimpleNamespace(config=SimpleNamespace(memory_dim=2))
    evaluator = OpeningFreezePlanEvaluator(policy, policy, acting_side=0,
                                           item_ids=[7,8,7], desired_frozen=[True,False,True])
    raw = AutoregressiveAction(ActionType.END_TURN)
    memory = torch.zeros(2)
    assert evaluator._continuation_choice(obs,memory,1,raw,memory)[0] == raw
    assert evaluator._continuation_choice(obs,memory,0,raw,memory)[0].action_type == ActionType.FREEZE
    obs['numeric'][0,12] = 2/25
    assert evaluator._continuation_choice(obs,memory,0,raw,memory)[0] == raw


def test_complete_two_freeze_prefix_executes_before_first_battle():
    class EndPolicy:
        config = SimpleNamespace(memory_dim=2)
        def __call__(self, observations, memory, deterministic):
            return SimpleNamespace(actions=torch.tensor([[0,10,5]]).repeat(len(memory),1),
                                   memory=memory+1)
    env = ParallelSelfPlayEnv()
    env.reset(seed=227)
    env.core.agent.team = Team(['otter'])
    env.core.agent.shop = Shop([Pet('otter'),Pet('beaver'),Pet('otter')], turn=1, fixed_rules=True)
    env.core.agent.gold = env.core.opponent.gold = 0
    env.core.agent._connect_objects()
    env.core.shop_agent, env.core.team_agent = env.core.agent.shop, env.core.agent.team
    env.core._agent_actions_cache = env.core._opponent_actions_cache = None
    obs = env.observations()[0]
    policy = EndPolicy()
    evaluator = OpeningFreezePlanEvaluator(policy, policy, acting_side=0, horizon_battles=1,
                                           item_ids=obs['item_ids'][6:9], desired_frozen=[True,False,True])
    first = next_freeze_plan_action(obs, obs['item_ids'][6:9], [True,False,True])
    result = evaluator.evaluate(env, [torch.zeros(2),torch.zeros(2)], 0, [first], [91])
    assert result[0].battles == 1
    assert evaluator.counters['FREEZE'] == 1  # second freeze; root is separately forced
    assert evaluator.counters['END_TURN'] == 1
    assert all(not slot.frozen for slot in env.core.agent.shop)
