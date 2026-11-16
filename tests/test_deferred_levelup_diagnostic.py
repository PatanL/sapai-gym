from types import SimpleNamespace

import numpy as np
import pytest
import torch

from sapai_gym1.rl.action_schema import ActionType, AutoregressiveAction
from sapai_gym1.rl.deferred_levelup_diagnostic import (
    DeferredLevelUpDiagnostic, constrained_hierarchical_choice, without_level_two,
)


def observation():
    ids = np.zeros(21, dtype=int)
    ids[[1, 2, 6]] = 7
    numeric = np.zeros((21,16), dtype=np.float32)
    numeric[0,12] = 2/25
    numeric[[1,2,6],2] = 1/3
    numeric[1,3] = 1/5
    mask = np.zeros((9,11,6), dtype=bool)
    for action in [(2,0,0),(2,0,1),(2,0,2),(5,0,1),(5,1,0),(1,10,5)]:
        mask[action] = True
    return {'item_ids': ids, 'numeric': numeric, 'target_mask': mask}


def test_only_level_completing_merges_are_removed():
    obs = observation()
    original = obs['target_mask'].copy()
    allowed = without_level_two(obs, 7)
    assert not allowed[2,0,0] and not allowed[5,0,1] and not allowed[5,1,0]
    assert allowed[2,0,1] and allowed[2,0,2] and allowed[1,10,5]
    np.testing.assert_array_equal(original, obs['target_mask'])
    np.testing.assert_array_equal(without_level_two(obs, 99), original)


def test_constraint_keeps_original_hierarchical_preferences():
    joint = torch.full((9,11,6), -torch.inf)
    joint[2,0,0], joint[2,0,2], joint[1,10,5] = torch.log(torch.tensor([.6,.1,.3]))
    allowed = observation()['target_mask'].copy()
    allowed[2,0,0] = False
    assert constrained_hierarchical_choice(joint, allowed) == AutoregressiveAction(2,0,2)
    with pytest.raises(ValueError, match='No legal alternative'):
        constrained_hierarchical_choice(joint, np.zeros_like(allowed))


def test_only_own_turn_two_is_constrained_and_live_memory_is_preserved():
    joint = torch.full((1,9,11,6), -torch.inf)
    joint[0,2,0,0], joint[0,2,0,2], joint[0,1,10,5] = torch.log(torch.tensor([.6,.1,.3]))
    calls = []
    def evaluate_all_actions(batch, memory):
        assert batch['target_mask'][0,2,0,0]  # inputs are not rewritten
        calls.append(1)
        return SimpleNamespace(joint_log_prob=joint)
    policy = SimpleNamespace(config=SimpleNamespace(memory_dim=2), evaluate_all_actions=evaluate_all_actions)
    evaluator = DeferredLevelUpDiagnostic(policy, policy, acting_side=0, species_id=7)
    obs, memory, next_memory = observation(), torch.zeros(2), torch.ones(2)
    raw = AutoregressiveAction(ActionType.BUY_PET,0,0)
    assert evaluator._continuation_choice(obs, memory, 1, raw, next_memory)[0] == raw
    action, updated = evaluator._continuation_choice(obs, memory, 0, raw, next_memory)
    assert action == AutoregressiveAction(2,0,2) and updated is next_memory
    obs['numeric'][0,12] = 3/25
    assert evaluator._continuation_choice(obs, memory, 0, raw, next_memory)[0] == raw
    assert len(calls) == 1
