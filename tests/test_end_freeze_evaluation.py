from collections import Counter
from pathlib import Path
import sys

import numpy as np
import pytest
import torch

from sapai_gym1.rl import AutoregressiveAction, ActionType
from sapai_gym1.rl.end_freeze import step_with_end_freeze
from test_end_freeze_policy import policies
from test_end_freeze import assert_same_environment

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
import evaluate_paired_policy as evaluation
import visualize_entity_game as viewer


@pytest.mark.parametrize('side', [0, 1])
def test_scalar_and_batched_evaluation_execute_identical_freeze_choices(side):
    env, primitive, joint = policies()
    env.core.agent.gold = env.core.opponent.gold = 0
    observations = env.observations()
    with torch.no_grad():
        joint.type_head.bias[0] = 20
        joint.end_freeze_head.scorer.bias.fill_(10)
    memory = joint.initial_memory(1)
    action, next_memory, mask = evaluation.policy_action(joint, observations[side], memory, torch.device('cpu'), True)
    actions, memories, masks = evaluation.batched_policy_actions(joint, [{'observation': observations[side], 'memory': memory[0]}],
                                                                 torch.device('cpu'), True)
    assert actions == [action] and masks == [mask] and mask > 0
    torch.testing.assert_close(memories[0], next_memory[0], rtol=0, atol=0)
    expected = env.clone()
    selected = [AutoregressiveAction(ActionType.END_TURN)] * 2
    selected[side] = action
    joint_masks = [None, None]; joint_masks[side] = mask
    wanted = step_with_end_freeze(expected, *selected, mask_ids=tuple(0 if m is None else m for m in joint_masks))
    actual = evaluation.step_policy_actions(env, selected, joint_masks)
    assert actual.rewards == wanted.result.rewards
    assert_same_environment(env, expected)


@pytest.mark.parametrize('side', [0, 1])
def test_viewer_records_each_toggle_without_advancing_policy_recurrence(side):
    env, _, joint = policies()
    env.core.agent.gold = env.core.opponent.gold = 0
    observations = env.observations()
    with torch.no_grad():
        joint.type_head.bias[0] = 20
        joint.end_freeze_head.scorer.bias.fill_(10)
    action, memory, diagnostics = viewer.policy_action(joint, observations[side], joint.initial_memory(1))
    mask = diagnostics['endFreezeMask']
    assert mask > 0
    selected = [AutoregressiveAction(ActionType.END_TURN)] * 2
    selected[side] = action
    masks = [None, None]; masks[side] = mask
    expected = env.clone()
    wanted = step_with_end_freeze(expected, *selected, mask_ids=tuple(0 if m is None else m for m in masks))
    counts = [Counter(), Counter()]
    frames, action_number = viewer.record_end_freeze_prefixes(env, selected, masks, side, 1, 0, counts)
    assert len(frames) == action_number == mask.bit_count()
    assert counts[side]['FREEZE'] == mask.bit_count()
    assert all(frame['policy'] is None and frame['jointEndFreezeMask'] == mask for frame in frames)
    assert not any(frame['warnings'] for frame in frames)
    actual = env.step(*selected)
    assert actual.rewards == wanted.result.rewards
    assert_same_environment(env, expected)
