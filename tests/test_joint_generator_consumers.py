import json
from pathlib import Path
import sys

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'scripts'))
from generate_search_demonstrations import policy_action, force_action, step_policy_decisions
from sapai_gym1.rl.joint_search_records import make_record, diagnostic_record_loss, rescore_record
from sapai_gym1.rl.joint_action import JointAction
from sapai_gym1.rl.action_schema import ActionType, AutoregressiveAction
from test_joint_search_roundtrip import fixture


def test_actual_generator_helpers_keep_joint_mask_and_one_memory_advance():
    env, policy, state, evaluator, memory = fixture()
    observation = state.observation()
    action, sampled_memory = policy_action(policy, observation, memory, torch.device('cpu'), True)
    assert isinstance(action, JointAction) and action.end_freeze_mask is not None
    calls = []
    hook = policy.memory_cell.register_forward_hook(lambda *args: calls.append(1))
    forced_memory = force_action(policy, observation, memory, action, torch.device('cpu'))
    hook.remove()
    assert len(calls) == 1
    torch.testing.assert_close(sampled_memory, forced_memory, atol=0, rtol=0)
    ends = [a for a in evaluator.evaluate(state, memory).priors if a.action_type == ActionType.END_TURN]
    nonzero = next(a for a in ends if a.end_freeze_mask)
    # The production generator's executor handles real two-seat transitions.
    env.core.agent.gold = env.core.opponent.gold = 0
    result = step_policy_decisions(env, (nonzero, JointAction(AutoregressiveAction(ActionType.END_TURN), 0)))
    assert result.battle_resolved


def test_restored_soft_targets_reach_differentiable_joint_consumer_once():
    env, policy, state, evaluator, memory = fixture()
    result = evaluator.evaluate(state, memory)
    actions = tuple(result.priors)
    record = make_record(state.observation(), memory, actions, tuple(result.priors.values()),
                         checkpoint_sha256='a'*64, omitted_probability_mass=result.omitted_probability_mass)
    record = json.loads(json.dumps(record))
    logps, _ = rescore_record(record, policy, checkpoint_sha256='a'*64)
    calls = []
    hook = policy.memory_cell.register_forward_hook(lambda *args: calls.append(1))
    loss = diagnostic_record_loss(record, policy, checkpoint_sha256='a'*64)
    hook.remove()
    assert len(calls) == 1
    assert loss.item() == pytest.approx(-sum(p*lp for p, lp in zip(record['targetProbabilities'], logps)), abs=2e-6)
    loss.backward()
    assert all(p.grad is None or torch.isfinite(p.grad).all() for p in policy.parameters())
    with pytest.raises(ValueError, match='Foreign checkpoint'):
        diagnostic_record_loss(record, policy, checkpoint_sha256='b'*64)
    record['productionTrainingAllowed'] = True
    with pytest.raises(ValueError, match='diagnostic-only'):
        diagnostic_record_loss(record, policy, checkpoint_sha256='a'*64)
