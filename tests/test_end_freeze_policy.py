from dataclasses import asdict, replace
import copy
import io

import numpy as np
import pytest
import torch

from sapai_gym1.rl import EntityTransformerConfig, EntityTransformerPolicy, ParallelSelfPlayEnv
from sapai_gym1.rl.training import BatchedSelfPlayCollector, learner_update, stack_observations


def policies():
    env = ParallelSelfPlayEnv()
    env.reset(seed=291)
    config = EntityTransformerConfig(
        num_items=env.encoder.num_items, num_statuses=env.encoder.num_statuses,
        d_model=32, nhead=4, num_layers=1, dim_feedforward=64, memory_dim=48,
        opponent_history_dim=16,
    )
    torch.manual_seed(923)
    primitive = EntityTransformerPolicy(config).eval()
    joint = EntityTransformerPolicy(replace(config, end_freeze=True)).eval()
    # Explicit test setup, not a permissive production load. Missing new-head
    # weights must fail a normal strict load rather than silently reinitialize.
    state = dict(primitive.state_dict())
    state.update({k: v for k, v in joint.state_dict().items() if k.startswith('end_freeze_head.')})
    joint.load_state_dict(state)
    return env, primitive, joint


def zero_gold_observations(env):
    env.core.agent.gold = env.core.opponent.gold = 0
    return stack_observations(env.observations(), 'cpu')


def test_legacy_greedy_actions_and_shared_outputs_are_unchanged():
    env, primitive, joint = policies()
    obs = zero_gold_observations(env)
    with torch.no_grad():
        old = primitive(obs, deterministic=True)
        new = joint(obs, deterministic=True, allow_end_freeze=True)
    for name in ('actions', 'value', 'memory', 'type_logits', 'source_logits', 'target_logits', 'battle_logits'):
        torch.testing.assert_close(getattr(old, name), getattr(new, name), rtol=0, atol=0)
    assert torch.count_nonzero(new.end_freeze_mask) == 0
    assert old.end_freeze_mask is None
    assert all(torch.equal(primitive.state_dict()[k], v) for k, v in joint.state_dict().items()
               if not k.startswith('end_freeze_head.'))
    assert joint.parameter_count() - primitive.parameter_count() == 65


def test_unsupported_consumers_and_missing_joint_actions_fail_closed():
    env, primitive, joint = policies()
    obs = zero_gold_observations(env)
    end = torch.tensor([[0, 10, 5]]).repeat(2, 1)
    with pytest.raises(ValueError, match='aware'):
        joint(obs)
    with pytest.raises(ValueError, match='Recorded joint'):
        joint(obs, actions=end, allow_end_freeze=True)
    with pytest.raises(ValueError, match='enumeration'):
        joint.evaluate_all_actions(obs)
    with pytest.raises(ValueError, match='Primitive policies'):
        primitive(obs, end_freeze_masks=torch.zeros(2, dtype=torch.long))
    sequence = {k: v[None] for k, v in obs.items()}
    with pytest.raises(ValueError, match='Recorded joint sequences'):
        joint.evaluate_sequence(sequence, joint.initial_memory(2), end[None])
    with pytest.raises(RuntimeError, match='end_freeze_head'):
        joint.load_state_dict(primitive.state_dict())


def test_recurrent_sequence_recomputes_joint_likelihood_and_memory():
    env, _, joint = policies()
    obs = zero_gold_observations(env)
    with torch.no_grad():
        joint.type_head.bias[0] = 15
        joint.end_freeze_head.scorer.weight.normal_(std=.03)
    memory = joint.initial_memory(2)
    results = []
    with torch.no_grad():
        for _ in range(3):
            output = joint(obs, memory=memory, allow_end_freeze=True)
            results.append(output)
            memory = output.memory
    actions = torch.stack([out.actions for out in results])
    masks = torch.stack([out.end_freeze_mask for out in results])
    sequence = {k: v[None].repeat(3, *([1] * v.ndim)) for k, v in obs.items()}
    evaluated = joint.evaluate_sequence(sequence, joint.initial_memory(2), actions,
                                        end_freeze_masks=masks)
    torch.testing.assert_close(evaluated.log_prob, torch.stack([out.log_prob for out in results]), atol=2e-6, rtol=2e-6)
    torch.testing.assert_close(evaluated.entropy, torch.stack([out.entropy for out in results]), atol=2e-6, rtol=2e-6)
    torch.testing.assert_close(evaluated.final_memory, memory, atol=2e-6, rtol=2e-6)
    (-evaluated.log_prob.mean()).backward()
    assert torch.isfinite(joint.end_freeze_head.scorer.weight.grad).all()
    assert joint.end_freeze_head.scorer.weight.grad.abs().sum() > 0


def test_checkpoint_round_trip_preserves_masks_and_likelihoods():
    env, _, joint = policies()
    obs = zero_gold_observations(env)
    original = joint(obs, allow_end_freeze=True)
    payload = io.BytesIO()
    torch.save({'config': asdict(joint.config), 'model': joint.state_dict()}, payload)
    payload.seek(0)
    saved = torch.load(payload, weights_only=False)
    restored = EntityTransformerPolicy(EntityTransformerConfig(**saved['config'])).eval()
    restored.load_state_dict(saved['model'])
    replay = restored(obs, actions=original.actions, end_freeze_masks=original.end_freeze_mask,
                      allow_end_freeze=True)
    torch.testing.assert_close(replay.log_prob, original.log_prob, rtol=0, atol=0)
    torch.testing.assert_close(replay.memory, original.memory, rtol=0, atol=0)


@pytest.mark.parametrize('legacy_resident', [False, True])
def test_collector_learner_preserves_joint_actions_with_legacy_opponents(legacy_resident):
    _, primitive, joint = policies()
    with torch.no_grad():
        joint.type_head.bias[0] = 20
        joint.end_freeze_head.scorer.bias.zero_()
    collector = BatchedSelfPlayCollector(joint, 1, 'cpu', seed=912, discount_mode='per_turn',
                                        opponent_policy=primitive if legacy_resident else None)
    for env in collector.envs:
        env.core.agent.gold = env.core.opponent.gold = 0
    collector.observations = [env.observations() for env in collector.envs]
    segments = collector.collect(2)
    transitions = [item for segment in segments for item in segment.transitions]
    assert all(item.end_freeze_mask is not None for item in transitions)
    assert any(item.end_freeze_mask > 0 for item in transitions)
    # Resident-controlled primitive actions are excluded, even when the main
    # policy now uses a different action schema in the same match.
    for segment in segments:
        batch = {k: torch.as_tensor(np.stack([item.observation[k] for item in segment.transitions]))[:, None]
                 for k in segment.transitions[0].observation}
        actions = torch.as_tensor(np.stack([item.action for item in segment.transitions]))[:, None]
        masks = torch.tensor([item.end_freeze_mask for item in segment.transitions])[:, None]
        evaluation = joint.evaluate_sequence(batch, segment.initial_memory[None], actions, end_freeze_masks=masks)
        torch.testing.assert_close(evaluation.log_prob[:, 0], torch.tensor([item.behavior_log_prob for item in segment.transitions]),
                                   atol=2e-5, rtol=2e-5)
        assert all(item.discount == 1 for item in segment.transitions[:-1])
        assert segment.transitions[-1].discount in (0., .99)
    optimizer = torch.optim.AdamW(joint.parameters(), lr=1e-4)
    before = joint.end_freeze_head.scorer.weight.detach().clone()
    metrics = learner_update(joint, optimizer, segments, 'cpu', use_bfloat16=False)
    assert np.isfinite(metrics.total_loss)
    assert not torch.equal(before, joint.end_freeze_head.scorer.weight)
    broken = copy.deepcopy(segments)
    broken[0].transitions[0].end_freeze_mask = None
    with pytest.raises(ValueError, match='action schema'):
        learner_update(joint, optimizer, broken, 'cpu')
    with pytest.raises(ValueError, match='action schema'):
        learner_update(primitive, torch.optim.AdamW(primitive.parameters()), segments, 'cpu')
