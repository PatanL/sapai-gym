"""Package A: same decision/return/optimizer semantics, fewer requested outputs."""
import copy
from dataclasses import asdict
import multiprocessing as mp
import threading

import numpy as np
import pytest
import torch

from sapai_gym1.rl.distributed import DynamicInferenceBatcher, SharedActorBuffers, DistributedSelfPlayRuntime
from sapai_gym1.rl.training import BatchedSelfPlayCollector, learner_update
from test_end_freeze_policy import policies, zero_gold_observations


@pytest.mark.parametrize('joint', [False, True])
@pytest.mark.parametrize('entropy_mode', ['legacy_path', 'exact_joint'])
@pytest.mark.parametrize('bf16', [False, True])
@pytest.mark.parametrize('gold', [0, 10])
def test_actor_outputs_and_rng_are_exact(joint, entropy_mode, bf16, gold):
    env, primitive, head = policies()
    policy = head if joint else primitive
    policy.config.entropy_mode = entropy_mode
    obs = zero_gold_observations(env)
    if gold:
        env.core.agent.gold = env.core.opponent.gold = gold
        from sapai_gym1.rl.training import stack_observations
        obs = stack_observations(env.observations(), 'cpu')
    if joint:
        with torch.no_grad():
            policy.type_head.bias[0] = 30
            policy.end_freeze_head.scorer.bias.fill_(1.)
    memory = torch.randn(2, policy.config.memory_dim)
    for deterministic in (False, True):
        with torch.no_grad(), torch.autocast('cpu', dtype=torch.bfloat16, enabled=bf16):
            torch.manual_seed(783)
            full = policy(obs, memory, deterministic=deterministic, allow_end_freeze=joint)
            full_rng = torch.get_rng_state().clone()
            torch.manual_seed(783)
            lean = policy.act(obs, memory, deterministic=deterministic)
            lean_rng = torch.get_rng_state().clone()
        assert set(vars(lean)) == {'actions', 'log_prob', 'memory', 'end_freeze_mask'}
        for key in vars(lean):
            actual, expected = getattr(lean, key), getattr(full, key)
            if expected is None:
                assert actual is None
            else:
                torch.testing.assert_close(actual, expected, rtol=0, atol=0)
        assert torch.equal(full_rng, lean_rng)


@pytest.mark.parametrize('joint', [False, True])
def test_actor_skips_unused_heads_and_every_entropy_call_but_retains_history(joint, monkeypatch):
    env, primitive, head = policies()
    policy = head if joint else primitive
    policy.config.entropy_mode = 'exact_joint'
    obs = zero_gold_observations(env)
    def forbidden(*args, **kwargs):
        raise AssertionError('Actor computed an unused output')
    for name in ('predict_opponent_board', 'exact_primitive_entropy'):
        monkeypatch.setattr(policy, name, forbidden)
    for module in (policy.value_head, policy.battle_head, policy.board_strength_head):
        monkeypatch.setattr(module, 'forward', forbidden)
    monkeypatch.setattr(torch.distributions.Categorical, 'entropy', forbidden)
    calls = []
    hook = policy.opponent_history_transformer.register_forward_hook(lambda *args: calls.append(1))
    with torch.no_grad():
        result = policy.act(obs)
    hook.remove()
    assert calls == [1]
    assert torch.isfinite(result.log_prob).all()


@pytest.mark.parametrize('joint', [False, True])
def test_value_only_matches_full_value_and_requires_no_decoder(joint, monkeypatch):
    env, primitive, head = policies()
    policy = head if joint else primitive
    obs = zero_gold_observations(env)
    memory = torch.randn(2, policy.config.memory_dim)
    with torch.no_grad():
        expected = policy(obs, memory, deterministic=True, allow_end_freeze=joint).value
    def forbidden(*args, **kwargs):
        raise AssertionError('Value-only decoded an action or forecast')
    monkeypatch.setattr(policy.type_head, 'forward', forbidden)
    monkeypatch.setattr(policy, 'predict_opponent_board', forbidden)
    with torch.no_grad():
        actual = policy.value_only(obs, memory)
    assert not actual.requires_grad
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)


@pytest.mark.parametrize('joint', [False, True])
@pytest.mark.parametrize('entropy_mode', ['legacy_path', 'exact_joint'])
def test_lean_bootstrap_preserves_complete_optimizer_step(joint, entropy_mode):
    _, primitive, head = policies()
    policy = head if joint else primitive
    policy.config.entropy_mode = entropy_mode
    data = BatchedSelfPlayCollector(policy, 2, 'cpu', seed=414, discount_mode='per_turn').collect(18)
    # Both last-step types, unequal lengths and multiple length buckets.
    data[0].terminal = True
    data[0].transitions[-1].discount = 0.
    old, new = copy.deepcopy(policy), copy.deepcopy(policy)
    old_opt = torch.optim.AdamW(old.parameters(), lr=1e-4)
    new_opt = torch.optim.AdamW(new.parameters(), lr=1e-4)
    for repeat in range(2):
        torch.manual_seed(500 + repeat)
        baseline = learner_update(old, old_opt, data, 'cpu', use_bfloat16=False)
        torch.manual_seed(500 + repeat)
        lean = learner_update(new, new_opt, data, 'cpu', use_bfloat16=False, lean_bootstrap=True)
        assert asdict(baseline) == asdict(lean)
        for left, right in zip(old.parameters(), new.parameters()):
            torch.testing.assert_close(left, right, rtol=0, atol=0)
            if left.grad is not None:
                torch.testing.assert_close(left.grad, right.grad, rtol=0, atol=0)
            else:
                assert right.grad is None
        for a, b in zip(old_opt.state.values(), new_opt.state.values()):
            for key in a:
                torch.testing.assert_close(a[key], b[key], rtol=0, atol=0)
                assert a[key].data_ptr() != b[key].data_ptr()


def test_server_mixed_residents_preserves_outputs_and_counts_cycles_not_groups():
    env, primitive, joint = policies()
    zero_gold_observations(env)
    observations = env.observations()
    context = mp.get_context('spawn')
    buffers = []
    batchers = []
    for lean in (False, True):
        shared = SharedActorBuffers.create(observations[0], joint.config.memory_dim, context)
        shared.active.fill_(1)
        shared.policy_ids[:] = torch.tensor([0, 1])
        for key in shared.observations:
            shared.observations[key].copy_(torch.as_tensor(np.stack([obs[key] for obs in observations])))
        batcher = DynamicInferenceBatcher(joint, [shared], 'cpu', threading.RLock(), context.Event(),
                                          resident_policies={1: primitive}, use_bfloat16=False,
                                          lean_policy_execution=lean)
        torch.manual_seed(118)
        batcher._serve([0])
        buffers.append(shared)
        batchers.append(batcher)
    for key in ('actions', 'log_probs', 'next_memories', 'end_freeze_masks', 'policy_versions'):
        assert torch.equal(getattr(buffers[0], key), getattr(buffers[1], key))
    for batcher in batchers:
        assert batcher.service_cycles == 1
        assert batcher.inference_batches == 2
        assert batcher.inference_items == 2
        assert 0 < batcher.policy_group_seconds <= batcher.inference_seconds


def test_multiprocess_lean_collection_and_learning_with_mixed_schemas():
    env, primitive, joint = policies()
    runtime = DistributedSelfPlayRuntime(joint, 2, env.observations()[0], 'cpu', seed=147,
        queue_size=16, games_per_actor=1, historical_policy=primitive,
        use_historical_opponent=True, learn_from_resident_sides=False,
        discount_mode='per_turn', lean_policy_execution=True)
    try:
        runtime.start()
        data = runtime.collect_segments(4, timeout=40)
        assert all(t.end_freeze_mask is not None for s in data for t in s.transitions)
        metrics = learner_update(joint, torch.optim.AdamW(joint.parameters(), lr=1e-4), data,
                                 'cpu', use_bfloat16=False, lean_bootstrap=True)
        assert np.isfinite(metrics.total_loss)
        runtime.publish_main_policy()
        assert runtime.batcher.policy_version == 1
    finally:
        runtime.stop()


def test_retained_full_actor_does_not_inherit_exact_entropy_work(monkeypatch):
    env, _, policy = policies()
    policy.config.entropy_mode = 'exact_joint'
    def forbidden(*args, **kwargs):
        raise AssertionError('Actor should not compute the learner exact entropy')
    monkeypatch.setattr(policy, 'exact_primitive_entropy', forbidden)
    collector=BatchedSelfPlayCollector(policy,1,'cpu',seed=939,lean_policy_execution=False)
    assert collector.collect(2)
    assert policy.config.entropy_mode == 'exact_joint'
