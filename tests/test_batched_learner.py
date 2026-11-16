import copy
from dataclasses import asdict

import pytest
import torch

from sapai_gym1.rl.vtrace import impala_actor_critic_loss, masked_impala_actor_critic_loss
from sapai_gym1.rl.training import BatchedSelfPlayCollector, learner_update
from test_end_freeze_policy import policies
from test_dense_trajectory import convert_through_ring


@pytest.mark.parametrize('dtype', [torch.float32, torch.float64])
@pytest.mark.parametrize('seed', range(36))
def test_masked_batched_vtrace_matches_per_segment_targets_losses_and_gradients(dtype, seed):
    torch.manual_seed(seed + 41000)
    t, b = (1, 7, 24, 64)[seed % 4], (1, 3, 9)[seed % 3]
    lengths = torch.randint(1, t+1, (b,))
    valid = torch.arange(t)[:, None] < lengths[None]
    behavior, target, entropy, reward, value = [torch.randn(t, b, dtype=dtype) for _ in range(5)]
    discount = torch.ones(t, b, dtype=dtype)
    discount[lengths - 1, torch.arange(b)] = torch.where(torch.arange(b) % 2 == 0, .99, 0.).to(dtype)
    bootstrap = torch.randn(b, dtype=dtype)
    # Invalid rows must not even enter the recursion or loss arithmetic.
    for tensor in (behavior, target, entropy, reward, value, discount):
        tensor[~valid] = torch.nan
    a = [tensor.clone().requires_grad_() for tensor in (target, entropy, value)]
    c = [tensor.clone().requires_grad_() for tensor in (target, entropy, value)]
    old = [impala_actor_critic_loss(behavior[:n, i:i+1], a[0][:n, i:i+1], a[1][:n, i:i+1],
                discount[:n, i:i+1], reward[:n, i:i+1], a[2][:n, i:i+1], bootstrap[i:i+1])
           for i, n in enumerate(lengths.tolist())]
    new = masked_impala_actor_critic_loss(behavior, c[0], c[1], discount, reward, c[2], bootstrap, lengths)
    atol, rtol = (3e-6, 3e-6) if dtype == torch.float32 else (1e-12, 1e-12)
    total = sum(lengths.tolist())
    for i, n in enumerate(lengths.tolist()):
        torch.testing.assert_close(new.vtrace.value_targets[:n, i], old[i].vtrace.value_targets[:, 0], atol=atol, rtol=rtol)
        torch.testing.assert_close(new.vtrace.policy_advantages[:n, i], old[i].vtrace.policy_advantages[:, 0], atol=atol, rtol=rtol)
    old_total = sum(loss.total * (int(n)/total) for loss, n in zip(old, lengths))
    for name in ('total', 'policy', 'value', 'entropy'):
        expected = sum(getattr(loss, name) * (int(n)/total) for loss, n in zip(old, lengths))
        torch.testing.assert_close(getattr(new, name), expected, atol=atol, rtol=rtol)
    old_total.backward()
    new.total.backward()
    assert not new.vtrace.value_targets.requires_grad
    for x, y in zip(a, c):
        torch.testing.assert_close(x.grad, y.grad, atol=atol, rtol=rtol)
        assert torch.isfinite(y.grad).all()
        assert torch.count_nonzero(y.grad[~valid]) == 0


@pytest.mark.parametrize('joint', [False, True])
@pytest.mark.parametrize('entropy_mode', ['legacy_path', 'exact_joint'])
@pytest.mark.parametrize('dense', [False, True])
def test_batched_full_learner_matches_gradients_parameters_and_adam(joint, entropy_mode, dense):
    _, primitive, head = policies()
    policy = head if joint else primitive
    policy.config.entropy_mode = entropy_mode
    data = BatchedSelfPlayCollector(policy, 2, 'cpu', seed=380, discount_mode='per_turn').collect(18)
    data[0].terminal = True
    data[0].transitions[-1].discount = 0.
    data[1].battle_outcome = None
    candidate_data = convert_through_ring(data) if dense else data
    old, new = copy.deepcopy(policy), copy.deepcopy(policy)
    a, b = torch.optim.AdamW(old.parameters(), lr=2e-6), torch.optim.AdamW(new.parameters(), lr=2e-6)
    for iteration in range(2):
        torch.manual_seed(803 + iteration)
        baseline = learner_update(old, a, data, 'cpu', use_bfloat16=False)
        torch.manual_seed(803 + iteration)
        result = learner_update(new, b, candidate_data, 'cpu', use_bfloat16=False,
                                 lean_bootstrap=True, batched_learner=True)
        for key, value in asdict(baseline).items():
            assert asdict(result)[key] == pytest.approx(value, abs=3e-6, rel=3e-5), key
        for (name, x), (_, y) in zip(old.named_parameters(), new.named_parameters()):
            torch.testing.assert_close(x, y, atol=2e-7, rtol=2e-6, msg=name)
            if x.grad is not None:
                torch.testing.assert_close(x.grad, y.grad, atol=3e-6, rtol=3e-5, msg=name)
            else:
                assert y.grad is None
        for x, y in zip(a.state.values(), b.state.values()):
            for key in x:
                torch.testing.assert_close(x[key], y[key], atol=3e-7, rtol=3e-5, msg=key)
                assert x[key].data_ptr() != y[key].data_ptr()
