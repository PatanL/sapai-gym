import copy

import pytest
import torch

from sapai_gym1.rl.opponent_distribution import AutoregressiveOpponentBoard, OpponentDistributionConfig
from sapai_gym1.rl.opponent_context_data import PublicOpponentContext
from sapai_gym1.rl.opponent_policy_adapter import CausalOpponentPolicy
from test_end_freeze_policy import policies, zero_gold_observations


def fixture():
    env, _, parent = policies(); observation = {k: v[:1] for k, v in zero_gold_observations(env).items()}
    forecaster = AutoregressiveOpponentBoard(OpponentDistributionConfig(('none', 'pet-fish'), ('none',), width=32))
    wrapper = CausalOpponentPolicy(copy.deepcopy(parent), forecaster)
    context = [PublicOpponentContext.capture(env.core, 0).features()]
    return parent, wrapper, observation, context


@pytest.mark.parametrize('stage', ['forecast', 'policy_control', 'policy_residual'])
def test_initial_logit_value_memory_and_rng_parity(stage):
    parent, wrapper, obs, context = fixture()
    parent.eval(); wrapper.eval(); wrapper.set_stage(stage)
    memory = parent.initial_memory(1)
    torch.manual_seed(611); rng = torch.get_rng_state()
    # Acting parity must compare the same inference execution mode. Stage 1
    # freezes the parent parameters; comparing that to a grad-enabled parent
    # selects different PyTorch attention kernels despite identical weights.
    with torch.inference_mode():
        before = parent(obs, memory, allow_end_freeze=True)
        after_rng = torch.get_rng_state(); torch.set_rng_state(rng)
        after = wrapper(obs, memory, causal_context=context, allow_end_freeze=True)
    assert torch.equal(after_rng, torch.get_rng_state())
    for key, value in vars(before).items():
        if torch.is_tensor(value):
            torch.testing.assert_close(value, getattr(after, key), atol=0, rtol=0)
    assert all(torch.equal(p, wrapper.base_policy.state_dict()[k]) for k, p in parent.state_dict().items())


def test_forecast_only_update_cannot_change_policy_or_projection():
    parent, wrapper, obs, context = fixture()
    saved = {k: v.detach().clone() for k, v in wrapper.base_policy.state_dict().items()}
    optimizer = torch.optim.Adam((p for p in wrapper.parameters() if p.requires_grad), lr=.01)
    optimizer.zero_grad(); wrapper.forecast_loss(context, [[None]*5]).backward(); optimizer.step()
    assert all(torch.equal(p, wrapper.base_policy.state_dict()[k]) for k, p in saved.items())
    assert wrapper.projection.weight.count_nonzero() == 0 and wrapper.projection.bias.count_nonzero() == 0


def test_sequence_residual_zero_parity_and_projection_gradient():
    parent, wrapper, obs, context = fixture()
    wrapper.set_stage('policy_residual')
    with torch.no_grad():
        action = parent.act(obs, deterministic=True)
    sequence = {k: v[None].repeat(2, *([1]*v.ndim)) for k, v in obs.items()}
    actions = action.actions[None].repeat(2, 1, 1)
    masks = action.end_freeze_mask[None].repeat(2, 1)
    before = parent.evaluate_sequence(sequence, parent.initial_memory(1), actions, end_freeze_masks=masks)
    after = wrapper.evaluate_sequence(sequence, wrapper.initial_memory(1), actions, end_freeze_masks=masks, causal_context=context*2)
    for key, value in vars(before).items():
        if torch.is_tensor(value):
            torch.testing.assert_close(value, getattr(after, key), atol=0, rtol=0)
    (-after.log_prob.mean()+after.value.square().mean()).backward()
    assert wrapper.projection.weight.grad.abs().sum() > 0
    assert all(p.grad is None for p in wrapper.forecast_model.parameters())


def test_enabled_residual_never_silently_uses_private_or_missing_context():
    _, wrapper, obs, context = fixture(); wrapper.set_stage('policy_residual')
    with pytest.raises(ValueError, match='explicit causal'):
        wrapper.act(obs)
    context[0]['opponent_shop'] = []
    with pytest.raises(ValueError, match='declared causal'):
        wrapper.act(obs, causal_context=context)
