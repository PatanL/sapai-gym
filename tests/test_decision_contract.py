import copy
import json
from dataclasses import replace
import math

import pytest
import torch

from sapai_gym1.rl.action_schema import ActionType, AutoregressiveAction, NO_SOURCE, NO_TARGET
from sapai_gym1.rl.decision_api import decode_decisions, propose_decisions
from sapai_gym1.rl.joint_action import JointAction, step_decisions
from test_end_freeze_policy import policies, zero_gold_observations


def test_schema_preserves_none_versus_zero_and_rejects_invalid_fields():
    end = AutoregressiveAction(ActionType.END_TURN)
    legacy, joint = JointAction(end), JointAction(end, 0)
    assert legacy != joint and legacy.schema != joint.schema
    for decision in (legacy, joint, JointAction(end, 1023)):
        assert JointAction.from_dict(json.loads(json.dumps(decision.to_dict()))) == decision
    for mask in (True, -1, 1024, 0.5):
        with pytest.raises(ValueError):
            JointAction(end, mask)
    with pytest.raises(ValueError):
        JointAction(AutoregressiveAction(ActionType.ROLL), 1)
    with pytest.raises(ValueError):
        JointAction(AutoregressiveAction(ActionType.END_TURN, 0))
    invalid = joint.to_dict()
    invalid['end_freeze_mask'] = None
    with pytest.raises(ValueError):
        JointAction.from_dict(invalid)


@pytest.mark.parametrize('enabled', [False, True])
def test_encode_once_rescore_and_propose_do_not_advance_memory(enabled):
    env, primitive, joint = policies()
    policy = joint if enabled else primitive
    obs = {k: v[:1] for k, v in zero_gold_observations(env).items()}
    memory = torch.randn(1, policy.config.memory_dim)
    before = memory.clone()
    calls = []
    hook = policy.memory_cell.register_forward_hook(lambda *args: calls.append(1))
    with torch.no_grad():
        features = policy.encode_decision(obs, memory)
        decisions, output = decode_decisions(policy, features, deterministic=True)
        _, replay = decode_decisions(policy, features, decisions=decisions)
        proposed = propose_decisions(policy, features, generator=torch.Generator().manual_seed(31))
    hook.remove()
    assert len(calls) == 1
    torch.testing.assert_close(memory, before, rtol=0, atol=0)
    torch.testing.assert_close(replay.log_prob, output.log_prob, rtol=0, atol=0)
    torch.testing.assert_close(replay.memory, output.memory, rtol=0, atol=0)
    for action, log_p in zip(proposed.decisions, proposed.log_probabilities):
        _, scored = decode_decisions(policy, features, decisions=(action,))
        assert float(scored.log_prob[0]) == pytest.approx(log_p, abs=2e-6)
        action.expand(env.core, 0)
    mass = sum(math.exp(p) for p in proposed.log_probabilities)
    assert mass + proposed.omitted_probability_mass == pytest.approx(1., abs=1e-6)
    ends = [d for d in proposed.decisions if d.primitive.action_type == ActionType.END_TURN]
    assert len(ends) <= 8
    assert any(d.end_freeze_mask == (0 if enabled else None) for d in ends)
    if enabled:
        with pytest.raises(ValueError, match='schema'):
            decode_decisions(policy, features, decisions=(JointAction(decisions[0].primitive),))


@pytest.mark.parametrize('seat', [0, 1])
def test_both_seats_expansion_and_battle_rng_match(seat):
    env, _, _ = policies()
    zero_gold_observations(env)
    other = copy.deepcopy(env)
    decisions = [JointAction(AutoregressiveAction(ActionType.END_TURN), 0) for _ in range(2)]
    decisions[seat] = replace(decisions[seat], end_freeze_mask=1)
    plans = [d.expand(other.core, s) for s, d in enumerate(decisions)]
    from sapai_gym1.rl.action_schema import FlatActionCodec
    for s, plan in enumerate(plans):
        for action in plan[:-1]:
            other.core.resolve_action(s, FlatActionCodec.encode(action))
    expected = other.step(*(p[-1] for p in plans))
    actual = step_decisions(env, decisions)
    assert actual.result.rewards == expected.rewards
    for left, right in zip(env.observations(), other.observations()):
        for key in left:
            assert (left[key] == right[key]).all()
    assert env.core.agent.shop.rs.get_state()[2:] == other.core.agent.shop.rs.get_state()[2:]
    assert (env.core.agent.shop.rs.get_state()[1] == other.core.agent.shop.rs.get_state()[1]).all()


def test_invalid_second_seat_is_atomic():
    env, _, _ = policies()
    zero_gold_observations(env)
    before = env.observations()
    with pytest.raises(ValueError):
        step_decisions(env, (JointAction(AutoregressiveAction(0), 1), JointAction(AutoregressiveAction(0), 512)))
    for a, b in zip(before, env.observations()):
        for key in a:
            assert (a[key] == b[key]).all()


@pytest.mark.parametrize('seat', [0, 1])
def test_public_joint_expansion_matches_primitive_rollout_without_hidden_access(seat):
    import numpy as np
    from sapai import Team
    from sapai_gym1.rl.rollout_search import PublicSearchState
    env, _, _ = policies()
    zero_gold_observations(env)
    public = PublicSearchState.from_core(env.core, seat, opening_prior=((Team(['pet-fish']),),))
    manual = public.clone()
    decision = JointAction(AutoregressiveAction(ActionType.END_TURN), 1)
    rng_a, rng_b = np.random.default_rng(491), np.random.default_rng(491)
    expected = None
    for primitive in decision.expand(manual.core, 0):
        expected = manual.step(primitive, rng_a)
    actual = public.step(decision, rng_b)
    assert actual == expected
    assert rng_a.bit_generator.state == rng_b.bit_generator.state
    for key, value in public.observation().items():
        np.testing.assert_array_equal(value, manual.observation()[key])
    # Construction supplied only an explicit public opening prior. The real
    # opposing shop was not copied into either public state.
    assert len(public.core.opponent.shop) == 0


def entropy_fixture(policy, observation):
    observation = {k: v[:1].clone() for k, v in observation.items()}
    for key in ('type_mask', 'source_mask', 'target_mask'):
        observation[key].zero_()
    observation['type_mask'][0, [ActionType.END_TURN, ActionType.FREEZE]] = True
    observation['source_mask'][0, ActionType.END_TURN, NO_SOURCE] = True
    observation['target_mask'][0, ActionType.END_TURN, NO_SOURCE, NO_TARGET] = True
    observation['source_mask'][0, ActionType.FREEZE, :3] = True
    observation['target_mask'][0, ActionType.FREEZE, :3, NO_TARGET] = True
    with torch.no_grad():
        for module in (policy.type_head, policy.source_query, policy.target_query):
            module.weight.zero_()
            module.bias.zero_()
    return observation


def test_exact_entropy_reproduces_analytical_value_and_missing_upstream_gradient():
    env, policy, _ = policies()
    obs = entropy_fixture(policy, zero_gold_observations(env))
    end = torch.tensor([[0, NO_SOURCE, NO_TARGET]])
    legacy = policy(obs, actions=end)
    legacy.entropy.sum().backward()
    assert legacy.entropy.item() == pytest.approx(math.log(2), abs=1e-6)
    assert policy.type_head.bias.grad.abs().max() < 1e-6
    policy.zero_grad()
    policy.config.entropy_mode = 'exact_joint'
    exact = policy(obs, actions=end)
    exact.entropy.sum().backward()
    assert exact.entropy.item() == pytest.approx(math.log(2) + .5 * math.log(3), abs=1e-6)
    assert policy.type_head.bias.grad[ActionType.END_TURN].item() == pytest.approx(-math.log(3)/4, abs=1e-6)
    assert policy.type_head.bias.grad[ActionType.FREEZE].item() == pytest.approx(math.log(3)/4, abs=1e-6)
    torch.testing.assert_close(exact.log_prob, legacy.log_prob, rtol=0, atol=0)
    torch.testing.assert_close(exact.value, legacy.value, rtol=0, atol=0)
    torch.testing.assert_close(exact.memory, legacy.memory, rtol=0, atol=0)


@pytest.mark.parametrize('enabled', [False, True])
@pytest.mark.parametrize('bf16', [False, True])
def test_exact_entropy_sequence_finite_gradients_and_action_independence(enabled, bf16):
    env, primitive, joint = policies()
    policy = joint if enabled else primitive
    obs = zero_gold_observations(env)
    policy.config.entropy_mode = 'exact_joint'
    with torch.autocast('cpu', dtype=torch.bfloat16, enabled=bf16):
        a = policy(obs, allow_end_freeze=enabled, deterministic=True)
        b = policy(obs, allow_end_freeze=enabled, deterministic=False)
        torch.testing.assert_close(a.entropy, b.entropy)
        seq = policy.evaluate_sequence({k: v[None] for k, v in obs.items()}, policy.initial_memory(2),
                                       a.actions[None], end_freeze_masks=a.end_freeze_mask[None] if enabled else None)
        torch.testing.assert_close(seq.entropy[0], a.entropy, atol=.01 if bf16 else 1e-6, rtol=.01 if bf16 else 1e-6)
        loss = seq.entropy.sum()
    loss.backward()
    assert torch.isfinite(loss)
    assert all(p.grad is None or torch.isfinite(p.grad).all() for p in policy.parameters())


def test_singleton_choice_has_zero_entropy_and_no_freeze_entropy_when_inactive():
    env, _, policy = policies()
    obs = zero_gold_observations(env)
    for key in ('type_mask', 'source_mask', 'target_mask'):
        obs[key].zero_()
    obs['type_mask'][:, 0] = True
    obs['source_mask'][:, 0, NO_SOURCE] = True
    obs['target_mask'][:, 0, NO_SOURCE, NO_TARGET] = True
    policy.config.entropy_mode = 'exact_joint'
    result = policy(obs, allow_end_freeze=True)
    assert torch.count_nonzero(result.entropy) == 0
    result.entropy.sum().backward()
    assert all(p.grad is None or torch.isfinite(p.grad).all() for p in policy.parameters())


def test_entropy_cli_mode_is_persisted_and_cannot_relabel_same_campaign(tmp_path):
    from dataclasses import asdict
    from pathlib import Path
    import subprocess
    import sys
    from sapai.turtle_pack import SIMULATOR_REVISION
    _, policy, _ = policies()
    parent = tmp_path / 'parent.pt'
    torch.save({'model': policy.state_dict(), 'config': asdict(policy.config),
                'simulator_revision': SIMULATOR_REVISION}, parent)
    run = tmp_path / 'exact'
    script = Path(__file__).resolve().parents[1] / 'scripts/train_entity_selfplay.py'
    common = [sys.executable, str(script), '--smoke', '--device', 'cpu',
              '--checkpoint-dir', str(run), '--disable-expert-anchor']
    result = subprocess.run(common + ['--updates', '1', '--init-checkpoint', str(parent),
                                     '--entropy-mode', 'exact_joint'], capture_output=True, text=True, timeout=45)
    assert result.returncode == 0, result.stdout + result.stderr
    first = run / 'update-000001.pt'
    saved = torch.load(first, weights_only=False)
    assert saved['config']['entropy_mode'] == saved['training_semantics']['entropy_mode'] == 'exact_joint'
    result = subprocess.run(common + ['--updates', '2', '--resume-checkpoint', str(first)],
                            capture_output=True, text=True, timeout=45)
    assert result.returncode == 0, result.stdout + result.stderr
    latest = run / 'update-000002.pt'
    saved = torch.load(latest, weights_only=False)
    assert saved['config']['entropy_mode'] == saved['training_semantics']['entropy_mode'] == 'exact_joint'
    result = subprocess.run(common + ['--updates', '3', '--resume-checkpoint', str(latest),
                                     '--entropy-mode', 'legacy_path'], capture_output=True, text=True, timeout=45)
    assert result.returncode != 0 and 'separate checkpoint directory' in result.stderr
    assert not (run / 'update-000003.pt').exists()
