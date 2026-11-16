from dataclasses import asdict
from types import SimpleNamespace

import pytest
import torch

from sapai_gym1.deployment.policy_adapter import LivePolicyEngine, live_state_from_environment
from sapai_gym1.deployment.inference_backend import inference_device
from sapai_gym1.rl.action_schema import FlatActionCodec
from sapai_gym1.rl.joint_action import JOINT_SCHEMA
from test_end_freeze_policy import policies, zero_gold_observations


def make_engine(tmp_path, enabled=True):
    env, primitive, joint = policies()
    zero_gold_observations(env)
    policy = joint if enabled else primitive
    with torch.no_grad():
        policy.type_head.bias[0] = 40
        if enabled:
            policy.end_freeze_head.scorer.bias.fill_(3)
    path = tmp_path / 'weights.pt'
    torch.save({'config': asdict(policy.config), 'model': policy.state_dict(),
                'training_semantics': {'action_schema': JOINT_SCHEMA if enabled else 'primitive-autoregressive-v1'}}, path)
    return env.core, LivePolicyEngine(path, threads=1)


@pytest.mark.parametrize('enabled', [False, True])
def test_recommendation_retry_pure_commit_once_and_manual_plan(enabled, tmp_path):
    env, engine = make_engine(tmp_path, enabled)
    engine.reset_memory()
    initial = engine.snapshot_memory()
    a = engine.recommend_environment(env, match_id='test')
    b = engine.recommend_environment(env, match_id='test')
    assert a is b
    torch.testing.assert_close(engine.memory, initial, rtol=0, atol=0)
    assert len(a.primitive_plan) == (5 if enabled else 1)
    assert a.to_dict()['decision']['end_freeze_mask'] == (15 if enabled else None)
    engine.validate_recommendation(a, env)
    with pytest.raises(ValueError, match='confirmation'):
        engine.commit_recommendation(a)
    for primitive in a.primitive_plan:
        env.resolve_action(0, FlatActionCodec.encode(primitive))
    engine.commit_recommendation(a, confirmed=True)
    assert not torch.equal(initial, engine.memory)
    committed = engine.snapshot_memory()
    with pytest.raises(ValueError, match='committed'):
        engine.commit_recommendation(a, confirmed=True)
    torch.testing.assert_close(engine.memory, committed, rtol=0, atol=0)


def test_stale_state_reset_and_model_replacement_reject_tokens(tmp_path):
    env, engine = make_engine(tmp_path)
    a = engine.recommend_environment(env)
    env.agent.gold = 1
    with pytest.raises(ValueError, match='changed'):
        engine.validate_recommendation(a, env)
    env.agent.gold = 0
    b = engine.recommend_environment(env)
    assert a.decision_id != b.decision_id
    engine.reset_memory()
    with pytest.raises(ValueError):
        engine.commit_recommendation(b, confirmed=True)
    c = engine.recommend_environment(env)
    _, _, replacement = policies()
    engine.policy = replacement
    with pytest.raises(ValueError, match='replaced'):
        engine.validate_recommendation(c, env)
    engine.memory.fill_(100.)
    engine.recommend_environment(env)
    assert torch.count_nonzero(engine.memory) == 0


def test_speculative_joint_plan_never_mutates_memory_or_pending(tmp_path):
    env, engine = make_engine(tmp_path)
    engine.reset_memory()
    state = live_state_from_environment(env)
    before = engine.snapshot_memory()
    plan = engine.plan_until_barrier(state)
    assert len(plan) == 1
    assert len(plan[0].recommendation.primitive_plan) == 5
    assert engine._pending is None
    torch.testing.assert_close(engine.memory, before, rtol=0, atol=0)


def test_backend_is_explicit_and_unavailable_mps_does_not_silently_fallback(monkeypatch):
    assert str(inference_device('cpu')) == 'cpu'
    monkeypatch.setattr(torch.backends.mps, 'is_available', lambda: False)
    with pytest.raises(RuntimeError, match='unavailable'):
        inference_device('mps')
    with pytest.raises(ValueError):
        inference_device('magic')


def controller_fixture(tmp_path, *, armed=True, fail_freeze=False):
    from sapai_gym1.deployment.arena_controller import AutonomousArenaController
    env, engine = make_engine(tmp_path)
    controller = AutonomousArenaController.__new__(AutonomousArenaController)
    controller.env, controller.engine = env, engine
    controller.match_id, controller.armed, controller.actions = 'fixture', armed, 0
    controller._state = lambda *args, **kwargs: live_state_from_environment(env)
    def observed():
        return SimpleNamespace(
            confidence=1., to_dict=lambda: {},
            pets=[SimpleNamespace(frozen=False if fail_freeze else s.frozen) for s in env.agent.shop if s.slot_type in {'pet', 'levelup'}],
            foods=[SimpleNamespace(frozen=False if fail_freeze else s.frozen) for s in env.agent.shop if s.slot_type == 'food'],
        )
    controller._stable_shop_observation = lambda **kwargs: (None, 0, observed())
    controller._changed_shop_observation = lambda *args, **kwargs: (None, 1, observed())
    controller.reconcile_observed_stats = lambda *args: []
    controller.observer = SimpleNamespace(verify_identities=lambda *args: None)
    controller._publish = lambda **kwargs: kwargs
    controller.mapper = SimpleNamespace(gesture=lambda action, **kwargs: SimpleNamespace(action=action, to_dict=lambda: {}))
    clicks = []
    controller.dashboard = SimpleNamespace(execute_gesture=lambda gesture: clicks.append(gesture.action))
    return controller, clicks


def test_live_joint_controller_reobserves_toggles_and_commits_one_gru_step(tmp_path):
    controller, clicks = controller_fixture(tmp_path)
    calls = []
    hook = controller.engine.policy.memory_cell.register_forward_hook(lambda *args: calls.append(1))
    result = controller.step_shop()
    hook.remove()
    assert result['endTurn']
    assert len(clicks) == 5 and len(calls) == 1
    assert all(s.frozen for s in controller.env.agent.shop)
    assert torch.count_nonzero(controller.engine.memory) == 0
    controller._commit_end_if_observed('unknown')
    controller._commit_end_if_observed('shop')
    assert torch.count_nonzero(controller.engine.memory) == 0
    controller._commit_end_if_observed('battle')
    after = controller.engine.snapshot_memory()
    assert torch.count_nonzero(after) > 0
    controller._commit_end_if_observed('battle')
    torch.testing.assert_close(controller.engine.memory, after, rtol=0, atol=0)


def test_live_joint_controller_aborts_before_end_when_toggle_is_not_confirmed(tmp_path):
    controller, clicks = controller_fixture(tmp_path, fail_freeze=True)
    from sapai_gym1.deployment.live_perception import PerceptionError
    with pytest.raises(PerceptionError, match='visually confirmed'):
        controller.step_shop()
    assert len(clicks) == 1
    assert controller.engine.memory is None or torch.count_nonzero(controller.engine.memory) == 0
    assert controller.engine._pending is None


def test_disarmed_live_controller_retries_do_not_advance_memory(tmp_path):
    controller, clicks = controller_fixture(tmp_path, armed=False)
    first = controller.step_shop()
    second = controller.step_shop()
    assert not first['executed'] and not second['executed'] and not clicks
    assert first['recommendation']['decisionId'] == second['recommendation']['decisionId']
    assert torch.count_nonzero(controller.engine.memory) == 0
