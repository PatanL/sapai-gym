import math
import pickle

import numpy as np
import pytest
import torch

from sapai import Shop, Pet, Food, Team
from sapai_gym1.rl import ParallelSelfPlayEnv
from sapai_gym1.rl.action_schema import ActionType, AutoregressiveAction, FlatActionCodec
from sapai_gym1.rl.end_freeze import EndFreezeHead, expand_end_freeze, step_with_end_freeze


def observation(count=0):
    obs = {'numeric': torch.zeros(1,21,16), 'type_mask': torch.zeros(1,9,dtype=torch.bool),
           'target_mask': torch.zeros(1,9,11,6,dtype=torch.bool)}
    obs['type_mask'][0,0] = True
    obs['target_mask'][0,6,0,5] = True
    obs['target_mask'][0,7,2,5] = True
    obs['numeric'][0,0,5] = count/50
    return obs


def test_subset_distribution_matches_independent_bits_when_budget_is_free():
    head = EndFreezeHead(4)
    logits = torch.full((1,10), math.log(.1/.9), requires_grad=True)
    dist, allowed = head.distribution(logits, observation(), torch.tensor([0]))
    assert torch.where(allowed[0])[0].tolist() == [0,1,4,5]
    torch.testing.assert_close(dist.probs[0,[0,1,4,5]], torch.tensor([.81,.09,.09,.01]))
    loss = -dist.log_prob(torch.tensor([5])).sum()
    loss.backward()
    assert torch.isfinite(logits.grad).all()
    assert logits.grad[0,0] < 0 and logits.grad[0,2] < 0
    assert torch.count_nonzero(logits.grad[0,[1,3,4,5,6,7,8,9]]) == 0


def test_budget_conditioning_and_non_end_actions_have_exact_support():
    head = EndFreezeHead(4)
    logits = torch.zeros(1,10)
    dist, allowed = head.distribution(logits, observation(49), torch.tensor([0]))
    assert torch.where(allowed[0])[0].tolist() == [0,1,4]
    torch.testing.assert_close(dist.probs[0,[0,1,4]], torch.full((3,),1/3))
    for obs, kind in [(observation(50),0),(observation(),2)]:
        dist, allowed = head.distribution(logits,obs,torch.tensor([kind]))
        assert torch.where(allowed[0])[0].tolist() == [0]
        assert float(dist.entropy()) == 0.


def test_initial_greedy_head_preserves_flags_and_rejects_bad_recorded_masks():
    head = EndFreezeHead(4)
    context, tokens = torch.randn(1,4), torch.randn(1,10,4)
    out = head(context,tokens,observation(),torch.tensor([0]),deterministic=True)
    assert int(out.mask_ids[0]) == 0
    for mask in (torch.tensor([2]), torch.tensor([1.5]), torch.tensor([1024])):
        with pytest.raises(ValueError):
            head(context,tokens,observation(),torch.tensor([0]),mask_ids=mask)


def environment():
    env=ParallelSelfPlayEnv(); env.reset(seed=411)
    env.core.agent.shop=Shop([Pet('fish'),Pet('otter'),Pet('cricket')],turn=1,fixed_rules=True)
    env.core.agent.gold=0; env.core.agent._connect_objects()
    env.core.shop_agent=env.core.agent.shop
    env.core._agent_actions_cache=None
    env.observations()
    return env


def test_plan_validation_precedes_execution_and_respects_existing_escape_hatch():
    env=environment()
    core=env.core
    with pytest.raises(ValueError,match='empty'):
        expand_end_freeze(core,0,(1<<0)|(1<<9))
    assert all(not slot.frozen for slot in core.agent.shop)
    core.shop_action_count_agent=49
    with pytest.raises(ValueError,match='budget'):
        expand_end_freeze(core,0,3)
    plan=expand_end_freeze(core,0,1)
    assert len(plan)==2
    for action in plan:
        assert FlatActionCodec.encode(action) in core._avail_actions(0)
        core.resolve_action(0,FlatActionCodec.encode(action))
    assert core.agent.turn_ended and core.shop_action_count_agent==51
    assert core.agent.shop[0].frozen


def test_head_support_matches_real_legal_plans():
    env=environment()
    env.core.agent.shop[1].frozen=True
    env.core._freeze_toggled_agent.add(id(env.core.agent.shop[2].obj))
    env.core.shop_action_count_agent=49
    obs={key:torch.as_tensor(value).unsqueeze(0) for key,value in env.observations()[0].items()}
    _, allowed=EndFreezeHead(4).distribution(torch.zeros(1,10),obs,torch.tensor([0]))
    for mask_id in range(1024):
        try:
            expand_end_freeze(env.core,0,mask_id)
            legal=True
        except ValueError:
            legal=False
        assert bool(allowed[0,mask_id]) == legal


def assert_same_environment(left, right):
    for left_obs, right_obs in zip(left.observations(), right.observations()):
        for key in left_obs:
            np.testing.assert_array_equal(left_obs[key], right_obs[key], err_msg=key)
    for side in ('agent', 'opponent'):
        a, b = getattr(left.core, side), getattr(right.core, side)
        assert pickle.dumps(a.shop.state) == pickle.dumps(b.shop.state)
        assert pickle.dumps(a.team.state) == pickle.dumps(b.team.state)
    assert pickle.dumps(left.core.rs_random.get_state()) == pickle.dumps(right.core.rs_random.get_state())


@pytest.mark.parametrize('side', [0, 1])
def test_joint_execution_matches_manual_clicks_through_battle_and_next_roll(side):
    env = environment()
    for seat, player in enumerate((env.core.agent, env.core.opponent)):
        player.team = Team(['fish'])
        player.shop = Shop([Pet('fish'), Food('honey'), Pet('otter')], turn=1, fixed_rules=True,
                           seed_state=np.random.RandomState(941 + seat).get_state())
        player.shop.levelup()
        player.gold = 0
        player._connect_objects()
    env.core.shop_agent, env.core.shop_opponent = env.core.agent.shop, env.core.opponent.shop
    env.core.team_agent, env.core.team_opponent = env.core.agent.team, env.core.opponent.team
    player = env.core.agent if side == 0 else env.core.opponent
    group = next(slot.reward_group for slot in player.shop if slot.reward_group is not None)
    pair = [index for index, slot in enumerate(player.shop) if slot.reward_group == group]
    assert len(pair) == 2 and group is not None
    # Freeze BOTH discovered choices, freeze one food and unfreeze a normal pet.
    food = next(i for i, slot in enumerate(player.shop) if isinstance(slot.obj, Food))
    normal = next(i for i, slot in enumerate(player.shop) if slot.reward_group is None and isinstance(slot.obj, Pet))
    player.shop[normal].frozen = True
    mask = sum(1 << i for i in (*pair, food, normal))
    expected = env.clone()
    masks = [0, 0]; masks[side] = mask
    end = AutoregressiveAction(ActionType.END_TURN)
    plans = [expand_end_freeze(expected.core, i, value) for i, value in enumerate(masks)]
    for i, plan in enumerate(plans):
        for action in plan[:-1]:
            expected.core.resolve_action(i, FlatActionCodec.encode(action))
    manual_step = expected.step(end, end)
    macro_step = step_with_end_freeze(env, end, end, mask_ids=masks)
    assert macro_step.primitive_actions == tuple(plans)
    assert macro_step.result.battle_resolved and manual_step.battle_resolved
    assert macro_step.result.rewards == manual_step.rewards
    assert macro_step.result.battle_result == manual_step.battle_result
    assert_same_environment(env, expected)
    player = env.core.agent if side == 0 else env.core.opponent
    assert sum(slot.reward_group == group and slot.frozen for slot in player.shop) == 2
    roll = AutoregressiveAction(ActionType.ROLL)
    step_with_end_freeze(env, roll, roll)
    expected.step(roll, roll)
    assert_same_environment(env, expected)
    indices = [i for i, slot in enumerate(player.shop) if slot.reward_group == group]
    assert len(indices) == 2
    player.buy_pet(indices[0], 1)
    assert not any(slot.reward_group == group for slot in player.shop)


def test_invalid_second_side_plan_cannot_partially_apply_first_side():
    env = environment()
    env.core.opponent.gold = 0
    before = env.clone()
    end = AutoregressiveAction(ActionType.END_TURN)
    with pytest.raises(ValueError, match='empty'):
        step_with_end_freeze(env, end, end, mask_ids=(1, 1 << 9))
    assert_same_environment(env, before)


def test_ordinary_actions_and_already_ended_sides_need_zero_subset():
    env = environment()
    end = AutoregressiveAction(ActionType.END_TURN)
    roll = AutoregressiveAction(ActionType.ROLL)
    before = env.clone()
    with pytest.raises(ValueError, match='only valid with END'):
        step_with_end_freeze(env, end, roll, mask_ids=(0, 1))
    assert_same_environment(env, before)
    step_with_end_freeze(env, end, roll)
    assert env.active_players() == (False, True)
    with pytest.raises(ValueError, match='Ended players'):
        step_with_end_freeze(env, None, roll, mask_ids=(1, 0))
    assert step_with_end_freeze(env, None, roll).primitive_actions[0] == ()


def test_log_probability_and_entropy_are_recomputable_for_training():
    torch.manual_seed(91)
    head = EndFreezeHead(4)
    context, tokens = torch.randn(1, 4), torch.randn(1, 10, 4)
    obs = observation()
    types = torch.tensor([int(ActionType.END_TURN)])
    out = head(context, tokens, obs, types)
    replay = head(context, tokens, obs, types, mask_ids=out.mask_ids)
    torch.testing.assert_close(out.log_prob, replay.log_prob, rtol=0, atol=0)
    torch.testing.assert_close(out.entropy, replay.entropy, rtol=0, atol=0)
    # This sub-action likelihood MUST be added to the ordinary END log-p.
    base_log_p = torch.tensor([-0.15])
    ratio = ((base_log_p + replay.log_prob) - (base_log_p + out.log_prob).detach()).exp()
    torch.testing.assert_close(ratio, torch.ones_like(ratio), rtol=0, atol=0)
    (-replay.log_prob.mean()).backward()
    assert torch.isfinite(head.scorer.weight.grad).all()


def test_full_ten_slot_support_and_positive_gold_escape_hatch():
    obs = observation()
    obs['target_mask'][0, ActionType.FREEZE, :10, 5] = True
    head = EndFreezeHead(4)
    logits = torch.zeros(1, 10)
    dist, allowed = head.distribution(logits, obs, torch.tensor([0]))
    assert int(allowed.sum()) == 1024
    torch.testing.assert_close(dist.probs, torch.full((1, 1024), 1 / 1024))
    obs['numeric'][0, 0, 5] = 48 / 50
    _, allowed = head.distribution(logits, obs, torch.tensor([0]))
    assert int(allowed.sum()) == 1 + 10 + 45  # empty / singleton / two toggles
    obs['numeric'][0, 0, 11] = 0.1
    dist, allowed = head.distribution(logits, obs, torch.tensor([0]))
    assert torch.where(allowed[0])[0].tolist() == [0]
    assert float(dist.log_prob(torch.tensor([0]))) == 0


def test_cpu_autocast_preserves_legal_support_and_finite_gradients():
    head = EndFreezeHead(4)
    obs = observation(49)
    with torch.autocast(device_type='cpu', dtype=torch.bfloat16):
        output = head(torch.randn(1, 4), torch.randn(1, 10, 4), obs,
                      torch.tensor([0]), mask_ids=torch.tensor([1]))
    assert torch.isfinite(output.log_prob).all() and torch.isfinite(output.entropy).all()
    (-output.log_prob.mean()).backward()
    assert torch.isfinite(head.scorer.weight.grad).all()
