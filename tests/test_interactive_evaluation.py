import numpy as np
import pytest

from sapai_gym1.rl import (
    EntityTransformerConfig,
    EntityTransformerPolicy,
    FlatActionCodec,
    InteractiveSelfPlayEvaluator,
    ParallelSelfPlayEnv,
)


@pytest.mark.parametrize('finish_match', [False, True])
def test_interactive_outer_evaluation_is_reproducible_and_nonmutating(finish_match):
    environment = ParallelSelfPlayEnv()
    original_observations = environment.reset(seed=771)
    if finish_match:
        environment.core.MAX_TURN = 2
    encoder = environment.encoder
    policy = EntityTransformerPolicy(
        EntityTransformerConfig(
            num_items=encoder.num_items,
            num_statuses=encoder.num_statuses,
            d_model=32,
            nhead=4,
            num_layers=1,
            dim_feedforward=64,
            memory_dim=24,
        )
    ).eval()
    legal = [
        FlatActionCodec.decode(action)
        for action in environment.core._avail_actions(0)
    ]
    actions = legal[:2]
    memories = [policy.initial_memory(1)[0] for _ in range(2)]
    evaluator = InteractiveSelfPlayEvaluator(
        policy,
        device="cpu",
        batch_size=16,
        horizon_battles=1,
    )
    first = evaluator.evaluate(
        environment, memories, 0, actions, seeds=(101, 202)
    )
    second = evaluator.evaluate(
        environment, memories, 0, actions, seeds=(101, 202)
    )
    assert first == second
    assert len(first) == 4
    assert [result.seed for result in first] == [101, 202, 101, 202]
    assert all(result.battles == 1 for result in first)
    assert all(result.terminated or result.truncated for result in first)
    if finish_match:
        assert all(result.terminated and result.match_score in (0, 0.5, 1) for result in first)
    else:
        assert all(result.truncated and result.match_score is None for result in first)
    for before, after in zip(
        original_observations, environment.observations()
    ):
        for key in before:
            np.testing.assert_array_equal(before[key], after[key])
    environment.core.agent.turn_ended = True
    with pytest.raises(ValueError, match='current nonterminal'):
        evaluator.evaluate(environment, memories, 0, actions, seeds=(101, 202))


@pytest.mark.parametrize('side', [0, 1])
def test_joint_interventions_and_continuations_preserve_original_state(side):
    from test_end_freeze_policy import policies, zero_gold_observations
    from sapai_gym1.rl.joint_action import JointAction
    from sapai_gym1.rl.action_schema import AutoregressiveAction, ActionType
    env, _, policy = policies()
    zero_gold_observations(env)
    before = env.observations()
    evaluator = InteractiveSelfPlayEvaluator(policy, horizon_battles=1, batch_size=8)
    actions = [JointAction(AutoregressiveAction(ActionType.END_TURN), mask) for mask in (0, 1)]
    memories = [policy.initial_memory(1)[0] for _ in range(2)]
    a = evaluator.evaluate(env, memories, side, actions, seeds=(312, 313))
    b = evaluator.evaluate(env, memories, side, actions, seeds=(312, 313))
    assert a == b and len(a) == 4
    assert all(result.battles == 1 for result in a)
    assert a[0].root_action.end_freeze_mask == 0
    assert a[2].root_action.end_freeze_mask == 1
    for left, right in zip(before, env.observations()):
        for key in left:
            np.testing.assert_array_equal(left[key], right[key])
    with pytest.raises(ValueError, match='explicit freeze masks'):
        evaluator.evaluate(env, memories, side, [actions[0].primitive], seeds=(312,))
