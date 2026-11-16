import multiprocessing as mp
import threading

import numpy as np
import pytest
import torch

from sapai_gym1.rl.distributed import SharedActorBuffers, SharedTrajectoryRing, DynamicInferenceBatcher, DistributedSelfPlayRuntime
from sapai_gym1.rl.training import Transition, TurnSegment, learner_update
from test_end_freeze_policy import policies


def test_shared_ring_preserves_legacy_none_zero_mask_and_full_mask():
    context = mp.get_context('spawn')
    obs = {'x': np.array([1.], dtype=np.float32)}
    ring = SharedTrajectoryRing.create(obs, 2, context, capacity=1, max_steps=3)
    segment = TurnSegment(
        [Transition(obs, np.array([0, 10, 5]), -.5, end_freeze_mask=mask)
         for mask in (None, 0, 1023)], torch.zeros(2), obs, False,
    )
    try:
        assert ring.write(segment, context.Event())
        restored = ring.read(2)
        assert [item.end_freeze_mask for item in restored.transitions] == [None, 0, 1023]
        segment.transitions[0].end_freeze_mask = -1
        with pytest.raises(ValueError, match='Invalid'):
            ring.write(segment, context.Event())
    finally:
        ring.close()


def test_inference_batches_mixed_schemas_and_overwrites_stale_mask():
    env, primitive, joint = policies()
    env.core.agent.gold = env.core.opponent.gold = 0
    observations = env.observations()
    with torch.no_grad():
        joint.type_head.bias[0] = 20
        joint.end_freeze_head.scorer.bias.fill_(10)
    context = mp.get_context('spawn')
    shared = SharedActorBuffers.create(observations[0], 48, context)
    shared.active.fill_(1)
    shared.policy_ids[:] = torch.tensor([0, 1])
    for key in shared.observations:
        shared.observations[key].copy_(torch.as_tensor(np.stack([obs[key] for obs in observations])))
    batcher = DynamicInferenceBatcher(joint, [shared], 'cpu', threading.RLock(), context.Event(),
                                      resident_policies={1: primitive}, use_bfloat16=False)
    batcher._serve([0])
    assert int(shared.end_freeze_masks[0]) > 0
    assert int(shared.end_freeze_masks[1]) == -1
    shared.policy_ids.fill_(1)
    batcher._serve([0])
    assert shared.end_freeze_masks.tolist() == [-1, -1]


def test_multiprocess_collect_and_learn_joint_actions_against_primitive_resident():
    env, primitive, joint = policies()
    with torch.no_grad():
        joint.type_head.bias[0] = 20
        joint.type_head.bias[1] = 10
        joint.end_freeze_head.scorer.bias.zero_()
    runtime = DistributedSelfPlayRuntime(
        joint, 2, env.observations()[0], 'cpu', seed=193,
        queue_size=16, games_per_actor=1, historical_policy=primitive,
        use_historical_opponent=True, learn_from_resident_sides=False,
        discount_mode='per_turn',
    )
    try:
        runtime.start()
        segments = runtime.collect_segments(4, timeout=30)
        assert all(t.end_freeze_mask is not None for s in segments for t in s.transitions)
        assert any(t.end_freeze_mask for s in segments for t in s.transitions)
        for segment in segments:
            assert all(t.discount == 1.0 for t in segment.transitions[:-1])
            assert segment.transitions[-1].discount == pytest.approx(.99)
            obs = {key: torch.as_tensor(np.stack([t.observation[key] for t in segment.transitions]))[:, None]
                   for key in segment.transitions[0].observation}
            actions = torch.tensor(np.stack([t.action for t in segment.transitions]))[:, None]
            masks = torch.tensor([t.end_freeze_mask for t in segment.transitions])[:, None]
            out = joint.evaluate_sequence(obs, segment.initial_memory[None], actions, end_freeze_masks=masks)
            torch.testing.assert_close(out.log_prob[:, 0], torch.tensor([t.behavior_log_prob for t in segment.transitions]),
                                       atol=2e-5, rtol=2e-5)
        metrics = learner_update(joint, torch.optim.AdamW(joint.parameters(), lr=1e-4), segments,
                                 'cpu', use_bfloat16=False)
        assert np.isfinite(metrics.total_loss)
        runtime.publish_main_policy()
        assert runtime.batcher.policy_version == 1
    finally:
        runtime.stop()
