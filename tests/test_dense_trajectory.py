import copy
import importlib.util
import multiprocessing as mp
from pathlib import Path
from queue import Empty
import sys
import threading

import numpy as np
import pytest
import torch

from sapai_gym1.rl.dense_trajectory import OwnedTrajectoryBatch, DenseTurnSegment
from sapai_gym1.rl.distributed import SharedTrajectoryRing, DistributedSelfPlayRuntime
from sapai_gym1.rl.learner_batch import pack_learner_batch
from sapai_gym1.rl.training import Transition, TurnSegment, BatchedSelfPlayCollector, learner_update
from test_end_freeze_policy import policies


def tiny_segment(value=1, length=3, masks=(None, 0, 1023)):
    rows = [Transition({'x': np.array([value, t], dtype=np.float32)}, np.array([0, 10, 5]),
                       -.1 * t, .2 * t, .99 if t == length-1 else 1.,
                       behavior_version=(-1, 3, 5)[t % 3], end_freeze_mask=masks[t % len(masks)]) for t in range(length)]
    return TurnSegment(rows, torch.tensor([value, -value], dtype=torch.float32),
                       {'x': np.array([value, 99], dtype=np.float32)}, False, 2)


def ring_fixture(capacity=3):
    ctx = mp.get_context('spawn')
    return SharedTrajectoryRing.create({'x': np.zeros(2, dtype=np.float32)}, 2, ctx,
                                        capacity=capacity, max_steps=8), ctx.Event()


def test_owned_copy_preserves_fifo_metadata_none_zero_and_survives_overwrite():
    ring, stop = ring_fixture()
    try:
        originals = [tiny_segment(i, i + 1) for i in range(3)]
        for s in originals:
            assert ring.write(s, stop)
        dense = []
        while len(dense) < 3:
            dense.extend(ring.read_dense(3-len(dense), 2).segments())
        assert [s.slot_id for s in dense] == [0, 1, 2]
        for shared in (ring.actions, ring.end_freeze_masks, ring.initial_memories, ring.behavior_versions,
                       ring.behavior_log_probs, ring.rewards, ring.discounts, *ring.observations.values(),
                       *ring.bootstrap_observations.values()):
            shared.fill_(0)
        for a, b in zip(originals, dense):
            assert b.length == len(a.transitions)
            assert b.freeze_masks.tolist() == [-1 if t.end_freeze_mask is None else t.end_freeze_mask for t in a.transitions]
            assert b.behavior_versions.tolist() == [t.behavior_version for t in a.transitions]
            assert b.battle_outcome == a.battle_outcome and b.terminal == a.terminal
            torch.testing.assert_close(b.initial_memory, a.initial_memory, rtol=0, atol=0)
            for t, row in enumerate(a.transitions):
                torch.testing.assert_close(b.observations['x'][t], torch.as_tensor(row.observation['x']), rtol=0, atol=0)
                torch.testing.assert_close(b.actions[t], torch.as_tensor(row.action), rtol=0, atol=0)
            torch.testing.assert_close(b.bootstrap_observation['x'], torch.as_tensor(a.bootstrap_observation['x']), rtol=0, atol=0)
        assert ring.stats()['read'] == 3 and not ring._read_leases
    finally:
        ring.close()


def test_lease_holds_producer_then_rejects_double_and_stale_generation_release():
    ring, stop = ring_fixture(1)
    try:
        ring.write(tiny_segment(1), stop)
        lease = ring.lease_ready(1, 2)[0]
        with pytest.raises(Empty):
            ring.free_queue.get(timeout=.02)
        ring.release_lease(lease)
        with pytest.raises(RuntimeError, match='released'):
            ring.release_lease(lease)
        ring.write(tiny_segment(2), stop)
        second = ring.lease_ready(1, 2)[0]
        assert second.slot == lease.slot and second.generation != lease.generation
        with pytest.raises(RuntimeError, match='stale'):
            ring.release_lease(lease)
        assert ring._read_leases == {second.slot: second.generation}
        ring.release_lease(second)
    finally:
        ring.close()


def test_injected_copy_failure_returns_all_leases_and_does_not_publish_partial_batch(monkeypatch):
    ring, stop = ring_fixture(1)
    original = OwnedTrajectoryBatch.copy_from_ring
    try:
        ring.write(tiny_segment(), stop)
        def fail(cls, *args):
            raise RuntimeError('injected copy failure')
        monkeypatch.setattr(OwnedTrajectoryBatch, 'copy_from_ring', classmethod(fail))
        with pytest.raises(RuntimeError, match='injected'):
            ring.read_dense(1, 2)
        assert not ring._read_leases
        monkeypatch.setattr(OwnedTrajectoryBatch, 'copy_from_ring', original)
        assert ring.write(tiny_segment(5), stop)
        assert ring.read_dense(1, 2).lengths == (3,)
    finally:
        ring.close()


def test_producer_copy_exception_and_shutdown_do_not_leak_slots():
    ring, stop = ring_fixture(1)
    try:
        malformed = tiny_segment()
        malformed.transitions[0].observation['x'] = np.zeros(3, dtype=np.float32)
        with pytest.raises(RuntimeError):
            ring.write(malformed, stop)
        assert ring.write(tiny_segment(), stop)
        ring.lease_ready(1, 2)
        assert ring._read_leases
        ring.close()
        assert not ring._read_leases
    finally:
        if ring._read_leases:
            ring.close()


def test_delayed_owned_copy_keeps_slot_leased_until_copy_returns(monkeypatch):
    ring, stop = ring_fixture(1)
    entered, proceed, produced = threading.Event(), threading.Event(), threading.Event()
    original = OwnedTrajectoryBatch.copy_from_ring
    result, errors = [], []
    def delayed(cls, *args):
        entered.set()
        assert proceed.wait(3)
        return original(*args)
    monkeypatch.setattr(OwnedTrajectoryBatch, 'copy_from_ring', classmethod(delayed))
    def consume():
        try:
            result.append(ring.read_dense(1, 2))
        except BaseException as exc:
            errors.append(exc)
    def produce():
        ring.write(tiny_segment(99), stop)
        produced.set()
    try:
        ring.write(tiny_segment(1), stop)
        consumer = threading.Thread(target=consume)
        producer = threading.Thread(target=produce)
        consumer.start()
        assert entered.wait(2)
        producer.start()
        assert not produced.wait(.03)
        proceed.set()
        consumer.join(3)
        producer.join(3)
        assert not errors and produced.is_set()
        assert result[0].observations['x'][0, 0].item() == 1
        assert ring.observations['x'][0, 0, 0].item() == 99
        assert not ring._read_leases
    finally:
        proceed.set()
        stop.set()
        ring.close()


def convert_through_ring(data):
    ctx = mp.get_context('spawn')
    ring = SharedTrajectoryRing.create(data[0].transitions[0].observation,
        len(data[0].initial_memory), ctx, capacity=max(1, len(data)))
    try:
        for segment in data:
            ring.write(segment, ctx.Event())
        result = []
        while len(result) < len(data):
            result.extend(ring.read_dense(len(data)-len(result), 3).segments())
        return result
    finally:
        ring.close()


@pytest.mark.parametrize('joint', [False, True])
def test_dense_packing_and_complete_learner_step_equal_legacy(joint):
    _, primitive, head = policies()
    policy = head if joint else primitive
    data = BatchedSelfPlayCollector(policy, 2, 'cpu', seed=927, discount_mode='per_turn').collect(18)
    dense = convert_through_ring(data)
    a = pack_learner_batch(data, 'cpu', joint_actions=joint, include_metadata=True)
    b = pack_learner_batch(dense, 'cpu', joint_actions=joint)
    for key in vars(a):
        x, y = getattr(a, key), getattr(b, key)
        if isinstance(x, dict):
            for k in x:
                torch.testing.assert_close(x[k], y[k], rtol=0, atol=0)
        elif x is not None:
            torch.testing.assert_close(x, y, rtol=0, atol=0)
        else:
            assert y is None
    old, new = copy.deepcopy(policy), copy.deepcopy(policy)
    old_opt, new_opt = torch.optim.AdamW(old.parameters()), torch.optim.AdamW(new.parameters())
    first = learner_update(old, old_opt, data, 'cpu', use_bfloat16=False, lean_bootstrap=True)
    second = learner_update(new, new_opt, dense, 'cpu', use_bfloat16=False, lean_bootstrap=True)
    assert vars(first) == vars(second)
    for x, y in zip(old.parameters(), new.parameters()):
        torch.testing.assert_close(x, y, rtol=0, atol=0)
        if x.grad is not None:
            torch.testing.assert_close(x.grad, y.grad, rtol=0, atol=0)
    for x, y in zip(old_opt.state.values(), new_opt.state.values()):
        for key in x:
            torch.testing.assert_close(x[key], y[key], rtol=0, atol=0)


def test_dense_lag_filter_and_execution_metrics_match_legacy():
    name = '_dense_cli_metrics'
    spec = importlib.util.spec_from_file_location(name, Path(__file__).resolve().parents[1] / 'scripts/train_entity_selfplay.py')
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    data = [tiny_segment(i, i+1) for i in range(3)]
    dense = convert_through_ring(data)
    assert module.action_execution_metrics(data) == module.action_execution_metrics(dense)
    for version in range(8):
        assert module.behavior_lag_metrics(data, version) == module.behavior_lag_metrics(dense, version)
        assert [module.max_segment_lag(s, version) for s in data] == [module.max_segment_lag(s, version) for s in dense]
