import copy
import importlib.util
import os
import sys
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from sapai_gym1.rl.training import Transition, TurnSegment, BatchedSelfPlayCollector, learner_update
from sapai_gym1.rl.learner_batch import pack_learner_batch
from sapai_gym1.rl.distributed import inference_results_on_cpu
from test_end_freeze_policy import policies


@pytest.mark.parametrize('joint', [False, True])
def test_dense_fields_and_padding_match_scalar_reference(joint):
    segments = []
    for size in (1, 3, 7):
        rows = [Transition({'ids': np.array([i, size], dtype=np.int64), 'values': np.array([i/7], dtype=np.float32)},
                           np.array([i % 9, 2, 3]), -.137 * i, reward=.2*i, discount=.99,
                           end_freeze_mask=i if joint else None) for i in range(size)]
        segments.append(TurnSegment(rows, torch.zeros(2), rows[-1].observation, False))
    result = pack_learner_batch(segments, 'cpu', joint_actions=joint)
    for t in range(7):
        for b, segment in enumerate(segments):
            row = segment.transitions[min(t, len(segment.transitions)-1)]
            assert torch.equal(result.actions[t,b], torch.as_tensor(row.action))
            for key, value in row.observation.items():
                assert torch.equal(result.observations[key][t,b], torch.as_tensor(value))
            for name, attribute in [('behavior_log_probs','behavior_log_prob'),('rewards','reward'),('discounts','discount')]:
                expected = getattr(row, attribute) if t < len(segment.transitions) else 0.
                assert getattr(result, name)[t,b] == torch.tensor(expected)
            if joint:
                assert int(result.freeze_masks[t,b]) == row.end_freeze_mask
    assert result.actions.is_contiguous()
    assert all(value.is_contiguous() for value in result.observations.values())


def test_batched_inference_copy_preserves_bfloat_values_and_none_masks():
    output = SimpleNamespace(actions=torch.arange(9).reshape(3,3), log_prob=torch.tensor([-.1,-.2,-.3],dtype=torch.bfloat16),
                             memory=torch.arange(12,dtype=torch.bfloat16).reshape(3,4), end_freeze_mask=torch.tensor([0,1,1023]))
    actions, log_probs, memories, masks = inference_results_on_cpu(output)
    for index in range(3):
        assert torch.equal(actions[index], output.actions[index].cpu())
        assert float(log_probs[index]) == float(output.log_prob[index])
        assert torch.equal(memories[index], output.memory[index].float().cpu())
    assert masks.tolist() == [0,1,1023]
    output.end_freeze_mask = None
    assert inference_results_on_cpu(output)[3] is None


@pytest.mark.parametrize('joint', [False, True])
def test_complete_learner_step_matches_frozen_reference(joint):
    path = os.environ.get('SAP_LEGACY_TRAINING')
    if not path:
        pytest.skip('Mount the frozen training module for the cross-revision equivalence gate')
    name = 'sapai_gym1.rl._packing_legacy_reference'
    spec = importlib.util.spec_from_file_location(name, path)
    legacy = importlib.util.module_from_spec(spec)
    sys.modules[name] = legacy
    spec.loader.exec_module(legacy)
    _, primitive, head = policies()
    policy = head if joint else primitive
    data = BatchedSelfPlayCollector(policy, 2, 'cpu', seed=771, discount_mode='per_turn').collect(8)
    before, after = copy.deepcopy(policy), copy.deepcopy(policy)
    old_optimizer = torch.optim.AdamW(before.parameters(), lr=1e-4)
    new_optimizer = torch.optim.AdamW(after.parameters(), lr=1e-4)
    old = legacy.learner_update(before, old_optimizer, data, 'cpu', use_bfloat16=False)
    new = learner_update(after, new_optimizer, data, 'cpu', use_bfloat16=False)
    assert vars(old) == vars(new)
    for first, second in zip(before.parameters(), after.parameters()):
        assert torch.equal(first, second)
        if first.grad is not None:
            assert torch.equal(first.grad, second.grad)
    for first, second in zip(old_optimizer.state.values(), new_optimizer.state.values()):
        for key in first:
            assert torch.equal(first[key], second[key])
