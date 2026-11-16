import json
from pathlib import Path
import sys

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'scripts'))
from train_entity_selfplay import demo_update
from sapai.turtle_pack import SIMULATOR_REVISION
from sapai_gym1.rl.joint_demonstrations import QUALITY_SCHEMA, make_joint_example, JointSearchDataset, collate_joint_examples, load_quality_gate, reconstruct_memory, runtime_fingerprint, config_hash
from sapai_gym1.rl.joint_action import JOINT_SCHEMA
from sapai_gym1.rl.action_schema import ActionType
from test_joint_search_roundtrip import fixture


def gate(tmp_path):
    path = tmp_path/'test-gate.json'
    path.write_text(json.dumps({'schema': QUALITY_SCHEMA, 'simulatorRevision': SIMULATOR_REVISION,
        'actionSchema': JOINT_SCHEMA, 'policySHA256': 'a'*64, 'executionMode': 'root-only-intervention',
        'approvedForDistillation': True, 'heldoutStates': True, 'freshFutureSeeds': True,
        'terminalContinuationFraction': 1., 'pairedStateMatchScoreImprovement95CI': [.01, .1],
        'statePanelSHA256': 'b'*64, 'seedPanelSHA256': 'c'*64,
        'sourceManifestSHA256': runtime_fingerprint(), 'plannerConfigSHA256': config_hash({})}))
    return path


def record(policy, state, evaluator, memory, quality=None, prefix=()):
    evaluation = evaluator.evaluate(state, memory)
    action = next(a for a in evaluation.priors if a.action_type == ActionType.END_TURN and a.end_freeze_mask)
    example = {'replay_id': 'test-root', 'teacherWeight': .75,
               'teacher': {'outerAuditReturnLowerBound': .02}}
    return make_joint_example(example, action, state.observation(), memory, policy=policy,
        checkpoint_sha256='a'*64, omitted_mass=evaluation.omitted_probability_mass,
        observation_prefix=prefix, quality_gate=quality)


def test_loader_actual_optimizer_step_preserves_masks_and_reconstructs_current_memory(tmp_path):
    env, policy, state, evaluator, memory = fixture(); approval = gate(tmp_path)
    example = record(policy, state, evaluator, memory, approval)
    path = tmp_path/'corpus.jsonl'; path.write_text(json.dumps(example)+'\n')
    dataset = JointSearchDataset([path], checkpoint_sha256='a'*64, quality_gate=approval)
    batch = collate_joint_examples([dataset[0]])
    assert batch['end_freeze_masks'].item() != 0 and batch['memory_prefixes'] == [[]]
    assert batch['initial_memories'] == [None]
    old = policy.end_freeze_head.scorer.weight.detach().clone()
    optimizer = torch.optim.Adam(policy.parameters(), lr=.001)
    loss, probability = demo_update(policy, optimizer, batch, torch.device('cpu'), 1.)
    assert torch.isfinite(torch.tensor(loss)) and 0 <= probability <= 1
    assert not torch.equal(old, policy.end_freeze_head.scorer.weight)
    prefix = [state.observation(), state.observation()]
    reconstructed = reconstruct_memory(policy, prefix, torch.device('cpu'))
    expected = policy.initial_memory(1)
    with torch.no_grad():
        for observation in prefix:
            expected = policy.encode_decision({k: torch.as_tensor(v)[None] for k, v in observation.items()}, expected, forecast=False).memory
    torch.testing.assert_close(reconstructed, expected[0], atol=0, rtol=0)


def test_unapproved_foreign_internal_and_prefixless_records_fail_closed(tmp_path):
    env, policy, state, evaluator, memory = fixture()
    example = record(policy, state, evaluator, memory)
    path = tmp_path/'diagnostic.jsonl'; path.write_text(json.dumps(example)+'\n')
    with pytest.raises(ValueError, match='not a teacher-quality'):
        JointSearchDataset([path], checkpoint_sha256='a'*64)
    with pytest.raises(ValueError, match='Foreign checkpoint'):
        JointSearchDataset([path], checkpoint_sha256='b'*64, diagnostic=True)
    approval = gate(tmp_path)
    with pytest.raises(ValueError, match='Unapproved'):
        JointSearchDataset([path], checkpoint_sha256='a'*64, quality_gate=approval)
    incomplete = record(policy, state, evaluator, memory, approval, prefix=None)
    assert incomplete['productionTrainingAllowed'] is False
    from sapai_gym1.replay.dataset import ReplayExampleDataset
    with pytest.raises(ValueError, match='joint-aware'):
        ReplayExampleDataset([path])


@pytest.mark.parametrize('key,value', [('executionMode', 'engine'), ('approvedForDistillation', False),
    ('terminalContinuationFraction', .9), ('pairedStateMatchScoreImprovement95CI', [-.01, .2])])
def test_quality_approval_requires_positive_independent_root_terminal_evidence(tmp_path, key, value):
    path = gate(tmp_path); payload = json.loads(path.read_text()); payload[key] = value; path.write_text(json.dumps(payload))
    with pytest.raises(ValueError):
        load_quality_gate(path, checkpoint_sha256='a'*64)
