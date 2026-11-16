"""Joint-aware corpus boundary, independent quality authorization and burn-in."""
import hashlib
import json
import math
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset

from sapai.turtle_pack import SIMULATOR_REVISION
from .joint_action import JointAction, JOINT_SCHEMA
from .joint_search_records import make_record, restore_record, rescore_record

QUALITY_SCHEMA = 'independent-terminal-search-quality-v1-20260906'


def config_hash(config):
    return hashlib.sha256(json.dumps(config, sort_keys=True, separators=(',', ':')).encode()).hexdigest()


def runtime_fingerprint():
    import sapai
    own = Path(__file__).parent
    simulator = Path(sapai.__file__).parent
    files = {f'rl/{p.name}': sha(p) for p in own.glob('*.py')}
    files.update({f'simulator/{p.relative_to(simulator)}': sha(p) for p in simulator.rglob('*.py')})
    files['torchVersion'] = torch.__version__
    return config_hash(files)


def sha(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def serialize_action(action):
    return action.to_dict() if isinstance(action, JointAction) else list(action.as_tuple())


def action_key(action):
    return json.dumps(action.to_dict(), sort_keys=True, separators=(',', ':')) if isinstance(action, JointAction) else ','.join(map(str, action.as_tuple()))


def load_quality_gate(path, *, checkpoint_sha256, planner_config=None):
    gate = json.loads(Path(path).read_text())
    if (gate.get('schema') != QUALITY_SCHEMA or gate.get('simulatorRevision') != SIMULATOR_REVISION
            or gate.get('actionSchema') != JOINT_SCHEMA or gate.get('policySHA256') != checkpoint_sha256
            or gate.get('executionMode') != 'root-only-intervention'
            or gate.get('approvedForDistillation') is not True or gate.get('heldoutStates') is not True
            or gate.get('freshFutureSeeds') is not True or gate.get('terminalContinuationFraction') != 1.):
        raise ValueError('Production joint demonstrations require an independently reviewed root-only terminal-quality gate')
    interval = gate.get('pairedStateMatchScoreImprovement95CI', [])
    if len(interval) != 2 or not all(math.isfinite(x) for x in interval) or not 0 < interval[0] <= interval[1] <= 1:
        raise ValueError('Held-out terminal match-score improvement is not established')
    for key in ('statePanelSHA256', 'seedPanelSHA256', 'sourceManifestSHA256'):
        value = gate.get(key, '')
        if len(value) != 64 or any(c not in '0123456789abcdef' for c in value):
            raise ValueError(f'Missing independent panel provenance: {key}')
    if gate['sourceManifestSHA256'] != runtime_fingerprint():
        raise ValueError('Quality gate does not validate this simulator/model/search source')
    if planner_config is not None and gate.get('plannerConfigSHA256') != config_hash(planner_config):
        raise ValueError('Quality gate does not validate this planner configuration')
    return gate


def serialize_observation(observation):
    return {k: {'dtype': np.asarray(v).dtype.str, 'shape': list(np.asarray(v).shape), 'data': np.asarray(v).tolist()}
            for k, v in observation.items()}


def make_joint_example(example, action, observation, memory, *, policy, checkpoint_sha256,
                       omitted_mass, observation_prefix=None, is_root=True, quality_gate=None):
    if not isinstance(action, JointAction):
        return example
    if policy is None:
        raise ValueError('Original joint behavior likelihood must be explicitly rescored')
    outer = example['teacher'].get('outerAuditReturnLowerBound')
    confirmed = bool(is_root and outer is not None and outer > 0)
    record = make_record(observation, memory, (action,), (1.,), checkpoint_sha256=checkpoint_sha256,
                         omitted_probability_mass=omitted_mass, confirmed=confirmed,
                         metadata={'replay_id': example['replay_id'], 'teacherWeight': example['teacherWeight'],
                                   'teacher': example['teacher'], 'isRoot': bool(is_root)})
    scores, _ = rescore_record(record, policy, checkpoint_sha256=checkpoint_sha256, device=next(policy.parameters()).device)
    record['originalJointLogProbabilities'] = list(scores)
    record['memoryActingPrecision'] = 'cuda-bf16' if next(policy.parameters()).device.type == 'cuda' else 'cpu-fp32'
    record['jointRescorePrecision'] = str(next(policy.parameters()).dtype)
    record['memoryObservationPrefix'] = None if observation_prefix is None else [serialize_observation(o) for o in observation_prefix]
    record['memoryReset'] = 'match-start-zero' if observation_prefix is not None else None
    record['recurrentTrainingMode'] = 'reconstruct-current-policy-from-reset' if observation_prefix is not None else 'diagnostic-checkpoint-owned-memory'
    if quality_gate is not None:
        if not confirmed or observation_prefix is None:
            # Internal search states without an exact causal prefix remain
            # diagnostics; no foreign hidden state is silently taught.
            return record
        load_quality_gate(quality_gate, checkpoint_sha256=checkpoint_sha256,
                          planner_config=example['teacher'].get('plannerConfig', example['teacher'].get('puctConfig', {})))
        record['productionTrainingAllowed'] = True
        record['qualityGateSHA256'] = sha(quality_gate)
    return record


def restore_prefix(record, *, checkpoint_sha256):
    prefix = record.get('memoryObservationPrefix')
    if prefix is None or record.get('memoryReset') != 'match-start-zero':
        raise ValueError('Production distillation requires a full observation prefix from a genuine match reset')
    observations = []
    for encoded in prefix:
        copy = {**record, 'observation': encoded}
        observations.append(restore_record(copy, checkpoint_sha256=checkpoint_sha256)[0])
    return observations


@torch.no_grad()
def reconstruct_memory(policy, prefix, device):
    memory = policy.initial_memory(1, device)
    for observation in prefix:
        batch = {k: torch.as_tensor(v, device=device)[None] for k, v in observation.items()}
        memory = policy.encode_decision(batch, memory, forecast=False).memory
    return memory[0].detach()


class JointSearchDataset(Dataset):
    def __init__(self, paths, *, checkpoint_sha256, quality_gate=None, diagnostic=False):
        self.items = []
        gate_hash = None
        if not diagnostic:
            if quality_gate is None:
                raise ValueError('A search round-trip is not a teacher-quality gate')
            load_quality_gate(quality_gate, checkpoint_sha256=checkpoint_sha256)
            gate_hash = sha(quality_gate)
        for path in paths:
            for line in Path(path).read_text().splitlines():
                if not line.strip():
                    continue
                record = json.loads(line)
                observation, memory, decisions, probabilities = restore_record(record, checkpoint_sha256=checkpoint_sha256)
                if record['actionSchema'] != JOINT_SCHEMA:
                    raise ValueError('Primitive records are not joint demonstrations')
                if not diagnostic and (record.get('productionTrainingAllowed') is not True
                        or record.get('qualityGateSHA256') != gate_hash
                        or record.get('rootIndependentlyConfirmed') is not True
                        or record['metadata'].get('isRoot') is not True):
                    raise ValueError('Unapproved/internal/unconfirmed record in production corpus')
                if not diagnostic:
                    teacher = record['metadata']['teacher']
                    load_quality_gate(quality_gate, checkpoint_sha256=checkpoint_sha256,
                                      planner_config=teacher.get('plannerConfig', teacher.get('puctConfig', {})))
                logps = record.get('originalJointLogProbabilities')
                if logps is None or len(logps) != len(decisions) or any(not math.isfinite(p) or p > 1e-6 for p in logps):
                    raise ValueError('Missing/invalid ORIGINAL joint likelihoods; target weights are not likelihoods')
                prefix = restore_prefix(record, checkpoint_sha256=checkpoint_sha256) if not diagnostic else None
                weight = float(record['metadata'].get('teacherWeight', 1.))
                if not math.isfinite(weight) or weight <= 0:
                    raise ValueError('Invalid teacher weight')
                for decision, probability in zip(decisions, probabilities):
                    if probability == 0:
                        continue
                    self.items.append({'observation': observation, 'action': np.asarray(decision.primitive.as_tuple()),
                        'end_freeze_mask': decision.end_freeze_mask, 'weight': weight*probability,
                        'initial_memory': memory if diagnostic else None, 'memory_prefix': prefix,
                        'battle_outcome': -1, 'return': float('nan'),
                        'replay_id': record['metadata'].get('replay_id', str(len(self.items)))})
        if not self.items:
            raise ValueError('Empty joint demonstration corpus')
        self.action_count = len(self.items)

    def __len__(self):
        return len(self.items)

    def __getitem__(self, index):
        return self.items[index]


def collate_joint_examples(batch):
    from sapai_gym1.replay.dataset import collate_replay_examples
    result = collate_replay_examples(batch)
    result['end_freeze_masks'] = torch.tensor([[item['end_freeze_mask'] for item in batch]], dtype=torch.long)
    result['memory_prefixes'] = [item['memory_prefix'] for item in batch]
    return result
