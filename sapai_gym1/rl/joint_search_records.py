"""Versioned diagnostic/demo/reservoir records preserving complete decisions."""
import json
import math
from pathlib import Path
import re

import numpy as np
import torch

from sapai.turtle_pack import SIMULATOR_REVISION
from .joint_action import JointAction
from .decision_api import decode_decisions
from .entity_observation import EntityObservationEncoder

LEGACY_RECORD_SCHEMA='typed-search-decision-record-v1-20260906'
RECORD_SCHEMA='typed-search-decision-record-v2-prefix-20260906'
RESERVOIR_SCHEMA='typed-search-decision-reservoir-v1-20260906'


def make_record(observation, memory, decisions, probabilities, *, checkpoint_sha256,
                omitted_probability_mass=0., confirmed=False, metadata=None):
    decisions=tuple(decisions);probabilities=tuple(map(float,probabilities))
    if not decisions or len(decisions)!=len(probabilities) or len(set(decisions))!=len(decisions):
        raise ValueError('Distinct complete decisions with matching weights are required')
    if not all(isinstance(d,JointAction) for d in decisions) or len({d.schema for d in decisions})!=1:
        raise ValueError('Records require one explicit action schema')
    if not re.fullmatch('[a-f0-9]{64}',checkpoint_sha256):
        raise ValueError('Exact recurrent-memory checkpoint ownership is required')
    if any(not math.isfinite(p) or p<0 for p in probabilities) or not math.isclose(sum(probabilities),1.,abs_tol=1e-6):
        raise ValueError('Invalid soft target distribution')
    if not 0 <= omitted_probability_mass <= 1:
        raise ValueError('Invalid omitted proposal mass')
    mem=np.asarray(torch.as_tensor(memory).detach().float().cpu()).reshape(-1)
    if not np.isfinite(mem).all():
        raise ValueError('Nonfinite recurrent state')
    return {'recordSchema':RECORD_SCHEMA,'actionSchema':decisions[0].schema,
        'simulatorRevision':SIMULATOR_REVISION,'checkpointSHA256':checkpoint_sha256,
        'observation':{k:{'dtype':np.asarray(v).dtype.str,'shape':list(np.asarray(v).shape),
                          'data':np.asarray(v).tolist()} for k,v in observation.items()},
        'memory':mem.tolist(),'decisions':[d.to_dict() for d in decisions],
        'targetProbabilities':probabilities,'omittedEndMaskProbabilityMass':float(omitted_probability_mass),
        'rootIndependentlyConfirmed':bool(confirmed),'productionTrainingAllowed':False,
        'metadata':dict(metadata or {})}


def restore_record(record, *, checkpoint_sha256, encoder=None):
    if record.get('recordSchema') not in (RECORD_SCHEMA, LEGACY_RECORD_SCHEMA) or record.get('simulatorRevision')!=SIMULATOR_REVISION:
        raise ValueError('Unsupported record or simulator revision')
    if record.get('checkpointSHA256')!=checkpoint_sha256:
        raise ValueError('Foreign checkpoint recurrent memory requires reconstruction/burn-in')
    decisions=tuple(JointAction.from_dict(d) for d in record['decisions'])
    if any(d.schema!=record['actionSchema'] for d in decisions):
        raise ValueError('Action schema mismatch; never discard a mask')
    space=(encoder or EntityObservationEncoder()).observation_space()
    if set(record['observation']) != set(space.spaces):
        raise ValueError('Unknown or missing observation fields')
    obs={}
    for key,value in record['observation'].items():
        array=np.asarray(value['data'],dtype=np.dtype(value['dtype']))
        if list(array.shape)!=value['shape'] or array.shape!=space[key].shape or array.dtype!=space[key].dtype:
            raise ValueError(f'Observation shape/dtype mismatch: {key}')
        if not np.isfinite(array).all():
            raise ValueError('Nonfinite observation')
        obs[key]=array
    memory=torch.tensor(record['memory'],dtype=torch.float32)
    # Reuse creation validation for probabilities, schema, memory and mass.
    make_record(obs,memory,decisions,record['targetProbabilities'],checkpoint_sha256=checkpoint_sha256,
                omitted_probability_mass=record['omittedEndMaskProbabilityMass'])
    return obs,memory,decisions,tuple(record['targetProbabilities'])


@torch.inference_mode()
def rescore_record(record, policy, *, checkpoint_sha256, device='cpu'):
    observation,memory,decisions,_=restore_record(record,checkpoint_sha256=checkpoint_sha256)
    if memory.shape!=(policy.config.memory_dim,):
        raise ValueError('Recurrent state shape does not match policy')
    batch={k:torch.as_tensor(v,device=device)[None] for k,v in observation.items()}
    features=policy.encode_decision(batch,memory.to(device)[None],forecast=False)
    scores=[]
    for decision in decisions:
        _,output=decode_decisions(policy,features,decisions=(decision,))
        scores.append(float(output.log_prob[0]))
    return tuple(scores),features.memory[0].detach().cpu()


def diagnostic_record_loss(record, policy, *, checkpoint_sha256, device='cpu'):
    """Exercise the real differentiable policy consumer, but NOT a production gate.

    No optimizer step and no mutation of caller memory. This is useful for
    contract tests; a numerical loss is not evidence that the teacher is good.
    Production corpus/CLI activation still requires an independent quality gate.
    """
    if record.get('productionTrainingAllowed') is not False:
        raise ValueError('This loss path is diagnostic-only, not a production training authorization')
    observation, memory, decisions, probabilities = restore_record(record, checkpoint_sha256=checkpoint_sha256)
    if memory.shape != (policy.config.memory_dim,):
        raise ValueError('Recurrent state shape does not match policy')
    batch = {k: torch.as_tensor(v, device=device)[None] for k, v in observation.items()}
    features = policy.encode_decision(batch, memory.to(device)[None], forecast=False)
    losses = []
    for decision, probability in zip(decisions, probabilities):
        _, output = decode_decisions(policy, features, decisions=(decision,))
        losses.append(-probability*output.log_prob[0])
    return torch.stack(losses).sum()


class JointSearchReservoir:
    def __init__(self, records=()):
        self.records=list(records)

    def save(self,path):
        path=Path(path)
        with path.open('x') as output:
            json.dump({'reservoirSchema':RESERVOIR_SCHEMA,'records':self.records},output)

    @classmethod
    def load(cls,path):
        payload=json.loads(Path(path).read_text())
        if payload.get('reservoirSchema')!=RESERVOIR_SCHEMA:
            raise ValueError('Unsupported typed search reservoir')
        for row in payload['records']:
            restore_record(row,checkpoint_sha256=row['checkpointSHA256'])
        return cls(payload['records'])

    def write_diagnostic_demonstrations(self,path):
        # Production corpus generation remains blocked until an independently
        # validated teacher panel and memory-reconstruction protocol are wired.
        with Path(path).open('x') as output:
            for row in self.records:
                if row.get('productionTrainingAllowed') is not False:
                    raise ValueError('This writer is diagnostic-only')
                output.write(json.dumps(row)+'\n')
