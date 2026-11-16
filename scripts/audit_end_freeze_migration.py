#!/usr/bin/env python3
"""CPU-only, read-only checkpoint migration audit; never saves model weights."""
import argparse
from dataclasses import replace
import hashlib
import json
from pathlib import Path

import torch

from sapai_gym1.rl import EntityTransformerConfig, EntityTransformerPolicy, ParallelSelfPlayEnv
from sapai_gym1.rl.training import stack_observations
from sapai_gym1.rl.end_freeze_checkpoint import initialize_end_freeze_from_primitive, attach_end_freeze_optimizer_group, HEAD_PREFIX


HISTORY_PREFIXES = ('opponent_history_', 'opponent_pet_head.', 'opponent_status_head.',
                    'opponent_level_head.', 'opponent_stats_head.', 'opponent_temporary_head.')


def legacy_groups(policy, saved_groups):
    named = [(name, p) for name, p in policy.named_parameters() if not name.startswith(HEAD_PREFIX)]
    if len(saved_groups) == 1 and saved_groups[0].get('name', 'base') == 'base':
        groups = [{'params': [p for _, p in named]}]
    elif [g.get('name') for g in saved_groups] == ['base', 'opponent_history', 'opponent_adapter']:
        groups = [
            {'params': [p for n, p in named if not n.startswith(HISTORY_PREFIXES) and not n.startswith('opponent_belief_adapter.')], 'name': 'base'},
            {'params': [p for n, p in named if n.startswith(HISTORY_PREFIXES)], 'name': 'opponent_history'},
            {'params': [p for n, p in named if n.startswith('opponent_belief_adapter.')], 'name': 'opponent_adapter'},
        ]
    else:
        raise ValueError('Unknown optimizer layout; do not guess parameter ownership')
    if [len(g['params']) for g in groups] != [len(g['params']) for g in saved_groups]:
        raise ValueError('Optimizer parameter counts do not match the declared legacy layout')
    return groups


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('checkpoint', type=Path)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    torch.set_num_threads(1)
    saved = torch.load(args.checkpoint, map_location='cpu', weights_only=False)
    config = EntityTransformerConfig(**saved['config'])
    if config.end_freeze:
        raise ValueError('Expected a primitive checkpoint')
    parent = EntityTransformerPolicy(config).eval()
    parent.load_state_dict(saved['model'])
    joint = EntityTransformerPolicy(replace(config, end_freeze=True)).eval()
    initialize_end_freeze_from_primitive(joint, saved['model'])
    optimizers = []
    for policy in (parent, joint):
        optimizer = torch.optim.AdamW(legacy_groups(policy, saved['optimizer']['param_groups']))
        optimizer.load_state_dict(saved['optimizer'])
        optimizers.append(optimizer)
    attach_end_freeze_optimizer_group(joint, optimizers[1], 5e-5)
    parent_named, joint_named = dict(parent.named_parameters()), dict(joint.named_parameters())
    moment_tensors = 0
    for name, parameter in parent_named.items():
        counterpart = joint_named[name]
        assert torch.equal(parameter, counterpart), name
        before, after = optimizers[0].state.get(parameter, {}), optimizers[1].state.get(counterpart, {})
        assert before.keys() == after.keys(), name
        for key in before:
            if torch.is_tensor(before[key]):
                assert torch.equal(before[key], after[key]), (name, key)
                moment_tensors += 1
            else:
                assert before[key] == after[key], (name, key)
    assert all(p not in optimizers[1].state for p in joint.end_freeze_head.parameters())
    env = ParallelSelfPlayEnv()
    env.reset(seed=440901)
    comparisons = []
    with torch.no_grad():
        for gold in (10, 0):
            env.core.agent.gold = env.core.opponent.gold = gold
            observation = stack_observations(env.observations(), 'cpu')
            original = parent(observation, deterministic=True)
            migrated = joint(observation, deterministic=True, allow_end_freeze=True)
            for field in ('actions', 'value', 'memory', 'type_logits', 'source_logits', 'target_logits', 'battle_logits'):
                assert torch.equal(getattr(original, field), getattr(migrated, field)), (gold, field)
            assert migrated.end_freeze_mask.tolist() == [0, 0]
            comparisons.append({'gold': gold, 'greedyActions': original.actions.tolist(), 'allSharedOutputsBitExact': True})
    digest = hashlib.sha256()
    with args.checkpoint.open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(chunk)
    report = {
        'checkpoint': str(args.checkpoint), 'sha256': digest.hexdigest(), 'update': saved['update'],
        'simulatorRevision': saved.get('simulator_revision'), 'parentParameterCount': parent.parameter_count(),
        'addedParameterCount': joint.parameter_count() - parent.parameter_count(),
        'preservedParameterTensors': len(parent_named), 'preservedOptimizerTensors': moment_tensors,
        'parentGroupLearningRates': [g['lr'] for g in optimizers[0].param_groups],
        'migratedGroupLearningRates': [g['lr'] for g in optimizers[1].param_groups],
        'comparisonStates': comparisons, 'cpuOnly': True, 'modelWeightsWritten': False,
        'note': 'Compatibility audit only. Stochastic exploration changes; no strength claim or production activation.',
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_suffix(args.output.suffix + '.tmp')
    temporary.write_text(json.dumps(report, indent=2) + '\n')
    temporary.replace(args.output)
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
