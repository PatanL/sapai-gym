from dataclasses import asdict
from pathlib import Path
import subprocess
import sys

import pytest
import torch

from sapai_gym1.rl import EntityTransformerConfig, EntityTransformerPolicy
from sapai_gym1.rl.end_freeze_checkpoint import initialize_end_freeze_from_primitive, attach_end_freeze_optimizer_group, HEAD_PREFIX, END_FREEZE_SCHEMA
from test_end_freeze_policy import policies


def test_explicit_optimizer_migration_preserves_all_parent_moments():
    _, parent, target = policies()
    old_optimizer = torch.optim.AdamW(parent.parameters(), lr=2e-6)
    sum(parameter.square().sum() for parameter in parent.parameters()).backward()
    old_optimizer.step()
    initialize_end_freeze_from_primitive(target, parent.state_dict())
    base_parameters = [parameter for name, parameter in target.named_parameters() if not name.startswith(HEAD_PREFIX)]
    optimizer = torch.optim.AdamW(base_parameters, lr=1e-3)
    optimizer.load_state_dict(old_optimizer.state_dict())
    attach_end_freeze_optimizer_group(target, optimizer, 5e-5)
    assert [group['lr'] for group in optimizer.param_groups] == [2e-6, 5e-5]
    for old_parameter, new_parameter in zip(parent.parameters(), base_parameters):
        torch.testing.assert_close(old_parameter, new_parameter, atol=0, rtol=0)
        for key in ('step', 'exp_avg', 'exp_avg_sq'):
            torch.testing.assert_close(old_optimizer.state[old_parameter][key], optimizer.state[new_parameter][key], atol=0, rtol=0)
    assert all(parameter not in optimizer.state for parameter in target.end_freeze_head.parameters())
    with pytest.raises(ValueError, match='already'):
        attach_end_freeze_optimizer_group(target, optimizer, 5e-5)
    with pytest.raises(ValueError, match='exact primitive'):
        initialize_end_freeze_from_primitive(target, target.state_dict())


@pytest.mark.parametrize('split_optimizer', [False, True])
def test_training_cli_can_migrate_and_resume_joint_checkpoint(tmp_path, split_optimizer):
    # CPU-only real entrypoint smoke: legacy optimizer -> add head -> update,
    # then reload the joint optimizer/schema and perform another update.
    from sapai.turtle_pack import SIMULATOR_REVISION
    _, parent, _ = policies()
    prefixes = ('opponent_history_', 'opponent_pet_head.', 'opponent_status_head.',
                'opponent_level_head.', 'opponent_stats_head.', 'opponent_temporary_head.')
    named = list(parent.named_parameters())
    if split_optimizer:
        optimizer = torch.optim.AdamW([
            {'params': [p for n, p in named if not n.startswith(prefixes) and not n.startswith('opponent_belief_adapter.')],
             'lr': 2e-6, 'name': 'base'},
            {'params': [p for n, p in named if n.startswith(prefixes)], 'lr': 1e-4, 'name': 'opponent_history'},
            {'params': [p for n, p in named if n.startswith('opponent_belief_adapter.')], 'lr': 1e-5, 'name': 'opponent_adapter'},
        ])
    else:
        optimizer = torch.optim.AdamW(parent.parameters(), lr=2e-6)
    expected_rates = [2e-6, 1e-4, 1e-5, 5e-5] if split_optimizer else [2e-6, 5e-5]
    checkpoint = tmp_path / 'parent.pt'
    torch.save({'config': asdict(parent.config), 'model': parent.state_dict(), 'optimizer': optimizer.state_dict(),
                'update': 0, 'transitions': 0, 'simulator_revision': SIMULATOR_REVISION,
                'training_semantics': {'discount_mode': 'per_turn'}}, checkpoint)
    script = Path(__file__).resolve().parents[1] / 'scripts/train_entity_selfplay.py'
    run_dir = tmp_path / 'joint-run'
    common = [sys.executable, str(script), '--device', 'cpu', '--smoke', '--discount-mode', 'per_turn',
              '--checkpoint-dir', str(run_dir), '--checkpoint-every', '1', '--disable-expert-anchor']
    if split_optimizer:
        common += ['--opponent-module-learning-rate', '0.0001', '--opponent-adapter-learning-rate', '0.00001']
    result = subprocess.run(common + ['--resume-checkpoint', str(checkpoint), '--updates', '1', '--end-freeze',
                                     '--end-freeze-learning-rate', '0.00005'], capture_output=True, text=True, timeout=40)
    assert result.returncode == 0, result.stdout + result.stderr
    assert 'migrated_parent=True' in result.stdout
    migrated = run_dir / 'update-000001.pt'
    state = torch.load(migrated, map_location='cpu', weights_only=False)
    assert state['training_semantics']['action_schema'] == END_FREEZE_SCHEMA
    assert state['config']['end_freeze']
    assert [group['lr'] for group in state['optimizer']['param_groups']] == expected_rates
    result = subprocess.run(common + ['--resume-checkpoint', str(migrated), '--updates', '2', '--distributed',
                                     '--num-envs', '2', '--trajectory-queue-size', '16'],
                            capture_output=True, text=True, timeout=40)
    assert result.returncode == 0, result.stdout + result.stderr
    assert 'migrated_parent=False' in result.stdout
    resumed = torch.load(run_dir / 'update-000002.pt', map_location='cpu', weights_only=False)
    assert resumed['transitions'] > state['transitions']
    assert [group['lr'] for group in resumed['optimizer']['param_groups']] == expected_rates
    restored = EntityTransformerPolicy(EntityTransformerConfig(**resumed['config']))
    restored.load_state_dict(resumed['model'])
