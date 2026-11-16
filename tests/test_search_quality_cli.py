from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import subprocess
import sys

import numpy as np
import torch

from sapai.turtle_pack import SIMULATOR_REVISION
from sapai_gym1.rl.decision_manifest import MANIFEST_VERSION, decision_record
from sapai_gym1.rl.joint_action import JointAction, step_decisions
from sapai_gym1.rl.action_schema import AutoregressiveAction
from sapai_gym1.rl.opponent_reservoir import RESERVOIR_SCHEMA_VERSION
from test_end_freeze_policy import policies


def test_full_quality_cli_completes_root_and_online_engine_without_approval(tmp_path):
    env, _, policy = policies(); env.reset(seed=609090)
    checkpoint = tmp_path/'tiny-diagnostic.pt'
    torch.save({'config': asdict(policy.config), 'model': policy.state_dict(),
                'simulator_revision': SIMULATOR_REVISION, 'diagnosticOnly': True}, checkpoint)
    checkpoint_hash = hashlib.sha256(checkpoint.read_bytes()).hexdigest()
    memory = [policy.initial_memory(1)[0] for _ in range(2)]; replay = []
    with torch.inference_mode():
        for _ in range(100):
            observations = env.observations(); decisions = []
            for side, active in enumerate(env.active_players()):
                if not active:
                    decisions.append(None); continue
                output = policy.act({k: torch.as_tensor(v)[None] for k, v in observations[side].items()}, memory[side][None], deterministic=True)
                memory[side] = output.memory[0]
                decisions.append(JointAction(AutoregressiveAction(*output.actions[0].tolist()), int(output.end_freeze_mask[0])))
            transition = step_decisions(env, decisions).result
            replay.append([None if a is None else a.to_dict() for a in decisions])
            if transition.battle_resolved:
                break
        else:
            raise AssertionError('Fixture failed to reach first battle')
    record = decision_record(env, memory, 0, (), {'resetSeed': 609090, 'actions': replay},
        search_seed=609091, validation_seed=609092, truth_seeds=(609093, 609094))
    record['checkpointSha256'] = checkpoint_hash
    panel = tmp_path/'panel'; panel.mkdir()
    (panel/'manifest.json').write_text(json.dumps({'version': MANIFEST_VERSION, 'checkpointSha256': checkpoint_hash,
        'simulatorRevision': SIMULATOR_REVISION, 'states': 1, 'truthRollouts': 2}))
    (panel/'state-0000.json').write_text(json.dumps(record))
    reservoir = tmp_path/'reservoir.json'
    reservoir.write_text(json.dumps({'schemaVersion': RESERVOIR_SCHEMA_VERSION,
        'metadata': {'simulatorRevision': SIMULATOR_REVISION}, 'trajectories': [
            {'id': 'fixture', 'boards': [{'turn': 1, 'team': env.core.previous_opponent_state.state,
                                        'opponentTeam': env.core.previous_agent_state.state}]}]}))
    output = tmp_path/'quality'
    command = [sys.executable, str(Path(__file__).parents[1]/'scripts/evaluate_joint_search_quality.py'), str(checkpoint),
        '--panel', str(panel), '--opponent-reservoir', str(reservoir), '--output-directory', str(output), '--device', 'cpu',
        '--states', '1', '--futures', '2', '--tree-transitions', '10', '--simulations', '2', '--searches-per-turn', '1',
        '--engine-match-tree-budget', '10', '--screening-rollouts', '2', '--validation-rollouts', '2', '--search-horizon', '1', '--diagnostic']
    completed = subprocess.run(command, capture_output=True, text=True, timeout=60)
    assert completed.returncode == 0, completed.stdout+completed.stderr
    result = json.loads((output/'summary.json').read_text())
    assert result['approvedForDistillation'] is False and result['diagnosticOnly'] is True
    assert result['allContinuationsTerminal']
    state = json.loads((output/'state-0000.json').read_text())
    assert state['engine'].get('searches', 0) > 0
    assert {o['mode'] for o in state['outcomes']} == {'root-only', 'engine'}
    assert all(o['terminated'] and o['score'] in (0, .5, 1) for o in state['outcomes'])
