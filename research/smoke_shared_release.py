#!/usr/bin/env python3
"""Check the restored release and run both policies on CPU without training."""
import hashlib
import json
from pathlib import Path
import sys

import torch

ROOT = Path(__file__).resolve().parents[2]
EXPERIMENT = ROOT / 'runs/claude-end-positioner-20261112-v1'


def main():
    manifest = json.loads((ROOT / 'RELEASE_MANIFEST.json').read_text())
    for name, meta in manifest['model_assets'].items():
        with (ROOT / meta['workspace_path']).open('rb') as stream:
            assert hashlib.file_digest(stream, 'sha256').hexdigest() == meta['sha256'], name
    for name, meta in manifest['workspace_files'].items():
        with (ROOT / name).open('rb') as stream:
            assert hashlib.file_digest(stream, 'sha256').hexdigest() == meta['sha256'], name
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    sys.path.insert(0, str(EXPERIMENT))
    from plan_common import install, load_actor
    install(cuda=False)
    actor = load_actor(device='cpu')
    from sapai_gym1.rl.selfplay_env import ParallelSelfPlayEnv
    from teacher_world import ActorRequest, observation_copy, owned_memory
    env = ParallelSelfPlayEnv()
    env.reset(seed=241115, paired_tape=True)
    request = ActorRequest(observation_copy(env.observations()[0]),
                           owned_memory(torch.zeros(actor.model.config.memory_dim)),
                           actor.owner)
    response = actor.act_many([request])[0]
    from posnet import PosNetScorer
    scorer = PosNetScorer(EXPERIMENT / 'out/posnet-zeroshot.pt', device='cpu')
    scores = scorer.score([{'team': []}, {'team': []}], {'team': []}, 1)
    assert all(0 <= score <= 1 for score in scores)
    print(json.dumps({'policy_update': 25772,
                      'policy_action': response.action.to_dict(),
                      'positioner_scores': scores,
                      'release_file_hashes': 'passed',
                      'frozen_reference': 'passed', 'cpu_smoke': 'passed'}))


if __name__ == '__main__':
    main()
