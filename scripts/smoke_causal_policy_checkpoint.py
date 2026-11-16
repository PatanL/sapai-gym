#!/usr/bin/env python3
"""Production-weight CPU neutrality gate; changes no saved training weights."""
import argparse
import copy
import hashlib
import json
from pathlib import Path

import torch
from sapai import Team
from sapai.data import data
from generate_search_demonstrations import load_policy
from sapai_gym1.rl.selfplay_env import ParallelSelfPlayEnv
from sapai_gym1.rl.opponent_context_data import PublicOpponentContext
from sapai_gym1.rl.opponent_distribution import OpponentDistributionConfig, AutoregressiveOpponentBoard
from sapai_gym1.rl.opponent_policy_adapter import CausalOpponentPolicy


def main():
    parser = argparse.ArgumentParser(); parser.add_argument('checkpoint', type=Path); parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--reference-rl', type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise ValueError('Preserve previous neutrality reports')
    torch.set_num_threads(1)
    parent, _ = load_policy(args.checkpoint, torch.device('cpu'), False)
    from gpu_actor_equivalence_v2 import load_production_package
    reference_module, _ = load_production_package(args.reference_rl)
    payload = torch.load(args.checkpoint, map_location='cpu', weights_only=False)
    reference = reference_module.EntityTransformerPolicy(reference_module.EntityTransformerConfig(**payload['config'])).eval()
    reference.load_state_dict(payload['model'])
    env = ParallelSelfPlayEnv(); env.reset(seed=609082)
    forecast_config = OpponentDistributionConfig(('none', *sorted(k for k in data['pets'] if k != 'pet-none')), tuple(env.encoder.status_to_id))
    wrapper = CausalOpponentPolicy(copy.deepcopy(parent), AutoregressiveOpponentBoard(forecast_config)).eval()
    checks = 0
    for populated in (False, True):
        if populated:
            env.core.previous_opponent_state = Team(['pet-monkey', 'pet-fish'])
            env.core.previous_previous_opponent_state = Team(['pet-fish'])
            env.core.previous_agent_state = Team(['pet-shark', 'pet-fly'])
        for gold in (0, 3, 10):
            env.core.agent.gold = gold
            observation = env.observations()[0]
            batch = {k: torch.as_tensor(v)[None] for k, v in observation.items()}
            context = [PublicOpponentContext.capture(env.core, 0).features()]
            for stage in ('forecast', 'policy_control', 'policy_residual'):
                wrapper.set_stage(stage)
                memory = parent.initial_memory(1)
                torch.manual_seed(609083); rng = torch.get_rng_state()
                with torch.inference_mode():
                    baseline = reference(batch, memory, allow_end_freeze=parent.config.end_freeze)
                    after_rng = torch.get_rng_state(); torch.set_rng_state(rng)
                    new_base = parent(batch, memory, allow_end_freeze=parent.config.end_freeze)
                    assert torch.equal(after_rng, torch.get_rng_state())
                    torch.set_rng_state(rng)
                    candidate = wrapper(batch, memory, causal_context=context, allow_end_freeze=parent.config.end_freeze)
                assert torch.equal(after_rng, torch.get_rng_state())
                for key, value in vars(baseline).items():
                    if torch.is_tensor(value):
                        torch.testing.assert_close(value, getattr(new_base, key), atol=0, rtol=0, msg='new base: '+key)
                        torch.testing.assert_close(value, getattr(candidate, key), atol=0, rtol=0, msg=key)
                checks += 1
    with args.checkpoint.open('rb') as stream:
        checkpoint_hash = hashlib.file_digest(stream, 'sha256').hexdigest()
    report = {'passed': True, 'device': 'cpu-fp32', 'checkpointSHA256': checkpoint_hash, 'pairedCases': checks,
        'comparisonChain': 'accepted frozen AB model -> current base -> causal wrapper',
        'referenceModelSHA256': hashlib.sha256((args.reference_rl/'model.py').read_bytes()).hexdigest(),
        'allOutputTensorsAndRngBitExact': True, 'oldPolicyParametersUnchanged': all(torch.equal(v, wrapper.base_policy.state_dict()[k]) for k, v in parent.state_dict().items()),
        'newParameters': sum(p.numel() for p in wrapper.forecast_model.parameters())+sum(p.numel() for p in wrapper.projection.parameters()),
        'policyTrainingActivated': False, 'gpuTested': False, 'strengthValidated': False}
    args.output.write_text(json.dumps(report, indent=2)+'\n'); print(json.dumps(report))


if __name__ == '__main__':
    main()
