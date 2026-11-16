#!/usr/bin/env python3
"""Read-only exact legacy parity check against a frozen model source file."""
import argparse
import importlib.util
import json
import sys
from pathlib import Path
from dataclasses import fields

import torch

from sapai_gym1.rl.model import EntityTransformerConfig, EntityTransformerPolicy
from sapai_gym1.rl.selfplay_env import ParallelSelfPlayEnv
from sapai_gym1.rl.training import stack_observations
from sapai_gym1.rl.decision_api import decode_decisions


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('checkpoint', type=Path)
    parser.add_argument('frozen_model', type=Path)
    parser.add_argument('--device', default='cpu')
    args = parser.parse_args()
    torch.set_num_threads(1)
    spec = importlib.util.spec_from_file_location('sapai_gym1.rl._frozen_parity_model', args.frozen_model)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    checkpoint = torch.load(args.checkpoint, map_location='cpu', weights_only=False)
    old = module.EntityTransformerPolicy(module.EntityTransformerConfig(**checkpoint['config'])).to(args.device).eval()
    new = EntityTransformerPolicy(EntityTransformerConfig(**checkpoint['config'])).to(args.device).eval()
    old.load_state_dict(checkpoint['model'])
    new.load_state_dict(checkpoint['model'])
    checked = 0
    with torch.inference_mode():
        for seed in (0, 123, 815):
            env = ParallelSelfPlayEnv()
            env.reset(seed=seed)
            memory = old.initial_memory(2, args.device)
            for gold in (10, 3, 1, 0):
                env.core.agent.gold = env.core.opponent.gold = gold
                env.core._agent_actions_cache = env.core._opponent_actions_cache = None
                obs = stack_observations(env.observations(), args.device)
                baseline = old(obs, memory, deterministic=True, allow_end_freeze=new.config.end_freeze)
                features = new.encode_decision(obs, memory)
                decisions, actual = decode_decisions(new, features, deterministic=True)
                _, scored = decode_decisions(new, features, decisions=decisions)
                for field in fields(baseline):
                    left, right = getattr(baseline, field.name), getattr(actual, field.name)
                    if left is not None:
                        torch.testing.assert_close(left, right, rtol=0, atol=0, msg=field.name)
                    else:
                        assert right is None
                torch.testing.assert_close(scored.log_prob, baseline.log_prob, rtol=0, atol=0)
                memory = baseline.memory
                checked += 2
    print(json.dumps({'checkpoint': str(args.checkpoint), 'device': args.device,
                      'states': checked, 'parameters': new.parameter_count(),
                      'all_policy_output_fields_bit_exact': True,
                      'joint_rescore_bit_exact': True}, indent=2))


if __name__ == '__main__':
    main()
