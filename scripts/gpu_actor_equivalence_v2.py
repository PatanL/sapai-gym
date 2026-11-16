#!/usr/bin/env python3
"""Frozen-production -> new full -> lean actor CUDA and RNG equivalence."""
import argparse
from dataclasses import fields
import hashlib
import importlib
import json
from pathlib import Path
import sys
import types

import torch
from sapai import Team
from sapai_gym1.rl.model import EntityTransformerPolicy, EntityTransformerConfig
from sapai_gym1.rl.selfplay_env import ParallelSelfPlayEnv


def load_production_package(directory):
    name = 'sapai_gym1.rl._frozen_production_v2'
    package = types.ModuleType(name)
    package.__path__ = [str(directory)]
    sys.modules[name] = package
    return importlib.import_module(name+'.model'), importlib.import_module(name+'.training')


def sha(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def observations(batch_size, case):
    env = ParallelSelfPlayEnv()
    env.reset(seed=123 + case)
    core = env.core
    core.agent.team = Team(['pet-fish', 'pet-ant', 'pet-cricket'])
    core.opponent.team = Team(['pet-pig', 'pet-fish'])
    core.agent._connect_objects()
    core.opponent._connect_objects()
    gold, cap, history = [(10,0,False),(0,0,False),(0,49,True),(0,50,True),(3,50,True),(1,0,True)][case]
    core.agent.gold = core.opponent.gold = gold
    core.shop_action_count_agent = core.shop_action_count_opponent = cap
    if history:
        core.agent.turn = core.opponent.turn = 5
        core.previous_opponent_state = Team(['pet-fish', 'pet-pig'])
        core.previous_agent_state = Team(['pet-cricket', 'pet-ant'])
        core.previous_previous_opponent_state = Team(['pet-fish'])
        core.previous_previous_agent_state = Team(['pet-ant'])
    core._agent_actions_cache = core._opponent_actions_cache = None
    rows = env.observations()
    return {k: torch.stack([torch.as_tensor(rows[i % 2][k]) for i in range(batch_size)]).cuda() for k in rows[0]}


def rng_state():
    return torch.get_rng_state().clone(), [state.clone() for state in torch.cuda.get_rng_state_all()]


def restore_rng(state):
    torch.set_rng_state(state[0])
    torch.cuda.set_rng_state_all(state[1])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', nargs='+', type=Path, required=True)
    parser.add_argument('--production-rl', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    report = {'version':'actor-gpu-chain-v2','passed':False,'comparisons':0,'zeroMasks':0,'nonzeroMasks':0,
              'checkpoints':{},'productionModelSHA256':sha(args.production_rl/'model.py'), 'runnerSHA256':sha(__file__)}
    try:
        if args.output.exists():
            raise FileExistsError('Never overwrite actor-equivalence evidence')
        torch.set_num_threads(1)
        if not torch.cuda.is_available():
            raise RuntimeError('CUDA required; no CPU fallback')
        production, _ = load_production_package(args.production_rl)
        for checkpoint in args.checkpoint:
            report['checkpoints'][str(checkpoint)] = sha(checkpoint)
            saved = torch.load(checkpoint, map_location='cpu', weights_only=False)
            torch.manual_seed(400)
            old = production.EntityTransformerPolicy(production.EntityTransformerConfig(**saved['config'])).cuda().eval()
            torch.manual_seed(400)
            new = EntityTransformerPolicy(EntityTransformerConfig(**saved['config'])).cuda().eval()
            old.load_state_dict(saved['model']); new.load_state_dict(saved['model'])
            assert new.config.entropy_mode == 'legacy_path'
            assert old.state_dict().keys() == new.state_dict().keys()
            for key in old.state_dict():
                torch.testing.assert_close(old.state_dict()[key], new.state_dict()[key], rtol=0, atol=0, msg=key)
            joint = bool(new.config.end_freeze)
            for bf16 in (False, True):
                for batch in (1, 4, 12, 32):
                    for case in range(6):
                        obs = observations(batch, case)
                        memory = new.initial_memory(batch, 'cuda')
                        for deterministic in (False, True):
                            torch.manual_seed(4031 + case)
                            state = rng_state()
                            with torch.inference_mode(), torch.autocast('cuda', dtype=torch.bfloat16, enabled=bf16):
                                baseline = old(obs, memory, deterministic=deterministic, allow_end_freeze=joint)
                                baseline_rng = rng_state()
                                restore_rng(state)
                                torch.testing.assert_close(new.value_only(obs, memory), baseline.value, rtol=0, atol=0,
                                                           msg=f'{checkpoint.name}/{bf16}/{batch}/{case}/value-only')
                                calls = (
                                    lambda: new(obs, memory, deterministic=deterministic, allow_end_freeze=joint),
                                    lambda: new.act(obs, memory, deterministic=deterministic),
                                    lambda: new.sample_joint_action(new.encode_decision(obs, memory, forecast=False), deterministic=deterministic),
                                )
                                for name, call in zip(('new_full','act','sample_joint_action'), calls):
                                    restore_rng(state)
                                    result = call()
                                    for key in ('actions','log_prob','memory','end_freeze_mask'):
                                        left, right = getattr(baseline,key), getattr(result,key)
                                        if left is None:
                                            assert right is None
                                        else:
                                            torch.testing.assert_close(left,right,rtol=0,atol=0,msg=f'{checkpoint.name}/{bf16}/{batch}/{case}/{name}/{key}')
                                    now = rng_state()
                                    assert torch.equal(now[0],baseline_rng[0])
                                    assert all(torch.equal(a,b) for a,b in zip(now[1],baseline_rng[1]))
                                    report['comparisons'] += batch
                            if baseline.end_freeze_mask is not None:
                                report['zeroMasks'] += int((baseline.end_freeze_mask == 0).sum())
                                report['nonzeroMasks'] += int((baseline.end_freeze_mask != 0).sum())
            del old, new
        if not report['zeroMasks'] or not report['nonzeroMasks']:
            raise RuntimeError('Insufficient mask coverage; add an explicitly labeled stress fixture before accepting')
        report['passed'] = True
    except BaseException as error:
        report['firstFailure'] = str(error)
        raise
    finally:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report,indent=2)+'\n')
        print(json.dumps(report,indent=2),flush=True)


if __name__ == '__main__':
    main()
