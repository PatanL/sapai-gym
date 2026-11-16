#!/usr/bin/env python3
"""Fixed-trajectory equivalence and timing against a frozen learner module."""
import argparse
import copy
from dataclasses import asdict
import importlib.util
import json
from pathlib import Path
import statistics
import sys
import time

import torch

from sapai_gym1.rl import EntityTransformerConfig, EntityTransformerPolicy
from sapai_gym1.rl.training import BatchedSelfPlayCollector, learner_update


def load_reference(path):
    name = 'sapai_gym1.rl._frozen_batch_benchmark'
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def model_optimizer(saved, device):
    model = EntityTransformerPolicy(EntityTransformerConfig(**saved['config'])).to(device)
    model.load_state_dict(saved['model'])
    history = ('opponent_history_', 'opponent_pet_head.', 'opponent_status_head.',
               'opponent_level_head.', 'opponent_stats_head.', 'opponent_temporary_head.')
    named = list(model.named_parameters())
    by_name = {
        'base': [p for n,p in named if not n.startswith(history) and not n.startswith(('opponent_belief_adapter.', 'end_freeze_head.'))],
        'opponent_history': [p for n,p in named if n.startswith(history)],
        'opponent_adapter': [p for n,p in named if n.startswith('opponent_belief_adapter.')],
        'end_freeze': [p for n,p in named if n.startswith('end_freeze_head.')],
    }
    old_groups = saved['optimizer']['param_groups']
    if len(old_groups) == 1 and 'name' not in old_groups[0]:
        groups = [{'params': list(model.parameters())}]
    else:
        groups = [{'params': by_name[group['name']], 'name': group['name']} for group in old_groups]
    assert [len(g['params']) for g in groups] == [len(g['params']) for g in old_groups]
    optimizer = torch.optim.AdamW(groups)
    # load_state_dict can otherwise alias CPU Adam step tensors between copies.
    optimizer.load_state_dict(copy.deepcopy(saved['optimizer']))
    return model, optimizer


def compare_state(first, second, first_opt, second_opt):
    maxima = {'parameters': 0., 'gradients': 0., 'optimizer': 0.}
    exact = True
    for a,b in zip(first.parameters(), second.parameters()):
        maxima['parameters'] = max(maxima['parameters'], float((a-b).abs().max()))
        exact &= torch.equal(a,b)
        if a.grad is not None:
            assert b.grad is not None
            maxima['gradients'] = max(maxima['gradients'], float((a.grad-b.grad).abs().max()))
            exact &= torch.equal(a.grad,b.grad)
        else:
            assert b.grad is None
        old, new = first_opt.state.get(a, {}), second_opt.state.get(b, {})
        assert old.keys() == new.keys()
        for key in old:
            if torch.is_tensor(old[key]):
                maxima['optimizer'] = max(maxima['optimizer'], float((old[key]-new[key]).abs().max()))
                exact &= torch.equal(old[key],new[key])
            else:
                assert old[key] == new[key]
    return maxima, bool(exact)


def capture_inputs(model, target):
    original = model.evaluate_sequence
    def wrapped(observation, initial_memory, actions, **kwargs):
        record = {f'observation:{key}': value.detach().cpu().clone() for key,value in observation.items()}
        record.update(memory=initial_memory.detach().cpu().clone(), actions=actions.detach().cpu().clone())
        record.update({key: value.detach().cpu().clone() for key,value in kwargs.items()})
        target.append(record)
        return original(observation, initial_memory, actions, **kwargs)
    model.evaluate_sequence = wrapped
    return original


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('mode', choices=('generate', 'compare'))
    parser.add_argument('--checkpoint', type=Path, required=True)
    parser.add_argument('--data', type=Path, required=True)
    parser.add_argument('--reference', type=Path)
    parser.add_argument('--output', type=Path)
    parser.add_argument('--transitions', type=int, default=8192)
    parser.add_argument('--repeats', type=int, default=4)
    parser.add_argument('--device', default='cuda')
    args = parser.parse_args()
    torch.set_num_threads(1)
    saved = torch.load(args.checkpoint, map_location='cpu', weights_only=False)
    if args.mode == 'generate':
        policy = EntityTransformerPolicy(EntityTransformerConfig(**saved['config'])).eval()
        policy.load_state_dict(saved['model'])
        collector = BatchedSelfPlayCollector(policy, 8, 'cpu', seed=4600101, discount_mode='per_turn')
        segments, count = [], 0
        while count < args.transitions:
            added = collector.collect(64)
            segments.extend(added)
            count += sum(len(segment.transitions) for segment in added)
        args.data.parent.mkdir(parents=True, exist_ok=True)
        temporary = args.data.with_suffix('.tmp')
        torch.save({'segments': segments, 'checkpoint': str(args.checkpoint), 'transitions': count}, temporary)
        temporary.replace(args.data)
        print(json.dumps({'segments': len(segments), 'transitions': count, 'data': str(args.data)}))
        return

    reference = load_reference(args.reference)
    payload = torch.load(args.data, map_location='cpu', weights_only=False)
    if payload['checkpoint'] != str(args.checkpoint):
        raise ValueError('Benchmark trajectories and policy checkpoint must have matching recurrent ownership')
    segments = payload['segments']
    old, old_opt = model_optimizer(saved, args.device)
    new, new_opt = model_optimizer(saved, args.device)
    old_inputs, new_inputs = [], []
    old_original = capture_inputs(old, old_inputs)
    new_original = capture_inputs(new, new_inputs)
    old_metrics = reference.learner_update(old, old_opt, segments, args.device)
    new_metrics = learner_update(new, new_opt, segments, args.device)
    old.evaluate_sequence, new.evaluate_sequence = old_original, new_original
    assert len(old_inputs) == len(new_inputs)
    for before, after in zip(old_inputs,new_inputs):
        assert before.keys() == after.keys()
        for key in before:
            assert torch.equal(before[key],after[key]), key
    maxima, exact = compare_state(old,new,old_opt,new_opt)
    comparisons = [{'legacy': asdict(old_metrics), 'batched': asdict(new_metrics)}]
    timings = {'legacy': [], 'batched': []}
    for repeat in range(args.repeats):
        results = {}
        ordered = [('legacy', reference.learner_update, old, old_opt), ('batched', learner_update, new, new_opt)]
        if repeat % 2:
            ordered.reverse()
        for name, function, model, optimizer in ordered:
            if args.device == 'cuda': torch.cuda.synchronize()
            start = time.perf_counter()
            metrics = function(model, optimizer, segments, args.device)
            if args.device == 'cuda': torch.cuda.synchronize()
            timings[name].append(time.perf_counter()-start)
            results[name] = asdict(metrics)
        comparisons.append(results)
    final_maxima, final_exact = compare_state(old,new,old_opt,new_opt)
    loss_error = max(abs(row['legacy']['total_loss']-row['batched']['total_loss']) for row in comparisons)
    passed = loss_error <= 1e-6 and all(value <= 1e-6 for value in final_maxima.values())
    report = {'checkpoint': str(args.checkpoint), 'data': str(args.data), 'device': args.device,
              'transitions': payload['transitions'], 'segments': len(segments), 'inputsBitExact': True,
              'warmupStateBitExact': exact, 'finalStateBitExact': final_exact,
              'warmupMaximumDifferences': maxima, 'finalMaximumDifferences': final_maxima,
              'maximumLossDifference': loss_error, 'numericalGatePassed': passed,
              'seconds': timings, 'learnerSpeedup': statistics.median(timings['legacy'])/statistics.median(timings['batched']),
              'metrics': comparisons, 'weightsUsedForTraining': False}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps({k:v for k,v in report.items() if k != 'metrics'}, indent=2))
    if not passed:
        raise SystemExit('Numerical equivalence gate failed')


if __name__ == '__main__':
    main()
