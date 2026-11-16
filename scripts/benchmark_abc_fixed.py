#!/usr/bin/env python3
"""A-C fixed-real-data gate; no live process, shared optimizer, or new trajectories."""
import argparse
from dataclasses import asdict
import hashlib
import json
import multiprocessing as mp
from pathlib import Path
import statistics
import time

import torch

from benchmark_batch_pipeline import load_reference, model_optimizer
from sapai_gym1.rl.distributed import SharedTrajectoryRing
from sapai_gym1.rl.learner_batch import pack_learner_batch
from sapai_gym1.rl import training
from sapai_gym1.rl import batched_learner as batched_module


def owned_segments(data):
    ctx = mp.get_context('spawn')
    ring = SharedTrajectoryRing.create(data[0].transitions[0].observation, len(data[0].initial_memory),
                                       ctx, capacity=64)
    result = []
    try:
        for start in range(0, len(data), 64):
            chunk = data[start:start+64]
            for segment in chunk:
                ring.write(segment, ctx.Event())
            received = 0
            while received < len(chunk):
                copied = ring.read_dense(len(chunk)-received, 5)
                result.extend(copied.segments())
                received += len(copied.lengths)
    finally:
        ring.close()
    return result


def max_error(a, b):
    return float((a.detach().float() - b.detach().float()).abs().max().cpu())


def assert_state(a, b, opt_a, opt_b, *, atol, rtol):
    errors = dict(parameters=0., gradients=0., optimizer=0.)
    for (name, x), (_, y) in zip(a.named_parameters(), b.named_parameters()):
        errors['parameters'] = max(errors['parameters'], max_error(x, y))
        torch.testing.assert_close(x, y, atol=min(atol, 1e-6), rtol=min(rtol, 1e-5), msg=name)
        if x.grad is not None:
            errors['gradients'] = max(errors['gradients'], max_error(x.grad, y.grad))
            torch.testing.assert_close(x.grad, y.grad, atol=atol, rtol=rtol, msg=f'gradient {name}')
        else:
            assert y.grad is None
        left, right = opt_a.state.get(x, {}), opt_b.state.get(y, {})
        assert left.keys() == right.keys()
        for key in left:
            if torch.is_tensor(left[key]):
                assert left[key].data_ptr() != right[key].data_ptr(), 'Optimizer tensors alias'
                errors['optimizer'] = max(errors['optimizer'], max_error(left[key], right[key]))
                torch.testing.assert_close(left[key], right[key], atol=atol, rtol=rtol, msg=f'{name} Adam {key}')
    return errors


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', type=Path, required=True)
    parser.add_argument('--data', type=Path, required=True)
    parser.add_argument('--reference', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--precision', choices=('fp32', 'bf16'), default='bf16')
    parser.add_argument('--warmup', type=int, default=5)
    parser.add_argument('--repeats', type=int, default=10)
    parser.add_argument('--entropy-mode', choices=('legacy_path', 'exact_joint'), default='legacy_path')
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError('Never overwrite an earlier fixed-data gate')
    torch.set_num_threads(1)
    payload = torch.load(args.data, map_location='cpu', weights_only=False)
    if payload['checkpoint'] != str(args.checkpoint):
        raise ValueError('Recurrent-state ownership does not match checkpoint')
    saved = torch.load(args.checkpoint, map_location='cpu', weights_only=False)
    data = payload['segments']
    dense = owned_segments(data)
    joint = bool(saved['config'].get('end_freeze', False))
    # Compare actual packed input fields/metadata, not just a checksum.
    for start in range(0, len(data), 64):
        old_pack = pack_learner_batch(data[start:start+64], 'cpu', joint_actions=joint, include_metadata=True)
        new_pack = pack_learner_batch(dense[start:start+64], 'cpu', joint_actions=joint)
        for key in vars(old_pack):
            x, y = getattr(old_pack, key), getattr(new_pack, key)
            if isinstance(x, dict):
                for k in x:
                    assert torch.equal(x[k], y[k]), f'{key}:{k}'
            elif x is not None:
                assert torch.equal(x, y), key
            else:
                assert y is None
    reference = load_reference(args.reference)
    old, opt_a = model_optimizer(saved, args.device)
    new, opt_b = model_optimizer(saved, args.device)
    old.config.entropy_mode = new.config.entropy_mode = args.entropy_mode
    bf16 = args.precision == 'bf16'
    # Declared before observing results. FP32 is strict; BF16 permits small
    # reduction/kernel rounding differences, not altered algorithms/targets.
    atol, rtol = (3e-5, 3e-4) if not bf16 else (3e-4, 1e-2)
    old_targets, new_targets = [], []
    original_ref_loss = reference.impala_actor_critic_loss
    original_batch_loss = batched_module.masked_impala_actor_critic_loss
    def capture_old(*a, **kw):
        result = original_ref_loss(*a, **kw)
        old_targets.append((result.vtrace.value_targets.detach().cpu(), result.vtrace.policy_advantages.detach().cpu()))
        return result
    def capture_new(*a, **kw):
        result = original_batch_loss(*a, **kw)
        lengths = a[7].cpu().tolist()
        for i, n in enumerate(lengths):
            new_targets.append((result.vtrace.value_targets[:n, i:i+1].cpu(), result.vtrace.policy_advantages[:n, i:i+1].cpu()))
        return result
    reference.impala_actor_critic_loss = capture_old
    batched_module.masked_impala_actor_critic_loss = capture_new
    def update(label):
        if label == 'baseline':
            return reference.learner_update(old, opt_a, data, args.device, use_bfloat16=bf16)
        return training.learner_update(new, opt_b, dense, args.device, use_bfloat16=bf16,
                                       lean_bootstrap=True, batched_learner=True)
    result = {'checkpoint': str(args.checkpoint), 'data': str(args.data), 'device': args.device,
              'precision': args.precision, 'entropyMode': args.entropy_mode,
              'tolerances': {'lossTargetsGradientsAdamAtol': atol, 'lossTargetsGradientsAdamRtol': rtol,
                             'parameterAtol': min(atol, 1e-6), 'parameterRtol': min(rtol, 1e-5)}, 'inputsBitExact': True,
              'segments': len(data), 'decisions': payload['transitions'],
              'weightsUsedForTraining': False, 'gatePassed': False}
    try:
        a, b = update('baseline'), update('abc')
        reference.impala_actor_critic_loss = original_ref_loss
        batched_module.masked_impala_actor_critic_loss = original_batch_loss
        assert len(old_targets) == len(new_targets)
        target_error = 0.
        for x, y in zip(old_targets, new_targets):
            for left, right in zip(x, y):
                target_error = max(target_error, max_error(left, right))
                torch.testing.assert_close(left, right, atol=atol, rtol=rtol)
        for key, value in asdict(a).items():
            torch.testing.assert_close(torch.tensor(value), torch.tensor(asdict(b)[key]), atol=atol, rtol=rtol, msg=key)
        result['initialStateErrors'] = assert_state(old, new, opt_a, opt_b, atol=atol, rtol=rtol)
        result['maximumTargetError'] = target_error
        result['initialMetrics'] = {'baseline': asdict(a), 'abc': asdict(b)}
        timings = {'baseline': [], 'abc': []}
        for iteration in range(args.warmup + args.repeats):
            for label in (('baseline', 'abc') if iteration % 2 == 0 else ('abc', 'baseline')):
                if args.device == 'cuda':
                    torch.cuda.synchronize()
                start = time.perf_counter()
                update(label)
                if args.device == 'cuda':
                    torch.cuda.synchronize()
                if iteration >= args.warmup:
                    timings[label].append(time.perf_counter()-start)
        result['finalStateErrors'] = assert_state(old, new, opt_a, opt_b, atol=atol, rtol=rtol)
        result['seconds'] = timings
        result['learnerSpeedup'] = statistics.median(timings['baseline']) / statistics.median(timings['abc'])
        result['gatePassed'] = True
    except BaseException as error:
        result['error'] = str(error)
        raise
    finally:
        reference.impala_actor_critic_loss = original_ref_loss
        batched_module.masked_impala_actor_critic_loss = original_batch_loss
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(result, indent=2) + '\n')
        print(json.dumps(result, indent=2), flush=True)


if __name__ == '__main__':
    main()
