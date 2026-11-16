#!/usr/bin/env python3
"""Frozen three-way forecast experiment, separate from policy-strength training.

Training selects epochs on validation NLL. A distinct --stage test invocation
opens the final test split once, after every model checkpoint is fixed.
"""
import argparse
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import time

import numpy as np
import torch
from sapai.data import data
from sapai_gym1.rl.entity_observation import EntityObservationEncoder
from sapai_gym1.rl.opponent_distribution import OpponentDistributionConfig, AutoregressiveOpponentBoard, MarginalOpponentBoard
from sapai_gym1.rl.opponent_forecast_data import DATASET_SCHEMA, freeze_match_splits
from sapai_gym1.rl.opponent_forecast_quality import VARIANTS, prepare_rows, select_context, evaluate_forecast, paired_match_interval


def sha(path):
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def write_new(path, payload):
    with path.open('x') as stream:
        json.dump(payload, stream, indent=2); stream.write('\n')


def build_model(config, variant, seed):
    torch.manual_seed(seed)
    cls = AutoregressiveOpponentBoard if variant == 'causal_autoregressive' else MarginalOpponentBoard
    return cls(config)


def synchronize(device):
    if device.type == 'cuda':
        torch.cuda.synchronize(device)


def validate_splits(dataset, diagnostic):
    metadata = dataset['metadata']
    if metadata['schema'] != DATASET_SCHEMA:
        raise ValueError('Expected a frozen whole-match dataset')
    if not diagnostic and (metadata.get('diagnosticOnly') or any(not dataset[s] for s in ('train', 'validation', 'test'))):
        raise ValueError('Pilot/empty splits cannot qualify a forecasting experiment')
    assignments = {}
    for split in ('train', 'validation', 'test'):
        for row in dataset[split]:
            key = row['metadata']['collection_id'], row['metadata']['match_id']
            if key in assignments and assignments[key] != split:
                raise ValueError('Match/perspective leakage between frozen splits')
            assignments[key] = split
    expected = {tuple(a['match']): a['split'] for a in metadata['assignments']}
    if assignments != expected:
        raise ValueError('Frozen split manifest differs from rows')
    all_rows = [r for s in ('train', 'validation', 'test') for r in dataset[s]]
    checked = freeze_match_splits(all_rows, heldout_populations=metadata['heldoutPopulations'],
        validation_fraction=metadata['validationFraction'], test_fraction=metadata['testFraction'],
        salt=metadata['salt'], require_nonempty=not diagnostic)
    if checked['metadata']['rejectedRows'] or any(checked[s] != dataset[s] for s in ('train', 'validation', 'test')):
        raise ValueError('Invalid/inconsistent complete-match dataset')
    if not diagnostic and len(assignments) < 200:
        raise ValueError('Predeclared first quality corpus requires >=200 complete matches; use diagnostic mode for plumbing only')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('dataset', type=Path)
    parser.add_argument('--output-directory', type=Path, required=True)
    parser.add_argument('--stage', choices=('train', 'test'), required=True)
    parser.add_argument('--device', default='cpu', choices=('cpu', 'cuda'))
    parser.add_argument('--gpu-boundary-approved', action='store_true')
    parser.add_argument('--epochs', type=int, default=30)
    parser.add_argument('--batch-size', type=int, default=128)
    parser.add_argument('--samples', type=int, default=128)
    parser.add_argument('--learning-rate', type=float, default=3e-4)
    parser.add_argument('--seed', type=int, default=609071)
    parser.add_argument('--diagnostic', action='store_true')
    args = parser.parse_args()
    if args.device == 'cuda' and not args.gpu_boundary_approved:
        raise ValueError('GPU forecast training requires a separately allocated post-entropy boundary')
    if min(args.epochs, args.batch_size) < 1 or args.samples < 2 or args.learning_rate <= 0:
        raise ValueError('Invalid experiment budget')
    torch.set_num_threads(1)
    device = torch.device(args.device)
    if device.type == 'cuda' and not torch.cuda.is_available():
        raise RuntimeError('CUDA requested but unavailable; no silent fallback')
    dataset = json.loads(args.dataset.read_text()); validate_splits(dataset, args.diagnostic)
    encoder = EntityObservationEncoder()
    config = OpponentDistributionConfig(('none', *sorted(k for k in data['pets'] if k != 'pet-none')), tuple(encoder.status_to_id))
    experiment = {'schema': 'forecast-quality-abc-v1-20260906', 'datasetSHA256': sha(args.dataset),
                  'modelConfig': asdict(config), 'seed': args.seed, 'epochs': args.epochs,
                  'batchSize': args.batch_size, 'learningRate': args.learning_rate, 'samples': args.samples,
                  'device': args.device, 'variants': VARIANTS, 'diagnosticOnly': args.diagnostic,
                  'policyTrainingActivated': False, 'runnerSHA256': sha(Path(__file__)),
                  'modelSourceSHA256': sha(Path(__file__).parents[1]/'sapai_gym1/rl/opponent_distribution.py'),
                  'metricSourceSHA256': sha(Path(__file__).parents[1]/'sapai_gym1/rl/opponent_forecast_quality.py')}
    root = args.output_directory
    if args.stage == 'train':
        root.mkdir(parents=True, exist_ok=False)
        write_new(root/'experiment.json', experiment)
        for variant in VARIANTS:
            model = build_model(config, variant, args.seed).to(device)
            optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)
            context, target = prepare_rows(dataset['train'], model, variant)
            validation_context, validation_target = prepare_rows(dataset['validation'], model, variant)
            history = []; best_nll = float('inf'); started = time.perf_counter(); valid_time = 0.; decisions = 0
            if device.type == 'cuda':
                torch.cuda.reset_peak_memory_stats(device)
            for epoch in range(args.epochs):
                model.train()
                # Identical row order per epoch for all three comparisons.
                order = np.random.default_rng(args.seed+epoch).permutation(len(target))
                for start in range(0, len(order), args.batch_size):
                    ids = order[start:start+args.batch_size]
                    optimizer.zero_grad(set_to_none=True)
                    loss = -model.log_prob(select_context(context, ids).to(device), target[ids].to(device)).mean()
                    if not torch.isfinite(loss):
                        raise RuntimeError(f'Nonfinite training loss: {variant}/{epoch}')
                    loss.backward()
                    norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 5., error_if_nonfinite=True)
                    optimizer.step(); decisions += len(ids)
                synchronize(device); valid_start = time.perf_counter(); model.eval(); nlls = []
                with torch.inference_mode():
                    for start in range(0, len(validation_target), args.batch_size):
                        sl = slice(start, start+args.batch_size)
                        nlls.extend((-model.log_prob(select_context(validation_context, sl).to(device), validation_target[sl].to(device))).cpu().tolist())
                value = float(np.mean(nlls)); synchronize(device); valid_time += time.perf_counter()-valid_start
                if not np.isfinite(value):
                    raise RuntimeError('Nonfinite/empty validation split')
                history.append({'epoch': epoch+1, 'validationNll': value, 'lastGradNorm': float(norm)})
                if value < best_nll:
                    best_nll = value
                    torch.save({'state_dict': {k: v.detach().cpu().clone() for k, v in model.state_dict().items()},
                                'config': asdict(config), 'variant': variant, 'epoch': epoch+1,
                                'datasetSHA256': experiment['datasetSHA256']}, root/f'{variant}.pt')
            synchronize(device)
            write_new(root/f'{variant}-training.json', {'history': history, 'examplesProcessed': decisions,
                'wallSeconds': time.perf_counter()-started, 'validationSeconds': valid_time,
                'parameterCount': sum(p.numel() for p in model.parameters()), 'bestValidationNll': best_nll,
                'checkpointSHA256': sha(root/f'{variant}.pt'),
                'peakAllocatedCudaBytes': torch.cuda.max_memory_allocated(device) if device.type == 'cuda' else None})
        write_new(root/'training-complete.json', {'datasetSHA256': experiment['datasetSHA256'],
            'checkpoints': {v: sha(root/f'{v}.pt') for v in VARIANTS}, 'finalTestOpened': False})
        return
    original = json.loads((root/'experiment.json').read_text())
    if original != json.loads(json.dumps(experiment)):
        raise ValueError('Final-test experiment/source/config changed since training')
    completed = json.loads((root/'training-complete.json').read_text())
    # A durable one-shot opening record is created BEFORE evaluation. Failures
    # require an explicit new review, never unplanned repeated test tuning.
    write_new(root/'final-test-opened.json', {'experiment': original, 'checkpoints': completed['checkpoints']})
    metrics = {}
    for variant in VARIANTS:
        if sha(root/f'{variant}.pt') != completed['checkpoints'][variant]:
            raise ValueError('Selected forecast checkpoint changed')
        model = build_model(config, variant, args.seed).to(device)
        model.load_state_dict(torch.load(root/f'{variant}.pt', map_location='cpu', weights_only=False)['state_dict'])
        synchronize(device); started = time.perf_counter()
        metrics[variant] = evaluate_forecast(model, dataset['test'], variant, batch_size=min(32, args.batch_size), samples=args.samples)
        synchronize(device)
        metrics[variant]['evaluationWallSeconds'] = time.perf_counter()-started
        metrics[variant]['parameterCount'] = sum(p.numel() for p in model.parameters())
        write_new(root/f'{variant}-test.json', metrics[variant])
    comparisons = {name: paired_match_interval(dataset['test'], metrics[first]['nllPerRow'], metrics[second]['nllPerRow'])
                   for name, first, second in [('publicContext', 'history_marginal', 'causal_marginal'),
                        ('jointDecoder', 'causal_marginal', 'causal_autoregressive')]}
    write_new(root/'quality-comparison.json', {'comparisons': comparisons, 'policyActivationAuthorized': False,
        'strengthValidated': False, 'diagnosticOnly': args.diagnostic, 'selectionAutomatic': False,
        'interpretation': 'Review held-out NLL, population/event calibration, validity limitations and cost together; sample aesthetics are not evidence.'})


if __name__ == '__main__':
    main()
