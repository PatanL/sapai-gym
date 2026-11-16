"""Standalone forecast quality metrics. No policy or search authority implied."""
from collections import defaultdict
import math

import numpy as np
import torch

from .opponent_forecast_data import PublicContextBatch, collate_contexts, collate_targets

VARIANTS = ('history_marginal', 'causal_marginal', 'causal_autoregressive')
EVENTS = ('shark_fly', 'elephant_camel', 'elephant_blowfish', 'mushroom_deer',
          'scorpion', 'level2_dragon', 'level2_monkey', 'two_tall_pets')


def prepare_rows(rows, model, variant):
    if variant not in VARIANTS:
        raise ValueError('Unknown forecast ablation')
    context = collate_contexts([r['context'] for r in rows], model.species_ids, model.perk_ids)
    if variant == 'history_marginal':
        context = context.history_only()
    targets = collate_targets([r['target'] for r in rows], model.species_ids, model.perk_ids)
    return context, targets


def select_context(batch, indices):
    return PublicContextBatch(*(getattr(batch, k)[indices] for k in batch.__dataclass_fields__))


def board_events(labels, config):
    """Verified board properties, not preferred SAP strategies or training labels."""
    species = {name: i for i, name in enumerate(config.species)}
    perks = {name: i for i, name in enumerate(config.perks)}
    ids, progress = labels[..., 0], labels[..., 1]
    def has(name):
        return (ids == species.get(name, -1)).any(-1)
    def leveled(name):
        return ((ids == species.get(name, -1)) & (progress >= 2)).any(-1)
    mushroom = ((ids == species.get('pet-deer', -1)) & (labels[..., 2] == perks.get('status-extra-life', -1))).any(-1)
    # Perk vocabulary differs between historical data packages: use only the
    # actual mushroom status; fail visibly in the experiment if absent.
    return torch.stack((has('pet-shark') & has('pet-fly'), has('pet-elephant') & has('pet-camel'),
                        has('pet-elephant') & has('pet-blowfish'), mushroom, has('pet-scorpion'),
                        leveled('pet-dragon'), leveled('pet-monkey'),
                        (((labels[..., 3] >= 25) & (labels[..., 4]+1 >= 25)) & (ids != 0)).sum(-1) >= 2), -1)


def calibration(probabilities, outcomes, bins=10):
    probabilities, outcomes = np.asarray(probabilities, dtype=float), np.asarray(outcomes, dtype=float)
    if (probabilities.shape != outcomes.shape or not np.isfinite(probabilities).all()
            or not np.isfinite(outcomes).all() or np.any((probabilities < 0) | (probabilities > 1))
            or np.any((outcomes != 0) & (outcomes != 1))):
        raise ValueError('Invalid calibration inputs')
    index = np.minimum((probabilities*bins).astype(int), bins-1)
    rows, weighted = [], 0.
    for b in range(bins):
        mask = index == b
        if not mask.any():
            continue
        p, y = float(probabilities[mask].mean()), float(outcomes[mask].mean())
        rows.append({'bin': b, 'count': int(mask.sum()), 'meanProbability': p, 'eventFrequency': y})
        weighted += abs(p-y)*mask.sum()
    return {'brier': float(np.mean((probabilities-outcomes)**2)),
            'ece': float(weighted/max(1, probabilities.size)), 'bins': rows,
            'positiveCount': int(outcomes.sum()), 'negativeCount': int(outcomes.size-outcomes.sum()),
            'meanProbability': float(probabilities.mean()), 'eventFrequency': float(outcomes.mean())}


def paired_match_interval(rows, first, second, *, seed=609061, repeats=2000):
    """Positive = first NLL minus second NLL (second forecasts better).

    Cluster by whole match, including both perspectives. Bootstrap uncertainty
    is over matches, not over a much larger fictitious independent row sample.
    """
    grouped = defaultdict(list)
    for row, a, b in zip(rows, first, second, strict=True):
        meta = row['metadata']; grouped[meta['collection_id'], meta['match_id']].append(float(a-b))
    values = np.array([np.mean(x) for _, x in sorted(grouped.items())])
    if not len(values) or not np.isfinite(values).all():
        raise ValueError('Missing/nonfinite match comparisons')
    interval = None
    if len(values) >= 2:
        rng = np.random.default_rng(seed)
        estimates = values[rng.integers(0, len(values), size=(repeats, len(values)))].mean(1)
        interval = np.quantile(estimates, [.025, .975]).tolist()
    return {'matches': len(values), 'meanMatchNllImprovement': float(values.mean()),
            'bootstrapMatch95CI': interval, 'bootstrapReplicates': repeats, 'seed': seed}


@torch.inference_mode()
def evaluate_forecast(model, rows, variant, *, batch_size=64, samples=128, seed=609062):
    if not rows or samples < 2 or batch_size < 1:
        raise ValueError('Evaluation requires rows and a declared positive sampling budget')
    model.eval()
    device = next(model.parameters()).device
    cpu_context, cpu_targets = prepare_rows(rows, model, variant)
    nlls, true_events, probabilities, correct, confidence, actual = [], [], [], defaultdict(list), defaultdict(list), defaultdict(list)
    invalid = 0; total_boards = 0
    # Evaluation RNG is isolated from training; identical sampling seed/budget
    # across models does not claim identical sampled stochastic boards.
    with torch.random.fork_rng(devices=[device.index or 0] if device.type == 'cuda' else []):
        torch.manual_seed(seed)
        for start in range(0, len(rows), batch_size):
            context = select_context(cpu_context, slice(start, start+batch_size)).to(device)
            targets = cpu_targets[start:start+batch_size].to(device)
            encoded = model.encode_context(context)
            nll = -model.decode(encoded, targets)[1]
            nlls.extend(nll.cpu().tolist())
            sampled, _ = model.decode(encoded.repeat_interleave(samples, 0))
            draws = sampled.reshape(len(targets), samples, 5, 6)
            probabilities.extend(board_events(sampled, model.config).float().reshape(len(targets), samples, -1).mean(1).cpu().tolist())
            true_events.extend(board_events(targets, model.config).cpu().tolist())
            occupied = sampled[..., 0] != 0
            # These checks measure syntax/range validity only. Unknown shops
            # mean exact reachability is NOT identifiable from public history.
            invalid += int(((sampled[..., 0] < 0) | (sampled[..., 0] >= len(model.config.species))
                            | ((sampled[..., 1] < 0) | (sampled[..., 1] >= 6)) & occupied
                            | ((sampled[..., 2] < 0) | (sampled[..., 2] >= len(model.config.perks))) & occupied
                            | ((sampled[..., 3] < 0) | (sampled[..., 3] > 50)) & occupied
                            | ((sampled[..., 4] < 0) | (sampled[..., 4] >= 50)) & occupied).any(-1).sum().cpu())
            total_boards += len(sampled)
            for name, field, size in [('species', 0, len(model.config.species)), ('perk', 2, len(model.config.perks)), ('level_progress', 1, 6)]:
                # For perk/level, sample joint P(occupied, field) rather than
                # silently replacing empty samples with a no-perk prediction.
                values = draws[..., field]
                if field != 0:
                    values = torch.where(draws[..., 0] != 0, values, size)
                    size += 1
                marginal = torch.nn.functional.one_hot(values, size).float().mean(1)
                conf, prediction = marginal.max(-1)
                truth = targets[..., field]
                if field != 0:
                    truth = torch.where(targets[..., 0] != 0, truth, size-1)
                matched = prediction == truth
                correct[name].extend(matched.cpu().flatten().tolist())
                confidence[name].extend(conf.cpu().flatten().tolist())
                actual[name].extend(matched.cpu().flatten().tolist())
    nlls = np.asarray(nlls)
    if not np.isfinite(nlls).all():
        raise ValueError('Nonfinite forecast NLL; quality gate failed')
    by_turn, by_population = defaultdict(list), defaultdict(list)
    for row, nll in zip(rows, nlls):
        by_turn[str(row['metadata']['turn'])].append(float(nll))
        by_population[row['metadata']['opponent_population']].append(float(nll))
    event_p, event_y = np.asarray(probabilities), np.asarray(true_events)
    return {'rows': len(rows), 'negativeLogLikelihoodPerBoard': float(nlls.mean()),
            'nllPerRow': nlls.tolist(), 'eventProbabilities': event_p.tolist(), 'eventOutcomes': event_y.tolist(),
            'events': {event: calibration(event_p[:, i], event_y[:, i]) for i, event in enumerate(EVENTS)},
            'secondaryAccuracy': {name: float(np.mean(values)) for name, values in correct.items()},
            'secondaryCategoricalCalibration': {name: calibration(confidence[name], actual[name]) for name in confidence},
            'nllByTurn': {key: {'rows': len(v), 'nll': float(np.mean(v))} for key, v in by_turn.items()},
            'nllByOpponentPopulation': {key: {'rows': len(v), 'nll': float(np.mean(v))} for key, v in by_population.items()},
            'syntacticallyInvalidBoardRate': invalid/max(1, total_boards), 'generatedBoards': total_boards,
            'exactHiddenShopReachabilityRate': None,
            'reachabilityLimitation': 'Syntax is checked, but hidden shops/actions prevent certifying complete reachability. No zero-impossibility claim.',
            'marginalMetricMethod': f'Monte Carlo with {samples} board samples/context; NLL is exact teacher-forced joint likelihood',
            'mushroomStatusPresent': 'status-extra-life' in model.config.perks}
