#!/usr/bin/env python3
"""Test a stated opening continuation plan on fresh interactive futures."""
import argparse
import json
from pathlib import Path
import time

import numpy as np
import torch

from generate_historical_search_manifest import load_policy
from benchmark_cached_tree_execution import effect, grouped
from sapai_gym1.rl.action_schema import ActionType, AutoregressiveAction
from sapai_gym1.rl.deferred_levelup_diagnostic import DeferredLevelUpDiagnostic
from sapai_gym1.rl.historical_evaluation import HistoricalOpponentEvaluator, restore_historical_decision


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('manifest_dir', type=Path)
    parser.add_argument('--comparison-row', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--futures', type=int, default=64)
    parser.add_argument('--seed', type=int, default=4309001)
    args = parser.parse_args()
    if args.futures < 2:
        raise ValueError('At least two matched futures are required')
    torch.set_num_threads(1)
    manifest = json.loads((args.manifest_dir / 'manifest.json').read_text())
    row = json.loads(args.comparison_row.read_text())
    record = json.loads((args.manifest_dir / f'state-{row["state"]:04d}.json').read_text())
    policy, main_hash, _ = load_policy(Path(manifest['checkpoint']), torch.device('cpu'))
    opponent, opponent_hash, _ = load_policy(Path(manifest['opponentCheckpoint']), torch.device('cpu'), allow_older_revision=True)
    env, memories, side, _ = restore_historical_decision(record, main_hash, opponent_hash)
    raw_info, freeze_info = row['rawAction'], row['root']['actionDetail']
    raw = AutoregressiveAction(ActionType[raw_info['type']], raw_info['source'], raw_info['target'])
    freeze = AutoregressiveAction(ActionType[freeze_info['type']], freeze_info['source'], freeze_info['target'])
    if record['turn'] != 1 or freeze.action_type != ActionType.FREEZE:
        raise ValueError('Expected an opening freeze proposal')
    player = env.core.agent if side == 0 else env.core.opponent
    species = player.shop[freeze.source].obj.name
    seeds = list(map(int, np.random.default_rng(args.seed).integers(0, 2**63 - 1, args.futures)))
    if set(seeds) & set(record['truthSeeds']):
        raise ValueError('Fresh diagnostic futures overlap the previous comparison')
    design = {'state': record['state'], 'checkpointSha256': main_hash,
              'opponentCheckpointSha256': opponent_hash, 'species': species, 'futures': seeds,
              'intervention': 'Freeze the offered copy; prevent this species reaching level two by buying/combining on turn two only.',
              'purpose': 'Expert-suggested continuation diagnosis, not a permanent rule or training label.'}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.with_suffix('.design.json').write_text(json.dumps(design, indent=2) + '\n')
    options = dict(acting_side=side, device='cpu', batch_size=64, horizon_battles=30, max_steps=2500)
    natural = HistoricalOpponentEvaluator(policy, opponent, **options)
    delayed = DeferredLevelUpDiagnostic(policy, opponent,
                                        species_id=env.encoder.item_to_id[species], **options)
    started = time.perf_counter()
    natural_results = natural.evaluate(env, memories, side, [raw, freeze], seeds)
    delayed_results = delayed.evaluate(env, memories, side, [freeze], seeds)
    complete = all(result.terminated for result in (*natural_results, *delayed_results))
    returns = grouped(natural_results, 'total_return')
    delayed_returns = grouped(delayed_results, 'total_return')[freeze]
    scores = grouped(natural_results, 'match_score') if complete else None
    delayed_scores = grouped(delayed_results, 'match_score')[freeze] if complete else None
    report = {
        'design': design, 'completeMatches': complete, 'wallSeconds': time.perf_counter() - started,
        'interventionCounts': dict(delayed.counters),
        'naturalFreezeReturnVsRaw': effect(returns[freeze] - returns[raw]),
        'delayedFreezeReturnVsRaw': effect(delayed_returns - returns[raw]),
        'delayedVsNaturalFreezeReturn': effect(delayed_returns - returns[freeze]),
        'naturalFreezeMatchVsRaw': effect(scores[freeze] - scores[raw]) if complete else None,
        'delayedFreezeMatchVsRaw': effect(delayed_scores - scores[raw]) if complete else None,
        'delayedVsNaturalFreezeMatch': effect(delayed_scores - scores[freeze]) if complete else None,
    }
    temporary = args.output.with_suffix('.json.tmp')
    temporary.write_text(json.dumps(report, indent=2) + '\n')
    temporary.replace(args.output)
    print(json.dumps({key: value for key, value in report.items() if key not in {'design'}}), flush=True)


if __name__ == '__main__':
    main()
