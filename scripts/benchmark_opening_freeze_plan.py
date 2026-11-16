#!/usr/bin/env python3
"""Diagnose one-versus-two copy preservation with fixed public-information plans."""
import argparse
from collections import Counter
import json
from pathlib import Path
import time

import numpy as np
import torch

from generate_historical_search_manifest import load_policy
from benchmark_cached_tree_execution import effect
from sapai_gym1.rl import ActionType, AutoregressiveAction, FlatActionCodec
from sapai_gym1.rl.historical_evaluation import HistoricalOpponentEvaluator, restore_historical_decision
from sapai_gym1.rl.opening_freeze_plan import OpeningFreezePlanEvaluator, next_freeze_plan_action


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('manifest_dir', type=Path)
    parser.add_argument('--state', type=int, default=2)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--futures', type=int, default=64)
    parser.add_argument('--seed', type=int, default=4310001)
    args = parser.parse_args()
    if args.futures < 2:
        raise ValueError('At least two futures are required')
    torch.set_num_threads(1)
    meta = json.loads((args.manifest_dir/'manifest.json').read_text())
    record = json.loads((args.manifest_dir/f'state-{args.state:04d}.json').read_text())
    policy, main_hash, _ = load_policy(Path(meta['checkpoint']), torch.device('cpu'))
    opponent, opponent_hash, _ = load_policy(Path(meta['opponentCheckpoint']), torch.device('cpu'), allow_older_revision=True)
    env, memories, side, _ = restore_historical_decision(record, main_hash, opponent_hash)
    player = env.core.agent if side == 0 else env.core.opponent
    if env.core.turn != 1 or player.gold != 0:
        raise ValueError('This diagnostic requires turn one at zero gold')
    owned = Counter()
    for slot in player.team:
        if not slot.empty:
            owned[slot.pet.name] += int(slot.pet.total_experience) + 1
    offers = {}
    legal = env.core._avail_actions(side)
    for index, slot in enumerate(player.shop):
        if (not slot.empty and slot.slot_type in {'pet','levelup'} and not slot.frozen
                and owned[slot.obj.name] == 1
                and FlatActionCodec.encode(AutoregressiveAction(ActionType.FREEZE,index)) in legal):
            offers.setdefault(slot.obj.name, []).append(index)
    names = sorted(name for name, indices in offers.items() if len(indices) >= 2)
    if not names:
        raise ValueError('No eligible two-copy public opportunity')
    species = names[0]
    positions = offers[species][:2]
    initial_mask = [bool(slot.frozen) for slot in player.shop]
    masks = {'none': initial_mask, 'one': initial_mask.copy(), 'both': initial_mask.copy()}
    masks['one'][positions[0]] = True
    for index in positions:
        masks['both'][index] = True
    obs = env.observations()[side]
    ids = obs['item_ids'][6:6+len(player.shop)]
    seeds = list(map(int, np.random.default_rng(args.seed).integers(0, 2**63-1, args.futures)))
    if set(seeds) & set(record['truthSeeds']):
        raise ValueError('Fresh plan futures overlap the earlier panel')
    options = dict(acting_side=side, device='cpu', batch_size=64, horizon_battles=30, max_steps=2500)
    raw = HistoricalOpponentEvaluator(policy, opponent, **options)._choose_many([(0,side,obs,memories[side])])[0][0]
    if raw.action_type != ActionType.END_TURN:
        raise ValueError('No-freeze control differs from the raw greedy action')
    design = {'state': args.state, 'species': species, 'positions': positions,
              'checkpointSha256': main_hash, 'opponentCheckpointSha256': opponent_hash,
              'futures': seeds, 'freezeMasks': masks,
              'purpose': 'Fixed-plan diagnostic selected from public information; no permanent strategy rule or training label.'}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.with_suffix('.design.json').write_text(json.dumps(design,indent=2)+'\n')
    results, telemetry = {}, {}
    started = time.perf_counter()
    for name, mask in masks.items():
        evaluator = OpeningFreezePlanEvaluator(policy, opponent, item_ids=ids, desired_frozen=mask, **options)
        first = next_freeze_plan_action(obs, ids, mask)
        results[name] = evaluator.evaluate(env, memories, side, [first], seeds)
        telemetry[name] = dict(evaluator.counters)
        print(f'completed plan={name}', flush=True)
    natural = HistoricalOpponentEvaluator(policy, opponent, **options)
    first = next_freeze_plan_action(obs, ids, masks['one'])
    results['natural_one'] = natural.evaluate(env, memories, side, [first], seeds)
    complete = all(result.terminated for group in results.values() for result in group)
    returns = {name: np.array([r.total_return for r in group]) for name,group in results.items()}
    scores = {name: np.array([r.match_score for r in group]) for name,group in results.items()} if complete else None
    report = {'design': design, 'completeMatches': complete, 'wallSeconds': time.perf_counter()-started,
              'planTelemetry': telemetry, 'plans': {}}
    for name in results:
        report['plans'][name] = {'returnVsNone': effect(returns[name]-returns['none']),
                                 'matchVsNone': effect(scores[name]-scores['none']) if complete else None}
    report['bothVsOne'] = effect(scores['both']-scores['one']) if complete else None
    report['bothVsNaturalOne'] = effect(scores['both']-scores['natural_one']) if complete else None
    temp = args.output.with_suffix('.json.tmp')
    temp.write_text(json.dumps(report,indent=2)+'\n')
    temp.replace(args.output)
    print(json.dumps({key:value for key,value in report.items() if key!='design'}), flush=True)


if __name__ == '__main__':
    main()
