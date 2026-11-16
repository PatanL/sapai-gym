#!/usr/bin/env python3
"""Track how a real policy uses an opening freeze, without supplying action labels."""
import argparse
from collections import Counter
import json
from pathlib import Path

import torch

from generate_historical_search_manifest import load_policy
from sapai_gym1.rl.action_schema import ActionType, AutoregressiveAction
from sapai_gym1.rl.historical_evaluation import HistoricalOpponentEvaluator, restore_historical_decision


def summarize(records):
    result = {}
    for name in sorted({record['name'] for record in records}):
        group = [record for record in records if record['name'] == name]
        result[name] = {
            'futures': len(group),
            'originalShopCopyPurchased': sum(record['originalPurchased'] for record in group),
            'originalShopCopyUnfrozen': sum(record['originalUnfrozen'] for record in group),
            'anyTargetCopyPurchased': sum(record['targetPurchases'] > 0 for record in group),
            'levelTwoReachedByTurn': dict(sorted(Counter(
                str(record['firstLevelTwoTurn']) if record['firstLevelTwoTurn'] is not None else 'not_by_prefix_end'
                for record in group
            ).items())),
            'targetPurchaseCount': sum(record['targetPurchases'] for record in group),
            'targetSaleCount': sum(record['targetSales'] for record in group),
        }
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('manifest_dir', type=Path)
    parser.add_argument('--comparison-row', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--last-turn', type=int, default=3)
    args = parser.parse_args()
    if not 2 <= args.last_turn <= 6:
        raise ValueError('Opening follow-through must end between turns two and six')
    torch.set_num_threads(1)
    meta = json.loads((args.manifest_dir / 'manifest.json').read_text())
    row = json.loads(args.comparison_row.read_text())
    record = json.loads((args.manifest_dir / f'state-{row["state"]:04d}.json').read_text())
    policy, main_hash, _ = load_policy(Path(meta['checkpoint']), torch.device('cpu'))
    opponent, opponent_hash, _ = load_policy(Path(meta['opponentCheckpoint']), torch.device('cpu'), allow_older_revision=True)
    initial, memories, side, _ = restore_historical_decision(record, main_hash, opponent_hash)
    raw = AutoregressiveAction(ActionType[row['rawAction']['type']],
                               row['rawAction']['source'], row['rawAction']['target'])
    freeze = AutoregressiveAction(ActionType[row['root']['actionDetail']['type']],
                                  row['root']['actionDetail']['source'], row['root']['actionDetail']['target'])
    if freeze.action_type != ActionType.FREEZE or record['turn'] != 1:
        raise ValueError('This diagnostic requires a turn-one freeze proposal')
    if row['truthSeeds'] != record['truthSeeds'][:row['truthRollouts']]:
        raise ValueError('Comparison futures differ from the frozen input')
    chooser = HistoricalOpponentEvaluator(policy, opponent, acting_side=side, device='cpu', batch_size=64)
    records = []
    for name, action in (('raw', raw), ('freeze', freeze)):
        for seed in row['truthSeeds']:
            env = initial.clone()
            env.redeterminize(seed)
            player = env.core.agent if side == 0 else env.core.opponent
            original = player.shop[freeze.source].obj
            records.append({
                'name': name, 'seed': seed, 'env': env, 'memories': [m.clone() for m in memories],
                'root': action, 'forced': False, 'original': original, 'species': original.name,
                'originalPurchased': False, 'originalUnfrozen': False,
                'targetPurchases': 0, 'targetSales': 0, 'firstLevelTwoTurn': None,
                'ends': [], 'steps': 0,
                'turnTwoOriginalPresent': None, 'turnTwoOriginalFrozen': None,
            })
    while True:
        active = [r for r in records if r['env'].core.turn <= args.last_turn and not r['env'].core.is_done()]
        if not active:
            break
        requests = []
        for index, item in enumerate(active):
            if item['steps'] >= 500:
                raise RuntimeError('Opening prefix exceeded its step safeguard')
            if item['env'].core.turn == 2 and item['turnTwoOriginalPresent'] is None:
                player = item['env'].core.agent if side == 0 else item['env'].core.opponent
                present = [slot for slot in player.shop if slot.obj is item['original']]
                item['turnTwoOriginalPresent'] = bool(present)
                item['turnTwoOriginalFrozen'] = any(slot.frozen for slot in present)
            obs = item['env'].observations()
            for seat, enabled in enumerate(item['env'].active_players()):
                if enabled:
                    requests.append((index, seat, obs[seat], item['memories'][seat]))
        choices = chooser._choose_many(requests)
        paired_actions = [[None, None] for _ in active]
        for request, (action, next_memory) in zip(requests, choices):
            index, seat, _, _ = request
            item = active[index]
            if seat == side and not item['forced']:
                action, item['forced'] = item['root'], True
            if seat == side:
                player = item['env'].core.agent if side == 0 else item['env'].core.opponent
                if action.action_type in (ActionType.BUY_PET, ActionType.UNFREEZE):
                    pet = player.shop[action.source].obj
                    if action.action_type == ActionType.BUY_PET and pet.name == item['species']:
                        item['targetPurchases'] += 1
                        item['originalPurchased'] |= pet is item['original']
                    if action.action_type == ActionType.UNFREEZE and pet is item['original']:
                        item['originalUnfrozen'] = True
                if action.action_type == ActionType.SELL and player.team[action.source].pet.name == item['species']:
                    item['targetSales'] += 1
                if action.action_type == ActionType.END_TURN:
                    item['ends'].append({'turn': item['env'].core.turn,
                                         'levels': [[int(slot.pet.level), int(slot.pet.experience)] for slot in player.team
                                                    if not slot.empty and slot.pet.name == item['species']]})
            paired_actions[index][seat] = action
            item['memories'][seat] = next_memory
        for item, actions in zip(active, paired_actions):
            turn = item['env'].core.turn
            item['env'].step(*actions)
            item['steps'] += 1
            player = item['env'].core.agent if side == 0 else item['env'].core.opponent
            if item['firstLevelTwoTurn'] is None and any(
                not slot.empty and slot.pet.name == item['species'] and slot.pet.level >= 2 for slot in player.team
            ):
                item['firstLevelTwoTurn'] = int(turn)
    public_records = [{key: value for key, value in item.items()
                       if key not in {'env', 'memories', 'root', 'original'}} for item in records]
    report = {'state': record['state'], 'lastTurn': args.last_turn,
              'interpretation': 'Diagnostic follow-through on previously used futures, not new win-rate evidence or training labels.',
              'summary': summarize(records), 'records': public_records}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_suffix('.json.tmp')
    temporary.write_text(json.dumps(report, indent=2) + '\n')
    temporary.replace(args.output)
    print(json.dumps(report['summary']), flush=True)


if __name__ == '__main__':
    main()
