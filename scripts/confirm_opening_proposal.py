#!/usr/bin/env python3
"""Independently recheck one already-fixed proposal, without selecting a winner."""
import argparse
import hashlib
import json
import math
from pathlib import Path
import time

import numpy as np
import torch

from generate_historical_search_manifest import load_policy
from benchmark_cached_tree_execution import effect, grouped
from sapai_gym1.rl.action_schema import ActionType, AutoregressiveAction
from sapai_gym1.rl.historical_evaluation import HistoricalOpponentEvaluator, restore_historical_decision
from sapai.turtle_pack import SIMULATOR_REVISION


def action_from_snapshot(snapshot):
    return AutoregressiveAction(ActionType[snapshot['type']], snapshot['source'], snapshot['target'])


def exact_paired_pvalue(differences):
    """One-sided exact test on discordant binary win/loss outcomes."""
    values = list(map(float, differences))
    if any(value not in (-1., 0., 1.) for value in values):
        return None
    positive = sum(value == 1. for value in values)
    negative = sum(value == -1. for value in values)
    discordant = positive + negative
    if not discordant:
        return 1.
    return sum(math.comb(discordant, k) for k in range(positive, discordant + 1)) / (1 << discordant)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('manifest_dir', type=Path)
    parser.add_argument('--proposal-row', type=Path, required=True)
    parser.add_argument('--planner', choices=('root','hybrid'), default='root')
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--futures', type=int, default=128)
    parser.add_argument('--seed', type=int, default=4309201)
    parser.add_argument('--alpha', type=float, default=0.025)
    args = parser.parse_args()
    if args.futures < 2:
        raise ValueError('At least two confirmation futures are required')
    if not 0 < args.alpha < 1:
        raise ValueError('alpha must be between zero and one')
    torch.set_num_threads(1)
    manifest = json.loads((args.manifest_dir / 'manifest.json').read_text())
    row = json.loads(args.proposal_row.read_text())
    record_path = args.manifest_dir / f'state-{row["state"]:04d}.json'
    record = json.loads(record_path.read_text())
    policy, main_hash, _ = load_policy(Path(manifest['checkpoint']), torch.device('cpu'))
    opponent, opponent_hash, _ = load_policy(Path(manifest['opponentCheckpoint']), torch.device('cpu'), allow_older_revision=True)
    env, memories, side, _ = restore_historical_decision(record, main_hash, opponent_hash)
    raw = action_from_snapshot(row['rawAction'])
    proposal = action_from_snapshot(row[args.planner]['actionDetail'])
    seeds = list(map(int, np.random.default_rng(args.seed).integers(0, 2**63 - 1, args.futures)))
    if set(seeds) & (set(record['truthSeeds']) | set(row['truthSeeds'])):
        raise ValueError('Confirmation futures must not overlap earlier selection/evaluation futures')
    design = {
        'state': row['state'], 'simulatorRevision': SIMULATOR_REVISION,
        'checkpointSha256': main_hash, 'opponentCheckpointSha256': opponent_hash,
        'recordSha256': hashlib.sha256(record_path.read_bytes()).hexdigest(),
        'proposalRowSha256': hashlib.sha256(args.proposal_row.read_bytes()).hexdigest(),
        'rawAction': list(raw.as_tuple()), 'fixedProposal': list(proposal.as_tuple()),
        'futures': seeds, 'seed': args.seed,
        'gate': 'exact-one-sided-discordant-binomial', 'alpha': args.alpha,
        'purpose': 'Fixed-proposal independent recheck; no search or post-outcome action selection.',
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    design_path = args.output.with_suffix('.design.json')
    if design_path.exists() and json.loads(design_path.read_text()) != design:
        raise ValueError('Refusing to overwrite another confirmation design')
    design_path.write_text(json.dumps(design, indent=2) + '\n')
    evaluator = HistoricalOpponentEvaluator(policy, opponent, acting_side=side,
                                             device='cpu', batch_size=64,
                                             horizon_battles=30, max_steps=2500)
    started = time.perf_counter()
    results = evaluator.evaluate(env, memories, side, tuple(dict.fromkeys([raw, proposal])), seeds)
    complete = all(result.terminated for result in results)
    returns = grouped(results, 'total_return')
    scores = grouped(results, 'match_score') if complete else None
    match_effect = effect(scores[proposal] - scores[raw]) if complete else None
    paired_p = exact_paired_pvalue(scores[proposal] - scores[raw]) if complete else None
    possible_cap = sum(record['turn'] + result.battles >= env.core.MAX_TURN for result in results)
    report = {
        'design': design, 'completeMatches': complete, 'possibleTurnLimitEndings': possible_cap,
        'wallSeconds': time.perf_counter() - started,
        'returnGain': effect(returns[proposal] - returns[raw]),
        'matchGain': match_effect,
        'rawMeanScore': float(scores[raw].mean()) if complete else None,
        'proposalMeanScore': float(scores[proposal].mean()) if complete else None,
        'positiveLowerBound': bool(complete and match_effect['fixedRootPairedFutureBootstrap95CI'][0] > 0),
        'exactPairedPValue': paired_p,
        'confirmationPassed': bool(complete and possible_cap == 0 and paired_p is not None
                                   and paired_p <= args.alpha and match_effect['mean'] > 0),
        'usedForTraining': False,
        'note': 'Turn-limit count is conservative because a natural death can coincide with the cap. This is one conditional-state result, not a policy win rate.',
    }
    temporary = args.output.with_suffix('.json.tmp')
    temporary.write_text(json.dumps(report, indent=2) + '\n')
    temporary.replace(args.output)
    print(json.dumps({key: value for key, value in report.items() if key != 'design'}), flush=True)


if __name__ == '__main__':
    main()
