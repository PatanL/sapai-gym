#!/usr/bin/env python3
"""Freeze main-vs-historical decisions with explicit per-seat policy lineage."""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import torch

from search_state_mining import (
    GOLD_BANDS, MINING_VERSION, OPENING_SELECTION_VERSION,
    mine_match_opportunity, opening_freeze_reasons,
)
from sapai.turtle_pack import SIMULATOR_REVISION
from sapai_gym1.rl import EntityTransformerConfig, EntityTransformerPolicy
from sapai_gym1.rl.decision_manifest import MANIFEST_VERSION, decision_record
from sapai_gym1.rl.historical_evaluation import POLICY_OWNERSHIP_VERSION, restore_historical_decision


def load_policy(path, device, *, allow_older_revision=False):
    payload = torch.load(path, map_location='cpu', weights_only=False)
    revision = payload.get('simulator_revision')
    if revision != SIMULATOR_REVISION and not allow_older_revision:
        raise ValueError(f'Checkpoint simulator revision mismatch: {path}')
    policy = EntityTransformerPolicy(EntityTransformerConfig(**payload['config'])).to(device).eval()
    policy.load_state_dict(payload['model'])
    return policy, hashlib.sha256(Path(path).read_bytes()).hexdigest(), revision


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('checkpoint', type=Path)
    parser.add_argument('--opponent-checkpoint', type=Path, required=True)
    parser.add_argument('--allow-older-opponent-revision', action='store_true')
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--states', type=int, default=16)
    parser.add_argument('--truth-rollouts', type=int, default=32)
    parser.add_argument('--seed', type=int, default=4005701)
    parser.add_argument('--threads', type=int, default=1)
    parser.add_argument('--opening-freeze-only', action='store_true')
    args = parser.parse_args()
    if args.states < 1 or args.truth_rollouts < 2 or args.threads < 1:
        raise SystemExit('Invalid count')
    torch.set_num_threads(args.threads)
    device = torch.device('cpu')
    policy, main_hash, _ = load_policy(args.checkpoint, device)
    opponent, opponent_hash, opponent_revision = load_policy(
        args.opponent_checkpoint, device, allow_older_revision=args.allow_older_opponent_revision
    )
    metadata = {
        'version': MANIFEST_VERSION, 'policyOwnershipVersion': POLICY_OWNERSHIP_VERSION,
        'miningVersion': MINING_VERSION, 'simulatorRevision': SIMULATOR_REVISION,
        'checkpoint': str(args.checkpoint), 'checkpointSha256': main_hash,
        'opponentCheckpoint': str(args.opponent_checkpoint), 'opponentCheckpointSha256': opponent_hash,
        'opponentTrainingSimulatorRevision': opponent_revision,
        'allowOlderOpponentRevision': args.allow_older_opponent_revision,
        'seed': args.seed, 'states': args.states, 'truthRollouts': args.truth_rollouts,
        'purpose': 'historical-interactive-evaluation-inputs-only; not training labels',
    }
    if args.opening_freeze_only:
        metadata['selectionVersion'] = OPENING_SELECTION_VERSION
    args.output_dir.mkdir(parents=True, exist_ok=True)
    config = args.output_dir / 'manifest.json'
    if config.exists() and json.loads(config.read_text()) != metadata:
        raise ValueError('Refusing mixed historical manifest configurations')
    temporary = config.with_suffix('.json.tmp')
    temporary.write_text(json.dumps(metadata, indent=2) + '\n')
    temporary.replace(config)
    windows = [(2, 4), (5, 8), (9, 12), (13, 16)]
    for index in range(args.states):
        path = args.output_dir / f'state-{index:04d}.json'
        if path.exists():
            restore_historical_decision(json.loads(path.read_text()), main_hash, opponent_hash)
            print(f'state={index} restored', flush=True)
            continue
        local_seed = args.seed + index * 1009
        torch.manual_seed(local_seed)
        rng = np.random.default_rng(local_seed)
        window = (index // len(GOLD_BANDS)) % len(windows)
        band_index = index % len(GOLD_BANDS)
        band = GOLD_BANDS[band_index]
        # Balance seats within every turn window and every gold band over 16 roots.
        side = (window + band_index) % 2
        minimum, maximum = windows[window]
        opportunity_filter = None
        if args.opening_freeze_only:
            minimum, maximum, band, side = 1, 1, 'zero', index % 2
            family = ('opening-levelup-copy-available', 'opening-two-shop-copies-available')[(index // 2) % 2]
            opportunity_filter = lambda core, side: tuple(reason for reason in opening_freeze_reasons(core, side) if reason == family)
        selected = None
        for attempt in range(512 if args.opening_freeze_only else 64):
            selected = mine_match_opportunity(
                policy, rng, device, minimum, maximum, band,
                opponent_policy=opponent, acting_side=side, with_replay=True,
                opportunity_filter=opportunity_filter,
            )
            if selected is not None:
                break
        if selected is None:
            raise RuntimeError(f'No historical decision for state {index}')
        env, memories, selected_side, reasons, replay = selected
        assert selected_side == side
        search_seed, validation_seed = map(int, rng.integers(0, 2**63 - 1, size=2))
        record = decision_record(
            env, memories, side, reasons, replay,
            search_seed=search_seed, validation_seed=validation_seed,
            truth_seeds=rng.integers(0, 2**63 - 1, size=args.truth_rollouts),
        )
        owners = [opponent_hash, opponent_hash]
        owners[side] = main_hash
        record.update({
            'state': index, 'checkpointSha256': main_hash,
            'opponentCheckpointSha256': opponent_hash,
            'policyOwnershipVersion': POLICY_OWNERSHIP_VERSION, 'policySha256BySeat': owners,
            'miningVersion': MINING_VERSION, 'goldBand': band,
            'turnWindow': [minimum, maximum], 'attempts': attempt + 1,
        })
        if args.opening_freeze_only:
            record['selectionVersion'] = OPENING_SELECTION_VERSION
            record['openingPriorSeed'] = int(rng.integers(0, 2**63 - 1))
        restore_historical_decision(record, main_hash, opponent_hash)
        temporary = path.with_suffix('.json.tmp')
        temporary.write_text(json.dumps(record, separators=(',', ':')) + '\n')
        temporary.replace(path)
        print(f'state={index} side={side} turn={record["turn"]} gold={record["gold"]} band={band}', flush=True)


if __name__ == '__main__':
    main()
