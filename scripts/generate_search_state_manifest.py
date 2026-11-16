#!/usr/bin/env python3
"""Freeze replayable states before any planner sees validation outcomes."""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import torch

from search_state_mining import GOLD_BANDS, MINING_VERSION, mine_match_opportunity
from sapai.turtle_pack import SIMULATOR_REVISION
from sapai_gym1.rl import EntityTransformerConfig, EntityTransformerPolicy
from sapai_gym1.rl.decision_manifest import MANIFEST_VERSION, decision_record, restore_decision


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('checkpoint', type=Path)
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--states', type=int, default=16)
    parser.add_argument('--truth-rollouts', type=int, default=32)
    parser.add_argument('--seed', type=int, default=4005001)
    parser.add_argument('--threads', type=int, default=1)
    parser.add_argument('--device', default='cpu')
    args = parser.parse_args()
    if args.states < 1 or args.truth_rollouts < 2 or args.threads < 1:
        raise SystemExit('Invalid state, rollout or thread count')
    torch.set_num_threads(args.threads)
    device = torch.device(args.device)
    payload = torch.load(args.checkpoint, map_location='cpu', weights_only=False)
    if payload.get('simulator_revision') != SIMULATOR_REVISION:
        raise SystemExit('Checkpoint simulator revision mismatch')
    policy = EntityTransformerPolicy(EntityTransformerConfig(**payload['config'])).to(device).eval()
    policy.load_state_dict(payload['model'])
    checkpoint_hash = hashlib.sha256(args.checkpoint.read_bytes()).hexdigest()
    metadata = {
        'version': MANIFEST_VERSION,
        'miningVersion': MINING_VERSION,
        'simulatorRevision': SIMULATOR_REVISION,
        'checkpoint': str(args.checkpoint),
        'checkpointSha256': checkpoint_hash,
        'seed': args.seed,
        'states': args.states,
        'truthRollouts': args.truth_rollouts,
        'device': str(device),
        'purpose': 'evaluation-inputs-only; not demonstration labels',
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    config = args.output_dir / 'manifest.json'
    if config.exists():
        if json.loads(config.read_text()) != metadata:
            raise SystemExit('Refusing to mix different manifest configurations')
    else:
        config.write_text(json.dumps(metadata, indent=2) + '\n')
    windows = [(2, 4), (5, 8), (9, 12), (13, 16)]
    for index in range(args.states):
        state_path = args.output_dir / f'state-{index:04d}.json'
        if state_path.exists():
            record = json.loads(state_path.read_text())
            if record['checkpointSha256'] != checkpoint_hash:
                raise SystemExit('Existing state checkpoint mismatch')
            restore_decision(record)
            print(f'state={index} restored', flush=True)
            continue
        local_seed = args.seed + index * 1009
        torch.manual_seed(local_seed)
        rng = np.random.default_rng(local_seed)
        band = GOLD_BANDS[index % len(GOLD_BANDS)]
        minimum, maximum = windows[(index // len(GOLD_BANDS)) % len(windows)]
        selected = None
        for attempt in range(64):
            selected = mine_match_opportunity(
                policy, rng, device, minimum, maximum, band, with_replay=True
            )
            if selected is not None:
                break
        if selected is None:
            raise RuntimeError(f'No {band} state in turns {minimum}-{maximum} after 64 games')
        env, memories, side, reasons, replay = selected
        search_seed, validation_seed = map(int, rng.integers(0, 2**63 - 1, size=2))
        truth_seeds = rng.integers(0, 2**63 - 1, size=args.truth_rollouts)
        record = decision_record(
            env, memories, side, reasons, replay,
            search_seed=search_seed, validation_seed=validation_seed,
            truth_seeds=truth_seeds,
        )
        record.update({
            'state': index, 'checkpointSha256': checkpoint_hash,
            'miningVersion': MINING_VERSION, 'goldBand': band,
            'turnWindow': [minimum, maximum], 'attempts': attempt + 1,
        })
        # Fail before accepting a state if deterministic replay does not
        # recreate both information sets, legal masks and shop histories.
        restore_decision(record)
        temporary = state_path.with_suffix('.json.tmp')
        temporary.write_text(json.dumps(record, separators=(',', ':')) + '\n')
        temporary.replace(state_path)
        print(f'state={index} turn={record["turn"]} gold={record["gold"]} band={band} reasons={reasons}', flush=True)


if __name__ == '__main__':
    main()
