#!/usr/bin/env python3
"""Audit actual joint generator outputs against their immutable teacher."""
import argparse
import json
from pathlib import Path

import numpy as np
import torch
from generate_search_demonstrations import load_policy
from sapai_gym1.rl.joint_demonstrations import sha, restore_prefix, reconstruct_memory
from sapai_gym1.rl.joint_search_records import rescore_record


def main():
    parser = argparse.ArgumentParser(); parser.add_argument('checkpoint', type=Path)
    parser.add_argument('corpus', type=Path); parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise ValueError('Choose a new audit output')
    torch.set_num_threads(1)
    policy, _ = load_policy(args.checkpoint, torch.device('cpu'), False)
    owner = sha(args.checkpoint); rows = []
    for line in args.corpus.read_text().splitlines():
        if not line.strip():
            continue
        record = json.loads(line)
        logps, _ = rescore_record(record, policy, checkpoint_sha256=owner)
        original = record.get('originalJointLogProbabilities')
        if original is None:
            raise ValueError('Missing original joint likelihood')
        error = float(np.max(np.abs(np.asarray(logps)-original)))
        memory_error = None
        if record.get('memoryObservationPrefix') is not None:
            memory = reconstruct_memory(policy, restore_prefix(record, checkpoint_sha256=owner), torch.device('cpu'))
            memory_error = float((memory-torch.tensor(record['memory'])).abs().max())
        # Cross-device BF16 memories need their own parity check; never claim
        # a CPU burn-in is bit-identical to CUDA actor arithmetic.
        same_precision = record.get('memoryActingPrecision', 'cpu-fp32') == 'cpu-fp32'
        if error > 3e-6 or (same_precision and memory_error is not None and memory_error > 3e-6):
            raise ValueError(f'First divergent joint rescore: logp={error}, memory={memory_error}')
        rows.append({'id': record['metadata'].get('replay_id'), 'maximumLogpError': error,
                     'maximumMemoryError': memory_error, 'sameMemoryPrecision': same_precision})
    if not rows:
        raise ValueError('Cannot validate an empty corpus')
    result = {'passed': True, 'rows': rows, 'checkpointSHA256': owner, 'corpusSHA256': sha(args.corpus),
              'productionTrainingAuthorized': False, 'device': 'cpu-fp32', 'tolerance': 3e-6}
    args.output.write_text(json.dumps(result, indent=2)+'\n'); print(json.dumps(result))


if __name__ == '__main__':
    main()
