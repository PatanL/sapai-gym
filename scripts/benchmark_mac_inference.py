#!/usr/bin/env python3
"""Actual-host FP32 policy latency; no simulated Mac numbers or hidden GPU fallback.

Run separately with --backend cpu and --backend mps on the Mac. Measures whole
state reconstruction + encoding + decision + validation, not browser/OCR time.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import platform
import resource
import time

import numpy as np
import torch

from sapai_gym1.deployment.policy_adapter import LivePolicyEngine
from sapai_gym1.deployment.state_schema import LiveGameState


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('checkpoint', type=Path)
    parser.add_argument('states', nargs='+', type=Path)
    parser.add_argument('--backend', choices=('cpu', 'mps'), default='cpu')
    parser.add_argument('--threads', type=int, default=2)
    parser.add_argument('--repeats', type=int, default=50)
    parser.add_argument('--warmup', type=int, default=5)
    args = parser.parse_args()
    if args.repeats <= 0 or args.warmup < 0:
        parser.error('Positive repeats and nonnegative warmup required')
    if args.backend == 'mps' and os.environ.get('PYTORCH_ENABLE_MPS_FALLBACK') == '1':
        parser.error('Disable PYTORCH_ENABLE_MPS_FALLBACK so unsupported operators remain visible')
    states = [LiveGameState.from_dict(json.loads(p.read_text())) for p in args.states]
    start = time.perf_counter()
    engine = LivePolicyEngine(args.checkpoint, threads=args.threads, backend=args.backend)
    load_seconds = time.perf_counter() - start
    timings = []
    actions = []
    for index in range(args.warmup + args.repeats):
        # Independent positions/cold memory, not cached identical-UI retries.
        engine.reset_memory()
        start = time.perf_counter()
        recommendation = engine.recommend(states[index % len(states)])
        if args.backend == 'mps':
            torch.mps.synchronize()
        elapsed = time.perf_counter() - start
        if index >= args.warmup:
            timings.append(elapsed)
            actions.append(recommendation.decision.to_dict())
    peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    peak_bytes = peak if platform.system() == 'Darwin' else peak * 1024
    print(json.dumps({
        'platform': platform.platform(), 'processor': platform.processor(),
        'torch': torch.__version__, 'backend': args.backend, 'dtype': 'float32',
        'threads': args.threads, 'loadSeconds': load_seconds,
        'medianMilliseconds': float(np.median(timings) * 1000),
        'p95Milliseconds': float(np.quantile(timings, .95) * 1000),
        'maxMilliseconds': float(max(timings) * 1000),
        'peakProcessRSSBytes': peak_bytes, 'swapMeasured': False,
        'checkpointSHA256': hashlib.file_digest(args.checkpoint.open('rb'), 'sha256').hexdigest(),
        'states': [str(p) for p in args.states], 'decisions': actions,
        'scope': 'Cold-memory policy decisions; excludes browser/OCR and search',
    }, indent=2))


if __name__ == '__main__':
    main()
