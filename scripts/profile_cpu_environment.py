#!/usr/bin/env python3
"""Profile the exact entity self-play environment without model inference."""

from __future__ import annotations

import argparse
import cProfile
import io
import pstats
import time

import numpy as np

from sapai_gym1.rl import FlatActionCodec, ParallelSelfPlayEnv


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--decisions", type=int, default=20_000)
    parser.add_argument("--seed", type=int, default=20260830)
    parser.add_argument("--top", type=int, default=40)
    return parser.parse_args()


def run(decisions: int, seed: int):
    rng = np.random.default_rng(seed)
    env = ParallelSelfPlayEnv(fast_battles=True)
    env.reset(seed=seed)
    matches = battles = actions = 0
    while actions < decisions:
        selected = []
        for side, active in enumerate(env.active_players()):
            if not active:
                selected.append(None)
                continue
            legal = tuple(env.core._avail_actions(side))
            flat = int(legal[int(rng.integers(0, len(legal)))])
            selected.append(FlatActionCodec.decode(flat))
            actions += 1
        step = env.step(selected[0], selected[1])
        battles += int(step.battle_resolved)
        if step.terminated:
            matches += 1
            env.reset(seed=int(rng.integers(0, 2**31 - 1)))
    return {"actions": actions, "battles": battles, "matches": matches}


def main():
    args = parse_args()
    profiler = cProfile.Profile()
    started = time.perf_counter()
    profiler.enable()
    counts = run(args.decisions, args.seed)
    profiler.disable()
    elapsed = time.perf_counter() - started
    print(
        f"actions={counts['actions']} battles={counts['battles']} "
        f"matches={counts['matches']} elapsed={elapsed:.3f}s "
        f"actions_per_second={counts['actions'] / elapsed:.1f}"
    )
    stream = io.StringIO()
    pstats.Stats(profiler, stream=stream).strip_dirs().sort_stats(
        "cumulative"
    ).print_stats(args.top)
    print(stream.getvalue())


if __name__ == "__main__":
    main()
