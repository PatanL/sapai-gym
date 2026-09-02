#!/usr/bin/env python3
"""Generate exact targeted Turtle P1 scenario demonstrations."""

from __future__ import annotations

import argparse
import hashlib
from pathlib import Path

from sapai_gym1.rl.curriculum import (
    CURRICULUM_VERSION,
    generate_curriculum_examples,
    scenario_counts,
    write_curriculum_jsonl,
)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--examples-per-scenario", type=int, default=128)
    parser.add_argument("--seed", type=int, default=20260901)
    return parser.parse_args()


def main():
    args = parse_args()
    examples = generate_curriculum_examples(
        examples_per_scenario=args.examples_per_scenario,
        seed=args.seed,
    )
    write_curriculum_jsonl(args.output, examples)
    digest = hashlib.sha256(args.output.read_bytes()).hexdigest()
    print(f"version={CURRICULUM_VERSION}")
    print(f"sequences={args.examples_per_scenario * len(scenario_counts(examples))}")
    print(f"actions={len(examples)}")
    print(f"scenario_actions={dict(sorted(scenario_counts(examples).items()))}")
    print(f"sha256={digest}")


if __name__ == "__main__":
    main()
