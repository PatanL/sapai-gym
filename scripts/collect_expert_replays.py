#!/usr/bin/env python3
"""Archive public Turtle replays. Raw evidence only; never starts training."""
import argparse
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from sapai_gym1.replay.archive import collect
from sapai_gym1.replay.evidence import EvidenceParser


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--calculator-root", type=Path, default=Path(__file__).resolve().parents[2] / "SAP-Calculator")
    parser.add_argument("--max-matches", type=int, default=40)
    parser.add_argument("--max-pages", type=int, default=20)
    args = parser.parse_args()
    if not 1 <= args.max_matches <= 500 or not 1 <= args.max_pages <= 100:
        parser.error("Bounded collection: 1..500 matches, 1..100 pages")
    collect(args.output, EvidenceParser(args.calculator_root), max_matches=args.max_matches, max_pages=args.max_pages)


if __name__ == "__main__":
    main()
