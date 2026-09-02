#!/usr/bin/env python3
"""Download, cache, reconstruct, validate, and quarantine SAP replays."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from sapai_gym1.replay import ReplayCode, ReplayParser, ReplayReconstructor, SapLibraryClient


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("replay_codes", nargs="*", help="Pid or replay JSON")
    parser.add_argument("--replay-file", type=Path)
    parser.add_argument("--cache-dir", type=Path, default=Path("replay-data/cache"))
    parser.add_argument("--output-dir", type=Path, default=Path("replay-data/reconstructed"))
    parser.add_argument("--calculator-root", type=Path, default=Path("../SAP-Calculator"))
    parser.add_argument("--allow-non-turtle", action="store_true")
    return parser.parse_args()


def load_codes(args):
    codes = list(args.replay_codes)
    if args.replay_file:
        for line in args.replay_file.read_text().splitlines():
            line = line.strip()
            if line and not line.startswith("#"):
                codes.append(line)
    if not codes:
        raise SystemExit("Provide replay codes or --replay-file")
    return [ReplayCode.parse(code) for code in codes]


def main():
    args = parse_args()
    client = SapLibraryClient(args.cache_dir)
    reconstructor = ReplayReconstructor(ReplayParser(args.calculator_root))
    accepted = 0
    quarantined = 0
    for replay_code in load_codes(args):
        bundle = client.fetch_bundle(replay_code)
        metadata = bundle.get("metadata", {})
        if not args.allow_non_turtle and (
            metadata.get("pack") != "Turtle"
            or metadata.get("opponent_pack") != "Turtle"
        ):
            print(
                f"skip replay={bundle['replayId']} pack={metadata.get('pack')} "
                f"opponent_pack={metadata.get('opponent_pack')}",
                flush=True,
            )
            continue
        report = reconstructor.reconstruct(bundle)
        report.write(args.output_dir)
        accepted += len(report.examples)
        quarantined += len(report.quarantine)
        print(
            f"replay={report.replay_id} examples={len(report.examples)} "
            f"quarantine={len(report.quarantine)} perspectives={len(report.perspectives)}",
            flush=True,
        )
    print(
        json.dumps(
            {
                "validatedExamples": accepted,
                "quarantineRecords": quarantined,
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()

