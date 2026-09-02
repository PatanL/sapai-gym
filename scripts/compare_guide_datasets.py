#!/usr/bin/env python3
"""Compare two exact guide demonstration corpora."""

from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path

from sapai_gym1.rl import ActionType
from sapai_gym1.rl.guide_teacher import LINE_MEMBERS


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("old", type=Path)
    parser.add_argument("new", type=Path)
    parser.add_argument("--json-output", type=Path, required=True)
    parser.add_argument("--markdown-output", type=Path, required=True)
    parser.add_argument("--sample-count", type=int, default=20)
    return parser.parse_args()


def key_for(row):
    return (
        row["replay_id"],
        row["perspective"],
        int(row["turn"]),
        int(row["decision"]),
    )


def observation_hash(row):
    encoded = json.dumps(
        row["observation"], sort_keys=True, separators=(",", ":")
    ).encode()
    return hashlib.sha256(encoded).hexdigest()


def action_is_legal(row):
    action_type, source, target = map(int, row["action"])
    observation = row["observation"]
    return bool(
        observation["type_mask"][action_type]
        and observation["source_mask"][action_type][source]
        and observation["target_mask"][action_type][source][target]
    )


def summarize(path, retain_examples=False):
    actions = Counter()
    lines = Counter()
    guide_versions = Counter()
    guide_hashes = Counter()
    executable_hashes = Counter()
    matchup_counters = Counter()
    end_gold = Counter()
    perspectives = Counter()
    replay_ids = set()
    illegal = 0
    examples = {} if retain_examples else None
    total = 0
    with path.open() as source:
        for raw in source:
            row = json.loads(raw)
            total += 1
            action_type = ActionType(int(row["action"][0])).name
            actions[action_type] += 1
            teacher = row.get("teacher") or {}
            line = teacher.get("line") or teacher.get("boardClass") or "unknown"
            lines[line] += 1
            guide_versions[teacher.get("version") or teacher.get("name") or "missing"] += 1
            guide_hashes[teacher.get("sha256") or "missing"] += 1
            executable_hashes[
                teacher.get("executableSha256") or "missing"
            ] += 1
            matchup_counters.update(teacher.get("matchupCounters") or [])
            perspectives[row["perspective"]] += 1
            replay_ids.add(row["replay_id"])
            illegal += int(not action_is_legal(row))
            if action_type == "END_TURN":
                gold = round(float(row["observation"]["numeric"][0][11]) * 30)
                end_gold[gold] += 1
            if retain_examples:
                examples[key_for(row)] = {
                    "action": tuple(map(int, row["action"])),
                    "observationHash": observation_hash(row),
                    "line": line,
                }
    return {
        "path": str(path),
        "examples": total,
        "replays": len(replay_ids),
        "perspectives": dict(sorted(perspectives.items())),
        "actionCounts": dict(sorted(actions.items())),
        "lineFrequencies": dict(lines.most_common()),
        "guideVersions": dict(guide_versions),
        "guideHashes": dict(guide_hashes),
        "executableTeacherHashes": dict(executable_hashes),
        "matchupCounterFrequencies": dict(matchup_counters.most_common()),
        "endTurnGoldHistogram": {
            str(gold): count for gold, count in sorted(end_gold.items())
        },
        "zeroGoldCompliance": (
            end_gold.get(0, 0) / sum(end_gold.values()) if end_gold else None
        ),
        "illegalActions": illegal,
        "_examples": examples,
    }


def markdown_table(headers, rows):
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join("---" for _ in headers) + " |",
    ]
    lines.extend("| " + " | ".join(map(str, row)) + " |" for row in rows)
    return "\n".join(lines)


def main():
    args = parse_args()
    old = summarize(args.old, retain_examples=True)
    new = summarize(args.new, retain_examples=True)
    old_examples = old.pop("_examples")
    new_examples = new.pop("_examples")
    aligned = sorted(set(old_examples) & set(new_examples))
    same_observation = 0
    changed_same_observation = 0
    changed_aligned = 0
    samples = []
    for key in aligned:
        previous = old_examples[key]
        current = new_examples[key]
        action_changed = previous["action"] != current["action"]
        changed_aligned += int(action_changed)
        if previous["observationHash"] != current["observationHash"]:
            continue
        same_observation += 1
        if not action_changed:
            continue
        changed_same_observation += 1
        if len(samples) < args.sample_count:
            samples.append(
                {
                    "replay": key[0],
                    "perspective": key[1],
                    "turn": key[2],
                    "decision": key[3],
                    "oldAction": list(previous["action"]),
                    "newAction": list(current["action"]),
                    "oldLine": previous["line"],
                    "newLine": current["line"],
                    "observationHash": previous["observationHash"],
                }
            )
    comparison = {
        "old": old,
        "new": new,
        "changedExamples": {
            "alignedKeys": len(aligned),
            "changedActionsAtAlignedKeys": changed_aligned,
            "sameObservationExamples": same_observation,
            "changedActionsAtSameObservation": changed_same_observation,
            "samples": samples,
        },
    }
    args.json_output.parent.mkdir(parents=True, exist_ok=True)
    args.json_output.write_text(json.dumps(comparison, indent=2, sort_keys=True) + "\n")

    action_names = sorted(
        set(old["actionCounts"]) | set(new["actionCounts"])
    )
    action_rows = [
        (
            name,
            old["actionCounts"].get(name, 0),
            new["actionCounts"].get(name, 0),
            new["actionCounts"].get(name, 0) - old["actionCounts"].get(name, 0),
        )
        for name in action_names
    ]
    line_names = list(
        dict.fromkeys(
            list(LINE_MEMBERS)
            + list(new["lineFrequencies"])
            + list(old["lineFrequencies"])
        )
    )
    line_rows = [
        (
            name,
            old["lineFrequencies"].get(name, 0),
            new["lineFrequencies"].get(name, 0),
        )
        for name in line_names
    ]
    sample_lines = [
        f"- turn {sample['turn']} {sample['perspective']}: "
        f"`{sample['oldAction']}` -> `{sample['newAction']}`; "
        f"{sample['oldLine']} -> {sample['newLine']}"
        for sample in samples
    ] or ["- No changed same-observation samples."]
    report = f"""# Turtle P1 Guide Dataset Comparison

Old dataset: `{args.old}`  
New dataset: `{args.new}`

## Summary

- Examples: {old['examples']:,} -> {new['examples']:,}
- Complete replay IDs: {old['replays']} -> {new['replays']}
- Illegal actions: {old['illegalActions']} -> {new['illegalActions']}
- Zero-gold END_TURN compliance: {old['zeroGoldCompliance']:.1%} -> {new['zeroGoldCompliance']:.1%}
- Aligned decision keys: {len(aligned):,}
- Changed actions at aligned keys: {changed_aligned:,}
- Identical-observation aligned examples: {same_observation:,}
- Changed actions on identical observations: {changed_same_observation:,}

## Action counts

{markdown_table(('Action', 'Old', 'New', 'Delta'), action_rows)}

## Persistent line frequencies

{markdown_table(('Line / old board class', 'Old', 'New'), line_rows)}

## Guide identity

- Old versions: `{json.dumps(old['guideVersions'], sort_keys=True)}`
- New versions: `{json.dumps(new['guideVersions'], sort_keys=True)}`
- Old hashes: `{json.dumps(old['guideHashes'], sort_keys=True)}`
- New hashes: `{json.dumps(new['guideHashes'], sort_keys=True)}`
- Old executable hashes: `{json.dumps(old['executableTeacherHashes'], sort_keys=True)}`
- New executable hashes: `{json.dumps(new['executableTeacherHashes'], sort_keys=True)}`

## Matchup-counter frequencies

- Old: `{json.dumps(old['matchupCounterFrequencies'], sort_keys=True)}`
- New: `{json.dumps(new['matchupCounterFrequencies'], sort_keys=True)}`

## Changed identical-state samples

{chr(10).join(sample_lines)}
"""
    args.markdown_output.write_text(report)
    print(json.dumps(comparison, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
