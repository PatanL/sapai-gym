#!/usr/bin/env python3
"""Recommend one action from a versioned live-state JSON document."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from urllib.request import Request, urlopen

from sapai_gym1.deployment.policy_adapter import LivePolicyEngine
from sapai_gym1.deployment.state_schema import LiveGameState


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("state", type=Path)
    parser.add_argument("--match-id")
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--backend", choices=('cpu', 'mps'), default='cpu')
    parser.add_argument("--dashboard", default="http://127.0.0.1:8770/agent")
    parser.add_argument("--no-post", action="store_true")
    parser.add_argument("--plan", action="store_true")
    parser.add_argument("--max-actions", type=int, default=20)
    return parser.parse_args()


def main():
    args = parse_args()
    state = LiveGameState.from_dict(json.loads(args.state.read_text()))
    engine = LivePolicyEngine(args.checkpoint, threads=args.threads, backend=args.backend)
    if args.plan:
        plan = engine.plan_until_barrier(
            state, max_actions=args.max_actions
        )
        recommendation = plan[0].recommendation
    else:
        recommendation = engine.recommend(state, match_id=args.match_id)
        plan = ()
    payload = {
        "mode": "dry-run",
        "checkpoint": str(args.checkpoint),
        "checkpointUpdate": engine.checkpoint.get("update"),
        "state": str(args.state),
        "recommendation": recommendation.to_dict(),
        "plan": [step.to_dict() for step in plan],
    }
    rendered = json.dumps(payload, indent=2)
    print(rendered)
    if not args.no_post:
        request = Request(
            args.dashboard,
            data=json.dumps(payload).encode(),
            headers={"content-type": "application/json"},
            method="POST",
        )
        with urlopen(request, timeout=5) as response:
            if response.status != 200:
                raise RuntimeError(f"Dashboard returned {response.status}")


if __name__ == "__main__":
    main()
