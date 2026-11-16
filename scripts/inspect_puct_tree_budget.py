#!/usr/bin/env python3
"""Measure deliberate tree work versus policy rollout work on a frozen root."""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import time

import torch

from benchmark_search_interactive_truth import attach_opponent_model, hybrid_planner
from generate_historical_search_manifest import load_policy
from sapai_gym1.rl.decision_manifest import restore_decision
from sapai_gym1.rl.game_trace import action_snapshot
from sapai_gym1.rl.opponent_reservoir import OpponentTrajectoryReservoir
from sapai_gym1.rl.puct_search import PUCT_VERSION
from sapai_gym1.rl.rollout_search import PublicSearchState


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('checkpoint', type=Path)
    parser.add_argument('--record', type=Path, required=True)
    parser.add_argument('--comparison-row', type=Path, required=True)
    parser.add_argument('--opponent-reservoir', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--search-horizon-battles', type=int, default=5)
    parser.add_argument('--opponent-candidates', type=int, default=32)
    parser.add_argument('--leaf-value-weight', type=float, default=0.5)
    parser.add_argument('--puct-transition-budget-scale', type=float, default=0.8)
    parser.add_argument('--batch-size', type=int, default=64)
    args = parser.parse_args()
    torch.set_num_threads(1)
    device = torch.device('cpu')
    policy, policy_hash, _ = load_policy(args.checkpoint, device)
    record = json.loads(args.record.read_text())
    comparison = json.loads(args.comparison_row.read_text())
    if record['checkpointSha256'] != policy_hash or record['state'] != comparison['state']:
        raise ValueError('Checkpoint or root mismatch')
    if record['searchSeed'] != comparison['searchSeed']:
        raise ValueError('Search seeds differ')
    env, memories, side, _ = restore_decision(record)
    public = PublicSearchState.from_core(env.core, side)
    reservoir = OpponentTrajectoryReservoir.load(args.opponent_reservoir)
    attach_opponent_model(public, env.core, side, reservoir,
                          args.search_horizon_battles, args.opponent_candidates)
    planner = hybrid_planner(policy, device, args.batch_size, args,
                             comparison['selectionTransitions']['root'])
    started = time.perf_counter()
    result = planner.search(public, memories[side], seed=record['searchSeed'],
                            validation_seed=record['validationSeed'])
    nodes = list(planner.nodes.values())
    tree_steps = sum(node.visits for node in nodes)
    payload = {
        'puctVersion': PUCT_VERSION, 'checkpointSha256': policy_hash,
        'recordSha256': hashlib.sha256(args.record.read_bytes()).hexdigest(),
        'state': record['state'], 'rootTurn': record['turn'], 'gold': record['gold'],
        'simulations': result.simulations, 'treeNodes': result.tree_nodes,
        'meanTreeDepth': result.mean_tree_depth, 'maxTreeDepth': result.max_tree_depth,
        'treeSelectedTransitions': tree_steps,
        'policyRolloutTransitions': result.simulator_transitions - tree_steps,
        'treeStageTransitions': result.simulator_transitions,
        'treeSelectionFraction': tree_steps / max(1, result.simulator_transitions),
        'screeningTransitions': result.screening_simulator_transitions,
        'expandedNodesByTurn': dict(sorted(Counter(node.turn for node in nodes).items())),
        'visitedNodesByTurn': dict(sorted(Counter(node.turn for node in nodes if node.visits).items())),
        'treeTransitionsByTurn': dict(sorted(
            (turn, sum(node.visits for node in nodes if node.turn == turn))
            for turn in {node.turn for node in nodes}
        )),
        'nodesVisitedAtLeastTwice': sum(node.visits >= 2 for node in nodes),
        'visitDepthHistogram': dict(sorted(Counter(planner.simulation_depths).items())),
        'treeProposal': action_snapshot(public.core.agent, result.tree_proposed_action),
        'screenedProposal': action_snapshot(public.core.agent, result.proposed_action),
        'wallSeconds': time.perf_counter() - started,
        'purpose': 'compute-allocation diagnosis, not training labels or strength evidence',
    }
    if payload['policyRolloutTransitions'] < 0:
        raise RuntimeError('Tree traversal accounting exceeds simulator transitions')
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_suffix('.json.tmp')
    temporary.write_text(json.dumps(payload, indent=2) + '\n')
    temporary.replace(args.output)
    print(json.dumps(payload), flush=True)


if __name__ == '__main__':
    main()
