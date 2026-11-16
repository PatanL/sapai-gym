#!/usr/bin/env python3
"""Independent terminal outcomes: raw vs one root intervention vs real engine.

This opens a frozen panel once and never authorizes distillation automatically.
Neither the public planner nor its callback receives the truth environment.
"""
import argparse
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import time

import numpy as np
import torch

from sapai.turtle_pack import SIMULATOR_REVISION
from generate_search_demonstrations import load_policy
from sapai_gym1.rl.decision_manifest import restore_decision
from sapai_gym1.rl.rollout_search import NeuralSearchEvaluator, PublicSearchState
from sapai_gym1.rl.puct_search import PUCTConfig, StochasticPUCTSearch
from sapai_gym1.rl.opponent_reservoir import OpponentTrajectoryReservoir
from sapai_gym1.rl.opponent_belief import OpponentBeliefMix
from sapai_gym1.rl.interactive_evaluation import InteractiveSelfPlayEvaluator
from sapai_gym1.rl.search_quality import PublicPUCTEngine, InteractiveEngineEvaluator
from sapai_gym1.rl.joint_demonstrations import serialize_action, sha, QUALITY_SCHEMA, runtime_fingerprint, config_hash
from sapai_gym1.rl.opponent_forecast_quality import paired_match_interval


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('checkpoint', type=Path)
    parser.add_argument('--panel', type=Path, required=True)
    parser.add_argument('--opponent-reservoir', type=Path, required=True)
    parser.add_argument('--output-directory', type=Path, required=True)
    parser.add_argument('--device', choices=('cpu', 'cuda'), default='cpu')
    parser.add_argument('--gpu-boundary-approved', action='store_true')
    parser.add_argument('--states', type=int, default=64)
    parser.add_argument('--futures', type=int, default=32)
    parser.add_argument('--tree-transitions', type=int, default=2000)
    parser.add_argument('--simulations', type=int, default=128)
    parser.add_argument('--screening-rollouts', type=int, default=4)
    parser.add_argument('--validation-rollouts', type=int, default=8)
    parser.add_argument('--search-horizon', type=int, default=5)
    parser.add_argument('--searches-per-turn', type=int, default=2)
    parser.add_argument('--engine-match-tree-budget', type=int, default=6000)
    parser.add_argument('--diagnostic', action='store_true')
    args = parser.parse_args()
    if args.device == 'cuda' and not args.gpu_boundary_approved:
        raise ValueError('Use a separately allocated post-entropy GPU boundary')
    if min(args.states, args.tree_transitions, args.simulations, args.searches_per_turn, args.engine_match_tree_budget) < 1 or args.futures < 2:
        raise ValueError('Invalid independent evaluation budget')
    torch.set_num_threads(1); device = torch.device(args.device)
    policy, _ = load_policy(args.checkpoint, device, False)
    checkpoint_hash = sha(args.checkpoint)
    metadata = json.loads((args.panel/'manifest.json').read_text())
    if metadata['checkpointSha256'] != checkpoint_hash or metadata['simulatorRevision'] != SIMULATOR_REVISION:
        raise ValueError('Frozen state panel/GRU ownership mismatch')
    if args.states > metadata['states'] or args.futures > metadata['truthRollouts']:
        raise ValueError('Budget exceeds the frozen held-out panel')
    if not args.diagnostic and args.states < 64:
        raise ValueError('Predeclared first screen requires 64 held-out states; smaller is diagnostic only')
    records = [json.loads((args.panel/f'state-{i:04d}.json').read_text()) for i in range(args.states)]
    panel_hash = hashlib.sha256(json.dumps(records, sort_keys=True).encode()).hexdigest()
    seed_hash = hashlib.sha256(json.dumps([r['truthSeeds'][:args.futures] for r in records]).encode()).hexdigest()
    args.output_directory.mkdir(parents=True, exist_ok=False)
    config = PUCTConfig(max_simulations=args.simulations, max_simulator_transitions=args.tree_transitions,
        horizon_battles=args.search_horizon, validation_rollouts=args.validation_rollouts,
        max_validation_rollouts=args.validation_rollouts, audit_rollouts=0,
        root_screening_rollouts=args.screening_rollouts, root_screening_rounds=2, tree_leaf_rollout_steps=12)
    specification = {'checkpointSHA256': checkpoint_hash, 'statePanelSHA256': panel_hash, 'seedPanelSHA256': seed_hash,
        'sourceManifestSHA256': runtime_fingerprint(), 'plannerConfigSHA256': config_hash(asdict(config)),
        'reservoirSHA256': sha(args.opponent_reservoir), 'runnerSHA256': sha(Path(__file__)),
        'config': asdict(config), 'arguments': {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
        'belief': 'existing marginal/persistence/reservoir mixture; no new opponent model',
        'distillationAuthorized': False, 'panelOpened': True, 'prefixEnhancedSearchImplemented': False}
    (args.output_directory/'specification.json').write_text(json.dumps(specification, indent=2)+'\n')
    reservoir = OpponentTrajectoryReservoir.load(args.opponent_reservoir)
    evaluator = NeuralSearchEvaluator(policy, device=device, chunk_size=64)
    root_truth = InteractiveSelfPlayEvaluator(policy, device=device, batch_size=64, horizon_battles=25, max_steps=5000)
    engine_truth = InteractiveEngineEvaluator(policy, device=device, batch_size=64, horizon_battles=25, max_steps=5000,
        engine_factory=lambda: PublicPUCTEngine(evaluator, config, match_tree_budget=args.engine_match_tree_budget,
                                                searches_per_turn=args.searches_per_turn, reservoir=reservoir))
    rows = []; scores = {'root': [], 'engine': []}; clusters = []; complete = True
    for index, record in enumerate(records):
        if record['checkpointSha256'] != checkpoint_hash:
            raise ValueError('Foreign recurrent state in panel')
        if record.get('policySha256BySeat', [checkpoint_hash]*2) != [checkpoint_hash]*2:
            raise ValueError('This initial gate requires both memory vectors owned by the same frozen policy')
        environment, memories, side, _ = restore_decision(record)
        public = PublicSearchState.from_core(environment.core, side)
        public.set_opponent_transition_model(reservoir, candidate_limit=32)
        public.enable_learned_opponent_belief(OpponentBeliefMix())
        before_metrics = evaluator.metrics(); started = time.perf_counter()
        result = StochasticPUCTSearch(evaluator, config).search(public, memories[side], seed=record['searchSeed'], validation_seed=record['validationSeed'])
        root_wall = time.perf_counter()-started
        actions = tuple(dict.fromkeys((result.policy_action, result.selected_action)))
        seeds = record['truthSeeds'][:args.futures]
        started = time.perf_counter()
        root_outcomes = root_truth.evaluate(environment, memories, side, actions, seeds)
        root_evaluation_wall = time.perf_counter()-started
        outcomes = {(r.root_action, r.seed): r for r in root_outcomes}
        started = time.perf_counter()
        engine_outcomes = engine_truth.evaluate(environment, memories, side, (result.selected_action,), seeds)
        engine_evaluation_wall = time.perf_counter()-started
        terminated = all(r.terminated and r.match_score is not None for r in (*root_outcomes, *engine_outcomes))
        complete &= terminated
        root_difference = engine_difference = None
        if terminated:
            raw = np.array([outcomes[result.policy_action, seed].match_score for seed in seeds])
            proposed = np.array([outcomes[result.selected_action, seed].match_score for seed in seeds])
            engine = np.array([r.match_score for r in engine_outcomes])
            root_difference, engine_difference = float((proposed-raw).mean()), float((engine-raw).mean())
            scores['root'].append(root_difference); scores['engine'].append(engine_difference)
            clusters.append({'metadata': {'collection_id': checkpoint_hash, 'match_id': str(record['replay']['resetSeed'])}})
        row = {'state': index, 'rawDecision': serialize_action(result.policy_action), 'executedRootDecision': serialize_action(result.selected_action),
            'rootUnchangedOrFallback': result.selected_action == result.policy_action, 'complete': terminated,
            'rootTerminalScoreGain': root_difference, 'engineTerminalScoreGain': engine_difference,
            'rootSearchWallSeconds': root_wall, 'rootTruthWallSeconds': root_evaluation_wall,
            'engineTruthAndPlanningWallSeconds': engine_evaluation_wall, 'engine': engine_truth.execution_metrics(),
            'rootTotalSimulatorTransitions': result.simulator_transitions+result.screening_simulator_transitions+result.validation_simulator_transitions+result.audit_simulator_transitions,
            'neuralCountersBefore': before_metrics, 'neuralCountersAfter': evaluator.metrics(),
            'omittedMaskMass': result.omitted_end_mask_probability_mass,
            'outcomes': [{'mode': mode, 'decision': serialize_action(r.root_action), 'seed': r.seed,
                          'score': r.match_score, 'terminated': r.terminated, 'truncated': r.truncated}
                         for mode, results in (('root-only', root_outcomes), ('engine', engine_outcomes)) for r in results]}
        rows.append(row)
        (args.output_directory/f'state-{index:04d}.json').write_text(json.dumps(row, indent=2)+'\n')
        print(json.dumps({'state': index, 'complete': terminated, 'rootGain': root_difference, 'engineGain': engine_difference}), flush=True)
    comparisons = {}
    if complete:
        for mode, values in scores.items():
            estimate = paired_match_interval(clusters, values, [0.]*len(values))
            comparisons[mode] = {'originatingMatches': estimate['matches'],
                'meanTerminalScoreGain': estimate['meanMatchNllImprovement'],
                'pairedOriginatingMatch95CI': estimate['bootstrapMatch95CI'],
                'bootstrapReplicates': estimate['bootstrapReplicates']}
    summary = {'schema': QUALITY_SCHEMA, 'policySHA256': checkpoint_hash, 'statePanelSHA256': panel_hash,
        'sourceManifestSHA256': specification['sourceManifestSHA256'], 'plannerConfigSHA256': specification['plannerConfigSHA256'],
        'seedPanelSHA256': seed_hash, 'simulatorRevision': SIMULATOR_REVISION,
        'approvedForDistillation': False, 'diagnosticOnly': args.diagnostic, 'allContinuationsTerminal': complete,
        'rootFallbackFraction': float(np.mean([r['rootUnchangedOrFallback'] for r in rows])),
        'unresolvedMissingOutcomesExcludedFromApproval': not complete,
        'comparisons': comparisons if complete else None,
        'interpretation': 'Terminal match score only, clustered by originating match, not independent future rows. No automatic quality approval.'}
    (args.output_directory/'summary.json').write_text(json.dumps(summary, indent=2)+'\n')


if __name__ == '__main__':
    main()
