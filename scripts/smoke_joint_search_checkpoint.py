#!/usr/bin/env python3
"""CPU production-checkpoint contract smoke. Never emits production labels."""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import torch
from sapai import Team
from generate_search_demonstrations import load_policy, policy_action, force_action, step_policy_decisions
from sapai_gym1.rl.selfplay_env import ParallelSelfPlayEnv
from sapai_gym1.rl.joint_action import JointAction
from sapai_gym1.rl.action_schema import ActionType, AutoregressiveAction
from sapai_gym1.rl.rollout_search import PublicSearchState, NeuralSearchEvaluator
from sapai_gym1.rl.puct_search import PUCTConfig, StochasticPUCTSearch
from sapai_gym1.rl.joint_search_records import make_record, JointSearchReservoir, rescore_record, diagnostic_record_loss


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('checkpoint', type=Path)
    parser.add_argument('--output-directory', type=Path, required=True)
    args = parser.parse_args()
    args.output_directory.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(1); torch.manual_seed(9061042)
    policy, _ = load_policy(args.checkpoint, torch.device('cpu'), False)
    if not policy.config.end_freeze:
        raise ValueError('This production smoke requires a joint policy')
    with args.checkpoint.open('rb') as stream:
        checkpoint_hash = hashlib.file_digest(stream, 'sha256').hexdigest()
    env = ParallelSelfPlayEnv(); env.reset(seed=9061042)
    env.core.agent.gold = env.core.opponent.gold = 0
    memory = policy.initial_memory(1)[0]
    state = PublicSearchState.from_core(env.core, 0, opening_prior=((Team(['pet-fish']),),))
    evaluator = NeuralSearchEvaluator(policy, device='cpu', joint_mask_budget=4, joint_exploration_masks=1)
    evaluation = evaluator.evaluate(state, memory)
    sampled, next_memory = policy_action(policy, state.observation(), memory, torch.device('cpu'), True)
    forced = force_action(policy, state.observation(), memory, sampled, torch.device('cpu'))
    torch.testing.assert_close(next_memory, forced, atol=0, rtol=0)
    record = make_record(state.observation(), memory, tuple(evaluation.priors), tuple(evaluation.priors.values()),
                         checkpoint_sha256=checkpoint_hash, omitted_probability_mass=evaluation.omitted_probability_mass,
                         metadata={'fixture': 'initial-shop-derived-zero-gold', 'qualityEvidence': False})
    path = args.output_directory/'reservoir.json'
    JointSearchReservoir([record]).save(path)
    restored = JointSearchReservoir.load(path).records[0]
    JointSearchReservoir([restored]).write_diagnostic_demonstrations(args.output_directory/'diagnostic.jsonl')
    logps, restored_memory = rescore_record(restored, policy, checkpoint_sha256=checkpoint_hash)
    np.testing.assert_allclose(logps, tuple(evaluation.exact_log_probabilities.values()), atol=3e-6, rtol=1e-6)
    torch.testing.assert_close(restored_memory, evaluation.next_memory, atol=0, rtol=0)
    loss = diagnostic_record_loss(restored, policy, checkpoint_sha256=checkpoint_hash)
    loss.backward()
    assert torch.isfinite(loss) and all(p.grad is None or torch.isfinite(p.grad).all() for p in policy.parameters())
    policy.zero_grad(set_to_none=True)
    ends = [a for a in evaluation.priors if a.action_type == ActionType.END_TURN]
    nonzero = next(a for a in ends if a.end_freeze_mask)
    branch = state.clone(); branch.step(nonzero, np.random.default_rng(71))
    result = step_policy_decisions(env, (nonzero, JointAction(AutoregressiveAction(ActionType.END_TURN), 0)))
    assert result.battle_resolved and branch.battles == 1
    planner = StochasticPUCTSearch(evaluator, PUCTConfig(max_simulations=2, max_simulator_transitions=20,
        max_tree_depth=2, max_rollout_steps=2, horizon_battles=1, validation_rollouts=2,
        max_validation_rollouts=2, audit_rollouts=0, root_screening_top_k=2,
        root_screening_min_survivors=2, root_screening_rollouts=2,
        root_screening_rounds=1, tree_leaf_rollout_steps=0))
    searched = planner.search(state, memory, seed=71, validation_seed=72)
    assert isinstance(searched.selected_action, JointAction)
    report = {'checkpointSHA256': checkpoint_hash, 'device': 'cpu', 'passed': True,
              'actionSchema': nonzero.schema, 'diagnosticOnly': True, 'productionLabels': 0,
              'optimizerSteps': 0, 'fixture': 'initial-shop-derived-zero-gold',
              'teacherStrengthProven': False, 'productionGeneratorFullyMigrated': False,
              'fullPuctSearchExecuted': True, 'jointEdges': len(searched.visit_counts),
              'omittedMaskMass': evaluation.omitted_probability_mass,
              'jointConsumerLoss': float(loss.detach()), 'recurrentDimension': memory.numel()}
    (args.output_directory/'report.json').write_text(json.dumps(report, indent=2)+'\n')
    print(json.dumps(report))


if __name__ == '__main__':
    main()
