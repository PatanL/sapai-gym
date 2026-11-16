import json
import math

import numpy as np
import pytest
import torch
from sapai import Team

from sapai_gym1.rl.action_schema import ActionType, FlatActionCodec
from sapai_gym1.rl.joint_action import JointAction
from sapai_gym1.rl.rollout_search import NeuralSearchEvaluator, PublicSearchState, CounterfactualRolloutSearch, RolloutSearchConfig
from sapai_gym1.rl.puct_search import StochasticPUCTSearch, PUCTConfig
from sapai_gym1.rl.joint_search_records import make_record, restore_record, rescore_record, JointSearchReservoir
from test_end_freeze_policy import policies, zero_gold_observations


def fixture():
    env, _, policy=policies();zero_gold_observations(env)
    with torch.no_grad():
        policy.type_head.bias[0]=30
        policy.end_freeze_head.scorer.bias.zero_()
    public=PublicSearchState.from_core(env.core,0,opening_prior=((Team(['pet-fish']),),))
    evaluator=NeuralSearchEvaluator(policy,joint_mask_budget=4,joint_exploration_masks=1)
    return env,policy,public,evaluator,policy.initial_memory(1)[0]


def test_joint_priors_keep_mask_edges_and_exact_likelihood_with_one_encoding():
    env,policy,state,evaluator,memory=fixture()
    calls=[];hook=policy.memory_cell.register_forward_hook(lambda *args:calls.append(1))
    evaluation=evaluator.evaluate(state,memory);hook.remove()
    ends=[a for a in evaluation.priors if a.action_type==ActionType.END_TURN]
    assert len(ends)==4 and len({a.end_freeze_mask for a in ends})==4
    assert len(calls)==1
    assert evaluation.policy_action in evaluation.priors
    assert evaluation.omitted_probability_mass>0
    assert sum(evaluation.priors.values())==pytest.approx(1.)
    retained=sum(math.exp(p) for p in evaluation.exact_log_probabilities.values())
    assert retained+evaluation.omitted_probability_mass==pytest.approx(1.,abs=1e-6)
    selector=CounterfactualRolloutSearch(evaluator)
    assert all(a in selector._candidate_actions(evaluation.priors,state) for a in ends)
    with pytest.raises(ValueError,match='discard'):
        ends[0].as_tuple()
    with pytest.raises(ValueError,match='never drop'):
        FlatActionCodec.encode(ends[0])


def test_policy_search_execution_reservoir_restore_and_rescore(tmp_path):
    env,policy,state,evaluator,memory=fixture()
    evaluation=evaluator.evaluate(state,memory)
    actions=tuple(evaluation.priors)
    record=make_record(state.observation(),memory,actions,tuple(evaluation.priors.values()),
                       checkpoint_sha256='a'*64,omitted_probability_mass=evaluation.omitted_probability_mass)
    reservoir=JointSearchReservoir([record]);path=tmp_path/'reservoir.json';reservoir.save(path)
    restored=JointSearchReservoir.load(path).records[0]
    demonstration=tmp_path/'diagnostic.jsonl'
    JointSearchReservoir([restored]).write_diagnostic_demonstrations(demonstration)
    restored=json.loads(demonstration.read_text())
    calls=[];hook=policy.memory_cell.register_forward_hook(lambda *args:calls.append(1))
    logps,next_memory=rescore_record(restored,policy,checkpoint_sha256='a'*64);hook.remove()
    assert len(calls)==1
    assert logps==pytest.approx(tuple(evaluation.exact_log_probabilities.values()),abs=2e-6)
    torch.testing.assert_close(next_memory,evaluation.next_memory,rtol=0,atol=0)
    action=next(a for a in actions if a.action_type==ActionType.END_TURN and a.end_freeze_mask)
    before=state.observation();branch=state.clone()
    branch.step(action,np.random.default_rng(543))
    assert branch.battles==1
    for key in before:np.testing.assert_array_equal(before[key],state.observation()[key])
    with pytest.raises(ValueError,match='Foreign checkpoint'):
        restore_record(restored,checkpoint_sha256='b'*64)
    from sapai_gym1.replay.dataset import ReplayExampleDataset
    with pytest.raises(ValueError,match='joint-aware'):
        ReplayExampleDataset([demonstration])


def test_puct_creates_distinct_joint_mask_edges_without_hidden_truth():
    env,policy,state,evaluator,memory=fixture()
    planner=StochasticPUCTSearch(evaluator,PUCTConfig(max_simulations=2,max_simulator_transitions=20,
        max_tree_depth=2,horizon_battles=1,validation_rollouts=2,max_validation_rollouts=2,
        audit_rollouts=0,root_screening_rollouts=2,root_screening_rounds=1,
        tree_leaf_rollout_steps=0))
    node,_=planner._expand(state,memory,root=True,rng=np.random.default_rng(8),exhaustive_root=False)
    ends=[a for a in node.edges if a.action_type==ActionType.END_TURN]
    assert len(ends)==4
    assert all(isinstance(a,JointAction) for a in node.edges)
    assert node.omitted_probability_mass>0
    assert len(state.core.opponent.shop)==0
    assert state.core.previous_opponent_state is None
    planner._simulate(state,memory,845,forced_root=ends[0])
    assert node.edges[ends[0]].visits==1
