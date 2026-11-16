from collections import Counter

import numpy as np
import torch

from sapai_gym1.rl.selfplay_env import ParallelSelfPlayEnv
from sapai_gym1.rl.joint_action import JointAction, step_decisions
from sapai_gym1.rl.action_schema import AutoregressiveAction
from sapai_gym1.rl.decision_manifest import decision_record, restore_decision
from sapai_gym1.rl.search_quality import InteractiveEngineEvaluator
from sapai_gym1.rl.interactive_evaluation import InteractiveSelfPlayEvaluator
from test_end_freeze_policy import policies


def test_typed_state_mining_manifest_replays_joint_decisions_exactly():
    env, _, policy = policies()
    seed = 191; observations = env.reset(seed=seed)
    batch = {k: torch.tensor(np.stack([o[k] for o in observations])) for k in observations[0]}
    output = policy.act(batch, policy.initial_memory(2), deterministic=True)
    actions = [JointAction(AutoregressiveAction(*a.tolist()), int(m)) for a, m in zip(output.actions, output.end_freeze_mask)]
    step_decisions(env, actions)
    record = decision_record(env, list(output.memory), 0, (), {'resetSeed': seed, 'actions': [[a.to_dict() for a in actions]]},
                             search_seed=1, validation_seed=2, truth_seeds=[31, 32])
    restored, memories, side, reasons = restore_decision(record)
    for expected, actual in zip(env.observations(), restored.observations()):
        for key in expected:
            np.testing.assert_array_equal(expected[key], actual[key])
    torch.testing.assert_close(torch.stack(memories), output.memory, atol=0, rtol=0)


def test_online_callback_gets_only_public_clone_and_raw_fallback_matches_baseline():
    env, _, policy = policies(); env.reset(seed=515)
    env.core.MAX_TURN = 3
    obs = env.observations()[0]
    batch = {k: torch.tensor(v)[None] for k, v in obs.items()}
    action = policy.act(batch, deterministic=True)
    root = JointAction(AutoregressiveAction(*action.actions[0].tolist()), int(action.end_freeze_mask[0]))
    memories = [policy.initial_memory(1)[0] for _ in range(2)]
    snapshots = env.observations()
    seen = []
    class PublicNoOp:
        def __init__(self):
            self.counts = Counter(); self.mask_mass = []
        def choose(self, public, memory, raw, next_memory):
            assert public.core is not env.core
            assert len(public.core.opponent.team) == 0 and len(public.core.opponent.shop) == 0
            assert public.core.previous_opponent_state is not None
            self.counts['publicCalls'] += 1; seen.append(public)
            return raw, next_memory
    baseline = InteractiveSelfPlayEvaluator(policy, horizon_battles=3, max_steps=500, batch_size=4)
    engine = InteractiveEngineEvaluator(policy, engine_factory=PublicNoOp, horizon_battles=3, max_steps=500, batch_size=4)
    first = baseline.evaluate(env, memories, 0, (root,), (12, 13))
    second = engine.evaluate(env, memories, 0, (root,), (12, 13))
    assert first == second and seen
    assert all(r.terminated for r in second)
    for before, after in zip(snapshots, env.observations()):
        for key in before:
            np.testing.assert_array_equal(before[key], after[key])
