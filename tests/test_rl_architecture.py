import copy
import multiprocessing as mp
import tempfile
import threading
import unittest
from pathlib import Path

import numpy as np
import torch

from sapai import Pet, Team
from sapai_gym1 import SuperAutoPetsEnv
from sapai_gym1.rl import (
    ActionType,
    AutoregressiveAction,
    BatchedSelfPlayCollector,
    EntityObservationEncoder,
    EntityObservationWrapper,
    EntityTransformerConfig,
    EntityTransformerPolicy,
    FlatActionCodec,
    LeagueManager,
    LeagueMember,
    ParallelSelfPlayEnv,
    SharedTrajectoryRing,
    SharedActorBuffers,
    DynamicInferenceBatcher,
    Transition,
    TurnSegment,
    impala_actor_critic_loss,
    learner_update,
    behavior_cloning_loss,
    action_shaping_reward,
    vtrace_from_log_probs,
)
from sapai_gym1.rl.distributed import _completed_match_result


def tensor_batch(observation, batch_size=2, device="cpu"):
    return {
        key: torch.as_tensor(
            np.stack([value] * batch_size), device=device
        )
        for key, value in observation.items()
    }


class TestActionSchema(unittest.TestCase):
    def test_generated_pet_tier_is_observed_as_tier_one(self):
        encoder = EntityObservationEncoder()
        bus = Pet("bus")
        bus._attack = 5
        bus._health = 3
        history = encoder._encode_opponent_history(None, Team([bus]))
        self.assertAlmostEqual(
            float(history["opponent_history_numeric"][1, 0, 4]),
            1.0 / 6.0,
        )

    def test_every_flat_action_round_trips(self):
        for flat_action in range(FlatActionCodec.NUM_ACTIONS):
            decoded = FlatActionCodec.decode(flat_action)
            self.assertEqual(FlatActionCodec.encode(decoded), flat_action)

    def test_entity_masks_are_lossless_with_flat_actions(self):
        env = SuperAutoPetsEnv(valid_actions_only=True)
        env.reset(seed=5)
        encoder = EntityObservationEncoder()
        observation = encoder.encode(env)
        triples = set()
        for flat_action in env._avail_actions(0):
            action = FlatActionCodec.decode(flat_action)
            triples.add(action.as_tuple())
            action_type, source, target = action.as_tuple()
            self.assertTrue(observation["type_mask"][action_type])
            self.assertTrue(observation["source_mask"][action_type, source])
            self.assertTrue(
                observation["target_mask"][action_type, source, target]
            )
        self.assertEqual(len(triples), len(env._avail_actions(0)))


class TestEntityTransformer(unittest.TestCase):
    def setUp(self):
        self.env = EntityObservationWrapper(
            SuperAutoPetsEnv(valid_actions_only=True)
        )
        self.observation, _ = self.env.reset(seed=11)

    def test_observation_and_random_autoregressive_rollout(self):
        self.assertTrue(self.env.observation_space.contains(self.observation))
        observation = self.observation
        for _ in range(100):
            action_type = int(
                np.random.choice(np.flatnonzero(observation["type_mask"]))
            )
            source = int(
                np.random.choice(
                    np.flatnonzero(observation["source_mask"][action_type])
                )
            )
            target = int(
                np.random.choice(
                    np.flatnonzero(
                        observation["target_mask"][action_type, source]
                    )
                )
            )
            observation, _, terminated, truncated, _ = self.env.step(
                np.array([action_type, source, target])
            )
            self.assertTrue(self.env.observation_space.contains(observation))
            if terminated or truncated:
                observation, _ = self.env.reset()

    def test_policy_sampling_evaluation_and_backward(self):
        encoder = self.env.encoder
        config = EntityTransformerConfig(
            num_items=encoder.num_items,
            num_statuses=encoder.num_statuses,
            d_model=64,
            nhead=4,
            num_layers=2,
            dim_feedforward=128,
            memory_dim=96,
        )
        model = EntityTransformerPolicy(config)
        batch = tensor_batch(self.observation, batch_size=3)
        sampled = model(batch)
        evaluated = model(batch, actions=sampled.actions)
        self.assertEqual(sampled.actions.shape, (3, 3))
        self.assertTrue(torch.allclose(sampled.log_prob, evaluated.log_prob))
        loss = -evaluated.log_prob.mean() + 0.5 * evaluated.value.square().mean()
        loss.backward()
        self.assertTrue(
            any(parameter.grad is not None for parameter in model.parameters())
        )

    def test_production_model_is_gpu_substantial_not_excessive(self):
        encoder = self.env.encoder
        model = EntityTransformerPolicy(
            EntityTransformerConfig(
                num_items=encoder.num_items,
                num_statuses=encoder.num_statuses,
            )
        )
        self.assertGreater(model.parameter_count(), 10_000_000)
        self.assertLess(model.parameter_count(), 30_000_000)

    def test_pre_v19_checkpoint_loads_with_neutral_new_adapters(self):
        encoder = self.env.encoder
        config = EntityTransformerConfig(
            num_items=encoder.num_items,
            num_statuses=encoder.num_statuses,
            d_model=32,
            nhead=4,
            num_layers=1,
            dim_feedforward=64,
            memory_dim=48,
        )
        original = EntityTransformerPolicy(config)
        v19_prefixes = (
            "legality_projection.",
            "opponent_history_",
            "opponent_belief_adapter.",
            "opponent_pet_head.",
            "opponent_status_head.",
            "opponent_level_head.",
            "opponent_stats_head.",
            "opponent_temporary_head.",
        )
        legacy_state = {
            key: value
            for key, value in original.state_dict().items()
            if not key.startswith(v19_prefixes)
        }
        restored = EntityTransformerPolicy(config)
        restored.load_state_dict(legacy_state)
        self.assertTrue(
            torch.equal(
                restored.legality_projection.weight,
                torch.zeros_like(restored.legality_projection.weight),
            )
        )
        self.assertTrue(
            torch.equal(
                restored.opponent_belief_adapter.weight,
                torch.zeros_like(restored.opponent_belief_adapter.weight),
            )
        )

    def test_v19_transfer_is_history_neutral_until_adapter_learns(self):
        encoder = self.env.encoder
        policy = EntityTransformerPolicy(
            EntityTransformerConfig(
                num_items=encoder.num_items,
                num_statuses=encoder.num_statuses,
                d_model=32,
                nhead=4,
                num_layers=1,
                dim_feedforward=64,
                memory_dim=48,
            )
        ).eval()
        baseline = tensor_batch(self.observation, batch_size=1)
        changed = {key: value.clone() for key, value in baseline.items()}
        changed["opponent_history_board_mask"][:] = 1
        changed["opponent_history_mask"][:] = 1
        changed["opponent_history_item_ids"][:] = min(
            3, encoder.num_items - 1
        )
        changed["opponent_history_numeric"][:] = 0.75
        memory = policy.initial_memory(1)
        with torch.no_grad():
            first = policy(baseline, memory=memory, deterministic=True)
            second = policy(changed, memory=memory, deterministic=True)
        self.assertTrue(torch.equal(first.actions, second.actions))
        self.assertTrue(torch.equal(first.type_logits, second.type_logits))
        self.assertTrue(torch.equal(first.value, second.value))

    def test_stateless_chess_engine_inference_matches_zero_memory(self):
        encoder = self.env.encoder
        policy = EntityTransformerPolicy(
            EntityTransformerConfig(
                num_items=encoder.num_items,
                num_statuses=encoder.num_statuses,
                d_model=32,
                nhead=4,
                num_layers=1,
                dim_feedforward=64,
                memory_dim=48,
            )
        ).eval()
        observation = tensor_batch(self.observation, batch_size=1)
        with torch.no_grad():
            implicit = policy(observation, memory=None, deterministic=True)
            explicit = policy(
                observation,
                memory=policy.initial_memory(1),
                deterministic=True,
            )
        self.assertTrue(torch.equal(implicit.actions, explicit.actions))
        self.assertTrue(torch.equal(implicit.type_logits, explicit.type_logits))
        self.assertTrue(torch.equal(implicit.value, explicit.value))

    def test_legality_projection_receives_value_gradient(self):
        encoder = self.env.encoder
        policy = EntityTransformerPolicy(
            EntityTransformerConfig(
                num_items=encoder.num_items,
                num_statuses=encoder.num_statuses,
                d_model=32,
                nhead=4,
                num_layers=1,
                dim_feedforward=64,
                memory_dim=48,
            )
        )
        observation = tensor_batch(self.observation, batch_size=1)
        policy(observation, deterministic=True).value.sum().backward()
        gradient = policy.legality_projection.weight.grad
        self.assertIsNotNone(gradient)
        self.assertGreater(float(gradient.abs().sum()), 0)

    def test_behavior_cloning_sequence_loss(self):
        encoder = self.env.encoder
        model = EntityTransformerPolicy(
            EntityTransformerConfig(
                num_items=encoder.num_items,
                num_statuses=encoder.num_statuses,
                d_model=64,
                nhead=4,
                num_layers=2,
                dim_feedforward=128,
                memory_dim=96,
            )
        )
        batch = tensor_batch(self.observation, batch_size=2)
        with torch.no_grad():
            actions = model(batch).actions.unsqueeze(0)
        sequence = {key: value.unsqueeze(0) for key, value in batch.items()}
        loss = behavior_cloning_loss(
            model,
            sequence,
            actions,
            model.initial_memory(2),
            valid_mask=torch.ones(1, 2, dtype=torch.bool),
            battle_outcomes=torch.tensor([[0, 2]]),
            returns=torch.tensor([[1.0, -1.0]]),
        )
        loss.total.backward()
        self.assertTrue(torch.isfinite(loss.total))
        self.assertGreaterEqual(
            float(loss.mean_expert_probability.detach()), 0.0
        )

    def test_behavior_cloning_action_type_weights_match_manual_loss(self):
        encoder = self.env.encoder
        model = EntityTransformerPolicy(
            EntityTransformerConfig(
                num_items=encoder.num_items,
                num_statuses=encoder.num_statuses,
                d_model=64,
                nhead=4,
                num_layers=2,
                dim_feedforward=128,
                memory_dim=96,
            )
        )
        batch = tensor_batch(self.observation, batch_size=2)
        sequence = {key: value.unsqueeze(0) for key, value in batch.items()}
        actions = torch.tensor(
            [[[ActionType.MOVE, 0, 0], [ActionType.ROLL, 10, 5]]]
        )
        # Make the requested teacher actions legal for this isolated loss test.
        sequence["type_mask"][0, :, ActionType.MOVE] = True
        sequence["source_mask"][0, :, ActionType.MOVE, 0] = True
        sequence["target_mask"][0, :, ActionType.MOVE, 0, 0] = True
        weights = torch.ones(len(ActionType))
        weights[ActionType.MOVE] = 4.0
        initial_memory = model.initial_memory(2)
        evaluated = model.evaluate_sequence(
            sequence, initial_memory, actions
        )
        expected = -(
            4.0 * evaluated.log_prob[0, 0]
            + evaluated.log_prob[0, 1]
        ) / 5.0
        loss = behavior_cloning_loss(
            model,
            sequence,
            actions,
            initial_memory,
            valid_mask=torch.ones(1, 2, dtype=torch.bool),
            battle_cost=0.0,
            value_cost=0.0,
            action_type_weights=weights,
        )
        self.assertTrue(torch.allclose(loss.action, expected))
        with self.assertRaisesRegex(ValueError, "finite and positive"):
            behavior_cloning_loss(
                model,
                sequence,
                actions,
                initial_memory,
                valid_mask=torch.ones(1, 2, dtype=torch.bool),
                action_type_weights=torch.zeros(len(ActionType)),
            )

    def test_behavior_cloning_combines_example_and_action_type_weights(self):
        encoder = self.env.encoder
        model = EntityTransformerPolicy(
            EntityTransformerConfig(
                num_items=encoder.num_items,
                num_statuses=encoder.num_statuses,
                d_model=64,
                nhead=4,
                num_layers=2,
                dim_feedforward=128,
                memory_dim=96,
            )
        )
        batch = tensor_batch(self.observation, batch_size=2)
        sequence = {key: value.unsqueeze(0) for key, value in batch.items()}
        actions = torch.tensor(
            [[[ActionType.MOVE, 0, 0], [ActionType.ROLL, 10, 5]]]
        )
        sequence["type_mask"][0, :, ActionType.MOVE] = True
        sequence["source_mask"][0, :, ActionType.MOVE, 0] = True
        sequence["target_mask"][0, :, ActionType.MOVE, 0, 0] = True
        type_weights = torch.ones(len(ActionType))
        type_weights[ActionType.MOVE] = 4.0
        example_weights = torch.tensor([[0.25, 2.0]])
        initial_memory = model.initial_memory(2)
        evaluated = model.evaluate_sequence(
            sequence, initial_memory, actions
        )
        expected = -(
            evaluated.log_prob[0, 0] + 2.0 * evaluated.log_prob[0, 1]
        ) / 3.0
        loss = behavior_cloning_loss(
            model,
            sequence,
            actions,
            initial_memory,
            valid_mask=torch.ones(1, 2, dtype=torch.bool),
            battle_cost=0.0,
            value_cost=0.0,
            action_type_weights=type_weights,
            example_weights=example_weights,
        )
        self.assertTrue(torch.allclose(loss.action, expected))


class TestVTraceAndLeague(unittest.TestCase):
    def test_optional_move_action_cost_only_penalizes_move(self):
        self.assertEqual(
            action_shaping_reward(
                [ActionType.MOVE, 0, 1], move_action_cost=0.001
            ),
            -0.001,
        )
        self.assertEqual(
            action_shaping_reward(
                [ActionType.ROLL, 10, 5], move_action_cost=0.001
            ),
            0.0,
        )
        with self.assertRaises(ValueError):
            action_shaping_reward(
                [ActionType.MOVE, 0, 1], move_action_cost=-0.001
            )

    def test_on_policy_vtrace_matches_discounted_return(self):
        behavior = torch.zeros(2, 1)
        target = torch.zeros(2, 1)
        discounts = torch.tensor([[1.0], [0.0]])
        rewards = torch.ones(2, 1)
        values = torch.zeros(2, 1)
        returns = vtrace_from_log_probs(
            behavior, target, discounts, rewards, values, torch.zeros(1)
        )
        self.assertTrue(
            torch.allclose(
                returns.value_targets, torch.tensor([[2.0], [1.0]])
            )
        )

        loss = impala_actor_critic_loss(
            behavior,
            target.requires_grad_(),
            torch.ones(2, 1),
            discounts,
            rewards,
            values.requires_grad_(),
            torch.zeros(1),
        )
        loss.total.backward()
        self.assertTrue(torch.isfinite(loss.total))

    def test_league_round_trip_and_sampling(self):
        league = LeagueManager()
        league.add(LeagueMember("main", "main.pt", role="main"))
        league.add(LeagueMember("old", "old.pt"))
        league.add(LeagueMember("exploit", "exploit.pt", role="exploiter"))
        for _ in range(8):
            league.record_result("main", "old", 1)
            league.record_result("main", "exploit", -1)
        rng = np.random.default_rng(1)
        draws = [league.sample_opponent("main", rng).member_id for _ in range(100)]
        self.assertGreater(draws.count("exploit"), draws.count("old"))

        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "league.json"
            league.save(path)
            restored = LeagueManager.load(path)
            self.assertEqual(set(restored.members), set(league.members))
            self.assertEqual(restored.win_rate("main", "old"), 1.0)

    def test_resident_members_keep_exploiters_and_cap_only_history(self):
        league = LeagueManager()
        for index in range(5):
            league.add(
                LeagueMember(
                    f"exploiter-{index}",
                    f"exploiter-{index}.pt",
                    role="exploiter",
                    created_step=index,
                )
            )
        for index in range(7):
            league.add(
                LeagueMember(
                    f"historical-{index}",
                    f"historical-{index}.pt",
                    role="historical",
                    created_step=100 + index,
                )
            )
        league.promote("champion-1", "champion-1.pt", created_step=999)

        resident_ids = [
            member.member_id for member in league.resident_members(4)
        ]
        self.assertEqual(
            resident_ids[:6],
            [f"exploiter-{index}" for index in range(5)]
            + ["champion-1"],
        )
        self.assertEqual(
            resident_ids[6:],
            [
                "historical-3",
                "historical-4",
                "historical-5",
                "historical-6",
            ],
        )
        self.assertEqual(
            [member.member_id for member in league.resident_members(0)],
            [f"exploiter-{index}" for index in range(5)]
            + ["champion-1"],
        )


class TestParallelSelfPlay(unittest.TestCase):
    @staticmethod
    def sample_action(observation, rng):
        action_type = int(rng.choice(np.flatnonzero(observation["type_mask"])))
        source = int(
            rng.choice(np.flatnonzero(observation["source_mask"][action_type]))
        )
        target = int(
            rng.choice(
                np.flatnonzero(
                    observation["target_mask"][action_type, source]
                )
            )
        )
        return AutoregressiveAction(action_type, source, target)

    def test_explicit_two_player_turn_and_zero_sum_reward(self):
        env = ParallelSelfPlayEnv()
        observations = env.reset(seed=3)
        env.core.agent.gold = 0
        env.core.opponent.gold = 0
        end = AutoregressiveAction(ActionType.END_TURN)
        step = env.step(end, end)
        self.assertTrue(step.battle_resolved)
        self.assertAlmostEqual(sum(step.rewards), 0.0)
        if not step.terminated:
            self.assertEqual(env.active_players(), (True, True))
            self.assertIsNotNone(env.core.agent.lf_winner)
            self.assertIsNotNone(env.core.opponent.lf_winner)
        self.assertEqual(len(observations), 2)

    def test_battle_boundary_shifts_both_opponent_board_histories(self):
        env = ParallelSelfPlayEnv()
        env.reset(seed=303)
        older_zero = Team(["ant"])
        older_one = Team(["fish"])
        env.core.previous_opponent_state = older_one
        env.core.previous_agent_state = older_zero
        env.core.agent.team = Team(["camel"])
        env.core.opponent.team = Team(["peacock"])
        env.core.agent.gold = 0
        env.core.opponent.gold = 0

        step = env.step(
            AutoregressiveAction(ActionType.END_TURN),
            AutoregressiveAction(ActionType.END_TURN),
        )
        self.assertTrue(step.battle_resolved)
        p0 = step.observations[0]
        p1 = step.observations[1]
        self.assertEqual(
            p0["opponent_history_item_ids"][0, 0],
            env.encoder.item_to_id["pet-fish"],
        )
        self.assertEqual(
            p0["opponent_history_item_ids"][1, 0],
            env.encoder.item_to_id["pet-peacock"],
        )
        self.assertEqual(
            p1["opponent_history_item_ids"][0, 0],
            env.encoder.item_to_id["pet-ant"],
        )
        self.assertEqual(
            p1["opponent_history_item_ids"][1, 0],
            env.encoder.item_to_id["pet-camel"],
        )

    def test_turn_two_restoration_has_zero_dense_life_reward(self):
        env = ParallelSelfPlayEnv()
        env.reset(seed=33)
        env.core.turn = 2
        env.core.agent.turn = 2
        env.core.opponent.turn = 2
        env.core.agent.team = Team(["ant"])
        env.core.opponent.team = Team(["ant"])
        env.core.agent.team[0].pet._attack = 50
        env.core.agent.team[0].pet._health = 50
        env.core.opponent.team[0].pet._attack = 1
        env.core.opponent.team[0].pet._health = 1
        env.core.agent.gold = 0
        env.core.opponent.gold = 0

        end = AutoregressiveAction(ActionType.END_TURN)
        step = env.step(end, end)
        self.assertEqual(step.battle_result, 0)
        self.assertEqual(
            (env.core.agent.lives, env.core.opponent.lives), (6, 6)
        )
        self.assertEqual(step.rewards, (0.0, 0.0))
        self.assertIs(env.core.opponent.lf_winner, False)

    def test_league_feedback_uses_completed_match_not_round_result(self):
        env = ParallelSelfPlayEnv()
        env.reset(seed=4)
        env.core.agent.lives = 4
        env.core.agent.wins = 5
        env.core.opponent.lives = 0
        env.core.opponent.wins = 3
        self.assertEqual(_completed_match_result(env.core), 0)
        env.core.agent.lives = 2
        env.core.opponent.lives = 2
        env.core.opponent.wins = 6
        self.assertEqual(_completed_match_result(env.core), 1)

    def test_random_parallel_self_play_rollout(self):
        env = ParallelSelfPlayEnv()
        observations = env.reset(seed=9)
        rng = np.random.default_rng(9)
        battles = 0
        for _ in range(500):
            active = env.active_players()
            actions = [
                self.sample_action(observations[index], rng)
                if active[index]
                else None
                for index in range(2)
            ]
            step = env.step(actions[0], actions[1])
            observations = step.observations
            self.assertAlmostEqual(sum(step.rewards), 0.0)
            battles += int(step.battle_resolved)
            if step.terminated:
                observations = env.reset()
        self.assertGreater(battles, 0)

    def test_batched_self_play_vtrace_update(self):
        probe = ParallelSelfPlayEnv()
        encoder = probe.encoder
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        policy = EntityTransformerPolicy(
            EntityTransformerConfig(
                num_items=encoder.num_items,
                num_statuses=encoder.num_statuses,
                d_model=64,
                nhead=4,
                num_layers=2,
                dim_feedforward=128,
                memory_dim=96,
            )
        ).to(device)
        collector = BatchedSelfPlayCollector(
            policy, num_envs=2, device=device, seed=17
        )
        collector.set_opponent_policy(copy.deepcopy(policy).to(device))
        segments = collector.collect(4)
        optimizer = torch.optim.AdamW(policy.parameters(), lr=3e-4)
        before = next(policy.parameters()).detach().clone()
        battle_head_before = policy.battle_head.weight.detach().clone()
        opponent_head_before = policy.opponent_pet_head.weight.detach().clone()
        metrics = learner_update(
            policy, optimizer, segments, device, battle_aux_cost=0.2
        )
        after = next(policy.parameters()).detach()
        self.assertEqual(len(segments), 4)
        self.assertTrue(
            all(segment.battle_outcome in {0, 1, 2} for segment in segments)
        )
        self.assertGreater(metrics.transitions, 0)
        self.assertTrue(np.isfinite(metrics.total_loss))
        self.assertGreater(metrics.battle_aux_loss, 0)
        self.assertGreater(metrics.opponent_aux_loss, 0)
        self.assertFalse(
            torch.equal(battle_head_before, policy.battle_head.weight)
        )
        self.assertFalse(
            torch.equal(opponent_head_before, policy.opponent_pet_head.weight)
        )
        self.assertFalse(torch.equal(before, after))

    def test_resident_sides_are_excluded_and_discount_is_per_turn(self):
        probe = ParallelSelfPlayEnv()
        encoder = probe.encoder
        policy = EntityTransformerPolicy(
            EntityTransformerConfig(
                num_items=encoder.num_items,
                num_statuses=encoder.num_statuses,
                d_model=32,
                nhead=4,
                num_layers=1,
                dim_feedforward=64,
                memory_dim=48,
            )
        )
        collector = BatchedSelfPlayCollector(
            policy,
            num_envs=1,
            device="cpu",
            seed=117,
            gamma=0.9,
            opponent_policy=copy.deepcopy(policy),
            learn_from_resident_sides=False,
            discount_mode="per_turn",
        )
        segments = collector.collect(2)

        # One learner segment per battle means the frozen resident side was
        # not emitted; with both sides this request would finish in one battle.
        self.assertEqual(collector.envs[0].core.turn, 3)
        self.assertEqual(len(segments), 2)
        for segment in segments:
            self.assertTrue(
                all(
                    transition.discount == 1.0
                    for transition in segment.transitions[:-1]
                )
            )
            self.assertIn(segment.transitions[-1].discount, {0.0, 0.9})
        self.assertTrue(
            any(segment.transitions[-1].discount == 0.9 for segment in segments)
        )

    def test_per_action_discount_retains_legacy_semantics(self):
        probe = ParallelSelfPlayEnv()
        encoder = probe.encoder
        policy = EntityTransformerPolicy(
            EntityTransformerConfig(
                num_items=encoder.num_items,
                num_statuses=encoder.num_statuses,
                d_model=32,
                nhead=4,
                num_layers=1,
                dim_feedforward=64,
                memory_dim=48,
            )
        )
        collector = BatchedSelfPlayCollector(
            policy,
            num_envs=1,
            device="cpu",
            seed=118,
            gamma=0.9,
            discount_mode="per_action",
        )
        segment = collector.collect(1)[0]
        self.assertTrue(
            all(
                transition.discount == 0.9
                for transition in segment.transitions
            )
        )

    def test_shared_ring_preserves_maximum_shop_turn(self):
        env = ParallelSelfPlayEnv()
        observation = env.reset(seed=19)[0]
        context = mp.get_context("spawn")
        ring = SharedTrajectoryRing.create(
            observation,
            memory_dim=8,
            context=context,
            capacity=2,
            max_steps=64,
        )
        transition = Transition(
            observation=observation,
            action=np.array([0, 10, 5], dtype=np.int64),
            behavior_log_prob=-1.0,
            behavior_version=7,
        )
        segment = TurnSegment(
            transitions=[copy.deepcopy(transition) for _ in range(51)],
            initial_memory=torch.zeros(8),
            bootstrap_observation=observation,
            terminal=False,
            battle_outcome=1,
        )
        stop = context.Event()
        self.assertTrue(ring.write(segment, stop))
        restored = ring.read(timeout=1.0)
        self.assertEqual(len(restored.transitions), 51)
        self.assertTrue(
            all(item.behavior_version == 7 for item in restored.transitions)
        )
        self.assertEqual(restored.battle_outcome, 1)
        self.assertEqual(ring.stats()["ready"], 0)
        self.assertEqual(ring.stats()["high_water"], 1)
        self.assertEqual(ring.stats()["written"], 1)
        self.assertEqual(ring.stats()["read"], 1)
        ring.close()

    def test_missing_resident_refuses_to_reinterpret_its_memory_as_main(self):
        env = ParallelSelfPlayEnv()
        observation = env.reset(seed=29)[0]
        context = mp.get_context("spawn")
        config = EntityTransformerConfig(
            num_items=env.encoder.num_items,
            num_statuses=env.encoder.num_statuses,
            d_model=64,
            nhead=4,
            num_layers=2,
            dim_feedforward=128,
            memory_dim=96,
        )
        policy = EntityTransformerPolicy(config)
        shared = SharedActorBuffers.create(
            observation, config.memory_dim, context, num_slots=1
        )
        for key, value in observation.items():
            shared.observations[key][0].copy_(torch.as_tensor(value))
        shared.active[0] = 1
        shared.policy_ids[0] = 12345  # Removed during a pool rotation.
        stop = context.Event()
        batcher = DynamicInferenceBatcher(
            policy,
            [shared],
            device="cpu",
            model_lock=threading.RLock(),
            stop_event=stop,
            max_batch_delay_ms=5.0,
        )
        self.assertAlmostEqual(batcher.max_batch_delay, 0.005)
        batcher.policy_version = 3
        with self.assertRaisesRegex(RuntimeError, "cross-policy memory"):
            batcher._serve([0])
        self.assertFalse(shared.response_event.is_set())


if __name__ == "__main__":
    unittest.main()
