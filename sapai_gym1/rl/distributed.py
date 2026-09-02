"""Multiprocess self-play actors with shared-memory dynamic GPU inference."""

from __future__ import annotations

from contextlib import nullcontext
import copy
from dataclasses import dataclass
import multiprocessing as mp
from queue import Empty, Full
import threading
import time

import numpy as np
import torch

from .action_schema import AutoregressiveAction
from .model import EntityTransformerPolicy
from .selfplay_env import ParallelSelfPlayEnv
from .training import (
    Transition,
    TurnSegment,
    action_shaping_reward,
    stack_observations,
)


def _shared_tensor(shape, dtype):
    return torch.zeros(shape, dtype=dtype).share_memory_()


@dataclass
class SharedActorBuffers:
    observations: dict[str, torch.Tensor]
    memories: torch.Tensor
    active: torch.Tensor
    policy_ids: torch.Tensor
    policy_versions: torch.Tensor
    actions: torch.Tensor
    log_probs: torch.Tensor
    next_memories: torch.Tensor
    request_event: object
    response_event: object

    @classmethod
    def create(cls, sample_observation, memory_dim, context, num_slots=2):
        observations = {}
        for key, value in sample_observation.items():
            tensor = torch.as_tensor(value)
            observations[key] = _shared_tensor(
                (num_slots, *tensor.shape), tensor.dtype
            )
        return cls(
            observations=observations,
            memories=_shared_tensor((num_slots, memory_dim), torch.float32),
            active=_shared_tensor((num_slots,), torch.int8),
            policy_ids=_shared_tensor((num_slots,), torch.int64),
            policy_versions=_shared_tensor((num_slots,), torch.int64),
            actions=_shared_tensor((num_slots, 3), torch.int64),
            log_probs=_shared_tensor((num_slots,), torch.float32),
            next_memories=_shared_tensor((num_slots, memory_dim), torch.float32),
            request_event=context.Event(),
            response_event=context.Event(),
        )


@dataclass
class SharedPolicyRouter:
    policy_ids: torch.Tensor
    probabilities: torch.Tensor
    count: torch.Tensor

    @classmethod
    def create(cls, max_policies=16):
        return cls(
            policy_ids=_shared_tensor((max_policies,), torch.int64),
            probabilities=_shared_tensor((max_policies,), torch.float32),
            count=_shared_tensor((1,), torch.int32),
        )

    def update(self, policy_ids, probabilities=None):
        policy_ids = list(policy_ids)
        if len(policy_ids) > len(self.policy_ids):
            raise ValueError("Resident policy pool exceeds shared router capacity")
        if not policy_ids:
            self.count[0] = 0
            return
        if probabilities is None:
            probabilities = np.ones(len(policy_ids), dtype=np.float64)
        probabilities = np.asarray(probabilities, dtype=np.float64)
        if len(probabilities) != len(policy_ids):
            raise ValueError("Each policy needs one sampling probability")
        probabilities = probabilities / probabilities.sum()
        # Publish count last so actors never read a partially written pool.
        self.count[0] = 0
        if policy_ids:
            self.policy_ids[: len(policy_ids)] = torch.tensor(policy_ids)
            self.probabilities[: len(policy_ids)] = torch.tensor(
                probabilities, dtype=torch.float32
            )
        self.count[0] = len(policy_ids)

    def sample(self, rng):
        count = int(self.count[0])
        if count == 0:
            return 0
        policy_ids = self.policy_ids[:count].numpy().copy()
        probabilities = self.probabilities[:count].numpy().copy()
        probabilities /= probabilities.sum()
        return int(rng.choice(policy_ids, p=probabilities))


@dataclass
class SharedTrajectoryRing:
    observations: dict[str, torch.Tensor]
    bootstrap_observations: dict[str, torch.Tensor]
    actions: torch.Tensor
    behavior_log_probs: torch.Tensor
    behavior_versions: torch.Tensor
    rewards: torch.Tensor
    discounts: torch.Tensor
    lengths: torch.Tensor
    initial_memories: torch.Tensor
    terminals: torch.Tensor
    battle_outcomes: torch.Tensor
    free_queue: object
    ready_queue: object
    ready_count: object
    ready_high_water: object
    segments_written: object
    segments_read: object
    capacity: int
    max_steps: int

    @classmethod
    def create(
        cls,
        sample_observation,
        memory_dim,
        context,
        capacity=512,
        max_steps=64,
    ):
        observations = {}
        bootstrap_observations = {}
        for key, value in sample_observation.items():
            tensor = torch.as_tensor(value)
            observations[key] = _shared_tensor(
                (capacity, max_steps, *tensor.shape), tensor.dtype
            )
            bootstrap_observations[key] = _shared_tensor(
                (capacity, *tensor.shape), tensor.dtype
            )
        free_queue = context.Queue(maxsize=capacity)
        ready_queue = context.Queue(maxsize=capacity)
        for slot_index in range(capacity):
            free_queue.put(slot_index)
        ready_count = context.Value("i", 0)
        ready_high_water = context.Value("i", 0)
        segments_written = context.Value("q", 0)
        segments_read = context.Value("q", 0)
        return cls(
            observations=observations,
            bootstrap_observations=bootstrap_observations,
            actions=_shared_tensor((capacity, max_steps, 3), torch.int64),
            behavior_log_probs=_shared_tensor(
                (capacity, max_steps), torch.float32
            ),
            behavior_versions=_shared_tensor(
                (capacity, max_steps), torch.int64
            ),
            rewards=_shared_tensor((capacity, max_steps), torch.float32),
            discounts=_shared_tensor((capacity, max_steps), torch.float32),
            lengths=_shared_tensor((capacity,), torch.int32),
            initial_memories=_shared_tensor(
                (capacity, memory_dim), torch.float32
            ),
            terminals=_shared_tensor((capacity,), torch.int8),
            battle_outcomes=_shared_tensor((capacity,), torch.int8),
            free_queue=free_queue,
            ready_queue=ready_queue,
            ready_count=ready_count,
            ready_high_water=ready_high_water,
            segments_written=segments_written,
            segments_read=segments_read,
            capacity=capacity,
            max_steps=max_steps,
        )

    def write(self, segment: TurnSegment, stop_event):
        length = len(segment.transitions)
        if length > self.max_steps:
            raise ValueError(
                f"Segment length {length} exceeds shared maximum {self.max_steps}"
            )
        while not stop_event.is_set():
            try:
                slot_index = self.free_queue.get(timeout=0.1)
                break
            except Empty:
                continue
        else:
            return False

        self.lengths[slot_index] = length
        self.terminals[slot_index] = int(segment.terminal)
        self.battle_outcomes[slot_index] = (
            -1 if segment.battle_outcome is None else int(segment.battle_outcome)
        )
        self.initial_memories[slot_index].copy_(segment.initial_memory)
        for time_index, transition in enumerate(segment.transitions):
            for key, value in transition.observation.items():
                self.observations[key][slot_index, time_index].copy_(
                    torch.as_tensor(value)
                )
            self.actions[slot_index, time_index].copy_(
                torch.as_tensor(transition.action)
            )
            self.behavior_log_probs[slot_index, time_index] = (
                transition.behavior_log_prob
            )
            self.behavior_versions[slot_index, time_index] = (
                transition.behavior_version
            )
            self.rewards[slot_index, time_index] = transition.reward
            self.discounts[slot_index, time_index] = transition.discount
        for key, value in segment.bootstrap_observation.items():
            self.bootstrap_observations[key][slot_index].copy_(
                torch.as_tensor(value)
            )
        while not stop_event.is_set():
            try:
                with self.ready_count.get_lock():
                    self.ready_count.value += 1
                    self.ready_high_water.value = max(
                        self.ready_high_water.value,
                        self.ready_count.value,
                    )
                self.ready_queue.put(slot_index, timeout=0.1)
                with self.segments_written.get_lock():
                    self.segments_written.value += 1
                return True
            except Full:
                with self.ready_count.get_lock():
                    self.ready_count.value -= 1
                continue
        self.free_queue.put(slot_index)
        return False

    def read(self, timeout):
        slot_index = self.ready_queue.get(timeout=timeout)
        with self.ready_count.get_lock():
            self.ready_count.value -= 1
        with self.segments_read.get_lock():
            self.segments_read.value += 1
        length = int(self.lengths[slot_index])
        transitions = []
        for time_index in range(length):
            observation = {
                key: values[slot_index, time_index].numpy().copy()
                for key, values in self.observations.items()
            }
            transitions.append(
                Transition(
                    observation=observation,
                    action=self.actions[slot_index, time_index].numpy().copy(),
                    behavior_log_prob=float(
                        self.behavior_log_probs[slot_index, time_index]
                    ),
                    behavior_version=int(
                        self.behavior_versions[slot_index, time_index]
                    ),
                    reward=float(self.rewards[slot_index, time_index]),
                    discount=float(self.discounts[slot_index, time_index]),
                )
            )
        segment = TurnSegment(
            transitions=transitions,
            initial_memory=self.initial_memories[slot_index].clone(),
            bootstrap_observation={
                key: values[slot_index].numpy().copy()
                for key, values in self.bootstrap_observations.items()
            },
            terminal=bool(self.terminals[slot_index]),
            battle_outcome=(
                None
                if int(self.battle_outcomes[slot_index]) < 0
                else int(self.battle_outcomes[slot_index])
            ),
        )
        self.free_queue.put(slot_index)
        return segment

    def stats(self):
        """Return exact shared occupancy/counter telemetry."""

        with self.ready_count.get_lock():
            ready = int(self.ready_count.value)
        with self.ready_high_water.get_lock():
            high_water = int(self.ready_high_water.value)
        with self.segments_written.get_lock():
            written = int(self.segments_written.value)
        with self.segments_read.get_lock():
            read = int(self.segments_read.value)
        return {
            "ready": ready,
            "high_water": high_water,
            "written": written,
            "read": read,
            "capacity": self.capacity,
        }

    def close(self):
        for queue in (self.free_queue, self.ready_queue):
            queue.cancel_join_thread()
            queue.close()


def _write_observation(shared, slot_index, observation):
    for key, value in observation.items():
        shared.observations[key][slot_index].copy_(torch.as_tensor(value))


def _completed_match_result(core):
    """Return 0/1/2 from player-zero perspective for a terminated match."""

    if core.opponent.lives <= 0 < core.agent.lives:
        return 0
    if core.agent.lives <= 0 < core.opponent.lives:
        return 1
    player_score = (core.agent.lives, core.agent.wins)
    opponent_score = (core.opponent.lives, core.opponent.wins)
    if player_score > opponent_score:
        return 0
    if opponent_score > player_score:
        return 1
    return 2


def _actor_main(
    actor_id,
    shared: SharedActorBuffers,
    trajectory_ring: SharedTrajectoryRing,
    result_queue,
    stop_event,
    seed,
    gamma,
    policy_router: SharedPolicyRouter,
    games_per_actor,
    move_action_cost,
    learn_from_resident_sides,
    discount_mode,
):
    # Prevent every actor from creating its own large BLAS thread pool.
    torch.set_num_threads(1)
    trajectory_ring.ready_queue.cancel_join_thread()
    trajectory_ring.free_queue.cancel_join_thread()
    result_queue.cancel_join_thread()
    rng = np.random.default_rng(seed)
    envs = [ParallelSelfPlayEnv() for _ in range(games_per_actor)]
    observations = [
        env.reset(seed=int(rng.integers(0, 2**31 - 1))) for env in envs
    ]
    memory_dim = shared.memories.shape[1]
    memories = np.zeros((games_per_actor, 2, memory_dim), dtype=np.float32)
    buffers = [[[], []] for _ in range(games_per_actor)]
    initial_memories = [
        [torch.zeros(memory_dim), torch.zeros(memory_dim)]
        for _ in range(games_per_actor)
    ]
    opponent_policy_ids = [
        policy_router.sample(rng) for _ in range(games_per_actor)
    ]

    while not stop_event.is_set():
        shared.active.zero_()
        active_by_game = []
        for game_index, env in enumerate(envs):
            active = env.active_players()
            active_by_game.append(active)
            for side in range(2):
                if not active[side]:
                    continue
                slot_index = game_index * 2 + side
                shared.active[slot_index] = 1
                shared.policy_ids[slot_index] = (
                    opponent_policy_ids[game_index] if side == 1 else 0
                )
                _write_observation(
                    shared, slot_index, observations[game_index][side]
                )
                shared.memories[slot_index].copy_(
                    torch.from_numpy(memories[game_index, side])
                )

        shared.response_event.clear()
        shared.request_event.set()
        while not stop_event.is_set():
            if shared.response_event.wait(timeout=0.1):
                break
        if stop_event.is_set():
            break

        for game_index, env in enumerate(envs):
            active = active_by_game[game_index]
            actions = [None, None]
            for side in range(2):
                if not active[side]:
                    continue
                slot_index = game_index * 2 + side
                action_array = shared.actions[slot_index].numpy().copy()
                actions[side] = AutoregressiveAction(*map(int, action_array))
                learns_from_side = (
                    side == 0
                    or opponent_policy_ids[game_index] == 0
                    or learn_from_resident_sides
                )
                if learns_from_side:
                    buffers[game_index][side].append(
                        Transition(
                            observation=observations[game_index][side],
                            action=action_array,
                            behavior_log_prob=float(shared.log_probs[slot_index]),
                            behavior_version=int(
                                shared.policy_versions[slot_index]
                            ),
                            reward=action_shaping_reward(
                                action_array, move_action_cost
                            ),
                            discount=(
                                gamma if discount_mode == "per_action" else 1.0
                            ),
                        )
                    )
                memories[game_index, side] = (
                    shared.next_memories[slot_index].numpy().copy()
                )

            step = env.step(actions[0], actions[1])
            observations[game_index] = step.observations
            if not step.battle_resolved:
                continue

            if opponent_policy_ids[game_index] != 0 and step.terminated:
                while not stop_event.is_set():
                    try:
                        result_queue.put(
                            (
                                opponent_policy_ids[game_index],
                                _completed_match_result(env.core),
                            ),
                            timeout=0.1,
                        )
                        break
                    except Full:
                        continue

            for side in range(2):
                side_buffer = buffers[game_index][side]
                learns_from_side = (
                    side == 0
                    or opponent_policy_ids[game_index] == 0
                    or learn_from_resident_sides
                )
                if not learns_from_side:
                    buffers[game_index][side] = []
                    initial_memories[game_index][side] = torch.from_numpy(
                        memories[game_index, side].copy()
                    )
                    continue
                if not side_buffer:
                    raise RuntimeError("Battle resolved without actor transitions")
                side_buffer[-1].reward += step.rewards[side]
                if step.terminated:
                    side_buffer[-1].discount = 0.0
                elif discount_mode == "per_turn":
                    side_buffer[-1].discount = gamma
                segment = TurnSegment(
                    transitions=side_buffer,
                    initial_memory=initial_memories[game_index][side],
                    bootstrap_observation=step.observations[side],
                    terminal=step.terminated,
                    battle_outcome=(
                        2
                        if step.battle_result == 2
                        else 0
                        if step.battle_result == side
                        else 1
                    ),
                )
                if not trajectory_ring.write(segment, stop_event):
                    break
                buffers[game_index][side] = []
                initial_memories[game_index][side] = torch.from_numpy(
                    memories[game_index, side].copy()
                )

            if step.terminated:
                observations[game_index] = env.reset(
                    seed=int(rng.integers(0, 2**31 - 1))
                )
                memories[game_index].fill(0.0)
                initial_memories[game_index] = [
                    torch.zeros(memory_dim),
                    torch.zeros(memory_dim),
                ]
                opponent_policy_ids[game_index] = policy_router.sample(rng)


class DynamicInferenceBatcher:
    """Continuously merge actor requests into main/historical GPU batches."""

    def __init__(
        self,
        main_policy: EntityTransformerPolicy,
        actor_buffers: list[SharedActorBuffers],
        device: torch.device | str,
        model_lock: threading.RLock,
        stop_event,
        resident_policies: dict[int, EntityTransformerPolicy] | None = None,
        max_batch_delay_ms: float = 2.0,
        use_bfloat16: bool = True,
    ):
        self.main_policy = main_policy
        self.resident_policies = {0: main_policy}
        if resident_policies:
            self.resident_policies.update(resident_policies)
        self.actor_buffers = actor_buffers
        self.device = torch.device(device)
        self.model_lock = model_lock
        self.stop_event = stop_event
        self.max_batch_delay = max_batch_delay_ms / 1000
        self.use_bfloat16 = use_bfloat16 and self.device.type == "cuda"
        self.thread = threading.Thread(target=self._run, daemon=True)
        self.inference_batches = 0
        self.inference_items = 0
        self.inference_seconds = 0.0
        self.coalesce_seconds = 0.0
        self.policy_version = 0

    def start(self):
        self.thread.start()

    def stop(self):
        self.thread.join(timeout=5)

    def set_policy_pool(self, policies):
        with self.model_lock:
            self.resident_policies = {0: self.main_policy, **policies}
            for policy in policies.values():
                policy.eval().requires_grad_(False)

    def _run(self):
        while not self.stop_event.is_set():
            ready = [
                index
                for index, shared in enumerate(self.actor_buffers)
                if shared.request_event.is_set()
            ]
            if not ready:
                time.sleep(0.0005)
                continue
            coalesce_started = time.perf_counter()
            deadline = time.perf_counter() + self.max_batch_delay
            while time.perf_counter() < deadline:
                additional = [
                    index
                    for index, shared in enumerate(self.actor_buffers)
                    if shared.request_event.is_set() and index not in ready
                ]
                ready.extend(additional)
                if len(ready) == len(self.actor_buffers):
                    break
                time.sleep(0.0001)
            self.coalesce_seconds += time.perf_counter() - coalesce_started
            self._serve(ready)

    @torch.no_grad()
    def _serve(self, ready):
        inference_started = time.perf_counter()
        with self.model_lock:
            # Pool rotation and request routing must be atomic. Actors can
            # already have an old policy ID in shared memory when the learner
            # removes that resident at a checkpoint. Route such in-flight
            # requests to main while holding the same lock used by
            # set_policy_pool(), rather than checking before the lock and then
            # indexing a changed dictionary.
            references_by_policy = {}
            for actor_index in ready:
                shared = self.actor_buffers[actor_index]
                shared.request_event.clear()
                for slot_index in range(len(shared.active)):
                    if int(shared.active[slot_index]) == 0:
                        continue
                    policy_id = int(shared.policy_ids[slot_index])
                    if policy_id not in self.resident_policies:
                        policy_id = 0
                    references_by_policy.setdefault(policy_id, []).append(
                        (actor_index, slot_index)
                    )

            for policy_id, references in references_by_policy.items():
                if not references:
                    continue
                policy = self.resident_policies[policy_id]
                policy.eval()
                observations = {
                    key: torch.stack(
                        [
                            self.actor_buffers[actor].observations[key][slot_index]
                            for actor, slot_index in references
                        ]
                    ).to(self.device)
                    for key in self.actor_buffers[0].observations
                }
                memories = torch.stack(
                    [
                        self.actor_buffers[actor].memories[slot_index]
                        for actor, slot_index in references
                    ]
                ).to(self.device)
                autocast = (
                    torch.autocast("cuda", dtype=torch.bfloat16)
                    if self.use_bfloat16
                    else nullcontext()
                )
                with autocast:
                    output = policy(observations, memory=memories)
                for batch_index, (actor, slot_index) in enumerate(references):
                    shared = self.actor_buffers[actor]
                    shared.actions[slot_index].copy_(
                        output.actions[batch_index].cpu()
                    )
                    shared.log_probs[slot_index] = float(
                        output.log_prob[batch_index]
                    )
                    shared.next_memories[slot_index].copy_(
                        output.memory[batch_index].float().cpu()
                    )
                    shared.policy_versions[slot_index] = (
                        self.policy_version if policy_id == 0 else -1
                    )
                self.inference_batches += 1
                self.inference_items += len(references)

        for actor_index in ready:
            self.actor_buffers[actor_index].response_event.set()
        self.inference_seconds += time.perf_counter() - inference_started


class DistributedSelfPlayRuntime:
    """Lifecycle wrapper for shared actors, inference thread, and segment queue."""

    def __init__(
        self,
        policy: EntityTransformerPolicy,
        num_actors: int,
        sample_observation: dict,
        device: torch.device | str,
        seed: int = 0,
        gamma: float = 0.99,
        queue_size: int = 512,
        games_per_actor: int = 1,
        historical_policy: EntityTransformerPolicy | None = None,
        use_historical_opponent: bool = False,
        move_action_cost: float = 0.0,
        inference_batch_delay_ms: float = 2.0,
        learn_from_resident_sides: bool = True,
        discount_mode: str = "per_action",
    ):
        self.policy = policy
        self.device = torch.device(device)
        if move_action_cost < 0:
            raise ValueError("move_action_cost must be non-negative")
        if inference_batch_delay_ms < 0:
            raise ValueError("inference_batch_delay_ms must be non-negative")
        if discount_mode not in {"per_action", "per_turn"}:
            raise ValueError("discount_mode must be 'per_action' or 'per_turn'")
        self.context = mp.get_context("spawn")
        self.stop_event = self.context.Event()
        self.trajectory_ring = SharedTrajectoryRing.create(
            sample_observation,
            policy.config.memory_dim,
            self.context,
            capacity=queue_size,
        )
        self.result_queue = self.context.Queue(maxsize=4096)
        self.model_lock = threading.RLock()
        self.inference_policy = copy.deepcopy(policy).to(self.device).eval()
        self.inference_policy.requires_grad_(False)
        self.policy_router = SharedPolicyRouter.create()
        initial_resident_policies = {}
        if historical_policy is not None:
            initial_resident_policies[1] = historical_policy
        if use_historical_opponent:
            self.policy_router.update([1] if historical_policy is not None else [0])
        else:
            self.policy_router.update([0])
        self.actor_buffers = [
            SharedActorBuffers.create(
                sample_observation,
                policy.config.memory_dim,
                self.context,
                num_slots=games_per_actor * 2,
            )
            for _ in range(num_actors)
        ]
        self.batcher = DynamicInferenceBatcher(
            self.inference_policy,
            self.actor_buffers,
            self.device,
            self.model_lock,
            self.stop_event,
            resident_policies=initial_resident_policies,
            max_batch_delay_ms=inference_batch_delay_ms,
        )
        self.processes = [
            self.context.Process(
                target=_actor_main,
                args=(
                    actor_id,
                    self.actor_buffers[actor_id],
                    self.trajectory_ring,
                    self.result_queue,
                    self.stop_event,
                    seed + actor_id * 1009,
                    gamma,
                    self.policy_router,
                    games_per_actor,
                    float(move_action_cost),
                    bool(learn_from_resident_sides),
                    discount_mode,
                ),
                daemon=True,
            )
            for actor_id in range(num_actors)
        ]

    def start(self):
        self.batcher.start()
        for process in self.processes:
            process.start()

    def collect_segments(self, count: int, timeout: float = 120.0):
        segments = []
        deadline = time.monotonic() + timeout
        while len(segments) < count:
            failed = [
                (index, process.exitcode)
                for index, process in enumerate(self.processes)
                if process.exitcode is not None and process.exitcode != 0
            ]
            if failed:
                raise RuntimeError(f"Actor process failed: {failed}")
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise TimeoutError(
                    f"Received {len(segments)}/{count} segments before timeout"
                )
            try:
                segments.append(self.trajectory_ring.read(min(1.0, remaining)))
            except Empty:
                failed = [p.exitcode for p in self.processes if p.exitcode]
                if failed:
                    raise RuntimeError(f"Actor process failed: {failed}")
        return segments

    def set_policy_pool(
        self, policies, probabilities=None, main_probability=1.0
    ):
        """Publish current plus resident policies and sampling weights."""

        self.batcher.set_policy_pool(policies)
        policy_ids = list(policies.keys())
        if probabilities is None:
            probabilities = [1.0] * len(policy_ids)
        else:
            probabilities = list(probabilities)
        if main_probability > 0:
            policy_ids = [0] + policy_ids
            probabilities = [float(main_probability)] + probabilities
        self.policy_router.update(policy_ids, probabilities)

    def drain_match_results(self):
        results = []
        while True:
            try:
                results.append(self.result_queue.get_nowait())
            except Empty:
                return results

    def publish_main_policy(self):
        """Briefly publish learner weights to the actor inference replica."""

        with self.model_lock:
            self.inference_policy.load_state_dict(self.policy.state_dict())
            self.inference_policy.eval()
            self.batcher.policy_version += 1

    def set_historical_policy(self, policy):
        # Compatibility helper for a single frozen snapshot.
        self.set_policy_pool({1: policy}, main_probability=0.0)

    def stop(self):
        self.stop_event.set()
        for shared in self.actor_buffers:
            shared.response_event.set()
        self.batcher.stop()
        deadline = time.monotonic() + 2.0
        for process in self.processes:
            process.join(timeout=max(0.0, deadline - time.monotonic()))
        for process in self.processes:
            if process.is_alive():
                process.terminate()
                process.join(timeout=2)
        self.trajectory_ring.close()
        self.result_queue.cancel_join_thread()
        self.result_queue.close()

    def __enter__(self):
        self.start()
        return self

    def __exit__(self, exc_type, exc, traceback):
        self.stop()
