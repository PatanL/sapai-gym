"""Multiprocess self-play actors with shared-memory dynamic GPU inference."""

from __future__ import annotations

from contextlib import nullcontext
from collections import Counter, defaultdict, deque
import copy
from dataclasses import dataclass, field
import multiprocessing as mp
from queue import Empty, Full
import threading
import time

import numpy as np
import torch

from .action_schema import AutoregressiveAction
from .model import EntityTransformerPolicy
from .selfplay_env import ParallelSelfPlayEnv
from .end_freeze import NUM_FREEZE_SUBSETS, step_with_end_freeze
from .training import (
    Transition,
    TurnSegment,
    action_shaping_reward,
    stack_observations,
)


def _shared_tensor(shape, dtype):
    return torch.zeros(shape, dtype=dtype).share_memory_()


def inference_results_on_cpu(output):
    """Complete whole-field device copies before publishing actor responses."""
    masks = getattr(output, 'end_freeze_mask', None)
    return (
        output.actions.detach().cpu(),
        output.log_prob.detach().float().cpu(),
        output.memory.detach().float().cpu(),
        None if masks is None else masks.detach().cpu(),
    )


@dataclass
class SharedActorBuffers:
    observations: dict[str, torch.Tensor]
    memories: torch.Tensor
    active: torch.Tensor
    policy_ids: torch.Tensor
    policy_versions: torch.Tensor
    actions: torch.Tensor
    end_freeze_masks: torch.Tensor
    log_probs: torch.Tensor
    next_memories: torch.Tensor
    request_event: object
    response_event: object
    telemetry_enabled: bool = False
    request_time: torch.Tensor | None = None
    request_generation: torch.Tensor | None = None
    completed_generation: torch.Tensor | None = None
    response_latency_ms: torch.Tensor | None = None
    completed_at: torch.Tensor | None = None
    completed_request_started: torch.Tensor | None = None

    @classmethod
    def create(cls, sample_observation, memory_dim, context, num_slots=2, telemetry=False):
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
            end_freeze_masks=_shared_tensor((num_slots,), torch.int16).fill_(-1),
            log_probs=_shared_tensor((num_slots,), torch.float32),
            next_memories=_shared_tensor((num_slots, memory_dim), torch.float32),
            request_event=context.Event(),
            response_event=context.Event(),
            telemetry_enabled=telemetry,
            request_time=_shared_tensor((1,), torch.float64),
            request_generation=_shared_tensor((1,), torch.int64),
            completed_generation=_shared_tensor((1,), torch.int64),
            response_latency_ms=_shared_tensor((1,), torch.float64),
            completed_at=_shared_tensor((1,), torch.float64),
            completed_request_started=_shared_tensor((1,), torch.float64),
        )


@dataclass
class SharedPolicyRouter:
    policy_ids: torch.Tensor
    probabilities: torch.Tensor
    count: torch.Tensor
    leases: torch.Tensor
    lock: object

    @classmethod
    def create(cls, max_policies=16, num_leases=0, context=None):
        context = context or mp.get_context("spawn")
        return cls(
            policy_ids=_shared_tensor((max_policies,), torch.int64),
            probabilities=_shared_tensor((max_policies,), torch.float32),
            count=_shared_tensor((1,), torch.int32),
            leases=_shared_tensor((num_leases,), torch.int64),
            lock=context.RLock(),
        )

    def update(self, policy_ids, probabilities=None):
        with self.lock:
            self._update(policy_ids, probabilities)

    def _update(self, policy_ids, probabilities=None):
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

    def sample(self, rng, lease_index=None):
        # Publish the game's assignment in the same transaction as sampling.
        # A pool update cannot retire a sampled policy before its actor records
        # that the match is using it, including between shop turns.
        with self.lock:
            count = int(self.count[0])
            if count == 0:
                policy_id = 0
            else:
                policy_ids = self.policy_ids[:count].numpy().copy()
                probabilities = self.probabilities[:count].numpy().copy()
                probabilities /= probabilities.sum()
                policy_id = int(rng.choice(policy_ids, p=probabilities))
            if lease_index is not None:
                self.leases[lease_index] = policy_id
            return policy_id


@dataclass(frozen=True)
class TrajectoryLease:
    slot: int
    generation: int


@dataclass
class SharedTrajectoryRing:
    observations: dict[str, torch.Tensor]
    bootstrap_observations: dict[str, torch.Tensor]
    actions: torch.Tensor
    end_freeze_masks: torch.Tensor
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
    _read_leases: dict[int, int] = field(default_factory=dict)
    _lease_generation: int = 0

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
            end_freeze_masks=_shared_tensor((capacity, max_steps), torch.int16).fill_(-1),
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
        for transition in segment.transitions:
            mask = transition.end_freeze_mask
            if mask is not None and (isinstance(mask, bool) or not isinstance(mask, (int, np.integer))
                                     or not 0 <= mask < NUM_FREEZE_SUBSETS):
                raise ValueError('Invalid end-freeze mask in trajectory')
        if length < 1 or length > self.max_steps:
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

        try:
            self._copy_segment_into_slot(slot_index, segment)
        except BaseException:
            # No ready publication occurred: return the producer's lease even
            # on malformed shapes or copy exceptions, then fail explicitly.
            self.free_queue.put(slot_index)
            raise
        while not stop_event.is_set():
            try:
                with self.ready_count.get_lock():
                    self.ready_count.value += 1
                    self.ready_high_water.value = max(self.ready_high_water.value, self.ready_count.value)
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

    def _copy_segment_into_slot(self, slot_index, segment):
        length = len(segment.transitions)
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
            self.end_freeze_masks[slot_index, time_index] = (
                -1 if transition.end_freeze_mask is None else transition.end_freeze_mask
            )
            self.rewards[slot_index, time_index] = transition.reward
            self.discounts[slot_index, time_index] = transition.discount
        for key, value in segment.bootstrap_observation.items():
            self.bootstrap_observations[key][slot_index].copy_(
                torch.as_tensor(value)
            )

    def read(self, timeout):
        lease = self.lease_ready(1, timeout)[0]
        try:
            return self._read_legacy_slot(lease.slot)
        finally:
            self.release_lease(lease)

    def lease_ready(self, max_count, timeout):
        """Single-consumer FIFO: wait for first item, then drain a bounded chunk."""
        if max_count <= 0 or max_count > self.capacity:
            raise ValueError('Invalid lease batch size')
        leases = []
        try:
            for i in range(max_count):
                try:
                    slot = self.ready_queue.get(timeout=timeout) if i == 0 else self.ready_queue.get_nowait()
                except Empty:
                    if not leases:
                        raise
                    break
                if slot in self._read_leases:
                    raise RuntimeError('Duplicate ready slot would violate trajectory ownership')
                self._lease_generation += 1
                lease = TrajectoryLease(slot, self._lease_generation)
                self._read_leases[slot] = lease.generation
                leases.append(lease)
                with self.ready_count.get_lock():
                    self.ready_count.value -= 1
                with self.segments_read.get_lock():
                    self.segments_read.value += 1
            return leases
        except BaseException:
            for lease in leases:
                self.release_lease(lease)
            raise

    def release_lease(self, lease):
        if self._read_leases.get(lease.slot) != lease.generation:
            raise RuntimeError('Unknown, stale, or already released trajectory lease')
        self.free_queue.put(lease.slot)
        del self._read_leases[lease.slot]

    def read_dense(self, max_count, timeout):
        from .dense_trajectory import OwnedTrajectoryBatch
        leases = self.lease_ready(max_count, timeout)
        try:
            return OwnedTrajectoryBatch.copy_from_ring(self, [lease.slot for lease in leases])
        finally:
            # Even copy exceptions return every lease; no partial batch is
            # published to the learner, and no owned tensor aliases the ring.
            for lease in leases:
                self.release_lease(lease)

    def _read_legacy_slot(self, slot_index):
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
                    end_freeze_mask=(
                        None if int(self.end_freeze_masks[slot_index, time_index]) == -1
                        else int(self.end_freeze_masks[slot_index, time_index])
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
        for slot, generation in list(self._read_leases.items()):
            self.release_lease(TrajectoryLease(slot, generation))
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
        policy_router.sample(rng, actor_id * games_per_actor + game_index)
        for game_index in range(games_per_actor)
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
        if shared.telemetry_enabled:
            shared.request_generation[0] += 1
            shared.request_time[0] = time.perf_counter()
        shared.request_event.set()
        while not stop_event.is_set():
            if shared.response_event.wait(timeout=0.1):
                break
        if stop_event.is_set():
            break

        if shared.telemetry_enabled:
            completed_at = time.perf_counter()
            shared.response_latency_ms[0] = 1000. * (completed_at - float(shared.request_time[0]))
            shared.completed_at[0] = completed_at
            shared.completed_request_started[0] = shared.request_time[0]
            shared.completed_generation[0] = shared.request_generation[0]

        for game_index, env in enumerate(envs):
            active = active_by_game[game_index]
            actions = [None, None]
            freeze_masks = [None, None]
            for side in range(2):
                if not active[side]:
                    continue
                slot_index = game_index * 2 + side
                action_array = shared.actions[slot_index].numpy().copy()
                actions[side] = AutoregressiveAction(*map(int, action_array))
                mask_id = int(shared.end_freeze_masks[slot_index])
                freeze_masks[side] = None if mask_id == -1 else mask_id
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
                            end_freeze_mask=freeze_masks[side],
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

            if any(mask is not None for mask in freeze_masks):
                step = step_with_end_freeze(
                    env, *actions, mask_ids=tuple(0 if mask is None else mask for mask in freeze_masks)
                ).result
            else:
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
                opponent_policy_ids[game_index] = policy_router.sample(
                    rng, actor_id * games_per_actor + game_index
                )


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
        lean_policy_execution: bool = False,
        telemetry: bool = False,
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
        self.lean_policy_execution = bool(lean_policy_execution)
        self.telemetry_enabled = bool(telemetry)
        self.telemetry_lock = threading.Lock()
        self.response_samples = deque(maxlen=200000)
        self.response_sample_count = 0
        self.batch_histograms = defaultdict(Counter)
        self.group_samples = deque(maxlen=200000)
        self.seen_generations = [0] * len(actor_buffers)
        self.thread = threading.Thread(target=self._run, daemon=True)
        self.inference_batches = 0
        self.inference_items = 0
        self.inference_seconds = 0.0
        self.coalesce_seconds = 0.0
        self.service_cycles = 0
        self.coalesce_cycles = 0
        self.policy_group_seconds = 0.0
        self.policy_version = 0

    def start(self):
        self.thread.start()

    def reset_telemetry(self):
        with self.telemetry_lock:
            self.response_samples.clear()
            self.response_sample_count = 0
            self.batch_histograms.clear()
            self.group_samples.clear()
            self.seen_generations = [int(shared.completed_generation[0]) for shared in self.actor_buffers]

    def telemetry_snapshot(self, start_ns=None, end_ns=None):
        with self.telemetry_lock:
            samples = list(self.response_samples)
            histograms = {str(k): dict(v) for k, v in self.batch_histograms.items()}
            count = self.response_sample_count
            groups = list(self.group_samples)
        if start_ns is not None and end_ns is not None:
            start, end = start_ns / 1e9, end_ns / 1e9
            samples = [row for row in samples if row[0] >= start and row[1] <= end]
            hist = defaultdict(Counter)
            for stamp, policy_id, size in groups:
                if start <= stamp <= end:
                    hist[policy_id][size] += 1
            histograms = {str(k): dict(v) for k, v in hist.items()}
        percentiles = np.percentile([row[2] for row in samples], [50, 95, 99]).tolist() if samples else [None] * 3
        return {'enabled': self.telemetry_enabled, 'responseSampleCount': count,
                'retainedResponseSamples': len(samples), 'responseWindowCapacity': 200000,
                'actorEnqueueToConsumeMilliseconds': dict(zip(('p50', 'p95', 'p99'), percentiles)),
                'batchSizeHistogramByPolicyId': histograms,
                'windowStartMonotonicNs': start_ns, 'windowEndMonotonicNs': end_ns,
                'completedInferenceItemsByPolicyId': {key: sum(int(size)*n for size,n in hist.items()) for key,hist in histograms.items()},
                'latencyNote': 'Completed responses observed on the next request; final in-flight responses may be absent.'}

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
            self.coalesce_cycles += 1
            self._serve(ready)

    @torch.no_grad()
    def _serve(self, ready):
        inference_started = time.perf_counter()
        with self.model_lock:
            # Leases keep retired opponents resident until their games end.
            # Missing IDs are an invariant violation; using main here would
            # import a foreign recurrent representation into another network.
            references_by_policy = {}
            for actor_index in ready:
                shared = self.actor_buffers[actor_index]
                if self.telemetry_enabled:
                    generation = int(shared.completed_generation[0])
                    with self.telemetry_lock:
                        if generation > self.seen_generations[actor_index]:
                            self.response_samples.append((float(shared.completed_request_started[0]),
                                float(shared.completed_at[0]), float(shared.response_latency_ms[0])))
                            self.response_sample_count += 1
                            self.seen_generations[actor_index] = generation
                shared.request_event.clear()
                for slot_index in range(len(shared.active)):
                    if int(shared.active[slot_index]) == 0:
                        continue
                    policy_id = int(shared.policy_ids[slot_index])
                    if policy_id not in self.resident_policies:
                        raise RuntimeError(f"Missing resident policy {policy_id}; refusing cross-policy memory fallback")
                    references_by_policy.setdefault(policy_id, []).append(
                        (actor_index, slot_index)
                    )

            for policy_id, references in references_by_policy.items():
                if not references:
                    continue
                group_started = time.perf_counter()
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
                    options = ({'allow_end_freeze': True}
                               if getattr(policy.config, 'end_freeze', False) else {})
                    # Even retained full-output actors must not inherit an
                    # exact-entropy learner's additional decoder workload.
                    options['entropy_mode_override'] = 'legacy_path'
                    output = (policy.act(observations, memory=memories) if self.lean_policy_execution
                              else policy(observations, memory=memories, **options))
                actions_cpu, log_probs_cpu, memories_cpu, masks_cpu = inference_results_on_cpu(output)
                for batch_index, (actor, slot_index) in enumerate(references):
                    shared = self.actor_buffers[actor]
                    shared.actions[slot_index].copy_(
                        actions_cpu[batch_index]
                    )
                    shared.end_freeze_masks[slot_index] = (
                        -1 if masks_cpu is None else int(masks_cpu[batch_index])
                    )
                    shared.log_probs[slot_index] = log_probs_cpu[batch_index]
                    shared.next_memories[slot_index].copy_(
                        memories_cpu[batch_index]
                    )
                    shared.policy_versions[slot_index] = (
                        self.policy_version if policy_id == 0 else -1
                    )
                self.inference_batches += 1
                self.inference_items += len(references)
                self.policy_group_seconds += time.perf_counter() - group_started
                if self.telemetry_enabled:
                    with self.telemetry_lock:
                        self.batch_histograms[policy_id][len(references)] += 1
                        self.group_samples.append((time.perf_counter(), policy_id, len(references)))

        for actor_index in ready:
            self.actor_buffers[actor_index].response_event.set()
        self.inference_seconds += time.perf_counter() - inference_started
        self.service_cycles += 1


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
        lean_policy_execution: bool = False,
        inference_telemetry: bool = False,
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
        self.policy_router = SharedPolicyRouter.create(
            num_leases=num_actors * games_per_actor, context=self.context
        )
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
                telemetry=inference_telemetry,
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
            lean_policy_execution=lean_policy_execution,
            telemetry=inference_telemetry,
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

    def collect_segments(self, count: int, timeout: float = 120.0, *, dense=False):
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
                if dense:
                    batch = self.trajectory_ring.read_dense(
                        min(count - len(segments), self.trajectory_ring.capacity), min(1.0, remaining))
                    segments.extend(batch.segments())
                else:
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

        policy_ids = list(policies.keys())
        if probabilities is None:
            probabilities = [1.0] * len(policy_ids)
        else:
            probabilities = list(probabilities)
        if main_probability > 0:
            policy_ids = [0] + policy_ids
            probabilities = [float(main_probability)] + probabilities
        with self.policy_router.lock:
            leased_ids = set(map(int, self.policy_router.leases.tolist()))
            retained = {
                policy_id: model
                for policy_id, model in self.batcher.resident_policies.items()
                if policy_id != 0 and policy_id in leased_ids
            }
            for policy_id, model in policies.items():
                if policy_id in retained and retained[policy_id] is not model:
                    raise ValueError(f"Cannot replace leased policy ID {policy_id}")
            self.batcher.set_policy_pool({**retained, **policies})
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
