"""Batched explicit self-play collection and V-trace learner updates."""

from __future__ import annotations

from dataclasses import dataclass
from contextlib import nullcontext

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from .action_schema import ActionType, AutoregressiveAction
from .model import EntityTransformerPolicy
from .selfplay_env import ParallelSelfPlayEnv
from .end_freeze import step_with_end_freeze
from .learner_batch import pack_learner_batch
from .dense_trajectory import segment_length
from .batched_learner import batched_sequence_losses, DeviceLearnerMetrics
from .vtrace import ImpalaLossOutput, impala_actor_critic_loss


@dataclass
class Transition:
    observation: dict[str, np.ndarray]
    action: np.ndarray
    behavior_log_prob: float
    reward: float = 0.0
    discount: float = 0.99
    behavior_version: int = -1
    # None is legacy primitive likelihood, NOT a zero-toggle joint action.
    end_freeze_mask: int | None = None


@dataclass
class TurnSegment:
    transitions: list[Transition]
    initial_memory: torch.Tensor
    bootstrap_observation: dict[str, np.ndarray]
    terminal: bool
    battle_outcome: int | None = None


@dataclass
class LearnerMetrics:
    total_loss: float
    policy_loss: float
    value_loss: float
    battle_aux_loss: float
    opponent_aux_loss: float
    entropy: float
    transitions: int
    grad_norm: float


def action_shaping_reward(action, move_action_cost=0.0):
    """Return a small optional cost for zero-gold positioning actions."""

    if move_action_cost < 0:
        raise ValueError("move_action_cost must be non-negative")
    action_type = int(action[0])
    return (
        -float(move_action_cost)
        if action_type == int(ActionType.MOVE)
        else 0.0
    )


def stack_observations(observations, device):
    return {
        key: torch.as_tensor(
            np.stack([observation[key] for observation in observations]),
            device=device,
        )
        for key in observations[0]
    }


class BatchedSelfPlayCollector:
    """Batch all active sides from several games through one GPU forward pass."""

    def __init__(
        self,
        policy: EntityTransformerPolicy,
        num_envs: int,
        device: torch.device | str,
        gamma: float = 0.99,
        seed: int = 0,
        use_bfloat16: bool = True,
        opponent_policy: EntityTransformerPolicy | None = None,
        move_action_cost: float = 0.0,
        learn_from_resident_sides: bool = False,
        discount_mode: str = "per_action",
        lean_policy_execution: bool = False,
    ):
        self.policy = policy
        self.device = torch.device(device)
        self.gamma = gamma
        self.use_bfloat16 = use_bfloat16 and self.device.type == "cuda"
        self.opponent_policy = opponent_policy
        self.learn_from_resident_sides = bool(learn_from_resident_sides)
        if discount_mode not in {"per_action", "per_turn"}:
            raise ValueError("discount_mode must be 'per_action' or 'per_turn'")
        self.discount_mode = discount_mode
        self.lean_policy_execution = bool(lean_policy_execution)
        if move_action_cost < 0:
            raise ValueError("move_action_cost must be non-negative")
        self.move_action_cost = float(move_action_cost)
        self.rng = np.random.default_rng(seed)
        self.envs = [ParallelSelfPlayEnv() for _ in range(num_envs)]
        self.game_opponents = [opponent_policy for _ in self.envs]
        self.game_started = [False for _ in self.envs]
        self.observations = [
            env.reset(seed=int(self.rng.integers(0, 2**31 - 1)))
            for env in self.envs
        ]
        self.memories = torch.zeros(
            num_envs,
            2,
            policy.config.memory_dim,
            device=self.device,
        )
        self.buffers = [[[], []] for _ in self.envs]
        self.segment_initial_memory = [
            [self.memories[index, side].detach().cpu().clone() for side in range(2)]
            for index in range(num_envs)
        ]

    def set_opponent_policy(
        self, opponent_policy: EntityTransformerPolicy | None
    ) -> None:
        """Choose the opponent for new matches, retaining started assignments.

        ``None`` means symmetric latest-v-latest. Unstarted games can adopt
        the assignment immediately; an existing resident's recurrent state
        must never be passed to a different frozen network mid-match.
        """

        self.opponent_policy = opponent_policy
        if opponent_policy is not None:
            opponent_policy.eval()
            opponent_policy.requires_grad_(False)
        for index, started in enumerate(self.game_started):
            if not started:
                self.game_opponents[index] = opponent_policy

    @torch.no_grad()
    def collect(self, num_segments: int) -> list[TurnSegment]:
        self.policy.eval()
        completed: list[TurnSegment] = []
        while len(completed) < num_segments:
            references = []
            observations = []
            memories = []
            for env_index, env in enumerate(self.envs):
                active = env.active_players()
                for side in range(2):
                    if not active[side]:
                        continue
                    references.append((env_index, side))
                    observations.append(self.observations[env_index][side])
                    memories.append(self.memories[env_index, side])

            action_lookup = {}
            freeze_lookup = {}
            # Batch by policy identity.  Latest-v-latest uses one large batch;
            # a frozen historical opponent uses a second batch without
            # entangling its recurrent state with the learner.
            policy_groups = {}
            for batch_index, (env_index, side) in enumerate(references):
                behavior_policy = (
                    self.game_opponents[env_index]
                    if side == 1 and self.game_opponents[env_index] is not None
                    else self.policy
                )
                policy_groups.setdefault(behavior_policy, []).append(batch_index)

            for behavior_policy, indices in policy_groups.items():
                behavior_policy.eval()
                batch = stack_observations(
                    [observations[index] for index in indices], self.device
                )
                memory_batch = torch.stack([memories[index] for index in indices])
                autocast = (
                    torch.autocast("cuda", dtype=torch.bfloat16)
                    if self.use_bfloat16
                    else nullcontext()
                )
                with autocast:
                    options = ({'allow_end_freeze': True}
                               if getattr(behavior_policy.config, 'end_freeze', False) else {})
                    options['entropy_mode_override'] = 'legacy_path'
                    output = (behavior_policy.act(batch, memory=memory_batch) if self.lean_policy_execution
                              else behavior_policy(batch, memory=memory_batch, **options))
                for group_index, batch_index in enumerate(indices):
                    env_index, side = references[batch_index]
                    self.game_started[env_index] = True
                    action = output.actions[group_index].detach().cpu().numpy()
                    masks = getattr(output, 'end_freeze_mask', None)
                    freeze_mask = None if masks is None else int(masks[group_index].cpu())
                    freeze_lookup[(env_index, side)] = freeze_mask
                    action_lookup[(env_index, side)] = AutoregressiveAction(
                        *map(int, action)
                    )
                    learns_from_side = (
                        side == 0
                        or self.game_opponents[env_index] is None
                        or self.learn_from_resident_sides
                    )
                    if learns_from_side:
                        self.buffers[env_index][side].append(
                            Transition(
                                observation=self.observations[env_index][side],
                                action=action,
                                behavior_log_prob=float(
                                    output.log_prob[group_index].cpu()
                                ),
                                end_freeze_mask=freeze_mask,
                                reward=action_shaping_reward(
                                    action, self.move_action_cost
                                ),
                                discount=(
                                    self.gamma
                                    if self.discount_mode == "per_action"
                                    else 1.0
                                ),
                            )
                        )
                    self.memories[env_index, side] = output.memory[group_index]

            for env_index, env in enumerate(self.envs):
                active = env.active_players()
                actions = [
                    action_lookup[(env_index, side)] if active[side] else None
                    for side in range(2)
                ]
                masks = [freeze_lookup.get((env_index, side)) for side in range(2)]
                if any(mask is not None for mask in masks):
                    step = step_with_end_freeze(
                        env, *actions, mask_ids=tuple(0 if mask is None else mask for mask in masks)
                    ).result
                else:
                    step = env.step(actions[0], actions[1])
                self.observations[env_index] = step.observations
                if not step.battle_resolved:
                    continue

                for side in range(2):
                    buffer = self.buffers[env_index][side]
                    learns_from_side = (
                        side == 0
                        or self.game_opponents[env_index] is None
                        or self.learn_from_resident_sides
                    )
                    if not learns_from_side:
                        self.buffers[env_index][side] = []
                        self.segment_initial_memory[env_index][side] = (
                            self.memories[env_index, side].detach().cpu().clone()
                        )
                        continue
                    if not buffer:
                        raise RuntimeError("Battle resolved without a player segment")
                    buffer[-1].reward += step.rewards[side]
                    if step.terminated:
                        buffer[-1].discount = 0.0
                    elif self.discount_mode == "per_turn":
                        buffer[-1].discount = self.gamma
                    completed.append(
                        TurnSegment(
                            transitions=buffer,
                            initial_memory=self.segment_initial_memory[env_index][side],
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
                    )
                    self.buffers[env_index][side] = []
                    self.segment_initial_memory[env_index][side] = (
                        self.memories[env_index, side].detach().cpu().clone()
                    )

                if step.terminated:
                    self.game_opponents[env_index] = self.opponent_policy
                    self.game_started[env_index] = False
                    self.observations[env_index] = env.reset(
                        seed=int(self.rng.integers(0, 2**31 - 1))
                    )
                    self.memories[env_index].zero_()
                    for side in range(2):
                        self.segment_initial_memory[env_index][side] = (
                            self.memories[env_index, side].detach().cpu().clone()
                        )
        return completed[:num_segments]


def _learner_update_batch(
    policy: EntityTransformerPolicy,
    optimizer: torch.optim.Optimizer,
    segments: list[TurnSegment],
    device: torch.device | str,
    max_grad_norm: float = 40.0,
    use_bfloat16: bool = True,
    zero_grad: bool = True,
    optimizer_step: bool = True,
    loss_scale: float = 1.0,
    battle_aux_cost: float = 0.05,
    opponent_aux_cost: float = 0.1,
    lean_bootstrap: bool = False,
    batched_learner: bool = False,
    defer_metrics: bool = False,
) -> LearnerMetrics:
    if not segments:
        raise ValueError("At least one segment is required")
    device = torch.device(device)
    policy.train()
    use_bfloat16 = use_bfloat16 and device.type == "cuda"
    joint_actions = bool(getattr(policy.config, 'end_freeze', False))
    packed = pack_learner_batch(segments, device, joint_actions=joint_actions, include_metadata=batched_learner)
    lengths = [segment_length(s) for s in segments]
    actions = packed.actions
    behavior_log_probs = packed.behavior_log_probs
    rewards, discounts = packed.rewards, packed.discounts
    freeze_masks = packed.freeze_masks
    observation_sequence = packed.observations
    memory = packed.initial_memories if packed.initial_memories is not None else torch.stack(
        [segment.initial_memory for segment in segments]
    ).to(device)
    def autocast_context():
        return (
            torch.autocast("cuda", dtype=torch.bfloat16)
            if use_bfloat16
            else nullcontext()
        )

    with autocast_context():
        sequence_output = policy.evaluate_sequence(
            observation_sequence,
            initial_memory=memory,
            actions=actions,
            **({'end_freeze_masks': freeze_masks} if joint_actions else {}),
            forecast_first_only=batched_learner,
        )
    if batched_learner:
        final_memories = sequence_output.memory_states[packed.lengths - 1, torch.arange(len(segments), device=device)]
    else:
        final_memories = torch.stack(
            [
                sequence_output.memory_states[lengths[batch_index] - 1, batch_index]
                for batch_index in range(len(segments))
            ]
        )

    bootstrap_observation = packed.bootstrap_observations if packed.bootstrap_observations is not None else stack_observations(
        [segment.bootstrap_observation for segment in segments], device
    )
    with autocast_context():
        if lean_bootstrap:
            # Targets are detached by V-trace. Avoid constructing a discarded
            # bootstrap graph and do not decode any actions/entropy/forecasts.
            with torch.no_grad():
                bootstrap_value = policy.value_only(bootstrap_observation, memory=final_memories)
        else:
            bootstrap_value = policy(
                bootstrap_observation, memory=final_memories, deterministic=True,
                **({'allow_end_freeze': True} if joint_actions else {}),
            ).value
    terminal = packed.terminals if packed.terminals is not None else torch.tensor(
        [segment.terminal for segment in segments],
        dtype=torch.bool,
        device=device,
    )
    bootstrap_value = torch.where(
        terminal, torch.zeros_like(bootstrap_value), bootstrap_value
    )

    if batched_learner:
        losses = batched_sequence_losses(policy, sequence_output, packed, bootstrap_value,
                                         battle_aux_cost, opponent_aux_cost)
    else:
        losses = _legacy_sequence_losses(policy, sequence_output, packed, bootstrap_value,
                                         segments, lengths, device, autocast_context,
                                         battle_aux_cost, opponent_aux_cost)
    total_loss, policy_loss, value_loss, entropy_loss, battle_aux_loss, opponent_aux_loss = losses
    transition_count = sum(lengths)
    if zero_grad:
        optimizer.zero_grad(set_to_none=True)
    (total_loss * loss_scale).backward()
    if optimizer_step:
        grad_norm = nn.utils.clip_grad_norm_(policy.parameters(), max_grad_norm)
        optimizer.step()
    else:
        grad_norm = torch.tensor(0.0)
    if batched_learner:
        metrics = DeviceLearnerMetrics(torch.stack([total_loss, policy_loss, value_loss,
                                                   battle_aux_loss, opponent_aux_loss, -entropy_loss]).detach(),
                                      transition_count)
        return metrics if defer_metrics else _metrics_on_cpu(metrics, grad_norm)
    return LearnerMetrics(
        total_loss=float(total_loss.detach().cpu()),
        policy_loss=float(policy_loss.detach().cpu()),
        value_loss=float(value_loss.detach().cpu()),
        battle_aux_loss=float(battle_aux_loss.detach().cpu()),
        opponent_aux_loss=float(opponent_aux_loss.detach().cpu()),
        entropy=float((-entropy_loss).detach().cpu()),
        transitions=transition_count,
        grad_norm=float(torch.as_tensor(grad_norm).detach().cpu()),
    )


def _metrics_on_cpu(metrics, grad_norm):
    values = torch.cat((metrics.values, torch.as_tensor(grad_norm, device=metrics.values.device).reshape(1))).cpu().tolist()
    return LearnerMetrics(*values[:6], transitions=metrics.transitions, grad_norm=values[6])



def _legacy_sequence_losses(policy, sequence_output, packed, bootstrap_value, segments,
                            lengths, device, autocast_context, battle_aux_cost, opponent_aux_cost):
    behavior_log_probs = packed.behavior_log_probs
    rewards, discounts = packed.rewards, packed.discounts
    target_log_probs = sequence_output.log_prob
    entropies = sequence_output.entropy
    values = sequence_output.value
    # V-trace's bootstrap sits immediately after each sequence.  Compute the
    # inexpensive return recursion per variable-length segment while retaining
    # fully batched transformer evaluation above.
    segment_losses = []
    transition_count = 0
    with autocast_context():
        for batch_index, segment in enumerate(segments):
            length = lengths[batch_index]
            transition_count += length
            segment_losses.append(
                impala_actor_critic_loss(
                    behavior_log_probs=behavior_log_probs[:length, batch_index : batch_index + 1],
                    target_log_probs=target_log_probs[:length, batch_index : batch_index + 1],
                    entropy=entropies[:length, batch_index : batch_index + 1],
                    discounts=discounts[:length, batch_index : batch_index + 1],
                    rewards=rewards[:length, batch_index : batch_index + 1],
                    values=values[:length, batch_index : batch_index + 1],
                    bootstrap_value=bootstrap_value[batch_index : batch_index + 1],
                )
            )
    weights = [length / transition_count for length in lengths]
    total_loss = sum(
        weight * segment_loss.total
        for weight, segment_loss in zip(weights, segment_losses)
    )
    policy_loss = sum(
        weight * segment_loss.policy
        for weight, segment_loss in zip(weights, segment_losses)
    )
    value_loss = sum(
        weight * segment_loss.value
        for weight, segment_loss in zip(weights, segment_losses)
    )
    entropy_loss = sum(
        weight * segment_loss.entropy
        for weight, segment_loss in zip(weights, segment_losses)
    )
    battle_logits = []
    battle_targets = []
    if battle_aux_cost > 0:
        for batch_index, segment in enumerate(segments):
            if segment.battle_outcome is None:
                continue
            length = lengths[batch_index]
            battle_logits.append(
                sequence_output.battle_logits[:length, batch_index]
            )
            battle_targets.append(
                torch.full(
                    (length,),
                    int(segment.battle_outcome),
                    dtype=torch.long,
                    device=device,
                )
            )
    if battle_logits:
        battle_aux_loss = F.cross_entropy(
            torch.cat(battle_logits), torch.cat(battle_targets)
        )
        total_loss = total_loss + battle_aux_cost * battle_aux_loss
    else:
        battle_aux_loss = torch.zeros((), device=device)
    prediction_indices = []
    target_items = []
    target_statuses = []
    target_numeric = []
    target_masks = []
    for batch_index, segment in enumerate(segments):
        bootstrap = segment.bootstrap_observation
        board_mask = bootstrap.get("opponent_history_board_mask")
        if board_mask is None or not bool(board_mask[-1]):
            continue
        prediction_indices.append(batch_index)
        target_items.append(bootstrap["opponent_history_item_ids"][-1])
        target_statuses.append(
            bootstrap["opponent_history_status_ids"][-1]
        )
        target_numeric.append(bootstrap["opponent_history_numeric"][-1])
        target_masks.append(bootstrap["opponent_history_mask"][-1])
    if prediction_indices and opponent_aux_cost > 0:
        indices = torch.tensor(
            prediction_indices, dtype=torch.long, device=device
        )
        items = torch.as_tensor(np.stack(target_items), device=device).long()
        statuses = torch.as_tensor(
            np.stack(target_statuses), device=device
        ).long()
        numeric_targets = torch.as_tensor(
            np.stack(target_numeric), device=device
        ).float()
        occupied = torch.as_tensor(
            np.stack(target_masks), device=device
        ).bool()
        pet_loss = F.cross_entropy(
            sequence_output.opponent_pet_logits[0, indices].float().reshape(
                -1, policy.config.num_items
            ),
            items.reshape(-1),
        )
        if occupied.any():
            status_loss = F.cross_entropy(
                sequence_output.opponent_status_logits[0, indices]
                .float()[occupied],
                statuses[occupied],
            )
            levels = torch.round(numeric_targets[..., 2] * 3).long()
            level_loss = F.cross_entropy(
                sequence_output.opponent_level_logits[0, indices]
                .float()[occupied],
                levels[occupied],
            )
            stats_loss = F.mse_loss(
                sequence_output.opponent_stats[0, indices]
                .float()[occupied],
                numeric_targets[..., :2][occupied],
            )
            temporary_loss = F.binary_cross_entropy_with_logits(
                sequence_output.opponent_temporary_logits[0, indices]
                .float()[occupied],
                numeric_targets[..., 5:7][occupied],
            )
        else:
            status_loss = level_loss = stats_loss = temporary_loss = (
                torch.zeros((), device=device)
            )
        opponent_aux_loss = (
            pet_loss
            + 0.25 * status_loss
            + 0.25 * level_loss
            + 0.5 * stats_loss
            + 0.1 * temporary_loss
        )
        total_loss = total_loss + opponent_aux_cost * opponent_aux_loss
    else:
        opponent_aux_loss = torch.zeros((), device=device)
    return total_loss, policy_loss, value_loss, entropy_loss, battle_aux_loss, opponent_aux_loss

def learner_update(
    policy: EntityTransformerPolicy,
    optimizer: torch.optim.Optimizer,
    segments: list[TurnSegment],
    device: torch.device | str,
    max_grad_norm: float = 40.0,
    use_bfloat16: bool = True,
    bucket_by_length: bool = True,
    battle_aux_cost: float = 0.05,
    opponent_aux_cost: float = 0.1,
    lean_bootstrap: bool = False,
    batched_learner: bool = False,
) -> LearnerMetrics:
    """Update from turn segments, bucketing lengths to avoid padded attention."""

    if not bucket_by_length or len(segments) < 16:
        return _learner_update_batch(
            policy,
            optimizer,
            segments,
            device,
            max_grad_norm=max_grad_norm,
            use_bfloat16=use_bfloat16,
            battle_aux_cost=battle_aux_cost,
            opponent_aux_cost=opponent_aux_cost,
            lean_bootstrap=lean_bootstrap,
            batched_learner=batched_learner,
        )

    buckets = {}
    for segment in segments:
        length = segment_length(segment)
        upper_bound = 1 << max(0, (length - 1).bit_length())
        buckets.setdefault(upper_bound, []).append(segment)

    bucket_groups = []
    for bucket in buckets.values():
        # Keep very large same-length populations as practical learner
        # minibatches while preserving recurrent ordering within each segment.
        for start in range(0, len(bucket), 128):
            bucket_groups.append(bucket[start : start + 128])

    total_transitions = sum(
        segment_length(segment)
        for group in bucket_groups
        for segment in group
    )
    optimizer.zero_grad(set_to_none=True)
    metrics = []
    for group in bucket_groups:
        group_transitions = sum(segment_length(segment) for segment in group)
        metrics.append(
            _learner_update_batch(
                policy,
                optimizer,
                group,
                device,
                max_grad_norm=max_grad_norm,
                use_bfloat16=use_bfloat16,
                zero_grad=False,
                optimizer_step=False,
                loss_scale=group_transitions / total_transitions,
                battle_aux_cost=battle_aux_cost,
                opponent_aux_cost=opponent_aux_cost,
                lean_bootstrap=lean_bootstrap,
                batched_learner=batched_learner,
                defer_metrics=batched_learner,
            )
        )
    grad_norm = nn.utils.clip_grad_norm_(policy.parameters(), max_grad_norm)
    optimizer.step()

    if batched_learner:
        aggregated = sum(metric.values * (metric.transitions / total_transitions) for metric in metrics)
        return _metrics_on_cpu(DeviceLearnerMetrics(aggregated, total_transitions), grad_norm)

    def weighted(field):
        return sum(
            getattr(metric, field) * metric.transitions for metric in metrics
        ) / total_transitions

    return LearnerMetrics(
        total_loss=weighted("total_loss"),
        policy_loss=weighted("policy_loss"),
        value_loss=weighted("value_loss"),
        battle_aux_loss=weighted("battle_aux_loss"),
        opponent_aux_loss=weighted("opponent_aux_loss"),
        entropy=weighted("entropy"),
        transitions=total_transitions,
        grad_norm=float(torch.as_tensor(grad_norm).detach().cpu()),
    )
