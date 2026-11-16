"""CPU batch assembly followed by one device transfer per dense field."""
from dataclasses import dataclass

import numpy as np
import torch
from .dense_trajectory import DenseTurnSegment, segment_length


@dataclass
class PackedLearnerBatch:
    observations: dict[str, torch.Tensor]
    actions: torch.Tensor
    behavior_log_probs: torch.Tensor
    rewards: torch.Tensor
    discounts: torch.Tensor
    freeze_masks: torch.Tensor | None
    lengths: torch.Tensor | None = None
    initial_memories: torch.Tensor | None = None
    bootstrap_observations: dict[str, torch.Tensor] | None = None
    terminals: torch.Tensor | None = None
    battle_outcomes: torch.Tensor | None = None


def pack_learner_batch(segments, device, *, joint_actions, include_metadata=False):
    if segments and isinstance(segments[0], DenseTurnSegment):
        if not all(isinstance(s, DenseTurnSegment) for s in segments):
            raise ValueError('Do not mix dense and legacy segments in a minibatch')
        return pack_dense_segments(segments, device, joint_actions=joint_actions)
    if not segments or any(not segment.transitions for segment in segments):
        raise ValueError('Learner batches require nonempty segments')
    for segment in segments:
        for transition in segment.transitions:
            if (transition.end_freeze_mask is not None) != joint_actions:
                raise ValueError('Trajectory action schema does not match learner; primitive and joint likelihoods cannot mix')
    time_steps = max(len(segment.transitions) for segment in segments)
    batch_size = len(segments)
    actions = torch.zeros(time_steps, batch_size, 3, dtype=torch.long, device='cpu')
    behavior = torch.zeros(time_steps, batch_size, device='cpu')
    rewards = torch.zeros_like(behavior)
    discounts = torch.zeros_like(behavior)
    freeze = torch.zeros(time_steps, batch_size, dtype=torch.long, device='cpu') if joint_actions else None

    for index, segment in enumerate(segments):
        rows = segment.transitions
        length = len(rows)
        padded = rows + [rows[-1]] * (time_steps - length)
        actions[:, index] = torch.as_tensor(np.stack([row.action for row in padded]), device='cpu')
        behavior[:length, index] = torch.as_tensor([row.behavior_log_prob for row in rows], dtype=behavior.dtype, device='cpu')
        rewards[:length, index] = torch.as_tensor([row.reward for row in rows], dtype=rewards.dtype, device='cpu')
        discounts[:length, index] = torch.as_tensor([row.discount for row in rows], dtype=discounts.dtype, device='cpu')
        if freeze is not None:
            freeze[:, index] = torch.as_tensor([row.end_freeze_mask for row in padded], dtype=torch.long, device='cpu')

    # Retain the legacy stack order and NumPy/PyTorch dtype-promotion behavior.
    # Padding copies the last legal observation/action, not an invalid zero state.
    padded_rows = [[segment.transitions[min(time_index, len(segment.transitions) - 1)]
                    for segment in segments] for time_index in range(time_steps)]
    observations = {
        key: torch.stack([
            torch.as_tensor(np.stack([row.observation[key] for row in rows]), device='cpu')
            for rows in padded_rows
        ]).to(device)
        for key in segments[0].transitions[0].observation
    }
    packed = PackedLearnerBatch(
        observations, actions.to(device), behavior.to(device), rewards.to(device),
        discounts.to(device), None if freeze is None else freeze.to(device),
    )
    return attach_metadata(packed, segments, device) if include_metadata else packed


def attach_metadata(packed, segments, device):
    packed.lengths = torch.tensor([segment_length(s) for s in segments], dtype=torch.long).to(device)
    packed.initial_memories = torch.stack([s.initial_memory for s in segments]).to(device)
    packed.bootstrap_observations = {
        k: torch.stack([torch.as_tensor(s.bootstrap_observation[k]) for s in segments]).to(device)
        for k in segments[0].bootstrap_observation
    }
    packed.terminals = torch.tensor([s.terminal for s in segments], dtype=torch.bool).to(device)
    packed.battle_outcomes = torch.tensor([-1 if s.battle_outcome is None else s.battle_outcome for s in segments],
                                          dtype=torch.long).to(device)
    return packed


def pack_dense_segments(segments, device, *, joint_actions):
    lengths = [s.length for s in segments]
    if not lengths or min(lengths) < 1:
        raise ValueError('Nonempty dense sequences are required')
    for s in segments:
        if not ((s.freeze_masks >= 0) if joint_actions else (s.freeze_masks == -1)).all():
            raise ValueError('Trajectory action schema does not match learner')
        if joint_actions and (s.freeze_masks >= 1024).any():
            raise ValueError('Invalid joint freeze mask')
    time_steps, batch_size = max(lengths), len(segments)
    observations = {k: torch.empty((time_steps, batch_size, *v.shape[1:]), dtype=v.dtype)
                    for k, v in segments[0].observations.items()}
    actions = torch.empty(time_steps, batch_size, 3, dtype=torch.long)
    behavior = torch.zeros(time_steps, batch_size, dtype=torch.float32)
    rewards, discounts = torch.zeros_like(behavior), torch.zeros_like(behavior)
    freeze = torch.empty(time_steps, batch_size, dtype=torch.long) if joint_actions else None
    for b, (s, length) in enumerate(zip(segments, lengths)):
        for key, tensor in observations.items():
            tensor[:length, b].copy_(s.observations[key])
            tensor[length:, b].copy_(s.observations[key][-1])
        actions[:length, b].copy_(s.actions)
        actions[length:, b].copy_(s.actions[-1])
        for output, source in ((behavior, s.behavior_log_probs), (rewards, s.rewards), (discounts, s.discounts)):
            output[:length, b].copy_(source)
        if freeze is not None:
            freeze[:length, b].copy_(s.freeze_masks)
            freeze[length:, b].copy_(s.freeze_masks[-1])
    packed = PackedLearnerBatch({k: v.to(device) for k, v in observations.items()}, actions.to(device),
                                behavior.to(device), rewards.to(device), discounts.to(device),
                                None if freeze is None else freeze.to(device))
    return attach_metadata(packed, segments, device)
