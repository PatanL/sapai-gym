"""Owned, ragged-dense CPU trajectory storage; no Transition objects per step.

All valid steps share flat field tensors. Length grouping/padding happens only
when a learner minibatch is assembled. Segment views alias this OWNED batch,
never reusable ring storage, and retain their tensors' storage references.
"""
from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class DenseTurnSegment:
    observations: dict[str, torch.Tensor]
    actions: torch.Tensor
    behavior_log_probs: torch.Tensor
    behavior_versions: torch.Tensor
    rewards: torch.Tensor
    discounts: torch.Tensor
    freeze_masks: torch.Tensor  # -1 is legacy None; zero is a real joint mask.
    initial_memory: torch.Tensor
    bootstrap_observation: dict[str, torch.Tensor]
    terminal: bool
    battle_outcome: int | None
    slot_id: int

    @property
    def length(self):
        return self.actions.shape[0]


def segment_length(segment):
    return segment.length if isinstance(segment, DenseTurnSegment) else len(segment.transitions)


@dataclass(frozen=True)
class OwnedTrajectoryBatch:
    observations: dict[str, torch.Tensor]
    actions: torch.Tensor
    behavior_log_probs: torch.Tensor
    behavior_versions: torch.Tensor
    rewards: torch.Tensor
    discounts: torch.Tensor
    freeze_masks: torch.Tensor
    lengths: tuple[int, ...]
    initial_memories: torch.Tensor
    bootstrap_observations: dict[str, torch.Tensor]
    terminals: torch.Tensor
    battle_outcomes: torch.Tensor
    slot_ids: tuple[int, ...]

    @classmethod
    def copy_from_ring(cls, ring, slot_ids):
        """Caller holds every slot lease until this synchronous copy returns."""
        slots = tuple(slot_ids)
        if not slots:
            raise ValueError('Cannot copy an empty ready-slot batch')
        lengths = tuple(int(ring.lengths[i]) for i in slots)
        if any(n < 1 or n > ring.max_steps for n in lengths):
            raise ValueError('Invalid ring sequence length')
        total, batch = sum(lengths), len(slots)
        observations = {k: torch.empty((total, *v.shape[2:]), dtype=v.dtype) for k, v in ring.observations.items()}
        bootstrap = {k: torch.empty((batch, *v.shape[1:]), dtype=v.dtype) for k, v in ring.bootstrap_observations.items()}
        step_fields = {}
        for target, source in (
            ('actions', 'actions'), ('behavior_log_probs', 'behavior_log_probs'),
            ('behavior_versions', 'behavior_versions'), ('rewards', 'rewards'),
            ('discounts', 'discounts'), ('freeze_masks', 'end_freeze_masks'),
        ):
            original = getattr(ring, source)
            step_fields[target] = torch.empty((total, *original.shape[2:]), dtype=original.dtype)
        initial = torch.empty((batch, ring.initial_memories.shape[1]), dtype=ring.initial_memories.dtype)
        terminals = torch.empty(batch, dtype=ring.terminals.dtype)
        outcomes = torch.empty(batch, dtype=ring.battle_outcomes.dtype)
        offset = 0
        for b, (slot, length) in enumerate(zip(slots, lengths)):
            section = slice(offset, offset + length)
            for key, tensor in observations.items():
                tensor[section].copy_(ring.observations[key][slot, :length])
                bootstrap[key][b].copy_(ring.bootstrap_observations[key][slot])
            for target, tensor in step_fields.items():
                source = 'end_freeze_masks' if target == 'freeze_masks' else target
                tensor[section].copy_(getattr(ring, source)[slot, :length])
            initial[b].copy_(ring.initial_memories[slot])
            terminals[b].copy_(ring.terminals[slot])
            outcomes[b].copy_(ring.battle_outcomes[slot])
            offset += length
        return cls(observations=observations, **step_fields, lengths=lengths,
                   initial_memories=initial, bootstrap_observations=bootstrap,
                   terminals=terminals, battle_outcomes=outcomes, slot_ids=slots)

    def segments(self):
        result, offset = [], 0
        for b, length in enumerate(self.lengths):
            section = slice(offset, offset + length)
            result.append(DenseTurnSegment(
                observations={k: v[section] for k, v in self.observations.items()},
                **{k: getattr(self, k)[section] for k in ('actions', 'behavior_log_probs', 'behavior_versions',
                                                       'rewards', 'discounts', 'freeze_masks')},
                initial_memory=self.initial_memories[b],
                bootstrap_observation={k: v[b] for k, v in self.bootstrap_observations.items()},
                terminal=bool(self.terminals[b]),
                battle_outcome=None if int(self.battle_outcomes[b]) < 0 else int(self.battle_outcomes[b]),
                slot_id=self.slot_ids[b],
            ))
            offset += length
        return result
