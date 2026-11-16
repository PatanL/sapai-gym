"""Interactive outer validation with separate, fixed policies for both seats."""
from __future__ import annotations

import torch

from .interactive_evaluation import InteractiveSelfPlayEvaluator
from .decision_manifest import restore_decision


POLICY_OWNERSHIP_VERSION = 'seat-owned-recurrence-v1-20260905'


def restore_historical_decision(record, main_hash, opponent_hash):
    """Fail closed on foreign hidden states, even when model shapes match."""
    side = record.get('side')
    if side not in (0, 1):
        raise ValueError('Invalid acting side')
    expected = [opponent_hash, opponent_hash]
    expected[side] = main_hash
    if (record.get('policyOwnershipVersion') != POLICY_OWNERSHIP_VERSION
            or record.get('policySha256BySeat') != expected
            or record.get('checkpointSha256') != main_hash
            or record.get('opponentCheckpointSha256') != opponent_hash):
        raise ValueError('Recurrent state policy ownership mismatch')
    return restore_decision(record)


class HistoricalOpponentEvaluator(InteractiveSelfPlayEvaluator):
    """Keep each network's hidden states on its own side of an actual match.

    The opponent is used only in outer validation, never in public planning.
    Both seats can be tested by constructing this evaluator with acting_side
    zero or one. Stored memories must have been generated with these policies.
    """

    def __init__(self, policy, opponent_policy, *, acting_side, **kwargs):
        if acting_side not in (0, 1):
            raise ValueError("acting_side must be zero or one")
        super().__init__(policy, **kwargs)
        self.acting_side = acting_side
        self.opponent_policy = opponent_policy
        self.opponent_evaluator = InteractiveSelfPlayEvaluator(opponent_policy, **kwargs)

    def _choose_many(self, requests):
        result = [None] * len(requests)
        for is_main in (True, False):
            indices = [index for index, request in enumerate(requests)
                       if (request[1] == self.acting_side) == is_main]
            if not indices:
                continue
            subset = [requests[index] for index in indices]
            choices = (super()._choose_many(subset) if is_main
                       else self.opponent_evaluator._choose_many(subset))
            for index, choice in zip(indices, choices):
                result[index] = choice
        return result

    def evaluate(self, environment, memories, acting_side, actions, seeds):
        if acting_side != self.acting_side:
            raise ValueError("Acting side differs from the historical policy assignment")
        if len(memories) != 2:
            raise ValueError("Both players' recurrent states are required")
        for side, memory in enumerate(memories):
            policy = self.policy if side == acting_side else self.opponent_policy
            expected = policy.config.memory_dim
            tensor = torch.as_tensor(memory)
            if tensor.shape != (expected,) or not torch.isfinite(tensor).all():
                raise ValueError(f"Invalid recurrent state for seat {side}")
        return super().evaluate(environment, memories, acting_side, actions, seeds)
