"""Teacher-forced expert imitation losses for reconstructed replay sequences."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch.nn import functional as F

from .model import EntityTransformerPolicy


@dataclass
class ImitationLossOutput:
    total: torch.Tensor
    action: torch.Tensor
    battle: torch.Tensor
    value: torch.Tensor
    mean_expert_probability: torch.Tensor


def behavior_cloning_loss(
    policy: EntityTransformerPolicy,
    observations: dict[str, torch.Tensor],
    actions: torch.Tensor,
    initial_memory: torch.Tensor,
    valid_mask: torch.Tensor,
    battle_outcomes: torch.Tensor | None = None,
    returns: torch.Tensor | None = None,
    battle_cost: float = 0.25,
    value_cost: float = 0.5,
    action_cost: float = 1.0,
    action_type_weights: torch.Tensor | None = None,
    example_weights: torch.Tensor | None = None,
    end_freeze_masks: torch.Tensor | None = None,
) -> ImitationLossOutput:
    """Compute masked autoregressive BC plus optional auxiliary targets.

    ``battle_outcomes`` uses class IDs 0/1/2 and ``-1`` for unavailable
    targets. ``returns`` may use NaN for unavailable targets.
    """

    output = policy.evaluate_sequence(observations, initial_memory, actions,
        **({'end_freeze_masks':end_freeze_masks} if end_freeze_masks is not None else {}))
    valid = valid_mask.to(output.log_prob.dtype)
    action_weights = valid
    if example_weights is not None:
        example_weights = torch.as_tensor(
            example_weights,
            device=actions.device,
            dtype=output.log_prob.dtype,
        )
        if example_weights.shape != valid_mask.shape:
            raise ValueError(
                "example_weights must have the same [T,B] shape as valid_mask"
            )
        if (
            not torch.all(torch.isfinite(example_weights))
            or torch.any(example_weights < 0)
        ):
            raise ValueError("example_weights must be finite and non-negative")
        action_weights = action_weights * example_weights
    if action_type_weights is not None:
        type_weights = torch.as_tensor(
            action_type_weights,
            device=actions.device,
            dtype=output.log_prob.dtype,
        )
        if type_weights.ndim != 1:
            raise ValueError("action_type_weights must be one-dimensional")
        if type_weights.numel() <= int(actions[..., 0].max()):
            raise ValueError("action_type_weights does not cover every action type")
        if not torch.all(torch.isfinite(type_weights)) or torch.any(type_weights <= 0):
            raise ValueError("action_type_weights must be finite and positive")
        action_weights = (
            action_weights * type_weights[actions[..., 0].long()]
        )
    denominator = action_weights.sum().clamp_min(1.0)
    action_loss = -(output.log_prob * action_weights).sum() / denominator

    battle_loss = torch.zeros((), device=actions.device)
    if battle_outcomes is not None:
        battle_valid = valid_mask.bool() & (battle_outcomes >= 0)
        if battle_valid.any():
            battle_loss = F.cross_entropy(
                output.battle_logits[battle_valid],
                battle_outcomes[battle_valid].long(),
            )

    value_loss = torch.zeros((), device=actions.device)
    if returns is not None:
        value_valid = valid_mask.bool() & torch.isfinite(returns)
        if value_valid.any():
            value_loss = F.mse_loss(
                output.board_strength[value_valid], returns[value_valid]
            )

    mean_expert_probability = (
        output.log_prob.exp() * action_weights
    ).sum() / denominator
    total = (
        action_cost * action_loss
        + battle_cost * battle_loss
        + value_cost * value_loss
    )
    return ImitationLossOutput(
        total=total,
        action=action_loss,
        battle=battle_loss,
        value=value_loss,
        mean_expert_probability=mean_expert_probability,
    )
