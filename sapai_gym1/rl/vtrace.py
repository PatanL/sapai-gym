"""IMPALA V-trace targets and actor-critic loss."""

from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass
class VTraceReturns:
    value_targets: torch.Tensor
    policy_advantages: torch.Tensor


@dataclass
class ImpalaLossOutput:
    total: torch.Tensor
    policy: torch.Tensor
    value: torch.Tensor
    entropy: torch.Tensor
    vtrace: VTraceReturns


def vtrace_from_log_probs(
    behavior_log_probs: torch.Tensor,
    target_log_probs: torch.Tensor,
    discounts: torch.Tensor,
    rewards: torch.Tensor,
    values: torch.Tensor,
    bootstrap_value: torch.Tensor,
    clip_rho_threshold: float = 1.0,
    clip_pg_rho_threshold: float = 1.0,
) -> VTraceReturns:
    """Compute V-trace targets for tensors shaped ``[time, batch]``."""

    if values.shape != rewards.shape:
        raise ValueError("values and rewards must have identical [T, B] shapes")
    log_rhos = target_log_probs - behavior_log_probs
    rhos = torch.exp(log_rhos)
    clipped_rhos = torch.clamp(rhos, max=clip_rho_threshold)
    clipped_pg_rhos = torch.clamp(rhos, max=clip_pg_rho_threshold)
    cs = torch.clamp(rhos, max=1.0)

    values_t_plus_1 = torch.cat(
        [values[1:], bootstrap_value.unsqueeze(0)], dim=0
    )
    deltas = clipped_rhos * (
        rewards + discounts * values_t_plus_1 - values
    )

    accumulator = torch.zeros_like(bootstrap_value)
    corrections = []
    for time_index in range(values.shape[0] - 1, -1, -1):
        accumulator = (
            deltas[time_index]
            + discounts[time_index] * cs[time_index] * accumulator
        )
        corrections.append(accumulator)
    corrections.reverse()
    value_targets = values + torch.stack(corrections, dim=0)

    targets_t_plus_1 = torch.cat(
        [value_targets[1:], bootstrap_value.unsqueeze(0)], dim=0
    )
    policy_advantages = clipped_pg_rhos * (
        rewards + discounts * targets_t_plus_1 - values
    )
    return VTraceReturns(value_targets, policy_advantages)


def impala_actor_critic_loss(
    behavior_log_probs: torch.Tensor,
    target_log_probs: torch.Tensor,
    entropy: torch.Tensor,
    discounts: torch.Tensor,
    rewards: torch.Tensor,
    values: torch.Tensor,
    bootstrap_value: torch.Tensor,
    value_cost: float = 0.5,
    entropy_cost: float = 0.01,
    valid_mask: torch.Tensor | None = None,
) -> ImpalaLossOutput:
    returns = vtrace_from_log_probs(
        behavior_log_probs=behavior_log_probs,
        target_log_probs=target_log_probs,
        discounts=discounts,
        rewards=rewards,
        values=values,
        bootstrap_value=bootstrap_value,
    )
    if valid_mask is None:
        valid_mask = torch.ones_like(rewards)
    valid_mask = valid_mask.to(dtype=rewards.dtype)
    denominator = valid_mask.sum().clamp_min(1.0)
    policy_loss = -(
        target_log_probs * returns.policy_advantages.detach() * valid_mask
    ).sum() / denominator
    value_loss = 0.5 * (
        torch.square(returns.value_targets.detach() - values) * valid_mask
    ).sum() / denominator
    entropy_loss = -(entropy * valid_mask).sum() / denominator
    total = policy_loss + value_cost * value_loss + entropy_cost * entropy_loss
    return ImpalaLossOutput(
        total=total,
        policy=policy_loss,
        value=value_loss,
        entropy=entropy_loss,
        vtrace=returns,
    )
