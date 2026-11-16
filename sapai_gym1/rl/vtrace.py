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


@torch.no_grad()
def masked_vtrace_from_log_probs(behavior_log_probs, target_log_probs, discounts, rewards,
                                 values, bootstrap_value, lengths, clip_rho_threshold=1.,
                                 clip_pg_rho_threshold=1., *, validate_lengths=True):
    """Variable-length V-trace, vectorized across B with a reverse T loop.

    Padding is removed BEFORE arithmetic and cannot carry corrections. Each
    sequence uses its own bootstrap at its last valid transition. The checked
    public entrypoint accepts arbitrary padding (including NaNs); the learner
    already validates lengths on CPU before transferring its owned batch.
    """
    if values.ndim != 2 or any(x.shape != values.shape for x in
            (behavior_log_probs, target_log_probs, discounts, rewards)):
        raise ValueError('Expected identical [T,B] inputs')
    time_steps, batch_size = values.shape
    if lengths.shape != (batch_size,) or bootstrap_value.shape != (batch_size,):
        raise ValueError('Lengths and bootstraps must have shape [B]')
    if lengths.dtype not in (torch.int16, torch.int32, torch.int64):
        raise ValueError('Lengths must be integer tensors')
    if validate_lengths and ((lengths < 1) | (lengths > time_steps)).any():
        raise ValueError('Invalid sequence lengths')
    steps = torch.arange(time_steps, device=values.device)[:, None]
    valid, has_next = steps < lengths[None], steps + 1 < lengths[None]
    v = torch.where(valid, values, 0.)
    r, d = torch.where(valid, rewards, 0.), torch.where(valid, discounts, 0.)
    lp, bp = torch.where(valid, target_log_probs, 0.), torch.where(valid, behavior_log_probs, 0.)
    rho = (lp - bp).exp()
    clipped = rho.clamp(max=clip_rho_threshold)
    c = rho.clamp(max=1.)
    next_v = torch.where(has_next, torch.cat((v[1:], torch.zeros_like(v[:1])), dim=0), bootstrap_value[None])
    delta = clipped * (r + d * next_v - v)
    correction = torch.zeros_like(bootstrap_value)
    corrections = []
    for t in range(time_steps - 1, -1, -1):
        correction = torch.where(valid[t], delta[t] + d[t] * c[t] * correction, 0.)
        corrections.append(correction)
    targets = torch.where(valid, v + torch.stack(corrections[::-1]), 0.)
    next_target = torch.where(has_next, torch.cat((targets[1:], torch.zeros_like(targets[:1])), dim=0), bootstrap_value[None])
    advantages = torch.where(valid, rho.clamp(max=clip_pg_rho_threshold) * (r + d * next_target - v), 0.)
    return VTraceReturns(targets, advantages)


def masked_impala_actor_critic_loss(behavior_log_probs, target_log_probs, entropy, discounts,
                                    rewards, values, bootstrap_value, lengths, value_cost=.5,
                                    entropy_cost=.01, *, validate_lengths=True):
    returns = masked_vtrace_from_log_probs(behavior_log_probs, target_log_probs, discounts, rewards,
        values, bootstrap_value, lengths, validate_lengths=validate_lengths)
    valid = torch.arange(values.shape[0], device=values.device)[:, None] < lengths[None]
    denominator = valid.sum().to(rewards.dtype).clamp_min(1.)
    safe_lp = torch.where(valid, target_log_probs, 0.)
    safe_values = torch.where(valid, values, 0.)
    safe_entropy = torch.where(valid, entropy, 0.)
    policy_loss = -(safe_lp * returns.policy_advantages).sum() / denominator
    value_loss = .5 * ((returns.value_targets - safe_values).square() * valid).sum() / denominator
    entropy_loss = -(safe_entropy * valid.to(rewards.dtype)).sum() / denominator
    return ImpalaLossOutput(policy_loss + value_cost * value_loss + entropy_cost * entropy_loss,
                            policy_loss, value_loss, entropy_loss, returns)
