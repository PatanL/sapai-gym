"""Batched variable-length return/auxiliary losses; old objective unchanged."""
from dataclasses import dataclass

import torch
from torch.nn import functional as F

from .vtrace import masked_impala_actor_critic_loss


@dataclass
class DeviceLearnerMetrics:
    # total, policy, value, battle auxiliary, opponent auxiliary, entropy
    values: torch.Tensor
    transitions: int


def batched_sequence_losses(policy, output, packed, bootstrap_value, battle_aux_cost, opponent_aux_cost):
    losses = masked_impala_actor_critic_loss(packed.behavior_log_probs, output.log_prob,
        output.entropy, packed.discounts, packed.rewards, output.value, bootstrap_value,
        packed.lengths, validate_lengths=False)
    valid = torch.arange(output.value.shape[0], device=output.value.device)[:, None] < packed.lengths[None]
    total = losses.total
    zero = torch.zeros((), device=total.device)
    battle_loss = zero
    if battle_aux_cost > 0:
        selected = valid & (packed.battle_outcomes[None] >= 0)
        # Preserve legacy segment-major concatenation order.
        if selected.any():
            logits = output.battle_logits.transpose(0, 1)[selected.T]
            labels = packed.battle_outcomes[:, None].expand(-1, valid.shape[0])[selected.T]
            battle_loss = F.cross_entropy(logits, labels)
            total = total + battle_aux_cost * battle_loss
    opponent_loss = zero
    bootstrap = packed.bootstrap_observations
    board_mask = bootstrap.get('opponent_history_board_mask')
    if opponent_aux_cost > 0 and board_mask is not None:
        selected = board_mask[:, -1].bool()
        if selected.any():
            items = bootstrap['opponent_history_item_ids'][selected, -1].long()
            statuses = bootstrap['opponent_history_status_ids'][selected, -1].long()
            numeric = bootstrap['opponent_history_numeric'][selected, -1].float()
            occupied = bootstrap['opponent_history_mask'][selected, -1].bool()
            pet_loss = F.cross_entropy(output.opponent_pet_logits[0, selected].float().reshape(-1, policy.config.num_items), items.reshape(-1))
            status_loss = level_loss = stats_loss = temporary_loss = zero
            if occupied.any():
                status_loss = F.cross_entropy(output.opponent_status_logits[0, selected].float()[occupied], statuses[occupied])
                levels = torch.round(numeric[..., 2] * 3).long()
                level_loss = F.cross_entropy(output.opponent_level_logits[0, selected].float()[occupied], levels[occupied])
                stats_loss = F.mse_loss(output.opponent_stats[0, selected].float()[occupied], numeric[..., :2][occupied])
                temporary_loss = F.binary_cross_entropy_with_logits(output.opponent_temporary_logits[0, selected].float()[occupied], numeric[..., 5:7][occupied])
            opponent_loss = pet_loss + .25 * status_loss + .25 * level_loss + .5 * stats_loss + .1 * temporary_loss
            total = total + opponent_aux_cost * opponent_loss
    return total, losses.policy, losses.value, losses.entropy, battle_loss, opponent_loss
