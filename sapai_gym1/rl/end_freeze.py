"""Experimental learned end-of-shop freeze subsets; no pet-specific strategy rules."""
from dataclasses import dataclass
import math

import numpy as np
import torch
from torch import nn
from torch.distributions import Categorical

from .action_schema import ActionType, AutoregressiveAction, FlatActionCodec, MAX_SHOP_SLOTS, NO_TARGET
from .selfplay_env import SelfPlayStep


SHOP_ACTION_CAP = 50
NUM_FREEZE_SUBSETS = 1 << MAX_SHOP_SLOTS


@dataclass
class EndFreezeOutput:
    mask_ids: torch.Tensor
    log_prob: torch.Tensor
    entropy: torch.Tensor | None
    slot_logits: torch.Tensor


@dataclass
class EndFreezeStep:
    result: SelfPlayStep
    # Retain the primitive actions for replay/telemetry. A future collector
    # must nevertheless record ONE sampled joint action and its joint log-p.
    primitive_actions: tuple[tuple[AutoregressiveAction, ...], tuple[AutoregressiveAction, ...]]


class EndFreezeHead(nn.Module):
    """Learn toggle subsets when END_TURN is selected at zero gold.

    The exact categorical distribution enumerates at most 1,024 subsets and
    conditions on legality and the remaining action budget. Positive bits
    toggle current freeze state. Zero means preserve all current flags.
    """
    def __init__(self, d_model, initial_toggle_probability=0.1):
        super().__init__()
        if not 0 < initial_toggle_probability < 0.5:
            raise ValueError('Initial toggle probability must be between zero and one half')
        self.scorer = nn.Linear(2 * d_model, 1)
        nn.init.zeros_(self.scorer.weight)
        nn.init.constant_(self.scorer.bias, math.log(initial_toggle_probability / (1 - initial_toggle_probability)))
        bits = ((torch.arange(NUM_FREEZE_SUBSETS)[:, None] >> torch.arange(MAX_SHOP_SLOTS)) & 1).bool()
        self.register_buffer('subsets', bits, persistent=False)
        self.register_buffer('subset_sizes', bits.sum(dim=1), persistent=False)

    def distribution(self, slot_logits, observation, action_types):
        if slot_logits.ndim != 2 or slot_logits.shape[1] != MAX_SHOP_SLOTS:
            raise ValueError('Expected one logit per physical shop slot')
        if not torch.isfinite(slot_logits).all():
            raise ValueError('Freeze logits must be finite')
        mask = observation['target_mask'].bool()
        eligible = (mask[:, ActionType.FREEZE, :MAX_SHOP_SLOTS, NO_TARGET]
                    | mask[:, ActionType.UNFREEZE, :MAX_SHOP_SLOTS, NO_TARGET])
        zero_gold = observation['numeric'][:, 0, 11].abs() < 1e-8
        active = ((action_types == int(ActionType.END_TURN)) & zero_gold
                  & observation['type_mask'][:, ActionType.END_TURN].bool())
        eligible = eligible & active[:, None]
        counts = torch.round(observation['numeric'][:, 0, 5].float() * SHOP_ACTION_CAP).long()
        # The existing environment permits 50 non-END actions, then its END
        # escape hatch. Do not reserve an additional slot and change that rule.
        remaining = (SHOP_ACTION_CAP - counts).clamp(0, MAX_SHOP_SLOTS)
        with torch.autocast(device_type=slot_logits.device.type, enabled=False):
            bits = self.subsets.to(device=slot_logits.device, dtype=torch.float32)
            illegal_slot = (~eligible).float() @ bits.T > 0
            allowed = ~illegal_slot & (self.subset_sizes.to(slot_logits.device)[None] <= remaining[:, None])
            logits = slot_logits.float() @ bits.T
            logits = logits.masked_fill(~allowed, -torch.inf)
            return Categorical(logits=logits), allowed

    def forward(self, context, shop_tokens, observation, action_types, *, mask_ids=None, deterministic=False,
                compute_entropy=True):
        if shop_tokens.shape[1] != MAX_SHOP_SLOTS:
            raise ValueError('All ten physical shop tokens are required')
        context = context[:, None].expand(-1, MAX_SHOP_SLOTS, -1)
        slot_logits = self.scorer(torch.cat([context, shop_tokens], dim=-1)).squeeze(-1)
        distribution, allowed = self.distribution(slot_logits, observation, action_types)
        if mask_ids is None:
            selected = distribution.logits.argmax(dim=-1) if deterministic else distribution.sample()
        else:
            selected = torch.as_tensor(mask_ids, device=slot_logits.device)
            if selected.dtype not in (torch.int16, torch.int32, torch.int64, torch.uint8):
                raise ValueError('Freeze subset IDs must be integer tensors')
            if selected.shape != (slot_logits.shape[0],) or ((selected < 0) | (selected >= NUM_FREEZE_SUBSETS)).any():
                raise ValueError('Invalid freeze subset IDs')
            selected = selected.long()
            if not allowed[torch.arange(len(selected), device=selected.device), selected].all():
                raise ValueError('Recorded freeze subset violates legality or action budget')
        return EndFreezeOutput(selected, distribution.log_prob(selected),
                               distribution.entropy() if compute_entropy else None, slot_logits)


def _validate_mask_id(mask_id):
    if isinstance(mask_id, bool) or not isinstance(mask_id, (int, np.integer)) or not 0 <= mask_id < NUM_FREEZE_SUBSETS:
        raise ValueError('Invalid freeze subset ID')


def expand_end_freeze(core, side, mask_id):
    """Validate and return an exact primitive plan; do not execute it here."""
    if side not in (0, 1):
        raise ValueError('side must be zero or one')
    _validate_mask_id(mask_id)
    player = core.agent if side == 0 else core.opponent
    if player.turn_ended:
        raise ValueError('Player already ended this shop')
    indices = [index for index in range(MAX_SHOP_SLOTS) if int(mask_id) & (1 << index)]
    count = core.shop_action_count_agent if side == 0 else core.shop_action_count_opponent
    if len(indices) > max(0, SHOP_ACTION_CAP - count):
        raise ValueError('Freeze plan exceeds remaining action budget')
    if indices and player.gold != 0:
        raise ValueError('End-of-turn freeze plans require zero gold')
    if any(index >= len(player.shop) or player.shop[index].empty for index in indices):
        raise ValueError('Freeze plan references an empty shop position')
    legal = core._avail_actions(side)
    end = AutoregressiveAction(ActionType.END_TURN)
    if FlatActionCodec.encode(end) not in legal:
        raise ValueError('END_TURN is not currently legal')
    plan = []
    for index in indices:
        action = AutoregressiveAction(ActionType.UNFREEZE if player.shop[index].frozen else ActionType.FREEZE, index)
        if FlatActionCodec.encode(action) not in legal:
            raise ValueError('Freeze plan contains an illegal toggle')
        plan.append(action)
    return tuple(plan) + (end,)


def step_with_end_freeze(env, player_zero_action, player_one_action, *, mask_ids=(0, 0)):
    """Execute an opt-in joint decision using only existing simulator actions.

    Validate both sides before changing gameplay state. Independent toggles
    reveal no shop information, consume no RNG/gold, and never resolve battle.
    The ordinary environment performs both final actions and the battle/turn
    transition exactly once. No policy or collector uses this helper by default.
    """
    if len(mask_ids) != 2:
        raise ValueError('Expected one freeze subset ID per player')
    plans = []
    for side, (action, mask_id, active) in enumerate(zip(
        (player_zero_action, player_one_action), mask_ids, env.active_players()
    )):
        _validate_mask_id(mask_id)
        if not active:
            if action is not None or mask_id != 0:
                raise ValueError('Ended players cannot submit actions or freeze plans')
            plans.append(())
            continue
        if action is None:
            raise ValueError('Active players must submit an action')
        action = env._coerce_and_validate(side, action)
        if action.action_type == ActionType.END_TURN:
            plan = expand_end_freeze(env.core, side, mask_id)
        elif mask_id:
            raise ValueError('Freeze subsets are only valid with END_TURN')
        else:
            plan = (action,)
        plans.append(plan)

    for side, plan in enumerate(plans):
        for action in plan[:-1]:
            env.core.resolve_action(side, FlatActionCodec.encode(action))
    result = env.step(*(plan[-1] if plan else None for plan in plans))
    return EndFreezeStep(result, tuple(plans))
