"""Pure decoding/scoring on one cached neural decision encoding.

This API is opt-in. Primitive-only search consumers continue to fail closed on
joint checkpoints until their execution and demonstration paths are migrated.
"""
from dataclasses import dataclass
import torch

from .action_schema import ActionType, AutoregressiveAction, MAX_TEAM_SLOTS, MAX_SHOP_SLOTS
from .joint_action import JointAction


@dataclass(frozen=True)
class DecisionCandidates:
    decisions: tuple[JointAction, ...]
    log_probabilities: tuple[float, ...]
    omitted_probability_mass: float


def decode_decisions(policy, features, *, decisions=None, deterministic=False):
    """Sample or score B decisions without changing caller-owned GRU memory."""
    actions = masks = None
    if decisions is not None:
        if len(decisions) != features.tokens.shape[0]:
            raise ValueError('One decision per encoded observation is required')
        if any((d.end_freeze_mask is not None) != policy.config.end_freeze for d in decisions):
            raise ValueError('Decision and checkpoint action schema disagree')
        actions = torch.tensor([d.primitive.as_tuple() for d in decisions], device=features.tokens.device)
        if policy.config.end_freeze:
            masks = torch.tensor([d.end_freeze_mask for d in decisions], device=features.tokens.device)
        obs = features.observation
        rows = torch.arange(len(decisions), device=actions.device)
        t, s, u = actions.unbind(-1)
        if not (obs['type_mask'][rows, t].bool() & obs['source_mask'][rows, t, s].bool()
                & obs['target_mask'][rows, t, s, u].bool()).all():
            raise ValueError('Cannot score an illegal decision')
    output = policy.decode_decision(features, actions, deterministic,
                                    allow_end_freeze=policy.config.end_freeze,
                                    end_freeze_masks=masks)
    triples = output.actions.detach().cpu().tolist()
    selected = [None] * len(triples) if output.end_freeze_mask is None else output.end_freeze_mask.detach().cpu().tolist()
    return tuple(JointAction(AutoregressiveAction(*a), m) for a, m in zip(triples, selected)), output


@torch.no_grad()
def propose_decisions(policy, features, *, mask_budget=8, exploration_masks=2, generator=None):
    """All legal primitives, budgeted END subsets; preserve exact raw mass.

    Zero and the greedy subset are retained. Exploration is uniform over valid
    omitted subsets, not based on pet-specific preferences. No full subset tree
    expansion or renormalization disguised as original policy probability.
    """
    if features.tokens.shape[0] != 1:
        raise ValueError('Candidate generation currently accepts one state')
    if mask_budget < 2 or not 0 <= exploration_masks <= mask_budget - 2:
        raise ValueError('Reserve at least two mask slots for zero and the mode')
    joint = policy.primitive_log_probabilities(features)[0].float().cpu()
    rows = torch.isfinite(joint).nonzero().tolist()
    mask_choices, mask_lp = [0], None
    if policy.config.end_freeze:
        start = 1 + MAX_TEAM_SLOTS
        end_types = torch.zeros(1, dtype=torch.long, device=features.tokens.device)
        head = policy.end_freeze_head(features.context, features.tokens[:, start:start + MAX_SHOP_SLOTS],
                                      features.observation, end_types, deterministic=True)
        distribution, allowed = policy.end_freeze_head.distribution(head.slot_logits, features.observation, end_types)
        mask_lp = distribution.logits[0].float().cpu()
        valid = allowed[0].cpu().nonzero().flatten()
        ordered = valid[torch.argsort(mask_lp[valid], descending=True)].tolist()
        mask_choices = list(dict.fromkeys([0, ordered[0]]))
        for mask in ordered:
            if len(mask_choices) >= mask_budget - exploration_masks:
                break
            if mask not in mask_choices:
                mask_choices.append(mask)
        remaining = [mask for mask in ordered if mask not in mask_choices]
        if remaining and exploration_masks:
            permutation = torch.randperm(len(remaining), generator=generator).tolist()
            mask_choices.extend(remaining[i] for i in permutation[:exploration_masks])
    decisions, log_probs = [], []
    for t, s, u in rows:
        primitive = AutoregressiveAction(t, s, u)
        for mask in mask_choices if t == ActionType.END_TURN else [0]:
            decisions.append(JointAction(primitive, mask if policy.config.end_freeze else None))
            log_probs.append(float(joint[t, s, u]) + (float(mask_lp[mask]) if mask_lp is not None and t == ActionType.END_TURN else 0.))
    mass = torch.tensor(log_probs, dtype=torch.float64).exp().sum().item()
    return DecisionCandidates(tuple(decisions), tuple(log_probs), max(0., min(1., 1. - mass)))
