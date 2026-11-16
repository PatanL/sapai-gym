"""A diagnostic continuation constraint, never a production strategy rule."""
from collections import Counter

import numpy as np
import torch

from .action_schema import ActionType, AutoregressiveAction, MAX_TEAM_SLOTS, MAX_SHOP_SLOTS
from .historical_evaluation import HistoricalOpponentEvaluator


def without_level_two(observation, species_id):
    """Keep legal alternatives but exclude merges that create level two."""
    allowed = np.asarray(observation['target_mask'], dtype=bool).copy()
    ids, numeric = observation['item_ids'], observation['numeric']
    for kind in (ActionType.BUY_PET, ActionType.COMBINE):
        sources = MAX_SHOP_SLOTS if kind == ActionType.BUY_PET else MAX_TEAM_SLOTS
        for source in range(sources):
            source_token = 1 + MAX_TEAM_SLOTS + source if kind == ActionType.BUY_PET else 1 + source
            for target in range(MAX_TEAM_SLOTS):
                if not allowed[kind, source, target]:
                    continue
                target_token = 1 + target
                if int(ids[source_token]) != species_id or int(ids[target_token]) != species_id:
                    continue
                levels = [round(float(numeric[token, 2]) * 3) for token in (source_token, target_token)]
                experience = sum(round(float(numeric[token, 3]) * 5) for token in (source_token, target_token)) + 1
                if levels == [1, 1] and experience >= 2:
                    allowed[kind, source, target] = False
    return allowed


def constrained_hierarchical_choice(joint_log_prob, allowed):
    """Preserve original type/source/target rankings without changing inputs."""
    allowed = torch.as_tensor(allowed, dtype=torch.bool, device=joint_log_prob.device)
    if not allowed.any():
        raise ValueError('No legal alternative under the diagnostic constraint')
    types = torch.logsumexp(joint_log_prob, dim=(1, 2))
    kind = int(types.masked_fill(~allowed.any(dim=(1, 2)), -torch.inf).argmax())
    sources = torch.logsumexp(joint_log_prob[kind], dim=1)
    source = int(sources.masked_fill(~allowed[kind].any(dim=1), -torch.inf).argmax())
    target = int(joint_log_prob[kind, source].masked_fill(~allowed[kind, source], -torch.inf).argmax())
    return AutoregressiveAction(kind, source, target)


class DeferredLevelUpDiagnostic(HistoricalOpponentEvaluator):
    """Test delaying one species' level-two merge on turn two only.

    This is expert-suggested diagnostic scaffolding, not an optimality claim.
    The real environment's rules and masks are unchanged. Ordinary choices
    are preserved unless the selected action would perform the delayed merge.
    """
    def __init__(self, *args, species_id, **kwargs):
        super().__init__(*args, **kwargs)
        self.species_id = int(species_id)
        self.counters = Counter()

    def evaluate(self, *args, **kwargs):
        self.counters.clear()
        return super().evaluate(*args, **kwargs)

    @torch.inference_mode()
    def _continuation_choice(self, observation, memory, side, action, next_memory):
        turn = round(float(observation['numeric'][0, 12]) * 25)
        if side != self.acting_side or turn != 2:
            return action, next_memory
        allowed = without_level_two(observation, self.species_id)
        if allowed[action.as_tuple()]:
            return action, next_memory
        batch = {key: torch.as_tensor(value, device=self.device).unsqueeze(0) for key, value in observation.items()}
        with self._autocast():
            output = self.policy.evaluate_all_actions(batch, memory=torch.as_tensor(memory, device=self.device).float().unsqueeze(0))
        replacement = constrained_hierarchical_choice(output.joint_log_prob[0].float(), allowed)
        self.counters['blockedActions'] += 1
        self.counters[f'blocked_{ActionType(action.action_type).name}'] += 1
        self.counters[f'replacement_{ActionType(replacement.action_type).name}'] += 1
        return replacement, next_memory
