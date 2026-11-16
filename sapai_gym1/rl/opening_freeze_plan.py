"""Execute a declared zero-gold freeze configuration as a diagnostic plan."""
from collections import Counter

import numpy as np

from .action_schema import ActionType, AutoregressiveAction, MAX_TEAM_SLOTS
from .cached_tree_evaluation import legal_observed_action
from .historical_evaluation import HistoricalOpponentEvaluator


def next_freeze_plan_action(observation, item_ids, desired_frozen):
    start = 1 + MAX_TEAM_SLOTS
    current_ids = observation['item_ids'][start:start + len(item_ids)]
    if not np.array_equal(current_ids, item_ids):
        raise ValueError('Opening freeze-plan inventory changed unexpectedly')
    for index, desired in enumerate(desired_frozen):
        frozen = bool(observation['numeric'][start + index, 6])
        if frozen != desired:
            action = AutoregressiveAction(ActionType.FREEZE if desired else ActionType.UNFREEZE, index)
            if not legal_observed_action(observation, action):
                raise ValueError('Declared freeze configuration is not legally reachable')
            return action
    action = AutoregressiveAction(ActionType.END_TURN)
    if not legal_observed_action(observation, action):
        raise ValueError('Freeze plan cannot end this shop')
    return action


class OpeningFreezePlanEvaluator(HistoricalOpponentEvaluator):
    """Force only the stated opening freezes, then end; later play stays raw.

    The plan is selected from public shop information before validation. This
    is not a rule that freezing those items is always preferable.
    """
    def __init__(self, *args, item_ids, desired_frozen, **kwargs):
        super().__init__(*args, **kwargs)
        if len(item_ids) != len(desired_frozen):
            raise ValueError('Freeze plan lengths differ')
        self.item_ids = np.asarray(item_ids).copy()
        self.desired_frozen = tuple(map(bool, desired_frozen))
        self.counters = Counter()

    def evaluate(self, *args, **kwargs):
        self.counters.clear()
        return super().evaluate(*args, **kwargs)

    def _continuation_choice(self, observation, memory, side, action, next_memory):
        turn = round(float(observation['numeric'][0, 12]) * 25)
        if side != self.acting_side or turn != 1:
            return action, next_memory
        if round(float(observation['numeric'][0, 11]) * 30) != 0:
            raise ValueError('Opening freeze plans require zero gold')
        planned = next_freeze_plan_action(observation, self.item_ids, self.desired_frozen)
        self.counters['plannedContinuationActions'] += 1
        self.counters['changedActions'] += int(planned != action)
        self.counters[ActionType(planned.action_type).name] += 1
        return planned, next_memory
