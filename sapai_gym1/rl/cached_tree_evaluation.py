"""Execute a frozen public PUCT plan without searching or observing private state."""
from collections import Counter, defaultdict
import math
from types import MappingProxyType

import numpy as np
import torch

from .decision_manifest import observation_hash
from .historical_evaluation import HistoricalOpponentEvaluator


class PublicObservationView:
    """The tree lookup sees only copied read-only observation arrays."""
    __slots__ = ('_observation',)

    def __init__(self, observation):
        arrays = {}
        for key, value in observation.items():
            array = np.asarray(value).copy()
            array.setflags(write=False)
            arrays[key] = array
        self._observation = MappingProxyType(arrays)

    def observation(self):
        return self._observation


def legal_observed_action(observation, action):
    kind, source, target = action.as_tuple()
    masks = (observation['type_mask'], observation['source_mask'], observation['target_mask'])
    if not (0 <= kind < masks[0].shape[0]
            and 0 <= source < masks[1].shape[1]
            and 0 <= target < masks[2].shape[2]):
        return False
    return bool(masks[0][kind] and masks[1][kind, source] and masks[2][kind, source, target])


class CachedTreeOpponentEvaluator(HistoricalOpponentEvaluator):
    """Apply cached own-side actions after the forced root; otherwise use policy.

    No new search is performed and the root is never overwritten. The opposing
    policy remains unchanged. Use this to distinguish root-label value from
    executing the cached plan. Do not equate it with repeated online replanning.
    """

    def __init__(self, policy, opponent_policy, *, tree, acting_side,
                 numerical_memory_tolerance=0.0, **kwargs):
        if not math.isfinite(numerical_memory_tolerance) or not 0 <= numerical_memory_tolerance <= 1e-5:
            raise ValueError('Numerical memory tolerance must be finite and in [0, 1e-5]')
        if getattr(tree.evaluator, 'policy', None) is not policy:
            raise ValueError('Cached tree belongs to a different policy')
        super().__init__(policy, opponent_policy, acting_side=acting_side, **kwargs)
        self.tree = tree
        self.numerical_memory_tolerance = numerical_memory_tolerance
        self.counters = Counter()
        self.observation_nodes = defaultdict(list)
        for node in tree.nodes.values():
            if node.visits:
                self.observation_nodes[observation_hash(node.observation)].append(node)

    def evaluate(self, *args, **kwargs):
        self.counters.clear()
        return super().evaluate(*args, **kwargs)

    def _continuation_choice(self, observation, memory, side, action, next_memory):
        if side != self.acting_side:
            return action, next_memory
        self.counters['ownContinuationDecisions'] += 1
        cached = self.tree.tree_action(PublicObservationView(observation), memory)
        if cached is None:
            nodes = self.observation_nodes.get(observation_hash(observation), ())
            if nodes:
                self.counters['observationMatchesWithoutExactMemory'] += 1
                current = torch.as_tensor(memory).float().cpu()
                candidates = [(node, float((current - node.memory.float().cpu()).abs().max()))
                              for node in nodes if node.memory.shape == current.shape]
                distances = [distance for _, distance in candidates]
                if distances and min(distances) <= 2e-6:
                    # Diagnose numerical instability; do not silently merge memories.
                    self.counters['nearNumericalMemoryMisses'] += 1
                if self.numerical_memory_tolerance:
                    matching = [(node, distance) for node, distance in candidates
                                if distance <= self.numerical_memory_tolerance]
                    # Agreement is required across all matching recurrent contexts.
                    # Do not pick whichever nearly equal memory gives a preferred action.
                    actions = {
                        max(node.priors, key=lambda candidate: (
                            node.edges[candidate].visits,
                            node.edges[candidate].mean_value,
                            node.priors[candidate],
                        )) for node, _ in matching
                    }
                    if len(actions) == 1:
                        cached = (next(iter(actions)), None)
                        self.counters['numericalCacheHits'] += 1
                        self.counters['maximumAcceptedMemoryDistance'] = max(
                            self.counters['maximumAcceptedMemoryDistance'],
                            max(distance for _, distance in matching),
                        )
                    elif len(actions) > 1:
                        self.counters['ambiguousNumericalMatches'] += 1
            if cached is None:
                return action, next_memory
        cached_action, _ = cached
        if not legal_observed_action(observation, cached_action):
            raise RuntimeError('Cached tree action violates the current public legal masks')
        self.counters['cacheHits'] += 1
        if cached_action != action:
            self.counters['changedActions'] += 1
        # Advance memory with the actual current observation's forward pass,
        # not a hidden vector imported from another rollout.
        return cached_action, next_memory
