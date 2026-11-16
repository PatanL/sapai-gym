"""Public-only online PUCT execution inside an independent truth evaluator."""
from collections import Counter
from dataclasses import replace
import hashlib
import time

import numpy as np

from .action_schema import ActionType
from .decision_manifest import observation_hash
from .interactive_evaluation import InteractiveSelfPlayEvaluator
from .puct_search import StochasticPUCTSearch
from .rollout_search import PublicSearchState
from .opponent_belief import OpponentBeliefMix


class PublicPUCTEngine:
    """Separate instance/counters per independent future. No truth-state input."""
    def __init__(self, evaluator, config, *, match_tree_budget=6000, searches_per_turn=2, reservoir=None):
        self.evaluator, self.config = evaluator, config
        self.match_tree_budget, self.searches_per_turn = match_tree_budget, searches_per_turn
        self.reservoir = reservoir
        self.counts = Counter(); self.by_turn = Counter(); self.mask_mass = []

    def choose(self, public, memory, raw_action, next_memory):
        self.counts['ownContinuationDecisions'] += 1
        turn = public.core.turn
        remaining = self.match_tree_budget-self.counts['treeTransitions']
        if remaining < 1 or self.by_turn[turn] >= self.searches_per_turn:
            self.counts['budgetFallbacks'] += 1
            return raw_action, next_memory
        self.by_turn[turn] += 1
        if self.reservoir is not None:
            public.set_opponent_transition_model(self.reservoir, candidate_limit=32)
        public.enable_learned_opponent_belief(OpponentBeliefMix())
        config = replace(self.config, max_simulator_transitions=min(remaining, self.config.max_simulator_transitions))
        planner = StochasticPUCTSearch(self.evaluator, config)
        # The planning random seed depends only on public state, not the outer
        # evaluator's hidden future seed or current enemy board/shop.
        seed = int(hashlib.sha256((observation_hash(public.observation())+'quality-engine-v1').encode()).hexdigest()[:15], 16)
        started = time.perf_counter()
        before = self.evaluator.metrics()
        result = planner.search(public, memory, seed=seed, validation_seed=seed+1)
        self.counts['wallSeconds'] += time.perf_counter()-started
        self.counts['searches'] += 1
        self.counts['treeTransitions'] += result.simulator_transitions
        self.counts['totalSimulatorTransitions'] += (result.simulator_transitions + result.screening_simulator_transitions
            + result.validation_simulator_transitions + result.audit_simulator_transitions)
        for key, value in self.evaluator.metrics().items():
            if key != 'maxOmittedEndMaskMass':
                self.counts['neural_'+key] += value-before[key]
        self.mask_mass.append(result.omitted_end_mask_probability_mass)
        if result.selected_action == raw_action:
            self.counts['unchangedOrConfidenceFallbacks'] += 1
        else:
            self.counts['changedDecisions'] += 1
        # This GRU depends on observation, not selected action. Reuse the
        # already computed live next memory; tree evaluation is speculative.
        return result.selected_action, next_memory


class InteractiveEngineEvaluator(InteractiveSelfPlayEvaluator):
    def __init__(self, policy, *, engine_factory, **kwargs):
        super().__init__(policy, **kwargs)
        self.engine_factory = engine_factory
        self.engines = {}

    def evaluate(self, *args, **kwargs):
        self.engines = {}
        return super().evaluate(*args, **kwargs)

    def _execution_choice(self, record, acting_side, observation, memory, side, action, next_memory):
        if side != acting_side:
            return action, next_memory
        engine = self.engines.setdefault(id(record), None)
        if engine is None:
            engine = self.engines[id(record)] = self.engine_factory()
        # Only this trusted evaluator sees the full environment. The planner
        # gets a sanitized public clone, never record/environment/truth seeds.
        try:
            public = PublicSearchState.from_core(record['environment'].core, side)
        except ValueError:
            core = record['environment'].core
            history = core.previous_opponent_state if side == 0 else core.previous_agent_state
            if history is not None:
                raise  # Do not disguise a real invalid-state bug as fallback.
            engine.counts['publicContextUnavailableFallbacks'] += 1
            return action, next_memory
        return engine.choose(public, memory, action, next_memory)

    def execution_metrics(self):
        totals = Counter(); masks = []
        for engine in self.engines.values():
            totals.update(engine.counts); masks.extend(engine.mask_mass)
        return {**totals, 'maxOmittedMaskMass': max(masks, default=0.),
                'meanOmittedMaskMass': float(np.mean(masks)) if masks else 0.,
                'budgetScope': 'tree transitions capped per future; confirmation/screening separately counted by planner; not a total-work or wall deadline claim'}
