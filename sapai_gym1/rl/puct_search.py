"""Globally budgeted public-information PUCT for stochastic SAP shops."""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field
import hashlib
import math
import time

import numpy as np
import torch

from .action_schema import ActionType, AutoregressiveAction
from .rollout_search import (
    CounterfactualRolloutSearch,
    RolloutSearchConfig,
)


PUCT_VERSION = "public-stochastic-puct-v14-joint-20260906"


@dataclass(frozen=True)
class PUCTConfig:
    max_simulations: int = 256
    max_simulator_transitions: int = 20_000
    max_tree_depth: int = 64
    max_rollout_steps: int = 180
    # Optional cheap leaf estimates during tree construction only. Screening,
    # validation and audits still use max_rollout_steps and the battle horizon.
    tree_leaf_rollout_steps: int | None = None
    horizon_battles: int = 5
    world_pool_size: int = 32
    min_root_visits: int = 1
    cpuct: float = 1.5
    prior_floor: float = 0.02
    root_dirichlet_alpha: float = 0.3
    root_noise_fraction: float = 0.0
    leaf_value_weight: float = 0.5
    # None preserves legacy full-parent first-play urgency. Explicit values
    # allow controlled comparison in the same scale as downweighted leaves.
    unvisited_value_weight: float | None = None
    value_scale: float = 2.5
    validation_rollouts: int = 32
    max_validation_rollouts: int = 128
    audit_rollouts: int = 64
    confidence_z: float = 1.96
    minimum_improvement: float = 0.0
    deployment_min_visits: int = 4
    deployment_confidence_z: float = 1.0
    root_screening_top_k: int = 8
    root_screening_rollouts: int = 8
    root_screening_rounds: int = 3
    root_screening_min_survivors: int = 4
    target_temperature: float = 0.10
    target_min_node_visits: int = 4
    target_min_action_visits: int = 2
    prune_affordable_freezes: bool = False
    exhaustive_root_probability: float = 0.05
    action_quotas: dict[ActionType, int] = field(default_factory=dict)
    # Scientific equal-budget comparisons can use the tree's own proposal and
    # a shared external evaluator. Production search leaves this disabled and
    # retains the independent screening/validation/audit pipeline below.
    selection_only: bool = False
    # Keep the matched-seed root tournament but omit the production
    # validation/audit stages when a benchmark supplies one shared untouched
    # comparison panel for every planner.
    external_confirmation_only: bool = False


@dataclass
class PUCTEdge:
    visits: int = 0
    value_sum: float = 0.0
    value_square_sum: float = 0.0

    @property
    def mean_value(self):
        return self.value_sum / self.visits if self.visits else 0.0

    @property
    def standard_error(self):
        if self.visits < 2:
            return float("inf")
        mean = self.mean_value
        variance = max(
            0.0,
            (self.value_square_sum - self.visits * mean * mean)
            / (self.visits - 1),
        )
        return math.sqrt(variance / self.visits)


@dataclass
class PUCTNode:
    key: bytes
    priors: dict[AutoregressiveAction, float]
    value: float
    next_memory: torch.Tensor
    observation: dict[str, np.ndarray]
    memory: torch.Tensor
    turn: int
    opponent_belief: object | None = None
    edges: dict[AutoregressiveAction, PUCTEdge] = field(default_factory=dict)
    visits: int = 0
    omitted_probability_mass: float = 0.0
    policy_action: object | None = None

    def __post_init__(self):
        if not self.edges:
            self.edges = {action: PUCTEdge() for action in self.priors}


@dataclass(frozen=True)
class PUCTTrainingTarget:
    observation: dict[str, np.ndarray]
    memory: torch.Tensor
    actions: tuple[AutoregressiveAction, ...]
    probabilities: tuple[float, ...]
    visits: int
    method: str = "q-lower-bound"
    is_root: bool = False
    turn: int = 0
    omitted_end_mask_probability_mass: float = 0.0


@dataclass(frozen=True)
class PUCTSearchResult:
    tree_proposed_action: AutoregressiveAction
    proposed_action: AutoregressiveAction
    selected_action: AutoregressiveAction
    policy_action: AutoregressiveAction
    accepted: bool
    candidate_eligible: bool
    exhaustive_root: bool
    full_candidate_count: int
    pruned_candidate_count: int
    tree_proposed_excluded_by_pruning: bool | None
    proposed_excluded_by_pruning: bool | None
    pruned_action_counts: dict[str, int]
    visit_counts: dict[AutoregressiveAction, int]
    q_values: dict[AutoregressiveAction, float]
    q_standard_errors: dict[AutoregressiveAction, float]
    q_lower_bounds: dict[AutoregressiveAction, float]
    priors: dict[AutoregressiveAction, float]
    root_screening_scores: dict[AutoregressiveAction, float]
    root_screening_counts: dict[AutoregressiveAction, int]
    simulations: int
    simulator_transitions: int
    tree_nodes: int
    mean_tree_depth: float
    max_tree_depth: int
    validation_return_improvement: float
    validation_return_standard_error: float
    validation_return_lower_bound: float
    validation_first_battle_score_improvement: float
    validation_horizon_battle_score_improvement: float
    validation_differences: tuple[float, ...]
    validation_rollouts_used: int
    validation_tree_actions_mean: float
    validation_planned_opponent_source_counts: dict[str, int]
    validation_policy_opponent_source_counts: dict[str, int]
    audit_return_improvement: float | None
    audit_return_standard_error: float | None
    audit_return_lower_bound: float | None
    audit_first_battle_score_improvement: float | None
    audit_horizon_battle_score_improvement: float | None
    audit_tree_actions_mean: float | None
    audit_planned_opponent_source_counts: dict[str, int]
    audit_policy_opponent_source_counts: dict[str, int]
    screening_simulator_transitions: int
    validation_simulator_transitions: int
    audit_simulator_transitions: int
    tree_wall_seconds: float
    screening_wall_seconds: float
    validation_wall_seconds: float
    audit_wall_seconds: float
    targets: tuple[PUCTTrainingTarget, ...]
    omitted_end_mask_probability_mass: float = 0.0


class StochasticPUCTSearch:
    """PUCT over public observations with sampled hidden future worlds.

    A finite pool of hidden shop/battle/opponent determinizations is reused
    across simulations. Deterministic shop prefixes therefore share tree
    nodes, while chance outcomes branch by their resulting public observation.
    Every simulator step, including leaf policy rollout, consumes one global
    transition budget.
    """

    def __init__(self, evaluator, config=None):
        self.evaluator = evaluator
        self.config = config or PUCTConfig()
        self._validate_config()
        self._selector = CounterfactualRolloutSearch(
            evaluator,
            RolloutSearchConfig(
                leaf_value_weight=self.config.leaf_value_weight,
                horizon_battles=self.config.horizon_battles,
                max_continuation_steps=self.config.max_rollout_steps,
                prune_affordable_freezes=(
                    self.config.prune_affordable_freezes
                ),
                action_quotas=(
                    dict(self.config.action_quotas)
                    if self.config.action_quotas
                    else RolloutSearchConfig().action_quotas
                ),
            ),
        )
        self.nodes = {}
        self.root_node = None
        self.simulator_transitions = 0
        self.rollout_simulator_transitions = 0
        self.simulation_depths = []

    def _validate_config(self):
        if self.config.selection_only and self.config.external_confirmation_only:
            raise ValueError(
                "selection_only and external_confirmation_only are exclusive"
            )
        if self.config.max_simulations <= 0:
            raise ValueError("max_simulations must be positive")
        if self.config.max_simulator_transitions <= 0:
            raise ValueError("max_simulator_transitions must be positive")
        if self.config.max_tree_depth <= 0:
            raise ValueError("max_tree_depth must be positive")
        if self.config.max_rollout_steps <= 0:
            raise ValueError("max_rollout_steps must be positive")
        if (self.config.tree_leaf_rollout_steps is not None
                and (not isinstance(self.config.tree_leaf_rollout_steps, int)
                     or isinstance(self.config.tree_leaf_rollout_steps, bool)
                     or self.config.tree_leaf_rollout_steps < 0)):
            raise ValueError("tree_leaf_rollout_steps must be None or a nonnegative integer")
        if self.config.horizon_battles <= 0:
            raise ValueError("horizon_battles must be positive")
        if self.config.world_pool_size <= 0:
            raise ValueError("world_pool_size must be positive")
        if self.config.min_root_visits < 0:
            raise ValueError("min_root_visits must be non-negative")
        if self.config.cpuct < 0:
            raise ValueError("cpuct must be non-negative")
        if not 0.0 <= self.config.prior_floor <= 1.0:
            raise ValueError("prior_floor must be in [0, 1]")
        if not 0.0 <= self.config.root_noise_fraction <= 1.0:
            raise ValueError("root_noise_fraction must be in [0, 1]")
        if self.config.root_dirichlet_alpha <= 0:
            raise ValueError("root_dirichlet_alpha must be positive")
        if not 0.0 <= self.config.leaf_value_weight <= 1.0:
            raise ValueError("leaf_value_weight must be in [0, 1]")
        if (self.config.unvisited_value_weight is not None
                and (isinstance(self.config.unvisited_value_weight, bool)
                     or not isinstance(self.config.unvisited_value_weight, (int, float))
                     or not math.isfinite(self.config.unvisited_value_weight)
                     or not 0 <= self.config.unvisited_value_weight <= 1)):
            raise ValueError("unvisited_value_weight must be None or finite and in [0, 1]")
        if self.config.value_scale <= 0:
            raise ValueError("value_scale must be positive")
        if self.config.validation_rollouts < 2:
            raise ValueError("validation_rollouts must be at least two")
        if self.config.max_validation_rollouts < self.config.validation_rollouts:
            raise ValueError(
                "max_validation_rollouts must be at least validation_rollouts"
            )
        if self.config.audit_rollouts not in {0} and self.config.audit_rollouts < 2:
            raise ValueError("audit_rollouts must be zero or at least two")
        if self.config.deployment_min_visits <= 0:
            raise ValueError("deployment_min_visits must be positive")
        if self.config.deployment_confidence_z < 0:
            raise ValueError("deployment_confidence_z must be non-negative")
        if self.config.root_screening_top_k <= 0:
            raise ValueError("root_screening_top_k must be positive")
        if self.config.root_screening_rollouts < 2:
            raise ValueError("root_screening_rollouts must be at least two")
        if self.config.root_screening_rounds <= 0:
            raise ValueError("root_screening_rounds must be positive")
        if self.config.root_screening_min_survivors <= 0:
            raise ValueError("root_screening_min_survivors must be positive")
        if (
            self.config.root_screening_min_survivors
            > self.config.root_screening_top_k
        ):
            raise ValueError(
                "root_screening_min_survivors cannot exceed "
                "root_screening_top_k"
            )
        if self.config.target_temperature <= 0:
            raise ValueError("target_temperature must be positive")
        if self.config.target_min_node_visits <= 0:
            raise ValueError("target_min_node_visits must be positive")
        if self.config.target_min_action_visits < 2:
            raise ValueError("target_min_action_visits must be at least two")
        if not 0.0 <= self.config.exhaustive_root_probability <= 1.0:
            raise ValueError("exhaustive_root_probability must be in [0, 1]")
        if any(value < 0 for value in self.config.action_quotas.values()):
            raise ValueError("action quotas must be non-negative")

    @staticmethod
    def _state_key(state, memory):
        digest = hashlib.blake2b(digest_size=20)
        observation = state.observation()
        for name in sorted(observation):
            digest.update(name.encode())
            digest.update(
                np.ascontiguousarray(observation[name]).tobytes()
            )
        digest.update(
            torch.as_tensor(memory)
            .detach()
            .float()
            .cpu()
            .numpy()
            .tobytes()
        )
        return digest.digest()

    def _normalize_priors(self, priors):
        actions = tuple(priors)
        if not actions:
            raise ValueError("PUCT node has no candidate actions")
        floor = self.config.prior_floor / len(actions)
        scaled = {
            action: (1.0 - self.config.prior_floor) * priors[action] + floor
            for action in actions
        }
        total = sum(scaled.values())
        return {action: value / total for action, value in scaled.items()}

    def _expand(
        self,
        state,
        memory,
        *,
        root=False,
        rng=None,
        exhaustive_root=False,
    ):
        key = self._state_key(state, memory)
        existing = self.nodes.get(key)
        if existing is not None:
            # Reused nodes bypass neural evaluation, which normally installs
            # the current public-history belief on the rollout state.
            if hasattr(state, "set_opponent_board_belief"):
                state.set_opponent_board_belief(existing.opponent_belief)
            return existing, False
        evaluation = self.evaluator.evaluate(state, memory)
        candidates = self._selector._candidate_actions(
            evaluation.priors, state
        )
        if root:
            if evaluation.policy_action is not None:
                policy_action = evaluation.policy_action
            else:
                policy_action, _, _ = self.evaluator.choose(state, memory, deterministic=True)
            if policy_action not in candidates:
                candidates.append(policy_action)
            self._root_full_actions = tuple(evaluation.priors)
            self._root_pruned_actions = frozenset(candidates)
            if exhaustive_root:
                candidates = list(self._root_full_actions)
        priors = self._normalize_priors(
            {action: evaluation.priors[action] for action in candidates}
        )
        if root and self.config.root_noise_fraction and len(priors) > 1:
            actions = tuple(priors)
            noise = rng.dirichlet(
                np.full(len(actions), self.config.root_dirichlet_alpha)
            )
            fraction = self.config.root_noise_fraction
            priors = {
                action: (1.0 - fraction) * priors[action]
                + fraction * float(noise[index])
                for index, action in enumerate(actions)
            }
        node = PUCTNode(
            key=key,
            priors=priors,
            value=float(evaluation.value),
            next_memory=evaluation.next_memory.float().cpu(),
            observation={
                name: np.asarray(value).copy()
                for name, value in state.observation().items()
            },
            memory=torch.as_tensor(memory).float().cpu().clone(),
            turn=int(
                getattr(getattr(state, "core", None), "turn", 0)
            ),
            opponent_belief=evaluation.opponent_belief,
            omitted_probability_mass=evaluation.omitted_probability_mass,
            policy_action=evaluation.policy_action,
        )
        self.nodes[key] = node
        return node, True

    def _select(self, node):
        sqrt_visits = math.sqrt(node.visits + 1.0)
        unvisited_weight = (1.0 if self.config.unvisited_value_weight is None
                            else self.config.unvisited_value_weight)

        def score(action):
            edge = node.edges[action]
            q_value = (
                edge.mean_value if edge.visits else unvisited_weight * float(node.value)
            ) / self.config.value_scale
            exploration = (
                self.config.cpuct
                * node.priors[action]
                * sqrt_visits
                / (1 + edge.visits)
            )
            return q_value + exploration, node.priors[action]

        return max(node.priors, key=score)

    def _step(self, state, action, rng):
        if self.simulator_transitions >= self.config.max_simulator_transitions:
            return None
        transition = state.step(action, rng)
        self.simulator_transitions += 1
        return transition

    def _policy_rollout(self, state, memory, rng, target_battles):
        total = 0.0
        discount = 1.0
        steps = 0
        step_limit = (self.config.max_rollout_steps if self.config.tree_leaf_rollout_steps is None
                      else min(self.config.tree_leaf_rollout_steps, self.config.max_rollout_steps))
        while (
            not state.terminated
            and state.battles < target_battles
            and steps < step_limit
            and self.simulator_transitions
            < self.config.max_simulator_transitions
        ):
            action, next_memory, _ = self.evaluator.choose(
                state, memory, deterministic=True
            )
            transition = self._step(state, action, rng)
            if transition is None:
                break
            total += discount * float(transition.reward)
            discount *= float(transition.discount)
            memory = next_memory
            steps += 1
        # The battle horizon is a truncation, not a terminal state. Include
        # the same discounted continuation value used by root rollout search.
        if not state.terminated:
            value = self.evaluator.value(state, memory)
            total += (
                discount * self.config.leaf_value_weight * float(value)
            )
        return total

    def _simulate(self, initial, initial_memory, world_seed, forced_root=None):
        state = initial.clone()
        rng = np.random.default_rng(int(world_seed))
        if hasattr(state, "redeterminize"):
            state.redeterminize(int(rng.integers(0, 2**32 - 1)))
        memory = torch.as_tensor(initial_memory).float().cpu().clone()
        target_battles = state.battles + self.config.horizon_battles
        path = []
        tree_depth = 0
        leaf_return = 0.0

        while (
            not state.terminated
            and state.battles < target_battles
            and tree_depth < self.config.max_tree_depth
            and self.simulator_transitions
            < self.config.max_simulator_transitions
        ):
            node, expanded = self._expand(state, memory)
            if expanded and path:
                if self.config.tree_leaf_rollout_steps == 0:
                    # Expansion just evaluated this exact leaf; do not issue
                    # another identical neural forward for value-only leaves.
                    leaf_return = self.config.leaf_value_weight * float(node.value)
                else:
                    leaf_return = self._policy_rollout(state, memory, rng, target_battles)
                break
            action = (
                forced_root
                if tree_depth == 0 and forced_root is not None
                else self._select(node)
            )
            if action not in node.edges:
                action = self._select(node)
            transition = self._step(state, action, rng)
            if transition is None:
                leaf_return = (
                    self.config.leaf_value_weight
                    * float(self.evaluator.value(state, memory))
                )
                break
            path.append(
                (
                    node,
                    action,
                    float(transition.reward),
                    float(transition.discount),
                )
            )
            memory = node.next_memory
            tree_depth += 1
        else:
            if not state.terminated:
                leaf_return = self._policy_rollout(
                    state, memory, rng, target_battles
                )

        backed_up = leaf_return
        for node, action, reward, discount in reversed(path):
            backed_up = reward + discount * backed_up
            edge = node.edges[action]
            edge.visits += 1
            edge.value_sum += backed_up
            edge.value_square_sum += backed_up * backed_up
            node.visits += 1
        self.simulation_depths.append(tree_depth)
        return bool(path)

    @staticmethod
    def _battle_score(record, first_only=False):
        results = record["battle_results"]
        if not results:
            return 0.5
        if first_only:
            results = results[:1]
        return float(
            np.mean(
                [
                    1.0 if result == 0 else 0.5 if result == 2 else 0.0
                    for result in results
                ]
            )
        )

    def tree_action(self, state, memory):
        """Return the learned tree action for this public state, if covered."""

        node = self.nodes.get(self._state_key(state, memory))
        if node is None or node.visits == 0:
            return None
        if hasattr(state, "set_opponent_board_belief"):
            state.set_opponent_board_belief(node.opponent_belief)
        action = max(
            node.priors,
            key=lambda candidate: (
                node.edges[candidate].visits,
                node.edges[candidate].mean_value,
                node.priors[candidate],
            ),
        )
        return action, node.next_memory

    def _validation_rollout(
        self,
        initial,
        memory,
        root_action,
        seed,
        *,
        use_tree,
    ):
        return self._validation_rollouts_many(
            initial,
            memory,
            (root_action,),
            (seed,),
            use_tree=use_tree,
        )[0]

    def _validation_rollouts_many(
        self,
        initial,
        memory,
        root_actions,
        seeds,
        *,
        use_tree,
    ):
        """Batch neural fallbacks while preserving exact tree decisions."""

        root = self.nodes[self._state_key(initial, memory)]
        specs = [
            (action, int(seed)) for action in root_actions for seed in seeds
        ]
        completed = []
        batch_limit = self._selector.config.max_parallel_rollouts
        for batch_start in range(0, len(specs), batch_limit):
            records = []
            for root_action, seed in specs[
                batch_start : batch_start + batch_limit
            ]:
                state = initial.clone()
                rng = np.random.default_rng(seed)
                if hasattr(state, "redeterminize"):
                    state.redeterminize(
                        int(rng.integers(0, 2**32 - 1))
                    )
                transition = state.step(root_action, rng)
                self.rollout_simulator_transitions += 1
                records.append(
                    {
                        "root_action": root_action,
                        "state": state,
                        "rng": rng,
                        "memory": root.next_memory.clone(),
                        "total": float(transition.reward),
                        "discount": float(transition.discount),
                        "steps": 0,
                        "battle_results": (
                            []
                            if transition.battle_result is None
                            else [int(transition.battle_result)]
                        ),
                        "tree_actions": 0,
                    }
                )
            target_battles = initial.battles + self.config.horizon_battles
            while True:
                active = [
                    record
                    for record in records
                    if not record["state"].terminated
                    and record["state"].battles < target_battles
                    and record["steps"] < self.config.max_rollout_steps
                ]
                if not active:
                    break
                choices = [None] * len(active)
                fallback_indices = []
                for index, record in enumerate(active):
                    tree_choice = (
                        self.tree_action(
                            record["state"], record["memory"]
                        )
                        if use_tree
                        else None
                    )
                    if tree_choice is None:
                        fallback_indices.append(index)
                    else:
                        choices[index] = tree_choice
                        record["tree_actions"] += 1
                if fallback_indices:
                    fallback = [active[index] for index in fallback_indices]
                    if hasattr(self.evaluator, "choose_many"):
                        fallback_choices = self.evaluator.choose_many(
                            [record["state"] for record in fallback],
                            [record["memory"] for record in fallback],
                            deterministic=True,
                        )
                    else:
                        fallback_choices = [
                            self.evaluator.choose(
                                record["state"],
                                record["memory"],
                                deterministic=True,
                            )
                            for record in fallback
                        ]
                    for index, choice in zip(
                        fallback_indices, fallback_choices
                    ):
                        choices[index] = choice[:2]
                for record, (action, next_memory) in zip(active, choices):
                    transition = record["state"].step(action, record["rng"])
                    self.rollout_simulator_transitions += 1
                    record["total"] += (
                        record["discount"] * float(transition.reward)
                    )
                    record["discount"] *= float(transition.discount)
                    record["memory"] = next_memory
                    record["steps"] += 1
                    if transition.battle_result is not None:
                        record["battle_results"].append(
                            int(transition.battle_result)
                        )
            bootstrap = [
                record
                for record in records
                if not record["state"].terminated
            ]
            if bootstrap:
                if hasattr(self.evaluator, "values_many"):
                    values = self.evaluator.values_many(
                        [record["state"] for record in bootstrap],
                        [record["memory"] for record in bootstrap],
                    )
                else:
                    values = [
                        self.evaluator.value(
                            record["state"], record["memory"]
                        )
                        for record in bootstrap
                    ]
                for record, value in zip(bootstrap, values):
                    record["total"] += (
                        record["discount"]
                        * self.config.leaf_value_weight
                        * float(value)
                    )
            for record in records:
                completed.append(
                    {
                        "root_action": record["root_action"],
                        "total": record["total"],
                        "battle_results": record["battle_results"],
                        "tree_actions": record["tree_actions"],
                        "opponent_trajectory_ids": tuple(
                            getattr(
                                record["state"],
                                "selected_opponent_trajectory_ids",
                                (),
                            )
                        ),
                    }
                )
        return completed

    @staticmethod
    def _opponent_source_counts(records):
        counts = Counter()
        for record in records:
            for trajectory_id in record.get("opponent_trajectory_ids", ()):
                text = str(trajectory_id)
                if text.startswith("source-"):
                    source = "-".join(text.split("-", 2)[:2])
                elif text.startswith("belief-"):
                    source = text
                else:
                    source = "unmixed"
                counts[source] += 1
        return dict(sorted(counts.items()))

    def _validate(self, state, memory, proposed, policy, rng):
        planned_records = []
        policy_records = []

        def values(records):
            return np.asarray(
                [record["total"] for record in records],
                dtype=np.float64,
            )

        while True:
            remaining = (
                self.config.max_validation_rollouts
                - len(planned_records)
            )
            batch_size = min(self.config.validation_rollouts, remaining)
            seeds = rng.integers(0, 2**63 - 1, size=batch_size)
            planned_records.extend(
                self._confirmation_rollouts(
                    state,
                    memory,
                    (proposed,),
                    seeds,
                    use_tree=True,
                )
            )
            policy_records.extend(
                self._confirmation_rollouts(
                    state,
                    memory,
                    (policy,),
                    seeds,
                    use_tree=False,
                )
            )
            differences = values(planned_records) - values(policy_records)
            mean = float(differences.mean())
            standard_error = float(
                differences.std(ddof=1) / math.sqrt(len(differences))
            )
            lower = mean - self.config.confidence_z * standard_error
            upper = mean + self.config.confidence_z * standard_error
            if (
                lower > self.config.minimum_improvement
                or upper <= self.config.minimum_improvement
                or len(planned_records)
                >= self.config.max_validation_rollouts
            ):
                break

        def battle_values(records, first_only=False):
            return np.asarray(
                [
                    self._battle_score(record, first_only=first_only)
                    for record in records
                ],
                dtype=np.float64,
            )

        first = float(
            (
                battle_values(planned_records, True)
                - battle_values(policy_records, True)
            ).mean()
        )
        horizon = float(
            (
                battle_values(planned_records)
                - battle_values(policy_records)
            ).mean()
        )
        tree_actions_mean = float(
            np.mean(
                [record.get("tree_actions", 0) for record in planned_records]
            )
        )
        order_only = (
            proposed != policy
            and not self._selector._root_moves_are_strategic(state)
            and (
                ActionType(proposed.action_type) is ActionType.MOVE
                or ActionType(policy.action_type) is ActionType.MOVE
            )
        )
        accepted = (
            not order_only
            and lower > self.config.minimum_improvement
        )
        return (
            accepted,
            mean,
            standard_error,
            lower,
            first,
            horizon,
            tuple(map(float, differences)),
            tree_actions_mean,
            self._opponent_source_counts(planned_records),
            self._opponent_source_counts(policy_records),
            len(differences),
        )

    def _confirmation_rollouts(
        self,
        state,
        memory,
        actions,
        seeds,
        *,
        use_tree,
    ):
        return self._validation_rollouts_many(
            state,
            memory,
            actions,
            seeds,
            use_tree=use_tree,
        )

    def _root_screening_candidates(
        self,
        root,
        policy_action,
        q_lower_bounds,
        q_values,
        visit_counts,
    ):
        """Select a diverse finalist set without trusting one tree statistic."""

        rankings = (
            sorted(root.priors, key=q_lower_bounds.get, reverse=True),
            sorted(root.priors, key=q_values.get, reverse=True),
            sorted(root.priors, key=visit_counts.get, reverse=True),
            sorted(root.priors, key=root.priors.get, reverse=True),
        )
        selected = [policy_action]
        rank = 0
        while (
            len(selected) < self.config.root_screening_top_k
            and rank < len(root.priors)
        ):
            for ranking in rankings:
                action = ranking[rank]
                if action not in selected:
                    selected.append(action)
                    if len(selected) >= self.config.root_screening_top_k:
                        break
            rank += 1
        return tuple(selected)

    def _screen_root_actions(
        self,
        state,
        memory,
        candidates,
        policy_action,
        rng,
    ):
        """Rank tree finalists on fresh paired worlds before confirmation."""

        samples = {action: [] for action in candidates}
        active = list(candidates)
        for round_index in range(self.config.root_screening_rounds):
            batch_size = self.config.root_screening_rollouts * (1, 3, 8)[
                min(round_index, 2)
            ]
            seeds = rng.integers(0, 2**63 - 1, size=batch_size)
            raw_records = self._screen_rollouts(
                state,
                memory,
                (policy_action,),
                seeds,
                use_tree=False,
            )
            raw = np.asarray(
                [record["total"] for record in raw_records],
                dtype=np.float64,
            )
            planned_actions = [
                action for action in active if action != policy_action
            ]
            planned_records = self._screen_rollouts(
                state,
                memory,
                planned_actions,
                seeds,
                use_tree=True,
            )
            by_action = {action: [] for action in planned_actions}
            for record in planned_records:
                by_action[record["root_action"]].append(record["total"])
            for action in active:
                if action == policy_action:
                    samples[action].extend([0.0] * batch_size)
                    continue
                planned = np.asarray(by_action[action], dtype=np.float64)
                samples[action].extend(map(float, planned - raw))
            if round_index + 1 >= self.config.root_screening_rounds:
                continue
            non_policy = [
                action for action in active if action != policy_action
            ]
            keep = min(
                len(non_policy),
                max(
                    self.config.root_screening_min_survivors,
                    math.ceil(len(non_policy) / 2),
                ),
            )
            non_policy.sort(
                key=lambda action: np.mean(samples[action]), reverse=True
            )
            active = non_policy[:keep]
            if policy_action in candidates:
                active.append(policy_action)
        scores = {
            action: (
                float(np.mean(values)) if values else float("-inf")
            )
            for action, values in samples.items()
        }
        counts = {action: len(values) for action, values in samples.items()}
        proposed = max(active, key=lambda action: scores[action])
        return proposed, scores, counts

    def _screen_rollouts(
        self,
        state,
        memory,
        actions,
        seeds,
        *,
        use_tree,
    ):
        return self._validation_rollouts_many(
            state,
            memory,
            actions,
            seeds,
            use_tree=use_tree,
        )

    def _audit(self, state, memory, proposed, policy, rng):
        if not self.config.audit_rollouts:
            return (None, None, None, None, None, None, {}, {})
        seeds = rng.integers(
            0, 2**63 - 1, size=self.config.audit_rollouts
        )
        planned = self._confirmation_rollouts(
            state,
            memory,
            (proposed,),
            seeds,
            use_tree=True,
        )
        raw = self._confirmation_rollouts(
            state,
            memory,
            (policy,),
            seeds,
            use_tree=False,
        )
        differences = np.asarray(
            [left["total"] - right["total"] for left, right in zip(planned, raw)],
            dtype=np.float64,
        )
        mean = float(differences.mean())
        standard_error = float(
            differences.std(ddof=1) / math.sqrt(len(differences))
        )
        lower = mean - self.config.confidence_z * standard_error
        planned_first = np.asarray(
            [self._battle_score(record, True) for record in planned]
        )
        raw_first = np.asarray(
            [self._battle_score(record, True) for record in raw]
        )
        planned_horizon = np.asarray(
            [self._battle_score(record) for record in planned]
        )
        raw_horizon = np.asarray(
            [self._battle_score(record) for record in raw]
        )
        return (
            mean,
            standard_error,
            lower,
            float((planned_first - raw_first).mean()),
            float((planned_horizon - raw_horizon).mean()),
            float(np.mean([record["tree_actions"] for record in planned])),
            self._opponent_source_counts(planned),
            self._opponent_source_counts(raw),
        )

    def _training_targets(self):
        targets = []
        for node in self.nodes.values():
            eligible = {
                action: edge
                for action, edge in node.edges.items()
                if edge.visits >= self.config.target_min_action_visits
            }
            if (
                node.visits < self.config.target_min_node_visits
                or len(eligible) < 2
            ):
                continue
            actions = tuple(eligible)
            scores = np.asarray(
                [
                    edge.mean_value
                    - self.config.deployment_confidence_z
                    * edge.standard_error
                    for edge in eligible.values()
                ],
                dtype=np.float64,
            )
            scores -= scores.max()
            weights = np.exp(scores / self.config.target_temperature)
            weights /= weights.sum()
            targets.append(
                PUCTTrainingTarget(
                    observation={
                        name: value.copy()
                        for name, value in node.observation.items()
                    },
                    memory=node.memory.clone(),
                    actions=actions,
                    probabilities=tuple(map(float, weights)),
                    visits=node.visits,
                    is_root=(node is self.root_node),
                    turn=node.turn,
                    omitted_end_mask_probability_mass=node.omitted_probability_mass,
                )
            )
        return tuple(targets)

    def search(self, state, memory, seed=0, validation_seed=None):
        tree_started = time.perf_counter()
        self.nodes = {}
        self.simulator_transitions = 0
        self.rollout_simulator_transitions = 0
        self.simulation_depths = []
        rng = np.random.default_rng(seed)
        exhaustive_root = (
            rng.random() < self.config.exhaustive_root_probability
        )
        self._root_full_actions = ()
        self._root_pruned_actions = frozenset()
        root, _ = self._expand(
            state,
            memory,
            root=True,
            rng=rng,
            exhaustive_root=exhaustive_root,
        )
        self.root_node = root
        if root.policy_action is not None:
            policy_action = root.policy_action
        else:
            policy_action, _, _ = self.evaluator.choose(state, memory, deterministic=True)
        world_seeds = rng.integers(
            0, 2**63 - 1, size=self.config.world_pool_size
        )
        forced = [
            (action, int(world_seeds[paired_index % len(world_seeds)]))
            for paired_index in range(self.config.min_root_visits)
            for action in root.priors
        ]
        simulations = 0
        while (
            simulations < self.config.max_simulations
            and self.simulator_transitions
            < self.config.max_simulator_transitions
        ):
            if simulations < len(forced):
                forced_action, world_seed = forced[simulations]
            else:
                forced_action = None
                world_seed = int(
                    world_seeds[simulations % len(world_seeds)]
                )
            progressed = self._simulate(
                state,
                memory,
                world_seed,
                forced_root=forced_action,
            )
            if not progressed:
                break
            simulations += 1
        tree_wall_seconds = time.perf_counter() - tree_started

        visit_counts = {
            action: edge.visits for action, edge in root.edges.items()
        }
        q_values = {
            action: edge.mean_value for action, edge in root.edges.items()
        }
        q_standard_errors = {
            action: edge.standard_error
            for action, edge in root.edges.items()
        }
        q_lower_bounds = {
            action: (
                float("-inf")
                if not np.isfinite(q_standard_errors[action])
                else q_values[action]
                - self.config.deployment_confidence_z
                * q_standard_errors[action]
            )
            for action in root.edges
        }
        deployable = [
            action
            for action, edge in root.edges.items()
            if edge.visits >= self.config.deployment_min_visits
        ] or list(root.priors)
        tree_proposed = max(
            deployable,
            key=lambda action: (
                q_lower_bounds[action],
                q_values[action],
                visit_counts[action],
                root.priors[action],
            ),
        )
        validation_rng = (
            rng
            if validation_seed is None
            else np.random.default_rng(validation_seed)
        )
        if self.config.selection_only:
            proposed = tree_proposed
            root_screening_scores = dict(q_values)
            root_screening_counts = dict(visit_counts)
            screening_wall_seconds = 0.0
            validation_wall_seconds = 0.0
            audit_wall_seconds = 0.0
            screening_simulator_transitions = 0
            validation_simulator_transitions = 0
            audit_simulator_transitions = 0
            accepted = False
            candidate_eligible = False
            validation_mean = 0.0
            validation_se = float("inf")
            validation_lower = float("-inf")
            validation_first = 0.0
            validation_horizon = 0.0
            validation_differences = ()
            validation_tree_actions_mean = 0.0
            validation_planned_source_counts = {}
            validation_policy_source_counts = {}
            validation_rollouts_used = 0
            audit = (None, None, None, None, None, None, {}, {})
            selected = policy_action
        else:
            screening_candidates = self._root_screening_candidates(
                root,
                policy_action,
                q_lower_bounds,
                q_values,
                visit_counts,
            )
            screening_started = time.perf_counter()
            screening_transition_start = self.rollout_simulator_transitions
            (
                proposed,
                root_screening_scores,
                root_screening_counts,
            ) = self._screen_root_actions(
                state,
                memory,
                screening_candidates,
                policy_action,
                validation_rng,
            )
            screening_wall_seconds = time.perf_counter() - screening_started
            screening_simulator_transitions = (
                self.rollout_simulator_transitions
                - screening_transition_start
            )
            if self.config.external_confirmation_only:
                validation_wall_seconds = 0.0
                audit_wall_seconds = 0.0
                validation_simulator_transitions = 0
                audit_simulator_transitions = 0
                accepted = False
                candidate_eligible = False
                validation_mean = 0.0
                validation_se = float("inf")
                validation_lower = float("-inf")
                validation_first = 0.0
                validation_horizon = 0.0
                validation_differences = ()
                validation_tree_actions_mean = 0.0
                validation_planned_source_counts = {}
                validation_policy_source_counts = {}
                validation_rollouts_used = 0
                audit = (None, None, None, None, None, None, {}, {})
                selected = policy_action
            else:
                validation_started = time.perf_counter()
                validation_transition_start = (
                    self.rollout_simulator_transitions
                )
                (
                    accepted,
                    validation_mean,
                    validation_se,
                    validation_lower,
                    validation_first,
                    validation_horizon,
                    validation_differences,
                    validation_tree_actions_mean,
                    validation_planned_source_counts,
                    validation_policy_source_counts,
                    validation_rollouts_used,
                ) = self._validate(
                    state, memory, proposed, policy_action, validation_rng
                )
                validation_wall_seconds = (
                    time.perf_counter() - validation_started
                )
                validation_simulator_transitions = (
                    self.rollout_simulator_transitions
                    - validation_transition_start
                )
                order_only_candidate = (
                    proposed != policy_action
                    and not self._selector._root_moves_are_strategic(state)
                    and (
                        ActionType(proposed.action_type) is ActionType.MOVE
                        or ActionType(policy_action.action_type)
                        is ActionType.MOVE
                    )
                )
                candidate_eligible = (
                    not order_only_candidate
                    and validation_mean > self.config.minimum_improvement
                )
                audit_started = time.perf_counter()
                audit_transition_start = self.rollout_simulator_transitions
                audit = (
                    self._audit(
                        state,
                        memory,
                        proposed,
                        policy_action,
                        validation_rng,
                    )
                    if candidate_eligible
                    else (None, None, None, None, None, None, {}, {})
                )
                audit_wall_seconds = time.perf_counter() - audit_started
                audit_simulator_transitions = (
                    self.rollout_simulator_transitions
                    - audit_transition_start
                )
                selected = proposed if accepted else policy_action
        full_counts = Counter(
            ActionType(action.action_type).name
            for action in self._root_full_actions
        )
        pruned_counts = Counter(
            ActionType(action.action_type).name
            for action in self._root_pruned_actions
        )
        removed_counts = {
            action_type: full_counts[action_type] - pruned_counts[action_type]
            for action_type in sorted(full_counts)
            if full_counts[action_type] > pruned_counts[action_type]
        }
        return PUCTSearchResult(
            tree_proposed_action=tree_proposed,
            proposed_action=proposed,
            selected_action=selected,
            policy_action=policy_action,
            accepted=accepted,
            candidate_eligible=candidate_eligible,
            exhaustive_root=exhaustive_root,
            full_candidate_count=len(self._root_full_actions),
            pruned_candidate_count=len(self._root_pruned_actions),
            tree_proposed_excluded_by_pruning=(
                tree_proposed not in self._root_pruned_actions
                if exhaustive_root
                else None
            ),
            proposed_excluded_by_pruning=(
                proposed not in self._root_pruned_actions
                if exhaustive_root
                else None
            ),
            pruned_action_counts=removed_counts,
            visit_counts=visit_counts,
            q_values=q_values,
            q_standard_errors=q_standard_errors,
            q_lower_bounds=q_lower_bounds,
            priors=dict(root.priors),
            root_screening_scores=root_screening_scores,
            root_screening_counts=root_screening_counts,
            simulations=simulations,
            simulator_transitions=self.simulator_transitions,
            tree_nodes=len(self.nodes),
            mean_tree_depth=(
                float(np.mean(self.simulation_depths))
                if self.simulation_depths
                else 0.0
            ),
            max_tree_depth=max(self.simulation_depths, default=0),
            validation_return_improvement=validation_mean,
            validation_return_standard_error=validation_se,
            validation_return_lower_bound=validation_lower,
            validation_first_battle_score_improvement=validation_first,
            validation_horizon_battle_score_improvement=validation_horizon,
            validation_differences=validation_differences,
            validation_rollouts_used=validation_rollouts_used,
            validation_tree_actions_mean=validation_tree_actions_mean,
            validation_planned_opponent_source_counts=(
                validation_planned_source_counts
            ),
            validation_policy_opponent_source_counts=(
                validation_policy_source_counts
            ),
            audit_return_improvement=audit[0],
            audit_return_standard_error=audit[1],
            audit_return_lower_bound=audit[2],
            audit_first_battle_score_improvement=audit[3],
            audit_horizon_battle_score_improvement=audit[4],
            audit_tree_actions_mean=audit[5],
            audit_planned_opponent_source_counts=audit[6],
            audit_policy_opponent_source_counts=audit[7],
            screening_simulator_transitions=(
                screening_simulator_transitions
            ),
            validation_simulator_transitions=(
                validation_simulator_transitions
            ),
            audit_simulator_transitions=audit_simulator_transitions,
            tree_wall_seconds=tree_wall_seconds,
            screening_wall_seconds=screening_wall_seconds,
            validation_wall_seconds=validation_wall_seconds,
            audit_wall_seconds=audit_wall_seconds,
            targets=self._training_targets(),
            omitted_end_mask_probability_mass=root.omitted_probability_mass,
        )
