"""Untouched interactive continuations for public-search evaluation."""

from __future__ import annotations

from contextlib import nullcontext
from dataclasses import dataclass

import numpy as np
import torch

from .action_schema import AutoregressiveAction
from .distributed import _completed_match_result
from .decision_api import decode_decisions
from .joint_action import JointAction, step_decisions


@dataclass(frozen=True)
class InteractiveContinuationResult:
    root_action: AutoregressiveAction | JointAction
    seed: int
    total_return: float
    battles: int
    terminated: bool
    truncated: bool
    match_score: float | None = None


class InteractiveSelfPlayEvaluator:
    """Evaluate public proposals in cloned, hidden, policy-driven matches.

    The planner never receives the full environment passed here. The evaluator
    is an outer holdout: it clones the already-collected two-player state,
    forces one public proposal, and lets both policies continue from their
    genuine recurrent states. Common seeds redeterminize only future RNG.
    """

    def __init__(
        self,
        policy,
        *,
        device="cpu",
        batch_size=128,
        horizon_battles=10,
        gamma=0.99,
        max_steps=1000,
    ):
        self.policy = policy
        self.device = torch.device(device)
        self.batch_size = int(batch_size)
        self.horizon_battles = int(horizon_battles)
        self.gamma = float(gamma)
        self.max_steps = int(max_steps)
        if self.batch_size <= 0:
            raise ValueError("batch_size must be positive")
        if self.horizon_battles <= 0:
            raise ValueError("horizon_battles must be positive")
        if not 0 <= self.gamma <= 1:
            raise ValueError("gamma must be in [0, 1]")
        if self.max_steps <= 0:
            raise ValueError("max_steps must be positive")

    def _autocast(self):
        return (
            torch.autocast("cuda", dtype=torch.bfloat16)
            if self.device.type == "cuda"
            else nullcontext()
        )

    @torch.inference_mode()
    def _choose_many(self, requests):
        results = []
        for start in range(0, len(requests), self.batch_size):
            chunk = requests[start : start + self.batch_size]
            observations = [request[2] for request in chunk]
            batch = {
                key: torch.as_tensor(
                    np.stack([observation[key] for observation in observations]),
                    device=self.device,
                )
                for key in observations[0]
            }
            memories = torch.stack(
                [torch.as_tensor(request[3]).float() for request in chunk]
            ).to(self.device)
            with self._autocast():
                if getattr(getattr(self.policy, 'config', None), 'end_freeze', False):
                    features = self.policy.encode_decision(batch, memories)
                    decisions, output = decode_decisions(self.policy, features, deterministic=True)
                else:
                    decisions = None
                    output = self.policy(batch, memory=memories, deterministic=True)
            for index in range(len(chunk)):
                results.append(
                    (
                        decisions[index] if decisions is not None else AutoregressiveAction(
                            *map(int, output.actions[index].cpu())
                        ),
                        output.memory[index].float().cpu(),
                    )
                )
        return results

    def _continuation_choice(self, observation, memory, side, action, next_memory):
        """Optional public-only continuation hook; never receives hidden state."""
        return action, next_memory

    def _execution_choice(self, record, acting_side, observation, memory, side, action, next_memory):
        """Truth-evaluator boundary; default preserves existing root-only behavior."""
        return self._continuation_choice(observation, memory, side, action, next_memory)

    def evaluate(self, environment, memories, acting_side, actions, seeds):
        if acting_side not in (0, 1):
            raise ValueError("acting_side must be zero or one")
        actions = tuple(actions)
        seeds = tuple(map(int, seeds))
        if not actions or not seeds:
            raise ValueError("actions and seeds must be non-empty")
        if getattr(getattr(self.policy, 'config', None), 'end_freeze', False) and any(
            not isinstance(action, JointAction) or action.end_freeze_mask is None for action in actions
        ):
            raise ValueError('Joint-policy interventions require explicit freeze masks, including zero')
        if environment.core.is_done() or not environment.active_players()[acting_side]:
            raise ValueError("The acting side must have a current nonterminal shop decision")
        if len(memories) != 2:
            raise ValueError("Both players' recurrent states are required")
        records = []
        for action in actions:
            for seed in seeds:
                cloned = environment.clone()
                cloned.redeterminize(seed)
                records.append(
                    {
                        "root_action": action,
                        "seed": seed,
                        "environment": cloned,
                        "memories": [
                            torch.as_tensor(memory).float().cpu().clone()
                            for memory in memories
                        ],
                        "root_forced": False,
                        "return": 0.0,
                        "discount": 1.0,
                        "battles": 0,
                        "steps": 0,
                        "terminated": False,
                    }
                )

        while True:
            active_records = [
                record
                for record in records
                if not record["terminated"]
                and record["battles"] < self.horizon_battles
                and record["steps"] < self.max_steps
            ]
            if not active_records:
                break
            requests = []
            for record_index, record in enumerate(active_records):
                environment_state = record["environment"]
                observations = environment_state.observations()
                for side, active in enumerate(
                    environment_state.active_players()
                ):
                    if active:
                        requests.append(
                            (
                                record_index,
                                side,
                                observations[side],
                                record["memories"][side],
                            )
                        )
            if not requests:
                break
            choices = self._choose_many(requests)
            per_record_actions = [[None, None] for _ in active_records]
            for request, (policy_action, next_memory) in zip(
                requests, choices
            ):
                record_index, side, observation, previous_memory = request
                record = active_records[record_index]
                action = policy_action
                if side == acting_side and not record["root_forced"]:
                    action = record["root_action"]
                    record["root_forced"] = True
                else:
                    action, next_memory = self._execution_choice(
                        record, acting_side, observation, previous_memory, side, action, next_memory
                    )
                per_record_actions[record_index][side] = action
                record["memories"][side] = next_memory
            for record, action_pair in zip(
                active_records, per_record_actions
            ):
                if any(isinstance(action, JointAction) for action in action_pair):
                    decisions = tuple(action if action is None or isinstance(action, JointAction)
                                      else JointAction(action) for action in action_pair)
                    transition = step_decisions(record['environment'], decisions).result
                else:
                    transition = record["environment"].step(*action_pair)
                record["steps"] += 1
                if transition.battle_resolved:
                    record["return"] += (
                        record["discount"]
                        * float(transition.rewards[acting_side])
                    )
                    record["discount"] *= self.gamma
                    record["battles"] += 1
                record["terminated"] = bool(transition.terminated)

        return tuple(
            InteractiveContinuationResult(
                root_action=record["root_action"],
                seed=record["seed"],
                total_return=float(record["return"]),
                battles=int(record["battles"]),
                terminated=bool(record["terminated"]),
                truncated=bool(
                    not record["terminated"]
                    and (
                        record["battles"] >= self.horizon_battles
                        or record["steps"] >= self.max_steps
                    )
                ),
                match_score=(
                    self._match_score(record["environment"].core, acting_side)
                    if record["terminated"] else None
                ),
            )
            for record in records
        )

    @staticmethod
    def _match_score(core, acting_side):
        result = _completed_match_result(core)
        return 0.5 if result == 2 else float(result == acting_side)
