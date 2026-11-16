"""Public-information neural belief over the opponent's next submitted board."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from sapai import Pet, Team


OPPONENT_BELIEF_VERSION = "neural-board-mixture-v1-20260904"


def _log_softmax(values):
    values = np.asarray(values, dtype=np.float64)
    shifted = values - values.max(axis=-1, keepdims=True)
    return shifted - np.log(np.exp(shifted).sum(axis=-1, keepdims=True))


@dataclass(frozen=True)
class OpponentBeliefMix:
    synthetic_weight: float = 0.50
    persistence_weight: float = 0.25
    reservoir_weight: float = 0.25
    reservoir_temperature: float = 0.25
    reservoir_uniform_mix: float = 0.10

    def __post_init__(self):
        weights = (
            self.synthetic_weight,
            self.persistence_weight,
            self.reservoir_weight,
        )
        if any(weight < 0 for weight in weights) or sum(weights) <= 0:
            raise ValueError("Opponent belief mixture weights are invalid")
        if self.reservoir_temperature <= 0:
            raise ValueError("reservoir_temperature must be positive")
        if not 0 <= self.reservoir_uniform_mix <= 1:
            raise ValueError("reservoir_uniform_mix must be in [0, 1]")

    @property
    def normalized_weights(self):
        weights = np.asarray(
            [
                self.synthetic_weight,
                self.persistence_weight,
                self.reservoir_weight,
            ],
            dtype=np.float64,
        )
        return weights / weights.sum()


@dataclass(frozen=True)
class OpponentBoardBelief:
    pet_log_prob: np.ndarray
    status_log_prob: np.ndarray
    level_log_prob: np.ndarray
    stats: np.ndarray
    temporary_probability: np.ndarray
    item_to_id: dict[str, int]
    status_to_id: dict[str, int]

    def __post_init__(self):
        expected_slots = 5
        if self.pet_log_prob.ndim != 2 or self.pet_log_prob.shape[0] != expected_slots:
            raise ValueError("Opponent pet belief must have five slots")
        if self.status_log_prob.shape[0] != expected_slots:
            raise ValueError("Opponent status belief must have five slots")
        if self.level_log_prob.shape != (expected_slots, 4):
            raise ValueError("Opponent level belief must have shape (5, 4)")
        if self.stats.shape != (expected_slots, 2):
            raise ValueError("Opponent stat belief must have shape (5, 2)")
        if self.temporary_probability.shape != (expected_slots, 2):
            raise ValueError("Opponent temporary belief must have shape (5, 2)")

    @classmethod
    def from_logits(
        cls,
        pet_logits,
        status_logits,
        level_logits,
        stats,
        temporary_logits,
        encoder,
    ):
        return cls(
            pet_log_prob=_log_softmax(pet_logits),
            status_log_prob=_log_softmax(status_logits),
            level_log_prob=_log_softmax(level_logits),
            stats=np.asarray(stats, dtype=np.float64).copy(),
            temporary_probability=(
                1.0
                / (1.0 + np.exp(-np.asarray(temporary_logits, dtype=np.float64)))
            ),
            item_to_id=dict(encoder.item_to_id),
            status_to_id=dict(encoder.status_to_id),
        )

    @property
    def id_to_item(self):
        return {value: key for key, value in self.item_to_id.items()}

    @property
    def id_to_status(self):
        return {value: key for key, value in self.status_to_id.items()}

    def _modal_pet_id(self, position):
        allowed = [0]
        allowed.extend(
            item_id
            for name, item_id in self.item_to_id.items()
            if name.startswith("pet-") and name != "pet-none"
        )
        return max(allowed, key=lambda item_id: self.pet_log_prob[position, item_id])

    def modal_team(self):
        team = Team()
        item_names = self.id_to_item
        status_names = self.id_to_status
        for position in range(5):
            item_id = self._modal_pet_id(position)
            if item_id == 0:
                continue
            pet = Pet(item_names[item_id])
            pet._attack = int(
                np.clip(
                    max(pet._attack, round(50 * self.stats[position, 0])),
                    1,
                    50,
                )
            )
            pet._health = int(
                np.clip(
                    max(pet._health, round(50 * self.stats[position, 1])),
                    1,
                    50,
                )
            )
            pet.level = int(
                np.clip(np.argmax(self.level_log_prob[position]), 1, 3)
            )
            pet.experience = 0
            status_id = int(np.argmax(self.status_log_prob[position]))
            status_name = status_names.get(status_id, "none")
            pet.status = (
                "none"
                if status_name in {"none", "status-none"}
                else status_name
            )
            team[position] = pet
        return team

    def team_score(self, team):
        if not isinstance(team, Team):
            team = Team.from_state(team)
        pet_scores = []
        status_scores = []
        level_scores = []
        stat_errors = []
        for position in range(5):
            slot = team[position]
            if slot.empty:
                item_id = 0
            else:
                item_id = self.item_to_id.get(slot.pet.name, 0)
            pet_scores.append(self.pet_log_prob[position, item_id])
            if slot.empty:
                continue
            pet = slot.pet
            status_id = self.status_to_id.get(pet.status, 0)
            status_scores.append(
                self.status_log_prob[position, status_id]
            )
            level_scores.append(
                self.level_log_prob[position, int(np.clip(pet.level, 0, 3))]
            )
            expected_stats = np.asarray(
                [pet.attack / 50, pet.health / 50], dtype=np.float64
            )
            stat_errors.extend(
                np.square(self.stats[position] - expected_stats)
            )
        score = float(np.mean(pet_scores))
        if status_scores:
            score += 0.25 * float(np.mean(status_scores))
            score += 0.25 * float(np.mean(level_scores))
            score -= 0.5 * float(np.mean(stat_errors))
        return score

    def reservoir_probabilities(
        self,
        teams,
        *,
        temperature=0.25,
        uniform_mix=0.10,
    ):
        if not teams:
            raise ValueError("Cannot score an empty opponent candidate set")
        if temperature <= 0:
            raise ValueError("temperature must be positive")
        if not 0 <= uniform_mix <= 1:
            raise ValueError("uniform_mix must be in [0, 1]")
        scores = np.asarray(
            [self.team_score(team) for team in teams], dtype=np.float64
        )
        scores = (scores - scores.max()) / float(temperature)
        probabilities = np.exp(scores)
        probabilities /= probabilities.sum()
        probabilities = (
            (1.0 - uniform_mix) * probabilities
            + uniform_mix / len(probabilities)
        )
        return probabilities
