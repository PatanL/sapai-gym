"""Public-board matching for coherent future opponent trajectories."""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path

from sapai import Team


RESERVOIR_SCHEMA_VERSION = 2
OLDER_BOARD_DISTANCE_WEIGHT = 0.5


def _team(team_or_state):
    if isinstance(team_or_state, Team):
        return team_or_state
    return Team.from_state(team_or_state)


def team_signature(team):
    team = _team(team)
    return tuple(
        None
        if team[position].empty
        else (
            team[position].pet.name,
            int(team[position].pet.attack),
            int(team[position].pet.health),
            int(team[position].pet.level),
            str(team[position].pet.status),
        )
        for position in range(5)
    )


def team_distance(observed: Team, candidate: Team) -> float:
    """Position-sensitive distance using only publicly visible attributes."""

    distance = 0.0
    for position in range(5):
        left = observed[position]
        right = candidate[position]
        if left.empty and right.empty:
            continue
        if left.empty != right.empty:
            distance += 4.0
            continue
        left_pet = left.pet
        right_pet = right.pet
        distance += 3.0 * (left_pet.name != right_pet.name)
        distance += 0.5 * abs(left_pet.level - right_pet.level)
        distance += 0.5 * (left_pet.status != right_pet.status)
        distance += abs(left_pet.attack - right_pet.attack) / 25.0
        distance += abs(left_pet.health - right_pet.health) / 25.0
    return float(distance)


@dataclass(frozen=True)
class MatchedOpponentTrajectory:
    trajectory_id: str
    distance: float
    teams: tuple[Team, ...]
    source_index: int | str | None = None


class OpponentTrajectoryReservoir:
    def __init__(self, trajectories, metadata=None):
        self.metadata = dict(metadata or {})
        components = self.metadata.get("components") or []
        self._source_weights = {
            component.get("sourceIndex"): float(component.get("games") or 0)
            for component in components
        }
        self.trajectories = []
        for trajectory in trajectories:
            boards = {
                int(board["turn"]): {
                    "team": _team(board["team"]),
                    "opponentTeam": (
                        None
                        if board.get("opponentTeam") is None
                        else _team(board["opponentTeam"])
                    ),
                }
                for board in trajectory.get("boards", [])
            }
            if boards:
                self.trajectories.append(
                    {
                        "id": str(trajectory.get("id", len(self.trajectories))),
                        "sourceIndex": trajectory.get("sourceIndex"),
                        "boards": boards,
                    }
                )
        if not self.trajectories:
            raise ValueError("Opponent reservoir has no non-empty trajectories")
        self._by_anchor_turn = {}
        for trajectory in self.trajectories:
            for turn in trajectory["boards"]:
                self._by_anchor_turn.setdefault(turn, []).append(trajectory)
        self._match_cache = {}

    def _source_quotas(self, sources, limit):
        """Allocate candidate slots by configured mixture with a diversity floor."""

        sources = sorted(sources, key=lambda source: str(source))
        if len(sources) >= limit:
            return {source: 1 for source in sources[:limit]}
        weights = {
            source: max(0.0, self._source_weights.get(source, 1.0))
            for source in sources
        }
        total = sum(weights.values())
        if total <= 0:
            weights = {source: 1.0 for source in sources}
            total = float(len(sources))
        targets = {
            source: limit * weights[source] / total for source in sources
        }
        quotas = {source: 1 for source in sources}
        for _ in range(limit - len(sources)):
            source = max(
                sources,
                key=lambda candidate: (
                    targets[candidate] - quotas[candidate],
                    weights[candidate],
                    str(candidate),
                ),
            )
            quotas[source] += 1
        return quotas

    @classmethod
    def load(cls, path):
        payload = json.loads(Path(path).read_text())
        if payload.get("schemaVersion") != RESERVOIR_SCHEMA_VERSION:
            raise ValueError("Unsupported opponent reservoir schema")
        return cls(payload["trajectories"], metadata=payload.get("metadata"))

    def opening_prior(self, *, seed, horizon=5, limit=32):
        """Sample source-balanced opening futures without a live-opponent query.

        Require recorded boards throughout the requested horizon. No empty
        fallback, future-board substitution or live hidden state is accepted.
        For ordinary six-heart matches, five opening rounds precede elimination.
        """
        import numpy as np

        if horizon <= 0 or limit <= 0:
            raise ValueError("horizon and limit must be positive")
        if horizon > 6:
            raise ValueError("Opening priors cannot condition on survival beyond the first six rounds")
        by_source = {}
        for trajectory in self._by_anchor_turn.get(1, ()):
            boards = trajectory['boards']
            if not all(turn in boards for turn in range(1, horizon + 1)):
                continue
            by_source.setdefault(trajectory['sourceIndex'], []).append(trajectory)
        if not by_source:
            raise ValueError("No complete opening trajectories for the requested horizon")
        rng = np.random.default_rng(seed)
        shuffled = {
            source: [entries[index] for index in rng.permutation(len(entries))]
            for source, entries in sorted(by_source.items(), key=lambda entry: str(entry[0]))
        }
        quotas = self._source_quotas(shuffled, limit)
        selected = [trajectory for source, quota in quotas.items() for trajectory in shuffled[source][:quota]]
        selected_ids = {trajectory['id'] for trajectory in selected}
        remaining = [trajectory for entries in shuffled.values() for trajectory in entries if trajectory['id'] not in selected_ids]
        for index in rng.permutation(len(remaining)):
            if len(selected) >= limit:
                break
            selected.append(remaining[index])
        return [MatchedOpponentTrajectory(
            trajectory_id=trajectory['id'], distance=0.,
            teams=tuple(trajectory['boards'][turn]['team'] for turn in range(1, horizon + 1)),
            source_index=trajectory['sourceIndex'],
        ) for trajectory in selected]

    def match(
        self,
        observed,
        *,
        current_turn,
        horizon=5,
        limit=32,
        observed_enemy=None,
        older_observed=None,
        older_observed_enemy=None,
    ):
        if horizon <= 0 or limit <= 0:
            raise ValueError("horizon and limit must be positive")
        observed = _team(observed)
        observed_enemy = (
            None if observed_enemy is None else _team(observed_enemy)
        )
        older_observed = (
            None if older_observed is None else _team(older_observed)
        )
        older_observed_enemy = (
            None
            if older_observed_enemy is None
            else _team(older_observed_enemy)
        )
        anchor_turn = int(current_turn) - 1
        cache_key = (
            anchor_turn,
            int(horizon),
            int(limit),
            team_signature(observed),
            None
            if observed_enemy is None
            else team_signature(observed_enemy),
            None
            if older_observed is None
            else team_signature(older_observed),
            None
            if older_observed_enemy is None
            else team_signature(older_observed_enemy),
        )
        cached = self._match_cache.get(cache_key)
        if cached is not None:
            return cached
        complete = []
        partial = []
        for trajectory in self._by_anchor_turn.get(anchor_turn, ()):
            boards = trajectory["boards"]
            if anchor_turn not in boards or int(current_turn) not in boards:
                continue
            available_turns = sorted(
                turn for turn in boards if turn >= int(current_turn)
            )
            if not available_turns:
                continue
            future = []
            last_state = None
            is_complete = True
            for turn in range(int(current_turn), int(current_turn) + horizon):
                if turn in boards:
                    last_state = boards[turn]["team"]
                else:
                    is_complete = False
                if last_state is None:
                    break
                future.append(last_state)
            if not future:
                continue
            while len(future) < horizon:
                future.append(future[-1])
            distance = team_distance(
                observed, boards[anchor_turn]["team"]
            )
            anchor_enemy = boards[anchor_turn].get("opponentTeam")
            if observed_enemy is not None and anchor_enemy is not None:
                distance += team_distance(
                    observed_enemy, anchor_enemy
                )
            older_anchor = boards.get(anchor_turn - 1)
            if older_anchor is not None:
                if older_observed is not None:
                    distance += OLDER_BOARD_DISTANCE_WEIGHT * team_distance(
                        older_observed, older_anchor["team"]
                    )
                older_anchor_enemy = older_anchor.get("opponentTeam")
                if (
                    older_observed_enemy is not None
                    and older_anchor_enemy is not None
                ):
                    distance += OLDER_BOARD_DISTANCE_WEIGHT * team_distance(
                        older_observed_enemy, older_anchor_enemy
                    )
            match = MatchedOpponentTrajectory(
                trajectory_id=trajectory["id"],
                distance=distance,
                teams=tuple(future),
                source_index=trajectory.get("sourceIndex"),
            )
            (complete if is_complete else partial).append(match)
        candidates = complete if complete else partial
        candidates.sort(key=lambda match: (match.distance, match.trajectory_id))
        # A merged reservoir may intentionally contain several frozen-policy
        # sources. Guarantee one nearest trajectory from each source before
        # filling by global distance; otherwise the largest component can
        # monopolize a small top-k despite the intended strategic mixture.
        by_source = {}
        for match in candidates:
            by_source.setdefault(match.source_index, []).append(match)
        result = []
        if len(by_source) > 1:
            quotas = self._source_quotas(by_source, limit)
            for source, quota in quotas.items():
                result.extend(by_source[source][:quota])
        selected_ids = {match.trajectory_id for match in result}
        for match in candidates:
            if len(result) >= limit:
                break
            if match.trajectory_id not in selected_ids:
                result.append(match)
                selected_ids.add(match.trajectory_id)
        result.sort(key=lambda match: (match.distance, match.trajectory_id))
        if len(self._match_cache) >= 10000:
            self._match_cache.clear()
        self._match_cache[cache_key] = result
        return result
