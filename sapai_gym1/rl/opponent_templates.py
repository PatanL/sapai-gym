"""Immutable opponent futures, materialized only when a rollout uses a board."""
from collections.abc import Sequence
from dataclasses import dataclass
import json

import numpy as np

from sapai import Team


def _json_scalar(value):
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    raise TypeError(f'Unsupported opponent-state value: {type(value).__name__}')


@dataclass(frozen=True)
class FrozenOpponentTrajectory(Sequence):
    """Strings and tuples are immutable; each accessed Team is an independent copy."""
    boards: tuple[str, ...]

    @classmethod
    def freeze(cls, teams):
        if isinstance(teams, cls):
            return teams
        # Preserve the old ingestion normalization once, not on every clone.
        return cls(tuple(json.dumps(Team.from_state(team.state).state,
                                    separators=(',', ':'), allow_nan=False,
                                    default=_json_scalar) for team in teams))

    def __len__(self):
        return len(self.boards)

    def __getitem__(self, index):
        if isinstance(index, slice):
            return tuple(Team.from_state(json.loads(board)) for board in self.boards[index])
        return Team.from_state(json.loads(self.boards[index]))

    def __deepcopy__(self, memo):
        return self


@dataclass(frozen=True)
class FrozenOpponentTrajectories(Sequence):
    trajectories: tuple[FrozenOpponentTrajectory, ...]

    @classmethod
    def freeze(cls, trajectories):
        if isinstance(trajectories, cls):
            return trajectories
        frozen = tuple(FrozenOpponentTrajectory.freeze(trajectory) for trajectory in trajectories)
        frozen = tuple(trajectory for trajectory in frozen if len(trajectory))
        if not frozen:
            raise ValueError('At least one opponent trajectory is required')
        return cls(frozen)

    def __len__(self):
        return len(self.trajectories)

    def __getitem__(self, index):
        return self.trajectories[index]

    def __deepcopy__(self, memo):
        return self
