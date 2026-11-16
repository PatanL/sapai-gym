"""Small, serializable self-play league with PFSP-style sampling."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import json
from pathlib import Path

import numpy as np


@dataclass
class LeagueMember:
    member_id: str
    checkpoint: str
    role: str = "historical"
    created_step: int = 0


class LeagueManager:
    """Track frozen policies and prioritize opponents exposing weaknesses."""

    def __init__(self):
        self.members: dict[str, LeagueMember] = {}
        self.results: dict[tuple[str, str], list[int]] = {}

    def add(self, member: LeagueMember):
        if member.member_id in self.members:
            raise ValueError(f"Duplicate league member {member.member_id}")
        self.members[member.member_id] = member

    def promote(self, member_id: str, checkpoint: str, created_step: int = 0):
        """Make a validated checkpoint a permanently resident champion."""

        if member_id in self.members:
            member = self.members[member_id]
            member.checkpoint = str(checkpoint)
            member.role = "champion"
            member.created_step = int(created_step)
            return member
        member = LeagueMember(
            member_id,
            str(checkpoint),
            role="champion",
            created_step=int(created_step),
        )
        self.add(member)
        return member

    def record_result(self, player_id: str, opponent_id: str, outcome: int):
        """Record outcome from player perspective: win=1, draw=0, loss=-1."""

        if outcome not in (-1, 0, 1):
            raise ValueError("outcome must be -1, 0, or 1")
        self.results.setdefault((player_id, opponent_id), []).append(outcome)

    def win_rate(self, player_id: str, opponent_id: str) -> float:
        outcomes = self.results.get((player_id, opponent_id), [])
        if not outcomes:
            return 0.5
        scores = [(outcome + 1) / 2 for outcome in outcomes]
        return float(np.mean(scores))

    def sample_opponent(
        self,
        player_id: str,
        rng: np.random.Generator,
        roles: set[str] | None = None,
    ) -> LeagueMember:
        candidates = [
            member
            for member in self.members.values()
            if member.member_id != player_id
            and (roles is None or member.role in roles)
        ]
        if not candidates:
            raise LookupError("No eligible league opponents")
        # Prioritized fictitious self-play: opponents with a low player win
        # rate are sampled more often, while every policy retains support.
        priorities = np.array(
            [
                max(0.05, 1.0 - self.win_rate(player_id, member.member_id)) ** 2
                for member in candidates
            ],
            dtype=np.float64,
        )
        priorities /= priorities.sum()
        return candidates[int(rng.choice(len(candidates), p=priorities))]

    def resident_members(self, max_historical: int = 4) -> list[LeagueMember]:
        """Keep every fixed exploiter/champion and only cap rolling history.

        Exploiters are permanent best-response anchors and champions are
        checkpoints that passed an external promotion gate. Historical
        snapshots are the only resident role that should age out as the main
        policy creates newer checkpoints.
        """

        if max_historical < 0:
            raise ValueError("max_historical must be non-negative")
        permanent = sorted(
            (
                member
                for member in self.members.values()
                if member.role in {"exploiter", "champion"}
            ),
            key=lambda member: (member.created_step, member.member_id),
        )
        historical = sorted(
            (
                member
                for member in self.members.values()
                if member.role == "historical"
            ),
            key=lambda member: (member.created_step, member.member_id),
        )
        historical = historical[-max_historical:] if max_historical else []
        return permanent + historical

    def save(self, path: str | Path):
        payload = {
            "members": [asdict(member) for member in self.members.values()],
            "results": {
                f"{player}|{opponent}": outcomes
                for (player, opponent), outcomes in self.results.items()
            },
        }
        path = Path(path)
        temporary = path.with_suffix(path.suffix + ".tmp")
        temporary.write_text(json.dumps(payload, indent=2, sort_keys=True))
        temporary.replace(path)

    @classmethod
    def load(cls, path: str | Path) -> "LeagueManager":
        payload = json.loads(Path(path).read_text())
        league = cls()
        for member in payload["members"]:
            league.add(LeagueMember(**member))
        for key, outcomes in payload["results"].items():
            player, opponent = key.split("|", 1)
            league.results[(player, opponent)] = list(outcomes)
        return league
