"""Rate-limited, cached client for SAP Library's public replay endpoints."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
import time
from urllib.parse import quote, urlencode
from urllib.request import Request, urlopen


@dataclass(frozen=True)
class ReplayCode:
    participation_id: str
    turn: int = 1

    @classmethod
    def parse(cls, value: str | dict) -> "ReplayCode":
        if isinstance(value, dict):
            payload = value
        else:
            value = value.strip()
            payload = json.loads(value) if value.startswith("{") else {"Pid": value}
        participation_id = payload.get("Pid") or payload.get("participationId")
        if not participation_id or not isinstance(participation_id, str):
            raise ValueError("Replay code requires a string Pid")
        return cls(participation_id, int(payload.get("T", 1)))


class SapLibraryClient:
    BASE_URL = "https://sap-library.vercel.app/api"

    def __init__(
        self,
        cache_dir: str | Path,
        min_request_interval: float = 0.25,
        timeout: float = 30.0,
    ):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.min_request_interval = min_request_interval
        self.timeout = timeout
        self._last_request = 0.0

    def lookup(self, replay_code: ReplayCode | str | dict) -> dict:
        replay_code = (
            replay_code
            if isinstance(replay_code, ReplayCode)
            else ReplayCode.parse(replay_code)
        )
        response = self._get_json(
            "/search", {"pid": replay_code.participation_id}
        )
        results = response.get("results") or []
        exact = [
            result
            for result in results
            if result.get("participation_id") == replay_code.participation_id
        ]
        if len(exact) != 1:
            raise LookupError(
                f"Expected one replay for {replay_code.participation_id}, "
                f"found {len(exact)}"
            )
        return exact[0]

    def resolve_replay_id(self, replay_code: ReplayCode | str | dict) -> str:
        return str(self.lookup(replay_code)["id"])

    def fetch_bundle(self, replay_code: ReplayCode | str | dict) -> dict:
        replay_code = (
            replay_code
            if isinstance(replay_code, ReplayCode)
            else ReplayCode.parse(replay_code)
        )
        metadata = self.lookup(replay_code)
        replay_id = str(metadata["id"])
        return {
            "replayCode": {
                "Pid": replay_code.participation_id,
                "T": replay_code.turn,
            },
            "replayId": replay_id,
            "metadata": metadata,
            "perspectives": self._get_json(
                f"/replays/{quote(replay_id)}/perspectives"
            ),
            "turns": self._get_json(f"/replays/{quote(replay_id)}/turns"),
        }

    def search_player(
        self,
        player_id: str,
        *,
        page: int = 1,
        page_size: int = 10,
        match_type: str = "private",
        version: str = "current",
        sort: str = "created_at",
        order: str = "desc",
    ) -> dict:
        """Return one cached SAP Library player-search page."""

        if not player_id:
            raise ValueError("player_id must be non-empty")
        return self._get_json(
            "/search",
            {
                "playerId": player_id,
                "matchType": match_type,
                "version": version,
                "page": max(1, int(page)),
                "pageSize": max(1, min(100, int(page_size))),
                "sort": sort,
                "order": order,
            },
        )

    def fetch_metadata_bundle(self, metadata: dict) -> dict:
        """Fetch a search result without performing a second PID lookup."""

        replay_id = str(metadata["id"])
        participation_id = str(metadata["participation_id"])
        return {
            "replayCode": {"Pid": participation_id, "T": 1},
            "replayId": replay_id,
            "metadata": metadata,
            "perspectives": self._get_json(
                f"/replays/{quote(replay_id)}/perspectives"
            ),
            "turns": self._get_json(f"/replays/{quote(replay_id)}/turns"),
        }

    def _get_json(self, path: str, query: dict | None = None) -> dict:
        url = f"{self.BASE_URL}{path}"
        if query:
            url = f"{url}?{urlencode(query)}"
        cache_key = hashlib.sha256(url.encode()).hexdigest()
        cache_path = self.cache_dir / f"{cache_key}.json"
        if cache_path.exists():
            return json.loads(cache_path.read_text())

        delay = self.min_request_interval - (time.monotonic() - self._last_request)
        if delay > 0:
            time.sleep(delay)
        request = Request(
            url,
            headers={"User-Agent": "sapai-rl-replay-ingestor/0.1"},
        )
        with urlopen(request, timeout=self.timeout) as response:
            payload = json.loads(response.read().decode("utf-8"))
        self._last_request = time.monotonic()
        temporary = cache_path.with_suffix(".tmp")
        temporary.write_text(json.dumps(payload, separators=(",", ":")))
        temporary.replace(cache_path)
        return payload
