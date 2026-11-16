"""Bounded, resumable public archive. No learner imports or training exports.

One process owns a collection directory. HTTP bodies and their receipts are
immutable; mutable progress is separate. A failed/partial download is never a
cache hit. Match identity, not perspective/replay ID, determines dataset split.
"""
from __future__ import annotations

from contextlib import contextmanager
from datetime import datetime, timezone
import fcntl
import hashlib
import json
import os
from pathlib import Path
import time
from urllib.error import HTTPError, URLError
from urllib.parse import urlencode
from urllib.request import Request, urlopen
from uuid import UUID, uuid4


SCHEMA = "sap-public-archive-v1"
BASE = "https://www.sap-library.com"


def utc():
    return datetime.now(timezone.utc).isoformat()


def sha_file(path):
    with Path(path).open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def atomic_json(path, data):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + f".{uuid4().hex}.tmp")
    try:
        with temporary.open("x") as handle:
            json.dump(data, handle, indent=2, allow_nan=False)
            handle.flush()
            os.fsync(handle.fileno())
        temporary.replace(path)
    finally:
        if temporary.exists():
            temporary.unlink()


def immutable_json(path, data):
    path = Path(path)
    if path.exists():
        if json.loads(path.read_text()) != data:
            raise ValueError(f"Immutable artifact differs: {path}")
        return
    atomic_json(path, data)


@contextmanager
def lock(path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a+") as handle:
        try:
            fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise RuntimeError(f"Another worker owns {path}") from exc
        try:
            yield
        finally:
            fcntl.flock(handle, fcntl.LOCK_UN)


def safe_id(value):
    return str(UUID(str(value)))


def split_for_match(match_id):
    bucket = int(hashlib.sha256(("sap-archive-split-v1:" + match_id).encode()).hexdigest()[:8], 16) % 10
    return "test" if bucket == 0 else "validation" if bucket == 1 else "train"


class PublicArchiveClient:
    def __init__(self, directory, interval=1.0, timeout=30.0):
        if interval < 1.0:
            raise ValueError("Public collector requires >=1 second between requests")
        self.directory = Path(directory)
        self.directory.mkdir(parents=True, exist_ok=True)
        self.interval, self.timeout = interval, timeout
        self.last = 0.0
        self.network_requests = 0

    def get(self, path, query=None):
        if not path.startswith("/api/") or ".." in path or "?" in path:
            raise ValueError("Only explicit SAP Library API paths are allowed")
        url = BASE + path + (("?" + urlencode(query)) if query else "")
        key = hashlib.sha256(url.encode()).hexdigest()
        body_path, receipt_path = self.directory / (key + ".json"), self.directory / (key + ".receipt.json")
        if receipt_path.exists():
            receipt = json.loads(receipt_path.read_text())
            if receipt["url"] != url or sha_file(body_path) != receipt["sha256"]:
                raise ValueError(f"Corrupt cache: {body_path}")
            return json.loads(body_path.read_text()), receipt
        # A body without a receipt is incomplete and is deliberately refetched.
        for attempt in range(3):
            time.sleep(max(0, self.interval - (time.monotonic() - self.last)))
            self.last = time.monotonic()
            self.network_requests += 1
            try:
                request = Request(url, headers={"User-Agent": "sapai-evidence-collector/1.0", "Accept": "application/json"})
                with urlopen(request, timeout=self.timeout) as response:
                    raw = response.read(32 * 1024 * 1024 + 1)
                    if len(raw) > 32 * 1024 * 1024:
                        raise ValueError("Response exceeds archive size bound")
                    value = json.loads(raw)
                    if not isinstance(value, dict):
                        raise ValueError("Expected API object")
                    receipt = {"schema": SCHEMA, "url": url, "status": response.status,
                               "fetchedAt": utc(), "bytes": len(raw),
                               "sha256": hashlib.sha256(raw).hexdigest()}
                temporary = body_path.with_suffix(".part")
                with temporary.open("wb") as handle:
                    handle.write(raw)
                    handle.flush()
                    os.fsync(handle.fileno())
                temporary.replace(body_path)
                atomic_json(receipt_path, receipt)
                return value, receipt
            except HTTPError as exc:
                if exc.code not in (429, 500, 502, 503, 504) or attempt == 2:
                    raise
                delay = exc.headers.get("Retry-After", "")
                if delay and (not delay.isdigit() or int(delay) > 60):
                    raise RuntimeError("Server requested a longer pause; worker stopped") from exc
                time.sleep(max(float(delay or 0), 2 ** (attempt + 1)))
            except (URLError, TimeoutError):
                if attempt == 2:
                    raise
                time.sleep(2 ** (attempt + 1))
        raise AssertionError("unreachable")


def collect(output, parser, *, max_matches=40, max_pages=20, version="48"):
    """Archive ranked 2-player Turtle mirrors in newest-first order, no W/L filter."""
    output = Path(output)
    config = {"schema": SCHEMA, "packA": "Turtle", "packB": "Turtle", "gameVersion": str(version),
              "matchType": "ranked", "maxPlayerCount": 2, "maxMatches": max_matches,
              "maxPages": max_pages, "pageSize": 50, "intervalSeconds": 1.0,
              "selection": "newest-first; no outcome or action selection", "trainingApproved": False}
    output.mkdir(parents=True, exist_ok=True)
    with lock(output / "collection.lock"):
        immutable_json(output / "CONFIG.json", config)
        client = PublicArchiveClient(output / "http")
        matches = {}
        errors, rejected, aliases = [], [], []
        report = {}
        def status(phase):
            report.update(schema=SCHEMA, phase=phase, updatedAt=utc(), matches=list(matches.values()),
                          matchCount=len(matches), rejected=rejected, aliases=aliases, errors=errors,
                          networkRequestsThisRun=client.network_requests, trainingApproved=False)
            atomic_json(output / "STATUS.json", report)
        try:
            for page in range(1, max_pages + 1):
                data, receipt = client.get("/api/search", {"packA": "Turtle", "packB": "Turtle", "version": "current", "matchType": "ranked",
                    "page": page, "pageSize": 50, "sort": "created_at", "order": "desc",
                    "uiView": "card", "uiEnabled": "matchup"})
                rows = data.get("results")
                if not isinstance(rows, list):
                    raise ValueError("Search response missing results list")
                for row in rows:
                    if len(matches) >= max_matches:
                        break
                    replay = safe_id(row["id"])
                    reasons = [key for key, value in {"pack": "Turtle", "opponent_pack": "Turtle", "match_type": "ranked", "max_player_count": 2}.items() if row.get(key) != value]
                    if str(row.get("game_version")) != str(version):
                        reasons.append("game_version")
                    if reasons:
                        rejected.append({"replayId": replay, "reasons": reasons})
                        continue
                    try:
                        perspectives, p_receipt = client.get(f"/api/replays/{replay}/perspectives")
                        turns, t_receipt = client.get(f"/api/replays/{replay}/turns")
                        match = safe_id(perspectives["matchId"])
                        if match in matches:
                            aliases.append({"replayId": replay, "matchId": match})
                            continue
                        bundle = {"replayId": replay, "metadata": row, "perspectives": perspectives, "turns": turns}
                        evidence = parser(bundle)
                        directory = output / "matches" / match
                        immutable_json(directory / "BUNDLE.json", bundle)
                        immutable_json(directory / "RECEIPTS.json", {"search": receipt, "perspectives": p_receipt, "turns": t_receipt})
                        immutable_json(directory / "EVIDENCE.json", evidence)
                        summary = {"matchId": match, "replayId": replay, "split": split_for_match(match),
                                   "players": [row.get("player_name"), row.get("opponent_name")],
                                   "directory": str(directory.relative_to(output)), "bundleSHA256": sha_file(directory / "BUNDLE.json"),
                                   "evidenceSHA256": sha_file(directory / "EVIDENCE.json"), "summary": evidence["summary"]}
                        immutable_json(directory / "MANIFEST.json", summary)
                        matches[match] = summary
                    except (ValueError, KeyError, TypeError) as exc:
                        errors.append({"replayId": replay, "error": str(exc)})
                    status("collecting")
                    print(json.dumps({"matches": len(matches), "errors": len(errors), "rejected": len(rejected)}), flush=True)
                if len(matches) >= max_matches or not rows or page * 50 >= data.get("total", 0):
                    break
            status("collection_complete" if len(matches) >= max_matches else "bounded_collection_exhausted")
        except BaseException as exc:
            errors.append({"fatal": type(exc).__name__, "error": str(exc)})
            status("failed_or_interrupted")
            raise
        return report
