"""No network, browser, service mutations or GPU required."""
from copy import deepcopy
import io
import json
from pathlib import Path
import subprocess
import sys
from unittest.mock import patch

import pytest

from sapai_gym1.replay.archive import PublicArchiveClient, collect, immutable_json, lock, safe_id, split_for_match
from sapai_gym1.replay.evidence import EvidenceParser

ROOT = Path(__file__).resolve().parents[2]
FIXTURE = ROOT / "runs/expert-replay-reconstruction-20260925-v1"


@pytest.fixture
def bundle():
    selection = json.loads((FIXTURE / "SELECTION.json").read_text())
    return {"replayId": selection["metadata"]["id"], "metadata": selection["metadata"],
            "perspectives": json.loads((FIXTURE / "perspectives.json").read_text()),
            "turns": json.loads((FIXTURE / "turns.json").read_text())}


@pytest.fixture
def parser():
    return EvidenceParser(ROOT / "SAP-Calculator")


def test_native_fixture_counts_and_no_fake_states(bundle, parser):
    result = parser(bundle)
    assert result["summary"]["transactions"] == 169
    assert result["summary"]["hashMismatches"] == 0
    assert result["summary"]["nativeEndpoints"] == 22
    assert result["summary"]["completeBeforeAfterStates"] == 0
    assert result["trainingApproved"] is False
    assert result["policyObservationsExported"] is False
    assert result["perspectives"][0]["expectedShopTurns"] == list(range(1, 12))
    assert result["perspectives"][0]["nativeLastTurnMetadata"] == 12
    assert all(a["beforeState"] is None and a["afterState"] is None
               for p in result["perspectives"] for a in p["actions"])


def test_zero_enum_and_free_food_and_nested_spell(bundle, parser):
    actions = parser(bundle)["perspectives"][0]["actions"]
    foods = [a["consumedFood"] for a in actions if a["type"] == 8]
    assert len(foods) == 13 and all(foods)
    assert any(f["name"] == "Apple" and f["enum"] == 0 for f in foods)
    assert any(f["name"] == "Bread Crumbs" and f["price"] == 0 for f in foods)
    assert any(f["name"] == "Sleeping Pill" for f in foods)


def test_hash_corruption_detected_not_repaired(bundle, parser):
    raw = bundle["perspectives"]["perspectives"][0]["raw"]
    request = json.loads(raw["Actions"][1]["Request"])
    request["Data"]["BoardHash"] += 1
    raw["Actions"][1]["Request"] = json.dumps(request)
    assert parser(bundle)["summary"]["hashMismatches"] == 1


@pytest.mark.parametrize("field", ["version", "replay", "match", "pid", "perspectives"])
def test_schema_fail_closed(bundle, parser, field):
    if field == "version": bundle["metadata"]["game_version"] = "49"
    elif field == "replay": bundle["replayId"] = "wrong"
    elif field == "match": bundle["perspectives"]["matchId"] = "wrong"
    elif field == "pid": bundle["perspectives"]["perspectives"][0]["participationId"] = "wrong"
    elif field == "perspectives": bundle["perspectives"]["perspectives"].pop()
    with pytest.raises(ValueError): parser(bundle)


def test_split_by_match_not_seat(bundle, parser):
    parsed = parser(bundle)
    assert parsed["split"] == split_for_match(bundle["perspectives"]["matchId"])
    assert len(parsed["perspectives"]) == 2
    assert all("split" not in p for p in parsed["perspectives"])


def test_future_identity_is_annotation_not_features(bundle, parser):
    result = parser(bundle)
    future = [a for p in result["perspectives"] for a in p["actions"] if a.get("identityUsesFutureEvidence")]
    assert future
    assert all(a["beforeState"] is None for a in future)


def test_inputs_not_mutated(bundle, parser):
    original = deepcopy(bundle)
    parser(bundle)
    assert bundle == original


def test_immutable_and_lock(tmp_path):
    path = tmp_path / "file.json"
    immutable_json(path, {"a": 1})
    immutable_json(path, {"a": 1})
    with pytest.raises(ValueError): immutable_json(path, {"a": 2})
    with lock(tmp_path / "lock"):
        with pytest.raises(RuntimeError):
            with lock(tmp_path / "lock"): pass
    with lock(tmp_path / "lock"): pass


def test_safe_paths_and_rate_limit(tmp_path):
    with pytest.raises(ValueError): safe_id("../../outside")
    with pytest.raises(ValueError): PublicArchiveClient(tmp_path, interval=0)
    with pytest.raises(ValueError): PublicArchiveClient(tmp_path).get("https://evil.example/")


class Response(io.BytesIO):
    status = 200


def test_cache_resume_and_corruption(tmp_path):
    with patch("sapai_gym1.replay.archive.urlopen", return_value=Response(b'{"a":1}')) as network:
        client = PublicArchiveClient(tmp_path)
        a, receipt = client.get("/api/meta/options")
        assert a == {"a": 1} and network.call_count == 1
        assert client.get("/api/meta/options")[1] == receipt
        assert network.call_count == 1
    raw = next(p for p in tmp_path.glob("*.json") if not p.name.endswith("receipt.json"))
    raw.write_text('{}')
    with pytest.raises(ValueError): client.get("/api/meta/options")


def test_archive_resume_and_match_dedupe(tmp_path, bundle, parser):
    other = deepcopy(bundle["metadata"])
    other["id"] = "aaaaaaaa-aaaa-4aaa-aaaa-aaaaaaaaaaaa"
    def fetch(self, path, query=None):
        if path == "/api/search":
            data = {"results": [bundle["metadata"], other], "total": 2}
        elif path.endswith("perspectives"):
            data = deepcopy(bundle["perspectives"])
            data["replayId"] = path.split("/")[-2]
        else:
            data = deepcopy(bundle["turns"])
            data["replayId"] = path.split("/")[-2]
        return data, {"url": path, "sha256": "fixture"}
    with patch.object(PublicArchiveClient, "get", fetch):
        first = collect(tmp_path, parser, max_matches=2)
        second = collect(tmp_path, parser, max_matches=2)
    assert first["matchCount"] == second["matchCount"] == 1
    assert len(second["aliases"]) == 1
    assert not second["errors"]


def test_lightweight_import_no_torch():
    code = "import sys; from sapai_gym1.replay import ReplayParser; from sapai_gym1.replay.evidence import EvidenceParser; assert 'torch' not in sys.modules"
    subprocess.run([sys.executable, "-c", code], check=True)
