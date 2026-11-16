"""General native replay evidence parser, deliberately not a simulator.

Identity annotation can use a later serialized object; that does NOT make its
later stats a valid earlier observation. No examples/behavior likelihoods or
invented intermediate states are exported by this module.
"""
from __future__ import annotations

from collections import Counter
from copy import deepcopy

from .archive import sha_file, split_for_match
from .parser import ACTION_NAMES, ReplayParser, decode_nested_json, object_key


SCHEMA = "sap-native-evidence-v1"
SHOP_TYPES = {5, 6, 7, 8, 9, 11}


def mapping(value, field, optional=True):
    value = decode_nested_json(value)
    if value is None and optional:
        return {}
    if not isinstance(value, dict):
        raise ValueError(f"Expected mapping at {field}")
    return value


class EvidenceParser:
    def __init__(self, calculator_root):
        self.names = ReplayParser(calculator_root)
        from pathlib import Path
        directory = Path(calculator_root) / "src/assets/data"
        self.vocabulary = {name: sha_file(directory / name) for name in ("pets.json", "food.json")}

    def __call__(self, bundle):
        metadata = bundle["metadata"]
        if str(metadata["game_version"]) != "48":
            raise ValueError("Only observed version-48 wire defaults are supported")
        payload = bundle["perspectives"]
        if payload.get("replayId") != bundle["replayId"] or bundle["turns"].get("replayId") != bundle["replayId"]:
            raise ValueError("Replay payload identity mismatch")
        perspectives = payload.get("perspectives")
        if not isinstance(perspectives, list) or len(perspectives) != 2:
            raise ValueError("Two perspectives required for this collector")
        pids = [p.get("participationId") for p in perspectives]
        if len(set(pids)) != 2 or None in pids:
            raise ValueError("Distinct participation IDs required")
        parsed = [self.perspective(p, i, payload["matchId"]) for i, p in enumerate(perspectives)]
        return {"schema": SCHEMA, "matchId": payload["matchId"], "replayId": bundle["replayId"],
                "split": split_for_match(payload["matchId"]), "gameVersion": "48", "vocabularySHA256": self.vocabulary,
                "perspectives": parsed, "trainingApproved": False, "policyObservationsExported": False,
                "summary": {"perspectives": len(parsed), "transactions": sum(p["summary"]["transactions"] for p in parsed),
                    "unknownSelectedIdentities": sum(p["summary"]["unknownSelectedIdentities"] for p in parsed),
                    "hashMismatches": sum(p["summary"]["hashMismatches"] for p in parsed),
                    "nativeEndpoints": sum(p["summary"]["nativeEndpoints"] for p in parsed),
                    "completeBeforeAfterStates": 0},
                "restrictions": ["annotation catalog may use future identity evidence; never copy future stats into past observations",
                    "native hashes are receipt comparisons, not independently recomputed state hashes",
                    "freeze/order batches do not reveal original click order or timestamps",
                    "both private perspectives are archival evidence; never concatenate them into player observations",
                    "original human likelihood and model GRU state are unavailable",
                    "requires visual state alignment and rules validation before any training export"]}

    def perspective(self, perspective, seat, match_id):
        raw = mapping(perspective.get("raw"), "raw", optional=False)
        if raw.get("MatchId") != match_id or raw.get("ParticipationId") != perspective["participationId"]:
            raise ValueError("Match/participation identity mismatch")
        actions = raw.get("Actions")
        if not isinstance(actions, list) or not actions:
            raise ValueError("Missing native actions")
        prefix = f"/perspectives/perspectives/{seat}/raw"
        catalog, entities, conflicts = {}, [], []

        def register(obj, kind, pointer, index):
            obj = mapping(obj, pointer, optional=False)
            uid = object_key(obj.get("Id"))
            if uid is None:
                raise ValueError(f"Object missing full native ID at {pointer}")
            # Type is established by its container. Serializer omits enum=0,
            # including Apple; preserve the reason rather than global fallback.
            enum = int(obj.get("Enu", 0))
            names = self.names.pet_names if kind == "pet" else self.names.food_names
            name = names.get(str(enum))
            info = {"id": uid, "kind": kind, "enum": enum, "name": name,
                    "evidence": {"file": "BUNDLE.json", "pointer": pointer, "actionIndex": index},
                    "enumDefaultApplied": "Enu" not in obj, "raw": obj}
            entities.append(info)
            existing = catalog.get(uid)
            if existing and (existing["kind"], existing["enum"]) != (kind, enum):
                conflicts.append({"id": uid, "before": [existing["kind"], existing["enum"]],
                                  "after": [kind, enum], "pointer": pointer})
                # Transforming pets exist; do not force one constant identity.
                existing["ambiguousOrTransformed"] = True
            elif existing:
                existing["evidence"].append(info["evidence"])
            else:
                catalog[uid] = {"kind": kind, "enum": enum, "name": name, "evidence": [info["evidence"]],
                                "ambiguousOrTransformed": False}
            return info

        snapshots = []
        def board(build, pointer, index):
            if not build:
                return
            build = mapping(build, pointer)
            bor = mapping(build.get("Bor", build), pointer + "/Bor")
            for group, items, kind in [("Mins/Items", (bor.get("Mins") or {}).get("Items", []), "pet"),
                                      ("MiSh", bor.get("MiSh", []), "pet"), ("SpSh", bor.get("SpSh", []), "food")]:
                for slot, item in enumerate(items or []):
                    if item is not None:
                        register(item, kind, f"{pointer}/Bor/{group}/{slot}", index)
            snapshots.append({"actionIndex": index, "turn": bor.get("Tur"), "hash": bor.get("Hash"),
                "phase": "genesis" if index < 0 else "native_submitted_build_not_pre_action",
                "evidence": {"file": "BUNDLE.json", "pointer": pointer}, "board": bor})

        board(raw.get("GenesisBuildModel"), prefix + "/GenesisBuildModel", -1)
        records = []
        for index, action in enumerate(actions):
            req, res = mapping(action.get("Request"), "Request"), mapping(action.get("Response"), "Response")
            typ, turn = int(action.get("Type", -1)), int(action.get("Turn", 0))
            if req.get("Version") not in (None, 48):
                raise ValueError("Mixed protocol versions")
            board(action.get("Build"), f"{prefix}/Actions/{index}/Build", index)
            source = object_key(req.get("MinionId") or req.get("SourceMinionId") or req.get("SpellId") or req.get("Minion"))
            target = object_key(req.get("TargetMinionId") or req.get("Aim"))
            consumed = []
            def response_foods(node, pointer):
                node = decode_nested_json(node)
                if isinstance(node, dict):
                    for k, value in node.items():
                        if k == "Spell" and isinstance(value, dict) and object_key(value.get("Id")):
                            consumed.append(register(value, "food", pointer + "/" + k, index))
                        else:
                            response_foods(value, pointer + "/" + k)
                elif isinstance(node, list):
                    for j, value in enumerate(node):
                        response_foods(value, pointer + f"/{j}")
            response_foods(res, f"{prefix}/Actions/{index}/Response (decoded)")
            spell = next((x for x in consumed if x["id"] == source), None)
            data, returned = req.get("Data") or {}, res.get("Data") or {}
            records.append({"index": index, "turn": turn, "type": typ,
                "action": ACTION_NAMES.get(typ, "unknown"),
                "sourceId": source, "targetId": target,
                "destination": deepcopy(req.get("Point")),
                "consumedFood": {"name": spell["name"], "enum": spell["enum"], "price": spell["raw"].get("Pri", 0),
                                 "priceDefaultApplied": "Pri" not in spell["raw"], "evidence": spell["evidence"]} if spell else None,
                "preHash": data.get("BoardHash"), "postHash": returned.get("Hash"),
                "freezeBatch": deepcopy(data.get("BoardFreezes")), "reorderBatch": deepcopy(data.get("BoardOrders")),
                "evidence": {"file": "BUNDLE.json", "pointer": f"{prefix}/Actions/{index}"},
                "nativeCreatedOn": action.get("CreatedOn"), "beforeState": None, "afterState": None,
                "beforeAfterStatus": "requires_visual_alignment_and_transition_validation"})

        # Propagate only explicit same-species combine equality; conflicts remain
        # unresolved (e.g. transformed IDs), rather than last-seen identity wins.
        for _ in range(len(records)):
            changed = False
            for rec in records:
                if rec["type"] != 7:
                    continue
                for a, b in [(rec["sourceId"], rec["targetId"]), (rec["targetId"], rec["sourceId"])]:
                    known = catalog.get(b)
                    if a and a not in catalog and known and known["kind"] == "pet" and not known["ambiguousOrTransformed"]:
                        catalog[a] = {**deepcopy(known), "derivedViaCombine": rec["index"]}
                        changed = True
            if not changed:
                break
        last_hash, last_turn = None, None
        checks, gaps = [], []
        for rec in records:
            source = catalog.get(rec["sourceId"])
            rec["sourceIdentityAnnotation"] = deepcopy(source)
            if source:
                rec["identityUsesFutureEvidence"] = all(e["actionIndex"] > rec["index"] for e in source["evidence"])
            if rec["turn"] != last_turn:
                last_hash, last_turn = None, rec["turn"]
            if rec["preHash"] is not None and last_hash is not None:
                checks.append({"kind": "receipt_chain", "index": rec["index"], "expected": last_hash,
                               "actual": rec["preHash"], "match": last_hash == rec["preHash"]})
            if rec["postHash"] is not None:
                last_hash = rec["postHash"]
            if rec["type"] in (6, 7, 8, 9) and (not source or source["ambiguousOrTransformed"] or not source["name"]):
                gaps.append({"index": rec["index"], "field": "sourceIdentity", "reason": "missing_or_ambiguous_native_identity"})
            if rec["freezeBatch"] or rec["reorderBatch"]:
                gaps.append({"index": rec["index"], "field": "originalLocalClickSequence", "reason": "not_uniquely_recorded; final batch retained"})
        end_by_turn = {r["turn"]: r for r in records if r["type"] == 11}
        endpoints = []
        for snap in snapshots:
            if snap["actionIndex"] < 0:
                continue
            end = end_by_turn.get(snap["turn"])
            if end:
                check = {"kind": "end_receipt", "index": end["index"], "turn": snap["turn"],
                         "expected": end["postHash"], "actual": snap["hash"],
                         "match": end["postHash"] is not None and end["postHash"] == snap["hash"]}
                checks.append(check)
                endpoints.append({"turn": snap["turn"], "endIndex": end["index"], "snapshotIndex": snap["actionIndex"], "hashMatches": check["match"]})
        turns = sorted({r["turn"] for r in records if r["type"] in SHOP_TYPES})
        return {"role": perspective["role"], "participationId": perspective["participationId"], "name": raw.get("UserName"),
                "replayCode": {"Pid": perspective["participationId"], "T": 1}, "expectedShopTurns": turns,
                "nativeLastTurnMetadata": raw.get("LastTurn"), "outcome": raw.get("Outcome"),
                "actions": records, "identityCatalog": catalog, "identityConflicts": conflicts,
                "serializedBoards": snapshots, "endpoints": endpoints, "checks": checks, "gaps": gaps,
                "summary": {"actions": len(records), "transactions": sum(r["type"] in SHOP_TYPES for r in records),
                    "unknownSelectedIdentities": sum(g["field"] == "sourceIdentity" for g in gaps),
                    "hashMismatches": sum(not c["match"] for c in checks), "nativeEndpoints": len(endpoints),
                    "actionCounts": dict(Counter(r["action"] for r in records))}}
