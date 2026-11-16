#!/usr/bin/env python3
"""Build a blinded SAP-native expert review queue for search disagreements."""

from __future__ import annotations

import argparse
import glob
import hashlib
import html
import json
from pathlib import Path

from sapai_gym1.rl.action_schema import ActionType, NO_TARGET
from sapai_gym1.rl.entity_observation import EntityKind, EntityObservationEncoder
from sapai_gym1.rl.game_trace import short_name


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--audit", action="append", default=[])
    parser.add_argument("--examples", action="append", default=[])
    parser.add_argument("--planner-report", action="append", default=[])
    parser.add_argument("--puct-report", action="append", default=[])
    parser.add_argument("--puct-scenario-report", action="append", default=[])
    parser.add_argument("--puct-audit", action="append", default=[])
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--limit", type=int, default=100)
    return parser.parse_args()


def expanded(patterns):
    paths = []
    for pattern in patterns:
        matches = glob.glob(pattern)
        paths.extend(matches or [pattern])
    return [Path(path) for path in sorted(dict.fromkeys(paths)) if Path(path).is_file()]


def load_jsonl(paths):
    rows = []
    for path in paths:
        with path.open() as source:
            for line in source:
                if line.strip():
                    row = json.loads(line)
                    row["_source"] = str(path)
                    rows.append(row)
    return rows


def current_experience(level, total):
    if level <= 1:
        return total
    if level == 2:
        return max(0, total - 2)
    return 0


def decode_observation(observation):
    encoder = EntityObservationEncoder()
    item_names = {value: key for key, value in encoder.item_to_id.items()}
    status_names = {value: key for key, value in encoder.status_to_id.items()}
    item_ids = observation["item_ids"]
    kind_ids = observation["kind_ids"]
    status_ids = observation["status_ids"]
    numeric = observation["numeric"]

    def item(token):
        kind = EntityKind(int(kind_ids[token]))
        if kind in {EntityKind.EMPTY, EntityKind.GLOBAL}:
            return None
        name = item_names.get(int(item_ids[token]), "none")
        status = status_names.get(int(status_ids[token]), "none")
        status = None if status in {"none", "status-none"} else status
        if kind in {
            EntityKind.TEAM_PET,
            EntityKind.SHOP_PET,
            EntityKind.OPPONENT_PET,
        }:
            level = max(1, round(float(numeric[token][2]) * 3))
            total = round(float(numeric[token][3]) * 5)
            return {
                "kind": "pet",
                "name": name,
                "label": short_name(name),
                "attack": round(float(numeric[token][0]) * 50),
                "health": round(float(numeric[token][1]) * 50),
                "level": level,
                "experience": current_experience(level, total),
                "status": status,
                "statusLabel": short_name(status) if status else None,
                "sellValue": max(1, round(float(numeric[token][15]) * 10)),
            }
        return {
            "kind": "food",
            "name": name,
            "label": short_name(name),
            "attack": round(float(numeric[token][0]) * 50),
            "health": round(float(numeric[token][1]) * 50),
            "status": status,
            "statusLabel": short_name(status) if status else None,
        }

    team = [item(1 + index) for index in range(5)]
    shop = []
    for index in range(10):
        token = 6 + index
        decoded = item(token)
        if decoded is None:
            shop.append(
                {
                    "index": index,
                    "slotType": "empty",
                    "cost": 0,
                    "frozen": False,
                    "rewardGroup": None,
                    "item": None,
                }
            )
            continue
        reward = float(numeric[token][11]) > 0.5
        shop.append(
            {
                "index": index,
                "slotType": (
                    "levelup"
                    if reward
                    else "pet"
                    if decoded["kind"] == "pet"
                    else "food"
                ),
                "cost": round(float(numeric[token][5]) * 3),
                "frozen": float(numeric[token][6]) > 0.5,
                "rewardGroup": (
                    round(float(numeric[token][12]) * 10)
                    if reward
                    else None
                ),
                "item": decoded,
            }
        )
    opponent = [item(16 + index) for index in range(5)]
    global_values = numeric[0]
    return {
        "player": {
            "turn": round(float(global_values[12]) * 25),
            "lives": round(float(global_values[13]) * 6),
            "wins": round(float(global_values[14]) * 10),
            "gold": round(float(global_values[11]) * 30),
            "team": team,
            "shop": shop,
        },
        "opponentScore": {
            "lives": round(float(global_values[9]) * 6),
            "wins": round(float(global_values[10]) * 10),
        },
        "lastOpponentTeam": opponent,
        "actionCount": round(float(global_values[5]) * 50),
        "reorderCount": round(float(global_values[0]) * 2),
    }


def action_description(state, action):
    action_type = ActionType(int(action[0]))
    source, target = int(action[1]), int(action[2])
    team = state["player"]["team"]
    shop = state["player"]["shop"]
    source_item = None
    if action_type in {
        ActionType.BUY_PET,
        ActionType.BUY_FOOD,
        ActionType.FREEZE,
        ActionType.UNFREEZE,
    } and 0 <= source < len(shop):
        source_item = shop[source]["item"]
    elif action_type in {ActionType.SELL, ActionType.COMBINE, ActionType.MOVE} and 0 <= source < len(team):
        source_item = team[source]
    source_name = source_item["label"] if source_item else "empty slot"
    target_item = team[target] if 0 <= target < len(team) else None
    target_name = target_item["label"] if target_item else "empty slot"
    if action_type is ActionType.END_TURN:
        return "End turn"
    if action_type is ActionType.ROLL:
        return "Roll shop"
    if action_type is ActionType.FREEZE:
        return f"Freeze {source_name} · shop {source}"
    if action_type is ActionType.UNFREEZE:
        return f"Unfreeze {source_name} · shop {source}"
    if action_type is ActionType.SELL:
        return f"Sell {source_name} · team {source}"
    if action_type is ActionType.BUY_PET:
        verb = f"combine into {target_name}" if target_item else f"place in team {target}"
        return f"Buy {source_name} · {verb}"
    if action_type is ActionType.BUY_FOOD:
        destination = "team" if target == NO_TARGET else target_name
        return f"Buy {source_name} · feed {destination}"
    if action_type is ActionType.COMBINE:
        return f"Combine {source_name} into {target_name}"
    if action_type is ActionType.MOVE:
        return f"Swap {source_name} with {target_name}"
    return action_type.name.replace("_", " ").title()


def audit_key(row):
    return (
        int(row["turn"]),
        tuple(row["policyAction"]),
        tuple(row["bestAction"]),
        round(float(row["improvement"]), 7),
    )


def example_key(row):
    teacher = row["teacher"]
    return (
        int(row["turn"]),
        tuple(teacher["policyAction"]),
        tuple(teacher["searchBestAction"]),
        round(float(teacher["improvement"]), 7),
    )


def make_case(case_id, state, policy_action, search_action, evidence, reasons, traces, source):
    blind_search_first = int(hashlib.sha256(case_id.encode()).hexdigest()[:2], 16) % 2 == 0
    search = {
        "tuple": search_action,
        "description": action_description(state, search_action),
        "origin": "search",
    }
    policy = {
        "tuple": policy_action,
        "description": action_description(state, policy_action),
        "origin": "policy",
    }
    options = [search, policy] if blind_search_first else [policy, search]
    return {
        "id": case_id,
        "state": state,
        "turn": state["player"]["turn"],
        "reasons": reasons,
        "options": options,
        "sequenceOrder": (
            int(policy_action[0]) == int(ActionType.MOVE)
            or int(search_action[0]) == int(ActionType.MOVE)
        ),
        "evidence": evidence,
        "traces": traces,
        "source": source,
    }


def cases_from_search(audits, examples):
    example_lookup = {}
    for row in examples:
        teacher = row.get("teacher", {})
        if int(teacher.get("searchTargetRank", 99)) != 0:
            continue
        example_lookup.setdefault(example_key(row), row)
    cases = []
    for index, audit in enumerate(audits):
        if audit["bestAction"] == audit["policyAction"]:
            continue
        example = example_lookup.get(audit_key(audit))
        if audit.get("publicState") is not None:
            state = audit["publicState"]
        elif example is not None:
            state = decode_observation(example["observation"])
        else:
            continue
        evidence = {
            "accepted": bool(audit.get("accepted")),
            "selectionMean": audit.get("improvement"),
            "selectionLower": audit.get("improvementLowerBound"),
            "confirmationMean": audit.get("validationReturnImprovement"),
            "confirmationLower": audit.get("validationReturnLowerBound"),
            "outerMean": audit.get("outerAuditReturnImprovement"),
            "outerLower": audit.get("outerAuditReturnLowerBound"),
            "outerFirstBattle": audit.get("outerAuditFirstBattleScoreImprovement"),
            "outerHorizonBattle": audit.get("outerAuditHorizonBattleScoreImprovement"),
        }
        traces = {
            "search": audit.get("bestRollouts", []),
            "policy": audit.get("policyRollouts", []),
        }
        source = audit.get("_source", "")
        digest = hashlib.sha256(
            (source + json.dumps(audit_key(audit))).encode()
        ).hexdigest()[:12]
        cases.append(
            make_case(
                f"search-{digest}",
                state,
                audit["policyAction"],
                audit["bestAction"],
                evidence,
                audit.get("reasons", []),
                traces,
                source,
            )
        )
    return cases


def cases_from_planners(paths):
    cases = []
    for path in paths:
        payload = json.loads(path.read_text())
        for row in payload.get("rows", []):
            state = row.get("publicState")
            if state is None:
                continue
            root = row["planners"]["root"]
            for planner_name in ("nested1", "nested2"):
                planner = row["planners"][planner_name]
                if planner["action"] == root["action"]:
                    continue
                evidence = {
                    "accepted": None,
                    "selectionMean": None,
                    "selectionLower": None,
                    "confirmationMean": planner["outerReturnAdvantageVsRaw"],
                    "confirmationLower": planner["outerReturnLowerBound"],
                    "outerMean": None,
                    "outerLower": None,
                    "outerFirstBattle": planner["outerFirstBattleScoreAdvantageVsRaw"],
                    "outerHorizonBattle": planner["outerHorizonBattleScoreAdvantageVsRaw"],
                }
                case_id = f"planner-{path.stem}-{row['state']}-{planner_name}"
                case = make_case(
                    case_id,
                    state,
                    root["action"],
                    planner["action"],
                    evidence,
                    [f"root-vs-{planner_name}"],
                    {},
                    str(path),
                )
                case["options"][0]["origin"] = (
                    planner_name if case["options"][0]["origin"] == "search" else "root"
                )
                case["options"][1]["origin"] = (
                    planner_name if case["options"][1]["origin"] == "search" else "root"
                )
                cases.append(case)
    return cases


def cases_from_puct(paths):
    cases = []
    for path in paths:
        payload = json.loads(path.read_text())
        for row in payload.get("rows", []):
            state = row.get("publicState")
            if state is None:
                continue
            for planner_name in ("root", "puct"):
                planner = row[planner_name]
                proposed = planner["proposedAction"]
                raw = planner["policyAction"]
                if proposed == raw:
                    continue
                evidence = {
                    "accepted": planner["accepted"],
                    "independentlyConfirmed": planner.get(
                        "independentlyConfirmed"
                    ),
                    "selectionMean": None,
                    "selectionLower": None,
                    "confirmationMean": planner.get(
                        "proposedReturnImprovement"
                    ),
                    "confirmationLower": planner.get(
                        "proposedReturnLowerBound"
                    ),
                    "outerMean": planner.get(
                        "outerAuditReturnImprovement"
                    ),
                    "outerLower": planner.get(
                        "outerAuditReturnLowerBound"
                    ),
                    "outerFirstBattle": None,
                    "outerHorizonBattle": planner.get(
                        "horizonBattleAdvantageVsRaw"
                    ),
                    "independentFull": planner.get(
                        "independentProposedReturnAdvantageVsRaw"
                    ),
                    "independentRootOnly": planner.get(
                        "independentProposedRootOnlyReturnAdvantageVsRaw"
                    ),
                    "continuationLift": planner.get(
                        "independentPlanningContinuationLift"
                    ),
                }
                case_id = (
                    f"equal-{path.stem}-{row['state']}-{planner_name}"
                )
                case = make_case(
                    case_id,
                    state,
                    raw,
                    proposed,
                    evidence,
                    list(row.get("opportunityReasons", []))
                    + [f"{planner_name}-proposal-vs-raw"],
                    {},
                    str(path),
                )
                for option in case["options"]:
                    option["origin"] = (
                        planner_name
                        if option["origin"] == "search"
                        else "raw"
                    )
                cases.append(case)
    return cases


def cases_from_puct_scenarios(paths):
    cases = []
    for path in paths:
        payload = json.loads(path.read_text())
        for row in payload.get("rows", []):
            state = row["publicState"]
            raw = row["rawAction"]
            for planner_name in ("root", "puct"):
                planner = row[planner_name]
                proposed = planner["proposedAction"]
                if proposed == raw:
                    continue
                case_id = (
                    f"scenario-{path.stem}-{row['scenario']}-"
                    f"{row['example']}-{planner_name}"
                )
                case = make_case(
                    case_id,
                    state,
                    raw,
                    proposed,
                    {
                        "accepted": planner["accepted"],
                        "selectionMean": None,
                        "selectionLower": None,
                        "confirmationMean": planner.get(
                            "validationReturnImprovement"
                        ),
                        "confirmationLower": planner.get(
                            "validationReturnLowerBound"
                        ),
                        "outerMean": None,
                        "outerLower": None,
                        "outerFirstBattle": None,
                        "outerHorizonBattle": planner.get(
                            "validationHorizonBattleScoreImprovement"
                        ),
                    },
                    [f"scenario:{row['scenario']}", f"{planner_name}-vs-raw"],
                    {},
                    str(path),
                )
                case["expertExpected"] = row["expectedActionDetail"][
                    "description"
                ]
                for option in case["options"]:
                    option["origin"] = (
                        planner_name
                        if option["origin"] == "search"
                        else "raw"
                    )
                cases.append(case)
    return cases


def cases_from_puct_audits(paths):
    cases = []
    for row in load_jsonl(paths):
        state = row.get("publicState")
        raw = row.get("policyAction")
        proposed = row.get("proposedAction")
        if state is None or raw is None or proposed is None or raw == proposed:
            continue
        case_id = (
            f"puct-audit-{Path(row['_source']).stem}-"
            f"{row.get('scenarioSeed', row.get('game', 0))}-"
            f"{row.get('side', 0)}"
        )
        case = make_case(
            case_id,
            state,
            raw,
            proposed,
            {
                "accepted": row.get("accepted"),
                "selectionMean": None,
                "selectionLower": None,
                "confirmationMean": row.get(
                    "validationReturnImprovement"
                ),
                "confirmationLower": row.get(
                    "validationReturnLowerBound"
                ),
                "outerMean": row.get("outerAuditReturnImprovement"),
                "outerLower": row.get("outerAuditReturnLowerBound"),
                "outerFirstBattle": row.get(
                    "outerAuditFirstBattleScoreImprovement"
                ),
                "outerHorizonBattle": row.get(
                    "outerAuditHorizonBattleScoreImprovement"
                ),
            },
            list(row.get("reasons", []))
            + (["pooled-candidate"] if row.get("candidateEligible") else []),
            {},
            row["_source"],
        )
        for option in case["options"]:
            option["origin"] = (
                "puct" if option["origin"] == "search" else "raw"
            )
        expected = row.get("expectedAction")
        if expected is not None:
            case["expertExpected"] = action_description(state, expected)
        cases.append(case)
    return cases


def priority(case):
    evidence = case["evidence"]
    reasons = set(case["reasons"])
    combo = bool(reasons & {"sleeping-pill", "mushroom", "full-board-pivot"})
    accepted = evidence["accepted"] is True
    has_outer_audit = evidence.get("outerMean") is not None
    near = abs(float(evidence.get("confirmationLower") or 0.0)) <= 0.10
    margin = abs(float(evidence.get("confirmationMean") or 0.0))
    return (
        has_outer_audit,
        accepted,
        not case.get("sequenceOrder", False),
        combo,
        near,
        margin,
    )


def render(cases, output):
    payload = json.dumps(cases, separators=(",", ":"), ensure_ascii=True).replace("</", "<\\/")
    document = TEMPLATE.replace("__CASES__", payload)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(document)


def main():
    args = parse_args()
    audits = load_jsonl(expanded(args.audit))
    examples = load_jsonl(expanded(args.examples))
    cases = cases_from_search(audits, examples)
    cases.extend(cases_from_planners(expanded(args.planner_report)))
    cases.extend(cases_from_puct(expanded(args.puct_report)))
    cases.extend(
        cases_from_puct_scenarios(expanded(args.puct_scenario_report))
    )
    cases.extend(cases_from_puct_audits(expanded(args.puct_audit)))
    cases.sort(key=priority, reverse=True)
    render(cases[: args.limit], args.output)
    print(f"cases={min(len(cases), args.limit)} output={args.output}")


TEMPLATE = r'''<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>SAP Search Expert Review</title>
<style>
@font-face{font-family:Lapsus;src:url('sap-fonts/LapsusPro-Regular.otf')}@font-face{font-family:Lapsus;src:url('sap-fonts/LapsusPro-Bold.otf');font-weight:700 1000}
:root{--cream:#f6f0d7;--paper:#fffdf2;--ink:#211b17;--muted:#6e655a;--line:#3c3028;--yellow:#ffd530;--orange:#f08b2f;--green:#4da66b;--red:#d9534f;--blue:#4b83d1;--ice:#c9eff5;--shadow:0 7px 0 #2b211b22}*{box-sizing:border-box}body{margin:0;background:linear-gradient(135deg,#fff7c9,#f4ecd7 50%,#ffef9a);color:var(--ink);font:14px/1.35 ui-sans-serif,system-ui,sans-serif}.app{max-width:1700px;margin:auto;padding:16px}.top{display:flex;align-items:center;gap:16px;margin-bottom:12px}.brand{font:1000 25px/1 Lapsus,ui-sans-serif,sans-serif}.progress{color:var(--muted)}.top button,.nav button,.rating button,.reveal{border:3px solid var(--line);border-radius:10px;background:var(--paper);color:var(--ink);padding:8px 12px;font-weight:850;cursor:pointer;box-shadow:0 3px 0 #2b211b33}.top button:hover,.nav button:hover,.rating button:hover,.reveal:hover{transform:translateY(-1px)}.top .export{margin-left:auto}.layout{display:grid;grid-template-columns:235px minmax(0,1fr);gap:13px}.rail,.main{background:var(--paper);border:3px solid var(--line);border-radius:15px;box-shadow:var(--shadow)}.rail{padding:9px;max-height:calc(100vh - 86px);overflow:auto;position:sticky;top:10px}.rail button{display:grid;grid-template-columns:27px 1fr 17px;gap:6px;align-items:center;width:100%;border:0;border-left:5px solid transparent;border-radius:8px;background:transparent;color:var(--ink);padding:8px 6px;text-align:left;cursor:pointer}.rail button:hover{background:#f4ead0}.rail button.active{background:#ffe47b;border-left-color:var(--orange)}.rail .done{color:var(--green);font-weight:1000}.main{padding:14px;min-width:0}.hud{display:flex;align-items:center;gap:12px;border-bottom:2px solid #d8cdb8;padding-bottom:11px}.hud-title{font:1000 21px/1.1 Lapsus,ui-sans-serif,sans-serif}.counters{display:flex;gap:19px;margin-left:auto}.counter{position:relative;display:flex;align-items:center;justify-content:center;min-width:67px;height:41px;padding-left:24px;background:#fff;border:3px solid #111;border-radius:8px;font:1000 25px/1 Lapsus}.counter img{position:absolute;left:-18px;width:43px;height:43px;object-fit:contain}.opp-label{font:900 11px/1 Lapsus;color:var(--muted);align-self:center}.boards-head,.boards{display:grid;grid-template-columns:1fr 44px 1fr;gap:6px}.boards-head{margin-top:13px;font-weight:900}.boards-head div:last-child{text-align:right}.boards{align-items:stretch}.slots{display:grid;grid-template-columns:repeat(5,minmax(90px,1fr));gap:5px}.versus{display:flex;align-items:center;justify-content:center;font:1000 18px Lapsus;color:var(--orange);border-left:2px solid #d8cdb8;border-right:2px solid #d8cdb8}.card{position:relative;min-height:193px;border:2px solid var(--line);border-radius:11px;background:#efe7d2;overflow:hidden}.card.empty{opacity:.38;background:#fff;border-style:dashed}.slot{position:absolute;left:7px;top:5px;z-index:5;font-size:10px;color:var(--muted);font-weight:800}.name{position:absolute;top:20px;left:4px;right:4px;text-align:center;font:900 13px Lapsus;z-index:5}.stage{position:relative;height:150px;margin-top:33px}.sprite{position:absolute;left:50%;top:42px;width:91px;height:91px;transform:translateX(-50%);object-fit:contain;filter:drop-shadow(0 4px 3px #0005)}.ours .sprite,.shop .sprite{transform:translateX(-50%) scaleX(-1)}.level{position:absolute;z-index:6;left:50%;top:0;transform:translateX(-50%);display:flex;flex-direction:column;align-items:center}.level img{width:42px;height:25px;object-fit:contain}.xp{display:flex;width:54px;height:10px;gap:2px;padding:2px 3px;border:2px solid #140d08;border-radius:8px;background:#4a2805}.xp i{flex:1;border-radius:2px;background:#7a4208}.xp i.on{background:#ffd31f}.perk{position:absolute;right:2px;top:41px;width:35px;height:35px;border:2px solid #111;border-radius:50%;background:#fff;padding:3px;z-index:7}.stat{position:absolute;bottom:0;width:46px;height:46px;z-index:8}.stat.atk{left:3px}.stat.hp{right:3px}.stat img{width:100%;height:100%}.stat b{position:absolute;inset:0;display:flex;align-items:center;justify-content:center;color:#fff;font:1000 20px Lapsus;-webkit-text-stroke:1.4px #111}.sell{position:absolute;bottom:4px;left:50%;transform:translateX(-50%);font-size:10px;color:var(--muted)}.frozen:after{content:'FROZEN';position:absolute;z-index:10;right:-24px;top:14px;transform:rotate(40deg);padding:2px 25px;background:#43a7c8;color:white;font:900 9px Lapsus}.shop-wrap{margin-top:13px}.section{font-weight:900;margin-bottom:6px}.shop{display:grid;grid-template-columns:repeat(10,minmax(74px,1fr));gap:4px}.shop .card{min-height:146px}.shop .stage{height:107px;margin-top:27px}.shop .sprite{top:30px;width:66px;height:66px}.shop .level img{width:31px;height:19px}.shop .xp{width:43px;height:8px;border-width:1px;padding:1px 2px}.shop .stat{width:34px;height:34px}.shop .stat b{font-size:15px}.shop .name{font-size:11px;top:18px}.shop .perk{width:27px;height:27px;top:31px}.price{position:absolute;right:4px;bottom:2px;color:#b87900;font-weight:1000}.choice{margin-top:14px;padding-top:13px;border-top:2px solid #d8cdb8}.choice h2{margin:0 0 9px;font:1000 20px Lapsus}.options{display:grid;grid-template-columns:1fr 1fr;gap:10px}.option{border:4px solid var(--line);border-radius:13px;background:#fff8d6;padding:12px}.option-letter{font:1000 24px Lapsus;color:var(--orange)}.option-action{font:1000 20px Lapsus;margin-top:2px}.option-tuple{color:var(--muted);font-size:11px;margin-top:4px}.rating{display:grid;grid-template-columns:repeat(5,1fr);gap:6px;margin-top:10px}.rating button.selected{background:var(--yellow)}textarea{width:100%;min-height:66px;margin-top:8px;padding:9px;border:2px solid var(--line);border-radius:9px;background:#fff;color:#211b17;font:14px ui-sans-serif;resize:vertical}.review-actions{display:flex;align-items:center;gap:8px;margin-top:8px}.reveal:disabled{opacity:.4;cursor:not-allowed}.evidence{display:none;margin-top:12px;border:2px dashed #8e7b63;border-radius:10px;padding:10px;background:#f5edd9}.evidence.show{display:block}.origins{font:1000 17px Lapsus;margin-bottom:7px}.metrics{display:grid;grid-template-columns:repeat(4,1fr);gap:6px}.metric{background:#fff;border:1px solid #cfc0a7;border-radius:7px;padding:7px;color:var(--muted);font-size:11px}.metric b{display:block;color:var(--ink);font-size:17px;margin-top:2px}.traces{display:grid;grid-template-columns:1fr 1fr;gap:8px;margin-top:8px}.trace{background:#fff;border:1px solid #cfc0a7;border-radius:7px;padding:7px}.trace b{font-size:12px}.trace pre{white-space:pre-wrap;font:11px/1.35 ui-monospace;margin:5px 0 0;max-height:135px;overflow:auto}.nav{display:flex;justify-content:space-between;margin-top:11px}.tag{display:inline-block;padding:2px 6px;margin:3px 3px 0 0;background:#eee3c9;border-radius:5px;font-size:10px;color:var(--muted)}
.choice h2 small{font:700 12px ui-sans-serif;color:var(--muted)}.rating{grid-template-columns:repeat(6,1fr)}
@media(max-width:1050px){.layout{grid-template-columns:1fr}.rail{position:static;max-height:160px}.shop{grid-template-columns:repeat(5,1fr)}.rating{grid-template-columns:1fr 1fr}.metrics{grid-template-columns:1fr 1fr}}@media(max-width:700px){.boards-head{display:none}.boards{grid-template-columns:1fr}.versus{min-height:32px;border:0}.slots{grid-template-columns:repeat(2,1fr)}.options,.traces{grid-template-columns:1fr}.counters{flex-wrap:wrap}.shop{grid-template-columns:repeat(2,1fr)}}
</style></head><body><div class="app"><div class="top"><div class="brand">SAP Search Review</div><div class="progress" id="progress"></div><button class="export" id="export">Download ratings</button></div><div class="layout"><aside class="rail" id="rail"></aside><main class="main"><div class="hud"><div><div class="hud-title" id="caseTitle"></div><div id="tags"></div></div><div class="counters" id="counters"></div></div><div class="boards-head"><div>Your team · back → front</div><div></div><div>front → back · Last opponent</div></div><div class="boards"><div class="slots ours" id="team"></div><div class="versus">VS</div><div class="slots" id="opponent"></div></div><div class="shop-wrap"><div class="section">Shop · physical slots 0–9</div><div class="shop" id="shop"></div></div><section class="choice"><h2>Which should be the very next click? <small>Both actions may still happen afterward.</small></h2><div class="options" id="options"></div><div class="rating" id="rating"></div><textarea id="note" placeholder="Short strategic reason or suspected mechanic issue…"></textarea><div class="review-actions"><button class="reveal" id="reveal" disabled>Reveal search evidence</button><span id="saved"></span></div><div class="evidence" id="evidence"></div><div class="nav"><button id="prev">← Previous</button><button id="next">Next unrated →</button></div></section></main></div></div><script id="data" type="application/json">__CASES__</script><script>
const cases=JSON.parse(document.getElementById('data').textContent),key='sap-search-expert-ratings-v1';let ratings=JSON.parse(localStorage.getItem(key)||'{}'),index=0;const $=id=>document.getElementById(id),esc=s=>String(s??'').replace(/[&<>"']/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));
const petOverride={'pet-zombie-cricket':'CricketToken','pet-dirty-rat':'RatToken','pet-zombie-fly':'FlyToken'},foodOverride={'food-sleeping-pill':'Pill','food-cupcake':'Muffin','food-cake':'BirthdayCake','food-better-apple':'Apple2','food-best-apple':'Apple3','food-better-milk':'Milk2','food-best-milk':'Milk3'},statusNames={'status-honey-bee':'Honey','status-bone-attack':'MeatBone','status-garlic-armor':'Garlic','status-splash-attack':'Chili','status-melon-armor':'Melon','status-extra-life':'Mushroom','status-steak-attack':'Steak','status-poison-attack':'Peanut','status-cake':'BirthdayCake','status-bread':'Bread','status-coconut-shield':'Coconut'};
function pascal(s){return s.split('-').map(x=>x[0].toUpperCase()+x.slice(1)).join('')}function sprite(x){return x.kind==='pet'?`sap-sprites/Pets/${petOverride[x.name]||pascal(x.name.slice(4))}.png`:`sap-sprites/Food/${foodOverride[x.name]||pascal(x.name.slice(5))}.png`}function perk(x){return x?.status&&statusNames[x.status]?`sap-sprites/Food/${statusNames[x.status]}.png`:null}function xp(x){let n=x.level===1?2:x.level===2?3:0;if(!n)return'';return `<span class="xp">${Array.from({length:n},(_,i)=>`<i class="${i<x.experience?'on':''}"></i>`).join('')}</span>`}
function card(item,slot,kind){if(!item)return `<div class="card empty"><span class="slot">${kind} ${slot.index}</span></div>`;let p=item.kind==='pet',frozen=slot.frozen?' frozen':'',perkSrc=perk(item);if(!p)return `<div class="card${frozen}"><span class="slot">${kind} ${slot.index}</span><div class="name">${esc(item.label)}</div><div class="stage"><img class="sprite" src="${sprite(item)}"></div><b class="price">${slot.cost}g</b></div>`;return `<div class="card${frozen}"><span class="slot">${kind} ${slot.index}</span><div class="name">${esc(item.label)}</div><div class="stage"><span class="level"><img src="sap-icons/l${item.level}.png">${xp(item)}</span><img class="sprite" src="${sprite(item)}">${perkSrc?`<img class="perk" src="${perkSrc}" title="${esc(item.statusLabel)}">`:''}<span class="stat atk"><img src="sap-icons/fist.png"><b>${item.attack}</b></span><span class="stat hp"><img src="sap-icons/heart.png"><b>${item.health}</b></span></div><span class="sell">sell ${item.sellValue}</span>${Number.isFinite(slot.cost)?`<b class="price">${slot.cost}g</b>`:''}</div>`}
function teamHtml(team,ours){let ids=[0,1,2,3,4];if(ours)ids.reverse();return ids.map(i=>card(team[i],{index:i},i===0?'front':'team')).join('')}function shopHtml(shop){let by=Object.fromEntries(shop.map(s=>[s.index,s]));return Array.from({length:10},(_,i)=>{let s=by[i]||{index:i,item:null};return card(s.item,s,s.slotType||'shop')}).join('')}function fmt(v){return v==null?'—':Number(v).toFixed(3)}function battleSeq(t){if(!t?.length)return'No trace stored';return t.slice(0,8).map((x,i)=>`#${i+1} return ${Number(x.totalReturn).toFixed(2)} · battles ${(x.battleResults||[]).map(v=>v===0?'W':v===1?'L':'D').join('')}`).join('\n')}
const labels=[['a','A clearly better'],['b','B clearly better'],['equal','Roughly equal'],['both','Both / order unimportant'],['uncertain','Uncertain'],['mechanic','Rule / simulator concern']];async function syncRatings(){try{let response=await fetch('/api/search-expert-ratings',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({schemaVersion:1,syncedAt:new Date().toISOString(),ratings})});if(!response.ok)throw new Error(`HTTP ${response.status}`);$('saved').textContent='Synced to DGX'}catch(error){$('saved').textContent='Saved locally · sync unavailable'}}function save(){let c=cases[index],r=ratings[c.id]||{};r.note=$('note').value;r.updatedAt=new Date().toISOString();ratings[c.id]=r;localStorage.setItem(key,JSON.stringify(ratings));$('saved').textContent='Saved locally';renderRail();renderProgress();syncRatings()}async function hydrate(){try{let response=await fetch('/api/search-expert-ratings',{cache:'no-store'});if(response.ok){let remote=(await response.json()).ratings||{};for(let [id,value] of Object.entries(remote)){let local=ratings[id],remoteTime=Date.parse(value.updatedAt||0),localTime=Date.parse(local?.updatedAt||0);if(!local||remoteTime>localTime)ratings[id]=value}localStorage.setItem(key,JSON.stringify(ratings));if(Object.keys(ratings).length)await syncRatings()}}catch(error){}render()}
function renderProgress(){let done=cases.filter(c=>ratings[c.id]?.choice).length;$('progress').textContent=`${done} / ${cases.length} rated · case ${cases.length?index+1:0}`}
function renderRail(){$('rail').innerHTML=cases.map((c,i)=>`<button data-i="${i}" class="${i===index?'active':''}"><b>${i+1}</b><span>Turn ${c.turn}<br><small>${esc((c.reasons||[]).join(' · ')||'general')}</small></span><span class="done">${ratings[c.id]?.choice?'✓':''}</span></button>`).join('');$('rail').querySelectorAll('button').forEach(b=>b.onclick=()=>{index=+b.dataset.i;render()})}
function render(){if(!cases.length){$('caseTitle').textContent='No review cases yet';return}let c=cases[index],s=c.state,p=s.player,r=ratings[c.id]||{};$('caseTitle').textContent=`Turn ${p.turn} decision`;$('tags').innerHTML=(c.reasons||[]).concat(c.sequenceOrder?['sequence-order comparison']:[]).map(x=>`<span class="tag">${esc(x)}</span>`).join('');$('counters').innerHTML=`<span class="counter"><img src="sap-icons/gold.png">${p.gold}</span><span class="counter"><img src="sap-icons/heart.png">${p.lives}</span><span class="counter">🏆 ${p.wins}</span><span class="opp-label">OPP</span><span class="counter"><img src="sap-icons/heart.png">${s.opponentScore?.lives??'?'}</span><span class="counter">🏆 ${s.opponentScore?.wins??'?'}</span>`;$('team').innerHTML=teamHtml(p.team,true);$('opponent').innerHTML=teamHtml(s.lastOpponentTeam||Array(5).fill(null),false);$('shop').innerHTML=shopHtml(p.shop);$('options').innerHTML=c.options.map((o,i)=>`<div class="option"><div class="option-letter">${i?'B':'A'}</div><div class="option-action">${esc(o.description)}</div></div>`).join('');$('rating').innerHTML=labels.map(([v,l])=>`<button data-v="${v}" class="${r.choice===v?'selected':''}">${l}</button>`).join('');$('rating').querySelectorAll('button').forEach(b=>b.onclick=()=>{ratings[c.id]={...ratings[c.id],choice:b.dataset.v,revealed:ratings[c.id]?.revealed||false};save();render()});$('note').value=r.note||'';$('note').oninput=()=>{clearTimeout(window.noteTimer);window.noteTimer=setTimeout(save,250)};$('reveal').disabled=!r.choice;$('reveal').textContent=r.revealed?'Evidence revealed':'Reveal search evidence';let e=c.evidence,origin=`A = ${c.options[0].origin} · B = ${c.options[1].origin}`+(c.expertExpected?` · scenario guide = ${c.expertExpected}`:'');$('evidence').className='evidence'+(r.revealed?' show':'');$('evidence').innerHTML=`<div class="origins">${esc(origin)}</div><div class="metrics"><div class="metric">Selection mean<b>${fmt(e.selectionMean)}</b></div><div class="metric">Fresh confirmation<b>${fmt(e.confirmationMean)}</b></div><div class="metric">Confirmation lower bound<b>${fmt(e.confirmationLower)}</b></div><div class="metric">Untouched audit<b>${fmt(e.outerMean)}</b></div><div class="metric">Audit lower bound<b>${fmt(e.outerLower)}</b></div><div class="metric">Independent full plan<b>${fmt(e.independentFull)}</b></div><div class="metric">First action + raw policy<b>${fmt(e.independentRootOnly)}</b></div><div class="metric">Later-tree lift<b>${fmt(e.continuationLift)}</b></div><div class="metric">First-battle gain<b>${fmt(e.outerFirstBattle)}</b></div><div class="metric">Five-battle gain<b>${fmt(e.outerHorizonBattle)}</b></div><div class="metric">Internal gate<b>${e.accepted===true?'PASS':e.accepted===false?'REJECT':'comparison'}</b></div><div class="metric">Independent gate<b>${e.independentlyConfirmed===true?'PASS':e.independentlyConfirmed===false?'REJECT':'—'}</b></div></div><div class="traces"><div class="trace"><b>Search rollouts</b><pre>${esc(battleSeq(c.traces?.search))}</pre></div><div class="trace"><b>Policy rollouts</b><pre>${esc(battleSeq(c.traces?.policy))}</pre></div></div>`;$('reveal').onclick=()=>{ratings[c.id]={...ratings[c.id],revealed:true};save();render()};$('prev').disabled=index===0;$('next').disabled=cases.length<2;renderRail();renderProgress()}
$('prev').onclick=()=>{if(index){index--;render()}};$('next').onclick=()=>{let next=cases.findIndex((c,i)=>i>index&&!ratings[c.id]?.choice);if(next<0)next=cases.findIndex(c=>!ratings[c.id]?.choice);index=next<0?(index+1)%cases.length:next;render()};$('export').onclick=()=>{let blob=new Blob([JSON.stringify({schemaVersion:1,exportedAt:new Date().toISOString(),ratings},null,2)],{type:'application/json'}),a=document.createElement('a');a.href=URL.createObjectURL(blob);a.download='sap-search-expert-ratings.json';a.click();URL.revokeObjectURL(a.href)};document.addEventListener('keydown',e=>{if(e.key==='ArrowLeft')$('prev').click();if(e.key==='ArrowRight')$('next').click()});hydrate();
</script></body></html>'''


if __name__ == "__main__":
    main()
