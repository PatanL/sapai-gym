"""Lossless, JSON-friendly snapshots for inspecting entity-policy matches."""

from __future__ import annotations

import html
import json
from pathlib import Path

from .action_schema import ActionType, AutoregressiveAction, NO_TARGET


def short_name(name: str) -> str:
    for prefix in ("pet-", "food-", "status-"):
        if name.startswith(prefix):
            name = name[len(prefix) :]
            break
    return name.replace("-", " ").title()


def pet_snapshot(pet) -> dict | None:
    if pet.name == "pet-none":
        return None
    status = None if pet.status in {"none", "status-none"} else pet.status
    return {
        "kind": "pet",
        "name": pet.name,
        "label": short_name(pet.name),
        "attack": int(pet.attack),
        "health": int(pet.health),
        "level": int(pet.level),
        "experience": int(pet.experience),
        "tier": int(pet.tier),
        "status": status,
        "statusLabel": short_name(status) if status else None,
        "sellValue": int(pet.sell_value),
        "abilityCounter": int(pet.ability_counter),
        "friendsHurtCounter": int(pet.friends_hurt_counter),
        "trigger": str(pet.ability.get("trigger", "None")),
    }


def food_snapshot(food) -> dict:
    status = None if food.status in {"none", "status-none"} else food.status
    return {
        "kind": "food",
        "name": food.name,
        "label": short_name(food.name),
        "attack": int(food.attack),
        "health": int(food.health),
        "status": status,
        "statusLabel": short_name(status) if status else None,
        "untilEndOfBattle": bool(food.apply_until_end_of_battle),
    }


def team_snapshot(team) -> list[dict | None]:
    return [None if slot.empty else pet_snapshot(slot.pet) for slot in team]


def shop_snapshot(shop) -> list[dict]:
    slots = []
    for index, slot in enumerate(shop):
        if slot.empty:
            item = None
        elif slot.slot_type in {"pet", "levelup"}:
            item = pet_snapshot(slot.obj)
        else:
            item = food_snapshot(slot.obj)
        slots.append(
            {
                "index": index,
                "slotType": slot.slot_type,
                "cost": int(slot.cost),
                "frozen": bool(slot.frozen),
                "rewardGroup": (
                    None if slot.reward_group is None else int(slot.reward_group)
                ),
                "item": item,
            }
        )
    return slots


def player_snapshot(player) -> dict:
    return {
        "turn": int(player.turn),
        "lives": int(player.lives),
        "wins": int(player.wins),
        "gold": int(player.gold),
        "turnEnded": bool(player.turn_ended),
        "lastBattleOutcome": player.last_battle_outcome,
        "shopAttack": int(player.shop.shop_attack),
        "shopHealth": int(player.shop.shop_health),
        "team": team_snapshot(player.team),
        "shop": shop_snapshot(player.shop),
    }


def view_snapshot(core, side: int) -> dict:
    player = core.agent if side == 0 else core.opponent
    opponent_player = core.opponent if side == 0 else core.agent
    previous_opponent = (
        core.previous_opponent_state if side == 0 else core.previous_agent_state
    )
    action_count = (
        core.shop_action_count_agent if side == 0 else core.shop_action_count_opponent
    )
    reorder_count = (
        core.reorder_count_agent if side == 0 else core.reorder_count_opponent
    )
    toggled = (
        core._freeze_toggled_agent
        if side == 0
        else core._freeze_toggled_opponent
    )
    seen_orders = (
        core._seen_team_orders_agent
        if side == 0
        else core._seen_team_orders_opponent
    )
    return {
        "player": player_snapshot(player),
        "opponentScore": {
            "lives": int(opponent_player.lives),
            "wins": int(opponent_player.wins),
        },
        "lastOpponentTeam": (
            None if previous_opponent is None else team_snapshot(previous_opponent)
        ),
        "actionCount": int(action_count),
        "reorderCount": int(reorder_count),
        "freezeToggledSources": [
            index
            for index, slot in enumerate(player.shop)
            if id(slot.obj) in toggled
        ],
        "seenOrderCount": int(len(seen_orders)),
    }


def action_snapshot(player, action: AutoregressiveAction) -> dict:
    action_type = ActionType(action.action_type)
    result = {
        "type": action_type.name,
        "source": int(action.source),
        "target": int(action.target),
        "sourceItem": None,
        "targetItem": None,
        "cost": 0,
    }
    if action_type in {
        ActionType.BUY_PET,
        ActionType.BUY_FOOD,
        ActionType.FREEZE,
        ActionType.UNFREEZE,
    }:
        slot = player.shop[action.source]
        result["cost"] = int(slot.cost) if action_type.name.startswith("BUY") else 0
        result["sourceItem"] = (
            pet_snapshot(slot.obj)
            if slot.slot_type in {"pet", "levelup"}
            else food_snapshot(slot.obj)
        )
    elif action_type in {ActionType.SELL, ActionType.COMBINE, ActionType.MOVE}:
        source_slot = player.team[action.source]
        result["sourceItem"] = (
            None if source_slot.empty else pet_snapshot(source_slot.pet)
        )
    if action_type in {ActionType.BUY_PET, ActionType.BUY_FOOD, ActionType.COMBINE, ActionType.MOVE}:
        if action.target != NO_TARGET:
            target_slot = player.team[action.target]
            result["targetItem"] = (
                None if target_slot.empty else pet_snapshot(target_slot.pet)
            )
    result["description"] = describe_action(result)
    return result


def describe_action(action: dict) -> str:
    action_type = action["type"]
    source = action.get("sourceItem")
    target = action.get("targetItem")
    source_label = source["label"] if source else "empty slot"
    target_label = target["label"] if target else "empty slot"
    if action_type == "END_TURN":
        return "End turn"
    if action_type == "ROLL":
        return "Roll shop"
    if action_type in {"FREEZE", "UNFREEZE"}:
        return f"{action_type.title()} {source_label} in shop slot {action['source']}"
    if action_type == "BUY_PET":
        verb = "combine into" if target else "buy into"
        return (
            f"Buy {source_label} from shop {action['source']} and {verb} "
            f"team {action['target']}"
        )
    if action_type == "BUY_FOOD":
        destination = "team" if action["target"] == NO_TARGET else f"{target_label} in team {action['target']}"
        return f"Buy {source_label} from shop {action['source']} for {destination}"
    if action_type == "SELL":
        return f"Sell {source_label} from team {action['source']}"
    if action_type == "COMBINE":
        return f"Combine {source_label} from team {action['source']} into team {action['target']}"
    if action_type == "MOVE":
        return f"Swap team {action['source']} ({source_label}) with team {action['target']} ({target_label})"
    return action_type.replace("_", " ").title()


def _item_signature(item):
    if item is None:
        return None
    keys = (
        "name",
        "attack",
        "health",
        "level",
        "experience",
        "status",
        "sellValue",
        "abilityCounter",
        "friendsHurtCounter",
    )
    return tuple(item.get(key) for key in keys)


def state_changes(before: dict, after: dict) -> list[str]:
    old, new = before["player"], after["player"]
    changes = []
    for key, label in (("gold", "Gold"), ("lives", "Hearts"), ("wins", "Wins")):
        if old[key] != new[key]:
            changes.append(f"{label}: {old[key]} → {new[key]}")
    for index, (old_pet, new_pet) in enumerate(zip(old["team"], new["team"])):
        if _item_signature(old_pet) == _item_signature(new_pet):
            continue
        old_text = "empty" if old_pet is None else _pet_text(old_pet)
        new_text = "empty" if new_pet is None else _pet_text(new_pet)
        changes.append(f"Team {index}: {old_text} → {new_text}")
    old_shop = {slot["index"]: slot for slot in old["shop"]}
    new_shop = {slot["index"]: slot for slot in new["shop"]}
    for index in sorted(set(old_shop) | set(new_shop)):
        old_slot, new_slot = old_shop.get(index), new_shop.get(index)
        old_sig = _shop_signature(old_slot)
        new_sig = _shop_signature(new_slot)
        if old_sig == new_sig:
            continue
        changes.append(
            f"Shop {index}: {_shop_text(old_slot)} → {_shop_text(new_slot)}"
        )
    if old["shopAttack"] != new["shopAttack"] or old["shopHealth"] != new["shopHealth"]:
        changes.append(
            "Future shop buff: "
            f"+{old['shopAttack']}/+{old['shopHealth']} → "
            f"+{new['shopAttack']}/+{new['shopHealth']}"
        )
    return changes


def _pet_text(pet: dict) -> str:
    status = f" {pet['statusLabel']}" if pet.get("statusLabel") else ""
    return (
        f"{pet['label']} {pet['attack']}/{pet['health']} "
        f"L{pet['level']}+{pet['experience']}{status}"
    )


def _shop_signature(slot):
    if slot is None:
        return None
    return (
        slot["slotType"],
        slot["cost"],
        slot["frozen"],
        slot["rewardGroup"],
        _item_signature(slot["item"]),
    )


def _shop_text(slot) -> str:
    if slot is None or slot["item"] is None:
        return "empty"
    item = slot["item"]
    text = _pet_text(item) if item["kind"] == "pet" else item["label"]
    flags = []
    if slot["frozen"]:
        flags.append("frozen")
    if slot["rewardGroup"] is not None:
        flags.append(f"discover #{slot['rewardGroup']}")
    suffix = f" ({', '.join(flags)})" if flags else ""
    return f"{text} ${slot['cost']}{suffix}"


def audit_state(action: dict | None, before: dict, after: dict) -> list[dict]:
    warnings = []
    player = after["player"]
    if len(player["shop"]) > 10:
        warnings.append({"level": "error", "text": "Shop exceeds ten slots"})
    groups = {}
    for slot in player["shop"]:
        group = slot["rewardGroup"]
        if group is not None:
            groups.setdefault(group, []).append(slot)
    before_groups = {
        slot["rewardGroup"]
        for slot in before["player"]["shop"]
        if slot["rewardGroup"] is not None
    }
    for group, slots in groups.items():
        names = [slot["item"]["name"] for slot in slots]
        if len(slots) not in {1, 2}:
            warnings.append(
                {"level": "error", "text": f"Discover #{group} has {len(slots)} choices"}
            )
        elif len(slots) == 1 and action is not None and group not in before_groups:
            warnings.append(
                {
                    "level": "error",
                    "text": f"New discover #{group} was generated with only one choice",
                }
            )
        elif len(slots) == 2 and len(set(names)) != 2:
            warnings.append(
                {"level": "error", "text": f"Discover #{group} choices are not distinct"}
            )
    if action is None:
        return warnings
    gold_delta = player["gold"] - before["player"]["gold"]
    expected = {
        "ROLL": -1,
        "FREEZE": 0,
        "UNFREEZE": 0,
        "COMBINE": 0,
        "MOVE": 0,
        "END_TURN": 0,
        "BUY_PET": -int(action.get("cost", 0)),
        "BUY_FOOD": -int(action.get("cost", 0)),
    }.get(action["type"])
    if action["type"] == "SELL" and action.get("sourceItem"):
        source = action["sourceItem"]
        pig_bonus = source["level"] if source["name"] == "pet-pig" else 0
        expected = int(source["sellValue"]) + int(pig_bonus)
    if expected is not None and gold_delta != expected:
        warnings.append(
            {
                "level": "error",
                "text": f"Gold changed by {gold_delta}; expected {expected} for {action['type']}",
            }
        )
    if action["type"] == "END_TURN" and before["player"]["gold"] > 0:
        level = "note" if before.get("actionCount", 0) >= 50 else "error"
        warnings.append(
            {
                "level": level,
                "text": f"Ended with {before['player']['gold']} unspent gold",
            }
        )
    if action["type"] == "SELL":
        source = int(action["source"])
        if after["player"]["team"][source] is not None:
            warnings.append(
                {
                    "level": "error",
                    "text": f"Sold team slot {source} is still occupied",
                }
            )
    if action["type"] in {"FREEZE", "UNFREEZE"}:
        source = int(action["source"])
        slots = {slot["index"]: slot for slot in after["player"]["shop"]}
        expected_frozen = action["type"] == "FREEZE"
        if source not in slots or slots[source]["frozen"] != expected_frozen:
            warnings.append(
                {
                    "level": "error",
                    "text": (
                        f"{action['type'].title()} did not leave shop slot "
                        f"{source} frozen={expected_frozen}"
                    ),
                }
            )
    return warnings


def render_game_html(trace: dict, output: Path) -> None:
    """Write a self-contained interactive match inspector."""

    payload = json.dumps(trace, separators=(",", ":"), ensure_ascii=True)
    payload = payload.replace("</", "<\\/")
    title = html.escape(trace["metadata"]["title"])
    document = _HTML_TEMPLATE.replace("__TITLE__", title).replace(
        "__TRACE_JSON__", payload
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(document)


_HTML_TEMPLATE = r'''<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>__TITLE__</title>
<style>
@font-face{font-family:Lapsus;src:url('sap-fonts/LapsusPro-Regular.otf')}@font-face{font-family:Lapsus;src:url('sap-fonts/LapsusPro-Bold.otf');font-weight:700 1000}
:root{color-scheme:light dark;--bg:light-dark(#f3f0e7,#111513);--panel:light-dark(#fffdf7,#1a211d);--panel2:light-dark(#ebe6d9,#252e28);--text:light-dark(#22241f,#edf3ec);--muted:light-dark(#686b61,#a7b0a8);--line:light-dark(#d2cbbc,#3c4840);--gold:#e5a816;--green:#4da56b;--red:#d85c55;--blue:#4f86d9;--purple:#9a6bdd;--orange:#d77d35;--shadow:0 10px 35px #0002}
 *{box-sizing:border-box}body{margin:0;background:var(--bg);color:var(--text);font:14px/1.4 ui-sans-serif,system-ui,-apple-system,"Segoe UI",sans-serif}.app{max-width:1800px;margin:auto;padding:18px}.top{display:flex;gap:16px;align-items:flex-start;justify-content:space-between;margin-bottom:14px}.title{font-size:23px;font-weight:800;letter-spacing:-.02em}.meta{color:var(--muted);font-size:12px;margin-top:3px}.controls{display:flex;gap:7px;align-items:center;flex-wrap:wrap}.controls button,.turns button,.mode button{border:1px solid var(--line);background:var(--panel);color:var(--text);padding:7px 11px;border-radius:8px;cursor:pointer}.controls button:hover,.turns button:hover,.mode button:hover{border-color:var(--blue)}.step{min-width:110px;text-align:center;font-variant-numeric:tabular-nums}.layout{display:grid;grid-template-columns:minmax(190px,225px) 1fr;gap:14px}.rail,.main{background:var(--panel);border:1px solid var(--line);border-radius:13px;box-shadow:var(--shadow)}.rail{padding:10px;max-height:calc(100vh - 105px);overflow:auto;position:sticky;top:10px}.turn-label{font-weight:800;margin:10px 4px 5px;color:var(--muted);text-transform:uppercase;font-size:11px;letter-spacing:.08em}.rail button{display:block;width:100%;text-align:left;background:transparent;color:var(--text);border:0;border-left:3px solid transparent;padding:7px 8px;margin:1px 0;cursor:pointer;border-radius:4px;font-size:12px}.rail button:hover{background:var(--panel2)}.rail button.active{background:var(--panel2);border-left-color:var(--blue);font-weight:700}.rail button.battle{border-left-color:var(--orange)}.main{padding:15px;min-width:0}.headline{display:flex;align-items:flex-start;justify-content:space-between;gap:16px;border-bottom:1px solid var(--line);padding-bottom:12px}.action-title{font-size:20px;font-weight:800}.action-sub{color:var(--muted);margin-top:3px}.score{display:flex;gap:13px;white-space:nowrap}.score b{font-size:17px}.gold{color:var(--gold)}.mode{display:flex;gap:5px;margin:11px 0}.mode button.active{background:var(--blue);border-color:var(--blue);color:white}.section-title{font-weight:800;margin:15px 0 7px}.slots{display:grid;grid-template-columns:repeat(5,minmax(110px,1fr));gap:8px}.shop{grid-template-columns:repeat(10,minmax(88px,1fr));gap:5px}.matchup-headings{display:grid;grid-template-columns:1fr 46px 1fr;gap:7px;margin-top:13px}.matchup-headings .section-title{margin:0}.matchup-headings .opponent-heading{text-align:right}.matchup{display:grid;grid-template-columns:minmax(0,1fr) 46px minmax(0,1fr);gap:7px;align-items:stretch}.matchup .slots{grid-template-columns:repeat(5,minmax(88px,1fr));gap:5px}.versus{display:flex;align-items:center;justify-content:center;font-weight:1000;font-size:16px;color:var(--orange);border-left:1px solid var(--line);border-right:1px solid var(--line)}.card{position:relative;min-height:118px;border:1px solid var(--line);border-radius:10px;padding:9px;background:var(--panel2);overflow:hidden}.card.empty{opacity:.45;background:transparent;border-style:dashed}.card.changed{outline:2px solid var(--orange);outline-offset:-2px}.card.selected{outline:3px solid var(--blue);outline-offset:-3px}.card.target{outline:3px solid var(--green);outline-offset:-3px}.card.frozen:after{content:"FROZEN";position:absolute;right:-22px;top:13px;transform:rotate(40deg);background:#4da7c7;color:white;padding:2px 24px;font-size:9px;font-weight:900}.slot-index{font-size:10px;color:var(--muted);text-transform:uppercase}.name{font-size:15px;font-weight:850;margin-top:4px}.stats{font-size:20px;font-weight:900;margin:5px 0}.detail{font-size:11px;color:var(--muted)}.status{display:inline-block;color:var(--purple);font-weight:800}.reward{position:absolute;left:7px;bottom:6px;font-size:10px;background:var(--purple);color:#fff;padding:2px 6px;border-radius:4px}.price{position:absolute;right:7px;bottom:5px;color:var(--gold);font-weight:900}.front{color:var(--red);font-weight:900}.changes{margin-top:14px;border-top:1px solid var(--line);padding-top:10px}.change-list{display:flex;gap:6px;flex-wrap:wrap}.change{background:var(--panel2);border-radius:6px;padding:5px 8px;font-size:11px}.warnings{margin-top:9px}.warning{border-left:4px solid var(--red);background:color-mix(in srgb,var(--red) 12%,transparent);padding:7px 9px;margin:5px 0}.warning.note{border-left-color:var(--blue)}.diagnostics{display:grid;grid-template-columns:180px 1fr;gap:12px;margin-top:14px;border-top:1px solid var(--line);padding-top:10px}.metric{font-size:12px;color:var(--muted)}.metric b{display:block;color:var(--text);font-size:16px}.probs{display:grid;gap:4px}.prob{display:grid;grid-template-columns:92px 1fr 45px;gap:7px;align-items:center;font-size:11px}.bar{height:7px;background:var(--line);border-radius:8px;overflow:hidden}.bar i{display:block;height:100%;background:var(--blue)}.battle-result{text-align:center;font-size:24px;font-weight:900;padding:13px}.kbd{color:var(--muted);font-size:11px}.hidden{display:none!important}
.card{min-height:128px}.name{max-width:68%}.detail{max-width:72%}.sprite{position:absolute;right:7px;top:27px;width:58px;height:58px;object-fit:contain;filter:drop-shadow(0 3px 3px #0004)}.equipment-sprite{position:absolute;right:42px;top:78px;width:27px;height:27px;object-fit:contain;background:var(--panel);border:1px solid var(--line);border-radius:50%;padding:2px}
.card.pet-card{min-height:232px;padding:7px 8px 26px;overflow:visible}.pet-card .slot-index{position:absolute;left:8px;top:7px;z-index:4}.pet-card .name{position:absolute;left:8px;top:24px;max-width:calc(100% - 16px);width:calc(100% - 16px);text-align:center;font-size:14px;z-index:4;text-shadow:0 1px var(--panel)}.pet-stage{position:relative;height:143px;margin-top:35px}.pet-card .sprite{position:absolute;left:50%;right:auto;top:43px;width:96px;height:96px;transform:translateX(-50%);object-fit:contain;filter:drop-shadow(0 5px 4px #0005)}#team .pet-card .sprite{transform:translateX(-50%) scaleX(-1)}.level-display{position:absolute;left:50%;top:0;width:72px;height:43px;transform:translateX(-50%);z-index:6;display:flex;flex-direction:column;align-items:center;justify-content:flex-start}.level-number{width:44px;height:27px;object-fit:contain;filter:drop-shadow(0 2px 1px #0005)}.xp-track{display:flex;width:58px;height:11px;gap:3px;margin-top:1px;padding:2px 4px;background:#24180e;border:2px solid #080706;border-radius:8px;box-shadow:0 1px 1px #0005}.xp-segment{flex:1;background:#6b3a05;border-radius:3px;box-shadow:inset 0 1px #a7670c}.xp-segment.filled{background:#ffd21f;box-shadow:inset 0 1px #fff27a}.pet-card .equipment-sprite{position:absolute;right:4px;top:43px;width:38px;height:38px;z-index:6;object-fit:contain;background:#fff;border:2px solid #151515;border-radius:50%;padding:3px;filter:drop-shadow(0 2px 2px #0005)}.pet-card .equipment-label{position:absolute;right:0;top:81px;max-width:64px;text-align:center;font-size:8px;font-weight:850;color:var(--purple);z-index:6}.stat-badge{position:absolute;bottom:0;width:52px;height:52px;z-index:7}.stat-badge.attack{left:5px}.stat-badge.health{right:5px}.stat-badge img{position:absolute;inset:0;width:100%;height:100%;object-fit:contain;filter:drop-shadow(0 2px 2px #0006)}.stat-badge b{position:absolute;left:50%;top:50%;transform:translate(-50%,-43%);color:#fff;font-size:21px;font-weight:1000;line-height:1;-webkit-text-stroke:1.6px #111;text-shadow:0 2px 1px #000}.pet-detail{font-size:10px;color:var(--muted);text-align:center;line-height:1.25;min-height:29px}.pet-card .status{display:none}.pet-card .price{bottom:5px}.pet-card .reward{bottom:5px}.pet-card.frozen:after{z-index:8}.shop .card{min-height:174px}.shop .pet-card{min-height:174px;padding-bottom:20px;overflow:hidden}.shop .pet-card .name{font-size:12px;top:20px}.shop .pet-stage{height:105px;margin-top:28px}.shop .pet-card .sprite{top:34px;width:68px;height:68px}.shop .level-display{width:55px;height:33px}.shop .level-number{width:34px;height:21px}.shop .xp-track{width:46px;height:9px;padding:1px 3px;gap:2px;border-width:1px}.shop .pet-card .equipment-sprite{top:33px;width:29px;height:29px}.shop .pet-card .equipment-label{display:none}.shop .stat-badge{width:39px;height:39px}.shop .stat-badge b{font-size:16px;-webkit-text-stroke:1.2px #111}.shop .pet-detail{font-size:8px;min-height:18px;white-space:nowrap;overflow:hidden;text-overflow:ellipsis}.shop .reward{font-size:8px;left:3px;bottom:3px;padding:1px 3px}.shop .price{font-size:10px;right:4px;bottom:3px}
.card.pet-card{min-height:250px}.pet-stage{height:161px}.stat-badge{width:48px;height:48px}.stat-badge b{font-size:20px}.shop .card,.shop .pet-card{min-height:187px}.shop .pet-stage{height:118px}.shop .pet-card .sprite{top:33px;width:66px;height:66px;transform:translateX(-50%) scaleX(-1)}.shop .stat-badge{width:36px;height:36px}.shop .stat-badge b{font-size:15px}
.headline{align-items:center;min-height:66px}.headline-right{margin-left:auto;display:flex;align-items:center;justify-content:flex-end;gap:20px}.action-summary{text-align:right;max-width:480px}.action-meta{margin-top:5px;color:var(--muted);font-size:11px;font-weight:700}.score,.opponent-score{display:flex;align-items:center;gap:5px;flex-wrap:wrap;white-space:normal}.score{padding-left:14px}.opponent-score{justify-content:flex-end;padding-left:13px}.hud-counter{position:relative;display:flex;align-items:center;justify-content:flex-end;height:44px;min-width:70px;margin-left:15px;padding:2px 10px 1px 32px;background:#fff;border:4px solid #111;border-radius:8px;color:#111;box-shadow:0 2px 0 #0003;font-family:Lapsus,ui-sans-serif,sans-serif}.hud-counter.wide{min-width:90px}.hud-counter img,.hud-counter .hud-glyph{position:absolute;left:-20px;top:50%;width:50px;height:50px;transform:translateY(-50%);object-fit:contain;filter:drop-shadow(0 2px 0 #0004)}.hud-counter .hud-glyph{display:flex;align-items:center;justify-content:center;font-size:39px;line-height:1}.hud-counter b{font-family:Lapsus,ui-sans-serif,sans-serif;font-size:27px;font-weight:1000;line-height:1;color:#111}.opponent-tag{padding:4px 3px 2px;color:var(--muted);font:900 11px/1 Lapsus,ui-sans-serif,sans-serif;letter-spacing:.08em}.hud-counter.opponent{height:38px;min-width:61px;padding-left:27px;margin-left:13px;border-width:3px}.hud-counter.opponent img,.hud-counter.opponent .hud-glyph{left:-17px;width:42px;height:42px;font-size:32px}.hud-counter.opponent b{font-size:23px}
@media(max-width:1100px){.layout{grid-template-columns:1fr}.rail{position:static;max-height:190px}.matchup{grid-template-columns:1fr 34px 1fr}.matchup .slots{grid-template-columns:repeat(5,minmax(74px,1fr))}.pet-card .sprite{width:86px;height:86px}.diagnostics{grid-template-columns:1fr}.top{display:block}.controls{margin-top:10px}.headline{display:block}.score{margin-top:8px}}@media(max-width:760px){.matchup-headings{display:none}.matchup{grid-template-columns:1fr}.matchup .slots,.slots,.shop{grid-template-columns:repeat(2,minmax(120px,1fr))}.versus{min-height:34px;border:0;border-top:1px solid var(--line);border-bottom:1px solid var(--line)}}
</style>
</head>
<body>
<div class="app">
 <div class="top"><div><div class="title" id="title"></div><div class="meta" id="meta"></div></div><div class="controls"><button id="prev">← Previous</button><span class="step" id="step"></span><button id="next">Next →</button><span class="kbd">←/→ step · B before/after</span></div></div>
 <div class="layout"><aside class="rail" id="rail"></aside><main class="main">
  <div class="headline"><div class="score" id="score"></div><div class="headline-right"><div class="opponent-score" id="opponentScore"></div><div class="action-summary"><div class="action-title" id="actionTitle"></div><div class="action-sub" id="actionSub"></div><div class="action-meta" id="actionMeta"></div></div></div></div>
  <div class="mode" id="mode"><button data-mode="after" class="active">After action</button><button data-mode="before">Before action</button></div>
  <div id="battleResult" class="battle-result hidden"></div>
  <div class="matchup-headings"><div class="section-title" id="teamTitle">Your team — back → <span class="front">front</span></div><div></div><div class="section-title opponent-heading" id="opponentTitle"><span class="front">front</span> → back — Last observed opponent</div></div>
  <div class="matchup"><div class="slots" id="team"></div><div class="versus">VS</div><div class="slots" id="opponent"></div></div>
  <div id="shopWrap"><div class="section-title">Shop — physical slots 0–9</div><div class="slots shop" id="shop"></div></div>
  <div class="changes"><div class="section-title">Exact state changes</div><div class="change-list" id="changes"></div><div class="warnings" id="warnings"></div></div>
  <div class="diagnostics" id="diagnostics"><div class="metric" id="modelMetrics"></div><div class="probs" id="probabilities"></div></div>
 </main></div>
</div>
<script id="trace" type="application/json">__TRACE_JSON__</script>
<script>
const trace=JSON.parse(document.getElementById('trace').textContent),frames=trace.frames;let index=0,mode='after';
const $=id=>document.getElementById(id),esc=s=>String(s??'').replace(/[&<>"']/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));
const petSpriteOverrides={'pet-zombie-cricket':'CricketToken','pet-dirty-rat':'RatToken','pet-zombie-fly':'FlyToken'};
const foodSpriteOverrides={'food-sleeping-pill':'Pill','food-cupcake':'Muffin','food-cake':'BirthdayCake','food-better-apple':'Apple2','food-best-apple':'Apple3','food-better-milk':'Milk2','food-best-milk':'Milk3'};
const statusSpriteNames={'status-honey-bee':'Honey','status-bone-attack':'MeatBone','status-garlic-armor':'Garlic','status-splash-attack':'Chili','status-melon-armor':'Melon','status-extra-life':'Mushroom','status-steak-attack':'Steak','status-poison-attack':'Peanut','status-cake':'BirthdayCake','status-bread':'Bread','status-coconut-shield':'Coconut','status-weak':'Weak'};
function pascal(name){return name.split('-').map(x=>x.charAt(0).toUpperCase()+x.slice(1)).join('')}
function spritePath(item){if(!item)return null;if(item.kind==='pet')return `sap-sprites/Pets/${petSpriteOverrides[item.name]||pascal(item.name.slice(4))}.png`;return `sap-sprites/Food/${foodSpriteOverrides[item.name]||pascal(item.name.slice(5))}.png`}
function statusSprite(item){return item?.status&&statusSpriteNames[item.status]?`sap-sprites/Food/${statusSpriteNames[item.status]}.png`:null}
function itemSig(x){return x?JSON.stringify([x.name,x.attack,x.health,x.level,x.experience,x.status,x.sellValue,x.abilityCounter,x.friendsHurtCounter]):''}
function xpBar(item){let total=item.level===1?2:item.level===2?3:0;if(!total)return '';let segments=Array.from({length:total},(_,i)=>`<i class="xp-segment${i<item.experience?' filled':''}"></i>`).join('');return `<span class="xp-track" title="${item.experience}/${total} experience">${segments}</span>`}
function card(item,slot,kind,changed,selected,target){let pet=item?.kind==='pet',cls='card'+(pet?' pet-card':'')+(!item?' empty':'')+(changed?' changed':'')+(selected?' selected':'')+(target?' target':'')+(slot&&slot.frozen?' frozen':'');if(!item)return `<div class="${cls}"><div class="slot-index">${kind} ${slot?.index??''}</div><div class="name">Empty</div></div>`;let details=pet?`sell ${item.sellValue}${item.abilityCounter?' · used '+item.abilityCounter:''}`:`${item.attack||item.health?`+${item.attack}/+${item.health}`:'effect'}${item.untilEndOfBattle?' until battle':''}`,sprite=spritePath(item),equipment=statusSprite(item),tooltip=pet?`${item.label} — ${item.attack}/${item.health}, level ${item.level} +${item.experience}${item.statusLabel?' · '+item.statusLabel:''}`:item.label,decorative='alt="" aria-hidden="true"';if(pet)return `<div class="${cls}" title="${esc(tooltip)}"><div class="slot-index">${kind} ${slot?.index??''}</div><div class="name">${esc(item.label)}</div><div class="pet-stage"><div class="level-display"><img class="level-number" src="sap-icons/l${item.level}.png" alt="Level ${item.level}">${xpBar(item)}</div><img class="sprite" src="${sprite}" ${decorative} onerror="this.style.display='none'">${equipment?`<img class="equipment-sprite" src="${equipment}" ${decorative} title="${esc(item.statusLabel)}" onerror="this.style.display='none'"><span class="equipment-label">${esc(item.statusLabel)}</span>`:''}<span class="stat-badge attack" aria-label="${item.attack} attack"><img src="sap-icons/fist.png" ${decorative}><b>${item.attack}</b></span><span class="stat-badge health" aria-label="${item.health} health"><img src="sap-icons/heart.png" ${decorative}><b>${item.health}</b></span></div><div class="pet-detail">${details}</div>${slot?.rewardGroup!=null?`<span class="reward">discover #${slot.rewardGroup}</span>`:''}${slot&&Number.isFinite(slot.cost)?`<span class="price">${slot.cost}g</span>`:''}</div>`;return `<div class="${cls}" title="${esc(tooltip)}"><div class="slot-index">${kind} ${slot?.index??''}</div><img class="sprite" src="${sprite}" ${decorative} onerror="this.style.display='none'"><div class="name">${esc(item.label)}</div><div class="detail">${details}</div>${item.statusLabel?`<div class="status">${esc(item.statusLabel)}</div>`:''}${equipment?`<img class="equipment-sprite" src="${equipment}" ${decorative} onerror="this.style.display='none'">`:''}${slot?.rewardGroup!=null?`<span class="reward">discover #${slot.rewardGroup}</span>`:''}${slot&&Number.isFinite(slot.cost)?`<span class="price">${slot.cost}g</span>`:''}</div>`}
function teamCards(team,other,action,reverse=false){let indices=Array.from({length:team.length},(_,i)=>i);if(reverse)indices.reverse();return indices.map(i=>{let p=team[i];return card(p,{index:i},i===0?'front':'team',itemSig(p)!==itemSig(other?.[i]),mode==='before'&&action?.source===i&&['SELL','COMBINE','MOVE'].includes(action.type),action?.target===i&&['BUY_PET','BUY_FOOD','COMBINE','MOVE'].includes(action.type))}).join('')}
function shopCards(shop,other,action){let a=Object.fromEntries((shop||[]).map(s=>[s.index,s])),b=Object.fromEntries((other||[]).map(s=>[s.index,s]));let out=[];for(let i=0;i<10;i++){let s=a[i],o=b[i],changed=JSON.stringify(s||null)!==JSON.stringify(o||null),selected=mode==='before'&&action?.source===i&&['BUY_PET','BUY_FOOD','FREEZE','UNFREEZE'].includes(action.type);out.push(card(s?.item,s||{index:i},s?.slotType||'shop',changed,selected,false))}return out.join('')}
function score(state){let player=state.player;return `<span class="hud-counter"><img src="sap-icons/gold.png" alt="Gold"><b>${player.gold}</b></span><span class="hud-counter"><img src="sap-icons/heart.png" alt="Hearts"><b>${player.lives}</b></span><span class="hud-counter"><img src="sap-ui/hourglass-twemoji.png" alt="Turn"><b>${player.turn}</b></span><span class="hud-counter wide"><span class="hud-glyph" aria-label="Wins">🏆</span><b>${player.wins}/10</b></span>`}
function opponentScore(state){let opp=state.opponentScore||{};return `<span class="opponent-tag">OPP</span><span class="hud-counter opponent"><img src="sap-icons/heart.png" alt="Opponent hearts"><b>${opp.lives??'?'}</b></span><span class="hud-counter opponent"><span class="hud-glyph" aria-label="Opponent wins">🏆</span><b>${opp.wins??'?'}</b></span>`}
function renderRail(){let lastTurn=null,html='';frames.forEach((f,i)=>{if(f.turn!==lastTurn){lastTurn=f.turn;html+=`<div class="turn-label">Turn ${f.turn}</div>`}html+=`<button data-index="${i}" class="${i===index?'active ':''}${f.phase==='battle'?'battle':''}">${f.phase==='battle'?'⚔ ':''}${esc(f.label)}</button>`});$('rail').innerHTML=html;$('rail').querySelectorAll('button').forEach(b=>b.onclick=()=>{index=+b.dataset.index;render()});$('rail').querySelector('.active')?.scrollIntoView({block:'nearest'})}
function render(){let f=frames[index],isBattle=f.phase==='battle',state=f[mode]||f.after||f.before,other=(mode==='after'?f.before:f.after)||state,player=state.player,toggled=(state.freezeToggledSources||[]).join(',');$('title').textContent=trace.metadata.title;$('meta').textContent=`seed ${trace.metadata.seed} · deterministic CPU · ${trace.metadata.simulatorRevision}`;$('step').textContent=`${index+1} / ${frames.length}`;$('actionTitle').textContent=f.label;$('actionSub').textContent=f.action?.description||f.subtitle||'';$('actionMeta').textContent=`actions ${state.actionCount??0}/50 · moves ${state.reorderCount??0}/2${toggled?' · toggled '+toggled:''}`;$('score').innerHTML=score(state);$('opponentScore').innerHTML=opponentScore(state);$('mode').classList.toggle('hidden',isBattle||!f.before||!f.after);$('battleResult').classList.toggle('hidden',!isBattle);$('shopWrap').classList.toggle('hidden',isBattle);$('diagnostics').classList.toggle('hidden',isBattle||!f.policy);if(isBattle){$('battleResult').textContent=f.battle.resultLabel;$('teamTitle').innerHTML='Your battle board — back → <span class="front">front</span>';$('opponentTitle').innerHTML='<span class="front">front</span> → back — Opponent battle board';$('team').innerHTML=teamCards(f.battle.viewerTeam,[],null,true);$('opponent').innerHTML=teamCards(f.battle.opponentTeam,[],null,false)}else{$('teamTitle').innerHTML='Your team — back → <span class="front">front</span>';$('opponentTitle').innerHTML='<span class="front">front</span> → back — Last observed opponent';$('team').innerHTML=teamCards(player.team,other.player.team,f.action,true);$('opponent').innerHTML=teamCards(state.lastOpponentTeam||Array(5).fill(null),other.lastOpponentTeam||[],null,false);$('shop').innerHTML=shopCards(player.shop,other.player.shop,f.action)}$('changes').innerHTML=(f.changes?.length?f.changes:['No state mutation']).map(x=>`<span class="change">${esc(x)}</span>`).join('');$('warnings').innerHTML=(f.warnings||[]).map(w=>`<div class="warning ${w.level}">${esc(w.text)}</div>`).join('');if(f.policy){$('modelMetrics').innerHTML=`Value estimate<b>${f.policy.value.toFixed(3)}</b>Chosen probability<b>${(100*f.policy.chosenProbability).toFixed(1)}%</b>Entropy<b>${f.policy.entropy.toFixed(3)}</b>`;$('probabilities').innerHTML=f.policy.typeProbabilities.map(p=>`<div class="prob"><span>${esc(p.label)}</span><span class="bar"><i style="width:${100*p.probability}%"></i></span><span>${(100*p.probability).toFixed(1)}%</span></div>`).join('')}renderRail();$('prev').disabled=index===0;$('next').disabled=index===frames.length-1}
$('prev').onclick=()=>{if(index>0){index--;render()}};$('next').onclick=()=>{if(index<frames.length-1){index++;render()}};document.querySelectorAll('#mode button').forEach(b=>b.onclick=()=>{mode=b.dataset.mode;document.querySelectorAll('#mode button').forEach(x=>x.classList.toggle('active',x===b));render()});document.addEventListener('keydown',e=>{if(e.key==='ArrowLeft')$('prev').click();if(e.key==='ArrowRight')$('next').click();if(e.key.toLowerCase()==='b'&&!$('mode').classList.contains('hidden')){mode=mode==='after'?'before':'after';document.querySelectorAll('#mode button').forEach(x=>x.classList.toggle('active',x.dataset.mode===mode));render()}});render();
</script>
</body>
</html>'''
