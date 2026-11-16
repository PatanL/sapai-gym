"""Structured Turtle Pack guide teacher for synthetic imitation trajectories."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
from pathlib import Path
import weakref

import numpy as np

from .action_schema import (
    ActionType,
    AutoregressiveAction,
    NO_TARGET,
)


GUIDE_VERSION = "turtle-p1-guide-v12-20260904"
GUIDE_PATH = Path(__file__).resolve().parents[2] / "TURTLE_P1_GUIDE.md"
GUIDE_SHA256 = hashlib.sha256(GUIDE_PATH.read_bytes()).hexdigest()
TEACHER_SHA256 = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()


SELL_PETS = {
    "pet-beaver",
    "pet-duck",
    "pet-pig",
    "pet-pigeon",
}
SUMMON_PETS = {
    "pet-cricket",
    "pet-horse",
    "pet-spider",
    "pet-sheep",
    "pet-deer",
    "pet-rooster",
    "pet-shark",
    "pet-turkey",
    "pet-fly",
}
SCALERS = {
    "pet-giraffe",
    "pet-rabbit",
    "pet-bison",
    "pet-penguin",
    "pet-monkey",
    "pet-cat",
    "pet-dragon",
}

IMMEDIATE_TEMPO_ITEMS = {
    "pet-deer",
    "pet-skunk",
    "pet-turtle",
    "pet-scorpion",
    "pet-crocodile",
    "pet-leopard",
    "food-pear",
    "food-salad-bowl",
    "food-sushi",
    "food-pizza",
    "food-chili",
    "food-melon",
    "food-steak",
}
SLOW_SCALING_ITEMS = {
    "pet-giraffe",
    "pet-penguin",
    "pet-monkey",
    "pet-cat",
    "pet-dragon",
    "food-canned-food",
}

TURN_HITS = {
    "early": {
        "pet-ant",
        "pet-beaver",
        "pet-cricket",
        "pet-duck",
        "pet-fish",
        "pet-horse",
        "pet-pig",
        "pet-pigeon",
        "food-apple",
    },
    "early_mid": {
        "pet-crab",
        "pet-flamingo",
        "pet-kangaroo",
        "pet-peacock",
        "pet-snail",
        "pet-spider",
        "pet-swan",
        "food-apple",
        "food-sleeping-pill",
    },
    "mid": {
        "pet-badger",
        "pet-deer",
        "pet-dolphin",
        "pet-giraffe",
        "pet-ox",
        "pet-parrot",
        "pet-rabbit",
        "pet-skunk",
        "pet-turtle",
        "food-garlic",
        "food-salad-bowl",
        "food-sleeping-pill",
    },
    "late_mid": {
        "pet-cow",
        "pet-deer",
        "pet-monkey",
        "pet-shark",
        "pet-skunk",
        "pet-turkey",
        "pet-turtle",
        "food-chocolate",
        "food-pear",
    },
    "late": {
        "pet-cat",
        "pet-cow",
        "pet-crocodile",
        "pet-dragon",
        "pet-fly",
        "pet-leopard",
        "pet-mammoth",
        "pet-monkey",
        "pet-scorpion",
        "pet-skunk",
        "pet-snake",
        "pet-tiger",
        "pet-turtle",
        "food-chili",
        "food-chocolate",
        "food-melon",
        "food-mushroom",
        "food-steak",
    },
}

FRONT_PETS = {
    "pet-badger",
    "pet-deer",
    "pet-elephant",
    "pet-hippo",
    "pet-rhino",
    "pet-scorpion",
    "pet-turtle",
}
BACK_PETS = {
    "pet-cat",
    "pet-dragon",
    "pet-fly",
    "pet-giraffe",
    "pet-monkey",
    "pet-penguin",
    "pet-shark",
    "pet-snake",
    "pet-tiger",
    "pet-turkey",
}

LINE_MEMBERS = {
    "sell-pet / triple economy": SELL_PETS | {"pet-fish", "pet-ant"},
    "big tier-1": {"pet-ant", "pet-cricket", "pet-fish", "pet-swan"},
    "horse-cricket summon tempo": {
        "pet-horse", "pet-cricket", "pet-spider", "pet-sheep", "pet-deer"
    },
    "generic stat-tech": {
        "pet-turtle", "pet-skunk", "pet-deer", "pet-monkey", "pet-scorpion"
    },
    "elephant hurt engine": {
        "pet-elephant", "pet-blowfish", "pet-camel", "pet-wolverine", "pet-deer"
    },
    "shark / faint": {
        "pet-shark", "pet-cricket", "pet-spider", "pet-sheep", "pet-turtle",
        "pet-deer", "pet-rooster", "pet-fly"
    },
    "turkey summon": {
        "pet-turkey", "pet-sheep", "pet-deer", "pet-whale", "pet-rooster",
        "pet-fly"
    },
    "rabbit / cat food": {"pet-rabbit", "pet-cat", "pet-cow", "pet-squirrel"},
    "cow scaling conversion": {"pet-cow", "pet-rabbit", "pet-cat", "pet-squirrel"},
    "dragon cycling": {"pet-dragon"} | SELL_PETS,
    "tiger-mammoth-mushroom": {"pet-tiger", "pet-mammoth"},
    "scam / out-fishing": {
        "pet-skunk", "pet-scorpion", "pet-crocodile", "pet-leopard", "pet-snake"
    },
}

LINE_REQUIRED_HITS = {
    "sell-pet / triple economy": (
        "pet-beaver", "pet-duck", "pet-pig", "pet-pigeon", "pet-fish"
    ),
    "big tier-1": ("copies", "food-apple", "pet-flamingo", "pet-bison"),
    "horse-cricket summon tempo": (
        "pet-horse", "pet-cricket", "pet-spider", "pet-sheep", "pet-deer"
    ),
    "generic stat-tech": (
        "food-pear", "pet-turtle", "pet-skunk", "pet-deer", "food-chocolate"
    ),
    "elephant hurt engine": (
        "pet-elephant", "pet-blowfish", "pet-camel", "pet-wolverine"
    ),
    "shark / faint": ("pet-shark", "pet-deer", "pet-turtle", "pet-fly"),
    "turkey summon": ("pet-turkey", "pet-deer", "pet-rooster", "pet-fly"),
    "rabbit / cat food": ("pet-rabbit", "pet-cow", "pet-cat", "stat-food"),
    "cow scaling conversion": ("pet-cow", "free-milk", "sell-slot"),
    "dragon cycling": ("pet-dragon", "tier-1-cycle", "sell-slot"),
    "tiger-mammoth-mushroom": (
        "pet-mammoth", "food-mushroom", "pet-tiger", "protection"
    ),
    "scam / out-fishing": (
        "pet-skunk", "pet-scorpion", "food-mushroom", "food-steak"
    ),
}


@dataclass(frozen=True)
class GuideDecision:
    action: AutoregressiveAction
    board_class: str
    score: float
    line_name: str
    line_confidence: float
    committed_slots: tuple[int, ...]
    flexible_slots: tuple[int, ...]
    sell_slot: int | None
    required_hits: tuple[str, ...]
    matchup_counters: tuple[str, ...]
    opponent_lives: int
    opponent_wins: int
    tempo_pressure: float


@dataclass
class LinePlan:
    line_name: str = "generic stat-tech"
    confidence: float = 0.15
    committed_slots: tuple[int, ...] = ()
    flexible_slots: tuple[int, ...] = ()
    sell_slot: int | None = None
    required_hits: tuple[str, ...] = LINE_REQUIRED_HITS["generic stat-tech"]
    matchup_counters: tuple[str, ...] = ()
    last_turn: int = 0
    last_signature: tuple = ()
    opponent_lives: int = 6
    opponent_wins: int = 0
    tempo_pressure: float = 0.0


def legal_actions(observation):
    actions = []
    for action_type in np.flatnonzero(observation["type_mask"]):
        for source in np.flatnonzero(observation["source_mask"][action_type]):
            for target in np.flatnonzero(
                observation["target_mask"][action_type, source]
            ):
                actions.append(
                    AutoregressiveAction(
                        int(action_type), int(source), int(target)
                    )
                )
    return actions


class TurtleGuideTeacher:
    """Turn-band guide distilled into a stochastic legal-action teacher.

    The teacher is deliberately a policy prior rather than a mechanics oracle.
    Every selected action still passes through the simulator's exact legality
    masks, and RL remains responsible for improving beyond these heuristics.
    """

    name = GUIDE_VERSION
    version = GUIDE_VERSION
    guide_sha256 = GUIDE_SHA256
    executable_sha256 = TEACHER_SHA256

    def __init__(self, rng: np.random.Generator, temperature=0.35):
        self.rng = rng
        self.temperature = float(temperature)
        self._plans = weakref.WeakKeyDictionary()

    def choose(self, core_env, agent_idx, observation, action_count):
        player = core_env.agent if agent_idx == 0 else core_env.opponent
        opponent_player = (
            core_env.opponent if agent_idx == 0 else core_env.agent
        )
        known_opponent = self._known_opponent_team(core_env, agent_idx)
        plan = self._update_plan(
            core_env,
            agent_idx,
            player,
            known_opponent,
            opponent_player=opponent_player,
        )
        actions = legal_actions(observation)
        board_class = plan.line_name
        scores = np.asarray(
            [
                self._score_action(
                    player,
                    action,
                    action_count,
                    board_class,
                    plan,
                    known_opponent,
                )
                for action in actions
            ],
            dtype=np.float64,
        )
        # Sample only among plausible near-best decisions. This preserves
        # trajectory diversity without teaching obviously dominated actions.
        best = float(scores.max())
        eligible = np.flatnonzero(scores >= best - 2.5)
        local = scores[eligible]
        local = (local - local.max()) / max(self.temperature, 1e-6)
        probabilities = np.exp(local)
        probabilities /= probabilities.sum()
        chosen = int(self.rng.choice(eligible, p=probabilities))
        return GuideDecision(
            actions[chosen],
            board_class,
            float(scores[chosen]),
            plan.line_name,
            plan.confidence,
            plan.committed_slots,
            plan.flexible_slots,
            plan.sell_slot,
            plan.required_hits,
            plan.matchup_counters,
            plan.opponent_lives,
            plan.opponent_wins,
            plan.tempo_pressure,
        )

    @staticmethod
    def _known_opponent_team(core_env, agent_idx):
        # Use only the last publicly observed board. The current opposing shop
        # team is private information and must not leak into teacher labels.
        return (
            core_env.previous_opponent_state
            if agent_idx == 0
            else core_env.previous_agent_state
        )

    def plan_for(self, core_env, agent_idx):
        """Return a copy-safe view of persistent line state for tests/audits."""

        return self._plans.get(core_env, {}).get(agent_idx)

    def _update_plan(
        self,
        core_env,
        agent_idx,
        player,
        known_opponent,
        opponent_player=None,
    ):
        plans = self._plans.setdefault(core_env, {})
        plan = plans.setdefault(agent_idx, LinePlan())
        if opponent_player is None:
            opponent_player = (
                core_env.opponent if agent_idx == 0 else core_env.agent
            )
        plan.opponent_lives = int(opponent_player.lives)
        plan.opponent_wins = int(opponent_player.wins)
        plan.tempo_pressure = float(
            max(0, 3 - plan.opponent_lives)
        )
        signature = tuple(
            (
                index,
                slot.pet.name,
                slot.pet.level,
                slot.pet.status,
                slot.pet.attack,
                slot.pet.health,
            )
            for index, slot in enumerate(player.team)
            if not slot.empty
        )
        scores = self._line_scores(player, known_opponent)
        # Persistence/inertia: a high-confidence line is not discarded merely
        # because one shop action temporarily changes the current board class.
        scores[plan.line_name] = scores.get(plan.line_name, 0.0) + (
            4.0 + 8.0 * plan.confidence
        )
        ranked = sorted(scores.items(), key=lambda item: (item[1], item[0]), reverse=True)
        selected, selected_score = ranked[0]
        runner_up = ranked[1][1] if len(ranked) > 1 else selected_score
        changed = selected != plan.line_name
        if changed:
            plan.line_name = selected
            plan.confidence = min(0.7, 0.25 + max(0.0, selected_score - runner_up) / 20)
        else:
            plan.confidence = min(1.0, plan.confidence + (0.08 if signature != plan.last_signature else 0.02))

        members = LINE_MEMBERS[plan.line_name]
        committed = []
        flexible = []
        for index, slot in enumerate(player.team):
            if slot.empty:
                continue
            pet = slot.pet
            is_committed = (
                pet.name in members
                or pet.level >= 2
                or pet.attack + pet.health >= 16
                or pet.status not in {"none", "status-cake"}
            )
            (committed if is_committed else flexible).append(index)
        plan.committed_slots = tuple(committed)
        plan.flexible_slots = tuple(flexible)
        plan.sell_slot = self._preferred_sell_slot(player, plan)
        plan.required_hits = tuple(LINE_REQUIRED_HITS[plan.line_name])
        plan.matchup_counters = self._matchup_counters(
            player, known_opponent, plan.line_name
        )
        plan.last_turn = player.turn
        plan.last_signature = signature
        return plan

    def _line_scores(self, player, known_opponent):
        pets = [slot.pet for slot in player.team if not slot.empty]
        names = [pet.name for pet in pets]
        name_set = set(names)
        scores = {line: 0.0 for line in LINE_MEMBERS}
        scores["generic stat-tech"] = 3.0 + 0.25 * player.turn
        scores["sell-pet / triple economy"] = (
            5.0 * sum(name in SELL_PETS for name in names)
            + (5.0 if player.turn <= 4 else -0.8 * (player.turn - 4))
        )
        tier_one_strength = max(
            (pet.attack + pet.health for pet in pets if pet.tier == 1),
            default=0,
        )
        scores["big tier-1"] = 0.8 * tier_one_strength + 3.0 * sum(
            pet.tier == 1 and pet.level >= 2 for pet in pets
        )
        scores["horse-cricket summon tempo"] = (
            8.0 * ("pet-horse" in name_set)
            + 8.0 * ("pet-cricket" in name_set)
            + 2.0 * len(name_set & SUMMON_PETS)
        )
        hurt_pairs = self._ordered_pair_count(
            player.team, "pet-elephant", {"pet-blowfish", "pet-camel"}
        )
        scores["elephant hurt engine"] = (
            18.0 * hurt_pairs
            + 7.0 * ("pet-wolverine" in name_set and hurt_pairs > 0)
            + 2.0 * ("pet-deer" in name_set and hurt_pairs > 0)
        )
        scores["shark / faint"] = 8.0 * ("pet-shark" in name_set) + 2.5 * len(
            name_set & SUMMON_PETS
        )
        scores["turkey summon"] = 9.0 * ("pet-turkey" in name_set) + 2.0 * len(
            name_set & SUMMON_PETS
        )
        scores["rabbit / cat food"] = (
            8.0 * ("pet-rabbit" in name_set)
            + 12.0 * ("pet-cat" in name_set)
            + 4.0 * ("pet-cow" in name_set)
        )
        scores["cow scaling conversion"] = (
            12.0 * ("pet-cow" in name_set)
            + 3.0 * len(name_set & SCALERS)
        )
        scores["dragon cycling"] = 16.0 * ("pet-dragon" in name_set) + 1.5 * sum(
            name in SELL_PETS for name in names
        )
        scores["tiger-mammoth-mushroom"] = (
            25.0 * self._has_mammoth_tiger_engine(player.team)
            + 9.0 * ("pet-mammoth" in name_set)
            + 9.0 * ("pet-tiger" in name_set)
            + 8.0 * ({"pet-mammoth", "pet-tiger"} <= name_set)
            + (3.0 if player.turn >= 9 else -5.0)
        )
        scores["scam / out-fishing"] = (
            max(0, player.turn - 8)
            + 3.0 * len(name_set & LINE_MEMBERS["scam / out-fishing"])
            + (5.0 if player.lives <= 2 else 0.0)
        )
        counters = self._matchup_counters(player, known_opponent, "")
        if "rhino-vs-summons" in counters:
            scores["shark / faint"] -= 8.0
            scores["turkey summon"] -= 8.0
        if "skunk-accelerates-mammoth" in counters:
            scores["tiger-mammoth-mushroom"] += 8.0
        if "elephant-blowfish-vs-fragile-engine" in counters:
            scores["tiger-mammoth-mushroom"] -= 7.0
            scores["turkey summon"] -= 4.0
        return scores

    @staticmethod
    def _ordered_pair_count(team, ahead_name, behind_names):
        count = 0
        for index in range(len(team) - 1):
            if team[index].empty or team[index + 1].empty:
                continue
            if (
                team[index].pet.name == ahead_name
                and team[index + 1].pet.name in behind_names
            ):
                count += 1
        return count

    @staticmethod
    def _has_mammoth_tiger_engine(team):
        for index in range(len(team) - 1):
            if team[index].empty or team[index + 1].empty:
                continue
            if (
                team[index].pet.name == "pet-mammoth"
                and team[index].pet.status == "status-extra-life"
                and team[index + 1].pet.name == "pet-tiger"
            ):
                return True
        return False

    @staticmethod
    def _preferred_sell_slot(player, plan):
        candidates = list(plan.flexible_slots)
        if not candidates:
            candidates = [
                index
                for index, slot in enumerate(player.team)
                if not slot.empty and slot.pet.name not in LINE_MEMBERS[plan.line_name]
            ]
        if not candidates:
            return None
        return min(
            candidates,
            key=lambda index: (
                player.team[index].pet.level,
                player.team[index].pet.attack + player.team[index].pet.health,
                index,
            ),
        )

    def _matchup_counters(self, player, known_opponent, line_name):
        if known_opponent is None:
            return ()
        own_pets = [slot.pet for slot in player.team if not slot.empty]
        enemy_pets = [slot.pet for slot in known_opponent if not slot.empty]
        own_names = {pet.name for pet in own_pets}
        enemy_names = {pet.name for pet in enemy_pets}
        counters = []
        if "pet-rhino" in enemy_names and own_names & SUMMON_PETS:
            counters.append("rhino-vs-summons")
        if "pet-skunk" in enemy_names:
            if self._has_mammoth_tiger_engine(player.team):
                counters.append("skunk-accelerates-mammoth")
            elif any(pet.attack + pet.health >= 40 for pet in own_pets):
                counters.append("skunk-vs-tall-stats")
        if self._ordered_pair_count(
            known_opponent, "pet-elephant", {"pet-blowfish"}
        ):
            counters.append("elephant-blowfish-vs-fragile-engine")
        if "pet-crocodile" in enemy_names and "pet-tiger" in own_names:
            counters.append("crocodile-vs-backline-tiger")
        melon_count = sum(pet.status == "status-melon-armor" for pet in own_pets)
        if "pet-leopard" in enemy_names and melon_count < 2:
            counters.append("leopard-vs-weak-melon")
        if "pet-scorpion" in enemy_names and own_pets:
            front = min(
                (
                    (index, slot.pet)
                    for index, slot in enumerate(player.team)
                    if not slot.empty
                ),
                default=None,
            )
            if front is not None and front[1].attack + front[1].health >= 35:
                counters.append("scorpion-vs-tall-front")
        return tuple(sorted(counters))

    @staticmethod
    def classify_board(player):
        pets = [slot.pet for slot in player.team if not slot.empty]
        names = {pet.name for pet in pets}
        if not pets:
            return "desperation / out-fishing board"
        if len(names & SELL_PETS) >= 2 and player.turn <= 4:
            return "sell-pet setup"
        if names & SUMMON_PETS and len(names & SUMMON_PETS) >= 2:
            return "summon pivot"
        if "pet-rabbit" in names or "pet-cat" in names:
            return "food board"
        if names & SCALERS:
            return "scaling board"
        if any(pet.tier == 1 and pet.attack + pet.health >= 12 for pet in pets):
            return "big tier 1"
        return "tempo board"

    @staticmethod
    def _turn_band(turn):
        if turn <= 2:
            return "early"
        if turn <= 4:
            return "early_mid"
        if turn <= 6:
            return "mid"
        if turn <= 8:
            return "late_mid"
        return "late"

    def _score_action(
        self,
        player,
        action,
        action_count,
        board_class,
        plan,
        known_opponent,
    ):
        action_type = ActionType(action.action_type)
        if action_type is ActionType.END_TURN:
            return self._score_end(player, action_count)
        if action_type is ActionType.ROLL:
            return self._score_roll(player, action_count, plan)
        if action_type is ActionType.BUY_PET:
            return self._score_buy_pet(player, action, plan, known_opponent)
        if action_type is ActionType.BUY_FOOD:
            return self._score_buy_food(
                player, action, board_class, plan, known_opponent
            )
        if action_type is ActionType.COMBINE:
            return self._score_combine(
                player,
                player.team[action.target].pet,
                player.team[action.source].pet,
                costs_gold=False,
            )
        if action_type is ActionType.SELL:
            return self._score_sell(player, action.source, plan)
        if action_type is ActionType.FREEZE:
            return self._score_freeze(
                player, action.source, plan, known_opponent
            )
        if action_type is ActionType.UNFREEZE:
            return -5.0
        if action_type is ActionType.MOVE:
            return self._score_move(
                player, action.source, action.target, plan, known_opponent
            )
        return -20.0

    def _score_end(self, player, action_count):
        score = 0.0
        if player.gold == 0:
            score += 16.0
        else:
            score -= 40.0 + 4.0 * player.gold
        if action_count >= 10:
            score += 0.8 * (action_count - 9)
        if action_count >= 18:
            score += 20.0
        if any(
            slot.slot_type == "levelup" and slot.cost <= player.gold
            for slot in player.shop
        ):
            score -= 18.0
        return score

    def _score_roll(self, player, action_count, plan=None):
        if player.gold <= 0:
            return -30.0
        visible_hits = sum(
            self._item_value(player, slot.obj.name, plan) >= 6.0
            and slot.cost <= player.gold
            for slot in player.shop
        )
        score = 5.5 - 2.0 * visible_hits
        if player.gold <= 2:
            # The gold expires, while a roll can reveal a useful freeze.
            score += 3.0
        if player.turn == 2 and player.lives == 6:
            # A deliberately weak turn-two board can activate turn-three
            # Snail without net heart loss after the P1 restore.
            score += 2.5
        if action_count >= 12:
            score -= 0.8 * (action_count - 11)
        return score

    def _score_buy_pet(self, player, action, plan, known_opponent):
        slot = player.shop[action.source]
        pet = slot.obj
        target = player.team[action.target]
        base = self._item_value(player, pet.name, plan, known_opponent)
        if slot.slot_type == "levelup":
            base += 13.0 + 1.5 * pet.tier
        if not target.empty:
            return base + self._score_combine(
                player, target.pet, pet, costs_gold=True
            )

        same_name = [
            team_slot.pet
            for team_slot in player.team
            if not team_slot.empty and team_slot.pet.name == pet.name
        ]
        if same_name:
            base += 5.0
            # On even turns, holding a separate copy preserves an odd-turn
            # discover rather than immediately cashing it.
            if player.turn % 2 == 0:
                base += 3.5
        elif player.turn == 2 and player.lives == 6:
            # Prefer future-value rolls/pairs over unnecessary tempo when the
            # turn-two heart restore makes a loss free.
            base -= 3.0
        unique_names = {
            team_slot.pet.name
            for team_slot in player.team
            if not team_slot.empty
        }
        if player.turn <= 2 and pet.name not in unique_names:
            if len(unique_names) >= 3:
                base -= 7.0
            elif len(unique_names) >= 2:
                base -= 2.0
        base += self._position_value(pet.name, action.target)
        if pet.name == "pet-cow" and plan.line_name in {
            "scaling board",
            "rabbit / cat food",
            "cow scaling conversion",
            "generic stat-tech",
        }:
            base += 20.0
        before = self._composition_value(
            player.team, plan.line_name, plan.matchup_counters
        )
        simulated = [
            None if team_slot.empty else team_slot.pet
            for team_slot in player.team
        ]
        simulated[action.target] = pet
        after = self._composition_value(
            simulated, plan.line_name, plan.matchup_counters
        )
        base += after - before
        return base

    def _score_combine(self, player, retained, merged, costs_gold):
        combined_total = (
            retained.total_experience + merged.total_experience + 1
        )
        final_level = 1 if combined_total < 2 else 2 if combined_total < 5 else 3
        previous_level = max(retained.level, merged.level)
        discover = (
            final_level > previous_level
            and not retained.level == merged.level == 2
        )
        score = 7.0 + 2.0 * (final_level - previous_level)
        if discover:
            score += 12.0 if player.turn % 2 == 1 else -3.0
        if all(not slot.empty for slot in player.team):
            score += 4.0
        if costs_gold:
            score -= 1.0
        return score

    def _score_buy_food(
        self, player, action, board_class, plan, known_opponent
    ):
        slot = player.shop[action.source]
        food = slot.obj.name
        score = self._item_value(player, food, plan, known_opponent)
        target = (
            None
            if action.target == NO_TARGET
            else player.team[action.target].pet
        )
        if food == "food-chocolate":
            score += 8.0 if player.turn >= 9 else 4.0
            if target is not None and target.level == 3:
                score -= 10.0
        elif food == "food-sleeping-pill":
            if target is not None and target.name in {
                "pet-flamingo",
                "pet-spider",
                "pet-turtle",
            }:
                score += 10.0
            else:
                score -= 12.0
        elif food == "food-apple" and player.turn <= 4:
            score += 4.0
            score += self._scaling_apple_bonus(player, target, plan)
        elif food == "food-canned-food":
            committed = sum(not team_slot.empty for team_slot in player.team)
            if player.turn >= 7 or committed >= 4 or player.lives <= 2:
                score -= 8.0
            score -= 3.0 * plan.tempo_pressure
        elif food in {"food-pear", "food-salad-bowl"}:
            score += 3.0
        elif food == "food-cake":
            if target is not None and target.name in SELL_PETS:
                score += 4.0
        elif food == "food-mushroom":
            if target is not None and target.name == "pet-mammoth":
                target_index = player.team.get_idx(target)
                behind = player.team.get_behind(target, n=1)
                if behind and behind[0].pet.name == "pet-tiger":
                    score += 24.0
        elif food == "food-milk":
            if target is not None and player.team.get_idx(target) in plan.committed_slots:
                score += 8.0
        if target is not None:
            score += 0.08 * (target.attack + target.health)
        if board_class == "food board":
            score += 2.0
        return score

    def _score_sell(self, player, source, plan=None):
        pet = player.team[source].pet
        score = -8.0 - 0.15 * (pet.attack + pet.health)
        if pet.name in SELL_PETS:
            score += 8.0 + 2.0 * pet.level
        if isinstance(pet.tier, int) and pet.tier <= 2 and player.turn >= 7:
            score += 4.0
        if (
            plan is not None
            and plan.tempo_pressure > 0
            and isinstance(pet.tier, int)
            and pet.tier <= 2
            and player.turn >= 5
        ):
            score += 2.0 * plan.tempo_pressure
        if all(not slot.empty for slot in player.team):
            score += 4.0
        if player.gold <= 2:
            score += 2.0
        cow_available = any(
            slot.obj.name == "pet-cow" and slot.cost <= player.gold
            for slot in player.shop
        )
        if (
            plan is not None
            and cow_available
            and all(not slot.empty for slot in player.team)
        ):
            score += 18.0 if source == plan.sell_slot else -8.0
        if pet.name == "pet-cow":
            free_milks = sum(
                slot.obj.name == "food-milk" and slot.cost == 0
                for slot in player.shop
            )
            if free_milks:
                score -= 24.0
            elif plan is not None and plan.line_name not in {
                "rabbit / cat food", "cow scaling conversion"
            }:
                score += 6.0
        return score

    def _score_freeze(self, player, source, plan=None, known_opponent=None):
        slot = player.shop[source]
        value = self._item_value(
            player, slot.obj.name, plan, known_opponent
        )
        if slot.slot_type == "levelup":
            if player.gold >= slot.cost:
                return -4.0
            frozen_sibling = any(
                other is not slot
                and other.slot_type == "levelup"
                and other.reward_group == slot.reward_group
                and other.frozen
                for other in player.shop
            )
            if not frozen_sibling:
                # Preserve the preferred unaffordable choice before rolling
                # or ending. Item value breaks ties between the two options.
                return 18.0 + 0.5 * value
            # A second frozen choice consumes another fresh pet slot. Keep it
            # only when its independent strategic value clears that cost.
            sibling_base = 8.0 if player.gold == 0 else 1.0
            return sibling_base + 0.5 * value
        if slot.slot_type == "pet":
            owned_copies = sum(
                (
                    1 + team_slot.pet.experience
                    if team_slot.pet.level == 1
                    else 3 + team_slot.pet.experience
                    if team_slot.pet.level == 2
                    else 6
                )
                for team_slot in player.team
                if not team_slot.empty
                and team_slot.pet.name == slot.obj.name
            )
            if owned_copies in {2, 5}:
                # An unaffordable copy completes level two or level three next
                # turn. It must beat both a low-gold roll and zero-gold end;
                # buying remains preferred whenever the copy is affordable.
                return 36.0 + value if slot.cost > player.gold else 4.0 + value
            if slot.cost <= player.gold:
                team_size = len(player.team)
                salad_isolation = (
                    player.turn in {5, 6} and team_size <= 3
                )
                late_full_board_hold = (
                    player.turn >= 7 and team_size == 5
                )
                if salad_isolation or late_full_board_hold:
                    return 2.0 + value
                return -12.0
        same_name = any(
            not team_slot.empty and team_slot.pet.name == slot.obj.name
            for team_slot in player.team
        )
        if same_name:
            value += 4.0
        return value - 5.0

    def _score_move(
        self, player, source, target, plan=None, known_opponent=None
    ):
        before = 0.0
        after = 0.0
        for index, slot in enumerate(player.team):
            if slot.empty:
                continue
            before += self._position_value(slot.pet.name, index)
            new_index = target if index == source else source if index == target else index
            after += self._position_value(slot.pet.name, new_index)
        improvement = after - before
        simulated = [
            None if slot.empty else slot.pet for slot in player.team
        ]
        simulated[source], simulated[target] = simulated[target], simulated[source]
        line_name = plan.line_name if plan is not None else "generic stat-tech"
        counters = plan.matchup_counters if plan is not None else ()
        composition_improvement = self._composition_value(
            simulated, line_name, counters
        ) - self._composition_value(player.team, line_name, counters)
        pill_available = any(
            not slot.empty
            and slot.obj.name == "food-sleeping-pill"
            and slot.cost <= player.gold
            for slot in player.shop
        )
        timing_penalty = -6.0
        if player.gold > 0 and not pill_available:
            timing_penalty = -20.0
        return timing_penalty + 3.0 * improvement + composition_improvement

    def _item_value(
        self, player, name, plan=None, known_opponent=None
    ):
        band = self._turn_band(player.turn)
        value = 2.0
        if name in TURN_HITS[band]:
            value += 6.0
        if name in SELL_PETS and player.turn <= 4:
            value += 3.0
        if plan is not None:
            if name in plan.required_hits:
                value += 6.0
            if name in LINE_MEMBERS[plan.line_name]:
                value += 3.0 * plan.confidence
            if name in IMMEDIATE_TEMPO_ITEMS:
                value += 2.5 * plan.tempo_pressure
            if name in SLOW_SCALING_ITEMS:
                value -= 1.5 * plan.tempo_pressure
        if name == "pet-cow" and plan is not None and (
            plan.line_name in {
                "rabbit / cat food", "cow scaling conversion", "generic stat-tech"
            }
            or any(
                not slot.empty and slot.pet.name in SCALERS
                for slot in player.team
            )
        ):
            value += 20.0
        if name == "pet-deer" and player.turn in {7, 8}:
            value += 13.0
            value += 4.0 * sum(
                not slot.empty and slot.pet.name == "pet-deer"
                for slot in player.team
            )
            if plan is not None and "rhino-vs-summons" in plan.matchup_counters:
                value -= 20.0
        if (
            name == "pet-wolverine"
            and self._ordered_pair_count(
                player.team, "pet-elephant", {"pet-blowfish", "pet-camel"}
            )
        ):
            value += 14.0
        if name.startswith("pet-"):
            slot = next(
                (
                    slot
                    for slot in player.shop
                    if slot.obj.name == name
                ),
                None,
            )
            if slot is not None:
                tier = getattr(slot.obj, "tier", 1)
                if isinstance(tier, int):
                    value += 0.8 * tier
        return value

    @staticmethod
    def _scaling_apple_bonus(player, target, plan):
        if target is None or player.turn not in {3, 4}:
            return 0.0
        if (
            plan.line_name == "horse-cricket summon tempo"
            or any(
                not slot.empty and slot.pet.name == "pet-dog"
                for slot in player.team
            )
        ):
            return 0.0
        scaling_lines = {
            "sell-pet / triple economy",
            "big tier-1",
            "generic stat-tech",
            "rabbit / cat food",
            "cow scaling conversion",
            "dragon cycling",
        }
        if plan.line_name not in scaling_lines:
            return 0.0
        if target.tier == 3:
            return 14.0
        if target.name in {"pet-peacock", "pet-kangaroo"}:
            copies = sum(
                not slot.empty and slot.pet.name == target.name
                for slot in player.team
            )
            if copies >= 2:
                return 12.0
        if plan.line_name == "big tier-1" and target.tier == 1:
            return 10.0
        return -2.0

    @classmethod
    def _composition_value(cls, team_or_pets, line_name, counters):
        pets = []
        for entry in team_or_pets:
            if entry is None:
                pets.append(None)
            elif hasattr(entry, "empty"):
                pets.append(None if entry.empty else entry.pet)
            else:
                pets.append(entry)
        value = 0.0
        names = {pet.name for pet in pets if pet is not None}
        for index in range(len(pets) - 1):
            ahead, behind = pets[index], pets[index + 1]
            if ahead is None or behind is None:
                continue
            if (
                ahead.name == "pet-elephant"
                and behind.name in {"pet-blowfish", "pet-camel"}
            ):
                value += 18.0
                if "pet-wolverine" in names:
                    value += 8.0
                if "pet-deer" in names:
                    value += 3.0
            if (
                ahead.name == "pet-mammoth"
                and ahead.status == "status-extra-life"
                and behind.name == "pet-tiger"
            ):
                value += 30.0
                if "skunk-accelerates-mammoth" in counters:
                    value += 8.0
                if "rhino-vs-summons" in counters:
                    value -= 12.0
                if "elephant-blowfish-vs-fragile-engine" in counters:
                    value -= 10.0
        deer_indices = [
            index
            for index, pet in enumerate(pets)
            if pet is not None and pet.name == "pet-deer"
        ]
        value += 5.0 * sum(index <= 1 for index in deer_indices)
        if "rhino-vs-summons" in counters:
            value -= 10.0 * len(deer_indices)
        if "pet-shark" in names:
            value += 2.0 * len(names & SUMMON_PETS)
        if "pet-turkey" in names:
            value += 2.0 * len(names & SUMMON_PETS)
        occupied = [index for index, pet in enumerate(pets) if pet is not None]
        if occupied:
            front = pets[min(occupied)]
            back = pets[max(occupied)]
            if (
                "crocodile-vs-backline-tiger" in counters
                and back.name == "pet-tiger"
            ):
                value -= 12.0
            if (
                "scorpion-vs-tall-front" in counters
                and front.attack + front.health >= 35
            ):
                value -= 10.0
        if "leopard-vs-weak-melon" in counters:
            fragile_engines = sum(
                pet is not None
                and pet.name in {
                    "pet-tiger", "pet-turkey", "pet-shark", "pet-snake"
                }
                and pet.status != "status-melon-armor"
                for pet in pets
            )
            value -= 5.0 * fragile_engines
        return value

    @staticmethod
    def _position_value(name, index):
        if name in FRONT_PETS:
            return max(0.0, 3.0 - index)
        if name in BACK_PETS:
            return max(0.0, index - 1.0)
        if name in {"pet-kangaroo", "pet-ox"}:
            return 2.0 if index in {1, 2} else 0.0
        return 0.0
