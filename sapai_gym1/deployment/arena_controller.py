"""Persistent, confidence-gated autonomous Arena controller."""

from __future__ import annotations

from dataclasses import dataclass, replace
import hashlib
import json
from pathlib import Path
import time

import numpy as np

from sapai import Food, Pet
from sapai.shop import ShopSlot
from sapai_gym1.rl import ActionType, FlatActionCodec

from .action_executor import DesktopActionMapper
from .geometry import NormalizedPoint
from .live_perception import LiveScreenObserver, PerceptionError
from .policy_adapter import (
    LivePolicyEngine,
    environment_from_live_state,
    live_state_from_environment,
)


@dataclass(frozen=True)
class ArenaLoopConfig:
    maximum_actions: int = 500
    shop_stable_wait: float = 12.0
    battle_wait: float = 150.0
    poll_interval: float = 1.0
    fast_battles: bool = True


class AutonomousArenaController:
    """Run one Arena match, observing and verifying every policy action."""

    autoplay_button = NormalizedPoint(0.5, 0.05)
    fast_button = NormalizedPoint(0.56, 0.05)
    # Empty background in shop, but any-click advances battle/result overlays.
    transition_click = NormalizedPoint(0.5, 0.20)
    return_to_menu_button = NormalizedPoint(0.5, 0.595)
    reload_button = NormalizedPoint(0.5, 0.605)
    confirm_button = NormalizedPoint(0.65, 0.55)
    continue_button = NormalizedPoint(0.88, 0.92)
    name_adjective = NormalizedPoint(0.5, 0.37)
    name_noun = NormalizedPoint(0.5, 0.64)

    def __init__(
        self,
        *,
        dashboard,
        observer: LiveScreenObserver,
        engine: LivePolicyEngine,
        initial_state,
        match_id: str,
        armed: bool,
        state_output: Path | None = None,
        config: ArenaLoopConfig | None = None,
    ):
        self.dashboard = dashboard
        self.observer = observer
        self.engine = engine
        self.env = environment_from_live_state(initial_state)
        self.match_id = match_id
        self.armed = bool(armed)
        self.state_output = (
            None if state_output is None else Path(state_output)
        )
        self.config = config or ArenaLoopConfig()
        self.mapper = DesktopActionMapper()
        self.opponent_history = tuple(initial_state.opponent_history)
        self.last_frame = None
        self.actions = 0
        self.terminal = False

    def _state(self, frame=None, confidence=1.0, phase="shop"):
        digest = None
        if frame is not None:
            digest = hashlib.sha256(frame.tobytes()).hexdigest()
        return live_state_from_environment(
            self.env,
            phase=phase,
            opponent_history=self.opponent_history,
            screenshot_sha256=digest,
            confidence=confidence,
            metadata={
                "controller": "autonomous-arena-v1",
                "matchId": self.match_id,
                "armed": self.armed,
            },
        )

    def _publish(self, **payload):
        state = self._state(
            self.last_frame,
            confidence=payload.pop("confidence", 1.0),
            phase=payload.pop("phase", "shop"),
        )
        message = {
            "mode": "armed" if self.armed else "dry-run",
            "matchId": self.match_id,
            "checkpointUpdate": self.engine.checkpoint.get("update"),
            "loopAction": self.actions,
            "state": state.to_dict(),
            **payload,
        }
        self.dashboard.publish_agent(message)
        if self.state_output is not None and (
            state.phase == "shop" or message.get("status") == "arena-complete"
        ):
            self.state_output.parent.mkdir(parents=True, exist_ok=True)
            temporary = self.state_output.with_suffix(
                self.state_output.suffix + ".tmp"
            )
            temporary.write_text(json.dumps(state.to_dict(), indent=2) + "\n")
            temporary.replace(self.state_output)
        return message

    def _stable_shop_observation(
        self, expected_shape=None, expected_state=None
    ):
        deadline = time.monotonic() + self.config.shop_stable_wait
        last_shape = None
        while time.monotonic() < deadline:
            frame, difference = self.dashboard.wait_for_stable_frame(
                maximum_wait=min(4.0, max(1.0, deadline - time.monotonic()))
            )
            self.last_frame = frame
            observation = self.observer.observe_shop(
                frame, expected_state=expected_state
            )
            last_shape = (len(observation.pets), len(observation.foods))
            if expected_shape is None or last_shape == expected_shape:
                return frame, difference, observation
            # Unity may briefly hold a partially populated but pixel-stable
            # frame while roll/tier-up items animate into their slots.
            time.sleep(0.5)
        raise PerceptionError(
            f"Shop did not reach expected shape {expected_shape}; last={last_shape}"
        )

    @staticmethod
    def _observation_signature(observation):
        return (
            tuple(observation.team_ids),
            tuple(item.item_id for item in observation.pets),
            tuple(item.item_id for item in observation.foods),
        )

    def _changed_shop_observation(
        self,
        reference,
        expected_shape,
        *,
        before_observation,
        expected_state,
        stochastic,
    ):
        frame, difference, _ = self.dashboard.wait_for_changed_then_stable(
            reference,
            maximum_wait=self.config.shop_stable_wait,
        )
        deadline = time.monotonic() + self.config.shop_stable_wait
        last_shape = None
        last_error = None
        before_signature = self._observation_signature(before_observation)
        while time.monotonic() < deadline:
            self.last_frame = frame
            try:
                observation = self.observer.observe_shop(
                    frame, expected_state=expected_state
                )
            except PerceptionError as error:
                last_error = str(error)
                time.sleep(0.5)
                frame, difference = self.dashboard.wait_for_stable_frame(
                    maximum_wait=min(
                        4.0, max(1.0, deadline - time.monotonic())
                    )
                )
                continue
            last_shape = (len(observation.pets), len(observation.foods))
            if last_shape == expected_shape:
                if stochastic:
                    if self._observation_signature(observation) != before_signature:
                        return frame, difference, observation
                    last_error = "stochastic shop still matches the pre-action frame"
                else:
                    try:
                        self.observer.verify_identities(
                            observation, expected_state
                        )
                        return frame, difference, observation
                    except PerceptionError as error:
                        last_error = str(error)
            time.sleep(0.5)
            frame, difference = self.dashboard.wait_for_stable_frame(
                maximum_wait=min(4.0, max(1.0, deadline - time.monotonic()))
            )
        raise PerceptionError(
            f"Changed shop did not reach expected shape {expected_shape}; "
            f"last={last_shape}; last_error={last_error}"
        )

    @staticmethod
    def _expected_ids(state):
        return {
            "team": tuple(
                None if pet is None else pet.item_id
                for pet in state.team.slots
            ),
            "pets": tuple(
                slot.pet.item_id
                for slot in state.shop
                if slot.kind in {"pet", "levelup"}
            ),
            "foods": tuple(
                slot.food.visual_id for slot in state.shop if slot.kind == "food"
            ),
        }

    def verify_current_shop(self):
        state_before = self._state()
        frame, difference, observation = self._stable_shop_observation(
            expected_state=state_before
        )
        state = self._state(frame, confidence=observation.confidence)
        self.observer.verify_identities(observation, state)
        corrections = self.reconcile_observed_stats(observation)
        self._publish(
            status="verified-shop",
            stableFrameDifference=difference,
            perception=observation.to_dict(),
            statCorrections=corrections,
            confidence=observation.confidence,
        )
        return frame, observation

    def reconcile_observed_stats(self, observation):
        """Make live digits authoritative over local random-target outcomes."""

        corrections = []
        for index, stats in enumerate(observation.team_stats):
            if stats is None or self.env.agent.team[index].empty:
                continue
            pet = self.env.agent.team[index].pet
            for field, observed in zip(("attack", "health"), stats):
                expected = int(getattr(pet, field))
                if expected == int(observed):
                    continue
                base_field = "_attack" if field == "attack" else "_health"
                setattr(
                    pet,
                    base_field,
                    int(getattr(pet, base_field)) + int(observed) - expected,
                )
                corrections.append(
                    {
                        "location": f"team-{index}",
                        "field": field,
                        "expected": expected,
                        "observed": int(observed),
                    }
                )
        for index, level_state in enumerate(observation.team_levels):
            if level_state is None or self.env.agent.team[index].empty:
                continue
            pet = self.env.agent.team[index].pet
            observed_level, observed_experience = map(int, level_state)
            if (
                int(pet.level) == observed_level
                and int(pet.experience) == observed_experience
            ):
                continue
            corrections.append(
                {
                    "location": f"team-{index}",
                    "field": "level-experience",
                    "expected": [int(pet.level), int(pet.experience)],
                    "observed": [observed_level, observed_experience],
                }
            )
            pet.level = observed_level
            pet.experience = observed_experience
        pet_slots = [
            slot
            for slot in self.env.agent.shop
            if slot.slot_type in {"pet", "levelup"}
        ]
        for visible, slot in zip(observation.pets, pet_slots):
            if visible.attack is None or visible.health is None:
                continue
            for field, observed in (
                ("attack", visible.attack),
                ("health", visible.health),
            ):
                expected = int(getattr(slot.obj, field))
                if expected == int(observed):
                    continue
                base_field = "_attack" if field == "attack" else "_health"
                setattr(
                    slot.obj,
                    base_field,
                    int(getattr(slot.obj, base_field))
                    + int(observed)
                    - expected,
                )
                corrections.append(
                    {
                        "location": f"shop-pet-{visible.screen_rank}",
                        "field": field,
                        "expected": expected,
                        "observed": int(observed),
                    }
                )
        if corrections:
            self.env._agent_actions_cache = None
        return corrections

    def _pet_from_observation(self, item_id, template_slot):
        if template_slot.frozen and template_slot.obj.name == item_id:
            return template_slot.obj
        pet = Pet(item_id)
        template_base = Pet(template_slot.obj.name)
        pet._attack += int(template_slot.obj.attack) - int(template_base.attack)
        pet._health += int(template_slot.obj.health) - int(template_base.health)
        pet.sell_value = int(template_slot.obj.sell_value)
        return pet

    def _food_from_observation(self, visual_id, template_slot):
        entry = self.observer.catalog.by_visual_id[visual_id]
        if template_slot.frozen:
            template_visual = self._visual_food_id(template_slot.obj)
            if template_visual == visual_id:
                return template_slot.obj
        food = Food(entry.canonical_id)
        if entry.rollable:
            # Carry only an actual shop-wide discount, never the template
            # item's absolute base price. Otherwise a local Sleeping Pill
            # roll (base 1) can make a real Honey/Pear look purchasable for 1.
            template_base_cost = int(Food(template_slot.obj.name).cost)
            discount = int(template_slot.cost) - template_base_cost
            food.cost = max(0, int(food.cost) + discount)
            return food
        # Generated scaled food is deterministic from the triggering pet and
        # should retain the local mirror's price/stats.
        food.cost = int(template_slot.cost)
        food.attack = int(template_slot.obj.attack)
        food.health = int(template_slot.obj.health)
        food.status = template_slot.obj.status
        return food

    @staticmethod
    def _visual_food_id(food):
        if food.name == "food-apple":
            if int(food.attack) >= 3:
                return "food-best-apple"
            if int(food.attack) >= 2:
                return "food-better-apple"
        if food.name == "food-milk":
            if int(food.attack) >= 3:
                return "food-best-milk"
            if int(food.attack) >= 2:
                return "food-better-milk"
        return food.name

    def reconcile_stochastic_shop(self, observation):
        """Replace local random identities while retaining live shop semantics."""

        expected_pets = [
            slot
            for slot in self.env.agent.shop
            if slot.slot_type in {"pet", "levelup"}
        ]
        expected_foods = [
            slot for slot in self.env.agent.shop if slot.slot_type == "food"
        ]
        if len(expected_pets) != len(observation.pets) or len(expected_foods) != len(
            observation.foods
        ):
            raise PerceptionError(
                "Stochastic shop shape mismatch: "
                f"expected {len(expected_pets)} pets/{len(expected_foods)} foods, "
                f"observed {len(observation.pets)} pets/{len(observation.foods)} foods"
            )
        slots = []
        def templates_in_visible_order(expected, visible):
            frozen = iter(slot for slot in expected if slot.frozen)
            ordinary = iter(slot for slot in expected if not slot.frozen)
            return [
                next(frozen if item.frozen else ordinary) for item in visible
            ]

        pet_templates = templates_in_visible_order(
            expected_pets, observation.pets
        )
        food_templates = templates_in_visible_order(
            expected_foods, observation.foods
        )
        for visible, template in zip(observation.pets, pet_templates):
            pet = self._pet_from_observation(visible.item_id, template)
            slots.append(
                ShopSlot(
                    obj=pet,
                    slot_type=template.slot_type,
                    frozen=template.frozen,
                    turn=self.env.agent.turn,
                    cost=template.cost,
                    pack=template.pack,
                    seed_state=self.env.agent.shop.rs.get_state(),
                    reward_group=template.reward_group,
                )
            )
        for visible, template in zip(observation.foods, food_templates):
            food = self._food_from_observation(visible.item_id, template)
            slot = ShopSlot(
                obj=food,
                slot_type="food",
                frozen=template.frozen,
                turn=self.env.agent.turn,
                cost=food.cost,
                pack=template.pack,
                seed_state=self.env.agent.shop.rs.get_state(),
                reward_group=template.reward_group,
            )
            slot.cost = int(food.cost)
            slot.obj.cost = int(food.cost)
            slots.append(slot)
        self.env.agent.shop.slots = slots
        self.env.shop_agent = self.env.agent.shop
        self.env.agent._connect_objects()
        self.env._agent_actions_cache = None

    def step_shop(self, *, _prepared=None):
        pre_action_state = self._state()
        frame, _, before = self._stable_shop_observation(
            expected_state=pre_action_state
        )
        current = self._state(frame, confidence=before.confidence)
        self.observer.verify_identities(before, current)
        before_corrections = self.reconcile_observed_stats(before)
        current = self._state(frame, confidence=before.confidence)
        memory_before = self.engine.snapshot_memory()
        recommendation = _prepared or self.engine.recommend_environment(
            self.env, match_id=self.match_id, state_confidence=before.confidence,
        )
        action = recommendation.action
        action_type = ActionType(action.action_type)
        barrier = self.engine._barrier_reason(self.env, action)
        gesture = self.mapper.gesture(action, state=current)
        payload = self._publish(
            status="recommended",
            perception=before.to_dict(),
            recommendation=recommendation.to_dict(),
            barrierAfter=barrier,
            gesture=gesture.to_dict(),
            confidence=before.confidence,
        )
        if not self.armed:
            return {**payload, "executed": False}

        if _prepared is None:
            self.engine.validate_recommendation(recommendation, self.env)
            if len(recommendation.primitive_plan) > 1:
                # Each toggle is re-observed/verified by the existing primitive
                # executor. All are ONE neural decision; no intermediate GRU
                # advancement. A mismatch abandons the remaining plan.
                try:
                    for primitive in recommendation.primitive_plan:
                        part = replace(recommendation, action=primitive,
                                       action_type=ActionType(primitive.action_type).name,
                                       source=primitive.source, target=primitive.target,
                                       primitive_plan=(primitive,))
                        result = self.step_shop(_prepared=part)
                    return result
                except Exception:
                    self.engine.restore_memory(memory_before)
                    raise
        if FlatActionCodec.encode(action) not in self.env._avail_actions(0):
            raise PerceptionError('Prepared primitive is no longer legal')

        self.dashboard.execute_gesture(gesture)
        if action_type is ActionType.BUY_FOOD:
            confirmation_visible = False
            confirmation_deadline = time.monotonic() + 2.5
            while time.monotonic() < confirmation_deadline:
                time.sleep(0.25)
                post_food_frame = self.dashboard.frame()
                if self.observer.confirmation_prompt_visible(post_food_frame):
                    confirmation_visible = True
                    break
            if confirmation_visible:
                self.dashboard.click(self.confirm_button)
                time.sleep(0.5)
                self.dashboard.click(self.mapper.dismiss_detail_point)
                time.sleep(0.5)
            else:
                self.dashboard.click(self.mapper.dismiss_detail_point)
        levelup_slots_before = sum(
            slot.slot_type == "levelup" for slot in self.env.agent.shop
        )
        self.env.resolve_action(0, FlatActionCodec.encode(action))
        levelup_slots_after = sum(
            slot.slot_type == "levelup" for slot in self.env.agent.shop
        )
        generated_levelup = levelup_slots_after > levelup_slots_before
        stochastic_shop_change = (
            barrier in {"new-shop", "level-up-discover"}
            or generated_levelup
        )
        self.actions += 1
        if action_type is ActionType.END_TURN:
            # A delivered click isn't yet a confirmed transition. Commit only
            # when battle/name/result UI proves that we left the shop.
            self._pending_end_recommendation = recommendation
            self._publish(
                status="waiting-for-battle",
                recommendation=recommendation.to_dict(),
                phase="battle",
            )
            return {"executed": True, "endTurn": True, "frameBefore": frame}

        try:
            expected_shape = (
                sum(
                    slot.slot_type in {"pet", "levelup"}
                    for slot in self.env.agent.shop
                ),
                sum(slot.slot_type == "food" for slot in self.env.agent.shop),
            )
            expected_before_reconcile = self._state(
                confidence=before.confidence
            )
            try:
                after_frame, difference, after = self._changed_shop_observation(
                    frame,
                    expected_shape,
                    before_observation=before,
                    expected_state=expected_before_reconcile,
                    stochastic=stochastic_shop_change,
                )
            except (TimeoutError, PerceptionError):
                late_action_frame = self.dashboard.frame()
                if (
                    action_type is ActionType.BUY_FOOD
                    and self.observer.confirmation_prompt_visible(
                        late_action_frame
                    )
                ):
                    self.dashboard.click(self.confirm_button)
                    time.sleep(0.5)
                    self.dashboard.click(self.mapper.dismiss_detail_point)
                    time.sleep(0.5)
                    after_frame, difference, after = (
                        self._changed_shop_observation(
                            late_action_frame,
                            expected_shape,
                            before_observation=before,
                            expected_state=expected_before_reconcile,
                            stochastic=stochastic_shop_change,
                        )
                    )
                else:
                # The HTTP bridge can confirm delivery while Unity drops an
                # occasional drag during a frame boundary. The live screen is
                # still authoritative: retry only if a fresh observation is
                # exactly the complete pre-action identity state. A partially
                # applied or differently applied action is never repeated.
                    retry_frame, _, retry_before = self._stable_shop_observation(
                        expected_shape=(len(before.pets), len(before.foods)),
                        expected_state=pre_action_state,
                    )
                    if self._observation_signature(
                        retry_before
                    ) != self._observation_signature(before):
                        raise
                    self.dashboard.execute_gesture(gesture)
                    after_frame, difference, after = (
                        self._changed_shop_observation(
                            retry_frame,
                            expected_shape,
                            before_observation=retry_before,
                            expected_state=expected_before_reconcile,
                            stochastic=stochastic_shop_change,
                        )
                    )
            if stochastic_shop_change:
                self.reconcile_stochastic_shop(after)
            after_corrections = self.reconcile_observed_stats(after)
            expected = self._state(after_frame, confidence=after.confidence)
            self.observer.verify_identities(after, expected)
            if action_type in (ActionType.FREEZE, ActionType.UNFREEZE):
                observed_flags = tuple(x.frozen for x in (*after.pets, *after.foods))
                expected_flags = tuple(slot.frozen for kind in ('pet', 'food') for slot in expected.shop
                                       if (slot.kind in {'pet', 'levelup'}) == (kind == 'pet'))
                if observed_flags != expected_flags:
                    raise PerceptionError('Freeze toggle was not visually confirmed')
        except Exception:
            self.engine.restore_memory(memory_before)
            raise
        if _prepared is None:
            self.engine.commit_recommendation(recommendation, confirmed=True)
        self._publish(
            status="executed-and-verified",
            recommendation=recommendation.to_dict(),
            effectiveBarrierAfter=(
                "level-up-discover" if generated_levelup else barrier
            ),
            stableFrameDifference=difference,
            perception=after.to_dict(),
            statCorrections=[*before_corrections, *after_corrections],
            confidence=after.confidence,
        )
        return {"executed": True, "endTurn": False}

    def _commit_end_if_observed(self, phase):
        if phase in {'battle', 'name', 'result', 'gameover'}:
            pending = getattr(self, '_pending_end_recommendation', None)
            if pending is not None:
                self.engine.commit_recommendation(pending, confirmed=True)
                self._pending_end_recommendation = None

    def _battle_to_next_shop(self, frame_before):
        before_scalars = self.observer.scalar_fingerprints(frame_before)
        deadline = time.monotonic() + self.config.battle_wait
        battle_controls_set = False
        left_shop = False
        name_selected = False
        last_transition_click = 0.0
        shop_streak = 0
        reload_attempts = 0
        while time.monotonic() < deadline:
            frame = self.dashboard.frame()
            phase = self.observer.phase(frame)
            self._commit_end_if_observed(phase)
            self.last_frame = frame
            self._publish(
                status=f"battle-transition-{phase}",
                phase="battle" if phase != "shop" else "shop",
                autoplayEnabled=(
                    self.observer.battle_is_playing(frame)
                    if self.observer.battle_controls_visible(frame)
                    else None
                ),
            )
            controls_visible = self.observer.battle_controls_visible(frame)
            if (
                controls_visible
                and self.config.fast_battles
                and not battle_controls_set
            ):
                self.dashboard.click(self.fast_button)
                time.sleep(0.5)
                control_frame = self.dashboard.frame()
                if not self.observer.battle_is_playing(control_frame):
                    self.dashboard.click(self.autoplay_button)
                    time.sleep(1.0)
                battle_controls_set = self.observer.battle_is_playing(
                    self.dashboard.frame()
                )
                last_transition_click = time.monotonic()
                continue
            if phase != "shop":
                left_shop = True
                shop_streak = 0
            elif left_shop:
                shop_streak += 1
            if phase == "shop" and left_shop and shop_streak >= 2:
                after_scalars = self.observer.scalar_fingerprints(frame)
                if self.observer.scalar_changed(
                    before_scalars, after_scalars, "wins"
                ):
                    outcome = "win"
                elif self.observer.scalar_changed(
                    before_scalars, after_scalars, "lives"
                ):
                    outcome = "loss"
                else:
                    outcome = "draw"
                self._start_next_turn(outcome)
                expected_shape = (
                    sum(
                        slot.slot_type in {"pet", "levelup"}
                        for slot in self.env.agent.shop
                    ),
                    sum(
                        slot.slot_type == "food"
                        for slot in self.env.agent.shop
                    ),
                )
                _, _, observation = self._stable_shop_observation(
                    expected_shape=expected_shape,
                    expected_state=self._state(),
                )
                self.reconcile_stochastic_shop(observation)
                corrections = self.reconcile_observed_stats(observation)
                expected = self._state(self.last_frame, confidence=observation.confidence)
                self.observer.verify_identities(observation, expected)
                self._publish(
                    status="new-turn-verified",
                    outcome=outcome,
                    perception=observation.to_dict(),
                    statCorrections=corrections,
                    confidence=observation.confidence,
                )
                return outcome
            if phase == "reload":
                if reload_attempts >= 3:
                    raise RuntimeError("Repeated live-client desync after reload")
                reload_attempts += 1
                self._publish(
                    status="client-desync-reloading",
                    reloadAttempt=reload_attempts,
                    phase="battle",
                )
                self.dashboard.click(self.reload_button)
                time.sleep(5.0)
                shop_streak = 0
                continue
            if phase == "gameover":
                outcome = (
                    "win"
                    if self.env.agent.wins >= 9
                    and self.env.agent.lives > 1
                    else "loss"
                )
                if outcome == "win":
                    self.env.agent.wins = 10
                    self.env.agent.last_battle_outcome = 1
                    self.env.agent.lf_winner = True
                else:
                    self.env.agent.lives = 0
                    self.env.agent.last_battle_outcome = -1
                    self.env.agent.lf_winner = False
                self.terminal = True
                self._publish(
                    status="arena-complete",
                    outcome=outcome,
                    phase="result",
                )
                self.dashboard.click(self.return_to_menu_button)
                return outcome
            if phase == "shop":
                # End-turn click has not reached Unity yet.
                time.sleep(self.config.poll_interval)
                continue
            if phase == "name" and not name_selected:
                self.dashboard.click(self.name_adjective)
                time.sleep(1.0)
                self.dashboard.click(self.name_noun)
                name_selected = True
                time.sleep(1.0)
                continue
            if phase != "name" and (
                time.monotonic() - last_transition_click >= 0.35
            ):
                # Constant-click through battle animation, results, heart
                # restoration, and Continue screens. The point is inert once
                # the next shop appears.
                self.dashboard.click(self.transition_click)
                last_transition_click = time.monotonic()
            time.sleep(self.config.poll_interval)
        raise TimeoutError("Arena battle did not return to a shop in time")

    def _start_next_turn(self, outcome):
        outcome_value = {"win": 1, "loss": -1, "draw": 0}[outcome]
        winner = outcome != "loss"
        self.env.agent.last_battle_outcome = outcome_value
        self.env.agent.lf_winner = winner
        if outcome == "win":
            self.env.agent.wins += 1
        elif outcome == "loss":
            self.env.agent.lives = max(0, self.env.agent.lives - 1)
        if self.env.agent.turn == 2:
            # Arena displays five starting hearts; preserve the observed cap.
            self.env.agent.lives = min(5, self.env.agent.lives + 1)
        self.env.agent.start_turn(winner=winner)
        self.env.turn = self.env.agent.turn
        self.env.shop_action_count_agent = 0
        self.env.reorder_count_agent = 0
        self.env._freeze_toggled_agent.clear()
        self.env._seen_team_orders_agent = {
            self.env._team_order_key(self.env.agent)
        }
        self.env._agent_actions_cache = None

    def run(self):
        self.verify_current_shop()
        completed_turns = 0
        while self.actions < self.config.maximum_actions:
            result = self.step_shop()
            if not self.armed:
                return result
            if result["endTurn"]:
                self._battle_to_next_shop(result["frameBefore"])
                completed_turns += 1
                if self.terminal:
                    break
                if self.env.agent.lives <= 0 or self.env.agent.wins >= 10:
                    break
        if self.actions >= self.config.maximum_actions:
            raise RuntimeError("Autonomous Arena action cap reached")
        final = self._publish(
            status="arena-complete",
            completedTurns=completed_turns,
            phase="result" if self.terminal else "shop",
        )
        return final
