"""Conservative visual parsing for the fixed 1920x1080 SAP client."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .pet_classifier import OnnxPetClassifier
from .screen_profile import DESKTOP_1080P
from .sprite_catalog import SpriteCatalog
from .sprite_matcher import SpriteMatcher
from .stat_ocr import DigitTemplateRecognizer, read_level_experience


class PerceptionError(RuntimeError):
    """Raised when the live frame is not safe enough to act on."""


@dataclass(frozen=True)
class VisibleItem:
    kind: str
    item_id: str
    screen_rank: int
    confidence: float
    attack: int | None = None
    health: int | None = None
    frozen: bool = False

    def to_dict(self):
        return {
            "kind": self.kind,
            "itemId": self.item_id,
            "screenRank": self.screen_rank,
            "confidence": self.confidence,
            "attack": self.attack,
            "health": self.health,
            "frozen": self.frozen,
        }


@dataclass(frozen=True)
class ShopObservation:
    phase: str
    pets: tuple[VisibleItem, ...]
    foods: tuple[VisibleItem, ...]
    team_ids: tuple[str | None, ...]
    team_stats: tuple[tuple[int, int] | None, ...]
    team_levels: tuple[tuple[int, int] | None, ...]
    confidence: float

    @property
    def items(self):
        return self.pets + self.foods

    def to_dict(self):
        return {
            "phase": self.phase,
            "pets": [item.to_dict() for item in self.pets],
            "foods": [item.to_dict() for item in self.foods],
            "teamIds": list(self.team_ids),
            "teamStats": [
                None if stats is None else list(stats)
                for stats in self.team_stats
            ],
            "teamLevels": [
                None if level is None else list(level)
                for level in self.team_levels
            ],
            "confidence": self.confidence,
        }


class LiveScreenObserver:
    """Identity/phase recognizer with strict abstention thresholds.

    Exact stat/equipment state is maintained by the simulator mirror. The
    observer independently verifies occupancy and identities after every UI
    action and fully reparses the stochastic shop after rolls/discovers.
    """

    def __init__(
        self,
        pet_model: Path,
        sprite_root: Path,
        digit_model: Path | None = None,
        *,
        minimum_pet_probability: float = 0.75,
        minimum_shop_pet_probability: float = 0.88,
        minimum_pet_margin: float = 0.50,
    ):
        try:
            import cv2
        except ImportError as error:
            raise RuntimeError("Live perception requires opencv-python") from error
        self.cv2 = cv2
        self.pet_classifier = OnnxPetClassifier(
            Path(pet_model),
            minimum_probability=minimum_pet_probability,
            minimum_margin=minimum_pet_margin,
        )
        self.catalog = SpriteCatalog.from_replay_bot(Path(sprite_root))
        self.food_matcher = SpriteMatcher(self.catalog)
        self.minimum_shop_pet_probability = float(
            minimum_shop_pet_probability
        )
        self.minimum_pet_margin = float(minimum_pet_margin)
        self.digit_recognizer = (
            None
            if digit_model is None
            else DigitTemplateRecognizer(Path(digit_model))
        )

    @staticmethod
    def _crop(frame, x1, y1, x2, y2):
        return frame[int(y1) : int(y2), int(x1) : int(x2)]

    def phase(self, frame) -> str:
        if frame.shape[:2] != (1080, 1920):
            raise PerceptionError(
                f"Expected 1920x1080 frame, received {frame.shape[1]}x{frame.shape[0]}"
            )
        hsv = self.cv2.cvtColor(frame, self.cv2.COLOR_BGR2HSV)
        orange = self.cv2.inRange(
            hsv,
            np.array([5, 170, 120], dtype=np.uint8),
            np.array([25, 255, 255], dtype=np.uint8),
        )

        def orange_ratio(x1, y1, x2, y2):
            return float(orange[y1:y2, x1:x2].mean() / 255.0)

        roll = orange_ratio(0, 930, 300, 1080)
        end_turn = orange_ratio(1500, 930, 1920, 1080)
        beige = self.cv2.inRange(
            hsv,
            np.array([5, 50, 120], dtype=np.uint8),
            np.array([30, 220, 255], dtype=np.uint8),
        )
        gameover_top = float(
            beige[380:580, 650:1270].mean() / 255.0
        )
        gameover_bottom = float(
            beige[590:700, 650:1270].mean() / 255.0
        )
        if gameover_top > 0.50 and gameover_bottom > 0.50:
            return "gameover"
        reload_orange = orange[605:700, 700:1220]
        if float(reload_orange.mean() / 255.0) > 0.45:
            return "reload"
        # Result and turn-two heart-restoration overlays can leave the shop
        # buttons visible underneath. Darkness must take precedence over the
        # ordinary Roll/End-Turn signature.
        if float(frame.mean()) < 75.0:
            return "result"
        if roll > 0.30 and end_turn > 0.30:
            return "shop"
        blue = self.cv2.inRange(
            hsv,
            np.array([100, 120, 60], dtype=np.uint8),
            np.array([135, 255, 220], dtype=np.uint8),
        )
        green = self.cv2.inRange(
            hsv,
            np.array([40, 100, 50], dtype=np.uint8),
            np.array([90, 255, 220], dtype=np.uint8),
        )
        name_blue = float(blue[340:470, 750:1170].mean() / 255.0)
        name_green = float(green[615:750, 750:1170].mean() / 255.0)
        if name_blue > 0.40 and name_green > 0.40:
            return "continue" if end_turn > 0.12 else "name"
        # A battle is split at the vertical center and has no Roll button.
        center = frame[:, 948:972]
        center_edge = float(
            np.abs(np.diff(center.astype(np.int16), axis=1)).mean()
        )
        if roll < 0.08 and center_edge > 4.0:
            return "battle"
        if end_turn > 0.12:
            return "continue"
        return "unknown"

    @staticmethod
    def _identity_crop(frame, center_x, *, shop):
        center = round(center_x * 1920)
        y1, y2 = (625, 765) if shop else (350, 490)
        return frame[y1:y2, center - 90 : center + 90]

    @staticmethod
    def _has_price_badge(frame, x):
        """Shop occupancy is unambiguous from the white freeze/price badge."""

        center = round(x * 1920)
        crop = frame[570:635, center - 34 : center + 34]
        # Pale cloud backgrounds are also nearly white. The badge's thick
        # black rounded border is the reliable occupancy signal.
        gray = np.mean(crop, axis=2)
        dark = gray < 45
        return float(dark.mean()) > 0.05

    @staticmethod
    def _is_frozen_item(frame, center_x):
        try:
            import cv2
        except ImportError as error:
            raise RuntimeError("Frozen-item detection requires OpenCV") from error
        center = round(center_x * 1920)
        crop = frame[620:770, center - 85 : center + 85]
        hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
        cyan = (
            (hsv[:, :, 0] > 80)
            & (hsv[:, :, 0] < 115)
            & (hsv[:, :, 1] > 20)
            & (hsv[:, :, 1] < 150)
            & (hsv[:, :, 2] > 130)
        )
        pale_ice = (hsv[:, :, 1] < 80) & (hsv[:, :, 2] > 150)
        return float(cyan.mean()) > 0.65 and float(pale_ice.mean()) > 0.75

    def observe_shop(self, frame, expected_state=None) -> ShopObservation:
        phase = self.phase(frame)
        if phase != "shop":
            raise PerceptionError(f"Expected shop frame, detected {phase}")

        pet_crops = []
        pet_ranks = []
        for rank, (point, badge) in enumerate(
            zip(
                DESKTOP_1080P.pet_shop_points,
                DESKTOP_1080P.pet_freeze_points,
            )
        ):
            if self._has_price_badge(frame, badge.x):
                pet_ranks.append(rank)
                pet_crops.append(self._identity_crop(frame, point.x, shop=True))

        pets = []
        if pet_crops:
            predictions = self.pet_classifier.predict(pet_crops)
            expected_pets = (
                []
                if expected_state is None
                else [
                    slot
                    for slot in expected_state.shop
                    if slot.kind in {"pet", "levelup"}
                ]
            )
            expected_frozen_pets = [
                slot for slot in expected_pets if slot.frozen
            ]
            frozen_pet_index = 0
            for bank_index, (rank, prediction) in enumerate(
                zip(pet_ranks, predictions)
            ):
                visually_frozen = self._is_frozen_item(
                    frame, DESKTOP_1080P.pet_shop_points[rank].x
                )
                if visually_frozen:
                    if frozen_pet_index >= len(expected_frozen_pets):
                        raise PerceptionError(
                            "Observed an untracked frozen pet slot"
                        )
                    expected = expected_frozen_pets[frozen_pet_index].pet
                    frozen_pet_index += 1
                    pets.append(
                        VisibleItem(
                            "pet",
                            expected.item_id,
                            rank,
                            0.95,
                            expected.attack,
                            expected.health,
                            True,
                        )
                    )
                    continue
                if (
                    prediction.probability
                    < self.minimum_shop_pet_probability
                    or prediction.margin < self.minimum_pet_margin
                ):
                    raise PerceptionError(
                        "Low-confidence shop pet at visual rank "
                        f"{rank}: {prediction.to_dict()}"
                    )
                pets.append(
                    VisibleItem(
                        "pet",
                        prediction.item_id,
                        rank,
                        prediction.probability,
                        *self._pet_stats(
                            frame,
                            DESKTOP_1080P.pet_shop_points[rank].x,
                            shop=True,
                        ),
                        False,
                    )
                )

        foods = []
        two_food_layout = self._has_price_badge(
            frame, DESKTOP_1080P.food_freeze_points[0].x
        )
        food_points = DESKTOP_1080P.food_points(2 if two_food_layout else 1)
        food_badges = DESKTOP_1080P.food_freeze_points_for_count(
            2 if two_food_layout else 1
        )
        expected_foods = (
            []
            if expected_state is None
            else [slot for slot in expected_state.shop if slot.kind == "food"]
        )
        expected_frozen_foods = [
            slot for slot in expected_foods if slot.frozen
        ]
        frozen_food_index = 0
        for rank, (point, badge) in enumerate(zip(food_points, food_badges)):
            if not self._has_price_badge(frame, badge.x):
                continue
            bank_index = len(foods)
            visually_frozen = self._is_frozen_item(frame, point.x)
            if visually_frozen:
                if frozen_food_index >= len(expected_frozen_foods):
                    raise PerceptionError(
                        "Observed an untracked frozen food slot"
                    )
                expected = expected_frozen_foods[frozen_food_index].food
                frozen_food_index += 1
                foods.append(
                    VisibleItem(
                        "food",
                        expected.visual_id,
                        rank,
                        0.95,
                        frozen=True,
                    )
                )
                continue
            center = round(point.x * 1920)
            crop = frame[570:790, center - 115 : center + 115]
            match = self.food_matcher.match_food(crop)
            if not match.accepted:
                expected_food = (
                    expected_foods[bank_index].food
                    if bank_index < len(expected_foods)
                    else None
                )
                expected_entry = (
                    None
                    if expected_food is None
                    else self.catalog.by_visual_id.get(
                        expected_food.visual_id
                    )
                )
                if expected_entry is not None and not expected_entry.rollable:
                    foods.append(
                        VisibleItem(
                            "food",
                            expected_food.visual_id,
                            rank,
                            0.75,
                            frozen=False,
                        )
                    )
                    continue
                raise PerceptionError(
                    f"Low-confidence food at visual rank {rank}: {match.to_dict()}"
                )
            foods.append(
                VisibleItem(
                    "food",
                    match.best.visual_id,
                    rank,
                    match.best.score,
                    frozen=False,
                )
            )

        team_crops = [
            self._identity_crop(frame, point.x, shop=False)
            for point in DESKTOP_1080P.team_points
        ]
        team_predictions = self.pet_classifier.predict(team_crops)
        expected_team = (
            (None,) * len(team_predictions)
            if expected_state is None
            else expected_state.team.slots
        )
        team_ids = []
        team_stats = []
        team_levels = []
        expected_identity_fallbacks = 0
        for index, prediction in enumerate(team_predictions):
            point = DESKTOP_1080P.team_points[index]
            if prediction.accepted:
                team_ids.append(prediction.item_id)
                team_stats.append(self._pet_stats(frame, point.x, shop=False))
                team_levels.append(self._pet_level(frame, point.x))
                continue
            expected = expected_team[index]
            if expected is None:
                team_ids.append(None)
                team_stats.append(None)
                team_levels.append(None)
                continue
            # A newly bought pet can occasionally be missed by the identity
            # model even though its independent stat digits and level bar are
            # clearly visible. In an expected-state verification pass, use
            # those two occupancy signals to retain the expected identity.
            # This cannot turn a visually empty slot into a pet: both parsers
            # must succeed and report positive health.
            try:
                stats = self._pet_stats(frame, point.x, shop=False)
                level = self._pet_level(frame, point.x)
            except PerceptionError:
                team_ids.append(None)
                team_stats.append(None)
                team_levels.append(None)
                continue
            if stats[1] <= 0 or level[0] not in {1, 2, 3}:
                team_ids.append(None)
                team_stats.append(None)
                team_levels.append(None)
                continue
            team_ids.append(expected.item_id)
            team_stats.append(stats)
            team_levels.append(level)
            expected_identity_fallbacks += 1
        team_ids = tuple(team_ids)
        team_stats = tuple(team_stats)
        team_levels = tuple(team_levels)
        confidences = [item.confidence for item in pets + foods]
        confidences.extend(
            prediction.probability
            for prediction in team_predictions
            if prediction.accepted
        )
        confidences.extend([0.75] * expected_identity_fallbacks)
        return ShopObservation(
            phase="shop",
            pets=tuple(pets),
            foods=tuple(foods),
            team_ids=team_ids,
            team_stats=team_stats,
            team_levels=team_levels,
            confidence=min(confidences, default=1.0),
        )

    def _pet_stats(self, frame, center_x, *, shop):
        if self.digit_recognizer is None:
            return (None, None)
        try:
            attack, health = self.digit_recognizer.read_pair(
                frame, center_x, shop=shop
            )
        except ValueError as error:
            raise PerceptionError(
                f"Could not read pet stats at x={center_x:.3f}: {error}"
            ) from error
        return (attack.value, health.value)

    @staticmethod
    def _pet_level(frame, center_x):
        try:
            reading = read_level_experience(frame, center_x)
        except ValueError as error:
            raise PerceptionError(
                f"Could not read pet level at x={center_x:.3f}: {error}"
            ) from error
        return (reading.level, reading.experience)

    @staticmethod
    def verify_identities(observation: ShopObservation, state):
        expected_team = tuple(
            None if pet is None else pet.item_id for pet in state.team.slots
        )
        if observation.team_ids != expected_team:
            raise PerceptionError(
                "Team identity mismatch: "
                f"expected={expected_team} observed={observation.team_ids}"
            )
        expected_pets = tuple(
            slot.pet.item_id
            for slot in state.shop
            if slot.kind in {"pet", "levelup"}
        )
        expected_foods = tuple(
            slot.food.visual_id for slot in state.shop if slot.kind == "food"
        )
        observed_pets = tuple(item.item_id for item in observation.pets)
        observed_foods = tuple(item.item_id for item in observation.foods)
        if observed_pets != expected_pets or observed_foods != expected_foods:
            raise PerceptionError(
                "Shop identity mismatch: "
                f"expected pets={expected_pets}, foods={expected_foods}; "
                f"observed pets={observed_pets}, foods={observed_foods}"
            )

    @staticmethod
    def scalar_fingerprints(frame):
        """Capture stable Arena life/win glyphs for post-battle deltas."""

        return {
            "lives": frame[18:52, 178:218].copy(),
            "wins": frame[18:52, 402:433].copy(),
        }

    @staticmethod
    def scalar_changed(before, after, key, threshold=8.0):
        difference = np.abs(
            before[key].astype(np.int16) - after[key].astype(np.int16)
        )
        return float(difference.mean()) > threshold

    def battle_is_playing(self, frame) -> bool:
        """The Play control shows two white pause bars while autoplay runs."""

        crop = frame[42:90, 815:875]
        white = np.all(crop > 225, axis=2).astype(np.uint8)
        count, _, stats, _ = self.cv2.connectedComponentsWithStats(white)
        bars = 0
        for index in range(1, count):
            _, _, width, height, area = map(int, stats[index])
            if area >= 60 and height >= 20 and width <= 20:
                bars += 1
        return bars >= 2

    def confirmation_prompt_visible(self, frame) -> bool:
        """Detect the centered two-button confirmation used by held food."""

        if frame.shape[:2] != (1080, 1920):
            return False
        hsv = self.cv2.cvtColor(frame, self.cv2.COLOR_BGR2HSV)
        orange = self.cv2.inRange(
            hsv,
            np.array([5, 170, 120], dtype=np.uint8),
            np.array([25, 255, 255], dtype=np.uint8),
        )
        white = np.all(frame > 225, axis=2)
        modal_white = float(white[405:665, 380:1540].mean())
        cancel_orange = float(orange[545:635, 400:950].mean() / 255.0)
        confirm_orange = float(orange[545:635, 970:1520].mean() / 255.0)
        return (
            modal_white > 0.32
            and cancel_orange > 0.45
            and confirm_orange > 0.45
        )

    @staticmethod
    def battle_controls_visible(frame) -> bool:
        """Detect Play/Autoplay/Fast controls independent of battle art."""

        regions = (
            frame[5:100, 800:890],
            frame[5:100, 915:1005],
            frame[5:100, 1030:1120],
        )
        return all(
            float(np.all(region > 220, axis=2).mean()) > 0.025
            for region in regions
        )
