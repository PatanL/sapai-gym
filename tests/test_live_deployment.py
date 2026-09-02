import json
import unittest
from pathlib import Path

import cv2

from sapai_gym1.deployment import (
    BoardView,
    FoodView,
    LiveGameState,
    LiveStateTracker,
    PetView,
    ShopSlotView,
    SpriteCatalog,
    DESKTOP_1080P,
    CanvasSpec,
    ViewportTransform,
)
from sapai_gym1.deployment.policy_adapter import (
    LivePolicyEngine,
    environment_from_live_state,
    live_state_from_environment,
)
from sapai_gym1.deployment.live_perception import LiveScreenObserver
from sapai_gym1.deployment.stat_ocr import DigitTemplateRecognizer
from sapai_gym1.rl import ActionType, EntityObservationEncoder
from sapai_gym1.rl import AutoregressiveAction
from sapai_gym1.deployment.action_executor import (
    ClickSequenceGesture,
    DesktopActionMapper,
    DragGesture,
)


ROOT = Path(__file__).resolve().parents[2]


class TestLiveDeployment(unittest.TestCase):
    def test_measured_shop_and_reversed_team_geometry(self):
        transform = ViewportTransform.full_viewport(CanvasSpec(1920, 1080))
        self.assertEqual(
            transform.point_to_pixels(DESKTOP_1080P.team_points[0]),
            (960, 448),
        )
        self.assertEqual(
            transform.point_to_pixels(DESKTOP_1080P.team_points[4]),
            (384, 448),
        )
        self.assertEqual(
            transform.point_to_pixels(DESKTOP_1080P.shop_point(0, turn=1)),
            (384, 707),
        )
        self.assertEqual(
            transform.point_to_pixels(DESKTOP_1080P.shop_point(3, turn=1)),
            (1536, 707),
        )

    def test_buy_pet_maps_to_reversed_team_drag(self):
        gesture = DesktopActionMapper().gesture(
            AutoregressiveAction(ActionType.BUY_PET, 0, 4),
            state=LiveGameState(
                schema_version=1,
                phase="shop",
                turn=1,
                gold=10,
                lives=6,
                wins=0,
                opponent_lives=6,
                opponent_wins=0,
                team=BoardView(),
                shop=(
                    ShopSlotView(
                        0, "pet", pet=PetView("pet-pig", 4, 1)
                    ),
                ),
            ),
        )
        self.assertIsInstance(gesture, DragGesture)
        self.assertEqual(gesture.source, DESKTOP_1080P.pet_shop_points[0])
        self.assertEqual(gesture.target, DESKTOP_1080P.team_points[4])

    def test_compacted_food_source_stays_in_food_bank(self):
        state = LiveGameState(
            schema_version=1,
            phase="shop",
            turn=1,
            gold=7,
            lives=6,
            wins=0,
            opponent_lives=6,
            opponent_wins=0,
            team=BoardView(
                (None, None, None, None, PetView("pet-pig", 4, 1))
            ),
            shop=(
                ShopSlotView(0, "pet", pet=PetView("pet-horse", 2, 1)),
                ShopSlotView(1, "pet", pet=PetView("pet-horse", 2, 1)),
                ShopSlotView(
                    2,
                    "food",
                    food=FoodView("food-apple", "food-apple"),
                ),
            ),
        )
        gesture = DesktopActionMapper().gesture(
            AutoregressiveAction(ActionType.BUY_FOOD, 2, 4),
            state=state,
        )
        self.assertEqual(
            gesture.source, DESKTOP_1080P.food_points(1)[0]
        )

    def test_two_food_shop_uses_left_shifted_bank(self):
        kinds = ("pet", "pet", "pet", "pet", "food", "food")
        self.assertEqual(
            DESKTOP_1080P.shop_point_for_kinds(4, kinds),
            DESKTOP_1080P.food_points(2)[0],
        )
        self.assertEqual(
            DESKTOP_1080P.shop_point_for_kinds(5, kinds),
            DESKTOP_1080P.food_points(2)[1],
        )

    def test_targetless_team_food_uses_upper_activation_area(self):
        state = LiveGameState.from_dict(
            json.loads(
                (Path(__file__).parent / "fixtures/live_arena_turn5_state.json").read_text()
            )
        )
        gesture = DesktopActionMapper().gesture(
            AutoregressiveAction(ActionType.BUY_FOOD, 5, 5), state=state
        )
        self.assertIsInstance(gesture, DragGesture)
        self.assertEqual(gesture.source, DESKTOP_1080P.food_points(2)[1])
        self.assertEqual(
            gesture.target, DesktopActionMapper.team_food_activation_point
        )

    def test_team_combine_maps_to_select_destination_and_dismiss(self):
        state = LiveGameState.from_dict(
            json.loads(
                (Path(__file__).parent / "fixtures/live_arena_turn3_state.json").read_text()
            )
        )
        gesture = DesktopActionMapper().gesture(
            AutoregressiveAction(ActionType.COMBINE, 1, 0), state=state
        )
        self.assertIsInstance(gesture, ClickSequenceGesture)
        self.assertEqual(
            gesture.points,
            (
                DESKTOP_1080P.team_points[1],
                DESKTOP_1080P.team_points[0],
                DesktopActionMapper.dismiss_detail_point,
            ),
        )

    def test_sell_maps_to_select_then_sell_button(self):
        state = LiveGameState(
            schema_version=1,
            phase="shop",
            turn=3,
            gold=7,
            lives=6,
            wins=0,
            opponent_lives=6,
            opponent_wins=0,
            team=BoardView(
                (None, None, None, None, PetView("pet-pig", 4, 1))
            ),
            shop=(),
        )
        gesture = DesktopActionMapper().gesture(
            AutoregressiveAction(ActionType.SELL, 4, 5), state=state
        )
        self.assertIsInstance(gesture, ClickSequenceGesture)
        self.assertEqual(gesture.points[0], DESKTOP_1080P.team_points[4])
        self.assertEqual(
            gesture.points[1], DesktopActionMapper.sell_button_point
        )

    def test_pack_one_sprite_catalog_is_complete(self):
        catalog = SpriteCatalog.from_replay_bot(ROOT / "sap-replay-bot")
        self.assertEqual(len(catalog.entries_of_kind("pet")), 60)
        self.assertEqual(
            len(
                [
                    entry
                    for entry in catalog.entries_of_kind("food")
                    if entry.rollable
                ]
            ),
            18,
        )
        self.assertEqual(catalog.validate_files(), [])
        self.assertEqual(
            catalog.by_visual_id["food-cupcake"].sprite_path.name,
            "Muffin.png",
        )
        self.assertEqual(
            catalog.by_visual_id["food-sleeping-pill"].sprite_path.name,
            "Pill.png",
        )
        self.assertEqual(
            catalog.by_visual_id["food-cake"].sprite_path.name,
            "BirthdayCake.png",
        )

    def test_live_state_round_trip_and_two_board_tracker(self):
        ant = PetView("pet-ant", 2, 2)
        fish = PetView("pet-fish", 3, 4, experience=1)
        older = BoardView((ant, None, None, None, None))
        latest = BoardView((fish, None, None, None, None))
        state = LiveGameState(
            schema_version=1,
            phase="shop",
            turn=3,
            gold=10,
            lives=6,
            wins=1,
            opponent_lives=5,
            opponent_wins=1,
            team=BoardView((fish, None, None, None, None)),
            shop=(
                ShopSlotView(0, "pet", pet=ant),
                ShopSlotView(
                    1,
                    "food",
                    food=FoodView("food-apple", "food-apple"),
                ),
            ),
            opponent_history=(older, latest),
            last_battle_outcome="win",
        )
        restored = LiveGameState.from_dict(
            json.loads(json.dumps(state.to_dict()))
        )
        self.assertEqual(restored, state)

        tracker = LiveStateTracker()
        tracked = tracker.update(restored)
        self.assertEqual(tracked.opponent_history, (older, latest))
        third = BoardView((PetView("pet-camel", 3, 3), None, None, None, None))
        tracker.observe_opponent_board(third)
        self.assertEqual(tuple(tracker.opponent_history), (latest, third))

    def test_live_state_reconstructs_exact_legal_information_set(self):
        state = LiveGameState(
            schema_version=1,
            phase="shop",
            turn=1,
            gold=5,
            lives=5,
            wins=0,
            opponent_lives=6,
            opponent_wins=0,
            team=BoardView(
                (PetView("pet-fish", 2, 3), None, None, None, None)
            ),
            shop=(
                ShopSlotView(0, "pet", pet=PetView("pet-mosquito", 2, 2)),
                ShopSlotView(1, "pet", pet=PetView("pet-pigeon", 3, 2)),
                ShopSlotView(2, "pet", pet=PetView("pet-otter", 1, 4)),
                ShopSlotView(
                    3,
                    "food",
                    food=FoodView(
                        "food-apple", "food-apple", attack=1, health=1
                    ),
                ),
            ),
            action_count=3,
        )
        env = environment_from_live_state(state)
        observation = EntityObservationEncoder().encode(env, 0)
        self.assertTrue(observation["type_mask"][ActionType.ROLL])
        self.assertTrue(observation["type_mask"][ActionType.BUY_PET])
        self.assertTrue(observation["type_mask"][ActionType.BUY_FOOD])
        self.assertTrue(observation["type_mask"][ActionType.FREEZE])
        self.assertFalse(observation["type_mask"][ActionType.END_TURN])

    def test_environment_live_state_projection_round_trip(self):
        state = LiveGameState.from_dict(
            json.loads(
                (Path(__file__).parent / "fixtures/live_arena_turn3_state.json").read_text()
            )
        )
        projected = live_state_from_environment(environment_from_live_state(state))
        self.assertEqual(projected.turn, state.turn)
        self.assertEqual(projected.gold, state.gold)
        self.assertEqual(projected.team, state.team)
        self.assertEqual(projected.shop, state.shop)

    def test_fixed_screen_phase_detection(self):
        observer = LiveScreenObserver.__new__(LiveScreenObserver)
        observer.cv2 = cv2
        shop = cv2.imread(
            str(ROOT / "runs/live-captures/arena-turn1-shop-clean-1920x1080.png")
        )
        battle = cv2.imread(
            str(ROOT / "runs/live-captures/private-solving-after-turn2-submit.png")
        )
        self.assertEqual(observer.phase(shop), "shop")
        self.assertEqual(observer.phase(battle), "battle")
        self.assertEqual(observer.phase((shop * 0.2).astype("uint8")), "result")

    def test_stat_ocr_reads_real_team_and_shop_badges(self):
        reader = DigitTemplateRecognizer(
            ROOT / "runs/live-vision/stat-digits-v1.npz"
        )
        frame = cv2.imread(
            str(ROOT / "runs/live-captures/arena-turn1-shop-clean-1920x1080.png")
        )
        self.assertEqual(
            tuple(item.value for item in reader.read_pair(frame, 0.5)),
            (2, 3),
        )
        self.assertEqual(
            tuple(
                item.value
                for item in reader.read_pair(frame, 0.2, shop=True)
            ),
            (2, 2),
        )
        self.assertEqual(
            tuple(
                item.value
                for item in reader.read_pair(frame, 0.275, shop=True)
            ),
            (3, 2),
        )


if __name__ == "__main__":
    unittest.main()
