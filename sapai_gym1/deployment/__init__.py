"""Live Super Auto Pets perception and control support."""

from .geometry import (
    CANONICAL_CANVAS,
    CanvasSpec,
    NormalizedPoint,
    NormalizedRect,
    ViewportTransform,
)
from .sprite_catalog import SpriteCatalog, SpriteEntry
from .sprite_matcher import CandidateScore, SpriteMatch, SpriteMatcher
from .pet_classifier import OnnxPetClassifier, PetPrediction
from .screen_profile import DESKTOP_1080P, DesktopScreenProfile, ShopLayout
from .state_schema import (
    BoardView,
    FoodView,
    LiveGameState,
    PetView,
    ShopSlotView,
)
from .state_tracker import LiveStateTracker

__all__ = [
    "CANONICAL_CANVAS",
    "CanvasSpec",
    "NormalizedPoint",
    "NormalizedRect",
    "ViewportTransform",
    "SpriteCatalog",
    "SpriteEntry",
    "CandidateScore",
    "SpriteMatch",
    "SpriteMatcher",
    "OnnxPetClassifier",
    "PetPrediction",
    "DESKTOP_1080P",
    "DesktopScreenProfile",
    "ShopLayout",
    "BoardView",
    "FoodView",
    "LiveGameState",
    "PetView",
    "ShopSlotView",
    "LiveStateTracker",
]
