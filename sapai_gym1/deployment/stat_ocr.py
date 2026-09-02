"""Template OCR for fixed-position attack/health badges."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np


@dataclass(frozen=True)
class StatReading:
    value: int
    confidence: float
    distances: tuple[float, ...]


@dataclass(frozen=True)
class LevelReading:
    level: int
    experience: int
    confidence: float


def read_level_experience(frame, center_x) -> LevelReading:
    """Decode the yellow level digit and discrete progress-bar fill."""

    try:
        import cv2
    except ImportError as error:
        raise RuntimeError("Level OCR requires opencv-python") from error
    center = round(center_x * 1920)
    crop = frame[265:335, center - 85 : center + 35]
    hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
    yellow = (
        (hsv[:, :, 0] > 15)
        & (hsv[:, :, 0] < 40)
        & (hsv[:, :, 1] > 130)
        & (hsv[:, :, 2] > 180)
    ).astype(np.uint8)
    count, _, stats, _ = cv2.connectedComponentsWithStats(yellow)
    components = [tuple(map(int, stats[index])) for index in range(1, count)]
    bars = [
        component
        for component in components
        if component[1] >= 40
        and component[2] >= 10
        and component[3] >= 8
        and component[4] >= 80
    ]
    bar_width = max((component[2] for component in bars), default=0)
    if bar_width >= 45:
        return LevelReading(3, 0, 1.0)
    digits = [
        component
        for component in components
        if component[1] < 25
        and component[3] >= 25
        and component[4] >= 150
    ]
    if not digits:
        raise ValueError("No level digit found")
    digit = max(digits, key=lambda component: component[4])
    if digit[2] <= 14:
        level = 1
        experience = 0 if bar_width == 0 else 1
        expected_width = 27 if experience else 0
    else:
        level = 2
        experience = int(np.clip(round(bar_width / 18), 0, 2))
        expected_width = 18 * experience
    confidence = max(0.0, 1.0 - abs(bar_width - expected_width) / 18)
    return LevelReading(level, experience, confidence)


def normalize_glyph(mask, size=32, content=28):
    try:
        import cv2
    except ImportError as error:
        raise RuntimeError("Stat OCR requires opencv-python") from error
    ys, xs = np.where(mask > 0)
    if not len(xs):
        raise ValueError("Cannot normalize an empty glyph")
    crop = mask[ys.min() : ys.max() + 1, xs.min() : xs.max() + 1]
    scale = min(content / crop.shape[1], content / crop.shape[0])
    width = max(1, round(crop.shape[1] * scale))
    height = max(1, round(crop.shape[0] * scale))
    crop = cv2.resize(
        crop.astype(np.uint8),
        (width, height),
        interpolation=cv2.INTER_NEAREST,
    ) > 0
    output = np.zeros((size, size), dtype=np.uint8)
    x = (size - width) // 2
    y = (size - height) // 2
    output[y : y + height, x : x + width] = crop
    return output


def extract_badge_glyphs(frame, center_x, *, shop=False, health=False):
    """Return left-to-right normalized glyph masks from one stat badge."""

    try:
        import cv2
    except ImportError as error:
        raise RuntimeError("Stat OCR requires opencv-python") from error
    x = round(center_x * 1920) + (36 if health else -36)
    y1, y2 = ((748, 806) if shop else (470, 520))
    crop = frame[y1:y2, x - 28 : x + 28]
    white = np.all(crop > 220, axis=2).astype(np.uint8)
    count, labels, stats, _ = cv2.connectedComponentsWithStats(white)
    components = []
    for label in range(1, count):
        cx, cy, width, height, area = map(int, stats[label])
        touches_edge = (
            cx == 0
            or cy == 0
            or cx + width == white.shape[1]
            or cy + height == white.shape[0]
        )
        if touches_edge or area < 70 or height < 18:
            continue
        components.append((cx, normalize_glyph(labels == label)))
    components.sort(key=lambda item: item[0])
    if not components:
        raise ValueError("No stat digit found")
    if len(components) > 2:
        components = sorted(
            components, key=lambda item: int(item[1].sum()), reverse=True
        )[:2]
        components.sort(key=lambda item: item[0])
    return tuple(mask for _, mask in components)


class DigitTemplateRecognizer:
    def __init__(
        self,
        model_path: Path,
        *,
        maximum_distance: float = 0.24,
        minimum_margin: float = 0.002,
    ):
        payload = np.load(Path(model_path))
        self.templates = payload["templates"].astype(bool)
        self.labels = payload["labels"].astype(np.int64)
        self.maximum_distance = float(maximum_distance)
        self.minimum_margin = float(minimum_margin)
        if self.templates.ndim != 3 or len(self.templates) != len(self.labels):
            raise ValueError("Invalid stat OCR template model")

    def _digit(self, glyph):
        distances = np.logical_xor(self.templates, glyph.astype(bool)).mean(
            axis=(1, 2)
        )
        by_digit = np.asarray(
            [distances[self.labels == digit].min() for digit in range(10)]
        )
        order = np.argsort(by_digit)
        best, second = int(order[0]), int(order[1])
        distance = float(by_digit[best])
        margin = float(by_digit[second] - distance)
        if distance > self.maximum_distance or margin < self.minimum_margin:
            raise ValueError(
                f"Ambiguous digit best={best} distance={distance:.3f} margin={margin:.3f}"
            )
        confidence = max(0.0, min(1.0, 1.0 - distance / self.maximum_distance))
        return best, confidence, tuple(float(value) for value in by_digit)

    def read(self, frame, center_x, *, shop=False, health=False):
        glyphs = extract_badge_glyphs(
            frame, center_x, shop=shop, health=health
        )
        digits = [self._digit(glyph) for glyph in glyphs]
        value = int("".join(str(digit[0]) for digit in digits))
        if not 0 <= value <= 50:
            raise ValueError(f"Stat value outside SAP bounds: {value}")
        return StatReading(
            value=value,
            confidence=min(digit[1] for digit in digits),
            distances=tuple(
                min(digit[2][candidate] for digit in digits)
                for candidate in range(10)
            ),
        )

    def read_pair(self, frame, center_x, *, shop=False):
        attack = self.read(frame, center_x, shop=shop, health=False)
        health = self.read(frame, center_x, shop=shop, health=True)
        return attack, health
