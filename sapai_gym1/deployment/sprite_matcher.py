"""Conservative exact-sprite matching for live pet/food crops."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .sprite_catalog import SpriteCatalog, SpriteEntry


@dataclass(frozen=True)
class CandidateScore:
    visual_id: str
    display_name: str
    score: float


@dataclass(frozen=True)
class SpriteMatch:
    best: CandidateScore
    alternatives: tuple[CandidateScore, ...]
    margin: float
    accepted: bool

    def to_dict(self):
        return {
            "best": self.best.__dict__,
            "alternatives": [item.__dict__ for item in self.alternatives],
            "margin": self.margin,
            "accepted": self.accepted,
        }


class SpriteMatcher:
    """Use exact source art first and defer ambiguous crops to a classifier."""

    def __init__(
        self,
        catalog: SpriteCatalog,
        *,
        pet_min_score: float = 0.60,
        pet_min_margin: float = 0.12,
        food_min_score: float = 0.92,
        food_min_margin: float = 0.04,
    ):
        try:
            import cv2
        except ImportError as error:
            raise RuntimeError("SpriteMatcher requires opencv-python") from error
        self.cv2 = cv2
        self.catalog = catalog
        self.pet_min_score = pet_min_score
        self.pet_min_margin = pet_min_margin
        self.food_min_score = food_min_score
        self.food_min_margin = food_min_margin
        self._rgba = {
            entry.visual_id: cv2.imread(
                str(entry.sprite_path), cv2.IMREAD_UNCHANGED
            )
            for entry in catalog.entries
        }

    def match_pet_body(self, crop: np.ndarray) -> SpriteMatch:
        cv2 = self.cv2
        gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
        dark = (gray < 55).astype("uint8") * 255
        dark = cv2.morphologyEx(
            dark,
            cv2.MORPH_CLOSE,
            np.ones((5, 5), dtype=np.uint8),
        )
        contours, _ = cv2.findContours(
            dark, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )
        eligible = []
        for contour in contours:
            x, y, width, height = cv2.boundingRect(contour)
            area = cv2.contourArea(contour)
            center_x = x + width / 2
            if area > 200 and height > 35 and 0.1 * crop.shape[1] < center_x < 0.9 * crop.shape[1]:
                eligible.append((area, contour))
        if not eligible:
            raise ValueError("No pet outline component found")
        contour = max(eligible, key=lambda item: item[0])[1]
        x, y, width, height = cv2.boundingRect(contour)
        mask = np.zeros(gray.shape, dtype=np.uint8)
        cv2.drawContours(mask, [contour], -1, 255, -1)
        sprite_crop = crop[y : y + height, x : x + width]
        sprite_mask = mask[y : y + height, x : x + width]
        scores = [
            (
                self._pet_similarity(sprite_crop, sprite_mask, entry),
                entry,
            )
            for entry in self.catalog.entries_of_kind("pet")
        ]
        return self._result(
            scores, self.pet_min_score, self.pet_min_margin
        )

    def _pet_similarity(self, image, mask, entry: SpriteEntry) -> float:
        cv2 = self.cv2
        asset = self._rgba[entry.visual_id]
        ys, xs = np.where(asset[:, :, 3] > 20)
        asset = asset[
            ys.min() : ys.max() + 1,
            xs.min() : xs.max() + 1,
        ]
        resized_image = cv2.resize(image, (128, 128))
        resized_mask = cv2.resize(mask, (128, 128)) > 40
        best = -1.0
        for candidate in (asset, asset[:, ::-1]):
            candidate_rgb = cv2.resize(candidate[:, :, :3], (128, 128))
            candidate_mask = cv2.resize(candidate[:, :, 3], (128, 128)) > 40
            intersection = candidate_mask & resized_mask
            union = candidate_mask | resized_mask
            if intersection.sum() < 100:
                continue
            iou = float(intersection.sum() / max(1, union.sum()))
            expected = candidate_rgb[intersection].astype(np.float64)
            observed = resized_image[intersection].astype(np.float64)
            expected = (expected - expected.mean(0)) / (expected.std(0) + 20)
            observed = (observed - observed.mean(0)) / (observed.std(0) + 20)
            cosine = float(
                (expected * observed).sum()
                / (
                    np.sqrt((expected * expected).sum())
                    * np.sqrt((observed * observed).sum())
                    + 1e-9
                )
            )
            best = max(best, 0.45 * iou + 0.55 * cosine)
        return best

    def match_food(self, crop: np.ndarray) -> SpriteMatch:
        cv2 = self.cv2
        scores = []
        for entry in self.catalog.entries_of_kind("food"):
            if not entry.rollable:
                continue
            asset = self._rgba[entry.visual_id]
            best = -1.0
            for size in range(96, 157, 4):
                template = cv2.resize(asset[:, :, :3], (size, size))
                mask = cv2.resize(asset[:, :, 3], (size, size))
                if crop.shape[0] < size or crop.shape[1] < size:
                    continue
                result = cv2.matchTemplate(
                    crop,
                    template,
                    cv2.TM_CCORR_NORMED,
                    mask=mask,
                )
                best = max(best, float(np.nanmax(result)))
            scores.append((best, entry))
        return self._result(
            scores, self.food_min_score, self.food_min_margin
        )

    @staticmethod
    def _result(scores, minimum_score, minimum_margin):
        scores.sort(key=lambda item: item[0], reverse=True)
        rendered = tuple(
            CandidateScore(entry.visual_id, entry.display_name, float(score))
            for score, entry in scores[:5]
        )
        margin = rendered[0].score - rendered[1].score
        return SpriteMatch(
            best=rendered[0],
            alternatives=rendered[1:],
            margin=margin,
            accepted=(
                rendered[0].score >= minimum_score
                and margin >= minimum_margin
            ),
        )
