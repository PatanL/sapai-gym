"""OpenCV-DNN inference for the exported live pet classifier."""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path

import numpy as np


@dataclass(frozen=True)
class PetPrediction:
    item_id: str
    probability: float
    margin: float
    accepted: bool
    alternatives: tuple[tuple[str, float], ...]

    def to_dict(self):
        return {
            "itemId": self.item_id,
            "probability": self.probability,
            "margin": self.margin,
            "accepted": self.accepted,
            "alternatives": [
                {"itemId": item_id, "probability": probability}
                for item_id, probability in self.alternatives
            ],
        }


class OnnxPetClassifier:
    def __init__(
        self,
        model_path: Path,
        *,
        minimum_probability: float = 0.90,
        minimum_margin: float = 0.50,
    ):
        try:
            import cv2
        except ImportError as error:
            raise RuntimeError("ONNX classifier requires opencv-python") from error
        self.cv2 = cv2
        self.model_path = Path(model_path)
        metadata = json.loads(self.model_path.with_suffix(".json").read_text())
        self.class_ids = tuple(metadata["classIds"])
        self.image_size = int(metadata["imageSize"])
        self.mean = np.asarray(
            metadata["normalization"]["mean"], dtype=np.float32
        )[None, None]
        self.std = np.asarray(
            metadata["normalization"]["std"], dtype=np.float32
        )[None, None]
        self.minimum_probability = float(minimum_probability)
        self.minimum_margin = float(minimum_margin)
        self.network = cv2.dnn.readNetFromONNX(str(self.model_path))

    def predict(self, crops) -> tuple[PetPrediction, ...]:
        batch = []
        for crop in crops:
            image = self.cv2.resize(
                crop,
                (self.image_size, self.image_size),
                interpolation=self.cv2.INTER_AREA,
            )
            image = self.cv2.cvtColor(image, self.cv2.COLOR_BGR2RGB)
            image = image.astype(np.float32) / 255.0
            image = (image - self.mean) / self.std
            batch.append(image.transpose(2, 0, 1))
        self.network.setInput(np.stack(batch))
        logits = self.network.forward()
        logits -= logits.max(axis=1, keepdims=True)
        probabilities = np.exp(logits)
        probabilities /= probabilities.sum(axis=1, keepdims=True)
        output = []
        for row in probabilities:
            order = np.argsort(row)[::-1][:5]
            first, second = int(order[0]), int(order[1])
            probability = float(row[first])
            margin = probability - float(row[second])
            output.append(
                PetPrediction(
                    item_id=self.class_ids[first],
                    probability=probability,
                    margin=margin,
                    accepted=(
                        probability >= self.minimum_probability
                        and margin >= self.minimum_margin
                    ),
                    alternatives=tuple(
                        (self.class_ids[int(index)], float(row[index]))
                        for index in order[1:]
                    ),
                )
            )
        return tuple(output)
