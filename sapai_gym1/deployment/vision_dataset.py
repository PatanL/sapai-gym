"""Synthetic exact-sprite crops for the lightweight live pet classifier."""

from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image, ImageEnhance, ImageFilter
import torch
from torch.utils.data import Dataset
from torchvision.transforms import v2

from .sprite_catalog import SpriteCatalog


class SyntheticPetDataset(Dataset):
    def __init__(
        self,
        sprite_root: Path,
        *,
        samples_per_class: int,
        seed: int,
        image_size: int = 128,
    ):
        self.sprite_root = Path(sprite_root)
        catalog = SpriteCatalog.from_replay_bot(self.sprite_root)
        self.entries = catalog.entries_of_kind("pet")
        self.samples_per_class = int(samples_per_class)
        self.seed = int(seed)
        self.image_size = int(image_size)
        if self.samples_per_class <= 0 or self.image_size <= 0:
            raise ValueError("Dataset sizes must be positive")
        self.assets = [
            Image.open(entry.sprite_path).convert("RGBA")
            for entry in self.entries
        ]
        background_paths = sorted(
            (self.sprite_root / "Sprite" / "Background").glob("*Build.png")
        )
        # Downsample once; workers can cheaply sample these exact game-art
        # patches without repeatedly decoding 2048x1680 PNGs.
        self.backgrounds = [
            Image.open(path).convert("RGB").resize((320, 262))
            for path in background_paths
        ]
        self.normalize = v2.Normalize(
            mean=(0.485, 0.456, 0.406),
            std=(0.229, 0.224, 0.225),
        )

    @property
    def class_ids(self):
        return [entry.visual_id for entry in self.entries]

    def __len__(self):
        return len(self.entries) * self.samples_per_class

    def __getitem__(self, index):
        class_index = index % len(self.entries)
        rng = np.random.default_rng(self.seed + index * 7919)
        canvas_width, canvas_height = 180, 140
        background = self.backgrounds[
            int(rng.integers(len(self.backgrounds)))
        ]
        max_x = background.width - canvas_width
        max_y = background.height - canvas_height
        x = int(rng.integers(max_x + 1))
        y = int(rng.integers(max_y + 1))
        canvas = background.crop(
            (x, y, x + canvas_width, y + canvas_height)
        ).convert("RGBA")
        if rng.random() < 0.35:
            canvas = canvas.filter(
                ImageFilter.GaussianBlur(float(rng.uniform(0.0, 1.2)))
            )

        sprite = self.assets[class_index]
        alpha = np.asarray(sprite.getchannel("A"))
        ys, xs = np.where(alpha > 10)
        sprite = sprite.crop(
            (xs.min(), ys.min(), xs.max() + 1, ys.max() + 1)
        )
        if rng.random() < 0.5:
            sprite = sprite.transpose(Image.Transpose.FLIP_LEFT_RIGHT)
        desired_height = int(rng.integers(100, 134))
        desired_width = round(sprite.width * desired_height / sprite.height)
        scale = min(1.0, 176 / desired_width)
        desired_width = max(1, round(desired_width * scale))
        desired_height = max(1, round(desired_height * scale))
        sprite = sprite.resize(
            (desired_width, desired_height), Image.Resampling.LANCZOS
        )
        paste_x = (canvas_width - desired_width) // 2 + int(
            rng.integers(-10, 11)
        )
        paste_y = (canvas_height - desired_height) // 2 + int(
            rng.integers(-6, 7)
        )
        paste_x = max(0, min(canvas_width - desired_width, paste_x))
        paste_y = max(0, min(canvas_height - desired_height, paste_y))
        canvas.alpha_composite(sprite, (paste_x, paste_y))
        canvas = ImageEnhance.Brightness(canvas.convert("RGB")).enhance(
            float(rng.uniform(0.88, 1.12))
        )
        canvas = ImageEnhance.Contrast(canvas).enhance(
            float(rng.uniform(0.9, 1.1))
        )
        tensor = v2.functional.to_image(canvas)
        tensor = v2.functional.resize(
            tensor, [self.image_size, self.image_size], antialias=True
        )
        tensor = v2.functional.to_dtype(
            tensor, dtype=torch.float32, scale=True
        )
        return self.normalize(tensor), class_index
