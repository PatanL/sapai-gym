#!/usr/bin/env python3
"""Extract calibrated native-resolution crops for vision labeling/training."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from PIL import Image

from sapai_gym1.deployment import CANONICAL_CANVAS, ViewportTransform
from sapai_gym1.deployment.screen_profile import (
    DESKTOP_1080P,
    ordinary_shop_layout,
)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("screenshot", type=Path)
    parser.add_argument("--turn", type=int, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def main():
    args = parse_args()
    image = Image.open(args.screenshot).convert("RGB")
    if image.size != (CANONICAL_CANVAS.width, CANONICAL_CANVAS.height):
        raise SystemExit(
            f"Expected canonical 1920x1080 screenshot, got {image.size}"
        )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    transform = ViewportTransform.full_viewport(CANONICAL_CANVAS)
    layout = ordinary_shop_layout(args.turn)
    crops = []

    def save_crop(name, kind, index, rectangle):
        x, y, width, height = transform.rect_to_pixels(rectangle)
        output = args.output_dir / f"{name}.png"
        image.crop((x, y, x + width, y + height)).save(output)
        crops.append(
            {
                "name": name,
                "kind": kind,
                "index": index,
                "file": output.name,
                "pixels": {"x": x, "y": y, "width": width, "height": height},
                "normalized": {
                    "x": rectangle.x,
                    "y": rectangle.y,
                    "width": rectangle.width,
                    "height": rectangle.height,
                },
            }
        )

    for index in range(5):
        save_crop(
            f"team-{index}", "team", index, DESKTOP_1080P.team_crop(index)
        )
    for index in range(layout.pet_slots):
        save_crop(
            f"shop-pet-{index}",
            "shop-pet",
            index,
            DESKTOP_1080P.pet_shop_crop(index),
        )
    for index in range(layout.food_slots):
        save_crop(
            f"shop-food-{index}",
            "shop-food",
            layout.pet_slots + index,
            DESKTOP_1080P.food_shop_crop(index, layout.food_slots),
        )
    for name in ("gold", "lives", "turn", "wins"):
        save_crop(name, "scalar", None, getattr(DESKTOP_1080P, f"{name}_rect"))

    source_digest = hashlib.sha256(args.screenshot.read_bytes()).hexdigest()
    manifest = {
        "schemaVersion": 1,
        "source": str(args.screenshot),
        "sourceSha256": source_digest,
        "canvas": {"width": image.width, "height": image.height},
        "turn": args.turn,
        "shopLayout": {
            "petSlots": layout.pet_slots,
            "foodSlots": layout.food_slots,
        },
        "crops": crops,
    }
    (args.output_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n"
    )
    print(f"crops={len(crops)} manifest={args.output_dir / 'manifest.json'}")


if __name__ == "__main__":
    main()
