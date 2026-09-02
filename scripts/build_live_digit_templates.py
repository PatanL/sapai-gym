#!/usr/bin/env python3
"""Build a tiny stat-glyph model from real labeled SAP frames and exact font."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from sapai_gym1.deployment.screen_profile import DESKTOP_1080P
from sapai_gym1.deployment.stat_ocr import (
    extract_badge_glyphs,
    normalize_glyph,
)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--font", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--pair",
        action="append",
        default=[],
        metavar="SCREENSHOT:STATE_JSON",
    )
    return parser.parse_args()


def rendered_templates(font_path):
    font = ImageFont.truetype(str(font_path), 40)
    for digit in range(10):
        image = Image.new("L", (64, 64))
        ImageDraw.Draw(image).text((2, -2), str(digit), font=font, fill=255)
        yield digit, normalize_glyph(np.asarray(image) > 0)


def real_templates(screenshot, state_path):
    frame = cv2.imread(str(screenshot))
    if frame is None or frame.shape[:2] != (1080, 1920):
        raise ValueError(f"Invalid screenshot {screenshot}")
    state = json.loads(state_path.read_text())
    for index, pet in enumerate(state["team"]["slots"]):
        if pet is None:
            continue
        center = DESKTOP_1080P.team_points[index].x
        for health, value in (
            (False, int(pet["attack"])),
            (True, int(pet["health"])),
        ):
            glyphs = extract_badge_glyphs(
                frame, center, shop=False, health=health
            )
            digits = tuple(map(int, str(value)))
            if len(glyphs) != len(digits):
                continue
            yield from zip(digits, glyphs)
    pet_rank = 0
    for slot in state["shop"]:
        pet = slot.get("pet")
        if pet is None:
            continue
        center = DESKTOP_1080P.pet_shop_points[pet_rank].x
        pet_rank += 1
        for health, value in (
            (False, int(pet["attack"])),
            (True, int(pet["health"])),
        ):
            glyphs = extract_badge_glyphs(
                frame, center, shop=True, health=health
            )
            digits = tuple(map(int, str(value)))
            if len(glyphs) != len(digits):
                continue
            yield from zip(digits, glyphs)


def main():
    args = parse_args()
    rows = list(rendered_templates(args.font))
    for pair in args.pair:
        screenshot, state = pair.split(":", 1)
        rows.extend(real_templates(Path(screenshot), Path(state)))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        args.output,
        templates=np.stack([mask for _, mask in rows]).astype(np.uint8),
        labels=np.asarray([label for label, _ in rows], dtype=np.int64),
    )
    print(
        f"output={args.output} templates={len(rows)} "
        f"counts={np.bincount([label for label, _ in rows], minlength=10).tolist()}"
    )


if __name__ == "__main__":
    main()
