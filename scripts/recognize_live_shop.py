#!/usr/bin/env python3
"""Run conservative gallery matching on calibrated live shop crops."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2

from sapai_gym1.deployment import SpriteCatalog, SpriteMatcher


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("screenshot", type=Path)
    parser.add_argument("--sprite-root", type=Path, required=True)
    return parser.parse_args()


def main():
    args = parse_args()
    image = cv2.imread(str(args.screenshot))
    if image is None or image.shape[:2] != (1080, 1920):
        raise SystemExit("Expected a readable 1920x1080 screenshot")
    catalog = SpriteCatalog.from_replay_bot(args.sprite_root)
    matcher = SpriteMatcher(catalog)
    results = {"pets": [], "foods": []}
    for index, center_x in enumerate((384, 528, 672)):
        crop = image[625:765, center_x - 90 : center_x + 90]
        results["pets"].append(
            {"index": index, **matcher.match_pet_body(crop).to_dict()}
        )
    crop = image[570:790, 1536 - 115 : 1536 + 115]
    results["foods"].append(
        {"index": 3, **matcher.match_food(crop).to_dict()}
    )
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
