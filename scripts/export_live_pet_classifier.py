#!/usr/bin/env python3
"""Export the trained pet classifier to lightweight ONNX inference."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from torch import nn
from torchvision.models import mobilenet_v3_small


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main():
    args = parse_args()
    checkpoint = torch.load(
        args.checkpoint, map_location="cpu", weights_only=False
    )
    classes = checkpoint["classIds"]
    model = mobilenet_v3_small(weights=None)
    model.classifier[-1] = nn.Linear(
        model.classifier[-1].in_features, len(classes)
    )
    model.load_state_dict(checkpoint["model"])
    model.eval()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    torch.onnx.export(
        model,
        torch.zeros(1, 3, 128, 128),
        args.output,
        input_names=["images"],
        output_names=["logits"],
        dynamic_axes={"images": {0: "batch"}, "logits": {0: "batch"}},
        opset_version=17,
        dynamo=False,
    )
    metadata = {
        "schemaVersion": 1,
        "architecture": checkpoint["architecture"],
        "classIds": classes,
        "imageSize": checkpoint["imageSize"],
        "normalization": {
            "mean": [0.485, 0.456, 0.406],
            "std": [0.229, 0.224, 0.225],
        },
        "realCalibration": checkpoint.get("realCalibration", []),
    }
    metadata_path = args.output.with_suffix(".json")
    metadata_path.write_text(json.dumps(metadata, indent=2) + "\n")
    print(f"onnx={args.output} metadata={metadata_path}")


if __name__ == "__main__":
    main()
