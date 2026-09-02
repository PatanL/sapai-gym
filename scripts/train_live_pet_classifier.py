#!/usr/bin/env python3
"""Train a compact exact-sprite classifier and gate it on real shop crops."""

from __future__ import annotations

import argparse
from contextlib import nullcontext
import json
from pathlib import Path
import time

import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision.models import (
    MobileNet_V3_Small_Weights,
    mobilenet_v3_small,
)
from torchvision.transforms import v2

from sapai_gym1.deployment.vision_dataset import SyntheticPetDataset


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--sprite-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--epochs", type=int, default=6)
    parser.add_argument("--samples-per-class", type=int, default=160)
    parser.add_argument("--validation-per-class", type=int, default=24)
    parser.add_argument("--batch-size", type=int, default=192)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--learning-rate", type=float, default=2e-4)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seed", type=int, default=19)
    parser.add_argument("--real-screenshot", type=Path)
    return parser.parse_args()


def real_crops(path):
    from PIL import Image

    image = Image.open(path).convert("RGB")
    rows = [
        ("pet-mosquito", image.crop((384 - 90, 625, 384 + 90, 765))),
        ("pet-pigeon", image.crop((528 - 90, 625, 528 + 90, 765))),
        ("pet-otter", image.crop((672 - 90, 625, 672 + 90, 765))),
        ("pet-fish", image.crop((960 - 90, 350, 960 + 90, 490))),
    ]
    transform = v2.Compose(
        [
            v2.ToImage(),
            v2.Resize((128, 128), antialias=True),
            v2.ToDtype(torch.float32, scale=True),
            v2.Normalize(
                mean=(0.485, 0.456, 0.406),
                std=(0.229, 0.224, 0.225),
            ),
        ]
    )
    return [(name, transform(image)) for name, image in rows]


def evaluate(model, loader, device):
    correct = total = 0
    model.eval()
    with torch.inference_mode():
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            predictions = model(images).argmax(-1)
            correct += int((predictions == labels).sum())
            total += labels.numel()
    return correct / max(1, total)


def main():
    args = parse_args()
    torch.manual_seed(args.seed)
    device = torch.device(
        args.device
        if args.device == "cpu" or torch.cuda.is_available()
        else "cpu"
    )
    train = SyntheticPetDataset(
        args.sprite_root,
        samples_per_class=args.samples_per_class,
        seed=args.seed,
    )
    validation = SyntheticPetDataset(
        args.sprite_root,
        samples_per_class=args.validation_per_class,
        seed=args.seed + 10_000_000,
    )
    train_loader = DataLoader(
        train,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.workers,
        persistent_workers=args.workers > 0,
        pin_memory=device.type == "cuda",
    )
    validation_loader = DataLoader(
        validation,
        batch_size=args.batch_size,
        num_workers=args.workers,
        persistent_workers=args.workers > 0,
        pin_memory=device.type == "cuda",
    )
    model = mobilenet_v3_small(weights=MobileNet_V3_Small_Weights.DEFAULT)
    model.classifier[-1] = nn.Linear(
        model.classifier[-1].in_features, len(train.class_ids)
    )
    model.to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.learning_rate, weight_decay=1e-4
    )
    scaler_context = (
        (lambda: torch.autocast("cuda", dtype=torch.bfloat16))
        if device.type == "cuda"
        else nullcontext
    )
    started = time.perf_counter()
    history = []
    for epoch in range(1, args.epochs + 1):
        model.train()
        running_loss = correct = total = 0
        for images, labels in train_loader:
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            with scaler_context():
                logits = model(images)
                loss = nn.functional.cross_entropy(logits, labels)
            loss.backward()
            optimizer.step()
            running_loss += float(loss.detach()) * labels.numel()
            correct += int((logits.argmax(-1) == labels).sum())
            total += labels.numel()
        validation_accuracy = evaluate(model, validation_loader, device)
        row = {
            "epoch": epoch,
            "loss": running_loss / total,
            "trainAccuracy": correct / total,
            "validationAccuracy": validation_accuracy,
        }
        history.append(row)
        print(json.dumps(row), flush=True)

    real_predictions = []
    if args.real_screenshot:
        class_index = {name: index for index, name in enumerate(train.class_ids)}
        model.eval()
        with torch.inference_mode():
            for expected, image in real_crops(args.real_screenshot):
                probabilities = model(image[None].to(device)).softmax(-1)[0]
                values, indices = probabilities.topk(5)
                real_predictions.append(
                    {
                        "expected": expected,
                        "correct": int(indices[0]) == class_index[expected],
                        "top": [
                            {
                                "itemId": train.class_ids[int(index)],
                                "probability": float(value),
                            }
                            for value, index in zip(values.cpu(), indices.cpu())
                        ],
                    }
                )
        print(json.dumps({"real": real_predictions}), flush=True)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model": model.state_dict(),
            "architecture": "mobilenet_v3_small",
            "classIds": train.class_ids,
            "imageSize": 128,
            "history": history,
            "realCalibration": real_predictions,
            "elapsedSeconds": time.perf_counter() - started,
        },
        args.output,
    )
    print(f"output={args.output}", flush=True)


if __name__ == "__main__":
    main()
