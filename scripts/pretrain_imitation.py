#!/usr/bin/env python3
"""Pretrain the entity transformer from validated replay examples."""

from __future__ import annotations

import argparse
from dataclasses import asdict
import json
from pathlib import Path

import torch
from torch.utils.data import DataLoader, Subset

from sapai.turtle_pack import SIMULATOR_REVISION, SOURCE_OF_TRUTH
from sapai_gym1.replay import (
    ReplaySequenceDataset,
    collate_replay_sequences,
)
from sapai_gym1.rl import (
    ActionType,
    EntityTransformerConfig,
    EntityTransformerPolicy,
    behavior_cloning_loss,
)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("data", nargs="+", type=Path)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--output", type=Path, default=Path("checkpoints/imitation.pt"))
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--include-end-only-action-loss", action="store_true")
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--validation-fraction", type=float, default=0.2)
    parser.add_argument(
        "--move-action-weight",
        type=float,
        default=1.0,
        help="Relative BC loss weight for MOVE-labeled decisions.",
    )
    parser.add_argument(
        "--init-checkpoint",
        type=Path,
        help=(
            "Initialize model weights from a revision-compatible policy "
            "checkpoint before supervised fine-tuning."
        ),
    )
    return parser.parse_args()


def move_batch(batch, device):
    for key, value in batch.items():
        if isinstance(value, dict):
            batch[key] = {name: tensor.to(device) for name, tensor in value.items()}
        elif torch.is_tensor(value):
            batch[key] = value.to(device)
    return batch


@torch.no_grad()
def evaluate_accuracy(model, loader, device):
    model.eval()
    correct = torch.zeros(4, dtype=torch.long)
    total = 0
    for batch in loader:
        batch = move_batch(batch, device)
        time_steps, batch_size = batch["actions"].shape[:2]
        memory = model.initial_memory(batch_size, device=device)
        for time_index in range(time_steps):
            observation = {
                key: value[time_index]
                for key, value in batch["observations"].items()
            }
            output = model(
                observation, memory=memory, deterministic=True
            )
            valid = batch["valid_mask"][time_index]
            target = batch["actions"][time_index]
            component_correct = output.actions.eq(target) & valid[:, None]
            correct[:3] += component_correct.sum(dim=0).cpu()
            correct[3] += (
                output.actions.eq(target).all(dim=-1) & valid
            ).sum().cpu()
            total += int(valid.sum())
            memory = torch.where(valid[:, None], output.memory, memory)
    model.train()
    return {
        "type": float(correct[0]) / total,
        "source": float(correct[1]) / total,
        "target": float(correct[2]) / total,
        "full": float(correct[3]) / total,
    }


def main():
    args = parse_args()
    if args.move_action_weight <= 0:
        raise SystemExit("--move-action-weight must be positive")
    torch.manual_seed(args.seed)
    device = torch.device(
        args.device if torch.cuda.is_available() or args.device == "cpu" else "cpu"
    )
    dataset = ReplaySequenceDataset(args.data)
    if dataset.rejected:
        print(
            f"rejected_incompatible_examples={len(dataset.rejected)}",
            flush=True,
        )
    encoder = dataset.encoder
    initialization = (
        torch.load(
            args.init_checkpoint, map_location="cpu", weights_only=False
        )
        if args.init_checkpoint is not None
        else None
    )
    if initialization is not None:
        checkpoint_revision = initialization.get("simulator_revision")
        if checkpoint_revision != SIMULATOR_REVISION:
            raise SystemExit(
                "Initialization checkpoint simulator mismatch: "
                f"checkpoint={checkpoint_revision!r} "
                f"current={SIMULATOR_REVISION!r}"
            )
        config = EntityTransformerConfig(**initialization["config"])
    elif args.smoke:
        config = EntityTransformerConfig(
            num_items=encoder.num_items,
            num_statuses=encoder.num_statuses,
            d_model=64,
            nhead=4,
            num_layers=2,
            dim_feedforward=128,
            memory_dim=96,
        )
        args.epochs = min(args.epochs, 2)
    else:
        config = EntityTransformerConfig(
            num_items=encoder.num_items,
            num_statuses=encoder.num_statuses,
        )
    model = EntityTransformerPolicy(config).to(device)
    if initialization is not None:
        model.load_state_dict(initialization["model"])
        print(f"initialized_from={args.init_checkpoint}", flush=True)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate)
    action_type_weights = torch.ones(len(ActionType), device=device)
    action_type_weights[int(ActionType.MOVE)] = args.move_action_weight
    replay_ids = sorted(set(dataset.replay_ids))
    validation_count = (
        max(1, round(len(replay_ids) * args.validation_fraction))
        if len(replay_ids) > 1
        else 0
    )
    validation_ids = set(replay_ids[-validation_count:]) if validation_count else set()
    train_indices = [
        index for index, replay_id in enumerate(dataset.replay_ids)
        if replay_id not in validation_ids
    ]
    validation_indices = [
        index for index, replay_id in enumerate(dataset.replay_ids)
        if replay_id in validation_ids
    ]
    if not validation_indices:
        validation_indices = list(train_indices)
    train_dataset = Subset(dataset, train_indices)
    validation_dataset = Subset(dataset, validation_indices)
    loader = DataLoader(
        train_dataset,
        batch_size=min(args.batch_size, len(train_dataset)),
        shuffle=True,
        collate_fn=collate_replay_sequences,
    )
    validation_loader = DataLoader(
        validation_dataset,
        batch_size=min(args.batch_size, len(validation_dataset)),
        shuffle=False,
        collate_fn=collate_replay_sequences,
    )
    all_end_turn = all(
        int(example["action"][0]) == int(ActionType.END_TURN)
        for example in dataset.base.examples
    )
    action_cost = (
        1.0
        if args.include_end_only_action_loss or not all_end_turn
        else 0.0
    )
    if all_end_turn and action_cost == 0:
        print(
            "dataset contains only validated end-turn actions; training "
            "representation/value auxiliaries without policy loss",
            flush=True,
        )

    best_epoch = 0
    best_accuracy = evaluate_accuracy(model, validation_loader, device)
    best_model = {
        name: tensor.detach().cpu().clone()
        for name, tensor in model.state_dict().items()
    }
    print(
        "epoch=0 loss=nan expert_probability=nan "
        f"val_type={best_accuracy['type']:.3f} "
        f"val_source={best_accuracy['source']:.3f} "
        f"val_target={best_accuracy['target']:.3f} "
        f"val_full={best_accuracy['full']:.3f}",
        flush=True,
    )
    for epoch in range(1, args.epochs + 1):
        totals = []
        probabilities = []
        for batch in loader:
            batch = move_batch(batch, device)
            initial_memory = model.initial_memory(
                batch["actions"].shape[1], device=device
            )
            autocast = (
                torch.autocast("cuda", dtype=torch.bfloat16)
                if device.type == "cuda"
                else torch.autocast("cpu", enabled=False)
            )
            with autocast:
                loss = behavior_cloning_loss(
                    model,
                    batch["observations"],
                    batch["actions"],
                    initial_memory,
                    batch["valid_mask"],
                    battle_outcomes=batch["battle_outcomes"],
                    returns=batch["returns"],
                    action_cost=action_cost,
                    action_type_weights=action_type_weights,
                )
            optimizer.zero_grad(set_to_none=True)
            loss.total.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 10.0)
            optimizer.step()
            totals.append(float(loss.total.detach()))
            probabilities.append(float(loss.mean_expert_probability.detach()))
        accuracy = evaluate_accuracy(model, validation_loader, device)
        if best_accuracy is None or accuracy["full"] > best_accuracy["full"]:
            best_epoch = epoch
            best_accuracy = dict(accuracy)
            best_model = {
                name: tensor.detach().cpu().clone()
                for name, tensor in model.state_dict().items()
            }
        print(
            f"epoch={epoch} loss={sum(totals)/len(totals):.5f} "
            f"expert_probability={sum(probabilities)/len(probabilities):.5f} "
            f"val_type={accuracy['type']:.3f} val_source={accuracy['source']:.3f} "
            f"val_target={accuracy['target']:.3f} val_full={accuracy['full']:.3f}",
            flush=True,
        )

    if best_model is None:
        raise RuntimeError("Pretraining produced no validation checkpoint")
    model.load_state_dict(best_model)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model": model.state_dict(),
            "config": asdict(config),
            "source_of_truth": SOURCE_OF_TRUTH,
            "simulator_revision": SIMULATOR_REVISION,
            "examples": dataset.action_count,
            "sequences": len(dataset),
            "train_examples": sum(
                len(dataset.sequences[index]) for index in train_indices
            ),
            "validation_examples": sum(
                len(dataset.sequences[index]) for index in validation_indices
            ),
            "rejected_examples": len(dataset.rejected),
            "all_end_turn": all_end_turn,
            "action_cost": action_cost,
            "move_action_weight": args.move_action_weight,
            "best_epoch": best_epoch,
            "validation_accuracy": best_accuracy,
            "initialized_from": (
                str(args.init_checkpoint)
                if args.init_checkpoint is not None
                else None
            ),
        },
        args.output,
    )
    args.output.with_suffix(".json").write_text(
        json.dumps(
            {
                "examples": dataset.action_count,
                "sequences": len(dataset),
                "trainExamples": sum(
                    len(dataset.sequences[index]) for index in train_indices
                ),
                "validationExamples": sum(
                    len(dataset.sequences[index])
                    for index in validation_indices
                ),
                "rejectedExamples": len(dataset.rejected),
                "allEndTurn": all_end_turn,
                "actionCost": action_cost,
                "moveActionWeight": args.move_action_weight,
                "bestEpoch": best_epoch,
                "validationAccuracy": best_accuracy,
                "initializedFrom": (
                    str(args.init_checkpoint)
                    if args.init_checkpoint is not None
                    else None
                ),
                "model": asdict(config),
                "simulatorRevision": SIMULATOR_REVISION,
            },
            indent=2,
            sort_keys=True,
        )
    )
    print(
        f"saved={args.output} best_epoch={best_epoch} "
        f"best_val_full={best_accuracy['full']:.3f} "
        f"parameters={model.parameter_count():,}"
    )


if __name__ == "__main__":
    main()
