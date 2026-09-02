#!/usr/bin/env python3
"""Train the entity transformer with batched symmetric V-trace self-play."""

from __future__ import annotations

import argparse
import copy
from dataclasses import asdict
import json
from pathlib import Path
import time

import numpy as np
import torch
from torch.utils.data import DataLoader

from sapai.turtle_pack import SIMULATOR_REVISION, SOURCE_OF_TRUTH
from sapai_gym1.replay import ReplaySequenceDataset, collate_replay_sequences
from sapai_gym1.rl import (
    ActionType,
    BatchedSelfPlayCollector,
    DistributedSelfPlayRuntime,
    EntityTransformerConfig,
    EntityTransformerPolicy,
    LeagueManager,
    LeagueMember,
    ParallelSelfPlayEnv,
    learner_update,
    behavior_cloning_loss,
)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--updates", type=int, default=1000)
    parser.add_argument("--num-envs", type=int, default=16)
    parser.add_argument("--segments-per-update", type=int, default=64)
    parser.add_argument("--transitions-per-update", type=int, default=8192)
    parser.add_argument(
        "--trajectory-queue-size",
        type=int,
        default=512,
        help="Shared turn-segment slots used to overlap actors and learner.",
    )
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--discount", type=float, default=0.99)
    parser.add_argument(
        "--battle-aux-cost",
        type=float,
        default=0.05,
        help="Weight for supervised per-battle win/loss/draw representation loss.",
    )
    parser.add_argument(
        "--opponent-aux-cost",
        type=float,
        default=0.1,
        help="Weight for public-history next-opponent-board prediction.",
    )
    parser.add_argument(
        "--opponent-module-learning-rate",
        type=float,
        help=(
            "Learning rate for the new opponent-history encoder and prediction "
            "heads. Defaults to --learning-rate."
        ),
    )
    parser.add_argument(
        "--opponent-adapter-learning-rate",
        type=float,
        help=(
            "Learning rate for the zero-initialized opponent-to-policy adapter. "
            "Defaults to --learning-rate."
        ),
    )
    parser.add_argument(
        "--discount-mode",
        choices=("per_action", "per_turn"),
        default="per_turn",
        help="Apply discount after every shop action or only across battles.",
    )
    parser.add_argument(
        "--learn-from-resident-sides",
        action="store_true",
        help=(
            "Also train main on frozen-resident trajectories. Disabled by "
            "default because resident GRU states are not main-policy states."
        ),
    )
    parser.add_argument(
        "--max-behavior-policy-lag",
        type=int,
        default=1,
        help="Discard segments older than this many main-policy updates; -1 disables.",
    )
    parser.add_argument(
        "--resume-learning-rate",
        type=float,
        help=(
            "Override optimizer param-group learning rates after loading a "
            "resume checkpoint. --learning-rate alone is intentionally "
            "restored from the checkpoint by PyTorch."
        ),
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--checkpoint-every", type=int, default=50)
    parser.add_argument("--checkpoint-dir", type=Path, default=Path("checkpoints/entity-selfplay"))
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--distributed", action="store_true")
    parser.add_argument("--games-per-actor", type=int, default=1)
    parser.add_argument(
        "--inference-batch-delay-ms",
        type=float,
        default=2.0,
        help="Maximum time to coalesce actor inference requests (milliseconds).",
    )
    parser.add_argument(
        "--move-action-cost",
        type=float,
        default=0.0,
        help="Optional per-MOVE reward cost; must be non-negative.",
    )
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--init-checkpoint", type=Path)
    parser.add_argument("--resume-checkpoint", type=Path)
    parser.add_argument("--allow-simulator-mismatch", action="store_true")
    parser.add_argument("--demo-data", nargs="*", type=Path)
    parser.add_argument("--demo-batch-size", type=int, default=64)
    parser.add_argument("--demo-until-update", type=int, default=250)
    parser.add_argument("--demo-every", type=int, default=4)
    parser.add_argument("--demo-learning-rate-scale", type=float, default=0.2)
    parser.add_argument(
        "--demo-move-action-weight",
        type=float,
        default=1.0,
        help="Relative MOVE weight in the optional demonstration update.",
    )
    parser.add_argument(
        "--expert-anchor-checkpoint",
        type=Path,
        help=(
            "Frozen expert policy kept in the resident league. Defaults to the "
            "initialization checkpoint; on resume it is recovered from league.json."
        ),
    )
    parser.add_argument(
        "--resident-checkpoint",
        action="append",
        type=Path,
        default=[],
        help=(
            "Additional revision-compatible frozen opponent checkpoint. "
            "May be repeated; useful for seeding a fresh league."
        ),
    )
    parser.add_argument(
        "--max-historical-residents",
        type=int,
        default=4,
        help=(
            "Newest rolling main snapshots kept resident in addition to all "
            "fixed exploiter checkpoints."
        ),
    )
    return parser.parse_args()


def save_checkpoint(
    path,
    policy,
    optimizer,
    config,
    update,
    transitions,
    *,
    discount_mode,
    discount,
    battle_aux_cost,
    opponent_aux_cost,
    opponent_module_learning_rate,
    opponent_adapter_learning_rate,
):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = path.with_suffix(path.suffix + ".tmp")
    torch.save(
        {
            "model": policy.state_dict(),
            "optimizer": optimizer.state_dict(),
            "config": asdict(config),
            "update": update,
            "transitions": transitions,
            "source_of_truth": SOURCE_OF_TRUTH,
            "simulator_revision": SIMULATOR_REVISION,
            "training_semantics": {
                "discount_mode": str(discount_mode),
                "discount": float(discount),
                "battle_aux_cost": float(battle_aux_cost),
                "opponent_aux_cost": float(opponent_aux_cost),
                "opponent_module_learning_rate": float(
                    opponent_module_learning_rate
                ),
                "opponent_adapter_learning_rate": float(
                    opponent_adapter_learning_rate
                ),
            },
        },
        temporary_path,
    )
    temporary_path.replace(path)


def move_batch(batch, device):
    for key, value in batch.items():
        if isinstance(value, dict):
            batch[key] = {
                name: tensor.to(device) for name, tensor in value.items()
            }
        elif torch.is_tensor(value):
            batch[key] = value.to(device)
    return batch


def demo_update(
    policy,
    optimizer,
    batch,
    device,
    learning_rate_scale,
    move_action_weight=1.0,
):
    batch = move_batch(batch, device)
    initial_memory = policy.initial_memory(
        batch["actions"].shape[1], device=device
    )
    autocast = (
        torch.autocast("cuda", dtype=torch.bfloat16)
        if device.type == "cuda"
        else torch.autocast("cpu", enabled=False)
    )
    with autocast:
        action_type_weights = torch.ones(len(ActionType), device=device)
        action_type_weights[int(ActionType.MOVE)] = move_action_weight
        loss = behavior_cloning_loss(
            policy,
            batch["observations"],
            batch["actions"],
            initial_memory,
            batch["valid_mask"],
            battle_outcomes=batch["battle_outcomes"],
            returns=batch["returns"],
            action_type_weights=action_type_weights,
        )
    original_rates = [group["lr"] for group in optimizer.param_groups]
    for group, rate in zip(optimizer.param_groups, original_rates):
        group["lr"] = rate * learning_rate_scale
    optimizer.zero_grad(set_to_none=True)
    loss.total.backward()
    torch.nn.utils.clip_grad_norm_(policy.parameters(), 10.0)
    optimizer.step()
    for group, rate in zip(optimizer.param_groups, original_rates):
        group["lr"] = rate
    return float(loss.total.detach()), float(
        loss.mean_expert_probability.detach()
    )


def behavior_lag_metrics(segments, current_version):
    versions = [
        transition.behavior_version
        for segment in segments
        for transition in segment.transitions
        if transition.behavior_version >= 0
    ]
    if not versions:
        return 0.0, 0, 0.0
    lags = [max(0, int(current_version) - version) for version in versions]
    return (
        float(sum(lags) / len(lags)),
        max(lags),
        float(sum(lag > 1 for lag in lags) / len(lags)),
    )


def main():
    args = parse_args()
    if args.move_action_cost < 0:
        raise SystemExit("--move-action-cost must be non-negative")
    if args.demo_move_action_weight <= 0:
        raise SystemExit("--demo-move-action-weight must be positive")
    if args.inference_batch_delay_ms < 0:
        raise SystemExit("--inference-batch-delay-ms must be non-negative")
    if args.trajectory_queue_size <= 0:
        raise SystemExit("--trajectory-queue-size must be positive")
    if not 0.0 < args.discount <= 1.0:
        raise SystemExit("--discount must be in (0, 1]")
    if args.battle_aux_cost < 0:
        raise SystemExit("--battle-aux-cost must be non-negative")
    if args.opponent_aux_cost < 0:
        raise SystemExit("--opponent-aux-cost must be non-negative")
    split_opponent_optimizer = (
        args.opponent_module_learning_rate is not None
        or args.opponent_adapter_learning_rate is not None
    )
    if args.opponent_module_learning_rate is None:
        args.opponent_module_learning_rate = args.learning_rate
    if args.opponent_adapter_learning_rate is None:
        args.opponent_adapter_learning_rate = args.learning_rate
    if args.opponent_module_learning_rate <= 0:
        raise SystemExit("--opponent-module-learning-rate must be positive")
    if args.opponent_adapter_learning_rate <= 0:
        raise SystemExit("--opponent-adapter-learning-rate must be positive")
    if args.max_behavior_policy_lag < -1:
        raise SystemExit("--max-behavior-policy-lag must be -1 or non-negative")
    if args.max_historical_residents < 0:
        raise SystemExit("--max-historical-residents must be non-negative")
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = torch.device(
        args.device if torch.cuda.is_available() or args.device == "cpu" else "cpu"
    )
    probe = ParallelSelfPlayEnv()
    encoder = probe.encoder
    if args.init_checkpoint and args.resume_checkpoint:
        raise SystemExit("Choose either --init-checkpoint or --resume-checkpoint")
    resume = (
        torch.load(args.resume_checkpoint, map_location="cpu", weights_only=False)
        if args.resume_checkpoint
        else None
    )
    if resume is not None:
        saved_semantics = resume.get("training_semantics", {})
        saved_discount_mode = saved_semantics.get("discount_mode")
        if saved_discount_mode == "per_turn" and args.discount_mode != "per_turn":
            raise SystemExit(
                "Refusing to resume a per-turn checkpoint with per-action "
                "discounting. Start a separately labeled diagnostic branch "
                "instead of changing the campaign semantics."
            )
    initialization = (
        torch.load(args.init_checkpoint, map_location="cpu", weights_only=False)
        if args.init_checkpoint
        else resume
    )
    if initialization is not None:
        checkpoint_revision = initialization.get("simulator_revision")
        if (
            checkpoint_revision != SIMULATOR_REVISION
            and not args.allow_simulator_mismatch
        ):
            raise SystemExit(
                "Checkpoint simulator revision mismatch: "
                f"checkpoint={checkpoint_revision!r} current={SIMULATOR_REVISION!r}. "
                "Use a freshly generated checkpoint or explicitly pass "
                "--allow-simulator-mismatch for diagnostic-only loading."
            )
    if initialization is not None:
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
    else:
        config = EntityTransformerConfig(
            num_items=encoder.num_items,
            num_statuses=encoder.num_statuses,
        )
    if args.smoke:
        args.updates = min(args.updates, 2)
        args.num_envs = min(args.num_envs, 2)
        args.segments_per_update = min(args.segments_per_update, 4)
        args.transitions_per_update = min(args.transitions_per_update, 16)
        args.games_per_actor = min(args.games_per_actor, 2)

    policy = EntityTransformerPolicy(config).to(device)
    if initialization is not None:
        policy.load_state_dict(initialization["model"])
        initialization_source = args.init_checkpoint or args.resume_checkpoint
        print(
            f"initialized_from={initialization_source} "
            f"examples={initialization.get('examples', 'unknown')}",
            flush=True,
        )
    if split_opponent_optimizer:
        history_prefixes = (
            "opponent_history_",
            "opponent_pet_head.",
            "opponent_status_head.",
            "opponent_level_head.",
            "opponent_stats_head.",
            "opponent_temporary_head.",
        )
        named_parameters = list(policy.named_parameters())
        base_parameters = [
            parameter
            for name, parameter in named_parameters
            if not name.startswith(history_prefixes)
            and not name.startswith("opponent_belief_adapter.")
        ]
        history_parameters = [
            parameter
            for name, parameter in named_parameters
            if name.startswith(history_prefixes)
        ]
        adapter_parameters = [
            parameter
            for name, parameter in named_parameters
            if name.startswith("opponent_belief_adapter.")
        ]
        optimizer = torch.optim.AdamW(
            [
                {
                    "params": base_parameters,
                    "lr": args.learning_rate,
                    "name": "base",
                },
                {
                    "params": history_parameters,
                    "lr": args.opponent_module_learning_rate,
                    "name": "opponent_history",
                },
                {
                    "params": adapter_parameters,
                    "lr": args.opponent_adapter_learning_rate,
                    "name": "opponent_adapter",
                },
            ],
            eps=1e-5,
        )
    else:
        optimizer = torch.optim.AdamW(
            policy.parameters(), lr=args.learning_rate, eps=1e-5
        )
    if resume is not None:
        optimizer.load_state_dict(resume["optimizer"])
        if args.resume_learning_rate is not None:
            if args.resume_learning_rate <= 0:
                raise SystemExit("--resume-learning-rate must be positive")
            for group in optimizer.param_groups:
                group_name = group.get("name", "base")
                if group_name == "opponent_history":
                    group["lr"] = args.opponent_module_learning_rate
                elif group_name == "opponent_adapter":
                    group["lr"] = args.opponent_adapter_learning_rate
                else:
                    group["lr"] = args.resume_learning_rate
            print(
                f"resume_learning_rate={args.resume_learning_rate} "
                f"opponent_module_learning_rate="
                f"{args.opponent_module_learning_rate} "
                f"opponent_adapter_learning_rate="
                f"{args.opponent_adapter_learning_rate}",
                flush=True,
            )
        print(
            f"resumed_from={args.resume_checkpoint} update={resume['update']} "
            f"transitions={resume['transitions']}",
            flush=True,
        )
    demo_loader = None
    demo_iterator = None
    if args.demo_data:
        demo_dataset = ReplaySequenceDataset(args.demo_data)
        demo_loader = DataLoader(
            demo_dataset,
            batch_size=min(args.demo_batch_size, len(demo_dataset)),
            shuffle=True,
            collate_fn=collate_replay_sequences,
        )
        demo_iterator = iter(demo_loader)
        print(
            f"demo_sequences={len(demo_dataset)} "
            f"demo_actions={demo_dataset.action_count}",
            flush=True,
        )
    collector = None
    runtime = None
    if args.distributed:
        sample_observation = probe.reset(seed=args.seed)[0]
        runtime = DistributedSelfPlayRuntime(
            policy,
            num_actors=args.num_envs,
            sample_observation=sample_observation,
            device=device,
            seed=args.seed,
            gamma=args.discount,
            use_historical_opponent=True,
            games_per_actor=args.games_per_actor,
            move_action_cost=args.move_action_cost,
            inference_batch_delay_ms=args.inference_batch_delay_ms,
            queue_size=args.trajectory_queue_size,
            learn_from_resident_sides=args.learn_from_resident_sides,
            discount_mode=args.discount_mode,
        )
        runtime.start()
    else:
        collector = BatchedSelfPlayCollector(
            policy,
            num_envs=args.num_envs,
            device=device,
            seed=args.seed,
            use_bfloat16=True,
            move_action_cost=args.move_action_cost,
            gamma=args.discount,
            learn_from_resident_sides=args.learn_from_resident_sides,
            discount_mode=args.discount_mode,
        )
    args.checkpoint_dir.mkdir(parents=True, exist_ok=True)
    league_path = args.checkpoint_dir / "league.json"
    if resume is not None and league_path.exists():
        league = LeagueManager.load(league_path)
        print(
            f"restored_league={league_path} members={len(league.members)}",
            flush=True,
        )
    else:
        league = LeagueManager()
    if "main" not in league.members:
        league.add(LeagueMember("main", "latest.pt", role="main"))
    def serialize_argument(value):
        if isinstance(value, Path):
            return str(value)
        if isinstance(value, list):
            return [serialize_argument(item) for item in value]
        return value

    serialized_arguments = {
        key: serialize_argument(value) for key, value in vars(args).items()
    }
    (args.checkpoint_dir / "config.json").write_text(
        json.dumps(
            {
                "model": asdict(config),
                "arguments": serialized_arguments,
                "source_of_truth": SOURCE_OF_TRUTH,
                "simulator_revision": SIMULATOR_REVISION,
            },
            indent=2,
            sort_keys=True,
        )
    )

    transitions = int(resume.get("transitions", 0)) if resume else 0
    first_update = int(resume.get("update", 0)) + 1 if resume else 1
    resident_opponents = {}
    resident_member_ids = {}
    resident_checkpoint_paths = set()
    next_policy_id = 1

    def load_frozen_policy(checkpoint_path):
        frozen_checkpoint = torch.load(
            checkpoint_path, map_location="cpu", weights_only=False
        )
        checkpoint_revision = frozen_checkpoint.get("simulator_revision")
        if (
            checkpoint_revision != SIMULATOR_REVISION
            and not args.allow_simulator_mismatch
        ):
            raise SystemExit(
                "Resident checkpoint simulator revision mismatch: "
                f"checkpoint={checkpoint_revision!r} "
                f"current={SIMULATOR_REVISION!r} path={checkpoint_path}"
            )
        frozen = EntityTransformerPolicy(
            EntityTransformerConfig(**frozen_checkpoint["config"])
        ).to(device)
        frozen.load_state_dict(frozen_checkpoint["model"])
        frozen.eval()
        return frozen

    expert_anchor_path = args.expert_anchor_checkpoint or args.init_checkpoint
    if expert_anchor_path is None and resume is not None:
        saved_expert = next(
            (
                member
                for member in league.members.values()
                if member.role == "expert"
            ),
            None,
        )
        if saved_expert is not None:
            expert_anchor_path = Path(saved_expert.checkpoint)
    if expert_anchor_path is not None:
        expert_anchor = load_frozen_policy(expert_anchor_path)
        resident_opponents[next_policy_id] = expert_anchor
        resident_member_ids[next_policy_id] = "expert-anchor"
        if "expert-anchor" not in league.members:
            league.add(
                LeagueMember(
                    "expert-anchor",
                    str(expert_anchor_path),
                    role="expert",
                    created_step=0,
                )
            )
        next_policy_id += 1
    if resume is not None:
        for member in league.resident_members(args.max_historical_residents):
            checkpoint_path = Path(member.checkpoint)
            if not checkpoint_path.exists():
                continue
            resident_opponents[next_policy_id] = load_frozen_policy(checkpoint_path)
            resident_member_ids[next_policy_id] = member.member_id
            resident_checkpoint_paths.add(checkpoint_path.resolve())
            next_policy_id += 1
    expert_resolved = (
        expert_anchor_path.resolve()
        if expert_anchor_path is not None
        else None
    )
    for explicit_index, checkpoint_path in enumerate(
        args.resident_checkpoint
    ):
        if not checkpoint_path.exists():
            raise SystemExit(
                f"Resident checkpoint does not exist: {checkpoint_path}"
            )
        if (
            expert_resolved is not None
            and checkpoint_path.resolve() == expert_resolved
        ):
            continue
        if checkpoint_path.resolve() in resident_checkpoint_paths:
            continue
        member_id = (
            f"seeded-{explicit_index:02d}-{checkpoint_path.stem}"
        )
        resident_opponents[next_policy_id] = load_frozen_policy(
            checkpoint_path
        )
        resident_member_ids[next_policy_id] = member_id
        resident_checkpoint_paths.add(checkpoint_path.resolve())
        if member_id not in league.members:
            league.add(
                LeagueMember(
                    member_id,
                    str(checkpoint_path),
                    role="exploiter",
                    created_step=transitions,
                )
            )
        next_policy_id += 1
    if runtime is not None and resident_opponents:
        runtime.set_policy_pool(
            resident_opponents,
            main_probability=float(len(resident_opponents)),
        )
        print(
            "resident_pool="
            + ",".join(
                resident_member_ids[policy_id]
                for policy_id in sorted(resident_member_ids)
            ),
            flush=True,
        )
    elif collector is not None and resident_opponents:
        collector.set_opponent_policy(resident_opponents[max(resident_opponents)])
    starting_transitions = transitions
    started = time.perf_counter()
    try:
        for update in range(first_update, args.updates + 1):
            inference_batches_before = (
                runtime.batcher.inference_batches if runtime is not None else 0
            )
            inference_items_before = (
                runtime.batcher.inference_items if runtime is not None else 0
            )
            inference_seconds_before = (
                runtime.batcher.inference_seconds if runtime is not None else 0.0
            )
            coalesce_seconds_before = (
                runtime.batcher.coalesce_seconds if runtime is not None else 0.0
            )
            collect_started = time.perf_counter()
            ring_before = (
                runtime.trajectory_ring.stats() if runtime is not None else None
            )
            segments = []
            collected_transitions = 0
            discarded_lag_segments = 0
            discarded_lag_transitions = 0
            while collected_transitions < args.transitions_per_update:
                new_segments = (
                    runtime.collect_segments(args.segments_per_update)
                    if runtime is not None
                    else collector.collect(args.segments_per_update)
                )
                if (
                    runtime is not None
                    and args.max_behavior_policy_lag >= 0
                ):
                    current_version = runtime.batcher.policy_version
                    accepted_segments = []
                    for segment in new_segments:
                        versions = [
                            transition.behavior_version
                            for transition in segment.transitions
                            if transition.behavior_version >= 0
                        ]
                        max_lag = max(
                            (
                                max(0, current_version - version)
                                for version in versions
                            ),
                            default=0,
                        )
                        if max_lag > args.max_behavior_policy_lag:
                            discarded_lag_segments += 1
                            discarded_lag_transitions += len(
                                segment.transitions
                            )
                        else:
                            accepted_segments.append(segment)
                    new_segments = accepted_segments
                segments.extend(new_segments)
                collected_transitions += sum(
                    len(segment.transitions) for segment in new_segments
                )
            collect_seconds = time.perf_counter() - collect_started
            ring_after_collect = (
                runtime.trajectory_ring.stats() if runtime is not None else None
            )
            behavior_lag_mean, behavior_lag_max, behavior_lag_over_one = (
                behavior_lag_metrics(
                    segments, runtime.batcher.policy_version
                )
                if runtime is not None
                else (0.0, 0, 0.0)
            )
            match_results = runtime.drain_match_results() if runtime is not None else []
            wins = draws = losses = 0
            for policy_id, result in match_results:
                member_id = resident_member_ids.get(policy_id)
                if member_id is None:
                    continue
                outcome = 1 if result == 0 else -1 if result == 1 else 0
                league.record_result("main", member_id, outcome)
                wins += int(result == 0)
                losses += int(result == 1)
                draws += int(result == 2)
            learn_started = time.perf_counter()
            metrics = learner_update(
                policy,
                optimizer,
                segments,
                device,
                use_bfloat16=True,
                battle_aux_cost=args.battle_aux_cost,
                opponent_aux_cost=args.opponent_aux_cost,
            )
            demo_loss = None
            demo_probability = None
            if (
                demo_loader is not None
                and update <= args.demo_until_update
                and update % args.demo_every == 0
            ):
                try:
                    demo_batch = next(demo_iterator)
                except StopIteration:
                    demo_iterator = iter(demo_loader)
                    demo_batch = next(demo_iterator)
                demo_loss, demo_probability = demo_update(
                    policy,
                    optimizer,
                    demo_batch,
                    device,
                    args.demo_learning_rate_scale,
                    args.demo_move_action_weight,
                )
            if runtime is not None:
                runtime.publish_main_policy()
            learn_seconds = time.perf_counter() - learn_started
            ring_after_learn = (
                runtime.trajectory_ring.stats() if runtime is not None else None
            )
            transitions += metrics.transitions
            elapsed = time.perf_counter() - started
            inference_batches = (
                runtime.batcher.inference_batches - inference_batches_before
                if runtime is not None
                else 0
            )
            inference_items = (
                runtime.batcher.inference_items - inference_items_before
                if runtime is not None
                else 0
            )
            mean_inference_batch = (
                inference_items / inference_batches if inference_batches else 0.0
            )
            inference_seconds = (
                runtime.batcher.inference_seconds - inference_seconds_before
                if runtime is not None
                else 0.0
            )
            coalesce_seconds = (
                runtime.batcher.coalesce_seconds - coalesce_seconds_before
                if runtime is not None
                else 0.0
            )
            mean_inference_ms = (
                1000.0 * inference_seconds / inference_batches
                if inference_batches
                else 0.0
            )
            mean_coalesce_ms = (
                1000.0 * coalesce_seconds / inference_batches
                if inference_batches
                else 0.0
            )
            branch_transitions = transitions - starting_transitions
            update_fps = metrics.transitions / max(
                collect_seconds + learn_seconds, 1e-9
            )
            opponent_adapter_norm = float(
                policy.opponent_belief_adapter.weight.detach()
                .float()
                .norm()
                .cpu()
            )
            print(
                f"update={update} transitions={transitions} "
                f"fps={branch_transitions / elapsed:.1f} "
                f"update_fps={update_fps:.1f} "
                f"loss={metrics.total_loss:.4f} policy={metrics.policy_loss:.4f} "
                f"value={metrics.value_loss:.4f} entropy={metrics.entropy:.4f} "
                f"battle_aux={metrics.battle_aux_loss:.4f} "
                f"opponent_aux={metrics.opponent_aux_loss:.4f} "
                f"opponent_adapter_norm={opponent_adapter_norm:.4f} "
                f"grad={metrics.grad_norm:.3f} collect_s={collect_seconds:.3f} "
                f"learn_s={learn_seconds:.3f} infer_batch={mean_inference_batch:.1f} "
                f"infer_ms={mean_inference_ms:.2f} "
                f"coalesce_ms={mean_coalesce_ms:.2f} "
                f"ring={ring_before['ready'] if ring_before else 0}/"
                f"{ring_after_collect['ready'] if ring_after_collect else 0}/"
                f"{ring_after_learn['ready'] if ring_after_learn else 0} "
                f"ring_high={ring_after_learn['high_water'] if ring_after_learn else 0} "
                f"segments={len(segments)} "
                f"lag={behavior_lag_mean:.2f}/{behavior_lag_max}/"
                f"{behavior_lag_over_one:.3f} "
                f"lag_drop={discarded_lag_segments}/"
                f"{discarded_lag_transitions} "
                f"league_wdl={wins}/{draws}/{losses} "
                f"demo_loss={demo_loss if demo_loss is not None else 0.0:.4f} "
                f"demo_p={demo_probability if demo_probability is not None else 0.0:.4f}",
                flush=True,
            )
            if update % args.checkpoint_every == 0 or update == args.updates:
                checkpoint = args.checkpoint_dir / f"update-{update:06d}.pt"
                save_checkpoint(
                    checkpoint,
                    policy,
                    optimizer,
                    config,
                    update,
                    transitions,
                    discount_mode=args.discount_mode,
                    discount=args.discount,
                    battle_aux_cost=args.battle_aux_cost,
                    opponent_aux_cost=args.opponent_aux_cost,
                    opponent_module_learning_rate=(
                        args.opponent_module_learning_rate
                    ),
                    opponent_adapter_learning_rate=(
                        args.opponent_adapter_learning_rate
                    ),
                )
                member_id = f"historical-{update:06d}"
                league.add(
                    LeagueMember(
                        member_id,
                        str(checkpoint),
                        role="historical",
                        created_step=transitions,
                    )
                )
                league.save(league_path)
                historical_opponent = copy.deepcopy(policy).to(device).eval()
                if runtime is not None:
                    resident_opponents[next_policy_id] = historical_opponent
                    resident_member_ids[next_policy_id] = member_id
                    next_policy_id += 1
                    retained_member_ids = {
                        member.member_id
                        for member in league.resident_members(
                            args.max_historical_residents
                        )
                    }
                    for policy_id, resident_member_id in list(
                        resident_member_ids.items()
                    ):
                        if (
                            league.members[resident_member_id].role == "historical"
                            and resident_member_id not in retained_member_ids
                        ):
                            del resident_opponents[policy_id]
                            del resident_member_ids[policy_id]
                    print(
                        "resident_pool="
                        + ",".join(
                            resident_member_ids[policy_id]
                            for policy_id in sorted(resident_member_ids)
                        ),
                        flush=True,
                    )
                else:
                    collector.set_opponent_policy(historical_opponent)
            if runtime is not None and resident_opponents:
                priorities = [
                    max(
                        0.05,
                        1.0
                        - league.win_rate(
                            "main", resident_member_ids[policy_id]
                        ),
                    )
                    ** 2
                    for policy_id in resident_opponents
                ]
                runtime.set_policy_pool(
                    resident_opponents,
                    priorities,
                    main_probability=sum(priorities),
                )
    finally:
        if runtime is not None:
            runtime.stop()

    print(
        f"finished updates={args.updates} transitions={transitions} "
        f"parameters={policy.parameter_count():,} device={device}"
    )


if __name__ == "__main__":
    main()
