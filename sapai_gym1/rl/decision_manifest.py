"""Replayable evaluation states with separate public and hidden evidence."""
from __future__ import annotations

import hashlib
import json

import numpy as np
import torch

from sapai.turtle_pack import SIMULATOR_REVISION
from .selfplay_env import ParallelSelfPlayEnv
from .joint_action import JointAction, step_decisions


MANIFEST_VERSION = "paired-interactive-state-v2-joint-20260906"


def observation_hash(observation):
    digest = hashlib.sha256()
    for key, value in sorted(observation.items()):
        array = np.ascontiguousarray(value)
        digest.update(json.dumps([key, array.dtype.str, array.shape]).encode())
        digest.update(array.tobytes())
    return digest.hexdigest()


def decision_record(env, memories, side, reasons, replay, *, search_seed, validation_seed, truth_seeds):
    observations = env.observations()
    if side not in (0, 1) or not env.active_players()[side]:
        raise ValueError("Manifest acting side must be active")
    return {
        "version": MANIFEST_VERSION,
        "simulatorRevision": SIMULATOR_REVISION,
        "replay": replay,
        "side": int(side),
        "turn": int(env.core.turn),
        "gold": int((env.core.agent if side == 0 else env.core.opponent).gold),
        "reasons": list(reasons),
        "observationHashes": [observation_hash(obs) for obs in observations],
        "memories": [torch.as_tensor(memory).detach().cpu().float().tolist() for memory in memories],
        "searchSeed": int(search_seed),
        "validationSeed": int(validation_seed),
        "truthSeeds": list(map(int, truth_seeds)),
    }


def restore_decision(record):
    if record.get("version") not in (MANIFEST_VERSION, 'paired-interactive-state-v1-20260905'):
        raise ValueError("Unsupported decision manifest version")
    if record.get("simulatorRevision") != SIMULATOR_REVISION:
        raise ValueError("Decision manifest simulator revision mismatch")
    env = ParallelSelfPlayEnv()
    env.reset(seed=int(record["replay"]["resetSeed"]))
    for action_pair in record["replay"]["actions"]:
        if any(isinstance(a, dict) for a in action_pair):
            if record['version'] != MANIFEST_VERSION:
                raise ValueError('Typed replay requires the joint-aware manifest version')
            if any(a is not None and not isinstance(a, dict) for a in action_pair):
                raise ValueError('Mixed implicit/typed replay decisions')
            step = step_decisions(env, [None if a is None else JointAction.from_dict(a) for a in action_pair]).result
        else:
            step = env.step(*action_pair)
        if step.terminated:
            raise ValueError("Manifest ends after the match is terminal")
    expected = record["observationHashes"]
    actual = [observation_hash(obs) for obs in env.observations()]
    if expected != actual:
        raise ValueError("Replayed decision differs from its saved observations")
    side = record["side"]
    if side not in (0, 1) or not env.active_players()[side]:
        raise ValueError("Manifest acting side is not active")
    memories = [torch.tensor(memory, dtype=torch.float32) for memory in record["memories"]]
    if len(memories) != 2 or any(m.ndim != 1 or not torch.isfinite(m).all() for m in memories):
        raise ValueError("Manifest requires two finite recurrent vectors")
    return env, memories, side, tuple(record["reasons"])
