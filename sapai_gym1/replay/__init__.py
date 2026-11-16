"""Replay tools; importing the archive/parser must not initialize Torch/CUDA."""

import importlib


def __getattr__(name):
    modules = {
        "ReplayCode": "client", "SapLibraryClient": "client",
        "ParsedAction": "parser", "ReplayParser": "parser",
        "ReplayReconstructor": "reconstruct", "ReconstructionReport": "reconstruct",
        "ReplayExampleDataset": "dataset", "ReplaySequenceDataset": "dataset",
        "collate_replay_examples": "dataset", "collate_replay_sequences": "dataset",
        "semantic_state_to_observation": "dataset",
    }
    if name not in modules:
        raise AttributeError(name)
    value = getattr(importlib.import_module(f".{modules[name]}", __name__), name)
    globals()[name] = value
    return value

__all__ = [
    "ReplayCode",
    "SapLibraryClient",
    "ParsedAction",
    "ReplayParser",
    "ReplayReconstructor",
    "ReconstructionReport",
    "ReplayExampleDataset",
    "ReplaySequenceDataset",
    "collate_replay_examples",
    "collate_replay_sequences",
    "semantic_state_to_observation",
]
