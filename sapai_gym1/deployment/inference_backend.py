"""Explicit deployment backend selection; never silently fall back from MPS."""
import torch


def inference_device(backend='cpu'):
    if backend not in ('cpu', 'mps'):
        raise ValueError('Deployment backend must be cpu or mps')
    if backend == 'mps' and not torch.backends.mps.is_available():
        raise RuntimeError('MPS requested but unavailable on this host/PyTorch build')
    return torch.device(backend)
