"""Battle-predictor board encoding (shared by data generation, training and the planner)."""
import json
from pathlib import Path
import numpy as np

HERE = Path(__file__).resolve().parent
_V = json.loads((HERE / 'out/vocab.json').read_text())
PETS = ['<empty>', '<other>'] + _V['pets']
STATUSES = ['none', '<other>', 'status-bread'] + [s for s in _V['statuses'] if s != 'status-bread']
PET_ID = {n: i for i, n in enumerate(PETS)}
STATUS_ID = {n: i for i, n in enumerate(STATUSES)}
SLOTS = 5


def encode(state):
    """Board state dict -> (ids[5] int, status[5] int, num[5,3] float: level, attack, health). Pets compacted to the
    front in team order; empty slots are id 0."""
    ids = np.zeros(SLOTS, np.int16); st = np.zeros(SLOTS, np.int16); num = np.zeros((SLOTS, 3), np.float32)
    k = 0
    for slot in state.get('team', []):
        p = slot.get('pet') or {}
        name = p.get('name')
        if name in (None, 'pet-none'): continue
        if k >= SLOTS: break
        ids[k] = PET_ID.get(name, 1); st[k] = STATUS_ID.get(p.get('status') or 'none', 1)
        num[k] = (p.get('level', 1), p.get('attack', 0), p.get('health', 0)); k += 1
    return ids, st, num


def encode_many(states):
    out = [encode(s) for s in states]
    return (np.stack([o[0] for o in out]), np.stack([o[1] for o in out]), np.stack([o[2] for o in out]))
