"""Learned END positioner.

PosNet scores one of our pre-END boards (pets in a candidate order, before END effects) against the opponent's
last revealed board, given the turn. It reads only what the deployed agent sees. It is the battle predictor
(bp_train.BattleNet) plus a turn embedding, initialised from battle_net.pt and trained on public labels
(planner._label_decision: every order, END effects included, against simulated opponent turns).
At END the planner scores every reachable order in one forward pass, with no clones and no battles.
"""
import os, sys
from pathlib import Path
import numpy as np
import torch, torch.nn as nn, torch.nn.functional as F

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from bp_train import BattleNet
from bp_common import encode, encode_many

MAX_TURN = 32


class PosNet(BattleNet):
    def __init__(self, **kw):
        super().__init__(**kw)
        d = self.cls.shape[-1]
        self.turn = nn.Embedding(MAX_TURN, d)
        nn.init.zeros_(self.turn.weight)  # starts as the battle predictor

    def forward(self, a_ids, a_st, a_num, b_ids, b_st, b_num, turn):
        cls = self.cls.expand(a_ids.shape[0], -1, -1) + self.turn(turn.clamp(0, MAX_TURN - 1))[:, None]
        x = torch.cat([cls, self.side_tokens(a_ids, a_st, a_num, 0), self.side_tokens(b_ids, b_st, b_num, 1)], 1)
        pad = torch.cat([torch.zeros_like(a_ids[:, :1], dtype=torch.bool), a_ids == 0, b_ids == 0], 1)
        return self.head(self.enc(x, src_key_padding_mask=pad)[:, 0])

    def value(self, *args):
        """Our equity: P(win) + 0.5 P(draw)."""
        p = F.softmax(self(*args).float(), -1)
        return p[:, 0] + 0.5 * p[:, 1]


def from_battle_net(path):
    net = PosNet()
    ck = torch.load(path, map_location='cpu', weights_only=False)
    missing, unexpected = net.load_state_dict(ck['model'], strict=False)
    assert set(missing) == {'turn.weight'} and not unexpected, (missing, unexpected)
    return net


class PosNetScorer:
    """Inference wrapper used by the planner. Device: MIRA_POSNET_DEVICE, else cuda when available (120 orders:
    about 11 ms on a shared GPU vs 54 ms on one CPU thread)."""
    def __init__(self, path, device=None):
        self.dev = device or os.environ.get('MIRA_POSNET_DEVICE') or ('cuda' if torch.cuda.is_available() else 'cpu')
        ck = torch.load(path, map_location='cpu', weights_only=False)
        self.net = PosNet().to(self.dev).eval()
        self.net.load_state_dict(ck['model'])
        self.score([{'team': []}, {'team': []}], {'team': []}, 1)  # warm-up

    @torch.no_grad()
    def score(self, boards, opp, turn):
        ids, st, num = encode_many(boards)
        oi, os_, on = encode(opp)
        n = len(boards)
        t = lambda x, dt: torch.as_tensor(np.ascontiguousarray(x).astype(dt), device=self.dev)
        a = (t(ids, np.int64), t(st, np.int64), t(num, np.float32))
        b = (t(np.repeat(oi[None], n, 0), np.int64), t(np.repeat(os_[None], n, 0), np.int64), t(np.repeat(on[None], n, 0), np.float32))
        tt = torch.full((n,), int(turn), dtype=torch.long, device=self.dev)
        return self.net.value(*a, *b, tt).tolist()
