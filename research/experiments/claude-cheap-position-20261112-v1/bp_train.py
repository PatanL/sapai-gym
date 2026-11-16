"""Train the battle predictor: P(A wins / draw / B wins) for two five-slot boards."""
import json, math, sys, time
import numpy as np
import torch, torch.nn as nn, torch.nn.functional as F
from pathlib import Path

HERE = Path(__file__).resolve().parent; OUT = HERE / 'out'
sys.path.insert(0, str(HERE))
from bp_common import PETS, STATUSES, SLOTS


class BattleNet(nn.Module):
    def __init__(self, d=192, layers=4, heads=6):
        super().__init__()
        self.pet = nn.Embedding(len(PETS), d); self.status = nn.Embedding(len(STATUSES), d)
        self.slot = nn.Embedding(SLOTS, d); self.side = nn.Embedding(2, d); self.cls = nn.Parameter(torch.zeros(1, 1, d))
        self.num = nn.Sequential(nn.Linear(5, d), nn.GELU(), nn.Linear(d, d))
        layer = nn.TransformerEncoderLayer(d, heads, 4 * d, dropout=0.0, batch_first=True, norm_first=True)
        self.enc = nn.TransformerEncoder(layer, layers, enable_nested_tensor=False)
        self.head = nn.Sequential(nn.LayerNorm(d), nn.Linear(d, 3))

    def side_tokens(self, ids, st, num, side):
        lvl, atk, hp = num[..., 0], num[..., 1], num[..., 2]
        feats = torch.stack([lvl / 3, torch.log1p(atk), torch.log1p(hp), atk / 50, hp / 50], -1)
        slots = torch.arange(SLOTS, device=ids.device).expand_as(ids)
        return (self.pet(ids) + self.status(st) + self.num(feats) + self.slot(slots)
                + self.side(torch.full_like(ids, side)))

    def forward(self, a_ids, a_st, a_num, b_ids, b_st, b_num):
        x = torch.cat([self.cls.expand(a_ids.shape[0], -1, -1), self.side_tokens(a_ids, a_st, a_num, 0),
                       self.side_tokens(b_ids, b_st, b_num, 1)], 1)
        pad = torch.cat([torch.zeros_like(a_ids[:, :1], dtype=torch.bool), a_ids == 0, b_ids == 0], 1)
        return self.head(self.enc(x, src_key_padding_mask=pad)[:, 0])

    @torch.no_grad()
    def equity(self, a, b):
        """Symmetrised P(A wins) + .5 P(draw); a, b = (ids, st, num) tensors."""
        p = F.softmax(self(*a, *b).float(), -1); q = F.softmax(self(*b, *a).float(), -1)
        win = (p[:, 0] + q[:, 2]) / 2; draw = (p[:, 1] + q[:, 1]) / 2
        return win + 0.5 * draw


def load(name, dev):
    z = np.load(OUT / f'{name}.npz')
    t = lambda k, dt: torch.as_tensor(z[k].astype(dt), device=dev)
    return ((t('a_ids', np.int64), t('a_st', np.int64), t('a_num', np.float32)),
            (t('b_ids', np.int64), t('b_st', np.int64), t('b_num', np.float32)), t('y', np.float32))


def main():
    torch.manual_seed(0); dev = 'cuda'
    A, B, Y = load('bp-train', dev); cA, cB, cY = load('bp-calib', dev)
    n = len(Y); val = torch.arange(n, device=dev) % 50 == 0; tr = ~val
    model = BattleNet().to(dev)
    steps = int(sys.argv[1]) if len(sys.argv) > 1 else 8000; bs = 4096
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=0.01)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, 1e-3, total_steps=steps, pct_start=0.05)
    idx_tr = torch.nonzero(tr).squeeze(1); began = time.time()

    def batch_loss(ix, swap):
        a = tuple(x[ix] for x in A); b = tuple(x[ix] for x in B); y = Y[ix]
        if swap: a, b, y = b, a, y.flip(-1)
        target = y / y.sum(-1, keepdim=True)
        return -(target * F.log_softmax(model(*a, *b).float(), -1)).sum(-1).mean()

    for step in range(steps):
        ix = idx_tr[torch.randint(len(idx_tr), (bs,), device=dev)]
        with torch.autocast('cuda', dtype=torch.bfloat16):
            loss = 0.5 * (batch_loss(ix, False) + batch_loss(ix, True))
        opt.zero_grad(set_to_none=True); loss.backward(); nn.utils.clip_grad_norm_(model.parameters(), 1.0); opt.step(); sched.step()
        if step % 1000 == 0 or step == steps - 1:
            model.eval()
            with torch.no_grad(), torch.autocast('cuda', dtype=torch.bfloat16):
                vix = torch.nonzero(val).squeeze(1)[:20000]
                vloss = 0.5 * (batch_loss(vix, False) + batch_loss(vix, True)).item()
                eq = torch.cat([model.equity(tuple(x[i:i + 2000] for x in cA), tuple(x[i:i + 2000] for x in cB)) for i in range(0, len(cY), 2000)])
            emp = (cY[:, 0] + 0.5 * cY[:, 1]) / cY.sum(-1)
            mae = (eq - emp).abs().mean().item(); corr = torch.corrcoef(torch.stack([eq, emp]))[0, 1].item()
            print(json.dumps(dict(step=step, loss=round(loss.item(), 4), valLoss=round(vloss, 4), calibMAE=round(mae, 4),
                                  calibCorr=round(corr, 4), seconds=round(time.time() - began))), flush=True)
            model.train()
    torch.save(dict(model=model.state_dict(), pets=PETS, statuses=STATUSES), OUT / 'battle_net.pt')


if __name__ == '__main__':
    main()
