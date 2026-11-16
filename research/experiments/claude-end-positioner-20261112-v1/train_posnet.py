"""Train the learned END positioner (posnet.PosNet) on public order labels (out/labels-*.jsonl).

Each label decision has every reachable order of our pre-END board, the opponent's last revealed board, the turn,
and y[i] = equity of order i (END effects included) against simulated opponent turns (32 battles per order).
Loss = MSE(value, y) + lam * listwise KL(softmax(y / tau) || softmax(value / tau)) within each decision.
Games are split by seed: one seed in ten is held out.
usage: ./run.sh posnet-train 6 16g train_posnet.py [steps] [out-name]
"""
import glob, json, sys, time, zlib
from pathlib import Path
import numpy as np
import torch, torch.nn.functional as F

HERE = Path(__file__).resolve().parent; OUT = HERE / 'out'
sys.path.insert(0, str(HERE))
from posnet import PosNet, from_battle_net, MAX_TURN

TAU, LAM, MAXN = 0.05, 1.0, 120


def load_decisions():
    dec = []
    for f in sorted(glob.glob(str(OUT / 'labels-*.jsonl'))):
        for line in open(f):
            r = json.loads(line)
            for d in r.get('labels') or []:
                if len(d['y']) >= 2: dec.append((r['seed'], d))
    return dec


def tensors(decs, dev):
    """Padded (D, MAXN, 5) boards, (D, 5) opponent, (D,) turn, (D, MAXN) y and mask."""
    D = len(decs)
    a_ids = np.zeros((D, MAXN, 5), np.int64); a_st = np.zeros((D, MAXN, 5), np.int64); a_num = np.zeros((D, MAXN, 5, 3), np.float32)
    b_ids = np.zeros((D, 5), np.int64); b_st = np.zeros((D, 5), np.int64); b_num = np.zeros((D, 5, 3), np.float32)
    turn = np.zeros(D, np.int64); y = np.zeros((D, MAXN), np.float32); mask = np.zeros((D, MAXN), bool)
    for k, (_, d) in enumerate(decs):
        n = min(len(d['y']), MAXN)
        for i in range(n):
            ids, st, num = d['orders'][i]; a_ids[k, i] = ids; a_st[k, i] = st; a_num[k, i] = num
        b_ids[k], b_st[k], b_num[k] = d['opp']; turn[k] = min(d['turn'], MAX_TURN - 1)
        y[k, :n] = d['y'][:n]; mask[k, :n] = True
    T = lambda x: torch.as_tensor(x, device=dev)
    return tuple(map(T, (a_ids, a_st, a_num, b_ids, b_st, b_num, turn, y, mask)))


def values(net, X, ix):
    a_ids, a_st, a_num, b_ids, b_st, b_num, turn, y, mask = (x[ix] for x in X)
    B, N = a_ids.shape[:2]
    flat = mask.reshape(-1)
    rep = lambda t: t[:, None].expand(B, N, *t.shape[1:]).reshape(B * N, *t.shape[1:])[flat]
    v = net.value(a_ids.reshape(B * N, 5)[flat], a_st.reshape(B * N, 5)[flat], a_num.reshape(B * N, 5, 3)[flat],
                  rep(b_ids), rep(b_st), rep(b_num), rep(turn))
    out = torch.full((B * N,), -1e4, device=v.device); out[flat] = v
    return out.reshape(B, N), y, mask


def loss_fn(v, y, mask):
    mse = ((v - y) ** 2)[mask].mean()
    neg = torch.finfo(torch.float32).min
    tgt = F.softmax(y.masked_fill(~mask, neg) / TAU, -1)
    lp = F.log_softmax(v.masked_fill(~mask, neg) / TAU, -1)
    kl = (tgt * (torch.log(tgt.clamp_min(1e-12)) - lp)).masked_fill(~mask, 0).sum(-1).mean()
    return mse + LAM * kl, mse, kl


@torch.no_grad()
def evaluate(net, X):
    net.eval(); D = X[0].shape[0]; gain = []; regret = []; best = []; vs = []
    for i in range(0, D, 64):
        ix = torch.arange(i, min(D, i + 64), device=X[0].device)
        v, y, mask = values(net, X, ix)
        pick = v.masked_fill(~mask, -1e9).argmax(-1)
        yb = y.masked_fill(~mask, -1e9).max(-1).values; yp = y.gather(1, pick[:, None])[:, 0]
        gain.append(yp - y[:, 0]); regret.append(yb - yp); best.append(yb - y[:, 0])
        vs.append(((v - y) ** 2)[mask])
    net.train(); c = lambda z: torch.cat(z).mean().item()
    return dict(labelGain=round(c(gain), 4), labelBestGain=round(c(best), 4), regret=round(c(regret), 4), valMse=round(c(vs), 4))


def main():
    steps = int(sys.argv[1]) if len(sys.argv) > 1 else 3000
    name = sys.argv[2] if len(sys.argv) > 2 else 'posnet-v1'
    torch.manual_seed(0); dev = 'cuda' if torch.cuda.is_available() else 'cpu'
    decs = load_decisions()
    val = [d for d in decs if zlib.crc32(str(d[0]).encode()) % 10 == 0]; tr = [d for d in decs if zlib.crc32(str(d[0]).encode()) % 10 != 0]
    if not val or not tr:  # too few seeds for a by-seed split (smoke runs): every tenth decision instead
        val = decs[::10]; tr = [d for k, d in enumerate(decs) if k % 10]
    print(json.dumps(dict(decisions=len(decs), train=len(tr), val=len(val), orders=sum(len(d['y']) for _, d in decs))), flush=True)
    Xtr, Xva = tensors(tr, dev), tensors(val, dev)
    net = from_battle_net(OUT / 'battle_net.pt').to(dev)
    print(json.dumps(dict(step='zero-shot', **evaluate(net, Xva))), flush=True)
    opt = torch.optim.AdamW(net.parameters(), lr=3e-4, weight_decay=0.01)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, 3e-4, total_steps=steps, pct_start=max(0.05, 3 / steps))
    began = time.time(); bs = 48; best_val = None
    for step in range(steps):
        ix = torch.randint(len(tr), (bs,), device=dev)
        with torch.autocast(dev, dtype=torch.bfloat16, enabled=dev == 'cuda'):
            v, y, mask = values(net, Xtr, ix)
        loss, mse, kl = loss_fn(v.float(), y, mask)
        opt.zero_grad(set_to_none=True); loss.backward(); torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0); opt.step(); sched.step()
        if step % 250 == 0 or step == steps - 1:
            ev = evaluate(net, Xva)
            print(json.dumps(dict(step=step, loss=round(loss.item(), 4), mse=round(mse.item(), 4), kl=round(kl.item(), 4), **ev,
                                  seconds=round(time.time() - began))), flush=True)
            if best_val is None or ev['labelGain'] > best_val:
                best_val = ev['labelGain']
                torch.save(dict(model=net.state_dict(), step=step, val=ev, tau=TAU, lam=LAM, decisions=len(tr)), OUT / f'{name}.pt')
    print(json.dumps(dict(saved=f'{name}.pt', bestValLabelGain=best_val)), flush=True)


if __name__ == '__main__':
    main()
