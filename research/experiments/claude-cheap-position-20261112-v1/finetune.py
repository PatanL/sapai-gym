"""Distil opponent-aware positioning into the policy network (no search at inference).

Data: out/<data>/game-*.pt from `match.py --record`. Every planner-seat decision has the observation, the recurrent
memory, the executed joint action and the policy's own greedy action. Loss: -log p(executed action) on decisions
the planner overrode (its reorders and the END after them), plus a smaller weight on all other decisions (the
policy's own greedy action, so behaviour elsewhere stays put). Writes a checkpoint in the original format.
usage: finetune.py DATA_DIR OUT_NAME [steps] [lr] [w_keep] [kl_weight] [train=decoder|all]
A KL penalty to a frozen copy of the original policy (type/source/target distributions along the recorded action
path, on every sampled decision) keeps the rest of its behaviour in place; by default only the decision heads,
the context projection and the opponent-belief adapter are trained.
"""
import glob, json, random, sys, time
import numpy as np
import torch
from plan_common import install, ROOT, CURRENT, OUT, sha


def main():
    data_dir, out_name = sys.argv[1], sys.argv[2]
    steps = int(sys.argv[3]) if len(sys.argv) > 3 else 3000
    lr = float(sys.argv[4]) if len(sys.argv) > 4 else 3e-6
    w_keep = float(sys.argv[5]) if len(sys.argv) > 5 else 0.2
    kl_weight = float(sys.argv[6]) if len(sys.argv) > 6 else 1.0
    train_what = sys.argv[7] if len(sys.argv) > 7 else 'decoder'
    install(cuda=True)
    from sapai_gym1.rl.model import EntityTransformerPolicy, EntityTransformerConfig
    torch.manual_seed(0); dev = 'cuda'
    ck = torch.load(ROOT / CURRENT['path'], map_location='cpu', weights_only=False)
    model = EntityTransformerPolicy(EntityTransformerConfig(**ck['config'])); model.load_state_dict(ck['model'], strict=True)
    model.to(dev).train()
    import copy
    original = copy.deepcopy(model).eval().requires_grad_(False)
    heads = ('action_type_embedding', 'no_source_embedding', 'no_target_embedding', 'type_head', 'source_query', 'source_key',
             'target_query', 'target_key', 'context_projection', 'end_freeze_head', 'opponent_belief_adapter')
    for name, prm in model.named_parameters():
        prm.requires_grad_(train_what == 'all' or name.split('.')[0] in heads)
    print(json.dumps(dict(trainable=sum(p.numel() for p in model.parameters() if p.requires_grad), total=sum(p.numel() for p in model.parameters()))), flush=True)

    files = sorted(glob.glob(f'{data_dir}/game-*.pt')); random.Random(0).shuffle(files)
    held = set(files[:max(1, len(files) // 10)])
    rows = {'train': [], 'val': []}
    for f in files:
        for r in torch.load(f, weights_only=False):
            rows['val' if f in held else 'train'].append(r)

    def tensors(rs):
        obs = {k: torch.as_tensor(np.stack([r['obs'][k] for r in rs]), device=dev) for k in rs[0]['obs']}
        mem = torch.stack([r['memory'] for r in rs]).to(dev)
        act = torch.tensor([r['action']['primitive'] for r in rs], device=dev)
        msk = torch.tensor([r['action']['end_freeze_mask'] or 0 for r in rs], device=dev)
        forced = torch.tensor([float(r['forced']) for r in rs], device=dev)
        return obs, mem, act, msk, forced
    tr, va = tensors(rows['train']), tensors(rows['val'])
    n_forced = int(tr[4].sum().item())
    print(json.dumps(dict(trainDecisions=len(rows['train']), trainForced=n_forced, valDecisions=len(rows['val']), valForced=int(va[4].sum().item()))), flush=True)

    def kl_rows(new, old):
        """Row-wise KL(old || new) of masked categorical logits (illegal entries contribute nothing)."""
        lo, ln = torch.log_softmax(old.float(), -1), torch.log_softmax(new.float(), -1)
        po = lo.exp()
        return torch.where(po > 0, po * (lo - ln), torch.zeros_like(po)).sum(-1)

    def logp(batch, idx, with_kl=False):
        obs, mem, act, msk, _ = batch
        o = {k: v[idx] for k, v in obs.items()}
        f = model.encode_decision(o, mem[idx], forecast=False)
        out = model.decode_decision(f, act[idx], False, allow_end_freeze=True, end_freeze_masks=msk[idx])
        if not with_kl: return out.log_prob, None
        with torch.no_grad():
            g = original.decode_decision(original.encode_decision(o, mem[idx], forecast=False), act[idx], False,
                                         allow_end_freeze=True, end_freeze_masks=msk[idx])
        kl = kl_rows(out.type_logits, g.type_logits) + kl_rows(out.source_logits, g.source_logits) + kl_rows(out.target_logits, g.target_logits)
        return out.log_prob, kl

    @torch.no_grad()
    def evaluate(batch):
        model.eval()
        obs, mem, act, msk, forced = batch; agree = []
        for lo in range(0, len(act), 512):
            idx = torch.arange(lo, min(len(act), lo + 512), device=dev)
            f = model.encode_decision({k: v[idx] for k, v in obs.items()}, mem[idx], forecast=False)
            g = model.decode_decision(f, None, True, allow_end_freeze=True, action_only=True)
            agree.append((g.actions == act[idx]).all(-1).float())
        agree = torch.cat(agree); model.train()
        return dict(agreeForced=round(agree[forced == 1].mean().item(), 4), agreeKept=round(agree[forced == 0].mean().item(), 4))

    print(json.dumps(dict(step=0, **{'val_' + k: v for k, v in evaluate(va).items()}, **{'train_' + k: v for k, v in evaluate(tr).items()})), flush=True)
    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=lr, weight_decay=0.0)
    forced_idx = torch.nonzero(tr[4] == 1).squeeze(1); kept_idx = torch.nonzero(tr[4] == 0).squeeze(1)
    began = time.time()
    for step in range(1, steps + 1):
        fi = forced_idx[torch.randint(len(forced_idx), (64,), device=dev)]
        ki = kept_idx[torch.randint(len(kept_idx), (192,), device=dev)]
        lp_f, kl_f = logp(tr, fi, True); lp_k, kl_k = logp(tr, ki, True)
        # forced decisions: learn the planner's move; kept decisions: stay with the policy's own action;
        # KL on kept decisions anchors everything else to the original policy
        loss = -lp_f.mean() - w_keep * lp_k.mean() + kl_weight * kl_k.mean()
        opt.zero_grad(set_to_none=True); loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0); opt.step()
        if step % 500 == 0 or step == steps:
            print(json.dumps(dict(step=step, loss=round(loss.item(), 4), seconds=round(time.time() - began),
                                  **{'val_' + k: v for k, v in evaluate(va).items()})), flush=True)
    model.eval()
    out = dict(ck); out['model'] = {k: v.detach().cpu() for k, v in model.state_dict().items()}
    path = OUT / f'{out_name}.pt'; torch.save(out, path)
    print(json.dumps(dict(saved=str(path.relative_to(ROOT)), sha256=sha(path))), flush=True)


if __name__ == '__main__':
    main()
