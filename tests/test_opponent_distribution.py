import copy

import torch
from sapai_gym1.rl.opponent_distribution import AutoregressiveOpponentBoard, OpponentDistributionConfig
from sapai_gym1.rl.opponent_context_data import PublicOpponentContext
from sapai_gym1.rl.selfplay_env import ParallelSelfPlayEnv


def fixture():
    env=ParallelSelfPlayEnv();env.reset(seed=71)
    context=PublicOpponentContext.capture(env.core,0).features()
    model=AutoregressiveOpponentBoard(OpponentDistributionConfig(('none','pet-fish','pet-ant'),('none','status-melon'),width=32))
    board=[{'species':'pet-fish','attack':3,'health':4,'level':1,'experience':1,'perk':'none',
            'temporaryAttack':False,'temporaryHealth':False},None,None,None,None]
    return model,context,board


def test_joint_likelihood_has_finite_gradients_and_respects_parameter_budget():
    model,context,board=fixture()
    assert sum(p.numel() for p in model.parameters())<1_000_000
    loss=-model.log_prob([context],[board]).mean();loss.backward()
    assert torch.isfinite(loss)
    assert all(p.grad is None or torch.isfinite(p.grad).all() for p in model.parameters())
    assert model.slot_cell.weight_hh.grad.abs().sum()>0
    assert model.field_cell.weight_hh.grad.abs().sum()>0


def test_sampling_returns_coherent_schema_and_scores_empty_fields_once():
    model,context,board=fixture();model.eval()
    samples=model.sample([context],16)
    assert len(samples)==16 and all(len(b)==5 for b in samples)
    assert torch.isfinite(model.log_prob([context]*16,samples)).all()
    embedding=model.encode_context([context])
    targets=model.target_tensor([[None]*5])
    altered=targets.clone();altered[:,:,1]=1;altered[:,:,2]=1;altered[:,:,3]=7;altered[:,:,4]=8;altered[:,:,5]=3
    first=model.decode(embedding,targets)[1]
    second=model.decode(embedding,altered)[1]
    torch.testing.assert_close(first,second,rtol=0,atol=0)


def test_earlier_slot_fields_condition_later_slot_probabilities():
    model,context,board=fixture();model.eval()
    embedding=model.encode_context([context])
    targets=model.target_tensor([board]);other=targets.clone();other[:,0,3]=20
    hidden=[]
    hook=model.species_head.register_forward_pre_hook(lambda module,args:hidden.append(args[0].detach().clone()))
    model.decode(embedding,targets);a=hidden[1].clone();hidden.clear()
    model.decode(embedding,other);b=hidden[1].clone();hook.remove()
    assert not torch.equal(a,b)
