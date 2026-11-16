import importlib.util
from pathlib import Path
import unittest
from unittest.mock import patch

import torch
from sapai import Pet, Food, Shop
from sapai_gym1.rl import ActionType, EntityObservationEncoder, EntityTransformerConfig, EntityTransformerPolicy, ParallelSelfPlayEnv

SCRIPT=Path(__file__).resolve().parents[1]/'scripts/visualize_entity_game.py'
s=importlib.util.spec_from_file_location('pet_probability_viewer',SCRIPT)
viewer=importlib.util.module_from_spec(s);s.loader.exec_module(viewer)

class PetProbabilityTests(unittest.TestCase):
    def setup_model(self,joint=True):
        env=ParallelSelfPlayEnv();env.reset(seed=318)
        env.core.agent.shop=Shop(slots=[Pet('ant'),Pet('ant'),Pet('fish'),Food('apple')],turn=1)
        env.core.agent.gold=10
        observation=env.observations()[0];encoder=EntityObservationEncoder()
        model=EntityTransformerPolicy(EntityTransformerConfig(num_items=encoder.num_items,num_statuses=encoder.num_statuses,
            d_model=32,nhead=4,num_layers=1,dim_feedforward=64,memory_dim=32,opponent_history_dim=32,end_freeze=joint)).eval()
        memory=model.initial_memory(1);batch={k:torch.as_tensor(v).unsqueeze(0) for k,v in observation.items()}
        return env,observation,model,memory,batch

    @torch.inference_mode()
    def test_conditional_and_overall_probabilities_match_exact_marginals(self):
        for joint in (False,True):
            _,obs,model,memory,batch=self.setup_model(joint)
            features=model.encode_decision(batch,memory)
            output=model.decode_decision(features,deterministic=True,allow_end_freeze=joint)
            data=viewer.buy_pet_source_probabilities(model,features,output)
            expected=model.primitive_log_probabilities(features)[0,int(ActionType.BUY_PET)].exp().sum(-1)
            self.assertAlmostEqual(sum(r['conditionalProbability'] for r in data['sources']),1.,places=6)
            self.assertAlmostEqual(sum(r['overallProbability'] for r in data['sources']),data['typeProbability'],places=6)
            self.assertEqual({r['source'] for r in data['sources']},{0,1,2})
            for r in data['sources']:self.assertAlmostEqual(r['overallProbability'],float(expected[r['source']]),places=6)

    @torch.inference_mode()
    def test_forced_buy_diagnostics_do_not_change_action_memory_or_rng(self):
        for joint in (False,True):
            _,obs,model,memory,batch=self.setup_model(joint)
            model.type_head.weight.zero_();model.type_head.bias.fill_(-10)
            model.type_head.bias[int(ActionType.ROLL)]=2;model.type_head.bias[int(ActionType.BUY_PET)]=1
            expected=model(batch,memory,deterministic=True,allow_end_freeze=joint)
            initial_memory=memory.clone();rng=torch.get_rng_state().clone()
            with patch.object(model.memory_cell,'forward',wraps=model.memory_cell.forward) as gru:
                action,next_memory,d=viewer.policy_action(model,obs,memory)
            self.assertEqual(action.as_tuple(),tuple(expected.actions[0].tolist()))
            self.assertEqual(int(action.action_type),int(ActionType.ROLL))
            self.assertTrue(torch.equal(next_memory,expected.memory));self.assertTrue(torch.equal(memory,initial_memory))
            self.assertTrue(torch.equal(torch.get_rng_state(),rng));self.assertEqual(gru.call_count,1)
            self.assertTrue(d['buyPetSources']['available'])

    @torch.inference_mode()
    def test_buy_selected_uses_existing_source_logits(self):
        _,obs,model,memory,batch=self.setup_model()
        model.type_head.weight.zero_();model.type_head.bias.fill_(-10);model.type_head.bias[int(ActionType.BUY_PET)]=10
        with patch.object(model,'decode_decision',wraps=model.decode_decision) as decoder:
            action,_,d=viewer.policy_action(model,obs,memory)
        self.assertEqual(action.action_type,ActionType.BUY_PET);self.assertEqual(decoder.call_count,1)
        self.assertEqual(len(d['buyPetSources']['sources']),3)

    @torch.inference_mode()
    def test_zero_gold_unavailable_is_not_a_fake_distribution(self):
        env,_,model,memory,_=self.setup_model();env.core.agent.gold=0;obs=env.observations()[0]
        with patch.object(model,'decode_decision',wraps=model.decode_decision) as decoder:
            _,_,d=viewer.policy_action(model,obs,memory)
        self.assertEqual(decoder.call_count,1)
        self.assertEqual(d['buyPetSources'],{'schema':'buy-pet-sources-v1','available':False,'typeProbability':0.0,'sources':[]})

if __name__=='__main__':unittest.main()
