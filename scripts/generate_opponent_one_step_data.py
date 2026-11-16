#!/usr/bin/env python3
"""CPU-only public next-board dataset; no future-survival filtering or training."""
import argparse
from dataclasses import asdict
import hashlib
import json
from pathlib import Path

import torch
from sapai.turtle_pack import SIMULATOR_REVISION
from sapai_gym1.rl.model import EntityTransformerConfig, EntityTransformerPolicy
from sapai_gym1.rl.selfplay_env import ParallelSelfPlayEnv
from sapai_gym1.rl.joint_action import JointAction, step_decisions
from sapai_gym1.rl.action_schema import AutoregressiveAction
from sapai_gym1.rl.opponent_context_data import OneStepRevealCollector, SCHEMA, split_by_match, validate_row


def sha(path):
    with path.open('rb') as stream:
        return hashlib.file_digest(stream,'sha256').hexdigest()


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('checkpoint',type=Path)
    parser.add_argument('--opponent-checkpoint',type=Path)
    parser.add_argument('--games',type=int,default=4)
    parser.add_argument('--seed',type=int,default=9069050)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--allow-checkpoint-training-revision-mismatch', action='store_true',
                        help='Use historical policy weights only; all games still execute current rules and record original training revisions')
    args=parser.parse_args()
    if args.output.exists() or args.games<=0:
        raise ValueError('Use a new output and positive game count')
    torch.set_num_threads(1);torch.manual_seed(args.seed)
    paths=[args.checkpoint,args.opponent_checkpoint or args.checkpoint]
    policies=[];training_revisions=[]
    for path in paths:
        data=torch.load(path,map_location='cpu',weights_only=False)
        if data['simulator_revision']!=SIMULATOR_REVISION and not args.allow_checkpoint_training_revision_mismatch:
            raise ValueError('Use matching-revision policies for this first corpus')
        policy=EntityTransformerPolicy(EntityTransformerConfig(**data['config'])).eval()
        policy.load_state_dict(data['model']);policies.append(policy)
        training_revisions.append(data['simulator_revision'])
    populations=tuple(sha(p) for p in paths)
    collection=hashlib.sha256(repr((populations,args.seed,SCHEMA,SIMULATOR_REVISION)).encode()).hexdigest()[:20]
    collector=OneStepRevealCollector(simulator_revision=SIMULATOR_REVISION,collection_id=collection)
    with torch.inference_mode():
        for game in range(args.games):
            env=ParallelSelfPlayEnv();env.reset(seed=args.seed+game)
            memory=[policy.initial_memory(1) for policy in policies]
            collector.begin_turn(env.core,str(game),populations=populations)
            for step_index in range(2000):
                obs=env.observations();decisions=[]
                turn=int(env.core.agent.turn)
                for side,active in enumerate(env.active_players()):
                    if not active:
                        decisions.append(None);continue
                    batch={k:torch.as_tensor(v)[None] for k,v in obs[side].items()}
                    output=policies[side].act(batch,memory[side],deterministic=False)
                    memory[side]=output.memory
                    mask=None if output.end_freeze_mask is None else int(output.end_freeze_mask[0])
                    decisions.append(JointAction(AutoregressiveAction(*output.actions[0].tolist()),mask))
                transition=step_decisions(env,decisions).result
                if transition.battle_resolved:
                    collector.revealed(match_id=str(game),turn=turn,
                        submitted_boards=(env.core.previous_agent_state,env.core.previous_opponent_state),terminal=transition.terminated)
                    if not transition.terminated:
                        collector.begin_turn(env.core,str(game),populations=populations)
                if transition.terminated:
                    break
            else:
                raise RuntimeError('Collection safety limit; do not silently label a partial match complete')
    if collector.pending:
        raise RuntimeError('Incomplete public-turn captures at match completion')
    for row in collector.examples:validate_row(row)
    train,test=split_by_match(collector.examples)
    payload={'schema':SCHEMA,'metadata':{'games':args.games,'populations':populations,'seed':args.seed,
        'continuation':'stochastic-policy','simulator_revision':SIMULATOR_REVISION,'generatorSHA256':sha(Path(__file__)),
        'checkpointTrainingRevisions':training_revisions,
        'allowedCheckpointTrainingRevisionMismatch':args.allow_checkpoint_training_revision_mismatch,
        'pendingTurnsAtCompletion':len(collector.pending),'rejectedRows':0,
        'trainingActivated':False},'train':train,'heldout':test}
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(payload,indent=2)+'\n')
    print(json.dumps({'games':args.games,'trainRows':len(train),'heldoutRows':len(test),
                      'terminalTurnRows':sum(r['metadata']['terminal_after_reveal'] for r in collector.examples),
                      'output':str(args.output),'trainingActivated':False}))


if __name__ == '__main__':main()
