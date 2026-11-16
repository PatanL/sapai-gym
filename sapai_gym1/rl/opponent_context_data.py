"""Causal public context and uncensored one-step reveal labels.

Not part of the production observation or policy. Context never reads either
current private shop/team. Targets are accepted only from a submitted-board
reveal event, including the last battle of a match.
"""
from dataclasses import asdict, dataclass
import hashlib
import json

SCHEMA = 'causal-opponent-one-step-v1-20260906'


def public_board(team):
    if team is None:
        return None
    return tuple(None if slot.empty else {
        'species':slot.pet.name,'attack':int(slot.pet.attack),'health':int(slot.pet.health),
        'level':int(slot.pet.level),'experience':int(slot.pet.experience),'perk':slot.pet.status,
        'temporaryAttack':bool(slot.pet._until_end_of_battle_attack_buff),
        'temporaryHealth':bool(slot.pet._until_end_of_battle_health_buff),
    } for slot in team)


@dataclass(frozen=True)
class PublicOpponentContext:
    opponent_last: tuple | None
    opponent_previous: tuple | None
    our_last: tuple | None
    turn: int
    our_lives: int
    opponent_lives: int
    our_wins: int
    opponent_wins: int
    pack: str = 'Turtle-P1-vs-P1'

    def features(self):
        result=asdict(self)
        result['missing_history']=[self.opponent_last is None,self.opponent_previous is None,self.our_last is None]
        return result

    def digest(self):
        return hashlib.sha256(json.dumps(self.features(),sort_keys=True,separators=(',',':')).encode()).hexdigest()

    @classmethod
    def capture(cls, core, side):
        if side not in (0,1):
            raise ValueError('Invalid perspective')
        own, enemy=(core.agent,core.opponent) if side==0 else (core.opponent,core.agent)
        latest=core.previous_opponent_state if side==0 else core.previous_agent_state
        previous=core.previous_previous_opponent_state if side==0 else core.previous_previous_agent_state
        own_last=core.previous_agent_state if side==0 else core.previous_opponent_state
        return cls(public_board(latest),public_board(previous),public_board(own_last),int(own.turn),
                   int(own.lives),int(enemy.lives),int(own.wins),int(enemy.wins))


class OneStepRevealCollector:
    def __init__(self, *, simulator_revision, collection_id):
        self.simulator_revision=simulator_revision
        self.collection_id=collection_id
        self.pending={}
        self.examples=[]

    def begin_turn(self, core, match_id, *, populations):
        if core.shop_action_count_agent or core.shop_action_count_opponent:
            raise ValueError('Capture context before either player starts hidden shopping')
        key=(str(match_id),int(core.agent.turn))
        if key in self.pending:
            raise ValueError('Duplicate turn context')
        self.pending[key]=(tuple(PublicOpponentContext.capture(core,side) for side in (0,1)),tuple(populations))

    def revealed(self, *, match_id, turn, submitted_boards, terminal=False):
        key=(str(match_id),int(turn))
        if key not in self.pending:
            raise ValueError('No matching before-shop context for this reveal')
        if len(submitted_boards)!=2:
            raise ValueError('Both actually submitted boards are required')
        contexts,populations=self.pending.pop(key)
        for side,context in enumerate(contexts):
            target=public_board(submitted_boards[1-side])
            if target is None or len(target)!=5:
                raise ValueError('A reveal must contain an ordered five-slot board')
            self.examples.append({'schema':SCHEMA,'context':context.features(),'target':target,
                'metadata':{'collection_id':self.collection_id,'match_id':str(match_id),'turn':int(turn),
                    'side':side,'checkpoint_populations':populations,'opponent_population':populations[1-side],
                    'simulator_revision':self.simulator_revision,'visibility':'submitted-pre-battle-reveal',
                    'terminal_after_reveal':bool(terminal)}})


def split_by_match(examples, *, heldout_populations=(), fraction=.2, salt='opponent-forecast-v1'):
    if not 0 <= fraction <= 1:
        raise ValueError('Invalid holdout fraction')
    grouped={}
    for row in examples:
        meta=row['metadata']
        key=(meta['collection_id'],meta['match_id'])
        grouped.setdefault(key,[]).append(row)
    train,test=[],[]
    heldout=set(heldout_populations)
    for key,rows in grouped.items():
        populations={p for row in rows for p in row['metadata']['checkpoint_populations']}
        score=int(hashlib.sha256((salt+repr(key)).encode()).hexdigest()[:16],16)/2**64
        is_test=bool(populations & heldout) or score < fraction
        (test if is_test else train).extend(rows)
    return train,test


def validate_row(row):
    if row.get('schema') != SCHEMA:
        raise ValueError('Unsupported causal dataset schema')
    required=set(PublicOpponentContext.__dataclass_fields__) | {'missing_history'}
    if set(row['context']) != required:
        raise ValueError('Unknown/missing context fields; do not import private features')
    if row['metadata']['visibility'] != 'submitted-pre-battle-reveal':
        raise ValueError('Targets must be actual public reveals')
    if row['context']['turn'] != row['metadata']['turn'] or len(row['target']) != 5:
        raise ValueError('Mismatched one-step label')
    pet_fields={'species','attack','health','level','experience','perk','temporaryAttack','temporaryHealth'}
    for board in [row['target'],row['context']['opponent_last'],row['context']['opponent_previous'],row['context']['our_last']]:
        if board is None:
            continue
        if len(board)!=5:
            raise ValueError('Invalid public board length')
        for pet in board:
            if pet is not None and set(pet)!=pet_fields:
                raise ValueError('Unknown/private pet feature')
    return row
