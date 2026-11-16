"""Standalone compact autoregressive public-board forecast. Not a policy adapter."""
from dataclasses import dataclass

import torch
from torch import nn
from torch.distributions import Categorical

from .opponent_forecast_data import (
    PROGRESS, PublicContextBatch, collate_contexts, collate_targets,
)


@dataclass
class OpponentDistributionConfig:
    species: tuple[str,...]
    perks: tuple[str,...]
    width: int = 128


class AutoregressiveOpponentBoard(nn.Module):
    """p(board|public context), including dependencies between slots/fields."""
    def __init__(self, config):
        super().__init__();self.config=config
        if not config.species or config.species[0]!='none' or not config.perks or config.perks[0]!='none':
            raise ValueError('Empty species and no-perk must have ID zero')
        if len(set(config.species))!=len(config.species) or len(set(config.perks))!=len(config.perks):
            raise ValueError('Duplicate vocabulary entries')
        self.species_ids={x:i for i,x in enumerate(config.species)}
        self.perk_ids={x:i for i,x in enumerate(config.perks)}
        d=config.width
        self.species_embedding=nn.Embedding(len(config.species),d)
        self.perk_embedding=nn.Embedding(len(config.perks),d)
        self.position=nn.Embedding(5,d);self.board_role=nn.Embedding(3,d)
        self.numeric=nn.Linear(7,d);self.global_projection=nn.Linear(8,d)
        layer=nn.TransformerEncoderLayer(d,4,d*2,dropout=0.,batch_first=True,norm_first=True,activation='gelu')
        self.context_encoder=nn.TransformerEncoder(layer,1,norm=nn.LayerNorm(d))
        self.cls=nn.Parameter(torch.zeros(d));self.bos=nn.Parameter(torch.zeros(d))
        self.slot_cell=nn.GRUCell(d,d);self.field_cell=nn.GRUCell(d,d)
        self.species_head=nn.Linear(d,len(config.species))
        sizes=(len(PROGRESS),len(config.perks),51,50,4)
        self.field_heads=nn.ModuleList(nn.Linear(d,size) for size in sizes)
        self.field_embeddings=nn.ModuleList(nn.Embedding(size,d) for size in sizes)
        if sum(p.numel() for p in self.parameters())>1_000_000:
            raise ValueError('Initial forecasting prototype exceeds the agreed one-million-parameter budget')

    def context_tensors(self, contexts):
        batch = contexts if isinstance(contexts, PublicContextBatch) else collate_contexts(
            contexts, self.species_ids, self.perk_ids)
        batch = batch.to(next(self.parameters()).device)
        return tuple(getattr(batch, key) for key in batch.__dataclass_fields__)

    def encode_context(self, contexts):
        species,perks,numbers,visible,globals=self.context_tensors(contexts)
        positions=torch.arange(15,device=species.device)%5
        roles=torch.arange(15,device=species.device)//5
        tokens=self.species_embedding(species)+self.perk_embedding(perks)+self.numeric(numbers)
        tokens=tokens+self.position(positions)[None]+self.board_role(roles)[None]
        cls=self.cls[None]+self.global_projection(globals)
        tokens=torch.cat((cls[:,None],tokens),dim=1)
        mask=torch.cat((torch.zeros(len(contexts),1,dtype=torch.bool,device=species.device),~visible),dim=1)
        return self.context_encoder(tokens,src_key_padding_mask=mask)[:,0]

    def target_tensor(self, boards):
        labels = boards if torch.is_tensor(boards) else collate_targets(
            boards, self.species_ids, self.perk_ids)
        return labels.to(next(self.parameters()).device)

    def decode(self, context_embedding, targets=None):
        batch=context_embedding.shape[0]
        hidden=context_embedding;previous=self.bos[None].expand(batch,-1)
        log_prob=torch.zeros(batch,device=hidden.device);slots=[]
        for slot in range(5):
            hidden=self.slot_cell(previous+self.position.weight[slot],hidden)
            distribution=Categorical(logits=self.species_head(hidden))
            species=distribution.sample() if targets is None else targets[:,slot,0]
            log_prob=log_prob+distribution.log_prob(species)
            occupied=species!=0
            state=hidden+self.species_embedding(species)
            fields=[species]
            for field,(head,embedding) in enumerate(zip(self.field_heads,self.field_embeddings)):
                distribution=Categorical(logits=head(state))
                choice=distribution.sample() if targets is None else targets[:,slot,field+1]
                # Empty-slot fields are marginalized, never scored or used to
                # condition later slots. Future dependence cannot leak them.
                log_prob=log_prob+torch.where(occupied,distribution.log_prob(choice),0.)
                fields.append(torch.where(occupied,choice,0))
                state=self.field_cell(embedding(choice),state)
            previous=torch.where(occupied[:,None],state,self.species_embedding.weight[0][None])
            slots.append(torch.stack(fields,dim=-1))
        return torch.stack(slots,dim=1),log_prob

    def log_prob(self, contexts, boards):
        return self.decode(self.encode_context(contexts),self.target_tensor(boards))[1]

    @torch.inference_mode()
    def sample(self, contexts, samples_per_context=1):
        if samples_per_context<=0:raise ValueError('Positive sample count required')
        encoding=self.encode_context(contexts).repeat_interleave(samples_per_context,dim=0)
        labels,_=self.decode(encoding)
        boards=[]
        for row in labels.cpu().tolist():
            board=[]
            for species,progress,perk,attack,health,temporary in row:
                board.append(None if species==0 else {'species':self.config.species[species],
                    'level':PROGRESS[progress][0],'experience':PROGRESS[progress][1],
                    'perk':self.config.perks[perk],'attack':attack,'health':health+1,
                    'temporaryAttack':bool(temporary&1),'temporaryHealth':bool(temporary&2)})
            boards.append(board)
        return boards


class MarginalOpponentBoard(AutoregressiveOpponentBoard):
    """Same public-context encoder; independent slot/field forecast baseline.

    This is a standalone retrained comparison, not the saved production head.
    Empty fields are marginalized in both models, so board log scores have the
    same definition. Samples deliberately retain the baseline's lack of joint
    dependencies for the planned forecasting comparison.
    """
    def __init__(self, config):
        super().__init__(config)
        del self.slot_cell, self.field_cell, self.field_embeddings, self.bos

    def decode(self, context_embedding, targets=None):
        log_prob = context_embedding.new_zeros(context_embedding.shape[0])
        slots = []
        for slot in range(5):
            hidden = context_embedding + self.position.weight[slot]
            distribution = Categorical(logits=self.species_head(hidden))
            species = distribution.sample() if targets is None else targets[:, slot, 0]
            log_prob = log_prob + distribution.log_prob(species)
            occupied = species != 0
            fields = [species]
            for field, head in enumerate(self.field_heads):
                distribution = Categorical(logits=head(hidden))
                choice = distribution.sample() if targets is None else targets[:, slot, field+1]
                log_prob = log_prob + torch.where(occupied, distribution.log_prob(choice), 0.)
                fields.append(torch.where(occupied, choice, 0))
            slots.append(torch.stack(fields, dim=-1))
        return torch.stack(slots, dim=1), log_prob
