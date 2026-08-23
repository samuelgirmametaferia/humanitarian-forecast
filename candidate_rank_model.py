from __future__ import annotations
import math,torch
from torch import nn

class ConflictCandidateRanker(nn.Module):
    def __init__(self,event_dim=19,candidate_dim=11,sequence_length=16,countries=121,conflicts=1014,d_model=128,heads=4,layers=3,ff_dim=384,dropout=.1):
        super().__init__();self.config=dict(event_dim=event_dim,candidate_dim=candidate_dim,sequence_length=sequence_length,countries=countries,conflicts=conflicts,d_model=d_model,heads=heads,layers=layers,ff_dim=ff_dim,dropout=dropout)
        self.event_norm=nn.LayerNorm(event_dim);self.event_projection=nn.Linear(event_dim,d_model);self.position=nn.Parameter(torch.randn(1,sequence_length,d_model)*.02)
        layer=nn.TransformerEncoderLayer(d_model,heads,ff_dim,dropout,'gelu',batch_first=True,norm_first=True);self.encoder=nn.TransformerEncoder(layer,layers)
        self.context=nn.Sequential(nn.LayerNorm(d_model),nn.Linear(d_model,d_model),nn.GELU())
        self.country=nn.Embedding(countries,d_model);self.conflict=nn.Embedding(conflicts,d_model)
        self.candidate=nn.Sequential(nn.LayerNorm(candidate_dim),nn.Linear(candidate_dim,d_model),nn.GELU(),nn.Linear(d_model,d_model))
        self.bias=nn.Sequential(nn.Linear(candidate_dim,64),nn.GELU(),nn.Linear(64,1))
    def forward(self,events,candidates,valid,country,conflict):
        emask=events[:,:,0]>.5;x=self.event_projection(self.event_norm(events))*math.sqrt(self.config['d_model']);x=self.encoder(x+self.position,src_key_padding_mask=~emask);context=self.context(x[:,-1]);candidate=self.candidate(candidates)
        context=context+self.country(country)+self.conflict(conflict)
        logits=(candidate*context[:,None]).sum(-1)/math.sqrt(self.config['d_model'])+self.bias(candidates).squeeze(-1);return logits.masked_fill(~valid,-1e9)

def loss_fn(logits,coordinates,target,label,distance_weight,center_weight=0.0):
    ce=nn.functional.cross_entropy(logits,label);probability=logits.softmax(-1);distance=torch.linalg.vector_norm(coordinates-target[:,None],dim=-1);expected=(probability*distance).sum(-1).mean()
    center=(probability[:,:,None]*coordinates).sum(1);center_distance=torch.linalg.vector_norm(center-target,dim=-1).mean()
    return ce+distance_weight*expected+center_weight*center_distance,ce,expected,center_distance
