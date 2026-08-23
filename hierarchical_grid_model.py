from __future__ import annotations

import math
import torch
from torch import nn


class HierarchicalGridClassifier(nn.Module):
    def __init__(self, feature_dim, sequence_length, countries, conflicts, classes,
                 d_model=128, heads=4, layers=3, ff_dim=384, dropout=.10):
        super().__init__()
        self.config=dict(feature_dim=feature_dim,sequence_length=sequence_length,
                         countries=countries,conflicts=conflicts,classes=classes,
                         d_model=d_model,heads=heads,layers=layers,ff_dim=ff_dim,dropout=dropout)
        self.norm=nn.LayerNorm(feature_dim);self.projection=nn.Linear(feature_dim,d_model)
        self.position=nn.Parameter(torch.randn(1,sequence_length,d_model)*.02)
        layer=nn.TransformerEncoderLayer(d_model,heads,ff_dim,dropout,"gelu",batch_first=True,norm_first=True)
        self.encoder=nn.TransformerEncoder(layer,layers)
        self.country=nn.Embedding(countries,d_model);self.conflict=nn.Embedding(conflicts,d_model)
        self.horizon=nn.Sequential(nn.Linear(2,32),nn.GELU(),nn.Linear(32,d_model))
        self.head=nn.Sequential(nn.LayerNorm(d_model),nn.Linear(d_model,d_model),nn.GELU(),nn.Dropout(dropout))
        self.classifier=nn.Linear(d_model,classes)

    def forward(self,features,country,conflict,horizon):
        valid=features[:,:,0]>.5
        x=self.projection(self.norm(features))*math.sqrt(self.config['d_model'])
        x=self.encoder(x+self.position,src_key_padding_mask=~valid)
        hfeatures=torch.stack((torch.log1p(horizon)/5.0,torch.sqrt(horizon)/10.0),-1)
        context=x[:,-1]+self.country(country)+self.conflict(conflict)+self.horizon(hfeatures)
        return self.classifier(self.head(context))


def distance_aware_loss(logits,target_class,target_latlon,class_latlon,country_mask,weight):
    masked=logits.masked_fill(~country_mask,-1e9)
    ce=nn.functional.cross_entropy(masked,target_class)
    if weight<=0:return ce,ce,ce.new_zeros(())
    # Equirectangular distance is stable here because all unmasked classes share a country.
    delta=class_latlon[None]-target_latlon[:,None]
    mean_lat=torch.deg2rad((class_latlon[None,:,0]+target_latlon[:,None,0])/2)
    east=delta[:,:,1]*torch.cos(mean_lat)*111.32;north=delta[:,:,0]*111.32
    distance=torch.sqrt(east.square()+north.square()+1e-6)
    expected=(masked.softmax(-1)*distance).sum(-1).mean()/1000.0
    return ce+weight*expected,ce,expected
