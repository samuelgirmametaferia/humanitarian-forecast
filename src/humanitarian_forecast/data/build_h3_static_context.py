#!/usr/bin/env python3
"""Project validated PRIO population/accessibility context onto dense H3 cells.

Population and accessibility were the strongest static cold-start combination in
this repository's rolling-origin ablations. This compiler keeps those feature
families together and uses the nearest materialized PRIO cell for boundary H3
cells that do not have an exact PRIO id in the Ethiopia feature tables.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def main() -> None:
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--h3',type=Path,required=True)
    p.add_argument('--population',type=Path,required=True)
    p.add_argument('--accessibility',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    a=p.parse_args()
    hz=np.load(a.h3,allow_pickle=True);pop=np.load(a.population,allow_pickle=True);access=np.load(a.accessibility,allow_pickle=True)
    if not np.array_equal(pop['gids'],access['gids']):raise ValueError('population/accessibility PRIO gids do not align')
    if not np.allclose(pop['centers'],access['centers']):raise ValueError('population/accessibility PRIO centers do not align')
    centers=pop['centers'].astype(np.float32);base=np.concatenate([pop['features'],access['features']],axis=1).astype(np.float32)
    names=np.concatenate([pop['feature_names'],access['feature_names']]).astype(str)
    # Normalize once over the 208 materialized PRIO cells so every H3 resolution
    # consumes the same static scale.
    mean=np.nanmean(base,axis=0);std=np.maximum(np.nanstd(base,axis=0),1e-4);normalized=np.nan_to_num((base-mean)/std,nan=0.,posinf=0.,neginf=0.)
    payload={'feature_names':np.concatenate([names,np.asarray(['nearest_prio_distance_100km'])]),'mean':mean.astype(np.float32),'std':std.astype(np.float32),'schema':np.asarray('h3-static-context-v1')}
    report={'schema':'h3-static-context-v1','sources':{'population':str(a.population),'accessibility':str(a.accessibility)},'base_prio_cells':int(len(centers)),'resolutions':{}}
    for r in map(int,hz['resolutions']):
        c=hz[f'centroids_r{r}'].astype(np.float32)
        # Equirectangular km is sufficient for Ethiopia-scale nearest-cell mapping.
        dlat=(c[:,None,0]-centers[None,:,0])*111.32
        mean_lat=(c[:,None,0]+centers[None,:,0])*.5
        dlon=(c[:,None,1]-centers[None,:,1])*111.32*np.cos(np.deg2rad(mean_lat))
        dist=np.sqrt(dlat*dlat+dlon*dlon);nearest=dist.argmin(axis=1);nearest_km=dist[np.arange(len(c)),nearest]
        features=np.concatenate([normalized[nearest],(nearest_km/100.).astype(np.float32)[:,None]],axis=1)
        payload[f'features_r{r}']=features.astype(np.float32);payload[f'nearest_prio_gid_r{r}']=pop['gids'][nearest].astype(np.int64);payload[f'nearest_prio_distance_km_r{r}']=nearest_km.astype(np.float32)
        report['resolutions'][str(r)]={'cells':int(len(c)),'feature_dim':int(features.shape[1]),'nearest_distance_mean_km':float(nearest_km.mean()),'nearest_distance_p90_km':float(np.quantile(nearest_km,.9)),'nearest_distance_max_km':float(nearest_km.max())}
    a.output.parent.mkdir(parents=True,exist_ok=True);np.savez_compressed(a.output,**payload);a.output.with_suffix('.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report,indent=2))
if __name__=='__main__':main()
