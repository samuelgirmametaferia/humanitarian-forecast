#!/usr/bin/env python3
"""Materialize monthly CHIRPS v3 rainfall context at H3 cell centroids.

Files are cached by year/month. The compiler records the source URL for every
month and never synthesizes a historical release date. Downstream retrospective
experiments must additionally enforce the CHIRPS preliminary/final availability
lag appropriate to their forecast cutoff.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import urllib.request
from datetime import date
from pathlib import Path

import numpy as np
import rasterio

BASE = "https://data.chc.ucsb.edu/products/CHIRPS/v3.0/monthly/africa/tifs"


def months_between(start: str, end: str) -> list[tuple[int, int]]:
    sy, sm = map(int, start.split("-")[:2]); ey, em = map(int, end.split("-")[:2])
    out=[]; y,m=sy,sm
    while (y,m) <= (ey,em):
        out.append((y,m)); m += 1
        if m == 13: y += 1; m = 1
    return out


def download(url: str, path: Path) -> None:
    if path.exists() and path.stat().st_size > 0:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp=path.with_suffix(path.suffix+'.part')
    req=urllib.request.Request(url,headers={"User-Agent":"humanitarian-forecast/0.1 research"})
    with urllib.request.urlopen(req,timeout=120) as r, tmp.open('wb') as f:
        while True:
            chunk=r.read(1024*1024)
            if not chunk: break
            f.write(chunk)
    tmp.replace(path)


def sample_raster(path: Path, latlon: np.ndarray) -> np.ndarray:
    with rasterio.open(path) as ds:
        coords=[(float(lon),float(lat)) for lat,lon in latlon]
        values=np.asarray([float(value[0]) for value in ds.sample(coords)],dtype=np.float32)
        nodata=ds.nodata
        if nodata is not None:
            values=np.where(np.isclose(values,nodata),np.nan,values)
        return values


def main() -> None:
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--h3',type=Path,required=True)
    p.add_argument('--start',required=True,help='YYYY-MM')
    p.add_argument('--end',required=True,help='YYYY-MM')
    p.add_argument('--cache-dir',type=Path,default=Path('data/geo/raw/chirps_v3_monthly_africa'))
    p.add_argument('--output',type=Path,required=True)
    a=p.parse_args()
    hz=np.load(a.h3,allow_pickle=True); resolutions=tuple(int(v) for v in hz['resolutions']); months=months_between(a.start,a.end)
    monthly_files=[];manifest=[]
    for y,m in months:
        name=f'chirps-v3.0.{y:04d}.{m:02d}.tif';url=f'{BASE}/{name}';path=a.cache_dir/name
        download(url,path);monthly_files.append(path)
        manifest.append({'year':y,'month':m,'url':url,'path':str(path),'sha256':hashlib.sha256(path.read_bytes()).hexdigest(),'bytes':path.stat().st_size})
        print(f'cached {name} {path.stat().st_size/1024/1024:.1f} MiB',flush=True)
    payload={'year_month':np.asarray([y*100+m for y,m in months],dtype=np.int32),'resolutions':np.asarray(resolutions,dtype=np.int16),'schema':np.asarray('chirps-h3-monthly-v1')}
    report={'schema':'chirps-h3-monthly-v1','source':'CHIRPS v3 monthly Africa GeoTIFF','start':a.start,'end':a.end,'months':len(months),'resolutions':{},'files':manifest,'causality_note':'CHIRPS v3 final is normally produced in the following month; historical feature builders must apply source availability time rather than event month alone.'}
    for r in resolutions:
        c=hz[f'centroids_r{r}'].astype(np.float32);raw=np.stack([sample_raster(path,c) for path in monthly_files]).astype(np.float32)
        log=np.log1p(np.maximum(raw,0.0)).astype(np.float32)
        delta=np.zeros_like(raw);delta[1:]=raw[1:]-raw[:-1]
        # Calendar-month standardized anomaly using only the materialized period.
        z=np.zeros_like(raw)
        for month in range(1,13):
            idx=np.asarray([i for i,(_,m) in enumerate(months) if m==month],dtype=np.int64)
            if not len(idx):continue
            mean=np.nanmean(raw[idx],axis=0);std=np.nanstd(raw[idx],axis=0);std=np.maximum(std,1.0);z[idx]=(raw[idx]-mean)/std
        feat=np.stack([raw,log,delta,z],axis=-1).astype(np.float32)
        payload[f'features_r{r}']=feat;payload[f'feature_names_r{r}']=np.asarray(['precip_mm','log1p_precip','delta_prev_month_mm','calendar_month_z'])
        report['resolutions'][str(r)]={'cells':int(len(c)),'coverage':float(np.isfinite(raw).mean()),'mean_precip_mm':float(np.nanmean(raw))}
    a.output.parent.mkdir(parents=True,exist_ok=True);np.savez_compressed(a.output,**payload);a.output.with_suffix('.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps({k:v for k,v in report.items() if k!='files'},indent=2))
if __name__=='__main__':main()
