#!/usr/bin/env python3
"""Extract conservative UCDP-gazetteer place mentions from ReliefWeb reports."""
from __future__ import annotations
import argparse,csv,gzip,io,json,re,unicodedata,zipfile
from collections import Counter,defaultdict
from pathlib import Path
import numpy as np
WORD=re.compile(r"[^\w\u1200-\u137f]+",re.UNICODE)
SUFFIX=re.compile(r"\b(town|city|district|province|region|zone|woreda|commune|county)\b")
def norm(v):return re.sub(r'\s+',' ',WORD.sub(' ',unicodedata.normalize('NFKC',str(v or '')).casefold())).strip()
def main():
    p=argparse.ArgumentParser();p.add_argument('--ucdp',type=Path,required=True);p.add_argument('--reliefweb',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args();obs=defaultdict(Counter);coords=defaultdict(list)
    with zipfile.ZipFile(a.ucdp) as z:
        name=next(v for v in z.namelist() if v.endswith('.csv'))
        with z.open(name) as raw:
            for r in csv.DictReader(io.TextIOWrapper(raw,encoding='utf-8-sig')):
                try:
                    if int(r['where_prec'])>4:continue
                    lat,lon=float(r['latitude']),float(r['longitude'])
                except (ValueError,KeyError):continue
                country=r['country'].casefold();grid=(round(lat*4),round(lon*4))
                for field in ('where_coordinates','adm_2'):
                    display=norm(r.get(field));variants={display,norm(SUFFIX.sub(' ',display))}
                    for alias in variants:
                        if len(alias)<4 or alias==country or len(alias.split())>5:continue
                        obs[(country,alias)][grid]+=1;coords[(country,alias,grid)].append((lat,lon))
    gaz=defaultdict(dict)
    for (country,alias),counts in obs.items():
        grid,count=counts.most_common(1)[0];total=sum(counts.values())
        if count/total<.8:continue
        values=coords[(country,alias,grid)];gaz[country][alias]=(sum(v[0] for v in values)/len(values),sum(v[1] for v in values)/len(values))
    records=[];reports=matched=0
    with gzip.open(a.reliefweb,'rt',encoding='utf-8') as stream:
        for line in stream:
            try:r=json.loads(line)
            except json.JSONDecodeError:continue
            date=(r.get('date') or {}).get('original','')[:10];pc=r.get('primary_country') or {};country=str(pc.get('name','')).casefold()
            if not date or country not in gaz:continue
            reports+=1;tokens=norm(f"{r.get('title','')} {r.get('body','')}").split();seen=set()
            for size in range(1,6):
                for i in range(len(tokens)-size+1):
                    alias=' '.join(tokens[i:i+size]);point=gaz[country].get(alias)
                    if point is not None:seen.add((round(point[0],5),round(point[1],5)))
            for lat,lon in seen:
                records.append((country,int(np.datetime64(date,'D').astype(int)),lat,lon))
            matched += bool(seen)
    dtype=np.dtype([('country','U80'),('day','i4'),('lat','f4'),('lon','f4')]);array=np.asarray(records,dtype=dtype);a.output.parent.mkdir(parents=True,exist_ok=True);np.savez_compressed(a.output,mentions=array)
    print(json.dumps({'gazetteer_aliases':sum(map(len,gaz.values())),'eligible_reports':reports,'reports_with_matches':matched,'mentions':len(records),'output':str(a.output)},indent=2))
if __name__=='__main__':main()
