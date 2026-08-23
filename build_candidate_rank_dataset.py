#!/usr/bin/env python3
"""Build cutoff-safe top-frequency conflict candidates for supervised ranking."""
from __future__ import annotations
import argparse,bisect,csv,io,json,math,zipfile
from collections import Counter,defaultdict
from pathlib import Path
import numpy as np

EARTH_KM=111.32
def pkey(lat,lon):return round(float(lat),5),round(float(lon),5)
def offset(point,anchor):
    north=(point[0]-anchor[0])*EARTH_KM;east=(point[1]-anchor[1])*EARTH_KM*math.cos(math.radians(anchor[0]));return east,north

def main():
    p=argparse.ArgumentParser();p.add_argument('--data',type=Path,required=True);p.add_argument('--output',type=Path,required=True);p.add_argument('--candidates',type=int,default=32);p.add_argument('--reliefweb-mentions',type=Path);p.add_argument('--ucdp-events',type=Path);a=p.parse_args()
    d=np.load(a.data);x=d['x'];y=d['y'];rows=[json.loads(str(v)) for v in d['meta']];n=len(rows);k=a.candidates
    mention_days=defaultdict(list)
    if a.reliefweb_mentions:
        for value in np.load(a.reliefweb_mentions)['mentions']:
            mention_days[(str(value['country']).casefold(),round(float(value['lat'])*4),round(float(value['lon'])*4))].append(int(value['day']))
        for days in mention_days.values():days.sort()
    raw_events=defaultdict(list)
    if a.ucdp_events:
        with zipfile.ZipFile(a.ucdp_events) as archive:
            name=next(value for value in archive.namelist() if value.endswith('.csv'))
            with archive.open(name) as raw:
                for row in csv.DictReader(io.TextIOWrapper(raw,encoding='utf-8-sig')):
                    try:
                        if int(row['where_prec'])>4 or int(row['date_prec'])>3:continue
                        event_day=int(np.datetime64(row['date_start'][:10],'D').astype(int));event_key=pkey(float(row['latitude']),float(row['longitude']));fatalities=max(0.,float(row.get('best') or 0))
                    except (ValueError,KeyError):continue
                    raw_events[(str(row['conflict_new_id']),event_key)].append((event_day,fatalities))
        for event_key,values in list(raw_events.items()):
            values.sort();days=[value[0] for value in values];prefix=[0.]
            for _,fatalities in values:prefix.append(prefix[-1]+fatalities)
            raw_events[event_key]=(days,prefix)
    # Base features include four cutoff-safe activity windows in addition to
    # lifetime frequency and last-seen recency.
    # Eight candidate-centered ring features summarize the encoded event
    # sequence at 25/50/100/250 km and 7/30/90-day scales.
    candidate_dim=23+(3 if mention_days else 0)+(5 if raw_events else 0)
    features=np.zeros((n,k,candidate_dim),np.float32);coordinates=np.zeros((n,k,2),np.float32);valid=np.zeros((n,k),bool);labels=np.zeros(n,np.int64);oracle=np.zeros(n,np.float32)
    locations=defaultdict(dict);transitions=defaultdict(lambda:defaultdict(Counter));cursor=0
    while cursor<n:
        date=rows[cursor]['target_date'];stop=cursor
        while stop<n and rows[stop]['target_date']==date:stop+=1
        day=int(np.datetime64(date,'D').astype(int))
        for index in range(cursor,stop):
            row=rows[index];conflict=str(row['conflict_id']);anchor=np.array([row['anchor_lat'],row['anchor_lon']],float);source=pkey(*anchor)
            history_valid=x[index,:,0]>.5;history_positions=x[index,history_valid,1:3]
            history_days=np.maximum(0.,np.expm1(x[index,history_valid,3]*6)-float(row['gap_days']))
            history_fatalities=np.maximum(0.,np.expm1(x[index,history_valid,4]*6))
            ranked=sorted(locations[conflict].items(),key=lambda item:(-item[1]['count'],-item[1]['last'],item[0]))
            cutoff=day-int(row['gap_days'])
            chosen=[(source,{'point':anchor,'count':1,'last':cutoff,'days':[cutoff]})]
            chosen.extend((key,value) for key,value in ranked if key!=source)
            chosen=chosen[:k]
            for j,(key,value) in enumerate(chosen):
                point=value['point'];east,north=offset(point,anchor);count=value['count'];cutoff=day-int(row['gap_days']);age=max(1,cutoff-value['last']);transition=transitions[conflict][source][key]
                activity=[bisect.bisect_right(value['days'],cutoff)-bisect.bisect_left(value['days'],cutoff-window) for window in (7,30,90,365)]
                values=[1,east/1000,north/1000,math.hypot(east,north)/1000,math.log1p(count)/6,math.log1p(age)/6,point[0]/90,point[1]/180,float(key==source),math.log1p(transition)/5,math.log1p(row['gap_days'])/5]
                values.extend(math.log1p(value)/4 for value in activity)
                history_distance=np.linalg.norm(history_positions-np.asarray([east/1000,north/1000]),axis=1)*1000
                rings=((25,7),(50,30),(100,90),(250,90))
                ring_masks=[(history_distance<=radius)&(history_days<=window) for radius,window in rings]
                values.extend(math.log1p(int(mask.sum()))/4 for mask in ring_masks)
                values.extend(math.log1p(float(history_fatalities[mask].sum()))/6 for mask in ring_masks)
                if raw_events:
                    raw_days,prefix=raw_events.get((conflict,key),([], [0.]))
                    right=bisect.bisect_right(raw_days,cutoff)
                    raw_counts=[]
                    for window in (7,30,90,365):raw_counts.append(right-bisect.bisect_left(raw_days,cutoff-window))
                    left90=bisect.bisect_left(raw_days,cutoff-90);fatalities90=prefix[right]-prefix[left90]
                    values.extend([*(math.log1p(value)/4 for value in raw_counts),math.log1p(fatalities90)/6])
                if mention_days:
                    days=mention_days[(str(row['country']).casefold(),round(point[0]*4),round(point[1]*4))]
                    values.extend(math.log1p(bisect.bisect_right(days,cutoff)-bisect.bisect_left(days,cutoff-window))/4 for window in (7,30,90))
                features[index,j]=values
                coordinates[index,j]=[east/1000,north/1000];valid[index,j]=True
            distance=np.linalg.norm(coordinates[index,:len(chosen)]-y[index,None],axis=1)*1000
            labels[index]=int(distance.argmin());oracle[index]=float(distance.min())
        for index in range(cursor,stop):
            row=rows[index];conflict=str(row['conflict_id']);source=pkey(row['anchor_lat'],row['anchor_lon']);target=pkey(row['target_lat'],row['target_lon']);day=int(np.datetime64(row['target_date'],'D').astype(int))
            value=locations[conflict].get(target)
            if value is None:value={'point':np.array([row['target_lat'],row['target_lon']],float),'count':0,'last':day,'days':[]};locations[conflict][target]=value
            value['count']+=1;value['last']=day;value['days'].append(day);transitions[conflict][source][target]+=1
        cursor=stop
    a.output.parent.mkdir(parents=True,exist_ok=True);np.savez_compressed(a.output,x=x,candidate_features=features,candidate_coordinates=coordinates,candidate_valid=valid,label=labels,y=y,meta=d['meta'])
    te=int(.7*n);ve=int(.85*n)
    print(json.dumps({'samples':n,'shape':list(features.shape),'oracle_mean_km':{'train':float(oracle[:te].mean()),'validation':float(oracle[te:ve].mean()),'test':float(oracle[ve:].mean())}},indent=2))
if __name__=='__main__':main()
