#!/usr/bin/env python3
"""Audit whether one coordinate represents a multimodal conflict-day label."""
from __future__ import annotations
import argparse,csv,io,json,zipfile
from collections import defaultdict
from pathlib import Path
import numpy as np
from full_history_forecaster import haversine
def main():
    p=argparse.ArgumentParser();p.add_argument('--data',type=Path,required=True);p.add_argument('--ucdp',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args();groups=defaultdict(list)
    with zipfile.ZipFile(a.ucdp) as z:
        name=next(v for v in z.namelist() if v.endswith('.csv'))
        with z.open(name) as raw:
            for r in csv.DictReader(io.TextIOWrapper(raw,encoding='utf-8-sig')):
                try:
                    if int(r['where_prec'])>4 or int(r['date_prec'])>3:continue
                    point=(float(r['latitude']),float(r['longitude']))
                except (ValueError,KeyError):continue
                groups[(r['conflict_new_id'],r['date_start'][:10])].append(point)
    d=np.load(a.data);n=len(d['meta']);start=int(.85*n);rows=[json.loads(str(v)) for v in d['meta'][start:]];counts=[];diameters=[]
    for r in rows:
        pts=np.asarray(groups[(str(r['conflict_id']),r['target_date'])]);counts.append(len(pts))
        if len(pts)<=1:diameters.append(0.0)
        else:
            # Maximum separation from the labeled first record is a conservative ambiguity measure.
            diameters.append(float(haversine(pts,np.broadcast_to(pts[0],pts.shape)).max()))
    counts=np.asarray(counts);diameters=np.asarray(diameters)
    report={'test_samples':len(rows),'multi_event_conflict_day_rate':float((counts>1).mean()),'events_per_labeled_day':{'mean':float(counts.mean()),'median':float(np.median(counts)),'p90':float(np.quantile(counts,.9))},'first_record_to_other_event_spread_km':{'mean':float(diameters.mean()),'median':float(np.median(diameters)),'p90':float(np.quantile(diameters,.9)),'over_40km_rate':float((diameters>40).mean())},'interpretation':'Dataset retains only the first ID-ordered event when a conflict has multiple events on the next active day.'}
    a.output.write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report,indent=2))
if __name__=='__main__':main()
