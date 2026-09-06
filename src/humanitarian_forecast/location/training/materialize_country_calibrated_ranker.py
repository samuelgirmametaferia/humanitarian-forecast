#!/usr/bin/env python3
"""Materialize a country-calibrated production wrapper around an all-data candidate ranker."""
from __future__ import annotations
import argparse, json
from pathlib import Path
import torch
from humanitarian_forecast.core.model_store import ModelInfo, read_info, write_info


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--resume',type=Path,required=True)
    p.add_argument('--output-dir',type=Path,required=True)
    p.add_argument('--version',required=True)
    p.add_argument('--country',required=True)
    p.add_argument('--reference-model-dir',type=Path)
    p.add_argument('--aggregation',default='weighted_geometric_median')
    p.add_argument('--temperature',type=float,default=1.0)
    p.add_argument('--history-reference',default='none')
    p.add_argument('--candidate-weight',type=float,default=1.0)
    p.add_argument('--selected-on',default='historical country validation; frozen before later-block check')
    a=p.parse_args()
    if not 0 <= a.candidate_weight <= 1: raise SystemExit('--candidate-weight must be in [0,1]')
    if a.output_dir.exists() and any(a.output_dir.iterdir()): raise FileExistsError(a.output_dir)
    state=torch.load(a.resume,map_location='cpu',weights_only=False)
    if not state.get('country_to_id') or not state.get('conflict_to_id'):
        raise SystemExit('resume checkpoint must contain production identity maps')
    if a.country not in state['country_to_id']:
        raise SystemExit(f'{a.country!r} is not present in parent identity map')
    calibration={
        'aggregation':a.aggregation,
        'temperature':a.temperature,
        'history_reference':a.history_reference,
        'candidate_weight':a.candidate_weight,
        'selected_on':a.selected_on,
    }
    out=dict(state)
    out['center_calibration']=calibration
    out['calibration_scope']=a.country
    out['country_filter']=a.country
    out['weight_parent']=str(a.resume)
    a.output_dir.mkdir(parents=True,exist_ok=False)
    torch.save(out,a.output_dir/'model.pt'); torch.save(out,a.output_dir/'candidate_ranker_calibrated.pt')
    report={
        'mode':'country-calibration-wrapper-no-weight-update',
        'country':a.country,
        'weight_parent':str(a.resume),
        'parent_training_samples':int(state.get('training_samples',0)),
        'calibration':calibration,
        'held_out_evaluation':False,
    }
    (a.output_dir/'training_metrics.json').write_text(json.dumps(report,indent=2)+'\n')
    ref={}
    if a.reference_model_dir and (a.reference_model_dir/'info.blt').exists(): ref=read_info(a.reference_model_dir)
    write_info(a.output_dir,ModelInfo(
        subsystem='location/candidate_ranker_ethiopia' if a.country.casefold()=='ethiopia' else 'location/candidate_ranker_country',
        version=a.version,
        status='production-all-data-country-calibrated',
        description=f'{a.country}-calibrated inference wrapper around the all-data v9-lineage candidate ranker; weights are unchanged.',
        metrics={'held_out_evaluation':False,'validated_ancestor':ref.get('metrics',{}),'training':report},
        lineage={'weight_parent':str(a.resume),'validated_reference':str(a.reference_model_dir) if a.reference_model_dir else None,'recipe':'v9 candidate ranker + country-specific calibrated center'},
        training={'mode':'calibration-only','parent_training_samples':int(state.get('training_samples',0)),'weight_update':False},
        calibration=calibration,
        notes=['No weights are fine-tuned in this wrapper.','The parent checkpoint was trained on all available labels.','Historical calibration selection/evaluation receipts are stored separately.'],
    ))
    print(json.dumps(report,indent=2))
if __name__=='__main__': main()
