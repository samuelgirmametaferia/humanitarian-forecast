#!/usr/bin/env python3
"""Fit a validation-only context-dependent gate over heterogeneous location experts.

The gate learns *when* to trust each frozen expert rather than retraining them.
It is deliberately low-capacity and strongly regularized so a small Ethiopia
validation slice cannot memorize outcomes. Development labels are never used to
fit gate parameters.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from scipy.optimize import minimize

SCALE_KM = 1000.0


def _softmax(score: np.ndarray, axis: int = -1) -> np.ndarray:
    score = score - np.max(score, axis=axis, keepdims=True)
    exp = np.exp(score)
    return exp / np.maximum(exp.sum(axis=axis, keepdims=True), 1e-12)


def _candidate_softmax(score: np.ndarray, valid: np.ndarray, temperature: float) -> np.ndarray:
    s = score / max(float(temperature), 1e-5)
    s = np.where(valid, s, -np.inf)
    m = np.max(s, axis=1, keepdims=True)
    e = np.exp(np.where(valid, s - m, -np.inf))
    e = np.where(valid, e, 0.0)
    return e / np.maximum(e.sum(axis=1, keepdims=True), 1e-12)


def _soft_target(coordinates: np.ndarray, valid: np.ndarray, target: np.ndarray, radius_km: float = 100.0) -> np.ndarray:
    distance = np.linalg.norm(coordinates - target[:, None], axis=-1) * SCALE_KM
    q = np.exp(-0.5 * (distance / radius_km) ** 2) * valid
    total = q.sum(axis=1, keepdims=True)
    bad = total[:, 0] <= 1e-12
    if np.any(bad):
        nearest = np.argmin(np.where(valid[bad], distance[bad], np.inf), axis=1)
        q[bad] = 0.0
        q[np.flatnonzero(bad), nearest] = 1.0
        total = q.sum(axis=1, keepdims=True)
    return q / total


def _context(events: np.ndarray, rows: list[dict[str, object]], expert_probability: np.ndarray) -> np.ndarray:
    valid = events[..., 0] > 0.5
    w = valid.astype(np.float32)
    count = np.maximum(w.sum(axis=1, keepdims=True), 1.0)
    mean = (events * w[..., None]).sum(axis=1) / count
    last = events[:, -1]
    recent = events[:, -4:].mean(axis=1)
    previous = events[:, -8:-4].mean(axis=1)
    delta = recent - previous
    selected = np.asarray([1,2,3,4,5,6,7,8,9,10,11,12,13,14,19,20,21,22,23,24], dtype=np.int64)
    parts = [last[:, selected], mean[:, selected], delta[:, selected]]
    # Explicit regime/state summaries.
    position = events[..., 1:3]
    position_mean = (position * w[..., None]).sum(axis=1) / count
    spread = np.sqrt(np.sum((position - position_mean[:, None]) ** 2, axis=-1))
    spread = (spread * w).sum(axis=1, keepdims=True) / count
    gaps = np.asarray([[np.log1p(float(row["gap_days"])) / 5.0] for row in rows], dtype=np.float32)
    activity = np.stack([
        w.sum(axis=1) / events.shape[1],
        np.mean(events[:, -4:, 0] > 0.5, axis=1),
        np.mean(np.linalg.norm(events[:, -4:, 19:21], axis=-1), axis=1) if events.shape[-1] >= 21 else np.zeros(len(events)),
    ], axis=1).astype(np.float32)
    parts.extend([spread.astype(np.float32), gaps, activity])
    # Prediction-state features are legal at inference and useful for gating.
    entropy = -np.sum(expert_probability * np.log(np.maximum(expert_probability, 1e-12)), axis=-1)
    peak = np.max(expert_probability, axis=-1)
    parts.extend([entropy.astype(np.float32), peak.astype(np.float32)])
    return np.concatenate(parts, axis=1).astype(np.float64)


def _metrics(probability: np.ndarray, coordinates: np.ndarray, valid: np.ndarray, target: np.ndarray, q: np.ndarray) -> dict[str, float]:
    distance = np.linalg.norm(coordinates - target[:, None], axis=-1) * SCALE_KM
    distance = np.where(valid, distance, np.inf)
    top = probability.argmax(axis=1)
    error = distance[np.arange(len(distance)), top]
    ce = float(np.mean(-np.sum(q * np.log(np.maximum(probability, 1e-12)), axis=1)))
    out: dict[str, float] = {
        "samples": float(len(error)),
        "distance_soft_cross_entropy": ce,
        "mean_error_km": float(error.mean()),
        "median_error_km": float(np.median(error)),
        "p90_error_km": float(np.quantile(error, 0.90)),
        "candidate_oracle_mean_km": float(np.mean(np.min(distance, axis=1))),
        "mean_entropy": float(np.mean(-np.sum(probability * np.log(np.maximum(probability, 1e-12)), axis=1))),
    }
    for radius in (25,50,100,200):
        inside = distance <= radius
        out[f"within_{radius}km"] = float(np.mean(error <= radius))
        out[f"probability_mass_within_{radius}km"] = float(np.mean(np.sum(probability * inside, axis=1)))
    order = np.argsort(-probability, axis=1)
    top3 = np.take_along_axis(distance, order[:, :3], axis=1)
    out["top3_within_100km"] = float(np.mean(np.any(top3 <= 100, axis=1)))
    out["top3_within_200km"] = float(np.mean(np.any(top3 <= 200, axis=1)))
    out["broad_area_score"] = (
        .55*out["probability_mass_within_100km"] + .20*out["probability_mass_within_200km"]
        + .15*out["within_100km"] + .10*out["top3_within_100km"]
    )
    return out


def _load_local_probability(path: Path, validation_indices: np.ndarray, development_indices: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    z = np.load(path)
    vi = z["validation_indices"].astype(np.int64)
    di = z["development_indices"].astype(np.int64)
    if not np.array_equal(vi, validation_indices) or not np.array_equal(di, development_indices):
        raise ValueError(f"expert index alignment failed for {path}")
    return z["validation_probability"].astype(np.float64), z["development_probability"].astype(np.float64)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--v9", type=Path, required=True)
    parser.add_argument("--trees", type=Path, required=True)
    parser.add_argument("--ensemble-metrics", type=Path, required=True)
    parser.add_argument("--shape", type=Path, required=True)
    parser.add_argument("--hawkes", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--country", default="Ethiopia")
    parser.add_argument("--l2", type=float, default=0.25)
    parser.add_argument("--max-iter", type=int, default=300)
    args = parser.parse_args()
    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        raise FileExistsError(f"refusing to overwrite non-empty directory: {args.output_dir}")

    z = np.load(args.data, allow_pickle=True)
    x = z["x"].astype(np.float32, copy=False)
    coords = z["candidate_coordinates"].astype(np.float32, copy=False)
    valid = z["candidate_valid"]
    target = z["y"].astype(np.float32, copy=False)
    rows = [json.loads(str(value)) for value in z["meta"]]
    n=len(x); tr=int(.70*n); va=int(.85*n)
    country = np.asarray([row["country"] == args.country for row in rows])
    vi = np.flatnonzero(country & (np.arange(n)>=tr) & (np.arange(n)<va))
    di = np.flatnonzero(country & (np.arange(n)>=va))
    local_val = vi - tr
    local_dev = di - va

    calibration = json.loads(args.ensemble_metrics.read_text())["expert_calibration"]
    v9z=np.load(args.v9); treez=np.load(args.trees)
    v9_val=_candidate_softmax(v9z["validation_logits"][local_val], valid[vi], calibration["v9_phase1"]["temperature"])
    v9_dev=_candidate_softmax(v9z["development_logits"][local_dev], valid[di], calibration["v9_phase1"]["temperature"])
    actor_val=_candidate_softmax(treez["actor_validation"][local_val], valid[vi], calibration["actor_transfer"]["temperature"])
    actor_dev=_candidate_softmax(treez["actor_development"][local_dev], valid[di], calibration["actor_transfer"]["temperature"])
    kernel_val=_candidate_softmax(treez["kernel_validation"][local_val], valid[vi], calibration["propagation_kernel"]["temperature"])
    kernel_dev=_candidate_softmax(treez["kernel_development"][local_dev], valid[di], calibration["propagation_kernel"]["temperature"])
    shape_val,shape_dev=_load_local_probability(args.shape,vi,di)
    hawkes_val,hawkes_dev=_load_local_probability(args.hawkes,vi,di)
    names=["v9","actor_transfer","propagation_kernel","shape_analogue","marked_hawkes"]
    pv=np.stack([v9_val,actor_val,kernel_val,shape_val,hawkes_val],axis=1)
    pd=np.stack([v9_dev,actor_dev,kernel_dev,shape_dev,hawkes_dev],axis=1)

    cv=_context(x[vi],[rows[i] for i in vi],pv)
    cd=_context(x[di],[rows[i] for i in di],pd)
    mean=cv.mean(axis=0);std=np.maximum(cv.std(axis=0),1e-4)
    cv=(cv-mean)/std;cd=(cd-mean)/std
    qv=_soft_target(coords[vi],valid[vi],target[vi]);qd=_soft_target(coords[di],valid[di],target[di])
    e=len(names);f=cv.shape[1]

    # Static convex baseline selected on the same validation slice.
    def static_obj(raw: np.ndarray) -> float:
        w=_softmax(raw[None],axis=1)[0]
        mix=np.einsum('e,nej->nj',w,pv)
        return float(np.mean(-np.sum(qv*np.log(np.maximum(mix,1e-12)),axis=1)))
    static=minimize(static_obj,np.zeros(e),method="BFGS")
    static_w=_softmax(static.x[None],axis=1)[0]
    static_val=np.einsum('e,nej->nj',static_w,pv);static_dev=np.einsum('e,nej->nj',static_w,pd)

    def unpack(theta: np.ndarray) -> tuple[np.ndarray,np.ndarray]:
        W=theta[:f*e].reshape(f,e);b=theta[f*e:]
        return W,b
    def objective(theta: np.ndarray) -> tuple[float,np.ndarray]:
        W,b=unpack(theta)
        gate=_softmax(cv@W+b,axis=1)
        mix=np.einsum('ne,nej->nj',gate,pv)
        ratio=-np.sum(qv[:,None,:]*pv/np.maximum(mix[:,None,:],1e-12),axis=2)
        centered=ratio-np.sum(gate*ratio,axis=1,keepdims=True)
        dz=gate*centered/len(cv)
        gradW=cv.T@dz + 2*args.l2*W/max(1,W.size)
        gradb=dz.sum(axis=0)
        loss=float(np.mean(-np.sum(qv*np.log(np.maximum(mix,1e-12)),axis=1)) + args.l2*np.mean(W*W))
        return loss,np.concatenate([gradW.ravel(),gradb]).astype(np.float64)
    initial=np.zeros(f*e+e,dtype=np.float64)
    result=minimize(lambda t:objective(t),initial,jac=True,method="L-BFGS-B",options={"maxiter":args.max_iter,"ftol":1e-10})
    W,b=unpack(result.x)
    gv=_softmax(cv@W+b,axis=1);gd=_softmax(cd@W+b,axis=1)
    mixv=np.einsum('ne,nej->nj',gv,pv);mixd=np.einsum('ne,nej->nj',gd,pd)

    # Disagreement is stored for abstention/calibration research.
    dev_disagreement=np.mean(np.var(pd,axis=1),axis=1)
    report={
        "schema":"ethiopia-regime-gate-v1",
        "experts":names,
        "fit":{"success":bool(result.success),"message":str(result.message),"iterations":int(result.nit),"objective":float(result.fun),"l2":args.l2,"context_dim":f},
        "static_weights":dict(zip(names,map(float,static_w))),
        "mean_gate_weights_validation":dict(zip(names,map(float,gv.mean(axis=0)))),
        "mean_gate_weights_development":dict(zip(names,map(float,gd.mean(axis=0)))),
        "static_validation":_metrics(static_val,coords[vi],valid[vi],target[vi],qv),
        "static_development":_metrics(static_dev,coords[di],valid[di],target[di],qd),
        "gated_validation":_metrics(mixv,coords[vi],valid[vi],target[vi],qv),
        "gated_development":_metrics(mixd,coords[di],valid[di],target[di],qd),
        "development_disagreement":{"mean":float(dev_disagreement.mean()),"p90":float(np.quantile(dev_disagreement,.9)),"p99":float(np.quantile(dev_disagreement,.99))},
        "protocol":"All expert weights and gate parameters fit on chronological validation only; development labels are not used for fitting.",
        "evaluation_caveat":"Development block has been inspected by prior research and is not a pristine prospective test.",
    }
    args.output_dir.mkdir(parents=True,exist_ok=True)
    np.savez_compressed(args.output_dir/'regime_gate_model.npz',W=W,b=b,context_mean=mean,context_std=std,expert_names=np.asarray(names))
    np.savez_compressed(args.output_dir/'regime_gate_probabilities.npz',validation_indices=vi,validation_probability=mixv.astype(np.float32),development_indices=di,development_probability=mixd.astype(np.float32),validation_weights=gv.astype(np.float32),development_weights=gd.astype(np.float32),development_disagreement=dev_disagreement.astype(np.float32))
    (args.output_dir/'regime_gate_metrics.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report,indent=2))

if __name__=='__main__':main()
