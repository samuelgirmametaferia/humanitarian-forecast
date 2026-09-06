#!/usr/bin/env python3
"""Quick signal check on the v6 spillover dataset: oracle coverage, baseline
rankers, and a boosted-tree candidate reranker, all on Ethiopia validation."""
import json
import time

import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier

data = np.load("data/location/conflict_candidates_64_spillover_v6.npz", allow_pickle=True)
meta = [json.loads(str(v)) for v in data["meta"]]
n = len(meta)
te, ve = int(.7 * n), int(.85 * n)
cc_all = data["candidate_coordinates"].astype(np.float64)
y_all = data["y"].astype(np.float64)
valid_all = data["candidate_valid"].copy()
cf_all = data["candidate_features"].astype(np.float64)
lab = data["label"].copy()
d_all = np.linalg.norm(cc_all - y_all[:, None, :], axis=2) * 1000

eth = np.asarray([i for i, m in enumerate(meta) if m["country"] == "Ethiopia"])
vr = eth[(eth >= te) & (eth < ve)]
dr = eth[eth >= ve]


def err_of(scores, rows):
    sc = np.where(valid_all[rows], scores, -np.inf)
    top = sc.argmax(1)
    e = d_all[rows][np.arange(len(rows)), top]
    return {"median_km": round(float(np.median(e)), 1), "w20": round(float(np.mean(e <= 20)), 3),
            "w50": round(float(np.mean(e <= 50)), 3)}


print(json.dumps({
    "ethiopia_oracle_w20": {"validation": round(float(np.mean(d_all[vr].min(1) <= 20)), 3),
                             "development": round(float(np.mean(d_all[dr].min(1) <= 20)), 3)},
    "ethiopia_oracle_median": {"validation": round(float(np.median(d_all[vr].min(1))), 1),
                                "development": round(float(np.median(d_all[dr].min(1))), 1)},
}))
print("freq-rank order:", json.dumps(err_of(np.tile(-np.arange(cc_all.shape[1], dtype=np.float64), (len(vr), 1)), vr)))

t0 = time.time()
rng = np.random.default_rng(0)
train_rows = np.arange(0, te)
sub = rng.choice(train_rows, size=min(60000, len(train_rows)), replace=False)
X = cf_all[sub][valid_all[sub]]
ys = (lab[sub][:, None] == np.arange(cc_all.shape[1])[None, :])[valid_all[sub]]
m = HistGradientBoostingClassifier(max_iter=300, learning_rate=0.08, early_stopping=True, validation_fraction=0.1, random_state=0)
m.fit(X, ys)
probs = np.zeros((len(vr), cc_all.shape[1]))
v = valid_all[vr]
probs[v] = m.predict_proba(cf_all[vr][v])[:, 1]
print("GBT reranker (validation):", json.dumps(err_of(probs, vr)), f"({time.time()-t0:.0f}s)")
