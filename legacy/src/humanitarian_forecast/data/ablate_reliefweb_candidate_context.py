#!/usr/bin/env python3
"""Create controlled ReliefWeb candidate-context ablations.

The input must be produced by augment_candidate_reliefweb_spatial.py.  This
utility keeps the original candidate features and selects either report-volume
channels, semantic-content channels, or all ReliefWeb channels.  It exists so a
semantic gain cannot be confused with simple publication-volume intensity.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--mode", choices=("count-only", "content-only", "full"), required=True)
    a = p.parse_args()
    z = np.load(a.data, allow_pickle=True)
    features = z["candidate_features"].astype(np.float32, copy=False)
    base_dim = int(np.asarray(z["base_candidate_dim"]).item())
    names = [str(v) for v in z["reliefweb_appended_feature_names"]]
    if features.shape[-1] != base_dim + len(names):
        raise ValueError("candidate feature dimension does not match ReliefWeb lineage")
    if a.mode == "full":
        keep = np.arange(len(names), dtype=np.int64)
    elif a.mode == "count-only":
        keep = np.asarray([i for i, name in enumerate(names) if "mention_count" in name], dtype=np.int64)
    else:
        keep = np.asarray([i for i, name in enumerate(names) if "mention_count" not in name], dtype=np.int64)
    selected = np.concatenate([features[..., :base_dim], features[..., base_dim:][..., keep]], axis=-1)
    payload = {k: z[k] for k in z.files if k not in {"candidate_features", "reliefweb_appended_feature_names"}}
    payload["candidate_features"] = selected
    payload["reliefweb_appended_feature_names"] = np.asarray([names[i] for i in keep])
    payload["reliefweb_ablation"] = np.asarray(json.dumps({"schema":"reliefweb-candidate-ablation-v1","mode":a.mode,"source":str(a.data),"selected_channels":len(keep)}, sort_keys=True))
    a.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(a.output, **payload)
    report={"schema":"reliefweb-candidate-ablation-v1","mode":a.mode,"candidate_dim":int(selected.shape[-1]),"base_dim":base_dim,"selected_reliefweb_channels":len(keep),"output":str(a.output)}
    a.output.with_suffix('.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report,indent=2))


if __name__ == "__main__":
    main()
