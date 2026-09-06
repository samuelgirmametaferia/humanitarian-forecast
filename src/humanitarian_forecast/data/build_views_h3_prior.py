#!/usr/bin/env python3
"""Project an immutable VIEWS PRIO-grid forecast run onto dense H3 cells.

The raw VIEWS run remains untouched. For each H3 cell, the containing 0.5-degree
PRIO-GRID id is computed deterministically and the selected VIEWS model output is
copied into that cell. Both raw values and within-month normalized probability
mass are stored so downstream models can choose how to consume the prior.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def priogrid_id(lat: np.ndarray, lon: np.ndarray) -> np.ndarray:
    row = np.floor((lat + 90.0) / 0.5).astype(np.int64)
    col = np.floor((lon + 180.0) / 0.5).astype(np.int64)
    row = np.clip(row, 0, 359)
    col = np.clip(col, 0, 719)
    return row * 720 + col + 1


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--views", type=Path, required=True)
    parser.add_argument("--h3", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--model", default="main_mean_ln")
    args = parser.parse_args()

    vz = np.load(args.views, allow_pickle=True)
    hz = np.load(args.h3, allow_pickle=True)
    models = [str(v) for v in vz["model_names"]]
    if args.model not in models:
        raise ValueError(f"model {args.model!r} not found; available={models}")
    model_col = models.index(args.model)
    months = np.unique(vz["month_id"].astype(np.int32))
    resolutions = tuple(int(v) for v in hz["resolutions"])
    values_by_key = {
        (int(gid), int(month)): float(value)
        for gid, month, value in zip(vz["pg_id"], vz["month_id"], vz["values"][:, model_col])
        if np.isfinite(value)
    }

    payload: dict[str, np.ndarray] = {
        "month_id": months,
        "run_id": vz["run_id"],
        "model_name": np.asarray(args.model),
        "schema": np.asarray("views-h3-prior-v1"),
        "resolutions": np.asarray(resolutions, dtype=np.int16),
    }
    report: dict[str, object] = {
        "schema": "views-h3-prior-v1",
        "run_id": str(vz["run_id"]),
        "model": args.model,
        "months": [int(months.min()), int(months.max())],
        "resolutions": {},
        "causality_rule": "Use only if this exact VIEWS production run existed by the forecast cutoff.",
    }

    for resolution in resolutions:
        centroids = hz[f"centroids_r{resolution}"].astype(np.float64)
        gids = priogrid_id(centroids[:, 0], centroids[:, 1])
        raw = np.full((len(months), len(gids)), np.nan, dtype=np.float32)
        for mi, month in enumerate(months):
            raw[mi] = np.asarray([values_by_key.get((int(gid), int(month)), np.nan) for gid in gids], dtype=np.float32)
        coverage = np.isfinite(raw)
        safe = np.nan_to_num(raw, nan=0.0, posinf=0.0, neginf=0.0)
        mass = safe / np.maximum(safe.sum(axis=1, keepdims=True), 1e-12)
        payload[f"prio_gid_r{resolution}"] = gids.astype(np.int64)
        payload[f"raw_r{resolution}"] = raw
        payload[f"mass_r{resolution}"] = mass.astype(np.float32)
        report["resolutions"][str(resolution)] = {
            "cells": int(len(gids)),
            "coverage": float(coverage.mean()),
            "month_mass_min": float(mass.sum(axis=1).min()),
            "month_mass_max": float(mass.sum(axis=1).max()),
        }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.output, **payload)
    args.output.with_suffix(".json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
