#!/usr/bin/env python3
"""Select historical candidates nearest to cutoff-safe forecast query centers."""

from __future__ import annotations

import argparse
from collections import defaultdict
import json
import math
from pathlib import Path
import sys

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from humanitarian_forecast.evaluation.full_history_forecaster import haversine, summarize
from humanitarian_forecast.location.models.mixture import MixtureLocationTransformer


def model_queries(data, checkpoint: Path, batch_size: int = 1024):
    state = torch.load(checkpoint, map_location="cpu", weights_only=False)
    model = MixtureLocationTransformer(**state["model_config"])
    model.load_state_dict(state["model_state"]); model.eval()
    top_centers = []; mean_centers = []
    x = torch.from_numpy(data["x"]).float()
    with torch.no_grad():
        for start in range(0, len(x), batch_size):
            logits, candidate_centers, _ = model(x[start:start + batch_size])
            top = logits.argmax(-1)
            top_centers.append(candidate_centers[torch.arange(len(top)), top].numpy())
            probability = logits.softmax(-1)
            mean_centers.append((candidate_centers * probability[..., None]).sum(1).numpy())
    return np.concatenate(top_centers), np.concatenate(mean_centers)


def absolute_query(row, offset):
    north, east = float(offset[1]) * 1000.0, float(offset[0]) * 1000.0
    lat = float(row["anchor_lat"]) + north / 111.32
    lon = float(row["anchor_lon"]) + east / (
        111.32 * max(.1, math.cos(math.radians(float(row["anchor_lat"]))))
    )
    return np.asarray([lat, lon])


class Banks:
    def __init__(self):
        self.conflict = defaultdict(list); self.country = defaultdict(list)

    def add(self, row):
        point = np.asarray([row["target_lat"], row["target_lon"]], dtype=float)
        self.conflict[str(row["conflict_id"])].append(point)
        self.country[str(row["country"])].append(point)

    @staticmethod
    def snap(query, anchor, points):
        candidates = np.asarray([anchor, *points])
        distance = haversine(candidates, np.broadcast_to(query, candidates.shape))
        return candidates[np.argmin(distance)]


def evaluate(rows, queries, seed_end, start, end, bank_name):
    banks = Banks()
    for row in rows[:seed_end]: banks.add(row)
    errors = []; cursor = start
    while cursor < end:
        date = rows[cursor]["target_date"]; group_end = cursor
        while group_end < end and rows[group_end]["target_date"] == date: group_end += 1
        for index in range(cursor, group_end):
            row = rows[index]
            anchor = np.asarray([row["anchor_lat"], row["anchor_lon"]], dtype=float)
            target = np.asarray([row["target_lat"], row["target_lon"]], dtype=float)
            points = getattr(banks, bank_name)[str(row[
                "conflict_id" if bank_name == "conflict" else "country"
            ])]
            prediction = banks.snap(queries[index], anchor, points)
            errors.append(float(haversine(prediction, target)))
        for row in rows[cursor:group_end]: banks.add(row)
        cursor = group_end
    return np.asarray(errors)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data", type=Path, required=True)
    p.add_argument("--checkpoint", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    a = p.parse_args()
    data = np.load(a.data); rows = [json.loads(str(v)) for v in data["meta"]]
    model_offset, mixture_mean_offset = model_queries(data, a.checkpoint)
    history_offset = np.asarray([
        .95 * row[row[:, 0] > 0, 1:3].mean(0) for row in data["x"]
    ])
    offset_sets = {"mixture_top1": model_offset}
    # Coarse validation established the top-1/history family as the winner;
    # refine only that family without consulting test results.
    for weight in (.50, .55, .60, .65, .70, .75, .80, .85, .90, .95):
        offset_sets[f"top1_history_blend_{weight:.2f}"] = (
            weight * model_offset + (1 - weight) * history_offset
        )
    query_sets = {
        name: np.asarray([absolute_query(row, off) for row, off in zip(rows, offsets)])
        for name, offsets in offset_sets.items()
    }
    n = len(rows); train_end = int(.70*n); validation_end = int(.85*n)
    trials = []
    for query_name, queries in query_sets.items():
        # The first experiment established that the broader country bank has
        # lower validation mean for both independent query families.
        for bank_name in ("country",):
            errors = evaluate(rows, queries, train_end, train_end, validation_end, bank_name)
            trials.append((float(errors.mean()), query_name, bank_name, summarize(errors)))
    trials.sort()
    _, query_name, bank_name, validation_metrics = trials[0]
    test_errors = evaluate(rows, query_sets[query_name], validation_end, validation_end, n, bank_name)
    report = {
        "selection": "query and candidate bank selected by validation mean error",
        "selected": {"query": query_name, "bank": bank_name},
        "validation": validation_metrics,
        "untouched_test": summarize(test_errors),
        "validation_trials": [
            {"query": q, "bank": b, "metrics": m} for _, q, b, m in trials
        ],
        "cutoff_policy": "candidate banks update after all forecasts on each target date",
    }
    a.output.parent.mkdir(parents=True, exist_ok=True)
    a.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__": main()
