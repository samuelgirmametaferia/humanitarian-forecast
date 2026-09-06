#!/usr/bin/env python3
"""Hierarchical transition modes with a learned-center candidate backoff."""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from dataclasses import dataclass
import json
from pathlib import Path
import sys

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from humanitarian_forecast.evaluation.evaluate_candidate_snap import absolute_query, model_queries
from humanitarian_forecast.evaluation.evaluate_candidate_ranker import Bank
from humanitarian_forecast.evaluation.full_history_forecaster import haversine, summarize


@dataclass(frozen=True)
class Config:
    scope: str
    source_resolution_degrees: float
    minimum_count: int


def point_key(lat, lon, resolution):
    if resolution <= 1e-4:
        return round(float(lat), 5), round(float(lon), 5)
    return round(float(lat) / resolution), round(float(lon) / resolution)


class Transitions:
    def __init__(self, resolutions):
        self.tables = {
            (scope, resolution): defaultdict(Counter)
            for scope in ("conflict", "country") for resolution in resolutions
        }
        self.points = {}

    def add(self, row):
        destination_key = point_key(row["target_lat"], row["target_lon"], 1e-5)
        self.points[destination_key] = np.asarray(
            [row["target_lat"], row["target_lon"]], dtype=float
        )
        for (scope, resolution), table in self.tables.items():
            identity = str(row["conflict_id"] if scope == "conflict" else row["country"])
            source = point_key(row["anchor_lat"], row["anchor_lon"], resolution)
            table[(identity, source)][destination_key] += 1

    def mode(self, row, config):
        identity = str(row["conflict_id"] if config.scope == "conflict" else row["country"])
        source = point_key(row["anchor_lat"], row["anchor_lon"], config.source_resolution_degrees)
        counts = self.tables[(config.scope, config.source_resolution_degrees)].get((identity, source))
        if not counts:
            return None
        target_key, count = counts.most_common(1)[0]
        return self.points[target_key] if count >= config.minimum_count else None


def fallback(bank, row, query):
    anchor = np.asarray([row["anchor_lat"], row["anchor_lon"]], dtype=float)
    points, country_count, _ = bank.arrays(row, anchor)
    query_distance = haversine(points, np.broadcast_to(query, points.shape))
    score = query_distance - 7.5 * np.log1p(country_count)
    return points[np.argmin(score)]


def evaluate(rows, queries, seed_end, start, end, choices):
    resolutions = sorted({c.source_resolution_degrees for c in choices})
    transitions = Transitions(resolutions); bank = Bank()
    for row in rows[:seed_end]: transitions.add(row); bank.add(row)
    errors = np.zeros((len(choices), end-start)); uses = np.zeros(len(choices), dtype=int)
    cursor = start
    while cursor < end:
        date = rows[cursor]["target_date"]; group_end = cursor
        while group_end < end and rows[group_end]["target_date"] == date: group_end += 1
        for index in range(cursor, group_end):
            row = rows[index]
            target = np.asarray([row["target_lat"], row["target_lon"]], dtype=float)
            backoff = fallback(bank, row, queries[index])
            for number, config in enumerate(choices):
                prediction = transitions.mode(row, config)
                if prediction is None: prediction = backoff
                else: uses[number] += 1
                errors[number, index-start] = float(haversine(prediction, target))
        for row in rows[cursor:group_end]: transitions.add(row); bank.add(row)
        cursor = group_end
    return errors, uses


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data", type=Path, required=True)
    p.add_argument("--checkpoint", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    a = p.parse_args()
    data = np.load(a.data); rows = [json.loads(str(v)) for v in data["meta"]]
    top, _ = model_queries(data, a.checkpoint)
    history = np.asarray([.95*r[r[:,0] > 0,1:3].mean(0) for r in data["x"]])
    offsets = .8*top + .2*history
    queries = np.asarray([absolute_query(row, off) for row, off in zip(rows, offsets)])
    choices = [
        Config(scope, resolution, count)
        for scope in ("conflict", "country")
        for resolution in (1e-5, .10, .25, .50, 1.0)
        for count in (1, 2, 4, 8, 16)
    ]
    n=len(rows); train_end=int(.7*n); validation_end=int(.85*n)
    ve, vu = evaluate(rows, queries, train_end, train_end, validation_end, choices)
    means=ve.mean(1); selected_index=int(means.argmin()); selected=choices[selected_index]
    te, tu = evaluate(rows, queries, validation_end, validation_end, n, [selected])
    order=np.argsort(means)
    report={
        "selection":"transition scope/resolution/support selected on validation mean error",
        "selected_config":selected.__dict__,
        "validation":{**summarize(ve[selected_index]),"transition_use_rate":float(vu[selected_index]/ve.shape[1])},
        "untouched_test":{**summarize(te[0]),"transition_use_rate":float(tu[0]/te.shape[1])},
        "top_validation_trials":[
            {"config":choices[i].__dict__,"metrics":{**summarize(ve[i]),"transition_use_rate":float(vu[i]/ve.shape[1])}}
            for i in order[:10]
        ],
        "cutoff_policy":"transitions and candidates update after all forecasts on each target date",
    }
    a.output.parent.mkdir(parents=True,exist_ok=True);a.output.write_text(json.dumps(report,indent=2)+"\n")
    print(json.dumps(report,indent=2))


if __name__=="__main__":main()
