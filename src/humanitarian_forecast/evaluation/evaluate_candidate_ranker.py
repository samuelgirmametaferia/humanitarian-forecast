#!/usr/bin/env python3
"""Rank cutoff-safe historical locations using validation-selected features."""

from __future__ import annotations

import argparse
from collections import defaultdict
from dataclasses import dataclass
import json
from pathlib import Path
import sys

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from humanitarian_forecast.evaluation.evaluate_candidate_snap import absolute_query, model_queries
from humanitarian_forecast.evaluation.full_history_forecaster import haversine, summarize


@dataclass(frozen=True)
class Config:
    anchor_weight: float
    country_frequency_bonus_km: float
    conflict_frequency_bonus_km: float
    recency_penalty_km: float = 0.0


@dataclass
class Candidate:
    point: np.ndarray
    country_count: int = 0
    conflict_counts: dict[str, int] | None = None
    last_date_ordinal: int = -1

    def __post_init__(self):
        if self.conflict_counts is None:
            self.conflict_counts = defaultdict(int)


class Bank:
    def __init__(self):
        self.countries: dict[str, dict[tuple[float, float], Candidate]] = defaultdict(dict)

    def add(self, row):
        country, conflict = str(row["country"]), str(row["conflict_id"])
        point = np.asarray([row["target_lat"], row["target_lon"]], dtype=float)
        # UCDP repeats exact coordinates heavily. Five decimals preserves a
        # roughly metre-scale identity while merging floating serialization noise.
        key = (round(float(point[0]), 5), round(float(point[1]), 5))
        candidate = self.countries[country].get(key)
        if candidate is None:
            candidate = Candidate(point=point)
            self.countries[country][key] = candidate
        candidate.country_count += 1
        candidate.conflict_counts[conflict] += 1
        candidate.last_date_ordinal = int(np.datetime64(row["target_date"], "D").astype(int))

    def arrays(self, row, anchor):
        candidates = list(self.countries[str(row["country"])].values())
        # The latest observed location is always a valid fallback candidate.
        points = np.asarray([anchor, *[c.point for c in candidates]])
        country_count = np.asarray([1, *[c.country_count for c in candidates]], dtype=float)
        conflict = str(row["conflict_id"])
        conflict_count = np.asarray(
            [1, *[c.conflict_counts.get(conflict, 0) for c in candidates]], dtype=float
        )
        return points, country_count, conflict_count

    def arrays_with_recency(self, row, anchor):
        candidates = list(self.countries[str(row["country"])].values())
        points, country_count, conflict_count = self.arrays(row, anchor)
        target_day = int(np.datetime64(row["target_date"], "D").astype(int))
        fallback_age = max(1, int(row.get("gap_days", 1)))
        age_days = np.asarray([
            fallback_age, *[max(1, target_day - c.last_date_ordinal) for c in candidates]
        ], dtype=float)
        return points, country_count, conflict_count, age_days


def configs():
    # Static validation selected zero anchor/conflict weights. Cross the useful
    # country-frequency range with a temporal activity penalty.
    return [
        Config(0.0, country, 0.0, recency)
        for country in (0.0, 5.0, 7.5, 10.0, 20.0)
        for recency in (0.0, 2.5, 5.0, 10.0, 20.0)
    ]


def evaluate(rows, queries, seed_end, start, end, choices):
    bank = Bank()
    for row in rows[:seed_end]: bank.add(row)
    errors = np.zeros((len(choices), end - start), dtype=float)
    cursor = start
    while cursor < end:
        date = rows[cursor]["target_date"]; group_end = cursor
        while group_end < end and rows[group_end]["target_date"] == date: group_end += 1
        for index in range(cursor, group_end):
            row = rows[index]
            anchor = np.asarray([row["anchor_lat"], row["anchor_lon"]], dtype=float)
            target = np.asarray([row["target_lat"], row["target_lon"]], dtype=float)
            points, country_count, conflict_count, age_days = bank.arrays_with_recency(row, anchor)
            query_distance = haversine(points, np.broadcast_to(queries[index], points.shape))
            anchor_distance = haversine(points, np.broadcast_to(anchor, points.shape))
            log_country = np.log1p(country_count); log_conflict = np.log1p(conflict_count)
            for number, choice in enumerate(choices):
                score = (query_distance + choice.anchor_weight * anchor_distance
                         - choice.country_frequency_bonus_km * log_country
                         - choice.conflict_frequency_bonus_km * log_conflict
                         + choice.recency_penalty_km * np.log1p(age_days))
                prediction = points[np.argmin(score)]
                errors[number, index - start] = float(haversine(prediction, target))
        for row in rows[cursor:group_end]: bank.add(row)
        cursor = group_end
    return errors


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data", type=Path, required=True)
    p.add_argument("--checkpoint", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    a = p.parse_args()
    data = np.load(a.data); rows = [json.loads(str(v)) for v in data["meta"]]
    top_offset, _ = model_queries(data, a.checkpoint)
    history_offset = np.asarray([
        .95 * row[row[:, 0] > 0, 1:3].mean(0) for row in data["x"]
    ])
    # The preceding validation search selected this query construction.
    offset = .80 * top_offset + .20 * history_offset
    queries = np.asarray([absolute_query(row, value) for row, value in zip(rows, offset)])
    n = len(rows); train_end = int(.70*n); validation_end = int(.85*n)
    choices = configs()
    validation_errors = evaluate(rows, queries, train_end, train_end, validation_end, choices)
    validation_means = validation_errors.mean(axis=1)
    selected_index = int(validation_means.argmin()); selected = choices[selected_index]
    test_errors = evaluate(rows, queries, validation_end, validation_end, n, [selected])[0]
    order = np.argsort(validation_means)
    report = {
        "selection": "ranking weights selected by chronological validation mean error",
        "query": "80% mixture top-1 + 20% shrunk history mean",
        "selected_config": selected.__dict__,
        "validation": summarize(validation_errors[selected_index]),
        "untouched_test": summarize(test_errors),
        "top_validation_trials": [
            {"config": choices[i].__dict__, "metrics": summarize(validation_errors[i])}
            for i in order[:10]
        ],
        "cutoff_policy": "candidate counts update after all forecasts on each target date",
    }
    a.output.parent.mkdir(parents=True, exist_ok=True)
    a.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__": main()
