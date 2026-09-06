#!/usr/bin/env python3
"""Measure cutoff-safe oracle recall for expanded historical candidate banks."""

from __future__ import annotations

import argparse
from collections import defaultdict
import json
from pathlib import Path

import numpy as np

from humanitarian_forecast.evaluation.full_history_forecaster import haversine, summarize


class Banks:
    def __init__(self) -> None:
        self.conflict = defaultdict(list)
        self.country = defaultdict(list)

    def add(self, row):
        point = np.asarray([row["target_lat"], row["target_lon"]], dtype=float)
        self.conflict[str(row["conflict_id"])].append(point)
        self.country[str(row["country"])].append(point)

    @staticmethod
    def nearest(target, points, fallback):
        candidates = np.asarray([fallback, *points])
        return float(haversine(candidates, np.broadcast_to(target, candidates.shape)).min())


def evaluate(rows, seed_end, start, end):
    banks = Banks()
    for row in rows[:seed_end]:
        banks.add(row)
    result = {"conflict_history": [], "country_history": []}
    cursor = start
    while cursor < end:
        date = rows[cursor]["target_date"]
        group_end = cursor
        while group_end < end and rows[group_end]["target_date"] == date:
            group_end += 1
        for row in rows[cursor:group_end]:
            target = np.asarray([row["target_lat"], row["target_lon"]], dtype=float)
            anchor = np.asarray([row["anchor_lat"], row["anchor_lon"]], dtype=float)
            result["conflict_history"].append(
                banks.nearest(target, banks.conflict[str(row["conflict_id"])], anchor)
            )
            result["country_history"].append(
                banks.nearest(target, banks.country[str(row["country"])], anchor)
            )
        for row in rows[cursor:group_end]:
            banks.add(row)
        cursor = group_end
    return {name: summarize(np.asarray(error)) for name, error in result.items()}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    a = p.parse_args()
    data = np.load(a.data)
    rows = [json.loads(str(value)) for value in data["meta"]]
    n = len(rows); train_end = int(.70 * n); validation_end = int(.85 * n)
    report = {
        "interpretation": "Retrospective candidate recall only; not forecast performance",
        "validation": evaluate(rows, train_end, train_end, validation_end),
        "untouched_test": evaluate(rows, validation_end, validation_end, n),
        "cutoff_policy": "Banks update only after every event on a target date is scored",
    }
    a.output.parent.mkdir(parents=True, exist_ok=True)
    a.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
