#!/usr/bin/env python3
"""Cutoff-safe full-history and transition-location forecaster.

Hyperparameters are selected on the chronological validation split. The test
split is evaluated once with the selected configuration while observations are
added only after each target date has been predicted.
"""

from __future__ import annotations

import argparse
from collections import defaultdict
from dataclasses import dataclass
import json
import math
from pathlib import Path

import numpy as np


def haversine(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    lat1, lon1 = np.radians(a[..., 0]), np.radians(a[..., 1])
    lat2, lon2 = np.radians(b[..., 0]), np.radians(b[..., 1])
    dlat, dlon = lat2 - lat1, lon2 - lon1
    h = np.sin(dlat / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2) ** 2
    return 6371.0088 * 2 * np.arcsin(np.minimum(1.0, np.sqrt(h)))


@dataclass(frozen=True)
class Config:
    neighbors: int
    bandwidth_km: float
    recency: int
    persistence_weight: float


class History:
    def __init__(self) -> None:
        self.rows: dict[str, list[tuple[np.ndarray, np.ndarray]]] = defaultdict(list)

    def add(self, conflict: str, anchor: np.ndarray, target: np.ndarray) -> None:
        self.rows[conflict].append((anchor, target))

    def predict(self, conflict: str, anchor: np.ndarray, config: Config) -> np.ndarray:
        rows = self.rows.get(conflict)
        if not rows:
            return anchor.copy()
        rows = rows[-config.recency:]
        sources = np.asarray([r[0] for r in rows])
        destinations = np.asarray([r[1] for r in rows])
        distance = haversine(sources, np.broadcast_to(anchor, sources.shape))
        take = np.argsort(distance)[:config.neighbors]
        weights = np.exp(-0.5 * (distance[take] / config.bandwidth_km) ** 2) + 1e-8
        destination = np.average(destinations[take], axis=0, weights=weights)
        return config.persistence_weight * anchor + (1.0 - config.persistence_weight) * destination


def summarize(errors: np.ndarray) -> dict[str, float | int | list[float]]:
    rng = np.random.default_rng(20260816)
    boot = np.asarray([rng.choice(errors, len(errors), replace=True).mean() for _ in range(1000)])
    return {
        "samples": int(len(errors)),
        "mean_error_km": float(errors.mean()),
        "mean_error_95ci_km": [float(v) for v in np.quantile(boot, [0.025, 0.975])],
        "median_error_km": float(np.median(errors)),
        "p90_error_km": float(np.quantile(errors, 0.9)),
        "within_25km": float((errors <= 25).mean()),
    }


def rows_from(data: np.lib.npyio.NpzFile) -> list[dict[str, object]]:
    return [json.loads(str(value)) for value in data["meta"]]


def seed_history(history: History, rows: list[dict[str, object]], end: int) -> None:
    for row in rows[:end]:
        history.add(
            str(row["conflict_id"]),
            np.asarray([row["anchor_lat"], row["anchor_lon"]], dtype=float),
            np.asarray([row["target_lat"], row["target_lon"]], dtype=float),
        )


def evaluate(
    rows: list[dict[str, object]], start: int, end: int, history: History, config: Config
) -> np.ndarray:
    errors: list[float] = []
    cursor = start
    while cursor < end:
        date = rows[cursor]["target_date"]
        group_end = cursor
        while group_end < end and rows[group_end]["target_date"] == date:
            group_end += 1
        pending = []
        for row in rows[cursor:group_end]:
            anchor = np.asarray([row["anchor_lat"], row["anchor_lon"]], dtype=float)
            target = np.asarray([row["target_lat"], row["target_lon"]], dtype=float)
            prediction = history.predict(str(row["conflict_id"]), anchor, config)
            errors.append(float(haversine(prediction, target)))
            pending.append((str(row["conflict_id"]), anchor, target))
        # Same-day targets are unavailable to one another at the forecast cutoff.
        for conflict, anchor, target in pending:
            history.add(conflict, anchor, target)
        cursor = group_end
    return np.asarray(errors)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    data = np.load(args.data)
    rows = rows_from(data)
    n = len(rows)
    train_end, validation_end = int(n * 0.70), int(n * 0.85)
    # A deliberately compact grid keeps this reproducible on a laptop. Wider
    # searches should be run only after this family clears the baseline gate.
    configs = [
        Config(k, bandwidth, recency, persistence)
        for k in (1, 4, 16)
        for bandwidth in (25.0, 100.0)
        for recency in (64, 4096)
        for persistence in (0.0, 0.25)
    ]
    trials = []
    for number, config in enumerate(configs, 1):
        history = History()
        seed_history(history, rows, train_end)
        error = evaluate(rows, train_end, validation_end, history, config)
        trials.append((float(error.mean()), config, summarize(error)))
        if number % 50 == 0:
            print(f"validation {number}/{len(configs)} best={min(t[0] for t in trials):.2f} km")
    trials.sort(key=lambda item: item[0])
    selected = trials[0][1]
    history = History()
    seed_history(history, rows, validation_end)
    test_error = evaluate(rows, validation_end, n, history, selected)
    result = {
        "selection": "minimum validation mean error; test evaluated once",
        "selected_config": selected.__dict__,
        "validation": trials[0][2],
        "untouched_test": summarize(test_error),
        "top_validation_trials": [
            {"config": config.__dict__, "metrics": metrics}
            for _, config, metrics in trials[:10]
        ],
        "data_policy": "UCDP only; full history is updated after each forecast date; Telegram excluded",
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
