#!/usr/bin/env python3
"""Run TheSwarm and emit a coarse humanitarian early-warning zone.

Thin wrapper around the swarm predictor pointed at the TheSwarm production
mixture (`models/location/theswarm/v1`): Ethiopia rows covered by the
packaged blend use the mixture probabilities over the shared H3 r4 support,
everything else routes through the v10-prized champion.
"""
from __future__ import annotations

from humanitarian_forecast.location.inference.predict_swarm import main as _predict


def main() -> None:
    _predict(
        default_swarm_dir="models/location/theswarm/v1",
        state_file="theswarm.json",
        model_label="theswarm",
    )


if __name__ == "__main__":
    main()
