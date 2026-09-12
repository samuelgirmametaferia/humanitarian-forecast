"""Build the Ethiopia population density dot layer from WorldPop.

Input:  WorldPop 2020 UN-adjusted population count, 30 arc-second (~1km)
        GeoTIFF (data.worldpop.org, CC BY 4.0).
Output: public/ethiopia-population-dots.geojson — one point per
        PEOPLE_PER_DOT residents, placed deterministically (golden-angle
        sunflower) inside each 4km grid cell. Dots are a density impression,
        not household locations.

Usage: .venv/bin/python scripts/build_population_dots.py <path-to-tif>
"""

from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import numpy as np
import tifffile

# Each dot stands for this many residents. ~115M population / 10k = ~11.5k dots.
PEOPLE_PER_DOT = 10_000
# Cells below this population are left empty: they are near-empty rangeland,
# and drawing them would blur the density impression without adding people.
MIN_CELL_POPULATION = 1_000
# Aggregate 4x4 of the 30 arc-second cells (~3.7km cells) before dotting, so
# rural dots do not jitter inside single noisy pixels.
BIN = 4
# WorldPop ETH 1km raster: EPSG:4326, origin at the top-left corner.
ORIGIN_LON = 32.99874987166672
ORIGIN_LAT = 14.899583476768186
PIXEL = 0.0083333333
GOLDEN_ANGLE = 2.399963229728653
KM_PER_DEGREE_LAT = 110.574


def main() -> None:
    if len(sys.argv) != 2:
        raise SystemExit(__doc__)
    source = Path(sys.argv[1])

    raster = tifffile.imread(source)
    rows, cols = raster.shape
    raster = np.where(raster == -99999, 0.0, raster.astype(np.float64))

    # Clip to whole bins and sum each BINxBIN block into a coarse grid.
    bin_rows, bin_cols = rows // BIN, cols // BIN
    clipped = raster[: bin_rows * BIN, : bin_cols * BIN]
    coarse = clipped.reshape(bin_rows, BIN, bin_cols, BIN).sum(axis=(1, 3))

    # Square kilometres covered by one coarse cell at its centre latitude.
    cell_size_km = BIN * PIXEL * KM_PER_DEGREE_LAT

    features = []
    total_dots = 0
    for row in range(bin_rows):
        for col in range(bin_cols):
            population = coarse[row, col]
            if population < MIN_CELL_POPULATION:
                continue
            centre_lon = ORIGIN_LON + (col + .5) * BIN * PIXEL
            centre_lat = ORIGIN_LAT - (row + .5) * BIN * PIXEL
            density = population / (cell_size_km ** 2)
            # max(1, ...) keeps lightly populated cells on the map instead of
            # rounding away the ~60% of Ethiopians who live outside towns.
            dot_count = max(1, round(population / PEOPLE_PER_DOT))
            total_dots += dot_count
            # Dots spread over most of the cell in a deterministic sunflower.
            half_lat = BIN * PIXEL * .5 * .96
            half_lon = half_lat / max(math.cos(math.radians(centre_lat)), .2)
            for index in range(dot_count):
                fraction = math.sqrt((index + .5) / dot_count)
                angle = index * GOLDEN_ANGLE
                latitude = centre_lat + math.sin(angle) * half_lat * fraction
                longitude = centre_lon + math.cos(angle) * half_lon * fraction
                features.append({
                    "type": "Feature",
                    "properties": {"d": round(density, 1)},
                    "geometry": {
                        "type": "Point",
                        "coordinates": [round(longitude, 4), round(latitude, 4)],
                    },
                })

    collection = {
        "type": "FeatureCollection",
        "properties": {
            "source": "WorldPop 2020 UN-adjusted population, 30 arc-second",
            "peoplePerDot": PEOPLE_PER_DOT,
        },
        "features": features,
    }

    destination = Path(__file__).parents[1] / "public" / "ethiopia-population-dots.geojson"
    destination.write_text(json.dumps(collection, separators=(",", ":")))
    print(
        f"{len(features)} dots for {coarse.sum():,.0f} residents "
        f"({PEOPLE_PER_DOT:,} per dot) -> {destination}"
    )


if __name__ == "__main__":
    main()
