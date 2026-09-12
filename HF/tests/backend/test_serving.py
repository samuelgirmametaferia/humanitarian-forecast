from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
from humanitarian_forecast.serving.feature_contract import ContractError, load_feature_contract
from humanitarian_forecast.serving.inference import OnnxEnsemble, greedy_order


def feature_payload() -> dict[str, object]:
    return {
        "schemaVersion": "feature-contract.v1",
        "country": "Ethiopia",
        "conflictId": "unseen-conflict",
        "conflictLabel": "Synthetic contract test",
        "anchor": {"latitude": 9.0, "longitude": 39.0},
        "observationCutoff": "2026-09-11T00:00:00Z",
        "horizonDays": 7,
        "events": [[1.0] + [0.0] * 18 for _ in range(16)],
        "candidates": [
            {
                "id": f"candidate-{index}",
                "valid": index < 8,
                "eastKm": float(index * 20),
                "northKm": 0.0,
                "features": [0.0] * 37,
            }
            for index in range(64)
        ],
    }


def test_feature_contract_has_strict_dimensions_and_anchor() -> None:
    contract = load_feature_contract(feature_payload())
    assert contract.events.shape == (16, 19)
    assert contract.candidate_features.shape == (64, 37)
    assert contract.candidate_coordinates[1].tolist() == pytest.approx([0.02, 0.0])
    invalid = feature_payload()
    invalid.pop("anchor")
    with pytest.raises(ContractError):
        load_feature_contract(invalid)


def test_greedy_order_preserves_probability_then_spread_semantics() -> None:
    probabilities = np.asarray([0.5, 0.4, 0.3])
    coordinates = np.asarray([[0.0, 0.0], [5.0, 0.0], [40.0, 0.0]])
    assert greedy_order(probabilities, coordinates, 3, 30.0) == [0, 2, 1]


def test_packaged_onnx_model_is_parity_validated() -> None:
    model_dir = Path(__file__).parents[3] / "models" / "location" / "ethiopia_serving"
    if not model_dir.exists():
        pytest.skip("packaged ONNX artifact is not present")
    manifest = json.loads((model_dir / "manifest.json").read_text())
    report = json.loads((model_dir / "parity-report.json").read_text())
    assert manifest["model"] == "theswarm_fine_v2"
    assert manifest["status"] == "parity-validated"
    assert report["passed"] is True
    assert len(report["rows"]) >= 4
    assert any(row["unseenIdentity"] for row in report["rows"])
    prediction = OnnxEnsemble(model_dir).predict(load_feature_contract(feature_payload()))
    assert prediction.unseen_identity is True
    assert len(prediction.candidates) == 5
    assert sum(prediction.probabilities) == pytest.approx(1.0, abs=1e-6)
