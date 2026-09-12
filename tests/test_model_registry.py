import json

import pytest

from humanitarian_forecast.core.model_registry import ModelRegistry


def _model(root, subsystem, version, status="production"):
    directory = root / subsystem / version
    directory.mkdir(parents=True)
    (directory / "info.blt").write_text(json.dumps({
        "subsystem": subsystem,
        "version": version,
        "status": status,
        "description": "test model",
        "artifacts": ["model.onnx"],
    }))


def test_registry_catalogs_all_but_only_runs_accepted_adapters(tmp_path):
    _model(tmp_path, "location/fine", "v1")
    _model(tmp_path, "location/ablation", "v2", "research-rejected")
    registry = ModelRegistry(tmp_path)
    registry.register_adapter("location/fine/v1", lambda payload: {"value": payload["value"] + 1})

    assert registry.manifest()["modelCount"] == 2
    assert [model.id for model in registry.ready_models()] == ["location/fine/v1"]
    assert registry.predict_all({"value": 2})["location/fine/v1"]["prediction"] == {"value": 3}
    with pytest.raises(ValueError):
        registry.predict_all({"value": 2}, ["location/ablation/v2"])
